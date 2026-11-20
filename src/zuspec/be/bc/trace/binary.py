"""
binary.py -- the compact binary trace form, ``.pstrace`` (pssc P4.4c).

The same records as the NDJSON form (``schema.py``), in fewer bytes, and
simple enough to write from C (the native engine's sink). A file is:

* the magic ``b"PSTRACE\\0"``, then the schema version (a uvarint), then the
  header: a map value with the NDJSON header's keys other than ``zbc_trace``
  (a test's name, seed, ...);
* then records, each a uvarint byte length and that many bytes:
  ``seq`` as a zigzag varint delta from ``previous seq + 1`` (0 when the
  sequence is unbroken; the first record's previous seq is -1), ``kind`` (a
  string value), ``coro`` and ``src_ref`` (uvarints), ``detail`` (a map
  value) and the extras (a map value: the line's other keys, such as
  ``file``, ``line`` and ``t``; an empty map when there are none).

A value is a one-byte tag and its payload:

==== ======== ===========================================================
Tag  Name     Payload
==== ======== ===========================================================
0    null
1    false
2    true
3    int      zigzag LEB128 (any size)
4    float    8 bytes, IEEE 754 binary64, little endian
5    str      uvarint byte length, UTF-8 bytes; the string gets the next id
6    strref   uvarint id of a string seen before in this file
7    list     uvarint count, the values
8    map      uvarint count, then key (a str or strref value) and value pairs,
              keys in sorted order
==== ======== ===========================================================

String ids count from 0, in the order the strings first appear in the file
(the header included), so repeated kinds, keys and names cost a byte or two.
Maps are written with sorted keys, so the bytes are a pure function of the
events, as in the NDJSON form, and converting either way is lossless:
``to_ndjson(from_ndjson(t)) == t`` for a canonical NDJSON trace ``t``.
"""

import json
import struct
from typing import Any, BinaryIO, Dict, Iterator, List, Tuple

from .schema import TRACE_SCHEMA_VERSION
from .sink import TraceSink

MAGIC = b"PSTRACE\0"

T_NULL, T_FALSE, T_TRUE, T_INT, T_FLOAT, T_STR, T_STRREF, T_LIST, T_MAP = \
    range(9)

#: the record's fixed fields; every other key of an NDJSON line is an extra
_FIXED = ("seq", "kind", "coro", "src_ref", "detail")
_HEADER_KEY = "zbc_trace"


def _dumps(d: Dict[str, Any]) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def _uvarint(out: bytearray, n: int) -> None:
    if n < 0:
        raise ValueError(f"negative uvarint {n}")
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def _zigzag(n: int) -> int:
    return n * 2 if n >= 0 else -n * 2 - 1


def _unzigzag(z: int) -> int:
    return z >> 1 if not z & 1 else -((z + 1) >> 1)


class Encoder:
    """Encodes values, interning strings across one file."""

    def __init__(self) -> None:
        self._ids: Dict[str, int] = {}

    def value(self, out: bytearray, v: Any) -> None:
        if v is None:
            out.append(T_NULL)
        elif v is True:
            out.append(T_TRUE)
        elif v is False:
            out.append(T_FALSE)
        elif isinstance(v, int):
            out.append(T_INT)
            _uvarint(out, _zigzag(int(v)))
        elif isinstance(v, float):
            out.append(T_FLOAT)
            out += struct.pack("<d", v)
        elif isinstance(v, str):
            self.string(out, v)
        elif isinstance(v, (list, tuple)):
            out.append(T_LIST)
            _uvarint(out, len(v))
            for x in v:
                self.value(out, x)
        elif isinstance(v, dict):
            out.append(T_MAP)
            _uvarint(out, len(v))
            for k in sorted(v):
                if not isinstance(k, str):
                    raise TypeError(f"map key {k!r} is not a string")
                self.string(out, k)
                self.value(out, v[k])
        else:
            raise TypeError(f"cannot encode {type(v).__name__} in a trace")

    def string(self, out: bytearray, s: str) -> None:
        i = self._ids.get(s)
        if i is not None:
            out.append(T_STRREF)
            _uvarint(out, i)
            return
        self._ids[s] = len(self._ids)
        b = s.encode("utf-8")
        out.append(T_STR)
        _uvarint(out, len(b))
        out += b


class Decoder:
    """Decodes values from ``buf``, keeping the file's string table."""

    def __init__(self, buf: bytes, pos: int = 0) -> None:
        self.buf, self.pos = buf, pos
        self._strs: List[str] = []

    def _byte(self) -> int:
        if self.pos >= len(self.buf):
            raise ValueError("truncated trace")
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def uvarint(self) -> int:
        n, shift = 0, 0
        while True:
            b = self._byte()
            n |= (b & 0x7F) << shift
            if not b & 0x80:
                return n
            shift += 7

    def _take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise ValueError("truncated trace")
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def value(self) -> Any:
        t = self._byte()
        if t == T_NULL:
            return None
        if t == T_FALSE:
            return False
        if t == T_TRUE:
            return True
        if t == T_INT:
            return _unzigzag(self.uvarint())
        if t == T_FLOAT:
            return struct.unpack("<d", self._take(8))[0]
        if t in (T_STR, T_STRREF):
            return self._string(t)
        if t == T_LIST:
            return [self.value() for _ in range(self.uvarint())]
        if t == T_MAP:
            d = {}
            for _ in range(self.uvarint()):
                k = self._string(self._byte())
                d[k] = self.value()
            return d
        raise ValueError(f"bad value tag {t} at byte {self.pos - 1}")

    def _string(self, t: int) -> str:
        if t == T_STR:
            s = self._take(self.uvarint()).decode("utf-8")
            self._strs.append(s)
            return s
        if t == T_STRREF:
            i = self.uvarint()
            if i >= len(self._strs):
                raise ValueError(f"string id {i} not defined yet")
            return self._strs[i]
        raise ValueError(f"expected a string, got tag {t}")


class Writer:
    """Writes a binary trace: the header on creation, then one record per
    :meth:`write` (an NDJSON line's dict: the fixed fields and any extras)."""

    def __init__(self, stream: BinaryIO, **header: Any) -> None:
        self.stream = stream
        self._enc = Encoder()
        self._prev = -1
        out = bytearray(MAGIC)
        _uvarint(out, TRACE_SCHEMA_VERSION)
        self._enc.value(out, header)
        stream.write(bytes(out))

    def write(self, d: Dict[str, Any]) -> None:
        body = bytearray()
        seq = d["seq"]
        _uvarint(body, _zigzag(seq - self._prev - 1))
        self._prev = seq
        self._enc.string(body, d["kind"])
        _uvarint(body, d.get("coro", 0))
        _uvarint(body, d.get("src_ref", 0))
        self._enc.value(body, d.get("detail", {}))
        self._enc.value(body, {k: v for k, v in d.items() if k not in _FIXED})
        out = bytearray()
        _uvarint(out, len(body))
        self.stream.write(bytes(out + body))


class BinarySink(TraceSink):
    """A trace sink writing the binary form to a byte stream."""

    def __init__(self, stream: BinaryIO, **header: Any) -> None:
        super().__init__()
        self.writer = Writer(stream, **header)

    def emit(self, event) -> None:
        self.writer.write(event.to_dict())


def is_binary(data: bytes) -> bool:
    """Whether ``data`` (a file's first bytes) is a binary trace."""
    return data[:len(MAGIC)] == MAGIC


def read(data: bytes) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
    """``(version, header, records)`` of a binary trace; each record is the
    dict its NDJSON line would hold. A truncated last record raises."""
    version, header, it = _open(data)
    return version, header, list(it)


def _open(data: bytes) -> Tuple[int, Dict[str, Any], Iterator[Dict[str, Any]]]:
    if not is_binary(data):
        raise ValueError("not a binary trace (bad magic)")
    dec = Decoder(data, len(MAGIC))
    version = dec.uvarint()
    if version > TRACE_SCHEMA_VERSION:
        raise ValueError(f"trace version {version} is newer than "
                         f"{TRACE_SCHEMA_VERSION}")
    header = dec.value()
    if not isinstance(header, dict):
        raise ValueError("trace header is not a map")

    def records() -> Iterator[Dict[str, Any]]:
        prev = -1
        while dec.pos < len(data):
            n = dec.uvarint()
            end = dec.pos + n
            if end > len(data):
                raise ValueError("truncated trace")
            seq = prev + 1 + _unzigzag(dec.uvarint())
            prev = seq
            d = {"seq": seq, "kind": dec.value(), "coro": dec.uvarint(),
                 "src_ref": dec.uvarint(), "detail": dec.value()}
            extra = dec.value()
            if dec.pos != end:
                raise ValueError(f"record {seq}: length {n} does not match")
            d.update(extra)
            yield d

    return version, header, records()


def to_ndjson(data: bytes) -> str:
    """The canonical NDJSON form of a binary trace."""
    version, header, recs = read(data)
    lines = [_dumps(dict(header, **{_HEADER_KEY: version}))]
    lines.extend(_dumps(d) for d in recs)
    return "\n".join(lines) + "\n"


def from_ndjson(text: str) -> bytes:
    """The binary form of an NDJSON trace (header line, then events)."""
    import io

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty trace")
    header = json.loads(lines[0])
    if _HEADER_KEY not in header:
        raise ValueError("missing trace header")
    if header[_HEADER_KEY] != TRACE_SCHEMA_VERSION:
        raise ValueError(f"trace version {header[_HEADER_KEY]} is not "
                         f"{TRACE_SCHEMA_VERSION}")
    del header[_HEADER_KEY]
    buf = io.BytesIO()
    w = Writer(buf, **header)
    for ln in lines[1:]:
        w.write(json.loads(ln))
    return buf.getvalue()


def load(path: str) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
    """``(version, header, records)`` of a trace file in either form."""
    with open(path, "rb") as f:
        data = f.read()
    if is_binary(data):
        return read(data)
    lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty trace")
    header = json.loads(lines[0])
    version = header.pop(_HEADER_KEY, None)
    if version is None:
        raise ValueError("missing trace header")
    return version, header, [json.loads(ln) for ln in lines[1:]]
