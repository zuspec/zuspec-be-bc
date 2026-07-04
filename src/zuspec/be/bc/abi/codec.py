"""
codec.py -- pack/unpack for the value ABI (design D§11).

Pure Python, used by the oracle and by the conformance tests. All encodings are
little-endian, two's complement for signed types, byte-granular for aggregates
(see :mod:`.value` for the canonical layout description).

Encoding is *total* over the type's value domain: a Python ``int`` is reduced
modulo ``2**width`` before encoding (mirroring hardware truncation / overflow
wraparound), so ``decode(encode(v)) == v & mask`` for unsigned and the
sign-extended equivalent for signed. This makes the codec the single authority on
truncation/overflow semantics (design D§15.5) rather than leaving it to ad-hoc
Python in op handlers.
"""

from typing import Any, List

from .value import (
    ScalarType,
    ArrayType,
    StructType,
    ValueType,
    INLINE_MAX_BITS,
)


def is_inline(t: ValueType) -> bool:
    """True if a value of ``t`` fits a ZBC register/slot (scalar <= 64 bits)."""
    return isinstance(t, ScalarType) and t.width_bits <= INLINE_MAX_BITS


# --------------------------------------------------------------------------- #
# Scalars
# --------------------------------------------------------------------------- #

def encode_scalar(value: int, t: ScalarType) -> bytes:
    """Encode ``value`` as ``t``: little-endian, two's complement if signed.

    ``value`` is reduced modulo ``2**width`` first, so out-of-range inputs wrap
    exactly as fixed-width hardware arithmetic would.
    """
    if not isinstance(t, ScalarType):
        raise TypeError(f"encode_scalar expects ScalarType, got {type(t).__name__}")
    v = int(value) & t.mask  # truncate/wrap to the field width
    return v.to_bytes(t.byte_len, byteorder="little", signed=False)


def decode_scalar(data: bytes, t: ScalarType) -> int:
    """Decode the first ``t.byte_len`` bytes of ``data`` as ``t``.

    Returns a signed Python ``int`` for signed types (the top unused bits of the
    final byte are interpreted as sign extension), unsigned otherwise.
    """
    if not isinstance(t, ScalarType):
        raise TypeError(f"decode_scalar expects ScalarType, got {type(t).__name__}")
    if len(data) < t.byte_len:
        raise ValueError(
            f"need {t.byte_len} bytes to decode a {t.width_bits}-bit scalar, "
            f"got {len(data)}"
        )
    raw = int.from_bytes(data[: t.byte_len], byteorder="little", signed=False)
    raw &= t.mask
    if t.signed and (raw >> (t.width_bits - 1)) & 1:
        raw -= 1 << t.width_bits
    return raw


# --------------------------------------------------------------------------- #
# Aggregates (byte-granular packing)
# --------------------------------------------------------------------------- #

def encode(value: Any, t: ValueType) -> bytes:
    """Encode a value of any :data:`.value.ValueType` to canonical bytes.

    * scalar  -> :func:`encode_scalar`
    * array   -> concatenation of ``count`` element encodings (``value`` is a
      sequence of length ``count``)
    * struct  -> concatenation of field encodings in **declaration order**
      (``value`` is a mapping ``{name: field_value}``)
    """
    if isinstance(t, ScalarType):
        return encode_scalar(value, t)
    if isinstance(t, ArrayType):
        if len(value) != t.count:
            raise ValueError(
                f"array expects {t.count} elements, got {len(value)}"
            )
        return b"".join(encode(elem, t.element) for elem in value)
    if isinstance(t, StructType):
        out: List[bytes] = []
        for name, ftype in t.fields:
            if name not in value:
                raise KeyError(f"struct value missing field {name!r}")
            out.append(encode(value[name], ftype))
        return b"".join(out)
    raise TypeError(f"unsupported ValueType: {type(t).__name__}")


def decode(data: bytes, t: ValueType) -> Any:
    """Inverse of :func:`encode`. Returns int / list / dict mirroring the type."""
    value, consumed = _decode_at(data, 0, t)
    return value


def _decode_at(data: bytes, off: int, t: ValueType):
    if isinstance(t, ScalarType):
        return decode_scalar(data[off:], t), t.byte_len
    if isinstance(t, ArrayType):
        elems = []
        cur = off
        for _ in range(t.count):
            v, n = _decode_at(data, cur, t.element)
            elems.append(v)
            cur += n
        return elems, cur - off
    if isinstance(t, StructType):
        result = {}
        cur = off
        for name, ftype in t.fields:
            v, n = _decode_at(data, cur, ftype)
            result[name] = v
            cur += n
        return result, cur - off
    raise TypeError(f"unsupported ValueType: {type(t).__name__}")


# --------------------------------------------------------------------------- #
# Constant pool (>64-bit literals -> SEC_CONST)
# --------------------------------------------------------------------------- #

def const_pool_record(value: int, t: ScalarType) -> bytes:
    """Bytes for one SEC_CONST entry holding a wide (>64-bit) literal.

    Layout (little-endian, self-describing so the C reader needs no side table):

    ``u32 width_bits | u32 byte_len | u8[byte_len] payload``  (payload 8-aligned)

    ``payload`` is the same canonical LE two's-complement encoding
    :func:`encode_scalar` produces, so a wide literal decodes identically whether
    it arrived inline or via the pool.
    """
    payload = encode_scalar(value, t)
    header = t.width_bits.to_bytes(4, "little") + t.byte_len.to_bytes(4, "little")
    rec = header + payload
    pad = (-len(rec)) % 8
    return rec + b"\x00" * pad
