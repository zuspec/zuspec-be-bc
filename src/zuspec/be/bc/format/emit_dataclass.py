"""
emit_dataclass.py -- generate in-memory dataclasses for every record (P0-7).

These are the natural Python form the oracle builds and manipulates. Each
generated class carries a ``struct``-based codec (``to_bytes`` / ``from_bytes``)
derived from the same spec, so the dataclass view and the ``ctypes`` overlay
decode identical bytes (asserted by the conformance tests).

Array fields are represented as tuples of ints.
"""

import dataclasses as dc
import struct
from typing import Any, Dict, List, Tuple, Type

from .spec import RECORDS, Record


def _default_factory(field):
    if field.array:
        n = field.count
        return lambda: tuple(0 for _ in range(n))
    return lambda: 0


def _make_class(record: Record) -> Type:
    fmt = record.struct_fmt()

    fields: List[Tuple[str, Any, Any]] = []
    for f in record.fields:
        fields.append((f.name, Any, dc.field(default_factory=_default_factory(f))))

    cls = dc.make_dataclass(record.name, fields, namespace={"__doc__": record.doc})

    # Attach the spec-derived codec.
    cls._RECORD = record
    cls._STRUCT_FMT = fmt

    def to_bytes(self) -> bytes:
        flat: List[int] = []
        for fld in record.fields:
            v = getattr(self, fld.name)
            if fld.array:
                seq = tuple(v)
                if len(seq) != fld.count:
                    raise ValueError(
                        f"{record.name}.{fld.name} expects {fld.count} elements, "
                        f"got {len(seq)}"
                    )
                flat.extend(int(x) for x in seq)
            else:
                flat.append(int(v))
        return struct.pack(fmt, *flat)

    @classmethod
    def from_bytes(klass, data: bytes):
        vals = struct.unpack(fmt, data[: struct.calcsize(fmt)])
        obj = klass()
        i = 0
        for fld in record.fields:
            if fld.array:
                setattr(obj, fld.name, tuple(vals[i : i + fld.count]))
                i += fld.count
            else:
                setattr(obj, fld.name, vals[i])
                i += 1
        return obj

    @staticmethod
    def size() -> int:
        return struct.calcsize(fmt)

    cls.to_bytes = to_bytes
    cls.from_bytes = from_bytes
    cls.size = size
    return cls


def build_classes() -> Dict[str, Type]:
    return {r.name: _make_class(r) for r in RECORDS}


#: Live dataclass types keyed by record name.
CLASSES: Dict[str, Type] = build_classes()


def dataclass_for(record_name: str) -> Type:
    return CLASSES[record_name]
