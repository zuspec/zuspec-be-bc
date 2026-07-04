"""
emit_ctypes.py -- generate a ``ctypes`` overlay for every ``.zbc`` record (P0-6).

The overlay is for zero-copy ``mmap`` reads in Python: a ``ctypes`` structure
whose byte layout matches the C header field-for-field. We use
``LittleEndianStructure`` so the overlay is correct regardless of host byte order,
and rely on natural alignment (no ``_pack_``) so it matches the C compiler's own
layout -- the conformance tests assert this against the C header and ``struct``.
"""

import ctypes
from typing import Dict, Type

from .spec import RECORDS

_CTYPE = {
    "u8": ctypes.c_uint8,
    "u16": ctypes.c_uint16,
    "u32": ctypes.c_uint32,
    "u64": ctypes.c_uint64,
}


def _fields_for(record):
    out = []
    for f in record.fields:
        base = _CTYPE[f.type]
        out.append((f.name, base * f.count) if f.array else (f.name, base))
    return out


def build_classes() -> Dict[str, Type[ctypes.LittleEndianStructure]]:
    """Return ``{record_name: ctypes.LittleEndianStructure subclass}``."""
    classes: Dict[str, Type[ctypes.LittleEndianStructure]] = {}
    for r in RECORDS:
        classes[r.name] = type(
            r.name,
            (ctypes.LittleEndianStructure,),
            {"_fields_": _fields_for(r), "__doc__": r.doc},
        )
    return classes


#: Live overlay classes keyed by record name.
CLASSES: Dict[str, Type[ctypes.LittleEndianStructure]] = build_classes()


def overlay(record_name: str) -> Type[ctypes.LittleEndianStructure]:
    return CLASSES[record_name]
