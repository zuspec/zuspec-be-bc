"""
writer.py -- serialize an in-memory image to ``.zbc`` bytes (P0-8).

Layout produced (D§12.2):

    +----------------+ offset 0
    |   zbc_header   |  (48 bytes)
    +----------------+ header.section_dir_off
    | zbc_section[N] |  section directory
    +----------------+
    |  payload 0     |  each payload 8-byte aligned, located by (offset,size)
    |  ...           |
    +----------------+

The container model here (``Section`` / ``ZbcImage``) is the P0 *container*-level
image -- enough to carry arbitrary section payloads. The richer per-section
in-memory model (opcode stream, descriptors) is P1-1; it serializes *through*
this writer.
"""

import dataclasses as dc
from typing import List

from . import spec
from .emit_dataclass import CLASSES as DC

#: Provenance-family section kinds (absent from the runtime profile).
PROV_SECTION_KINDS = frozenset(
    spec.sec_kind(n)
    for n in ("ZBC_SEC_STRB", "ZBC_SEC_STRO", "ZBC_SEC_FILE",
              "ZBC_SEC_PROV", "ZBC_SEC_CMNT", "ZBC_SEC_LINE")
)

_ALIGN = 8


def _align_up(n: int, a: int = _ALIGN) -> int:
    return (n + a - 1) & ~(a - 1)


def _fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


@dc.dataclass
class Section:
    """One container section: a kind tag plus an opaque payload."""

    kind: int
    payload: bytes
    flags: int = 0
    count: int = 0       # element count for record arrays (0 = N/A)
    elem_size: int = 0   # per-element stride (0 = N/A)


@dc.dataclass
class ZbcImage:
    """Container-level image: a header's worth of scalars + a list of sections."""

    sections: List[Section] = dc.field(default_factory=list)
    abi_id: int = 0
    entry_coro: int = 0
    profile: str = "codegen"          # "codegen" | "runtime"
    #: a coroutine constructs the component tree before the entry runs (P1.5)
    comp_init: bool = False
    version_major: int = spec.VERSION_MAJOR
    version_minor: int = spec.VERSION_MINOR
    compute_hash: bool = True

    def has_provenance(self) -> bool:
        return any(s.kind in PROV_SECTION_KINDS for s in self.sections)


def _header_flags(image: ZbcImage) -> int:
    flags = 0
    if image.has_provenance():
        flags |= spec.flag_value("zbc_hdr_flags", "ZBC_HDR_HAS_PROV")
    if image.profile == "runtime":
        flags |= spec.flag_value("zbc_hdr_flags", "ZBC_HDR_PROFILE_RUNTIME")
    if image.comp_init:
        flags |= spec.flag_value("zbc_hdr_flags", "ZBC_HDR_COMP_INIT")
    return flags


def write_image(image: ZbcImage) -> bytes:
    if image.profile == "runtime" and image.has_provenance():
        raise ValueError(
            "runtime profile must not contain provenance sections "
            "(STRB/STRO/FILE/PROV/CMNT/LINE); strip them before writing"
        )

    header_cls = DC["zbc_header"]
    section_cls = DC["zbc_section"]
    header_size = header_cls.size()          # 48
    section_size = section_cls.size()         # 32
    n = len(image.sections)

    section_dir_off = _align_up(header_size)  # 48, already aligned
    payload_start = _align_up(section_dir_off + n * section_size)

    # Assign payload offsets (each 8-aligned).
    offsets: List[int] = []
    cur = payload_start
    for s in image.sections:
        cur = _align_up(cur)
        offsets.append(cur)
        cur += len(s.payload)
    file_size = _align_up(cur) if image.sections else payload_start

    # Build directory.
    dir_bytes = b""
    for s, off in zip(image.sections, offsets):
        se = section_cls()
        se.kind = s.kind
        se.flags = s.flags
        se.offset = off
        se.size = len(s.payload)
        se.count = s.count
        se.elem_size = s.elem_size
        dir_bytes += se.to_bytes()

    # Assemble body (everything from offset 0) with padding between regions.
    buf = bytearray(file_size)

    # Directory.
    buf[section_dir_off : section_dir_off + len(dir_bytes)] = dir_bytes
    # Payloads.
    for s, off in zip(image.sections, offsets):
        buf[off : off + len(s.payload)] = s.payload

    # Header (last, so file_size/hash are known). content_hash covers bytes
    # after the content_hash field == everything from offset header_size? No:
    # "hash of all bytes after this field". The field is the final 8 bytes of the
    # header (offsets 40..48), so hashed range is buf[48:].
    hdr = header_cls()
    hdr.magic = spec.MAGIC
    hdr.version_major = image.version_major
    hdr.version_minor = image.version_minor
    hdr.flags = _header_flags(image)
    hdr.header_size = header_size
    hdr.abi_id = image.abi_id
    hdr.section_count = n
    hdr.section_dir_off = section_dir_off
    hdr.entry_coro = image.entry_coro
    hdr.file_size = file_size
    hdr.content_hash = 0

    # Write header without hash first so we can hash the tail deterministically.
    buf[0:header_size] = hdr.to_bytes()
    if image.compute_hash:
        hdr.content_hash = _fnv1a64(bytes(buf[header_size:]))
        buf[0:header_size] = hdr.to_bytes()

    return bytes(buf)
