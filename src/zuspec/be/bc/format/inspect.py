"""
inspect.py -- a generic ``.zbc`` disassembler / inspector (P0-9).

Walks the section directory *generically* (kind + elem_size), needing no
hardcoded per-section layout. Used by tests to assert profile contents and to
produce reviewable golden dumps, and later by debug tooling. Output is
deterministic so it can be checked in as a golden.
"""

from typing import Union

from . import spec
from .emit_dataclass import CLASSES as DC
from .reader import read_image
from .writer import ZbcImage


def _header_summary(data: bytes) -> str:
    hdr = DC["zbc_header"].from_bytes(data)
    profile = (
        "runtime"
        if hdr.flags & spec.flag_value("zbc_hdr_flags", "ZBC_HDR_PROFILE_RUNTIME")
        else "codegen"
    )
    has_prov = bool(hdr.flags & spec.flag_value("zbc_hdr_flags", "ZBC_HDR_HAS_PROV"))
    lines = [
        "zbc_header:",
        f"  magic          {' '.join('%02x' % b for b in hdr.magic)}",
        f"  version        {hdr.version_major}.{hdr.version_minor}",
        f"  abi_id         {hdr.abi_id}",
        f"  flags          0x{hdr.flags:08x} (profile={profile}, has_prov={has_prov})",
        f"  entry_coro     {hdr.entry_coro}",
        f"  section_count  {hdr.section_count}",
        f"  file_size      {hdr.file_size}",
        f"  content_hash   0x{hdr.content_hash:016x}",
    ]
    return "\n".join(lines)


def inspect_bytes(data: bytes) -> str:
    """Return a deterministic textual dump of a ``.zbc`` byte image."""
    out = [_header_summary(data), "sections:"]
    section_cls = DC["zbc_section"]
    hdr = DC["zbc_header"].from_bytes(data)
    dir_off = hdr.section_dir_off
    for i in range(hdr.section_count):
        se = section_cls.from_bytes(data[dir_off + i * section_cls.size():])
        kind_name = spec.sec_kind_name(se.kind) or f"UNKNOWN({se.kind})"
        out.append(
            f"  [{i}] {kind_name:<14} off={se.offset} size={se.size} "
            f"count={se.count} elem_size={se.elem_size} flags=0x{se.flags:04x}"
        )
    return "\n".join(out) + "\n"


def inspect_image(image: ZbcImage) -> str:
    """Inspect an in-memory image by round-tripping it through the writer."""
    from .writer import write_image

    return inspect_bytes(write_image(image))


def inspect(obj: Union[bytes, ZbcImage]) -> str:
    if isinstance(obj, (bytes, bytearray)):
        return inspect_bytes(bytes(obj))
    return inspect_image(obj)
