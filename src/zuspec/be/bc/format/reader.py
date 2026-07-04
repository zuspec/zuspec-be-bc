"""
reader.py -- parse ``.zbc`` bytes back into a container image (P0-8).

Self-describing walk (D§12.2): read the header, walk the section directory,
slice payloads by ``(offset, size)``. Unknown section kinds are preserved as raw
``Section`` objects (forward-compat); absent sections simply don't appear. The
reader validates the fields the engine validates on load -- ``magic``,
``version_major``, ``file_size`` -- and, if present, ``content_hash``.
"""

from typing import Optional

from . import spec
from .emit_dataclass import CLASSES as DC
from .writer import Section, ZbcImage, _fnv1a64


class ZbcFormatError(Exception):
    """Raised when bytes are not a valid/compatible ``.zbc`` image."""


def read_image(data: bytes, *, verify_hash: bool = True,
               expect_abi_id: Optional[int] = None) -> ZbcImage:
    header_cls = DC["zbc_header"]
    section_cls = DC["zbc_section"]
    header_size = header_cls.size()

    if len(data) < header_size:
        raise ZbcFormatError(f"truncated: {len(data)} bytes < header {header_size}")

    hdr = header_cls.from_bytes(data)
    if tuple(hdr.magic) != tuple(spec.MAGIC):
        raise ZbcFormatError(f"bad magic {tuple(hdr.magic)!r}")
    if hdr.version_major != spec.VERSION_MAJOR:
        raise ZbcFormatError(
            f"incompatible major version {hdr.version_major} "
            f"(reader supports {spec.VERSION_MAJOR})"
        )
    if hdr.file_size != len(data):
        raise ZbcFormatError(
            f"file_size {hdr.file_size} != actual length {len(data)}"
        )
    if expect_abi_id is not None and hdr.abi_id != expect_abi_id:
        raise ZbcFormatError(
            f"abi_id {hdr.abi_id} != expected {expect_abi_id}"
        )
    if verify_hash and hdr.content_hash != 0:
        actual = _fnv1a64(data[header_size:])
        if actual != hdr.content_hash:
            raise ZbcFormatError(
                f"content_hash mismatch: stored {hdr.content_hash:#x}, "
                f"computed {actual:#x}"
            )

    # Walk the directory. elem_size is the forward-compat stride, but the
    # directory entries themselves are fixed; we read section_count of them.
    sections = []
    dir_off = hdr.section_dir_off
    for i in range(hdr.section_count):
        base = dir_off + i * section_cls.size()
        if base + section_cls.size() > len(data):
            raise ZbcFormatError(f"section directory entry {i} out of range")
        se = section_cls.from_bytes(data[base:])
        if se.offset + se.size > len(data):
            raise ZbcFormatError(
                f"section {i} payload [{se.offset},{se.offset + se.size}) "
                f"out of range (len {len(data)})"
            )
        payload = data[se.offset : se.offset + se.size]
        sections.append(
            Section(kind=se.kind, payload=payload, flags=se.flags,
                    count=se.count, elem_size=se.elem_size)
        )

    profile = (
        "runtime"
        if hdr.flags & spec.flag_value("zbc_hdr_flags", "ZBC_HDR_PROFILE_RUNTIME")
        else "codegen"
    )
    return ZbcImage(
        sections=sections,
        abi_id=hdr.abi_id,
        entry_coro=hdr.entry_coro,
        profile=profile,
        version_major=hdr.version_major,
        version_minor=hdr.version_minor,
    )
