"""T0-B -- golden format bytes; profile presence/absence of provenance sections."""

from zuspec.be.bc.abi import ABI_ID
from zuspec.be.bc.format import spec
from zuspec.be.bc.format.writer import Section, ZbcImage, write_image, PROV_SECTION_KINDS
from zuspec.be.bc.format.reader import read_image
from zuspec.be.bc.format.inspect import inspect_bytes


def _codegen_image():
    return ZbcImage(
        sections=[
            Section(spec.sec_kind("ZBC_SEC_CODE"), b"\x01\x02\x03\x04", count=4, elem_size=1),
            Section(spec.sec_kind("ZBC_SEC_STRB"), b"\x00hi\x00"),
            Section(spec.sec_kind("ZBC_SEC_PROV"), b"\x00" * 32, count=1, elem_size=32),
        ],
        abi_id=ABI_ID,
        profile="codegen",
    )


def test_header_magic_and_fields():
    data = write_image(_codegen_image())
    hdr = spec.RECORDS_BY_NAME  # noqa: F841 (spec exists)
    from zuspec.be.bc.format.emit_dataclass import CLASSES as DC

    h = DC["zbc_header"].from_bytes(data)
    assert tuple(h.magic) == spec.MAGIC
    assert h.version_major == spec.VERSION_MAJOR
    assert h.version_minor == spec.VERSION_MINOR
    assert h.abi_id == ABI_ID
    assert h.header_size == 48
    assert h.section_count == 3
    assert h.file_size == len(data)
    # HAS_PROV set, PROFILE_RUNTIME clear
    assert h.flags & spec.flag_value("zbc_hdr_flags", "ZBC_HDR_HAS_PROV")
    assert not (h.flags & spec.flag_value("zbc_hdr_flags", "ZBC_HDR_PROFILE_RUNTIME"))


def test_all_payloads_8_byte_aligned():
    data = write_image(_codegen_image())
    img = read_image(data)
    # We can't see raw offsets from the image, so re-derive from the directory.
    from zuspec.be.bc.format.emit_dataclass import CLASSES as DC

    h = DC["zbc_header"].from_bytes(data)
    scls = DC["zbc_section"]
    for i in range(h.section_count):
        se = scls.from_bytes(data[h.section_dir_off + i * scls.size():])
        assert se.offset % 8 == 0, f"section {i} not 8-aligned"


def test_runtime_profile_omits_provenance():
    # A runtime profile carries only non-provenance sections and clears HAS_PROV.
    rt = ZbcImage(
        sections=[Section(spec.sec_kind("ZBC_SEC_CODE"), b"\xaa\xbb", count=2, elem_size=1)],
        abi_id=ABI_ID,
        profile="runtime",
    )
    data = write_image(rt)
    from zuspec.be.bc.format.emit_dataclass import CLASSES as DC

    h = DC["zbc_header"].from_bytes(data)
    assert h.flags & spec.flag_value("zbc_hdr_flags", "ZBC_HDR_PROFILE_RUNTIME")
    assert not (h.flags & spec.flag_value("zbc_hdr_flags", "ZBC_HDR_HAS_PROV"))
    img = read_image(data)
    assert all(s.kind not in PROV_SECTION_KINDS for s in img.sections)


def test_inspect_is_deterministic():
    data = write_image(_codegen_image())
    assert inspect_bytes(data) == inspect_bytes(data)
    assert "ZBC_SEC_CODE" in inspect_bytes(data)


def test_unknown_section_kind_tolerated():
    # A future section kind (99) is preserved as raw, not rejected.
    img = ZbcImage(
        sections=[Section(99, b"\xde\xad\xbe\xef")],
        abi_id=ABI_ID,
    )
    data = write_image(img)
    back = read_image(data)
    assert back.sections[0].kind == 99
    assert back.sections[0].payload == b"\xde\xad\xbe\xef"
    assert "UNKNOWN(99)" in inspect_bytes(data)
