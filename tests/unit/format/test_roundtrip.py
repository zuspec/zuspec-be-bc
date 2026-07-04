"""T0-C -- round-trip identity: image == read(write(image))."""

import pytest

from zuspec.be.bc.abi import ABI_ID
from zuspec.be.bc.format import spec
from zuspec.be.bc.format.writer import Section, ZbcImage, write_image
from zuspec.be.bc.format.reader import read_image, ZbcFormatError


def _images():
    yield ZbcImage(sections=[], abi_id=ABI_ID)  # empty
    yield ZbcImage(
        sections=[Section(spec.sec_kind("ZBC_SEC_CODE"), b"\x00\x11\x22\x33", count=4, elem_size=1)],
        abi_id=ABI_ID,
    )
    yield ZbcImage(
        sections=[
            Section(spec.sec_kind("ZBC_SEC_CODE"), bytes(range(7)), count=7, elem_size=1),
            Section(spec.sec_kind("ZBC_SEC_CONST"), b"\x10" * 24, count=1, elem_size=24),
            Section(spec.sec_kind("ZBC_SEC_STRB"), b"\x00abc\x00def\x00"),
        ],
        abi_id=ABI_ID,
        entry_coro=2,
    )


@pytest.mark.parametrize("img", list(_images()))
def test_roundtrip_identity(img):
    assert read_image(write_image(img)) == img


@pytest.mark.parametrize("img", list(_images()))
def test_reserialize_is_byte_stable(img):
    data1 = write_image(img)
    data2 = write_image(read_image(data1))
    assert data1 == data2


def test_bad_magic_rejected():
    data = bytearray(write_image(ZbcImage(sections=[], abi_id=ABI_ID)))
    data[0] = 0
    with pytest.raises(ZbcFormatError):
        read_image(bytes(data))


def test_abi_id_mismatch_rejected():
    data = write_image(ZbcImage(sections=[], abi_id=ABI_ID))
    with pytest.raises(ZbcFormatError):
        read_image(data, expect_abi_id=ABI_ID + 1)


def test_content_hash_tamper_rejected():
    data = bytearray(write_image(
        ZbcImage(sections=[Section(spec.sec_kind("ZBC_SEC_CODE"), b"\x01\x02")], abi_id=ABI_ID)
    ))
    data[-1] ^= 0xFF
    with pytest.raises(ZbcFormatError):
        read_image(bytes(data))
