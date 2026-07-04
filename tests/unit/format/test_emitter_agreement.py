"""T0-A -- the three emitters (C header, ctypes overlay, dataclass) cannot drift."""

import ctypes
import shutil
import struct
import subprocess
import tempfile
import os

import pytest

from zuspec.be.bc.format import spec
from zuspec.be.bc.format.emit_ctypes import CLASSES as CT
from zuspec.be.bc.format.emit_dataclass import CLASSES as DC
from zuspec.be.bc.format import emit_c


@pytest.mark.parametrize("record", spec.RECORDS, ids=lambda r: r.name)
def test_size_agreement(record):
    spec_sz = record.size()
    assert struct.calcsize(record.struct_fmt()) == spec_sz
    assert ctypes.sizeof(CT[record.name]) == spec_sz
    assert DC[record.name].size() == spec_sz


@pytest.mark.parametrize("record", spec.RECORDS, ids=lambda r: r.name)
def test_field_offsets_no_padding(record):
    """ctypes offsets match the spec's cumulative offsets (natural alignment)."""
    ct = CT[record.name]
    off = 0
    for f in record.fields:
        assert getattr(ct, f.name).offset == off, f"{record.name}.{f.name}"
        off += f.byte_len
    assert off == record.size()


def test_cross_emitter_bytes_identical():
    """A dataclass serialized round-trips through the ctypes overlay identically."""
    dh = DC["zbc_header"]()
    dh.magic = spec.MAGIC
    dh.version_major = spec.VERSION_MAJOR
    dh.abi_id = 7
    dh.header_size = DC["zbc_header"].size()
    dh.section_count = 3
    dh.file_size = 4096
    dh.content_hash = 0xDEADBEEFCAFEBABE
    b = dh.to_bytes()

    cv = CT["zbc_header"].from_buffer_copy(b)
    assert bytes(cv.magic) == bytes(spec.MAGIC)
    assert cv.abi_id == 7
    assert cv.section_count == 3
    assert cv.file_size == 4096
    assert cv.content_hash == 0xDEADBEEFCAFEBABE
    assert DC["zbc_header"].from_bytes(b) == dh


@pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")
def test_generated_header_compiles():
    """The generated header's _Static_asserts prove the C layout matches the spec."""
    header_dir = None
    try:
        import zuspec.rt.core as rt
        header_dir = rt.include_dir()
    except Exception:
        pytest.skip("rt-core not importable")
    header = os.path.join(header_dir, "zbc_format.h")
    assert os.path.exists(header), "run emit_c to generate the header"

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "check.c")
        with open(src, "w") as fp:
            fp.write(f'#include "{header}"\nint main(void){{return 0;}}\n')
        r = subprocess.run(
            ["gcc", "-std=c11", "-Wall", "-Wextra", "-c", src, "-o",
             os.path.join(td, "check.o")],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
