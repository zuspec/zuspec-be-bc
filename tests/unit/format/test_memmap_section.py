"""The MEMMAP section: the host's address spaces and regions survive write
and read (F7.2c)."""

from zuspec.be.bc.format import spec
from zuspec.be.bc.model import CoroDescriptor, ZbcModel

MEMMAP = {
    "spaces": [{"id": 0, "path": "top.sys_mem", "kind": "transparent",
                "type": "transparent_addr_space_c"}],
    "regions": [{"id": 0, "space": 0, "tag": "ram", "addr": 4096, "size": 65536,
                 "allocatable": True, "trait": {"kind": 1}, "file": "t.pss", "line": 12}],
}


def _model(profile="debug", memmap=MEMMAP):
    m = ZbcModel(coros=[CoroDescriptor(name="t_a")], profile=profile)
    m.memmap = dict(memmap)
    return m


def test_memmap_round_trip():
    assert ZbcModel.from_bytes(_model().to_bytes()).memmap == MEMMAP


def test_the_image_has_one_memmap_record():
    img = _model().to_container()
    sec = [s for s in img.sections if s.kind == spec.sec_kind("ZBC_SEC_MEMMAP")]
    assert len(sec) == 1 and sec[0].count == 1
    assert sec[0].elem_size == spec.RECORDS_BY_NAME["zbc_memmap"].size() == 4


def test_no_spaces_no_memmap():
    for mm in ({}, {"spaces": [], "regions": []}):
        img = _model(memmap=mm).to_container()
        assert spec.sec_kind("ZBC_SEC_MEMMAP") not in [s.kind for s in img.sections]
    assert ZbcModel.from_bytes(_model(memmap={}).to_bytes()).memmap == {}


def test_runtime_image_has_no_memmap():
    m = _model(profile="runtime")
    m.coros[0].name = ""
    img = m.to_container()
    assert spec.sec_kind("ZBC_SEC_MEMMAP") not in [s.kind for s in img.sections]
