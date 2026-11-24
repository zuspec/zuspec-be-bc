"""The REGMAP section: the host's register map survives write and read
(F1.7b)."""

from zuspec.be.bc.format import spec
from zuspec.be.bc.model import CoroDescriptor, ZbcModel

REGS = {
    3: {"path": "regs.ctrl", "comp": "", "reg": "regs.ctrl", "type": "ctrl_r",
        "comp_type": "top_c", "offset": 4, "width": 32, "access": "READWRITE",
        "fields": [{"name": "en", "lsb": 0, "width": 1, "kind": "RW"},
                   {"name": "mode", "lsb": 1, "width": 2, "kind": "RW",
                    "enum": {"A": 0, "B": 1}}]},
    1: {"path": "regs.stat", "comp": "", "reg": "regs.stat", "type": "stat_r",
        "comp_type": "top_c", "offset": 0, "width": 32, "access": "READONLY",
        "fields": []},
}


def _model(profile="debug", regs=REGS):
    m = ZbcModel(coros=[CoroDescriptor(name="t_a")], profile=profile)
    m.registers = dict(regs)
    return m


def test_regmap_round_trip():
    back = ZbcModel.from_bytes(_model().to_bytes())
    assert back.registers == REGS
    assert all(isinstance(k, int) for k in back.registers)


def test_the_image_has_one_regmap_record():
    img = _model().to_container()
    sec = [s for s in img.sections if s.kind == spec.sec_kind("ZBC_SEC_REGMAP")]
    assert len(sec) == 1 and sec[0].count == 1
    assert sec[0].elem_size == spec.RECORDS_BY_NAME["zbc_regmap"].size() == 4


def test_no_registers_no_regmap():
    img = _model(regs={}).to_container()
    assert spec.sec_kind("ZBC_SEC_REGMAP") not in [s.kind for s in img.sections]
    assert ZbcModel.from_bytes(_model(regs={}).to_bytes()).registers == {}


def test_runtime_image_has_no_regmap():
    m = _model(profile="runtime")
    m.coros[0].name = ""
    img = m.to_container()
    assert spec.sec_kind("ZBC_SEC_REGMAP") not in [s.kind for s in img.sections]
