"""The ENTRY section: ``ZbcModel.entries`` survives write and read (pssc P1.3d2)."""

from zuspec.be.bc.format import spec
from zuspec.be.bc.interp import oracle
from zuspec.be.bc.model import CoroDescriptor, EntryPoint, ZbcModel

ENUM = {"kind": "enum", "width": 32, "signed": True, "items": [["A", 0], ["B", 1]]}
U8 = {"kind": "int", "width": 8, "signed": False}


def _model(profile="codegen"):
    # A runtime image has no string table, so its coroutines are unnamed.
    names = ("t_a", "dev_c::f") if profile == "codegen" else ("", "")
    m = ZbcModel(coros=[CoroDescriptor(name=n) for n in names], profile=profile)
    m.entries = {
        "t_a": EntryPoint(coro=0),
        "dev_c::f": EntryPoint(coro=1, kind="function", comp_type="dev_c",
                               params=[("bank", ENUM), ("cfg.len", U8)], returns=U8),
    }
    return m


def test_entries_round_trip():
    m = _model()
    back = ZbcModel.from_bytes(m.to_bytes())
    assert back.entries == m.entries
    assert list(back.entries) == ["t_a", "dev_c::f"]


def test_the_image_has_one_entry_record_per_entry():
    img = _model().to_container()
    sec = [s for s in img.sections if s.kind == spec.sec_kind("ZBC_SEC_ENTRY")]
    assert len(sec) == 1 and sec[0].count == 2
    assert sec[0].elem_size == spec.RECORDS_BY_NAME["zbc_entry"].size() == 32


def test_no_entries_no_section():
    img = ZbcModel(coros=[CoroDescriptor(name="t_a")]).to_container()
    assert spec.sec_kind("ZBC_SEC_ENTRY") not in [s.kind for s in img.sections]


def test_oracle_roundtrip_keeps_entries_in_both_profiles():
    for profile in ("codegen", "runtime"):
        m = _model(profile)
        assert oracle.roundtrip(m).entries == m.entries
