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


def test_an_action_entry_keeps_its_type_and_object_layout():
    m = _model()
    m.entries["t_a"] = EntryPoint(coro=0, qname="pkg::t_a",
                                  obj=[("x", dict(U8, rand=True)), ("s.f", ENUM)])
    back = ZbcModel.from_bytes(m.to_bytes())
    assert back.entries["t_a"].qname == "pkg::t_a"
    assert back.entries["t_a"].obj == m.entries["t_a"].obj
    # ... and the action object gets its field names from it.
    assert back.obj_layouts == {0: ["x", "s.f"]}


def test_lowering_gives_each_action_entry_its_object_layout():
    from zuspec.ir.core import scenario as SC
    from zuspec.ir.core.data_type import DataTypeInt
    from zuspec.be.bc.lower import lower_scenario
    coro = SC.ScCoroutine(name="t_a", body=[], action_type="pkg::t_a", fields=[
        SC.ScField(name="b", slot=1, datatype=DataTypeInt(bits=8, signed=False)),
        SC.ScField(name="a", slot=0, datatype=DataTypeInt(bits=16), rand=True)])
    m = lower_scenario([coro])
    e = m.entries["t_a"]
    assert e.qname == "pkg::t_a"
    assert e.obj == [("a", {"kind": "int", "width": 16, "signed": True, "rand": True}),
                     ("b", U8)]
    assert m.obj_layouts == {0: ["a", "b"]}
    assert ZbcModel.from_bytes(m.to_bytes()).entries["t_a"] == e
