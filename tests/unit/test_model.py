"""P1-1 -- the in-memory ZBC model and its round-trip through the container."""

import pytest

from zuspec.be.bc.model import (
    Op, Instr, Block, CoroDescriptor, SolveProblem, ZbcModel,
    ConstPool, StringTable, TypeTable,
    ORCH_OPS, is_orchestration, INSTR_F_BLOCKING, INSTR_F_FROM_POOL,
)
from zuspec.be.bc.abi.value import ScalarType, ArrayType, StructType
from zuspec.be.bc.format.inspect import inspect_bytes


def _sample_model():
    root = CoroDescriptor(
        name="root",
        code=[
            Instr(Op.CONST, (0,), imm=5, src_ref=1),
            Instr(Op.ST_LOCAL, (0, 0), src_ref=1),
            Instr(Op.WAIT, (), imm=10, src_ref=2),
            Instr(Op.LD_LOCAL, (1, 0), src_ref=3),
            Instr(Op.RET, ()),
        ],
        blocks=[Block(0, 0, 3, int(Op.WAIT)), Block(1, 3, 5, 0)],
        frame_locals=["x"],
        src_ref=1,
    )
    child = CoroDescriptor(
        name="child",
        code=[
            Instr(Op.SOLVE, (0,), src_ref=4),
            Instr(Op.IMPORT, (1,), flags=INSTR_F_BLOCKING, src_ref=5),
            Instr(Op.RET, ()),
        ],
        blocks=[Block(0, 0, 2, int(Op.IMPORT)), Block(1, 2, 3, 0)],
        src_ref=4,
    )
    m = ZbcModel(coros=[root, child], entry_coro=0)
    m.consts.add((1 << 100) | 7, 128)
    return m


def test_roundtrip_identity():
    m = _sample_model()
    assert ZbcModel.from_bytes(m.to_bytes()) == m


def test_reserialize_is_byte_stable():
    m = _sample_model()
    data = m.to_bytes()
    assert ZbcModel.from_bytes(data).to_bytes() == data


def test_problem_table_is_in_memory_only():
    m = _sample_model()
    m.problems.append(SolveProblem(var_names=["a", "z", "m"]))
    # Not serialized; the round-tripped model has no problems but is still ==.
    m2 = ZbcModel.from_bytes(m.to_bytes())
    assert m2 == m
    assert m2.problems == []


def test_empty_model_roundtrips():
    m = ZbcModel(coros=[], entry_coro=0)
    assert ZbcModel.from_bytes(m.to_bytes()) == m


def test_const_bits_preserved():
    m = ZbcModel()
    m.consts.add((1 << 100) | 0xABCD, 128)
    m.consts.add(0xFF, 8)
    m2 = ZbcModel.from_bytes(m.to_bytes())
    assert m2.consts.entries == [((1 << 100) | 0xABCD, 128), (0xFF, 8)]


def test_frame_locals_roundtrip():
    m = ZbcModel(coros=[CoroDescriptor(name="c", frame_locals=["a", "b", "c"])])
    m2 = ZbcModel.from_bytes(m.to_bytes())
    assert m2.coros[0].frame_locals == ["a", "b", "c"]


def test_suspend_classification():
    assert Instr(Op.WAIT).is_suspend()
    assert Instr(Op.JOIN).is_suspend()
    assert Instr(Op.PAR).is_suspend()
    assert Instr(Op.YIELD).is_suspend()
    assert Instr(Op.IMPORT, flags=INSTR_F_BLOCKING).is_suspend()
    assert not Instr(Op.IMPORT).is_suspend()          # pure import, no suspend
    assert not Instr(Op.INVOKE).is_suspend()
    assert Instr(Op.INVOKE, flags=INSTR_F_BLOCKING).is_suspend()
    assert not Instr(Op.ADD).is_suspend()


def test_orchestration_set():
    assert is_orchestration(Op.SPAWN) and is_orchestration(Op.SOLVE)
    assert not is_orchestration(Op.ADD)
    assert Op.SOLVE in ORCH_OPS


def test_too_many_args_rejected():
    with pytest.raises(ValueError):
        Instr(Op.ADD, (1, 2, 3, 4, 5))


def test_const_from_pool_flag_roundtrips():
    m = ZbcModel(coros=[CoroDescriptor(name="c", code=[
        Instr(Op.CONST, (0,), imm=0, flags=INSTR_F_FROM_POOL),
        Instr(Op.RET),
    ], blocks=[Block(0, 0, 2, 0)])])
    ins = ZbcModel.from_bytes(m.to_bytes()).coros[0].code[0]
    assert ins.flags & INSTR_F_FROM_POOL


def test_inspect_shows_model_sections():
    dump = inspect_bytes(_sample_model().to_bytes())
    for kind in ("ZBC_SEC_CODE", "ZBC_SEC_CORO", "ZBC_SEC_BLOCK",
                 "ZBC_SEC_CONST", "ZBC_SEC_TYPE", "ZBC_SEC_STRB", "ZBC_SEC_STRO"):
        assert kind in dump


# --- table units --------------------------------------------------------- #

def test_string_table_roundtrip():
    st = StringTable()
    ids = [st.intern(s) for s in ("root", "child", "root", "")]
    assert ids[0] == 1 and ids[2] == 1 and ids[3] == 0  # dedup; "" is StrId 0
    st2 = StringTable.parse(st.strb_bytes(), st.stro_bytes())
    assert st2.get(1) == "root" and st2.get(2) == "child" and st2.get(0) == ""


def test_type_table_children_interned_first():
    tt = TypeTable()
    arr = ArrayType(ScalarType(16), 4)
    tid = tt.intern(arr)
    # element interned before the array
    assert tt._map[ScalarType(16)] < tid


def test_type_table_struct_roundtrip():
    st = StringTable()
    tt = TypeTable()
    struct = StructType((("a", ScalarType(8)), ("b", ScalarType(32, signed=True))))
    tt.intern(struct)
    tt2 = TypeTable.parse(tt.to_bytes(st), len(tt.types), st)
    assert struct in tt2._map
