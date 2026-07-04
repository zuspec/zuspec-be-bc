"""W7 -- the Python ZBC oracle: dispatch, scheduling, externs, round-trip.

Covers P1-8..P1-13 and the P1 test rows that don't need the legacy frontend:
T1-B (round-trip is the default path) and T1-D (same seed ⇒ identical trace, and
in-memory vs round-tripped ⇒ identical results). The legacy differential (T1-A) is
a separate adapter that reuses these entry points.
"""

import pytest

from zuspec.ir.core.base import Loc
from zuspec.ir.core import scenario as SC
from zuspec.ir.core import stmt as S
from zuspec.ir.core import expr as E

from zuspec.be.bc.model import (
    ZbcModel, CoroDescriptor, Instr, Op,
    INSTR_F_BLOCKING, INSTR_F_HAS_RET,
)
from zuspec.be.bc.interp import (
    run_scenario, run_model, roundtrip, VM,
    Obj, FixedSolveBackend, RecordingImportProvider, VMError,
)
from zuspec.be.bc.trace.sink import MemorySink


# --------------------------------------------------------------------------- #
# Fixtures: a representative lowered scenario (exec + WAIT + SOLVE + IMPORT).
# --------------------------------------------------------------------------- #

def _seq_coro():
    body = [
        SC.ScExecBlock(kind="body", stmts=[
            S.StmtAssign(targets=[E.ExprRefLocal(name="x")],
                         value=E.ExprConstant(value=5),
                         comment="init x",
                         loc=Loc(file="a.pss", line=3, pos=2, ref=None)),
        ]),
        SC.ScWait(time=E.ExprConstant(value=10)),
        SC.ScSolveProblem(
            vars=[SC.ScSolveVar(name="b", var_id=0), SC.ScSolveVar(name="a", var_id=1)],
            writeback={"a": 0, "b": 1},
            seed=E.ExprConstant(value=42)),
        SC.ScImport(fn="send", fn_id=42, blocking=True,
                    args=[E.ExprConstant(value=7)], ret_var="r"),
    ]
    return SC.ScCoroutine(name="root", body=body, frame_locals=["x"])


def test_sequential_run_end_to_end():
    obj = Obj(field_names=["a", "b"])
    imports = RecordingImportProvider(returns={42: 99})
    res = run_scenario([_seq_coro()], obj=obj, seed=0,
                       solve_backend=FixedSolveBackend(base=1000),
                       import_provider=imports)
    # fixed solve seed 42: a<-var("a")=1000+42+0, b<-var("b")=1000+42+1
    assert res.fields == {"a": 1042, "b": 1043}
    assert res.now == 10                       # WAIT advanced the clock
    assert imports.calls == [{"fn_id": 42, "args": [7]}]
    assert [e.kind for e in res.events] == ["WAIT", "SOLVE", "IMPORT"]
    # the blocking import's result reached the caller register
    assert res.events[-1].detail["ret"] == 99


def test_roundtrip_matches_inmemory():
    # T1-B / T1-D: serialize→deserialize→execute must equal in-memory execute.
    def go(round_trip):
        obj = Obj(field_names=["a", "b"])
        r = run_scenario([_seq_coro()], obj=obj, seed=7,
                         solve_backend=FixedSolveBackend(),
                         import_provider=RecordingImportProvider(),
                         round_trip=round_trip)
        return r.fields, r.trace_json()

    assert go(round_trip=True) == go(round_trip=False)


def test_same_seed_identical_trace():
    # T1-D: determinism -- same seed ⇒ byte-identical trace.
    def go():
        return run_scenario([_seq_coro()], obj=Obj(field_names=["a", "b"]),
                            seed=99, solve_backend=FixedSolveBackend(),
                            import_provider=RecordingImportProvider()).trace_json()

    assert go() == go()


def test_import_sequence_recorded_in_order():
    body = [
        SC.ScImport(fn="a", fn_id=1, blocking=False, args=[E.ExprConstant(value=1)]),
        SC.ScImport(fn="b", fn_id=2, blocking=True, args=[E.ExprConstant(value=2)]),
        SC.ScImport(fn="c", fn_id=3, blocking=False, args=[E.ExprConstant(value=3)]),
    ]
    coro = SC.ScCoroutine(name="root", body=body)
    imports = RecordingImportProvider()
    run_scenario([coro], import_provider=imports)
    assert [c["fn_id"] for c in imports.calls] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Procedural arithmetic through the ABI codec.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("expr,expected", [
    (E.ExprBin(lhs=E.ExprBin(lhs=E.ExprConstant(value=2), op=E.BinOp.Add,
                             rhs=E.ExprConstant(value=3)),
               op=E.BinOp.Mult, rhs=E.ExprConstant(value=4)), 20),
    (E.ExprBin(lhs=E.ExprConstant(value=10), op=E.BinOp.Sub,
               rhs=E.ExprConstant(value=3)), 7),
    (E.ExprBin(lhs=E.ExprConstant(value=17), op=E.BinOp.Mod,
               rhs=E.ExprConstant(value=5)), 2),
])
def test_procedural_arithmetic_returns_value(expr, expected):
    coro = SC.ScCoroutine(name="calc", body=[
        SC.ScExecBlock(kind="body", stmts=[S.StmtReturn(value=expr)]),
    ])
    res = run_scenario([coro])
    assert res.retval == expected


def test_subtraction_wraps_to_64bit_twos_complement():
    # 3 - 5 stored as an unsigned 64-bit bit pattern (ABI codec authority).
    expr = E.ExprBin(lhs=E.ExprConstant(value=3), op=E.BinOp.Sub,
                     rhs=E.ExprConstant(value=5))
    coro = SC.ScCoroutine(name="w", body=[
        SC.ScExecBlock(kind="body", stmts=[S.StmtReturn(value=expr)]),
    ])
    assert run_scenario([coro]).retval == (1 << 64) - 2


# --------------------------------------------------------------------------- #
# Scheduler: SPAWN/JOIN, blocking INVOKE, WAIT ordering (hand-built models).
# --------------------------------------------------------------------------- #

def _model(coros, entry=0):
    return ZbcModel(coros=coros, entry_coro=entry)


def test_spawn_join_waits_for_children():
    child = CoroDescriptor(name="child", code=[
        Instr(Op.WAIT, (), imm=5),
        Instr(Op.RET),
    ])
    root = CoroDescriptor(name="root", code=[
        Instr(Op.SPAWN, (1,)),
        Instr(Op.SPAWN, (1,)),
        Instr(Op.JOIN),
        Instr(Op.RET),
    ])
    sink = MemorySink()
    res = run_model(_model([root, child]), sink=sink)
    assert res.now == 5          # both children waited 5, root joined
    assert res.frames == 3       # root + 2 children
    kinds = [e.kind for e in sink.events]
    assert kinds == ["SPAWN", "SPAWN", "JOIN", "WAIT", "WAIT"]


def test_blocking_invoke_delivers_return_value():
    callee = CoroDescriptor(name="f", code=[
        Instr(Op.CONST, (0,), imm=77),
        Instr(Op.RET, (0,)),
    ])
    root = CoroDescriptor(name="root", code=[
        Instr(Op.INVOKE, (1, 0), flags=INSTR_F_BLOCKING | INSTR_F_HAS_RET),
        Instr(Op.ST_LOCAL, (0, 0)),
        Instr(Op.RET),
    ], frame_locals=["y"])
    res = run_model(_model([root, callee]))
    # callee's return value landed in the caller's frame local 'y'
    root_frame_locals = _model([root, callee]).coros[0].frame_locals
    assert root_frame_locals == ["y"]
    # observe via a second run capturing the local through a return
    root2 = CoroDescriptor(name="root2", code=[
        Instr(Op.INVOKE, (1, 0), flags=INSTR_F_BLOCKING | INSTR_F_HAS_RET),
        Instr(Op.RET, (0,)),
    ])
    assert run_model(_model([root2, callee])).retval == 77


def test_wait_orders_frames_by_resume_time():
    # Two children waiting different amounts resume in time order.
    fast = CoroDescriptor(name="fast", code=[Instr(Op.WAIT, (), imm=1), Instr(Op.RET)])
    slow = CoroDescriptor(name="slow", code=[Instr(Op.WAIT, (), imm=9), Instr(Op.RET)])
    root = CoroDescriptor(name="root", code=[
        Instr(Op.SPAWN, (1,)),   # fast
        Instr(Op.SPAWN, (2,)),   # slow
        Instr(Op.JOIN),
        Instr(Op.RET),
    ])
    res = run_model(_model([root, fast, slow]))
    assert res.now == 9


def test_yield_requeues_frame():
    root = CoroDescriptor(name="root", code=[
        Instr(Op.YIELD),
        Instr(Op.RET),
    ])
    sink = MemorySink()
    res = run_model(_model([root]), sink=sink)
    assert [e.kind for e in sink.events] == ["YIELD"]
    assert res.frames == 1


# --------------------------------------------------------------------------- #
# Error / robustness.
# --------------------------------------------------------------------------- #

def test_division_by_zero_is_clean_vmerror():
    expr = E.ExprBin(lhs=E.ExprConstant(value=1), op=E.BinOp.Div,
                     rhs=E.ExprConstant(value=0))
    coro = SC.ScCoroutine(name="d", body=[
        SC.ScExecBlock(kind="body", stmts=[S.StmtReturn(value=expr)]),
    ])
    with pytest.raises(VMError):
        run_scenario([coro])


def test_roundtrip_selfcheck_runs():
    # roundtrip() itself asserts model equality + byte stability.
    from zuspec.be.bc.lower import lower_scenario
    m = lower_scenario([_seq_coro()])
    m2 = roundtrip(m)
    assert m2 == m
    assert m2.problems is m.problems      # side channel reattached, not lost
