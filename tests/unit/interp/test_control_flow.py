"""Control-flow lowering (deferred item #1): ScIf / ScLoop -> BR/BRZ.

End-to-end through the oracle: the FSM pass keeps these constructs opaque within a
block (no suspend inside), the lowering emits code-absolute BR/BRZ, and the VM's
existing branch handler executes them. We assert on observable results (retval /
fields), so a wrong branch target or off-by-one iteration count fails loudly.
"""

import pytest

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import stmt as S
from zuspec.ir.core import expr as E

from zuspec.be.bc.interp import run_scenario, Obj
from zuspec.be.bc.lower import lower_scenario
from zuspec.be.bc.model import Op


# --------------------------------------------------------------------------- #
# Small IR builders.
# --------------------------------------------------------------------------- #

def _assign(name, value):
    return S.StmtAssign(targets=[E.ExprRefLocal(name=name)], value=value)


def _exec(*stmts):
    return SC.ScExecBlock(kind="body", stmts=list(stmts))


def _local(name):
    return E.ExprRefLocal(name=name)


def _const(v):
    return E.ExprConstant(value=v)


def _field(i):
    # base is ignored by the M1 lowering (only .index is used).
    return E.ExprRefField(base=E.TypeExprRefSelf(), index=i)


def _add(a, b):
    return E.ExprBin(lhs=a, op=E.BinOp.Add, rhs=b)


def _lt(a, b):
    return E.ExprCompare(left=a, ops=[E.CmpOp.Lt], comparators=[b])


def _ret(expr):
    return S.StmtReturn(value=expr)


# --------------------------------------------------------------------------- #
# ScIf
# --------------------------------------------------------------------------- #

def _if_coro():
    # r = 0; if field0: r = 100 else: r = 200; return r
    body = [
        _exec(_assign("r", _const(0))),
        SC.ScIf(
            cond=_field(0),
            then_body=[_exec(_assign("r", _const(100)))],
            else_body=[_exec(_assign("r", _const(200)))],
        ),
        _exec(_ret(_local("r"))),
    ]
    return SC.ScCoroutine(name="root", body=body, frame_locals=["r"])


@pytest.mark.parametrize("cond,expected", [(1, 100), (0, 200)])
def test_if_takes_correct_arm(cond, expected):
    res = run_scenario([_if_coro()], obj=Obj(field_names=["c"], values=[cond]))
    assert res.retval == expected


def test_if_without_else_falls_through():
    # r = 5; if field0: r = 99; return r  -> 99 when true, 5 when false
    def coro():
        return SC.ScCoroutine(name="root", frame_locals=["r"], body=[
            _exec(_assign("r", _const(5))),
            SC.ScIf(cond=_field(0),
                    then_body=[_exec(_assign("r", _const(99)))]),
            _exec(_ret(_local("r"))),
        ])
    assert run_scenario([coro()], obj=Obj(field_names=["c"], values=[1])).retval == 99
    assert run_scenario([coro()], obj=Obj(field_names=["c"], values=[0])).retval == 5


# --------------------------------------------------------------------------- #
# ScLoop
# --------------------------------------------------------------------------- #

def test_repeat_loop_iterates_count_times():
    # acc = 0; repeat 5: acc = acc + 1; return acc
    coro = SC.ScCoroutine(name="root", frame_locals=["acc"], body=[
        _exec(_assign("acc", _const(0))),
        SC.ScLoop(kind="repeat", count=_const(5),
                  body=[_exec(_assign("acc", _add(_local("acc"), _const(1))))]),
        _exec(_ret(_local("acc"))),
    ])
    assert run_scenario([coro]).retval == 5


def test_repeat_zero_count_skips_body():
    coro = SC.ScCoroutine(name="root", frame_locals=["acc"], body=[
        _exec(_assign("acc", _const(7))),
        SC.ScLoop(kind="repeat", count=_const(0),
                  body=[_exec(_assign("acc", _const(999)))]),
        _exec(_ret(_local("acc"))),
    ])
    assert run_scenario([coro]).retval == 7


def test_repeat_loop_index_var_is_live_in_body():
    # acc = 0; repeat 5 index i: acc = acc + i; return acc  -> 0+1+2+3+4 = 10
    coro = SC.ScCoroutine(name="root", frame_locals=["acc"], body=[
        _exec(_assign("acc", _const(0))),
        SC.ScLoop(kind="repeat", count=_const(5), index_var="i",
                  body=[_exec(_assign("acc", _add(_local("acc"), _local("i"))))]),
        _exec(_ret(_local("acc"))),
    ])
    assert run_scenario([coro]).retval == 10


def test_whiledo_loop():
    # i = 0; while i < 3: i = i + 1; return i  -> 3
    coro = SC.ScCoroutine(name="root", frame_locals=["i"], body=[
        _exec(_assign("i", _const(0))),
        SC.ScLoop(kind="whiledo", cond=_lt(_local("i"), _const(3)),
                  body=[_exec(_assign("i", _add(_local("i"), _const(1))))]),
        _exec(_ret(_local("i"))),
    ])
    assert run_scenario([coro]).retval == 3


def test_dowhile_runs_body_at_least_once():
    # i = 5; do i = i + 1 while i < 3; return i  -> body runs once despite false cond
    coro = SC.ScCoroutine(name="root", frame_locals=["i"], body=[
        _exec(_assign("i", _const(5))),
        SC.ScLoop(kind="dowhile", cond=_lt(_local("i"), _const(3)),
                  body=[_exec(_assign("i", _add(_local("i"), _const(1))))]),
        _exec(_ret(_local("i"))),
    ])
    assert run_scenario([coro]).retval == 6


# --------------------------------------------------------------------------- #
# Nesting: if inside a loop exercises control.lower recursion.
# --------------------------------------------------------------------------- #

def test_if_nested_in_loop():
    # acc = 0; repeat 3: if field0: acc = acc + 10; return acc  -> 30
    coro = SC.ScCoroutine(name="root", frame_locals=["acc"], body=[
        _exec(_assign("acc", _const(0))),
        SC.ScLoop(kind="repeat", count=_const(3), body=[
            SC.ScIf(cond=_field(0),
                    then_body=[_exec(_assign("acc", _add(_local("acc"), _const(10))))]),
        ]),
        _exec(_ret(_local("acc"))),
    ])
    assert run_scenario([coro], obj=Obj(field_names=["c"], values=[1])).retval == 30


# --------------------------------------------------------------------------- #
# Structure + round-trip.
# --------------------------------------------------------------------------- #

def test_loop_emits_branches_and_roundtrips():
    coro = SC.ScCoroutine(name="root", frame_locals=["acc"], body=[
        _exec(_assign("acc", _const(0))),
        SC.ScLoop(kind="repeat", count=_const(4),
                  body=[_exec(_assign("acc", _add(_local("acc"), _const(1))))]),
        _exec(_ret(_local("acc"))),
    ])
    model = lower_scenario([coro])
    ops = [i.op for i in model.coros[0].code]
    assert Op.BR in ops and Op.BRZ in ops
    # in-memory == round-tripped
    a = run_scenario([coro], round_trip=False).retval
    b = run_scenario([coro], round_trip=True).retval
    assert a == b == 4
