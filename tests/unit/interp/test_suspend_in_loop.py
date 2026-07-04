"""Suspend inside a loop / branch (Phase-5 territory, oracle path).

The oracle runs a flat, resumable code stream: a loop back-edge (BR) and an interior
suspend (blocking INVOKE / WAIT) interleave freely because the VM saves the pc across
a suspend and frame locals + registers persist. So `repeat(n) { do blocking }` and
friends execute directly, without the stackless FSM split (which is still deferred to
Phase 5 for the native backend).
"""

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import stmt as S
from zuspec.ir.core import expr as E

from zuspec.be.bc.interp import run_scenario, Obj
from zuspec.be.bc.lower import lower_scenario
from zuspec.be.bc.model import Op, INSTR_F_BLOCKING


def _act(wait=2):
    """A blocking sub-coroutine: it suspends (waits) then returns."""
    return SC.ScCoroutine(name="act", body=[SC.ScWait(time=E.ExprConstant(value=wait))])


def _invoke():
    return SC.ScInvoke(target="act")


def test_blocking_invoke_in_repeat_loop():
    # repeat 3 { do act }  -- three sequential blocking invokes: t = 3 * 2 = 6.
    root = SC.ScCoroutine(name="root", body=[
        SC.ScLoop(kind="repeat", count=E.ExprConstant(value=3), body=[_invoke()]),
    ])
    res = run_scenario([root, _act()], obj=Obj(field_names=[]),
                       blocking_targets=["act"])
    assert res.now == 6
    assert res.frames == 4          # root + 3 act invocations


def test_loop_index_persists_across_suspend():
    # acc = 0; repeat 3 index i { do act; acc = acc + i }; -> acc = 0+1+2 = 3.
    # The induction var i is read *after* the interior suspend each iteration.
    acc = E.ExprRefField(base=E.TypeExprRefSelf(), index=0)
    body = [
        _invoke(),
        SC.ScExecBlock(kind="body", stmts=[
            S.StmtAssign(targets=[acc],
                         value=E.ExprBin(lhs=acc, op=E.BinOp.Add,
                                         rhs=E.ExprRefLocal(name="i")))]),
    ]
    root = SC.ScCoroutine(name="root", frame_locals=["i"], body=[
        SC.ScLoop(kind="repeat", count=E.ExprConstant(value=3), index_var="i", body=body),
    ])
    res = run_scenario([root, _act()], obj=Obj(field_names=["acc"]),
                       blocking_targets=["act"])
    assert res.fields["acc"] == 3
    assert res.now == 6


def test_wait_in_whiledo_loop():
    # i = 0; while i < 3 { wait 5; i = i + 1 }  -- three waits: t = 15.
    inc = S.StmtAssign(targets=[E.ExprRefLocal(name="i")],
                       value=E.ExprBin(lhs=E.ExprRefLocal(name="i"), op=E.BinOp.Add,
                                       rhs=E.ExprConstant(value=1)))
    root = SC.ScCoroutine(name="root", frame_locals=["i"], body=[
        SC.ScExecBlock(kind="body", stmts=[
            S.StmtAssign(targets=[E.ExprRefLocal(name="i")], value=E.ExprConstant(value=0))]),
        SC.ScLoop(kind="whiledo",
                  cond=E.ExprCompare(left=E.ExprRefLocal(name="i"), ops=[E.CmpOp.Lt],
                                     comparators=[E.ExprConstant(value=3)]),
                  body=[SC.ScWait(time=E.ExprConstant(value=5)),
                        SC.ScExecBlock(kind="body", stmts=[inc])]),
    ])
    res = run_scenario([root], obj=Obj(field_names=[]))
    assert res.now == 15


def test_blocking_invoke_in_if_branch():
    # if field0 { do act } -- suspend only on the taken arm.
    root = SC.ScCoroutine(name="root", body=[
        SC.ScIf(cond=E.ExprRefField(base=E.TypeExprRefSelf(), index=0),
                then_body=[_invoke()]),
    ])
    hit = run_scenario([root, _act()], obj=Obj(field_names=["c"], values=[1]),
                       blocking_targets=["act"])
    miss = run_scenario([root, _act()], obj=Obj(field_names=["c"], values=[0]),
                        blocking_targets=["act"])
    assert hit.now == 2 and hit.frames == 2
    assert miss.now == 0 and miss.frames == 1     # branch not taken: no invoke


def test_suspend_in_loop_lowers_flat_and_roundtrips():
    root = SC.ScCoroutine(name="root", body=[
        SC.ScLoop(kind="repeat", count=E.ExprConstant(value=2), body=[_invoke()]),
    ])
    model = lower_scenario([root, _act()], blocking_targets=["act"])
    code = model.coros[0].code
    # A back-edge BR and a BLOCKING invoke coexist in one coroutine's flat stream.
    assert any(i.op == Op.BR for i in code)
    assert any(i.op == Op.INVOKE and (i.flags & INSTR_F_BLOCKING) for i in code)
    a = run_scenario([root, _act()], obj=Obj(field_names=[]),
                     blocking_targets=["act"], round_trip=False).now
    b = run_scenario([root, _act()], obj=Obj(field_names=[]),
                     blocking_targets=["act"], round_trip=True).now
    assert a == b == 4
