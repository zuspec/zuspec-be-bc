"""ScMatch lowering (activity vocabulary): multi-way branch -> compare/BRZ chain.

Non-suspending, procedural (like ScIf). The subject is evaluated once; each case
compares for equality and branches over its body on a miss; a default case
(pattern=None) runs when nothing matched.
"""

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import stmt as S
from zuspec.ir.core import expr as E

from zuspec.be.bc.interp import run_scenario, Obj


def _set(name, v):
    return SC.ScExecBlock(kind="body", stmts=[
        S.StmtAssign(targets=[E.ExprRefLocal(name=name)], value=E.ExprConstant(value=v))])


def _match_coro(default_body=None):
    # r = 0; match (field0) { 1: r=10; 2: r=20; [default]: r=99 }; return r
    cases = [
        SC.ScMatchCase(pattern=E.ExprConstant(value=1), body=[_set("r", 10)]),
        SC.ScMatchCase(pattern=E.ExprConstant(value=2), body=[_set("r", 20)]),
    ]
    if default_body is not None:
        cases.append(SC.ScMatchCase(pattern=None, body=default_body))
    return SC.ScCoroutine(name="root", frame_locals=["r"], body=[
        _set("r", 0),
        SC.ScMatch(subject=E.ExprRefField(base=E.TypeExprRefSelf(), index=0), cases=cases),
        SC.ScExecBlock(kind="body", stmts=[S.StmtReturn(value=E.ExprRefLocal(name="r"))]),
    ])


def _run(coro, subject):
    return run_scenario([coro], obj=Obj(field_names=["s"], values=[subject])).retval


def test_match_selects_first_case():
    assert _run(_match_coro(), 1) == 10


def test_match_selects_second_case():
    assert _run(_match_coro(), 2) == 20


def test_match_falls_to_default():
    assert _run(_match_coro(default_body=[_set("r", 99)]), 7) == 99


def test_match_no_default_no_case_leaves_prior_value():
    # no match, no default -> r keeps its pre-match value (0)
    assert _run(_match_coro(), 7) == 0


def test_match_only_one_case_body_runs():
    # subject matches case 1; case 2's body must NOT run (would overwrite to 20)
    assert _run(_match_coro(default_body=[_set("r", 99)]), 1) == 10
