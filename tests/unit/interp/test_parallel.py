"""ScPar lowering (deferred item #2): parallel branches -> SPAWN/JOIN.

Each branch becomes a synthesized sub-coroutine; the ScPar desugars to one SPAWN
per branch + a JOIN. End-to-end through the oracle asserts the branches actually
run concurrently under the shared object and the parent blocks until all finish.
"""

import pytest

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import stmt as S
from zuspec.ir.core import expr as E
from zuspec.ir.core.activity import JoinSpec, JoinKind

from zuspec.be.bc.interp import run_scenario, Obj
from zuspec.be.bc.lower import lower_scenario, LoweringError
from zuspec.be.bc.model import Op


def _set_field(idx, value):
    return SC.ScExecBlock(kind="body", stmts=[
        S.StmtAssign(targets=[E.ExprRefField(base=E.TypeExprRefSelf(), index=idx)],
                     value=E.ExprConstant(value=value))])


def test_parallel_all_branches_run():
    # parallel { field0 = 11; field1 = 22 }  -- both branches touch the shared obj
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScPar(branches=[_set_field(0, 11), _set_field(1, 22)]),
    ])
    res = run_scenario([coro], obj=Obj(field_names=["a", "b"]))
    assert res.fields == {"a": 11, "b": 22}
    assert res.frames == 3          # root + 2 branch children


def test_parallel_join_waits_for_slowest_branch():
    # parallel { wait 3; wait 7 } -- parent resumes at t=7
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScPar(branches=[
            SC.ScWait(time=E.ExprConstant(value=3)),
            SC.ScWait(time=E.ExprConstant(value=7)),
        ]),
    ])
    res = run_scenario([coro], obj=Obj(field_names=[]))
    assert res.now == 7


def test_parallel_desugars_to_spawn_join_and_synthesizes_branches():
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScPar(branches=[_set_field(0, 1), _set_field(1, 2), _set_field(2, 3)]),
    ])
    model = lower_scenario([coro])
    assert len(model.coros) == 4                     # root + 3 branches
    ops = [i.op for i in model.coros[0].code]
    assert ops.count(Op.SPAWN) == 3 and ops.count(Op.JOIN) == 1
    # branch coros carry a derived, unique name and follow the top-level coro
    assert model.coros[0].name == "root"
    assert all(c.name.startswith("root$par") for c in model.coros[1:])


def test_parallel_runs_after_and_before_sequential_work():
    # x = field0; parallel { field1 = 5; field2 = 6 } (sequential prefix then fork)
    coro = SC.ScCoroutine(name="root", body=[
        _set_field(0, 9),
        SC.ScPar(branches=[_set_field(1, 5), _set_field(2, 6)]),
    ])
    res = run_scenario([coro], obj=Obj(field_names=["a", "b", "c"]))
    assert res.fields == {"a": 9, "b": 5, "c": 6}


def test_parallel_roundtrip_matches_inmemory():
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScPar(branches=[_set_field(0, 11), _set_field(1, 22)]),
    ])
    def go(rt):
        return run_scenario([coro], obj=Obj(field_names=["a", "b"]),
                            round_trip=rt).fields
    assert go(True) == go(False) == {"a": 11, "b": 22}


@pytest.mark.parametrize("kind", [JoinKind.SELECT, JoinKind.BRANCH])
def test_unsupported_join_policies_rejected(kind):
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScPar(branches=[_set_field(0, 1)],
                 join_spec=JoinSpec(kind=kind)),
    ])
    with pytest.raises(LoweringError):
        lower_scenario([coro])


def _wait(t):
    return SC.ScWait(time=E.ExprConstant(value=t))


def test_join_none_does_not_block_parent():
    # parallel(NONE) { wait 5 }; wait 1  -- parent resumes at t=1, branch detaches
    # and finishes at t=5 in the drain. (ALL would give t=6.)
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScPar(branches=[_wait(5)], join_spec=JoinSpec(kind=JoinKind.NONE)),
        _wait(1),
    ])
    res = run_scenario([coro], obj=Obj(field_names=[]))
    assert res.now == 5
    assert res.frames == 2


@pytest.mark.parametrize("n,resume", [(1, 1), (2, 5)])
def test_join_first_n_resumes_after_n_complete(n, resume):
    # parallel(FIRST n) { wait 1; wait 5; wait 20 }; wait 100
    # Parent resumes when the n-th branch completes, then waits 100. The trailing
    # wait dominates the detached branches, so `now` reveals the resume instant.
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScPar(branches=[_wait(1), _wait(5), _wait(20)],
                 join_spec=JoinSpec(kind=JoinKind.FIRST, count=E.ExprConstant(value=n))),
        _wait(100),
    ])
    res = run_scenario([coro], obj=Obj(field_names=[]))
    assert res.now == resume + 100
    assert res.frames == 4


def test_join_first_cancels_surplus_side_effects():
    # FIRST(1) { (wait 1; a=1) ; (wait 10; b=2) } -- branch A wins at t=1; branch B
    # is cancelled before its wait completes, so its write (b=2) never happens.
    def timed_write(delay, idx, val):
        return SC.ScSeq(body=[_wait(delay), _set_field(idx, val)])
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScPar(branches=[timed_write(1, 0, 1), timed_write(10, 1, 2)],
                 join_spec=JoinSpec(kind=JoinKind.FIRST, count=E.ExprConstant(value=1))),
    ])
    res = run_scenario([coro], obj=Obj(field_names=["a", "b"]))
    assert res.fields == {"a": 1, "b": 0}     # B cancelled: b stays 0
    assert res.now == 1                        # cancelled wait-10 never advances now


def test_join_first_requires_constant_count():
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScPar(branches=[_wait(1)],
                 join_spec=JoinSpec(kind=JoinKind.FIRST,
                                    count=E.ExprRefField(base=E.TypeExprRefSelf(), index=0))),
    ])
    with pytest.raises(LoweringError):
        lower_scenario([coro])


def test_nested_parallel():
    # parallel { parallel { field0=1; field1=2 }; field2=3 }
    inner = SC.ScPar(branches=[_set_field(0, 1), _set_field(1, 2)])
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScPar(branches=[inner, _set_field(2, 3)]),
    ])
    res = run_scenario([coro], obj=Obj(field_names=["a", "b", "c"]))
    assert res.fields == {"a": 1, "b": 2, "c": 3}
