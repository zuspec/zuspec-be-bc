"""ScSelect lowering (deferred item #2): weighted-choice SELECT.

Each branch becomes a synthesized sub-coroutine; the ScSelect lowers to a SELECT op
backed by an in-memory branch table (indices + weights). The VM draws one branch in
declaration order against cumulative weights using the frame's seed stream, then runs
it as a blocking child. The load-bearing property is that the oracle's choice matches
the determinism primitive exactly.
"""

import pytest

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import stmt as S
from zuspec.ir.core import expr as E

from zuspec.be.bc.interp import run_scenario, Obj, VMError
from zuspec.be.bc.lower import lower_scenario, LoweringError
from zuspec.be.bc.model import Op
from zuspec.be.bc.determinism import SeedStream, select_choice


def _set_field(idx, value):
    return SC.ScExecBlock(kind="body", stmts=[
        S.StmtAssign(targets=[E.ExprRefField(base=E.TypeExprRefSelf(), index=idx)],
                     value=E.ExprConstant(value=value))])


def _field(idx):
    return E.ExprRefField(base=E.TypeExprRefSelf(), index=idx)


def _branch(sentinel, weight, body=None, guard=None):
    return SC.ScSelectBranch(
        guard=guard,
        weight=E.ExprConstant(value=weight),
        body=body if body is not None else [_set_field(0, sentinel)])


def _select_coro(sentinels, weights):
    branches = [_branch(s, w) for s, w in zip(sentinels, weights)]
    return SC.ScCoroutine(name="root", body=[SC.ScSelect(branches=branches)])


def test_select_choice_matches_determinism_primitive():
    # The oracle's chosen branch must equal determinism.select_choice for each seed.
    sentinels, weights = [10, 20, 30], [1, 1, 1]
    for seed in range(12):
        expected = select_choice(SeedStream(seed), weights)
        res = run_scenario([_select_coro(sentinels, weights)],
                           obj=Obj(field_names=["x"]), seed=seed)
        assert res.fields["x"] == sentinels[expected]
        assert res.frames == 2          # root + the one chosen branch


def test_select_weighted_choice_matches_primitive():
    sentinels, weights = [1, 2], [3, 7]
    for seed in range(12):
        expected = select_choice(SeedStream(seed), weights)
        res = run_scenario([_select_coro(sentinels, weights)],
                           obj=Obj(field_names=["x"]), seed=seed)
        assert res.fields["x"] == sentinels[expected]


def test_select_is_deterministic_same_seed():
    coro = _select_coro([10, 20, 30], [1, 1, 1])
    def go():
        return run_scenario([coro], obj=Obj(field_names=["x"]), seed=5).trace_json()
    assert go() == go()


def test_select_all_branches_reachable_across_seeds():
    coro = _select_coro([10, 20, 30], [1, 1, 1])
    seen = {run_scenario([coro], obj=Obj(field_names=["x"]), seed=s).fields["x"]
            for s in range(40)}
    assert seen == {10, 20, 30}


def test_select_lowers_to_select_op_and_branch_table():
    model = lower_scenario([_select_coro([10, 20, 30], [2, 3, 5])])
    assert len(model.coros) == 4                 # root + 3 branches
    assert [i.op for i in model.coros[0].code] == [Op.SELECT]
    assert len(model.selects) == 1
    assert model.selects[0].weights == [2, 3, 5]
    assert model.selects[0].branches == [1, 2, 3]


def test_select_branch_can_suspend():
    # A single branch that waits: always chosen, and its WAIT advances the clock.
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScSelect(branches=[_branch(0, 1, body=[SC.ScWait(time=E.ExprConstant(value=5))])])])
    res = run_scenario([coro], obj=Obj(field_names=[]), seed=1)
    assert res.now == 5
    assert res.frames == 2


def test_select_roundtrip_matches_inmemory():
    coro = _select_coro([10, 20, 30], [1, 2, 3])
    def go(rt):
        return run_scenario([coro], obj=Obj(field_names=["x"]), seed=9,
                            round_trip=rt).fields["x"]
    assert go(True) == go(False)


def test_parent_resumes_after_chosen_branch():
    # The parent must run its post-select work once the chosen branch completes.
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScSelect(branches=[_branch(0, 1, body=[_set_field(0, 7)])]),
        _set_field(1, 8),          # only runs if the parent resumes past the SELECT
    ])
    res = run_scenario([coro], obj=Obj(field_names=["x", "y"]), seed=3)
    assert res.fields == {"x": 7, "y": 8}


# --------------------------------------------------------------------------- #
# Guards + allow_none.
# --------------------------------------------------------------------------- #

def _guarded_coro(specs, allow_none=False):
    # specs: (sentinel, weight, guard_field_index_or_None); sentinel written to field0.
    branches = [_branch(sent, w, guard=(_field(g) if g is not None else None))
                for sent, w, g in specs]
    return SC.ScCoroutine(name="root", body=[
        SC.ScSelect(branches=branches, allow_none=allow_none)])


def test_guard_filters_to_eligible_branch():
    # branch0 guarded by field1, branch1 by field2; field1=0 -> only branch1 eligible.
    coro = _guarded_coro([(10, 1, 1), (20, 1, 2)])
    for seed in range(6):
        res = run_scenario([coro], obj=Obj(field_names=["x", "g0", "g1"],
                                           values=[0, 0, 1]), seed=seed)
        assert res.fields["x"] == 20


def test_guarded_choice_matches_primitive_over_eligible():
    # 3 branches guarded by fields 1..3; field2 off -> eligible = [0, 2].
    coro = _guarded_coro([(10, 1, 1), (20, 1, 2), (30, 1, 3)])
    sentinels = [10, 20, 30]
    eligible = [0, 2]
    for seed in range(12):
        local = select_choice(SeedStream(seed), [1, 1])   # weights of eligible
        res = run_scenario([coro], obj=Obj(field_names=["x", "g0", "g1", "g2"],
                                           values=[0, 1, 0, 1]), seed=seed)
        assert res.fields["x"] == sentinels[eligible[local]]


def test_allow_none_runs_nothing_when_no_branch_eligible():
    coro = _guarded_coro([(10, 1, 1), (20, 1, 2)], allow_none=True)
    res = run_scenario([coro], obj=Obj(field_names=["x", "g0", "g1"],
                                       values=[0, 0, 0]), seed=1)
    assert res.fields["x"] == 0          # nothing ran
    assert res.frames == 1               # no branch child spawned


def test_no_eligible_without_allow_none_is_vmerror():
    coro = _guarded_coro([(10, 1, 1)], allow_none=False)
    with pytest.raises(VMError):
        run_scenario([coro], obj=Obj(field_names=["x", "g0"], values=[0, 0]))


# --------------------------------------------------------------------------- #
# Deferred features reject cleanly.
# --------------------------------------------------------------------------- #

def test_non_constant_weight_rejected():
    br = SC.ScSelectBranch(weight=E.ExprRefField(base=E.TypeExprRefSelf(), index=0),
                           body=[_set_field(0, 1)])
    coro = SC.ScCoroutine(name="root", body=[SC.ScSelect(branches=[br])])
    with pytest.raises(LoweringError):
        lower_scenario([coro])
