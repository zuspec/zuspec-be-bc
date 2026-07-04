"""T1-A (widened) -- differential over the *activity* vocabulary.

The atomic differential (``test_t1a_differential.py``) pins the import sequence of
a straight-line exec body. This file widens that to the compound-activity
constructs the oracle grew in M2 -- ``repeat`` / sequence / ``if`` / ``match`` /
``parallel`` / ``select`` -- driving one hand-built ``zuspec.ir.core`` activity
fixture through **both** runtimes and comparing the deterministic import-call
sequence.

Each atomic sub-action's body calls a single ``mark(k)`` target import with a
distinct ``k``, so the recorded call sequence names exactly which sub-actions ran
and in what order. The activity constructs are load-bearing here: the marks reveal
loop iteration counts, sequential ordering, branch/case selection, and parallel
membership -- all fully deterministic on both sides.

Two constructs are only *structurally* differential, by design:

* **parallel(ALL)** -- both runtimes run every branch, but the interleaving of a
  cooperative scheduler (oracle) vs. ``asyncio.gather`` (legacy) is not a contract;
  we compare the call *multiset*, not the order.
* **select** -- the legacy branch choice is unseeded ``random.choices`` (an
  acknowledged non-determinism, mirrored by the SELECT-join note in ``adapter.py``),
  so we assert only that *exactly one* eligible branch ran on each side.
"""

import pytest

pytest.importorskip("zuspec.be.py")

from zuspec.be.py.rt.legacy_status import legacy_reference_enabled

pytestmark = pytest.mark.skipif(
    not legacy_reference_enabled(),
    reason="legacy runtime disabled as differential reference (ZUSPEC_LEGACY_REFERENCE=0)")

import zuspec.ir.core as ir
from zuspec.ir.core.stmt import StmtExpr
from zuspec.ir.core.expr import ExprConstant, ExprAttribute, TypeExprRefSelf, ExprCall
from zuspec.ir.core.activity import (
    ActivitySequenceBlock, ActivityAnonTraversal, ActivityRepeat, ActivityIfElse,
    ActivityMatch, MatchCase, ActivityParallel, ActivitySelect, SelectBranch,
)
from zuspec.ir.core.activity import JoinSpec, JoinKind
from zuspec.be.py.rt.import_resolver import ImportSpec

from tests.diff.adapter import run_oracle, run_legacy


# --------------------------------------------------------------------------- #
# Fixture construction
# --------------------------------------------------------------------------- #

_MARK_SPECS = {"mark": ImportSpec(name="mark", is_target=True)}


class _Marker:
    """The import impl: ``mark(k)`` is a void target import (recorded by name+arg)."""

    def mark(self, k):
        pass


def _do(sub):
    return ActivityAnonTraversal(action_type=sub)


def _c(v):
    return ExprConstant(value=v)


def _sub(name, k):
    """An atomic sub-action whose body is `mark(k);`."""
    body = ir.Function(name="body", body=[
        StmtExpr(expr=ExprCall(
            func=ExprAttribute(value=TypeExprRefSelf(), attr="mark"),
            args=[_c(k)]))])
    return ir.DataTypeClass(name=name, super=None, functions=[body])


def _build_ctx(activity, subs):
    """Compound ``Root`` with *activity*, owning the given ``{name: k}`` sub-actions."""
    root = ir.DataTypeClass(name="Root", super=None)
    root.activity_ir = ActivitySequenceBlock(stmts=activity)
    top = ir.DataTypeComponent(name="Top", super=None)
    type_m = {"Top": top, "Top::Root": root}
    for name, k in subs.items():
        type_m["Top::" + name] = _sub(name, k)
    return ir.Context(type_m=type_m)


def _both(activity, subs):
    """Run the fixture through both runtimes; return (legacy_calls, oracle_calls)."""
    lf = run_legacy(_build_ctx(activity, subs), "Top::Root",
                    imports_impl=_Marker(), import_specs=_MARK_SPECS)
    of = run_oracle(_build_ctx(activity, subs), "Top::Root",
                    imports_impl=_Marker(), import_specs=_MARK_SPECS)
    return lf[1], of[1]


def _ks(calls):
    """Extract the ordered mark argument list from a call log."""
    return [a[0] for (n, a) in calls if n == "mark"]


# --------------------------------------------------------------------------- #
# Fully deterministic constructs -- exact ordered sequence must match.
# --------------------------------------------------------------------------- #

def test_repeat_traversal_sequence_matches():
    # repeat(3) { do A }  ->  mark(1) x3
    legacy, oracle = _both(
        [ActivityRepeat(count=_c(3), body=[_do("A")])],
        {"A": 1})
    assert _ks(legacy) == [1, 1, 1]
    assert _ks(oracle) == _ks(legacy)


def test_sequential_traversal_order_matches():
    # { do A; do B; do A }  ->  mark(1), mark(2), mark(1)  (ordering is load-bearing)
    legacy, oracle = _both(
        [_do("A"), _do("B"), _do("A")],
        {"A": 1, "B": 2})
    assert _ks(legacy) == [1, 2, 1]
    assert _ks(oracle) == _ks(legacy)


@pytest.mark.parametrize("cond,expect", [(1, [1]), (0, [2])])
def test_if_else_branch_selection_matches(cond, expect):
    # if (cond) { do A } else { do B }
    legacy, oracle = _both(
        [ActivityIfElse(condition=_c(cond), if_body=[_do("A")], else_body=[_do("B")])],
        {"A": 1, "B": 2})
    assert _ks(legacy) == expect
    assert _ks(oracle) == _ks(legacy)


@pytest.mark.parametrize("subject,expect", [(1, [1]), (2, [2]), (99, [3])])
def test_match_case_selection_matches(subject, expect):
    # match (subject) { 1: do A; 2: do B; default: do C }
    legacy, oracle = _both(
        [ActivityMatch(subject=_c(subject), cases=[
            MatchCase(pattern=_c(1), body=[_do("A")]),
            MatchCase(pattern=_c(2), body=[_do("B")]),
            MatchCase(pattern=None, body=[_do("C")]),
        ])],
        {"A": 1, "B": 2, "C": 3})
    assert _ks(legacy) == expect
    assert _ks(oracle) == _ks(legacy)


def test_nested_repeat_over_sequence_matches():
    # repeat(2) { do A; do B }  ->  1,2,1,2
    legacy, oracle = _both(
        [ActivityRepeat(count=_c(2), body=[_do("A"), _do("B")])],
        {"A": 1, "B": 2})
    assert _ks(legacy) == [1, 2, 1, 2]
    assert _ks(oracle) == _ks(legacy)


# --------------------------------------------------------------------------- #
# Structurally differential constructs -- membership / cardinality only.
# --------------------------------------------------------------------------- #

def test_parallel_all_runs_every_branch_matches():
    # parallel { do A; do B; do C }  -- both run all three; interleave is not pinned.
    legacy, oracle = _both(
        [ActivityParallel(stmts=[_do("A"), _do("B"), _do("C")],
                          join_spec=JoinSpec(kind=JoinKind.ALL))],
        {"A": 1, "B": 2, "C": 3})
    assert sorted(_ks(legacy)) == [1, 2, 3]
    assert sorted(_ks(oracle)) == sorted(_ks(legacy))


def test_select_runs_exactly_one_branch_matches():
    # select { do A; do B }  -- legacy uses unseeded random; assert cardinality only.
    legacy, oracle = _both(
        [ActivitySelect(branches=[
            SelectBranch(guard=None, weight=None, body=[_do("A")]),
            SelectBranch(guard=None, weight=None, body=[_do("B")]),
        ])],
        {"A": 1, "B": 2})
    assert len(_ks(legacy)) == 1 and _ks(legacy)[0] in (1, 2)
    assert len(_ks(oracle)) == 1 and _ks(oracle)[0] in (1, 2)


def test_guarded_select_only_eligible_branch_runs_matches():
    # select { [false]: do A; [true]: do B }  -- only B is eligible on both sides.
    legacy, oracle = _both(
        [ActivitySelect(branches=[
            SelectBranch(guard=_c(0), weight=None, body=[_do("A")]),
            SelectBranch(guard=_c(1), weight=None, body=[_do("B")]),
        ])],
        {"A": 1, "B": 2})
    assert _ks(legacy) == [2]
    assert _ks(oracle) == _ks(legacy)
