"""T1-A -- differential: legacy ``zuspec-be-py`` runtime vs. the ZBC oracle.

One hand-built ``zuspec.ir.core`` fixture drives both runtimes; the adapter
compares the import call sequence (deterministic, load-bearing) and the solved
field set (pinned-solver write-back routing). See ``adapter.py`` for the M1
comparison contract and why bit-exact independent solving is not asserted.
"""

import pytest

# The differential reference is a test-only dependency; skip cleanly if absent.
pytest.importorskip("zuspec.be.py")

# W8: the legacy runtime is retained only as the gated differential reference.
from zuspec.be.py.rt.legacy_status import legacy_reference_enabled

pytestmark = pytest.mark.skipif(
    not legacy_reference_enabled(),
    reason="legacy runtime disabled as differential reference (ZUSPEC_LEGACY_REFERENCE=0)")

import zuspec.ir.core as ir
from zuspec.ir.core.stmt import StmtExpr
from zuspec.ir.core.expr import (
    ExprConstant, ExprAttribute, TypeExprRefSelf, ExprCall,
)
from zuspec.be.py.rt.import_resolver import ImportSpec

from tests.diff.adapter import differential, run_oracle, run_legacy


# --------------------------------------------------------------------------- #
# Fixtures (mirror zuspec-be-py/tests/unit/test_export_api.py)
# --------------------------------------------------------------------------- #

def _int(bits=8, signed=False):
    return ir.DataTypeInt(bits=bits, signed=signed)


def _self_call(name, *args):
    return ExprCall(func=ExprAttribute(value=TypeExprRefSelf(), attr=name),
                    args=list(args))


def _ctx_atomic():
    """component Top { action Go { rand bit[8] addr; } }"""
    go = ir.DataTypeClass(
        name="Go", super=None,
        fields=[ir.Field(name="addr", datatype=_int(8), rand_kind=ir.RandKind.RAND)])
    top = ir.DataTypeComponent(name="Top", super=None)
    return ir.Context(type_m={"Top": top, "Top::Go": go})


def _ctx_with_imports():
    """action Go { exec body { doit(getval(7)); } } with target/solve imports."""
    body = ir.Function(name="body", body=[
        StmtExpr(expr=_self_call("doit", _self_call("getval", ExprConstant(value=7))))])
    go = ir.DataTypeClass(name="Go", super=None, functions=[body])
    top = ir.DataTypeComponent(name="Top", super=None)
    return ir.Context(type_m={"Top": top, "Top::Go": go})


_IMPORT_SPECS = {
    "doit": ImportSpec(name="doit", is_target=True),
    "getval": ImportSpec(name="getval", is_solve=True),
}


class _Imp:
    def getval(self, i):
        return i + 5

    def doit(self, i):
        pass


# --------------------------------------------------------------------------- #
# Import sequence -- the load-bearing, fully deterministic differential.
# --------------------------------------------------------------------------- #

def test_import_sequence_matches_legacy():
    d = differential(_ctx_with_imports(), "Top::Go",
                     imports_impl=_Imp(), import_specs=_IMPORT_SPECS)
    # Both runtimes call getval(7) -> 12, then doit(12), in that order.
    assert d.legacy_calls == [("getval", [7]), ("doit", [12])]
    assert d.oracle_calls == d.legacy_calls


def test_import_nonblocking_result_feeds_blocking_call():
    # The value returned by the non-blocking `getval` must reach the blocking
    # `doit` -- proves the IMPORT return register threads through the nested call.
    _, calls = run_oracle(_ctx_with_imports(), "Top::Go",
                          imports_impl=_Imp(), import_specs=_IMPORT_SPECS)
    doit = next(a for n, a in calls if n == "doit")
    assert doit == [12]


# --------------------------------------------------------------------------- #
# Solve -- write-back routing (pinned solver) + oracle self-determinism.
# --------------------------------------------------------------------------- #

def test_atomic_solve_field_writeback_matches_legacy():
    d = differential(_ctx_atomic(), "Top::Go", seed=42)
    # Legacy solves exactly the rand field set...
    assert set(d.legacy_fields) == {"addr"}
    assert 0 <= d.legacy_fields["addr"] <= 255
    # ...and the oracle routes the (shared-solver) value to the same field.
    assert d.oracle_fields == d.legacy_fields


def test_oracle_solve_is_self_deterministic():
    # T1-D via the adapter: same seed + same solver values ⇒ identical result.
    ctx1, ctx2 = _ctx_atomic(), _ctx_atomic()
    f1, _ = run_oracle(ctx1, "Top::Go", seed=7, solve_values={"addr": 200})
    f2, _ = run_oracle(ctx2, "Top::Go", seed=7, solve_values={"addr": 200})
    assert f1 == f2 == {"addr": 200}


def test_legacy_and_oracle_both_complete_atomic():
    # Smoke: the whole path (lower -> serialize round-trip is exercised elsewhere;
    # here we assert both runtimes produce a field dict for the same fixture).
    lf, _ = run_legacy(_ctx_atomic(), "Top::Go", seed=1)
    of, _ = run_oracle(_ctx_atomic(), "Top::Go", seed=1, solve_values=lf)
    assert set(lf) == set(of) == {"addr"}
