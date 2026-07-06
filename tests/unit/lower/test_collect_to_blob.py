"""End-to-end constraint wiring: frontend-style constraint IR -> solved values.

This closes the loop the earlier slices left open. It starts from the shape the
*frontend* produces -- a rand-field ``DataTypeClass`` plus constraint ``Function``s
whose bodies are ``Stmt`` nodes with *name-based* field references
(``ExprAttribute(TypeExprRefSelf(), attr)``) -- and drives:

    collect_solve_problem   (ir-core: Stmt->Constraint + resolve refs to slots)
        -> build_solve_blob (be-bc: structured Constraint -> dv-solve blob)
        -> SolveCtx.solve   (dv-solve)

and asserts the solution satisfies the original constraint. That proves the
resolution + conversion feed the solver lowering correctly, not just that the
lowering works on hand-built ``ExprRefField`` IR.
"""

import ctypes

import pytest

pytest.importorskip("dv_solve.problem")

from zuspec.ir.core import expr as E
from zuspec.ir.core import stmt as S
from zuspec.ir.core.data_type import DataTypeClass, DataTypeInt, Function
from zuspec.ir.core.fields import Field, RandKind
from zuspec.ir.core.scenario import ScCoroutine
from zuspec.ir.core.xf.pss_lower.constraints import collect_solve_problem

from zuspec.be.bc.lower.constraints import build_solve_blob

from dv_solve.ctx import SolveCtx, SOLVE_OK


def _field(name, bits=8, rand=True):
    return Field(name=name, datatype=DataTypeInt(bits=bits),
                 rand_kind=RandKind.RAND if rand else None)

def SELF(attr):     return E.ExprAttribute(value=E.TypeExprRefSelf(), attr=attr)
def K(n):           return E.ExprConstant(value=n)
def BIN(l, op, r):  return E.ExprBin(lhs=l, op=op, rhs=r)


def _solve(dt, *stmts, seed=7):
    """collect -> blob -> solve; return {field_name: value}."""
    fn = Function(name="c", body=list(stmts), is_async=False,
                  metadata={"_is_constraint": True})
    coro = ScCoroutine(name="A", body=[], pending_constraints=[fn])
    problem = collect_solve_problem(coro, dt)
    blob, _ = build_solve_blob(problem)
    raw = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
    ctx = SolveCtx(raw)
    try:
        rc = ctx.solve(seed=seed)
        vals = {v.name: ctx.get_value(v.var_id) for v in problem.vars}
    finally:
        ctx.destroy()
    return rc, vals


def test_expr_constraint_end_to_end():
    dt = DataTypeClass(name="A", super=None, fields=[_field("x"), _field("y")])
    rc, v = _solve(dt,
                   S.StmtExpr(expr=BIN(BIN(SELF("x"), E.BinOp.Add, SELF("y")),
                                       E.BinOp.Eq, K(42))),
                   S.StmtExpr(expr=BIN(SELF("x"), E.BinOp.GtE, K(10))))
    assert rc == SOLVE_OK
    assert v["x"] + v["y"] == 42 and v["x"] >= 10


def test_if_else_constraint_end_to_end():
    # if (mode == 1) { x < 10 } else { x > 100 }, with mode pinned to 0 -> else.
    dt = DataTypeClass(name="A", super=None, fields=[_field("mode"), _field("x")])
    rc, v = _solve(
        dt,
        S.StmtExpr(expr=BIN(SELF("mode"), E.BinOp.Eq, K(0))),
        S.StmtIf(test=BIN(SELF("mode"), E.BinOp.Eq, K(1)),
                 body=[S.StmtExpr(expr=BIN(SELF("x"), E.BinOp.Lt, K(10)))],
                 orelse=[S.StmtExpr(expr=BIN(SELF("x"), E.BinOp.Gt, K(100)))]))
    assert rc == SOLVE_OK
    assert v["mode"] == 0 and v["x"] > 100


def test_implies_constraint_end_to_end():
    # (mode == 1) -> (x in [50..55]), mode pinned to 1.
    dt = DataTypeClass(name="A", super=None, fields=[_field("mode"), _field("x")])
    implies = E.ExprCall(func=E.ExprRefUnresolved(name="implies"), args=[
        BIN(SELF("mode"), E.BinOp.Eq, K(1)),
        E.ExprIn(value=SELF("x"), container=E.ExprRange(lower=K(50), upper=K(55)))])
    rc, v = _solve(dt,
                   S.StmtExpr(expr=BIN(SELF("mode"), E.BinOp.Eq, K(1))),
                   S.StmtExpr(expr=implies))
    assert rc == SOLVE_OK
    assert v["mode"] == 1 and 50 <= v["x"] <= 55


def test_unique_constraint_end_to_end():
    dt = DataTypeClass(name="A", super=None,
                       fields=[_field("x"), _field("y"), _field("z")])
    rc, v = _solve(dt, S.StmtUnique(vars=["x", "y", "z"]))
    assert rc == SOLVE_OK
    assert len({v["x"], v["y"], v["z"]}) == 3


def test_interleaved_nonrand_slot_end_to_end():
    # A non-rand pad interleaves the rand fields, so the constraint's field slots
    # differ from the solver var_ids; resolution + writeback must still line up.
    dt = DataTypeClass(name="A", super=None,
                       fields=[_field("pad", rand=False), _field("x"), _field("y")])
    rc, v = _solve(dt, S.StmtExpr(expr=BIN(SELF("x"), E.BinOp.Eq,
                                           BIN(SELF("y"), E.BinOp.Add, K(1)))),
                   S.StmtExpr(expr=BIN(SELF("y"), E.BinOp.GtE, K(5))))
    assert rc == SOLVE_OK
    assert v["x"] == v["y"] + 1 and v["y"] >= 5
