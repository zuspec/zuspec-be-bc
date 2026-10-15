"""bc's constraint lowering is zuspec-solver's walk (its own tests are in
zuspec-solver ``tests/translate``); bc owns only how a refusal reads."""

import pytest

pytest.importorskip("dv_solve.builder")

from zuspec.ir.core import expr as E
from zuspec.ir.core import constraint as C
from zuspec.ir.core import scenario as SC

from zuspec.be.bc.lower.constraints import build_solve_blob
from zuspec.be.bc.lower.errors import LoweringError


def test_a_refused_constraint_is_a_lowering_error():
    p = SC.ScSolveProblem(
        vars=[SC.ScSolveVar(name="f0", var_id=0, width=8)],
        constraints=[C.ConstraintExpr(expr=E.ExprBin(
            lhs=E.ExprRefField(base=E.TypeExprRefSelf(), index=5),
            op=E.BinOp.Eq, rhs=E.ExprConstant(value=1)))])
    with pytest.raises(LoweringError, match="slot 5"):
        build_solve_blob(p)
