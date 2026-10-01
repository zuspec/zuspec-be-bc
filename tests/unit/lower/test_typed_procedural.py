"""Typed Layer-0 procedural lowering: LRM 8.7 widths/signedness on 64-bit registers.

Each case runs straight through lowering and the oracle and reads the value back
through message(), which is how the lowering is observed end to end.
"""

import pytest

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import stmt as S
from zuspec.ir.core import expr as E
from zuspec.ir.core.data_type import DataTypeInt

from zuspec.be.bc.interp import run_model
from zuspec.be.bc.lower import LoweringError, lower_scenario
from zuspec.be.bc.lower.errors import PssSemanticError


def ty(bits, signed):
    return DataTypeInt(bits=bits, signed=signed)


def c(v):
    return E.ExprConstant(value=v)


def loc(n):
    return E.ExprRefLocal(name=n)


def decl(n, t, v=None):
    return S.StmtAnnAssign(target=loc(n), annotation=t, value=v)


def bin_(a, op, b):
    return E.ExprBin(lhs=a, op=op, rhs=b)


def msg(fmt, *args):
    return S.StmtExpr(expr=E.ExprCall(func=E.ExprRefUnresolved(name="message"),
                                      args=[c(0), c(fmt)] + list(args)))


def run(*stmts, functions=None):
    coro = SC.ScCoroutine(name="A", body=[SC.ScExecBlock(kind="body", stmts=list(stmts))])
    lines = []
    run_model(lower_scenario([coro], functions=functions), out=lines.append)
    return lines


def test_assignment_truncates_to_the_target():
    assert run(decl("x", ty(8, False), c(300)), decl("z", ty(8, True), c(200)),
               msg("%u %d", loc("x"), loc("z"))) == ["44 -56"]


def test_widening_follows_the_source_signedness():
    assert run(decl("a", ty(8, True), E.ExprUnary(op=E.UnaryOp.USub, operand=c(3))),
               decl("f", ty(16, False), loc("a")),
               msg("%u", loc("f"))) == ["65533"]


def test_mixed_compare_is_unsigned():
    assert run(decl("x", ty(32, False), c(1)),
               decl("y", ty(32, True), E.ExprUnary(op=E.UnaryOp.USub, operand=c(10))),
               decl("b", DataTypeInt(name="bool", bits=1, signed=False),
                    bin_(loc("x"), E.BinOp.Lt, loc("y"))),
               msg("%n", loc("b"))) == ["true"]


def test_signed_division_truncates_toward_zero():
    neg7 = E.ExprUnary(op=E.UnaryOp.USub, operand=c(7))
    assert run(decl("a", ty(32, True), neg7),
               msg("%d %d", bin_(loc("a"), E.BinOp.Div, c(2)),
                   bin_(loc("a"), E.BinOp.Mod, c(2)))) == ["-3 -1"]


def test_arithmetic_right_shift():
    assert run(decl("a", ty(32, True), E.ExprUnary(op=E.UnaryOp.USub, operand=c(16))),
               msg("%d", bin_(loc("a"), E.BinOp.RShift, c(2)))) == ["-4"]


def test_a_wider_target_propagates_its_size():
    x, y = decl("x", ty(8, False), c(0xF0)), decl("y", ty(12, False), c(0xFF0))
    s = bin_(loc("x"), E.BinOp.Add, loc("y"))
    assert run(x, y, decl("z", ty(16, False), s), msg("%u %u", loc("z"), s)) == ["4320 224"]


def test_break_and_continue():
    body = [S.StmtIf(test=bin_(loc("i"), E.BinOp.Eq, c(1)), body=[S.StmtContinue()]),
            S.StmtIf(test=bin_(loc("i"), E.BinOp.Eq, c(3)), body=[S.StmtBreak()]),
            msg("%d", loc("i"))]
    assert run(S.StmtFor(target=loc("i"), iter=c(10), body=body)) == ["0", "2"]


def test_functions_are_inlined_with_by_value_parameters():
    from zuspec.ir.core import Function
    from zuspec.ir.core.stmt import Arg, Arguments
    inc = Function(name="inc", args=Arguments(args=[Arg(arg="x", annotation=ty(32, True))]),
                   returns=ty(32, True),
                   body=[S.StmtAugAssign(target=E.ExprAttribute(value=E.TypeExprRefSelf(), attr="x"),
                                         op=E.AugOp.Add, value=c(1)),
                         S.StmtReturn(value=E.ExprAttribute(value=E.TypeExprRefSelf(), attr="x"))])
    call = E.ExprCall(func=E.ExprRefUnresolved(name="inc"), args=[loc("a")])
    assert run(decl("a", ty(32, True), c(5)), decl("b", ty(32, True), call),
               msg("%d %d", loc("a"), loc("b")), functions={"inc": inc}) == ["5 6"]


def test_unbounded_recursion_is_a_run_time_error_not_a_hang():
    """A recursive call is a CALL (bc procedural gaps B-D5); the depth limit
    stops one that never ends."""
    from zuspec.ir.core import Function
    from zuspec.ir.core.stmt import Arguments
    from zuspec.be.bc.interp import VMError
    f = Function(name="f", args=Arguments(), returns=None,
                 body=[S.StmtExpr(expr=E.ExprCall(func=E.ExprRefUnresolved(name="f"), args=[]))])
    with pytest.raises(VMError, match="nested deeper"):
        run(S.StmtExpr(expr=E.ExprCall(func=E.ExprRefUnresolved(name="f"), args=[])),
            functions={"f": f})


def test_a_bad_format_is_a_semantic_error():
    with pytest.raises(PssSemanticError):
        run(msg("%d %d", c(1)))


def test_match_with_no_arm_and_no_default_is_a_runtime_error():
    from zuspec.be.bc.interp import VMError
    m = S.StmtMatch(subject=c(9), cases=[
        S.StmtMatchCase(pattern=S.PatternValue(value=c(1)), body=[msg("one")])])
    with pytest.raises(VMError, match="no branch"):
        run(m)
