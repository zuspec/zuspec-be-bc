"""
procedural.py -- Layer-0 Stmt/Expr -> register-SSA procedural ops (P1-3).

M1 covers the common cases the differential corpus exercises: integer literals,
local/field references, binary/unary/compare expressions, and assign/expr/return
statements. Unsupported nodes raise :class:`LoweringError` (never a silent
miscompile). Operator semantics are intended to match the legacy ``Executor`` so
the oracle reproduces legacy arithmetic exactly.
"""

from zuspec.ir.core import expr as E
from zuspec.ir.core import stmt as S

from ..model import Op, INSTR_F_HAS_RET, INSTR_F_BLOCKING
from .context import CoroCtx
from .errors import LoweringError

_VOID = 0xFFFFFFFF


def _import_call_name(func):
    """Resolve the import-function name a call refers to, or ``None``.

    Matches the two shapes the frontend produces: ``self.<name>(...)``
    (``ExprAttribute`` on ``TypeExprRefSelf``) and a bare unresolved ``<name>(...)``
    (``ExprRefUnresolved``) -- the same forms ``pss_lower._as_blocking_import`` keys on.
    """
    if isinstance(func, E.ExprAttribute) and isinstance(func.value, E.TypeExprRefSelf):
        return func.attr
    if isinstance(func, E.ExprRefUnresolved):
        return func.name
    return None

_BINOP = {
    E.BinOp.Add: Op.ADD, E.BinOp.Sub: Op.SUB, E.BinOp.Mult: Op.MUL,
    E.BinOp.Div: Op.DIV, E.BinOp.FloorDiv: Op.DIV, E.BinOp.Mod: Op.MOD,
    E.BinOp.BitAnd: Op.AND, E.BinOp.BitOr: Op.OR, E.BinOp.BitXor: Op.XOR,
    E.BinOp.LShift: Op.SHL, E.BinOp.RShift: Op.SHR,
    E.BinOp.Eq: Op.CMP_EQ, E.BinOp.NotEq: Op.CMP_NE,
    E.BinOp.Lt: Op.CMP_LT, E.BinOp.LtE: Op.CMP_LE,
    E.BinOp.Gt: Op.CMP_GT, E.BinOp.GtE: Op.CMP_GE,
    E.BinOp.And: Op.AND, E.BinOp.Or: Op.OR,
}

_CMPOP = {
    E.CmpOp.Eq: Op.CMP_EQ, E.CmpOp.NotEq: Op.CMP_NE,
    E.CmpOp.Lt: Op.CMP_LT, E.CmpOp.LtE: Op.CMP_LE,
    E.CmpOp.Gt: Op.CMP_GT, E.CmpOp.GtE: Op.CMP_GE,
}

_BOOLOP = {E.BoolOp.And: Op.AND, E.BoolOp.Or: Op.OR}


def eval_expr(ctx: CoroCtx, expr) -> int:
    """Lower ``expr``, returning the register that holds its value."""
    if isinstance(expr, E.ExprConstant):
        v = expr.value
        if isinstance(v, bool):
            v = int(v)
        if not isinstance(v, int):
            raise LoweringError(
                f"non-integer constant {v!r} not supported in M1",
                loc=getattr(expr, "loc", None),
            )
        return ctx.emit_const(v)

    if isinstance(expr, E.ExprRefLocal):
        r = ctx.new_reg()
        ctx.emit(Op.LD_LOCAL, (r, ctx.local_slot(expr.name)))
        return r

    if isinstance(expr, E.ExprRefField):
        r = ctx.new_reg()
        ctx.emit(Op.LD_FIELD, (r, expr.index))
        return r

    if isinstance(expr, E.ExprBin):
        a = eval_expr(ctx, expr.lhs)
        b = eval_expr(ctx, expr.rhs)
        op = _BINOP.get(expr.op)
        if op is None:
            raise LoweringError(f"unsupported binop {expr.op}",
                                loc=getattr(expr, "loc", None))
        r = ctx.new_reg()
        ctx.emit(op, (r, a, b))
        return r

    if isinstance(expr, E.ExprUnary):
        operand = eval_expr(ctx, expr.operand)
        if expr.op == E.UnaryOp.USub:
            r = ctx.new_reg()
            ctx.emit(Op.NEG, (r, operand))
            return r
        if expr.op == E.UnaryOp.Invert:
            r = ctx.new_reg()
            ctx.emit(Op.NOT, (r, operand))
            return r
        if expr.op == E.UnaryOp.Not:
            r = ctx.new_reg()
            ctx.emit(Op.NOT, (r, operand))
            return r
        if expr.op == E.UnaryOp.UAdd:
            return operand
        raise LoweringError(f"unsupported unary op {expr.op}",
                            loc=getattr(expr, "loc", None))

    if isinstance(expr, E.ExprCompare):
        if len(expr.ops) != 1 or len(expr.comparators) != 1:
            raise LoweringError("chained comparison not supported in M1",
                                loc=getattr(expr, "loc", None))
        a = eval_expr(ctx, expr.left)
        b = eval_expr(ctx, expr.comparators[0])
        op = _CMPOP.get(expr.ops[0])
        if op is None:
            raise LoweringError(f"unsupported compare {expr.ops[0]}",
                                loc=getattr(expr, "loc", None))
        r = ctx.new_reg()
        ctx.emit(op, (r, a, b))
        return r

    if isinstance(expr, E.ExprBool):
        op = _BOOLOP[expr.op]
        acc = eval_expr(ctx, expr.values[0])
        for v in expr.values[1:]:
            b = eval_expr(ctx, v)
            r = ctx.new_reg()
            ctx.emit(op, (r, acc, b))
            acc = r
        return acc

    if isinstance(expr, E.ExprCall):
        return _eval_import_call(ctx, expr)

    raise LoweringError(
        f"unsupported expression {type(expr).__name__} in M1",
        loc=getattr(expr, "loc", None),
    )


def _eval_import_call(ctx: CoroCtx, expr) -> int:
    """Lower a value-returning import call (e.g. ``getval(7)``) to an ``IMPORT``.

    The result lands in a fresh register (``INSTR_F_HAS_RET``). Nested calls fall
    out naturally -- ``doit(getval(7))`` lowers ``getval`` here (returning a reg),
    then the enclosing blocking ``doit`` consumes that reg. Only calls that resolve
    to a declared import are supported in M1; anything else is a clear error.
    """
    name = _import_call_name(expr.func)
    decl = ctx.lowerer.imports.get(name) if name is not None else None
    if decl is None:
        raise LoweringError(
            f"call to {name!r} is not a known import (only import calls are "
            f"supported in M1 procedural code)",
            loc=getattr(expr, "loc", None),
        )
    arg_regs = [eval_expr(ctx, a) for a in expr.args]
    ret_reg = ctx.new_reg()
    flags = INSTR_F_HAS_RET | (INSTR_F_BLOCKING if decl["blocking"] else 0)
    ctx.emit(Op.IMPORT, tuple([decl["fn_id"], ret_reg] + arg_regs), flags=flags)
    ctx.lowerer.import_decls.setdefault(
        decl["fn_id"], {"fn": name, "blocking": decl["blocking"]})
    return ret_reg


def _store(ctx: CoroCtx, target, reg: int) -> None:
    if isinstance(target, E.ExprRefLocal):
        ctx.emit(Op.ST_LOCAL, (reg, ctx.local_slot(target.name)))
    elif isinstance(target, E.ExprRefField):
        ctx.emit(Op.ST_FIELD, (reg, target.index))
    else:
        raise LoweringError(
            f"unsupported assignment target {type(target).__name__} in M1",
            loc=getattr(target, "loc", None),
        )


def lower_stmt(ctx: CoroCtx, stmt) -> None:
    """Lower one Layer-0 statement into ``ctx.code``."""
    sr = ctx.lowerer.prov.src_ref(stmt)
    saved, ctx.cur_src_ref = ctx.cur_src_ref, (sr or ctx.cur_src_ref)
    try:
        if isinstance(stmt, S.StmtAssign):
            r = eval_expr(ctx, stmt.value)
            for tgt in stmt.targets:
                _store(ctx, tgt, r)
        elif isinstance(stmt, S.StmtExpr):
            eval_expr(ctx, stmt.expr)  # evaluated for side effects
        elif isinstance(stmt, S.StmtReturn):
            if stmt.value is not None:
                r = eval_expr(ctx, stmt.value)
                ctx.emit(Op.RET, (r,))
            else:
                ctx.emit(Op.RET)
        elif isinstance(stmt, S.StmtPass):
            pass
        else:
            raise LoweringError(
                f"unsupported statement {type(stmt).__name__} in M1",
                loc=getattr(stmt, "loc", None),
            )
    finally:
        ctx.cur_src_ref = saved


def lower_exec_block(ctx: CoroCtx, exec_block) -> None:
    """Lower a ScExecBlock's Layer-0 statements."""
    sr = ctx.lowerer.prov.src_ref(exec_block)
    saved, ctx.cur_src_ref = ctx.cur_src_ref, (sr or ctx.cur_src_ref)
    try:
        for st in exec_block.stmts:
            lower_stmt(ctx, st)
    finally:
        ctx.cur_src_ref = saved
