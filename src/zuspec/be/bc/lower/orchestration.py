"""
orchestration.py -- Scenario orchestration stmts -> orchestration opcodes (P1-2).

Each orchestration statement maps 1:1 to an orchestration opcode (design D§4.1).
The FSM pass has already split the coroutine at suspend points, so the driver
knows which statement ends a block; :func:`lower_orch_stmt` takes an
``is_suspend`` hint so INVOKE/IMPORT get the blocking flag exactly when they are
the block's suspend.

M1 scope: WAIT/JOIN/INVOKE/SPAWN/IMPORT/SOLVE, plus ScPar (ALL join) which desugars
to branch sub-coroutines + SPAWN/JOIN (see :mod:`.parallel`) and non-suspending
ScIf/ScLoop (see :mod:`.control`). ScSelect and non-ALL joins are still deferred and
rejected cleanly.
"""

from typing import Optional

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import expr as E

from ..model import (
    Op, SolveProblem, INSTR_F_BLOCKING, INSTR_F_HAS_RET,
)
from ..abi.value import solver_var_map
from .context import CoroCtx
from .procedural import eval_expr, lower_exec_block
from . import control
from . import parallel
from . import select
from .errors import LoweringError

_VOID = 0xFFFFFFFF


def _const_int(expr) -> Optional[int]:
    if isinstance(expr, E.ExprConstant) and isinstance(expr.value, (int, bool)):
        return int(expr.value)
    return None


def lower_orch_stmt(ctx: CoroCtx, s, is_suspend: bool = False) -> Optional[Op]:
    """Lower one orchestration/exec statement.

    Returns the :class:`Op` that ends a block if this statement suspends, else
    ``None``.
    """
    sr = ctx.lowerer.prov.src_ref(s)
    saved, ctx.cur_src_ref = ctx.cur_src_ref, (sr or ctx.cur_src_ref)
    try:
        return _dispatch(ctx, s, is_suspend, sr)
    finally:
        ctx.cur_src_ref = saved


def _dispatch(ctx: CoroCtx, s, is_suspend, sr) -> Optional[Op]:
    if isinstance(s, SC.ScExecBlock):
        lower_exec_block(ctx, s)
        return None

    if isinstance(s, SC.ScWait):
        c = _const_int(s.time)
        if c is not None:
            ctx.emit(Op.WAIT, (), imm=int(c) & ((1 << 64) - 1), src_ref=sr)
        else:
            r = eval_expr(ctx, s.time)
            ctx.emit(Op.WAIT, (r,), src_ref=sr)
        return Op.WAIT

    if isinstance(s, SC.ScJoin):
        ctx.emit(Op.JOIN, (), src_ref=sr)
        return Op.JOIN

    if isinstance(s, SC.ScSpawn):
        target = ctx.lowerer._coro_index.get(s.target, 0)
        ctx.emit(Op.SPAWN, (target,), src_ref=sr)
        return None

    if isinstance(s, SC.ScInvoke):
        target = ctx.lowerer._coro_index.get(s.target, 0)
        # Blocking is intrinsic to the callee, not positional: an invoke of a
        # blocking sub-coroutine suspends wherever it appears -- including inside a
        # loop/branch body (where the positional ``is_suspend`` hint is False).
        blocking = s.target in ctx.lowerer.blocking_targets
        flags = INSTR_F_BLOCKING if blocking else 0
        ctx.emit(Op.INVOKE, (target,), flags=flags, src_ref=sr)
        return Op.INVOKE if blocking else None

    if isinstance(s, SC.ScImport):
        return _lower_import(ctx, s, sr)

    if isinstance(s, SC.ScSolveProblem):
        return _lower_solve(ctx, s, sr)

    if isinstance(s, (SC.ScSeq, SC.ScAtomic)):
        # Region wrappers with no suspend of their own: lower the body inline.
        for st in s.body:
            lower_orch_stmt(ctx, st, is_suspend=False)
        return None

    if isinstance(s, SC.ScIf):
        control.lower_if(ctx, s)
        return None

    if isinstance(s, SC.ScLoop):
        control.lower_loop(ctx, s)
        return None

    if isinstance(s, SC.ScMatch):
        control.lower_match(ctx, s)
        return None

    if isinstance(s, SC.ScPar):
        return parallel.lower_par(ctx, s, sr)

    if isinstance(s, SC.ScSelect):
        return select.lower_select(ctx, s, sr)

    raise LoweringError(
        f"unsupported orchestration statement {type(s).__name__}",
        loc=getattr(s, "loc", None),
    )


def _lower_import(ctx: CoroCtx, s, sr) -> Optional[Op]:
    if len(s.args) > 2:
        raise LoweringError(
            f"import {s.fn!r}: M1 supports <=2 args, got {len(s.args)}",
            loc=getattr(s, "loc", None),
        )
    arg_regs = [eval_expr(ctx, a) for a in s.args]

    flags = INSTR_F_BLOCKING if s.blocking else 0
    ret_reg = 0
    if s.ret_var is not None:
        ret_reg = ctx.new_reg()
        flags |= INSTR_F_HAS_RET

    args = [s.fn_id, ret_reg if (flags & INSTR_F_HAS_RET) else _VOID] + arg_regs
    ctx.emit(Op.IMPORT, tuple(args), flags=flags, src_ref=sr)

    # Record the import declaration (fn_id -> blocking) for the provider table.
    ctx.lowerer.import_decls.setdefault(s.fn_id, {"fn": s.fn, "blocking": s.blocking})

    if s.ret_var is not None:
        ctx.emit(Op.ST_LOCAL, (ret_reg, ctx.local_slot(s.ret_var)), src_ref=sr)

    return Op.IMPORT if s.blocking else None


def _lower_solve(ctx: CoroCtx, s, sr) -> Optional[Op]:
    # var_id assignment is the value ABI's sorted-by-name rule; keep the var names
    # so the interpreter maps get_value(var_id) back to the right field.
    names = [v.name for v in s.vars]
    var_map = solver_var_map(names)
    problem = SolveProblem(
        var_names=sorted(names, key=lambda n: var_map[n]),
        writeback=dict(s.writeback),
    )
    if s.seed is not None:
        c = _const_int(s.seed)
        if c is not None:
            problem.seed_kind = "fixed"
            problem.seed_value = c
    pid = ctx.lowerer.add_problem(problem)
    ctx.emit(Op.SOLVE, (pid,), src_ref=sr)
    return None
