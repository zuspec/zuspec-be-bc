"""
select.py -- ScSelect -> weighted-choice SELECT (deferred item #2).

A ``select { [w1]: a; [w2]: b; ... }`` block chooses **one** branch at random,
weighted by declaration-order cumulative weights, then runs it.  Unlike ScPar (which
forks every branch), the choice is a runtime RNG draw, so this needs a real SELECT
op backed by a branch table (:class:`..model.SelectTable`, an in-memory M1 side
channel like the SOLVE problem table).

M1 scope: constant weights (default 1). Guards and ``allow_none`` are supported --
each guard expression lowers to a register the VM reads to filter eligible branches
before the weighted draw. Non-constant weights are still rejected.
"""

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import expr as E

from ..model import Op, SelectTable
from .context import CoroCtx
from .procedural import eval_expr
from .errors import LoweringError


def _const_weight(w) -> int:
    if w is None:
        return 1
    if isinstance(w, E.ExprConstant) and isinstance(w.value, (int, bool)):
        v = int(w.value)
        if v <= 0:
            raise LoweringError(f"select branch weight must be positive, got {v}")
        return v
    raise LoweringError(
        "select branch weight must be a positive integer constant in M1",
    )


def lower_select(ctx: CoroCtx, s: SC.ScSelect, sr) -> Op:
    """Lower a ScSelect to a SELECT op + branch table. Returns ``Op.SELECT``."""
    if not s.branches:
        raise LoweringError("select has no branches", loc=getattr(s, "loc", None))

    from .driver import lower_coroutine  # late import: break the cycle

    # Guard expressions are evaluated *before* the SELECT (procedural ops in this
    # block write their result registers); the VM reads those registers at runtime
    # to decide which branches are eligible. -1 marks an unguarded (always-eligible)
    # branch.
    guards = []
    for br in s.branches:
        guards.append(eval_expr(ctx, br.guard) if br.guard is not None else -1)

    seq = ctx.lowerer.next_synth_id()
    branch_idxs = []
    weights = []
    for j, br in enumerate(s.branches):
        sub = SC.ScCoroutine(name=f"{ctx.coro_name}$sel{seq}_{j}", body=list(br.body))
        desc = lower_coroutine(sub, ctx.lowerer,
                               blocking_targets=ctx.lowerer.blocking_targets)
        branch_idxs.append(ctx.lowerer.add_branch_coro(desc))
        weights.append(_const_weight(br.weight))

    sid = ctx.lowerer.add_select(SelectTable(
        branches=branch_idxs, weights=weights, guards=guards,
        allow_none=bool(s.allow_none)))
    ctx.emit(Op.SELECT, (sid,), src_ref=sr)
    return Op.SELECT
