"""
parallel.py -- ScPar -> branch sub-coroutines + SPAWN/JOIN (deferred item #2).

A ``parallel { a; b; ... }`` block forks its branches concurrently and joins them.
The FSM pass already classifies ``ScPar`` as a suspend point, so the driver hands
it to us as a block's trailing suspend op.

M1 lowering, ``JoinKind.ALL`` (the default): each branch becomes a synthesized
**sub-coroutine** (lowered independently, so a branch may itself suspend on a
blocking invoke/wait), and the ScPar desugars to one ``SPAWN`` per branch followed
by a ``JOIN``.  This reuses the fully-tested scheduler machinery -- ``SPAWN`` forks
a child with a seed forked in declaration order (determinism spec), ``JOIN`` blocks
the parent until every live child completes.  The compact single-``PAR``-op + OPLIST
encoding is deferred; they change the *encoding*, not
the executed semantics of the ALL case.
"""

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import expr as E
from zuspec.ir.core.activity import JoinKind

from ..model import Op, INSTR_F_SUBSET
from .context import CoroCtx, branch_inherit
from .errors import LoweringError

_SUPPORTED_JOINS = (JoinKind.ALL, JoinKind.NONE, JoinKind.FIRST,
                    JoinKind.SELECT, JoinKind.BRANCH)
#: join_branch names branches by a bit mask in one 32-bit argument
_MAX_SUBSET_BRANCHES = 32


def _join_count(spec, kind) -> int:
    c = spec.count
    if isinstance(c, E.ExprConstant) and isinstance(c.value, (int, bool)):
        n = int(c.value)
        if n < 0:
            raise LoweringError(f"{kind.name} join count must not be negative, got {n}")
        return n
    raise LoweringError(f"{kind.name} join count must be an integer constant")


def lower_par(ctx: CoroCtx, s: SC.ScPar, sr):
    """Lower a ScPar to branch sub-coroutines + SPAWN/JOIN.

    ALL   -> SPAWN each + JOIN (wait for all).       Returns ``Op.JOIN``.
    FIRST -> SPAWN each + JOIN(imm=n) (wait for n).  Returns ``Op.JOIN``.
    NONE  -> SPAWN each, no JOIN (detached).          Returns ``None`` (no suspend).
    SELECT(n) / BRANCH(labels) -> SPAWN each + JOIN with ``INSTR_F_SUBSET``:
             wait for n random branches / the labelled ones; the rest run on
             (LRM 13.3.6). Returns ``Op.JOIN``.
    A FIRST or SELECT count of 0 is NONE (13.3.6 c.1, d.1).
    """
    kind = s.join_spec.kind if s.join_spec is not None else JoinKind.ALL
    count = 0
    if kind in (JoinKind.FIRST, JoinKind.SELECT):
        count = _join_count(s.join_spec, kind)
        if count == 0:
            kind = JoinKind.NONE
        elif kind == JoinKind.SELECT and count >= len(s.branches):
            kind = JoinKind.ALL
    if kind in (JoinKind.SELECT, JoinKind.BRANCH) \
            and len(s.branches) > _MAX_SUBSET_BRANCHES:
        raise LoweringError(
            f"{kind.name} join over {len(s.branches)} branches: at most "
            f"{_MAX_SUBSET_BRANCHES} are supported", loc=getattr(s, "loc", None))
    if kind not in _SUPPORTED_JOINS:
        raise LoweringError(
            f"ScPar join policy {kind.name} lowering is deferred (M1 covers "
            f"ALL / NONE / FIRST(n))",
            loc=getattr(s, "loc", None),
        )
    if not s.branches:
        return None                    # nothing to fork or join

    from .driver import lower_coroutine  # late import: break the cycle

    seq = ctx.lowerer.next_synth_id()
    for j, branch in enumerate(s.branches):
        sub = SC.ScCoroutine(name=f"{ctx.coro_name}$par{seq}_{j}", body=[branch],
                             **branch_inherit(ctx))
        desc = lower_coroutine(sub, ctx.lowerer,
                               blocking_targets=ctx.lowerer.blocking_targets)
        idx = ctx.lowerer.add_branch_coro(desc)
        ctx.emit(Op.SPAWN, (idx,), src_ref=sr)

    if kind == JoinKind.NONE:
        return None                    # fire-and-forget: branches run detached

    k = len(s.branches)
    if kind == JoinKind.SELECT:
        ctx.emit(Op.JOIN, (k, 0), imm=count, flags=INSTR_F_SUBSET, src_ref=sr)
    elif kind == JoinKind.BRANCH:
        mask = 0
        for i in s.join_branches:
            mask |= 1 << i
        ctx.emit(Op.JOIN, (k, mask), imm=0, flags=INSTR_F_SUBSET, src_ref=sr)
    else:
        ctx.emit(Op.JOIN, (), imm=count if kind == JoinKind.FIRST else 0, src_ref=sr)
    return Op.JOIN
