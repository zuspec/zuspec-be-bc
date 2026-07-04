"""
control.py -- non-suspending structured control flow -> BR/BRZ (deferred item #1).

``CoroutineFSMPass`` keeps ``ScIf`` / ``ScLoop`` / ``ScMatch`` opaque *within* a
block as long as they contain no suspend point, so their bodies are
straight-line/structured orchestration + procedural code that executes inside one
``idx`` block.  This module lowers those constructs to **code-absolute** ``BR`` /
``BRZ`` (the VM already executes those; see ``interp.ops_proc.exec_branch``).

A construct that *does* contain a suspend is rejected upstream by the FSM pass
(loop/branch-aware FSM lowering is Phase 5), so nothing here has to split blocks:
every branch target is a plain instruction index within the current coroutine.
"""

from zuspec.ir.core import scenario as SC

from ..model import Op
from .context import CoroCtx
from .procedural import eval_expr
from .errors import LoweringError


def _patch_target(ctx: CoroCtx, at: int, target: int) -> None:
    """Backpatch the branch emitted at instruction ``at`` to jump to ``target``.

    Both ``BR`` and ``BRZ`` carry the target as their *last* arg (``BR`` = (target,),
    ``BRZ`` = (cond, target)), so patching ``args[-1]`` works for either.
    """
    ins = ctx.code[at]
    args = list(ins.args)
    args[-1] = int(target)
    ins.args = tuple(args)


def lower_if(ctx: CoroCtx, s: SC.ScIf) -> None:
    """``if cond: then_body [else: else_body]`` -> BRZ over the taken arm."""
    cond = eval_expr(ctx, s.cond)
    brz = ctx.emit(Op.BRZ, (cond, 0))   # target backpatched below
    _lower_body(ctx, s.then_body)
    if s.else_body:
        br_end = ctx.emit(Op.BR, (0,))  # then-arm skips the else-arm
        _patch_target(ctx, brz, len(ctx.code))      # BRZ -> else-arm start
        _lower_body(ctx, s.else_body)
        _patch_target(ctx, br_end, len(ctx.code))   # BR  -> join point
    else:
        _patch_target(ctx, brz, len(ctx.code))       # BRZ -> join point


def lower_loop(ctx: CoroCtx, s: SC.ScLoop) -> None:
    """Counted (``repeat``) and conditional (``whiledo`` / ``dowhile``) loops.

    ``foreach`` (collection iteration) is deferred -- it needs collection value
    semantics not present in the M1 procedural subset -- and rejected cleanly.
    """
    if s.kind == "repeat":
        _lower_counted(ctx, s)
    elif s.kind == "whiledo":
        _lower_whiledo(ctx, s)
    elif s.kind == "dowhile":
        _lower_dowhile(ctx, s)
    else:
        raise LoweringError(
            f"ScLoop kind {s.kind!r} lowering is deferred (M1 covers "
            f"repeat / whiledo / dowhile)",
            loc=getattr(s, "loc", None),
        )


def _lower_counted(ctx: CoroCtx, s: SC.ScLoop) -> None:
    if s.count is None:
        raise LoweringError("repeat loop has no count expression",
                            loc=getattr(s, "loc", None))
    # Evaluate the bound once; keep the induction variable in a frame-local slot
    # so the body's references to ``index_var`` resolve to the live counter.
    n = eval_expr(ctx, s.count)
    slot = ctx.local_slot(s.index_var or _synthetic_index(ctx))
    ctx.emit(Op.ST_LOCAL, (ctx.emit_const(0), slot))

    top = len(ctx.code)
    i = ctx.new_reg()
    ctx.emit(Op.LD_LOCAL, (i, slot))
    lt = ctx.new_reg()
    ctx.emit(Op.CMP_LT, (lt, i, n))
    brz = ctx.emit(Op.BRZ, (lt, 0))      # exit when !(i < n)

    _lower_body(ctx, s.body)

    i2 = ctx.new_reg()
    ctx.emit(Op.LD_LOCAL, (i2, slot))
    inc = ctx.new_reg()
    ctx.emit(Op.ADD, (inc, i2, ctx.emit_const(1)))
    ctx.emit(Op.ST_LOCAL, (inc, slot))
    ctx.emit(Op.BR, (top,))
    _patch_target(ctx, brz, len(ctx.code))


def _lower_whiledo(ctx: CoroCtx, s: SC.ScLoop) -> None:
    if s.cond is None:
        raise LoweringError("whiledo loop has no condition",
                            loc=getattr(s, "loc", None))
    top = len(ctx.code)
    c = eval_expr(ctx, s.cond)
    brz = ctx.emit(Op.BRZ, (c, 0))       # exit when cond is false
    _lower_body(ctx, s.body)
    ctx.emit(Op.BR, (top,))
    _patch_target(ctx, brz, len(ctx.code))


def _lower_dowhile(ctx: CoroCtx, s: SC.ScLoop) -> None:
    if s.cond is None:
        raise LoweringError("dowhile loop has no condition",
                            loc=getattr(s, "loc", None))
    top = len(ctx.code)
    _lower_body(ctx, s.body)
    c = eval_expr(ctx, s.cond)
    # Loop back while cond is true: test (cond == 0); BRZ branches when that is 0,
    # i.e. when cond is non-zero.
    ez = ctx.new_reg()
    ctx.emit(Op.CMP_EQ, (ez, c, ctx.emit_const(0)))
    ctx.emit(Op.BRZ, (ez, top))


def lower_match(ctx: CoroCtx, s: SC.ScMatch) -> None:
    """``match (subject) { p0: b0; p1: b1; default: bd }`` -> a compare/BRZ chain.

    The subject is evaluated once; each non-default case compares it for equality
    and branches over its body on a miss. A default case (``pattern is None``) runs
    when no pattern matched. Cases are tried in declaration order.
    """
    if not s.cases:
        return
    subj = eval_expr(ctx, s.subject)
    default = None
    end_jumps = []
    for case in s.cases:
        if case.pattern is None:
            default = case          # run only if nothing else matched
            continue
        pat = eval_expr(ctx, case.pattern)
        eq = ctx.new_reg()
        ctx.emit(Op.CMP_EQ, (eq, subj, pat))
        miss = ctx.emit(Op.BRZ, (eq, 0))   # skip this body if subject != pattern
        _lower_body(ctx, case.body)
        end_jumps.append(ctx.emit(Op.BR, (0,)))   # matched: jump past the rest
        _patch_target(ctx, miss, len(ctx.code))    # miss -> next case
    if default is not None:
        _lower_body(ctx, default.body)
    for j in end_jumps:
        _patch_target(ctx, j, len(ctx.code))


def _lower_body(ctx: CoroCtx, body) -> None:
    from .orchestration import lower_orch_stmt  # late import: break the cycle

    for st in body:
        lower_orch_stmt(ctx, st, is_suspend=False)


def _synthetic_index(ctx: CoroCtx) -> str:
    """A stable, collision-free counter name when the loop has no index_var."""
    n = 0
    while f"$i{n}" in ctx.local_slots:
        n += 1
    return f"$i{n}"
