"""Which constraints of an unsat solve conflict (F3.8).

dv-solve answers *unsat* without saying why. When a problem's source (the
ir-core ``ScSolveProblem``) is at hand, :func:`unsat_core` finds a minimal
set of its constraints that has no solution on its own: it drops each
constraint in turn and keeps the drop when what is left is still unsat
(deletion filter). Each constraint keeps the location of the PSS statement
it came from, so the error can name the lines to look at.

Only constraints with a location are candidates; the rest (a variable's
own domain, say) always stay in. A ``soft`` is never one: the solver drops
a soft that conflicts. A trial that runs out of budget proves nothing, so
the constraint it left out stays in the core.
"""
from __future__ import annotations

import ctypes
import dataclasses as dc
from typing import Dict, List, Optional, Tuple

#: Restarts each trial solve gets; a diagnosis, so kept small.
TRIAL_RESTARTS = 50

#: More candidates than this are not narrowed (each costs a compile).
MAX_CANDIDATES = 64


def _is_unsat(source, constraints, pins) -> bool:
    """Whether *source* with only *constraints* is proven to have no
    solution under *pins* (``{var_id: value}``)."""
    from dv_solve.ctx import SOLVE_UNSAT, SolveCtx, CompileUnsatError
    from zuspec.solver.translate import TranslateError, build_solve_blob

    try:
        blob, _ = build_solve_blob(dc.replace(source, constraints=list(constraints)))
    except TranslateError:
        return False
    raw = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
    try:
        ctx = SolveCtx(raw)
    except CompileUnsatError:
        return True
    try:
        for vid, val in sorted((pins or {}).items()):
            if not ctx.pin(vid, val):
                return True
        return ctx.solve(seed=1, max_restarts=TRIAL_RESTARTS) == SOLVE_UNSAT
    finally:
        ctx.destroy()


def _is_candidate(c) -> bool:
    from zuspec.ir.core import constraint as C
    return getattr(c, "loc", None) is not None and not isinstance(c, C.ConstraintSoft)


def unsat_core(source, pins: Optional[Dict[int, int]] = None) -> List[object]:
    """A minimal list of *source*'s located constraints that conflict, in
    source order; empty when it cannot tell (the whole problem is not
    proven unsat, or there are too many to narrow)."""
    cons = list(source.constraints)
    cand = [i for i, c in enumerate(cons) if _is_candidate(c)]
    if not cand or len(cand) > MAX_CANDIDATES or not _is_unsat(source, cons, pins):
        return []
    keep = set(range(len(cons)))
    for i in cand:
        trial = keep - {i}
        if _is_unsat(source, [cons[j] for j in sorted(trial)], pins):
            keep = trial
    core = [cons[i] for i in sorted(keep) if i in cand]
    # Several IR constraints from one statement count once.
    seen, out = set(), []
    for c in core:
        k = _where(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _where(c) -> Tuple[str, int]:
    return (c.loc.file or "", c.loc.line or 0)


def describe(core) -> str:
    """``"conflicting constraints: a.pss:6, a.pss:8"``, or "" for none."""
    if not core:
        return ""
    return "conflicting constraints: " + ", ".join(
        "%s:%d" % _where(c) for c in core)


def dropped_softs(source, values: Dict[int, int]) -> List[object]:
    """The soft constraints of *source* that the solution *values*
    (``{var_id: value}``, every variable) breaks (F3.10). dv-solve drops a
    soft only when it conflicts with what is kept, and a dropped soft
    cannot hold in a solution of what is kept, so these are the dropped
    ones. Each is checked as a hard constraint with every variable pinned."""
    from zuspec.ir.core import constraint as C

    out = []
    for c in source.constraints:
        if isinstance(c, C.ConstraintSoft):
            hard = C.ConstraintExpr(expr=c.expr)
            if _is_unsat(source, [hard], values):
                out.append(c)
    return out


def soft_detail(c, text: str) -> Dict[str, object]:
    """A dropped soft as the ``solve`` event lists it."""
    loc = getattr(c, "loc", None)
    return {"text": text, "file": (loc.file or "") if loc else "",
            "line": (loc.line or 0) if loc else 0}
