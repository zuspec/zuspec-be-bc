"""
constraints.py -- lower an ``ScSolveProblem``'s constraints to a dv-solve blob.

The translation is zuspec-solver's (:mod:`zuspec.solver.translate`), shared with
every other backend that hands a scenario solve problem to dv-solve; this module
only turns its refusal into a :class:`LoweringError`, so a constraint bc cannot
lower reads as one wherever the refusal came from.
"""
from __future__ import annotations

from typing import Dict, Tuple

from zuspec.ir.core import scenario as SC

from .errors import LoweringError


def build_solve_blob(problem: SC.ScSolveProblem) -> Tuple[bytes, Dict[int, int]]:
    """Translate *problem*'s constraints to a (problem_bytes, writeback_slots) pair.

    ``writeback_slots`` maps object field slot -> solver var_id. Raises
    :class:`LoweringError` on an unsupported constraint form.
    """
    from zuspec.solver.translate import TranslateError, build_solve_blob as _build
    try:
        return _build(problem)
    except TranslateError as e:
        raise LoweringError(str(e), loc=e.loc) from e


def solve_blob_vars(problem: SC.ScSolveProblem) -> int:
    """How many variables *problem*'s blob declares: more than the base's
    when a constraint needs selectors (a disjunction with a conjunction in
    it). An added blob numbers them from the base's count, so two such
    blobs cannot both be added to one base (dv-solve ``add_constraint`` -4)."""
    from zuspec.solver.translate import TranslateError, builder_sink, translate
    sp = builder_sink()
    n = [0]

    class _Count:
        def __getattr__(self, name):
            return getattr(sp, name)

        def add_var(self, *a, **k):
            n[0] += 1
            return sp.add_var(*a, **k)
    try:
        translate(problem, _Count())
    except TranslateError as e:
        raise LoweringError(str(e), loc=e.loc) from e
    finally:
        sp.destroy()
    return n[0]
