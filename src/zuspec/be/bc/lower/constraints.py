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
