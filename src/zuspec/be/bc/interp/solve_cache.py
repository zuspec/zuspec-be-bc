"""
solve_cache.py -- compiled solve problems, reused across solves of one run.

Compiling a problem (``dv_solve.ctx.SolveCtx``) costs far more than solving it:
a cone of a dozen nodes compiles in tens of milliseconds and solves, pinned, in
tens of microseconds. A run solves the same problem many times -- every
traversal of one site, every iteration of a loop, every IO of a long scenario
-- so each distinct problem is compiled once and each solve works on it
between a checkpoint and a restore::

    cp = ctx.checkpoint(); ctx.pin(...); ctx.solve(...); ...; ctx.restore(cp)

A restore returns the context to the state it compiled to, so a solve on a
reused context gives exactly the values a fresh compile would. That is a
requirement, not an optimisation's side effect: a run's values must not depend
on what was solved before it (``test_solve_cache.py`` holds runs with and
without the cache to byte-identical logs).

A problem is keyed by its blob, which determines it completely. Entries are
evicted least-recently-used beyond ``capacity``; each holds the solver's working
memory (about 1 MiB), so the default keeps a run's footprint bounded no matter
how many distinct problems it meets. ``capacity=0`` compiles every solve
afresh -- the reference the equivalence tests compare against.

**Budget.** dv-solve's search restarts after ``luby(i) * 100`` conflicts and
gives up (``SOLVE_TIMEOUT``) after ``max_restarts`` restarts: a deterministic
bound on one solve. The default is dv-solve's own, so caching changes no
outcome; what changes is that giving up is a located :class:`SolveBudgetError`
(it used to be reported as unsat). The restart unit is left at dv-solve's
default on purpose: it is not a total budget, and a large one turns restarts
off, which makes heavy-tailed problems slower rather than bounded.

**Learning first.** Each solve first runs a short clause-learning search
(``use_lcg``, ``lcg_restarts`` restarts) and keeps its solution if it finds
one; otherwise the plain search settles it under the full budget (see
:meth:`SolveCache.solve`). On the benchmark IO problems this is 2-6x faster
overall than the plain search alone (nvme-bench plan §5).
"""

from __future__ import annotations

import collections
import contextlib
import ctypes
from typing import Iterator, Optional

from .ops_proc import VMError

#: Compiled problems one run keeps.
DEFAULT_CAPACITY = 64
#: Restarts one solve may take before it is reported as out of budget
#: (dv-solve's default).
DEFAULT_MAX_RESTARTS = 10_000
#: Learn clauses from conflicts (dv-solve's lazy clause generation) first.
DEFAULT_USE_LCG = True
#: Restarts the learning search gets before the plain search takes over.
DEFAULT_LCG_RESTARTS = 5


class SolveBudgetError(VMError):
    """A solve exhausted its budget: the problem may or may not have a
    solution, and the run cannot tell which."""


class SolveUnsatError(VMError):
    """No values satisfy an action's own constraints."""


class SolveCache:
    """Compiled dv-solve contexts, keyed by problem blob, least recently used
    evicted first."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY,
                 max_restarts: int = DEFAULT_MAX_RESTARTS,
                 use_lcg: bool = DEFAULT_USE_LCG,
                 lcg_restarts: int = DEFAULT_LCG_RESTARTS,
                 fair_pick: bool = False):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        if max_restarts < 1:
            raise ValueError("max_restarts must be >= 1")
        self.capacity = capacity
        self.max_restarts = max_restarts
        self.use_lcg = use_lcg
        self.lcg_restarts = lcg_restarts
        #: dv-solve's fair decision tie-break (D-B7). It changes the values
        #: a seed gives, so it is off unless asked for.
        self.fair_pick = fair_pick
        #: solves LCG could not finish, settled by the confirming plain solve
        self.lcg_retries = 0
        self._ctxs: "collections.OrderedDict[bytes, object]" = collections.OrderedDict()
        #: problems compiled, and solves that reused a compiled problem
        self.compiles = 0
        self.hits = 0
        #: solves made (one session may hold several, or none)
        self.solves = 0
        #: solves not made: a cone's last solution answered them
        #: (``activation.Activation._reuse``)
        self.reused = 0

    def __len__(self) -> int:
        return len(self._ctxs)

    @contextlib.contextmanager
    def session(self, blob: bytes) -> Iterator[object]:
        """The compiled context of *blob*, for one solve: pins and the solve
        made in the ``with`` body are undone when it exits.

        Raises ``dv_solve.ctx.CompileUnsatError`` if the problem is unsat
        before any pin; nothing is cached then.
        """
        ctx = self._ctxs.get(blob)
        if ctx is not None:
            self._ctxs.move_to_end(blob)
            self.hits += 1
        else:
            ctx = self._compile(blob)
            if self.capacity == 0:
                try:
                    yield ctx
                finally:
                    ctx.destroy()
                return
            self._ctxs[blob] = ctx
            while len(self._ctxs) > self.capacity:
                _, old = self._ctxs.popitem(last=False)
                old.destroy()
        cp = ctx.checkpoint()
        try:
            yield ctx
        finally:
            ctx.restore(cp)

    def solve(self, ctx, seed: int) -> int:
        """``ctx.solve`` under this cache's budget.

        With ``use_lcg``, a solve is first a short search that learns clauses
        from its conflicts and backjumps (``lcg_restarts`` restarts): on this
        kind of problem it is usually far faster than chronological search,
        and on the rest it gives up quickly. A solution it finds is kept
        (every constraint holds on it). Anything else -- it gave up, or it
        claims there is no solution -- is settled by the plain search from the
        same state, under the full ``max_restarts`` budget. Both steps are
        deterministic in the seed, so a run is still reproducible and a
        reused context still answers as a fresh one.
        """
        self.solves += 1
        fair = self.fair_pick
        if not self.use_lcg:
            return ctx.solve(seed=seed, max_restarts=self.max_restarts, fair_pick=fair)
        from dv_solve.ctx import SOLVE_OK
        cp = ctx.checkpoint()
        rc = ctx.solve(seed=seed, max_restarts=self.lcg_restarts, use_lcg=True,
                       fair_pick=fair)
        if rc == SOLVE_OK:
            return rc                  # the session's restore pops `cp`
        ctx.restore(cp)
        self.lcg_retries += 1
        return ctx.solve(seed=seed, max_restarts=self.max_restarts, fair_pick=fair)

    def clear(self) -> None:
        """Release every compiled context."""
        while self._ctxs:
            _, ctx = self._ctxs.popitem()
            ctx.destroy()

    def _compile(self, blob: bytes):
        from dv_solve.ctx import SolveCtx
        raw = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        ctx = SolveCtx(raw)
        self.compiles += 1
        return ctx

    def budget_message(self) -> str:
        return ("the solver gave up after %d restarts; the constraints may have "
                "no solution, or need more search than the budget allows "
                "(raise it with run_model(max_restarts=...))" % self.max_restarts)


__all__ = ["SolveCache", "SolveBudgetError", "DEFAULT_CAPACITY",
           "DEFAULT_USE_LCG", "DEFAULT_LCG_RESTARTS",
           "DEFAULT_MAX_RESTARTS"]
