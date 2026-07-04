"""
harness.py -- microbenchmark scaffold for the four hot ops (P0-13, design D§6a).

Built in "week one" so there is a rig to regress against before there is anything
worth regressing. It measures per-op cost for the four hot operations the design
calls out -- **coroutine resume, pure-import crossing, blocking-import crossing,
solve** -- against explicit ~1 microsecond budgets.

In M1 the harness runs against the **Python oracle** only (to establish the shape
and a Python baseline); it is deliberately callable with placeholder ops so the
rig itself is exercised before the oracle exists. The cross-tier regression gate
(oracle vs rt-eng) arrives with the native engine (roadmap P3).
"""

import dataclasses as dc
import time
from typing import Callable, Dict, List, Optional

#: The four hot ops (D§6a). Order is stable for reporting.
HOT_OPS = ("coro_resume", "pure_import", "blocking_import", "solve")

#: Target per-op budgets in nanoseconds (the ~1 microsecond envelope, D§6a).
#: These are *targets*, not gates, in M1 -- the Python oracle is not expected to
#: hit them; the C engine (P3) is where the regression gate bites.
BUDGET_NS: Dict[str, float] = {
    "coro_resume": 1000.0,
    "pure_import": 500.0,
    "blocking_import": 1000.0,
    "solve": 5000.0,
}


@dc.dataclass
class BenchResult:
    name: str
    iters: int
    ns_per_op: float
    budget_ns: float

    @property
    def within_budget(self) -> bool:
        return self.ns_per_op <= self.budget_ns

    def format(self) -> str:
        flag = "ok" if self.within_budget else "OVER"
        return (
            f"{self.name:<18} {self.ns_per_op:10.1f} ns/op  "
            f"(budget {self.budget_ns:8.1f} ns) [{flag}]"
        )


def bench(name: str, fn: Callable[[], None], iters: int = 100_000,
          warmup: int = 1_000, budget_ns: Optional[float] = None) -> BenchResult:
    """Time ``fn`` over ``iters`` calls after ``warmup`` untimed calls."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter_ns() - start
    return BenchResult(
        name=name,
        iters=iters,
        ns_per_op=elapsed / iters,
        budget_ns=budget_ns if budget_ns is not None else BUDGET_NS.get(name, 0.0),
    )


def run_suite(ops: Dict[str, Callable[[], None]], iters: int = 100_000
              ) -> List[BenchResult]:
    """Run the provided ops (subset of :data:`HOT_OPS`) and return results.

    Unknown op names are allowed (custom benches); missing hot ops are simply not
    run -- callers that want the full four-op picture pass all four.
    """
    results = []
    for name in HOT_OPS:
        if name in ops:
            results.append(bench(name, ops[name], iters=iters))
    for name, fn in ops.items():
        if name not in HOT_OPS:
            results.append(bench(name, fn, iters=iters))
    return results


def _placeholder_ops() -> Dict[str, Callable[[], None]]:
    """Trivial no-op stand-ins so the rig is self-exercising before the oracle."""
    def noop():
        pass
    return {name: noop for name in HOT_OPS}


if __name__ == "__main__":
    print("ZBC perf harness (placeholder ops -- Python baseline shape only)\n")
    for r in run_suite(_placeholder_ops(), iters=200_000):
        print(r.format())
