"""Perf-harness self-test (backs P0-13). The oracle-driven run is T1-G (P1)."""

from tests.perf.harness import (
    HOT_OPS, BUDGET_NS, bench, run_suite, _placeholder_ops,
)


def test_hot_ops_have_budgets():
    assert set(HOT_OPS) == set(BUDGET_NS)
    assert all(v > 0 for v in BUDGET_NS.values())


def test_run_suite_reports_all_four_ops():
    results = run_suite(_placeholder_ops(), iters=2000)
    assert [r.name for r in results] == list(HOT_OPS)
    for r in results:
        assert r.ns_per_op >= 0.0
        assert r.budget_ns == BUDGET_NS[r.name]


def test_bench_custom_op():
    calls = {"n": 0}

    def op():
        calls["n"] += 1

    r = bench("coro_resume", op, iters=500, warmup=10)
    assert calls["n"] == 510
    assert r.iters == 500
