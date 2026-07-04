"""T1-G -- perf smoke: run the P0-13 harness against the Python oracle.

Establishes a Python baseline for the four hot ops (D§6a). No gate in M1 -- the
oracle is a semantics reference, not a speed target; the regression gate arrives
with the native engine (roadmap P3). This test only asserts the rig runs the
oracle and produces results, and prints the baseline for the record.
"""

from tests.perf.harness import run_suite, HOT_OPS

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import expr as E

from zuspec.be.bc.lower import lower_scenario
from zuspec.be.bc.interp import (
    run_model, Obj, FixedSolveBackend, RecordingImportProvider,
)


def _coro(name, body, **kw):
    return SC.ScCoroutine(name=name, body=body, **kw)


def _oracle_ops():
    # Pre-lower each micro-scenario once; time only the execution.
    resume_m = lower_scenario([_coro("r", [])])  # empty body -> single terminal block
    pure_m = lower_scenario([_coro("r", [
        SC.ScImport(fn="p", fn_id=1, blocking=False, args=[E.ExprConstant(value=1)])])])
    blk_m = lower_scenario([_coro("r", [
        SC.ScImport(fn="b", fn_id=2, blocking=True, args=[E.ExprConstant(value=1)])])])
    solve_m = lower_scenario([_coro("r", [
        SC.ScSolveProblem(vars=[SC.ScSolveVar(name="a", var_id=0)],
                          writeback={"a": 0}, seed=E.ExprConstant(value=1))])])

    imports = RecordingImportProvider()
    solver = FixedSolveBackend()

    return {
        "coro_resume": lambda: run_model(resume_m),
        "pure_import": lambda: run_model(pure_m, import_provider=imports),
        "blocking_import": lambda: run_model(blk_m, import_provider=imports),
        "solve": lambda: run_model(solve_m, obj=Obj(["a"]), solve_backend=solver),
    }


def test_oracle_perf_smoke(capsys):
    results = run_suite(_oracle_ops(), iters=2_000)
    assert {r.name for r in results} == set(HOT_OPS)
    for r in results:
        assert r.ns_per_op > 0
    with capsys.disabled():
        print("\nZBC Python oracle baseline (T1-G, no gate):")
        for r in results:
            print("  " + r.format())
