# The ZBC Python Oracle

The oracle (`zuspec.be.bc.interp`) is the **differential reference implementation**
of ZBC execution. It runs the same lowered bytecode the native engine will run and
reproduces the legacy runtime's results, so simulation semantics are validated in
pure Python before any native build exists (roadmap P1 / design `[D§3a #1, §5]`).

## Role

- **Semantics oracle.** For any lowered scenario, the oracle's field values, import
  call sequence, and trace *are* the specification the native engine (P3) must match.
- **Round-trip guard.** The canonical run path serializes and deserializes the model
  before executing, so a writer/reader bug shows up as an execution divergence rather
  than hiding off the tested path.
- **Determinism reference.** Given a seed, the run is a pure function of the model —
  same seed ⇒ byte-identical trace (the property the differential and golden suites
  rely on).

## Architecture

| Module        | Responsibility |
|---------------|----------------|
| `scheduler.py`| Frames + ready queue + timed-event min-heap (a Python analogue of `zsp_timebase`). Models *ordering*, not speed. |
| `vm.py`       | The `switch`-style dispatch loop: resume one frame until its next suspend op or completion. |
| `ops_proc.py` | Procedural (register-SSA) handlers. Register values are unsigned 64-bit bit patterns; the value codec is the truncation/overflow authority. |
| `ops_orch.py` | Orchestration handlers (drive the scheduler) + trace emission — one event per orchestration op, carrying `src_ref`. |
| `extern.py`   | The two external seams: `SolveBackend` (SOLVE) and `ImportProvider` (IMPORT), plus the field-addressable `Obj`. |
| `oracle.py`   | The canonical round-trip run configuration and the `RunResult`. |

## Execution model

A coroutine is a flat CODE stream that the FSM split at suspend points. The VM runs a
frame from its `pc`, executing procedural ops in-line, until:

- an **orchestration op** decides whether the frame continues (SPAWN, non-blocking
  INVOKE/IMPORT, SOLVE, BIND) or suspends (WAIT, JOIN, blocking INVOKE, YIELD,
  SELECT); or
- **RET** / end-of-code completes the frame and wakes anyone JOINing / blocking-
  INVOKING on it.

Structured control flow lowers to plain branches and sub-coroutines: non-suspending
`if`/`loop`/`match` become code-absolute `BR`/`BRZ` within a block (`lower.control`);
a `parallel` desugars to one `SPAWN` per branch + a `JOIN` over synthesized branch
sub-coroutines (`lower.parallel`) — `JOIN` with `imm=n` implements a `FIRST(n)` wait
(detaching the surplus), `imm=0` a full `ALL` wait, and `NONE` skips the join
entirely; a `select` lowers to a `SELECT` op backed by a branch table
(`lower.select`) that evaluates per-branch guards, filters to the eligible set, draws
one branch in declaration order against cumulative weights
(`determinism.select_choice`), and runs it as a blocking child (or runs nothing under
`allow_none` when nothing is eligible).

M1 externs are synchronous and pure-Python: a blocking IMPORT completes immediately
(the "suspend" is only the FSM block boundary); a blocking INVOKE suspends the caller
until the callee returns, delivering the return value into the caller's result
register. SOLVE runs the backend and writes results back per the value ABI's
sorted-by-name / `var_id` mapping.

## Determinism

Seeds derive from the determinism spec (`docs/spec/determinism.md`): the root frame
gets the run seed; child frames (SPAWN / INVOKE) fork via `fork_seed(parent.state,
child_index)`. An "inherit" SOLVE draws its seed from the frame's stream; a "fixed"
SOLVE uses the literal seed lowered from the source. Ready order is FIFO by
enqueue/spawn order; timed-heap ties break by insertion sequence. Nothing reads
wall-clock time.

## Usage

```python
from zuspec.be.bc.interp import run_scenario, Obj, FixedSolveBackend, RecordingImportProvider

res = run_scenario(
    coros,                          # list[ScCoroutine]
    obj=Obj(field_names=["a", "b"]),
    seed=42,
    solve_backend=FixedSolveBackend(),
    import_provider=RecordingImportProvider(),
    round_trip=True,                # serialize→deserialize→execute (default)
)
res.fields          # {name: value} written back by SOLVE / ST_FIELD
res.retval          # coroutine return value
res.events          # trace events (orchestration ops)
res.now             # final scheduler clock
```

The in-memory SOLVE problem table is an M1 side channel (no SEC_SOLVE bytes yet), so
`round_trip` reattaches it to the deserialized model rather than round-tripping it
through bytes — exactly how the native path treats it until P4.

## As the successor execution engine (W8 / P1-14)

`run_module(scenario_module, entry_action=..., ...)` is the public "run a lowered
PSS scenario through the oracle" surface — the designated successor to the legacy
`zuspec-be-py` `ScenarioRunner`. A frontend (or `PSSToScenarioPass`) produces the
`ScenarioModule`; `run_module` lowers it (`lower_module`) and executes it through
the round-trip serialize path with the value-ABI solve/import seams.

Per the accepted plan (§7 #2 / roadmap open-Q2), the legacy runtime is **not**
deleted at G1: it is retained, importable, as a *second differential reference*
through P3 (gated by `zuspec.be.py.rt.legacy_status`), then deleted at P3 exit. The
ZBC oracle is the strategic engine; full production cutover of the export API lands
once activity lowering (parallel/select/loop) is complete. See
`zuspec-be-py/docs/retirement.md`.

## Boundaries (M1)

Structured control flow lowers: `if`/`loop`/`match`, `parallel` (`ALL` / `NONE` /
`FIRST(n)` joins), and `select` (constant weights, runtime guards, `allow_none`) —
**including a suspend point *inside* a loop or branch** (e.g. `repeat(n) { do
blocking }`). The oracle runs the resulting flat code stream directly: the FSM pass
keeps the loop/branch opaque (`allow_nested_suspend`) and the back-edge + interior
suspend interleave because the VM saves the pc across a suspend. The *stackless FSM
split* of such a construct is still deferred to Phase 5 for the native backend — a
codegen concern, not an oracle-semantics gap.

Still deferred: `SELECT`/`BRANCH` join policies, `foreach` collection iteration, and
full DPI import marshaling (the P4 calling ABI). PAR desugars to `SPAWN`/`JOIN`, so
the VM never sees a `PAR` op; the compact single-`PAR`-op + OPLIST *encoding* is a
later representation change, not a semantics gap. The differential suite against the
legacy `zuspec-be-py` runtime (T1-A) reuses `run_scenario` / `run_model` through a
fixture adapter.
