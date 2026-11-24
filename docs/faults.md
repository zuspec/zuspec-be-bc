# Fault injection: the kill matrix

The corpus checker's cheat mutations prove the checker catches bad traces. The
fault matrix proves the reverse: the corpus catches a bad tool. Each fault breaks
one scheduling rule of the pyvm on purpose; at least one corpus test must then
fail (its checker verdict is not PASS, or the run raises).

## Where it lives

- `src/zuspec/be/bc/interp/faults.py`: `DOC` (fault name -> what it breaks),
  `ALL` (the names in `DOC`), `PENDING` (faults no corpus test kills yet; they
  can be injected but are not in the matrix), `on(name)`, and the context
  manager `inject(name)`. The environment variable `ZSP_BC_FAULT` (a comma
  list) turns faults on for a whole process. Test-only; the C engine is not
  faulted.
- Each fault is one guarded site in the interpreter, `if faults.on("x"):`, so a
  normal run pays one set lookup there.
- pssc `tests/compliance/expected/fault_kill.toml`: one table per fault, with
  `tests`, the corpus test ids that kill it.
- pssc `tests/compliance/test_fault_kill.py` checks that every fault in `ALL` is
  in the matrix and the reverse; that no killer is an expected bc failure; and,
  per fault, runs the listed tests over their corpus seeds with the fault on
  until one (test, seed) does not pass. It then reruns that (test, seed) without
  the fault and requires a pass, so a broken harness cannot fake a kill.

## The faults

| fault | breaks | killed by |
|---|---|---|
| `ignore-lock` | a lock excludes no one (activation `_busy`) | `resource.lock.array.001`, `resource.select.lock.enum.001` |
| `skip-buffer-wait` | a buffer is offered when its producer is solved, not when it completes | `flow.buffer.implicit.001` |
| `drop-stream-partner` | an inferred stream partner is never started (ops_orch `_start_partner`) | `flow.stream.infer.001` |
| `stale-prev` | a state write lands one write late (`publish_states`) | `flow.state.prev.001`, `flow.state.basic.001` |
| `select-no-prune` | a select branch that cannot run is chosen (`select_blocked`) | `resource.select.lock.enum.001` |
| `atomic-infer` | an action is inferred inside `atomic` (`_in_atomic`) | `flow.atomic.no_infer_inside.001` |
| `race-state` | a state output is current when its writer is solved, not completed, and readers do not wait for it | `flow.state.par_rw.001` |
| `skip-state-wait` | a state reader or writer starts while a parallel writer of its pool runs, or a writer while a reader runs (`state_wait`, LRM 13.3.1) | `flow.state.par_rw.001`, `flow.state.par_rw.002` |
| `no-lookahead` | constraints on nodes not yet committed are ignored (`_in_force`), as with the tree's `lookahead` off | `act.lookahead.001`, `act.lookahead.sub.001`, `act.constraint.001`, `act.symbol.001` |

Not faults on bc: `bad-cut` (no cone-break analysis), `commit-aux` (deferred),
`live-escape` (no live arbitration until OTF). Not a fault: inferring a buffer
producer although an explicit producer's object fits (LRM 3.0 Figure 5 allows
it; only an inferred action that feeds nothing is illegal).

pssc `tests/compliance/test_corpus_lookahead_calibrated.py` stays as well: it
switches lookahead off at lowering and requires *each* lookahead test to fail on
some seed, where the matrix needs only one.

## Adding a fault

1. Add the name and a one-line description to `DOC` in `faults.py`.
2. Add one guarded site, `if faults.on("name"):`, where the rule is enforced.
3. Find a killer: run candidate corpus tests with `ZSP_BC_FAULT=name` (or
   `with faults.inject("name")`) through pssc's bc adapter and the checker, and
   list the tests whose verdict is not PASS.
4. Add `["name"] tests = [...]` to `fault_kill.toml`. If no corpus test kills
   it, write one in pss-corpus (`compliance/<area>/<name>/test.{pss,json}`), or
   put the fault in `PENDING` with the reason until one exists.
