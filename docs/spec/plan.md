# Planner v0: schedules, joins, streams and the realised plan (M8b)

How the activity statements whose order or membership the source leaves open
reach bytecode, and what a run records about the choices it made. The rule is
**hooks first**: no new VM op and no new image section. The compiler
(zuspec-ir-core) works out the legal orders and lowers them onto ops that
already exist (`PAR`, `SELECT`, `JOIN`); the run picks one from the frame's
seed (`determinism.md` §3), so the same seed gives the same order on both
engines. Pre-computed plans, plan opcodes, state tickets and a `PLAN` section
are not part of v0.

## `schedule`

ir-core `xf/plan.py` computes a `Footprint` per schedule member (action
tree `schedule_footprints`): buffer, state and stream types in and out, and
resource types locked and shared (type identity = the resolved type object).
`plan_schedule(fps, before, together)` returns `SchedulePlan(edges, groups,
orders)`:

| Members i, j | Result |
|---|---|
| i outputs buffer type T, j inputs T | edge i → j |
| both use state type T, one writes it | tie: serialized, either order |
| both claim resource type R, one with `lock` | tie |
| scheduling constraint `sequence {i, j}` | edge i → j |
| stream out in i, stream in in j | `together` |
| `parallel {i, j}` constraint | `together` |

Members joined by edges or ties form a group (union-find). The legal orders
of a group are its linear extensions, at most `MAX_ORDERS` (24); above that a
sample of 24 drawn with `Random(0)`, so the image does not depend on the run.
A cycle is `CycleError`; `together` members that are tied, or in a group of
more than one, are `ParallelError`. Both are compile errors located at the
schedule.

Lowering (`pss_lower/lower.py` `_lower_planned`): the schedule becomes a
`ScPar` with the schedule's scope. A member alone in its group is a `ScSeq`
branch; a group with one order is a `ScSeq` in that order; a group with
several orders is a `ScSelect` over them (weight 1 each, no scope), so the
seed draws the order. Duplicated member statements keep their node and site
ids: only one branch of the select runs. The action tree (`_Walker.walk`)
gives each group of more than one a `SEQUENCE` sub-scope, so the activation
sees its members as ordered (not concurrent) and does not refuse two lockers
of a one-instance pool.

## `join_select` / `join_branch`: `JOIN` with `INSTR_F_SUBSET` (0x40)

`ScPar.join_branches` (labels or handles resolved to branch indices) lowers
(`lower/parallel.py` `lower_par`) to a `JOIN` with `INSTR_F_SUBSET`:

- `arg0` = k, the block's branch count (the last k children spawned);
- `imm` > 0: `join_select(imm)`: draw `imm` branches without replacement, each
  one draw of `select_choice(frame.seed, [1]*remaining)`, in draw order;
- `imm` = 0: `join_branch`: `arg1` is the mask of joined branches.

A count of 0 lowers to join none, n ≥ k to join all; more than 32 branches is
refused. Branches not joined are not counted and run on; they are never
cancelled. The `JOIN` trace event carries `joined` (sorted indices). The C
engine refuses `JOIN`+`SUBSET` in `zbc_image_check` (`UNSUPPORTED_OP`).

## Streams

Paired in the action tree (`_flow` → `_streams`), not in the VM: each pair
(bound by `bind`, or unbound outs and ins grouped by scope, pool and type and
paired in walk order) becomes `==` constraints on the stream's leaves, of kind
ACTIVITY at the enclosing `PARALLEL`/`SCHEDULE` scope, plus a guard that both
ends reach the same pool instance. Each end must be a direct branch of that
scope.

## Dead `select` branches

ir-core `FlowChecks.dead_branches` marks a branch `SelectBranch.infeasible`
when, in every instance of its action, a must-run traversal inputs a buffer no
action type outputs, or its starting locks plus the other parallel branches'
exceed the pool. The branch is not walked and lowers to
`ScSelectBranch(guard=false, body=[])`, keeping branch indices and weights.
All branches dead and no `allow_none` is a compile error at the select.
Pruning on run-time state: see "Inference" below.

## Inference

LRM 18.2 / Annex E. A buffer, state or stream input the activity does not
feed gets an action the activity does not traverse. Which actions could be
inferred is decided at compile time; which one runs is decided at run time.

- **Candidates (ir-core).** `xf/pss_lower/infer.py` `InferPlanner` lists, per
  demand, the producers the ICL tables give for the input's pool whose output
  type fits, and recursively the producers of their own buffer inputs (state
  inputs read the current object; stream inputs pair). Limits (D5):
  `MAX_DEPTH` 8 actions per chain, `MAX_SLOTS` 32 candidates per demand,
  an action already on the chain is not a candidate again. A traversal whose
  input has producers but no viable chain is a located compile error
  (`FlowProblem` kind `infer`, "no inference chain can feed input ...";
  corpus `neg.flow.infer_limit`). `action_tree.py` `_infer_nodes` then adds
  one hidden node per candidate (`ScActionNode.inferred_for`, path
  `<consumer>.<ref>$infer<k>`) with `INFER` scope constraints tying its
  output to the consumer's input, so the pair is solved jointly. The pick
  records them (`ScBufferPick.infer_nodes`; `state_infer`, `stream_infer`
  on the node).
- **Choice (be-bc).** A hidden node's constraints are in force only while it
  is turned on (`Activation.infer_on`). At a consumer's solve, when no object
  already in the pool fits, `Activation._infer_next` turns on the next untried
  candidate in seeded order and `ops_orch._run_inferred` runs it before the
  consumer (a stream partner starts with it instead). If the consumer still
  cannot solve, the next candidate is tried (chronological retry, bounded by
  the candidate count); none left is `ScopeUnsatError` naming the tried
  candidates, or "no action can be inferred to produce one".
- **atomic (11.3.7).** Nothing is inferred inside an `atomic` block: at its
  `SCOPE_ENTER`, `Activation.atomic_infer` runs the producers its members will
  need before the block; one that cannot be fed there fails with "no action
  is inferred inside an atomic block (11.3.7)".
- **select.** `Activation.select_blocked` drops, at choice time, a branch
  whose first traversal can neither pick an object nor infer one; all
  blocked and no `allow_none` is `VMError` "select: no branch can run".
- **Naming.** An inferred action's frame has `inferred_for` = the stimulus
  id it serves; it is in the trace's action-start detail, `bench_report`
  in-flight entries, the debugger's thread list and the attribution line.
  In the realised plan it is an ordinary occurrence.
- **C engine.** rt-eng never turns an `INFER` constraint on
  (`zbc_interp.c`); models that need inference run on pyvm until M14.


## The realised plan: `PlanLog` and `pss-plan` 1.0

`interp/activation.py` `PlanLog` is opt-in (`RunOpts.plan`, or a session's
`vm.plan`); without one nothing is recorded. It is filled from the
activation's hooks as the run decides: `enter_node`/`exit_node`
(occurrences, buffer and state writes), `commit` (fields, state reads),
`solve` (buffer picks), `_claim` (resource claims), `enter_scope`
(`if`/`repeat` outcomes) and `_op_select` (select choice). `to_json(seed)`
gives:

```json
{"format": "pss-plan", "version": "1.0", "seed": 1,
 "occurrences": [{"id": 0, "site": null, "path": "", "stim": "...", "type": "...",
                  "comp": 0, "fields": {}, "start": 0, "end": 9, "completed": true}],
 "flow": [{"id": 0, "kind": "buffer", "pool": "bp@0", "producer": 1,
           "consumers": [2], "fields": {"v": 3}}],
 "resources": [{"pool": "rp@0", "instance": 0, "claims": [{"occ": 1, "lock": true}]}],
 "order": [[1, 2]],
 "control": [{"occ": 0, "kind": "select", "select": 4, "value": 1}]}
```

- Occurrence ids are start order; `start`/`end` are the log's own event
  counter (not the activation clock).
- `flow.kind` is buffer, state or stream; `pool` is `name@inst`, or null for
  a pair joined by `bind`; an initial state object has producer null.
- `order` holds `[a, b]` when a ended before b started, transitively reduced;
  the root and still-running occurrences are left out.
- `control.kind` select (branch index, null when none was eligible), if
  (true = then) or repeat (iterations entered). `if`/`repeat` come from
  `SCOPE_ENTER`, emitted only in a model with a cone; a repeat of zero
  iterations has no record.

The schema is `checker/schema/plan-1.0.json` in pss-corpus. Because bc decides
at run time, its plan records the run rather than predicting it. pssc writes
it with `pssc test --plan-out DIR` (Python engine only; the C engine keeps no
plan log).
