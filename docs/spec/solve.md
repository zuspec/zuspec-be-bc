# Solving: constraints, seeds and the `solve` event

How a bc image picks random values: which constraint forms reach the solver,
how soft constraints rank, how seeds are derived, and what a solve reports
when it fails or when it is watched. Both engines (the Python VM in
`interp/`, the C engine in zuspec-rt-eng) follow this page; where they differ
it says so.

## The solver

Every solve goes to dv-solve. A traversal's solve (`SOLVE_NODE`) solves one
*cone* of the activation table ([activation.md](activation.md)): the cone's
base problem (its variables) plus one problem per constraint in force,
composed with dv-solve `add_constraint`, then pins (values already fixed:
earlier traversals, pre_solve inputs, pool instances), then the search.
A `SOLVE` op (procedural `randomize x with {...}`) solves one problem built
at lowering time, with its inputs pinned.

Constraint forms on bc:

| Form | Where | Notes |
|---|---|---|
| hard expression, `if`/`implies` | type, `with`, activity | |
| `soft e` | type, `with`, activity | top level only; `soft` under `if` is refused by line |
| `dist` (`:=`, `:/`, ranges) | type, `with`, activity | top level only |
| `unique {...}` | type, `with`, activity | path operands; arrays refused until M6 |
| `forall (it: A [in h])` | activity | expanded per handle in pssc |

**Soft and dist on a cone.** When a soft or a dist is in force the Python VM
solves one whole problem of the constraints in force (`ActivationTable.blob`).
The C engine composes per constraint with dv-solve `add_constraint`, which
keeps an added problem's softs (ranked with the others) and constant-weight
dists; it refuses a dist with expressions (`DVS_COMPILE_ADD_UNSUPPORTED`).
Both give the same values (pssc `test_bc_solve_trace.py`).

## Soft ranks

A hard constraint always wins. Among softs (the rule for `default`, LRM 13.1.11 d):
later > earlier, derived or extension > base, an action > its struct, and at
one node `with` > activity constraint > type. ir-core gives each
`ConstraintSoft` a `rank`; zuspec-solver maps it to an absolute dv-solve
priority (`(_SOFT_RANKS-1-rank)<<8 | ties`), absolute because cone
constraints are compiled apart. dv-solve drops a soft only on a conflict.

## Seeds

The run seed is the root. By default seeds are *keyed* (`RunOpts.seed_mode =
"keyed"`; C `pssvm_load` option `seed_mode`, keyed when the image has a
SEEDKEY section): each traversal's seed is forked from its parent's origin
and a key from the child's name and position, so adding an unrelated action
leaves the other actions' values unchanged. `seed_mode=stream` (DPI:
`+pssvm.seed_mode=stream`) restores the older positional seeds. Both engines
give the same values for the same seed and mode, in and out of simulation.

## When a solve fails

An unsatisfiable solve is an error naming the conflicting constraints by
file and line (`; conflicting constraints: a.pss:5, a.pss:9`): a deletion
filter over the problem's source (`interp/unsat_core.py`). The C engine has
no source: its error JSON names the failing solve (`"solve"`, or
`"act"`/`"cone"`/`"enabled"`) and its `"pins"`, and the host maps that back
(pssc `testing/native.py` `explain_unsat`).

## The `solve` debug event

While `EV_SOLVE` is watched ([trace-schema.md](trace-schema.md)), each solve
raises `solve`:

```
{site, node, type, seed, values: {name: value}, softs, dropped: [{text, file, line}]}
```

`softs` counts the soft constraints in force; `dropped` lists those the
values break. A problem loaded from an image without source lists none.

The C engine sends raw keys instead of `softs`/`dropped`, which need the
source: for a cone solve `act`, `cone`, `enabled` (the constraints in force)
and `witness` (the cone's values); for a `SOLVE` op `solve` and `solved`.
A host that has the model fills the rest (pssc `testing/native.py`
`explain_solve`, called for every event of a native run). Events raised in
simulation (DPI) go to the bench's sink unchanged, so they keep the raw keys.
