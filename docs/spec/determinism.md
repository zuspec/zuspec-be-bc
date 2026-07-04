# Determinism spec

Status: **P0 contract (M1) — the highest-leverage P0 deliverable** · Reference
impl: `zuspec.be.bc.determinism` · Companion to design `[D§15.1]` and the value
ABI (`value-abi.md`, §4). This document is authoritative; the module is its
executable form.

Cross-tier determinism is a **hard requirement** `[D§2a, §15.1]`: identical seed +
identical traversal/solve order must yield **bit-identical** results and
**byte-identical** traces in the Python oracle, the C interpreter, and the AOT
model. Differential-testing-in-simulation (the compiled tier's *only* validation)
is meaningless otherwise. This spec writes down the **specified** order for
everything an execution observes.

The anchors are **inherited, not invented** — we formalize existing behavior
rather than design a new order.

## 1. Seed progression (the LCG)

The seed stream is the legacy LCG, unchanged:

```
state' = (state * 6364136223846793005 + 1442695040888963407)  mod 2**64
```

- A `SeedStream` is created from the user-supplied root seed `S0`.
- Each draw calls `next_raw()` = one LCG step, returning the new 64-bit state.
- A bounded draw is `next_below(n) = next_raw() % n`. The modest modulo bias is
  **accepted on purpose** in M1: the contract here is *reproducibility*, not
  statistical quality (constraint randomness is the solver's job, §4). Changing
  this mapping is a conscious, versioned change.

## 2. Seed forking (child coroutines / PAR branches)

When execution forks — a `SPAWN` of a child coroutine, or creation of the i-th
`PAR` branch — the child gets a derived stream:

```
fork_seed(parent_state, index) = lcg_next(parent_state XOR (index * 0x9E3779B97F4A7C15))
```

- `index` is the branch/child index in **declaration order** (§3).
- The derivation reads the parent's stream state **at the fork point**; because
  traversal order is itself specified, the entire fork tree is reproducible.
- `0x9E3779B97F4A7C15` (the golden-ratio odd constant) decorrelates sibling
  streams so branch 0 and branch 1 do not share a trajectory.

## 3. Traversal / SELECT / loop order

- **PAR branches** are created and seeded in **source declaration order**
  (`index = 0, 1, 2, ...`). The scheduler may *run* ready frames in its own ready-
  queue order, but branch *creation* and *seeding* order is fixed here so seeds
  are stable regardless of scheduling.
- **SELECT** draws once against the cumulative weights, branches considered in
  declaration order; the first branch whose cumulative weight exceeds the draw
  wins (`select_choice`). Zero-weight branches are unreachable.
- **Loop iterations** sequence in order; each iteration that forks/draws does so
  against the loop body's stream in iteration order.
- **Flow-object binding** resolves consumers in declaration order, and for each
  consumer, candidate producers in **sorted-by-name** order (the same ordering
  discipline as the solver var-map), so a bind choice is reproducible.

## 4. Solver variable ordering

A solver `var_id` is the index of the field name in the alphabetically **sorted**
list of the scope's randomizable field names — see `value-abi.md` §4 and
`zuspec.be.bc.abi.value.solver_var_map`. This spec **references** that rule (it is
the value ABI's to own) and adds one external-dependency note:

> **dv-solve's internal search order is a pinned external determinism
> dependency.** We specify the seed we hand the solver and the var ordering we map
> results back through; we do **not** re-derive the solver's internal search. The
> solver version is version-locked, and any solver upgrade that perturbs results
> is treated as an ABI-relevant change (re-baseline goldens). This is the resolved
> "depth call" of roadmap open-question #5 (plan §7.1).

## 5. Trace determinism

The trace serialization (`trace/schema.py`) emits JSON with sorted keys and
compact separators, and events carry a monotonic `seq`. Given §1–§4, two runs with
the same root seed produce a **byte-identical** trace — the property T1-D checks.

## 6. Checklist (referenced by later phases' tests)

A construct is determinism-complete when all of these have a specified,
implemented order:

- [ ] root seed → `SeedStream(S0)`
- [ ] child/branch seed via `fork_seed(parent_state, index)`, `index` in decl order
- [ ] PAR branch creation/seeding in declaration order
- [ ] SELECT via `select_choice` (cumulative weights, declaration order)
- [ ] loop iterations in order; per-iteration draws in iteration order
- [ ] flow-object binding: consumers in decl order, producers sorted by name
- [ ] solver var_id = index in `sorted(names)` (value ABI §4)
- [ ] solver version pinned; result mapping via `solver_var_map`
- [ ] trace serialized canonically (sorted keys, monotonic seq)

## 7. Versioning

The LCG constants, the fork mixing constant, the `next_below` mapping, and the
SELECT rule are all part of the determinism contract. Changing any of them changes
observable results and therefore requires a determinism-version bump (and re-
baselining of golden traces); today that version is carried alongside `abi_id`.
