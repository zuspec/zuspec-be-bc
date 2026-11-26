# Activations: base offsets and the scope solve (P1.4)

An exported action runs as one **activation**: one object holding every action
its activity can run, each a node of the action tree with its own slot range
(ir-core `ScActionTree`, design P1-D1). This page specifies the bytecode that
runs it. The Python oracle implements all of it; the native engine implements
it from the `ACT` section and refuses what it cannot solve yet (below).

## The frame's base

A frame runs at its node's **base**: `LD_FIELD`/`ST_FIELD` slot `k` is object
slot `base + k`, and so is a `SOLVE` write-back pair's slot. The root frame's
base is 0.

| INVOKE flag | Meaning |
|---|---|
| `INSTR_F_NODE` (0x08) | The callee is a node of the caller's activation. It runs on the caller's object at `caller.base + imm`. `arg2` is the traversal site's index among the caller type's sites, which names the node in the activation table. |
| `INSTR_F_INITED` (0x10, with `NODE`) | The caller has already applied the callee's attribute initial values and its initializers to the callee's slots (LRM 11.3.1 b i-ii). The callee starts at pc `arg3`, past its own initial values. |

An INVOKE without `NODE` keeps the M1 meaning: the oracle gives an action
coroutine a fresh object, and the engine shares the caller's at the caller's
base. A synthesized PAR/SELECT branch runs at its parent's node.

## SCOPE_ENTER (0x81, pss)

`imm` is the local index of an activity block within the frame's action type.
The node's first scope, plus `imm`, is the block in the activation table.
Entering it:

- resets every node traversed in the block and its nested blocks, with their
  subtrees: no committed values, no traversal run (LRM 13.4.8);
- for a branch body (select branch, if/else arm, match case) or a loop body
  that may not run, commits its structure: its traversals become lookahead.

Entering a node through INVOKE resets the node's subtree the same way, because
a node's handles are uninitialized on entry to its activity.

Only a model with a cone emits `SCOPE_ENTER`.

## SOLVE_NODE (0x82, pss)

`arg0` is the problem the frame's node would solve alone. When the node is a
member of a cone of the activation, the node is solved in that cone instead:

1. **In force** — every constraint of the cone whose nodes are all live. A
   node is live if it holds values, has started, or a traversal of it is still
   to come in a live block. Beyond that:
   - a type constraint holds while its owner is live;
   - an activity constraint holds while its block is;
   - a `with` holds at its own traversal, as lookahead before it, and after
     it while the values it chose stand.
2. **Pinned** — committed values, and non-rand values: the object's value for
   a node that has started, otherwise the attribute's constant initial value.
3. **Free** — the node's own rand values, and the rand values of live nodes
   still to be traversed. That freedom is the lookahead (LRM 13.4.9).
4. **Solve** — one seed draw from the frame's stream, as `SOLVE` makes. Only
   the node's own rand slots are written back; the node is then committed.
   When the node chooses its component instance, the frame then runs in the
   instance chosen ([components.md](components.md)).

If no values satisfy the constraints in force, the run fails with an error
naming the traversal, the constraints in force and the pinned values. There is
no fallback.

**The cone's last solution.** A solve finds values for every variable of the
cone, and the cone keeps them as its witness, with the constraints that were
in force and the variables that were free. Step 4 takes the node's values
from the witness instead of solving when all of these hold:

- every value pinned in step 2 equals the witness's value;
- the constraints in force are among those the witness satisfied;
- every variable of the node that is free now was free in the witness's
  solve;
- the node has not taken values from this witness already.

The witness is then a solution of this step's problem, so the result is one
the solve could have returned. The last two conditions keep it random: the
node gets values the witness's solve chose for it, never values that were
pinned there. A loop iteration therefore never repeats the values of the
iteration before. No seed is drawn beyond the one step 4 always draws.
`ActivationTable.reuse = False` solves every traversal; that is for
comparison only.

**Calibration only.** When the tree's `lookahead` is false
(`PSSToScenarioPass(lookahead=False)`), step 1 also drops every constraint
that reads a node that is neither the traversed node nor committed. The solve
is then greedy: nothing still to come constrains the choice. Tests use it to
show that a lookahead test fails without lookahead. It is not a mode a model
can select.

A node in no cone solves `arg0` exactly as `SOLVE` would (P1-D3). Only a type
with a node in some cone emits `SOLVE_NODE`, so a model with no cone keeps its
bytecode.

## The activation table

`ZbcModel.activations` maps an entry coroutine to its `ActivationTable`
(`interp/activation.py`). The table holds the tree's nodes, blocks, sites and
cones, with each cone's constraints as IR. It is built at lowering. The
writer stores it in the `ACT` section (kind 27; P4.10a). A reader rebuilds it as
`ZbcModel.image_activations`: the constraints there are None, and each cone
solves from the dv-solve blobs the section carries, so a model read from an
image runs the same table as the one that wrote it. The section is absent when
no coroutine has an activity.

### The `ACT` section

One `zbc_acttab` (64 bytes) of counts, then, in this order: `zbc_actinit[n_init]`,
`zbc_act[n_acts]`, `zbc_actnode[n_nodes]`, `zbc_actscope[n_scopes]`,
`zbc_actsite[n_sites]`, `zbc_actcone[n_cones]`, `zbc_actvar[n_vars]`,
`zbc_actcon[n_cons]`, `zbc_actpool[n_pools]`, `zbc_actclaim[n_claims]`,
`zbc_actwrite[n_writes]`, `zbc_actpick[n_picks]`, a pool of u32 lists
(`n_u32`), the u32 string offsets (`n_strings`), the NUL-terminated string text
(`strb_size`, padded to 8) and the blob area (`blob_size`; each blob 8-aligned,
offsets relative to the area's start).

Record indices (the `*0` fields) are section-global; ids inside an activation
(node, scope, site, cone, pool) are local to it. Strings are section-local
indices; `0xFFFFFFFF` is "none". A list is `<x>0` (first entry in the u32 pool)
and `n_<x>`.

| Record | Bytes | Holds |
|---|---|---|
| `zbc_actinit` | 16 | a non-rand attribute's initial value: `slot`, `unknown` (1: computed at traversal), `value` |
| `zbc_act` | 80 | one activation: `coro` it is rooted at, object `size`, `root` and `type` strings, `flags` (bit 0: lookahead), its node/scope/site/cone/pool ranges, `names0` (a string per object slot) and its `init` range |
| `zbc_actnode` | 80 | an action node: `path`, `type`, `base`, `size`, `parent`, `decl_scope`, candidate instances (`comp0`), `comp_slot` (the slot its solve writes the chosen instance to), its claims, state and buffer writes, picks and children |
| `zbc_actscope` | 16 | an activity block: `kind` (index in ir-core `ScopeKind` order), `parent`, `node` |
| `zbc_actsite` | 16 | a traversal site: `owner` node, `target` node, `scope` |
| `zbc_actcone` | 40 | a solve cone: `id`, its nodes, variables and constraints, and `base_off`/`base_size`, the dv-solve blob of its variables |
| `zbc_actvar` | 48 | a cone variable: `name`, `node`, `slot`, `width`, `flags` (1 signed, 2 rand, 4 busy_lock), `comp_read`, loop-local binding (`loop_local`, `loop_node`, `loop0`), `live_from`, `busy_pool`, `pick` |
| `zbc_actcon` | 40 | a cone constraint: `kind` (0 type, 1 activity, 2 with, 3 comp), `owner`, `scope`, `site`, its nodes, its blob (the cone's variables and this constraint) and `text`, the constraint as written for unsat messages |
| `zbc_actpool` | 48 | a pool instance: `id`, `inst`, `name`, `kind` (buffer/stream/state/resource), `size`, slots, leaves and initial values |
| `zbc_actclaim` | 24 | a resource claim: `ref`, `lock` (1 lock, 0 share), `iid_slot`, its (inst, pool) pairs |
| `zbc_actwrite` | 24 | a state or buffer output: `ref`, source slots, (inst, pool) pairs |
| `zbc_actpick` | 24 | an unbound buffer input: `ref`, slots, `sel_slot`, (inst, pool) pairs |

The field-level definitions are the records in `format/spec.py` (and the
generated `zbc_format.h`). A solve compiles the cone's base blob, then adds the
blob of each constraint in force (dv-solve `add_constraint`); the IR of the
constraints is not in the section.

## Native engine

The engine reads the `ACT` section and implements `INSTR_F_NODE`, the base
offset, `INSTR_F_INITED`, `SCOPE_ENTER` and `SOLVE_NODE` (P4.10). `zbc_run`
still refuses, before running any of it, with `ZBC_ERR_UNSUPPORTED_OP` and
`halted_op` set to the opcode (P1-D6):

- a `SOLVE_NODE` in an image with a cone whose activation has a pool other than
  state, buffer, or a resource pool of at most 64 instances
  (`act_solve_supported`).

It refuses with `ZBC_ERR_COMP_INIT` an image whose nodes run in an instance
other than their parent's (`ZBC_COMP_F_NODE_INST`) but that has no `ACT`
section ([components.md](components.md)).
