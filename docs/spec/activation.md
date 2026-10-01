# Activations: base offsets and the scope solve (P1.4)

An exported action runs as one **activation**: one object holding every action
its activity can run, each a node of the action tree with its own slot range
(ir-core `ScActionTree`, design P1-D1). This page specifies the bytecode that
runs it. The Python oracle implements all of it; the native engine implements
the base offset and refuses the rest (below).

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

## SCOPE_ENTER (0x4A)

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

## SOLVE_NODE (0x4B)

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
cones, with each cone's constraints as IR. It is built at lowering and is
in-memory only, like the M1 problem table, until P7 specifies its bytes.

## Native engine

`zbc_run` refuses an image containing `SCOPE_ENTER`, `SOLVE_NODE`, or an
`INVOKE` with `INSTR_F_INITED` before running any of it. The result is
`ZBC_ERR_UNSUPPORTED_OP`, with `halted_op` set to the opcode (P1-D6). P8
ports these ops. The engine does implement `INSTR_F_NODE` and the base offset.
