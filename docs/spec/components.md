# Components: the component object and `comp` (P1.5)

The elaborated component tree under the root component (ir-core
`ScComponentTree`, LRM 9.1.4) is ONE object, the **component object**. Each
instance owns a slot range of it, laid out the way an action's subtree is
(P1-D1): a component type's data attributes, a struct attribute one slot per
scalar, with each sub-instance's subtree in place of its field, in declaration
order, base type's fields first. A type's subtree has the same layout wherever
it is instantiated, so a sub-instance sits at a static slot offset from its
parent.

Instances are numbered in pre-order, the root 0. A sub-instance's number is its
parent's plus a static offset, so `comp.sub1` is `comp + k`: a linear
expression, which the solve can choose over.

The Python oracle implements all of this. The native engine refuses it (below).

## The frame's instance

A frame runs in one instance, `comp`, and at that instance's first slot, its
component base. The root frame of an activation runs in its node's instance. A
traversal's child runs in its parent's instance plus the node's static offset,
or, when the node has more than one candidate, in the instance its solve
chooses (below). Any other child (a PAR/SELECT branch, an INVOKE without
`INSTR_F_NODE`) runs in its parent's instance. A frame with no component tree
runs in instance 0.

| Op | Operands | Meaning |
|---|---|---|
| `LD_COMP` (0x16) | `arg0` = rd, `arg1` = slot | rd ← component object slot `(cbase + arg1) mod 2^32` |
| `ST_COMP` (0x17) | `arg0` = rs, `arg1` = slot | component object slot `(cbase + arg1) mod 2^32` ← rs |

`arg1` is static, and the sum wraps at 2^32, so a slot before the frame's
instance (one a component reference reaches outside the frame's subtree,
F6.8b3b) is `2^32 - n`. A path below the frame's instance (`comp.a.sub.k`, or
`self.k` in a component function inlined at a call through `comp.a.sub`) is a
fixed slot offset, and so is an element of a component array: the slots are
static per element. Lowering makes every index static. A `foreach` over a
component array is unrolled, one copy per element. An index known only at run
time is a dispatch: the access, call or store is lowered once per element,
behind a compare of the index with that element's number. An index matching
no element reaches a `BUILTIN_ERROR` ("index out of bounds"). No opcode takes
a slot from a register.

`LD_COMP`/`ST_COMP` in a frame whose instance is not chosen yet (an action's
`pre_solve`, when its solve chooses its instance) is a run-time error.

## Channels

A `channel_c<T, D>` attribute (LRM 21.9.1) is `2 + D` slots of the instance that
declares it, laid out by ir-core's `comp_tree`: `c.$count`, `c.$head`, then
the ring `c.$buf0` .. `c.$buf<D-1>`. All are 0 at construction. A channel whose
element is not a scalar keeps one opaque slot, and calling its functions is a
lowering error.

The four functions are inlined at each call, with `LD_COMP`/`ST_COMP`:

- `try_put(v)`: `v` is evaluated; if `count < D`, it is stored at
  `(head + count) % D` and `count` grows. The result is whether it was stored.
- `try_get(t)`: if `count > 0`, the element at `head` is assigned to `t` (an
  lvalue; the parameter is `output`), `head` advances modulo D, and `count`
  shrinks. Otherwise `t` is untouched. The result is whether an element was
  taken.
- `put(v)` / `get()`: wait until there is room or an element, then as above.

A ring position is a register value, so the element slot is chosen by a
compare-and-branch per element, as for a component-array index.

The wait is a loop around `YIELD` with `INSTR_F_SPIN` (0x20): the frame tests
its condition, yields with the flag while it is false, and tests again. A frame
that resumes just past a SPIN yield and reaches the same one again has made no
progress. When more such runs have happened in a row than there are ready
frames, every ready frame is waiting. Then the oracle advances time to the
next timed frame, or, if there is none, stops with a `deadlock` error. This
form is temporary; an event wait will replace it.

Independently of channels, a run that goes quiet with its entry coroutine
unfinished is an error ("the entry unfinished"), never a success.

## Construction

A coroutine, `$comp_init`, constructs the tree. It runs to completion before
the entry action starts (LRM 20.1.3: before the root action's `pre_solve`), in
instance 0, so each block's slots are absolute:

1. every instance's declared attribute initial values (9.1.4.1 d);
2. each instance's `exec init_down`, top-down (pre-order);
3. each instance's `exec init_up`, bottom-up (post-order).

This is the order LRM Example 281 lists. An instance runs its type's block, or
its nearest base's when its type declares none.

The construction coroutine's index is in the in-memory table
(`ZbcModel.components`, `interp.components.CompTable`) until P7 specifies it.
The image header carries `ZBC_HDR_COMP_INIT` (0x0004) when the image has one.

## The instance an action runs in (P1-D4)

Each node of an activation has **candidates**: the instances of its action's
component type in the subtree of its parent's instance (9.1.5.1), as offsets
from it (the root's, from instance 0). None is a lowering error (LRM Ex 51).

- **One candidate:** the instance is static, relative to the parent's.
- **More:** the node's instance is a variable of its cone, held in a slot of
  the activation's object past the action subtrees (`ScActionNode.comp_slot`).
  A `COMP` constraint keeps it among the candidates while the node is live.
  `SOLVE_NODE` chooses it with the node's other values and sets the frame's
  instance from it.

A traversal's `comp == X` (`with { comp == this.comp.sub1; }`, LRM Ex 143) is a
`WITH` constraint over the two instances. When both are static, it is decided at
lowering: it holds and is dropped, or it never holds and is an error.

## Native engine

The native engine reads the `COMP` section (kind 21), written in every profile
when the model has components: one `zbc_comp` (16 bytes: `n_slots`,
`init_coro` (`0xFFFFFFFF` = none), `n_inst`, `flags`) then `n_inst` u32
instance bases. `zbc_run` allocates `n_slots` component slots and runs
`init_coro` to completion before the entry, so construction is never skipped.
`LD_COMP`/`ST_COMP` address instance 0 (base 0); a slot past `n_slots` halts
with `ZBC_ERR_OOB_COMP` (-26). `ZBC_COMP_F_NODE_INST` says an activation node
runs in a non-root instance; `zbc_run` refuses it with `ZBC_ERR_COMP_INIT`, as
it does `ZBC_HDR_COMP_INIT` without a `COMP` section. All refusals happen
before anything runs (P1-D6).

`ZBC_COMP_F_RUN_HOOKS` says the tree has `exec run_start` or `exec run_end`
blocks. Coroutine `init_coro + 1` (`$run_start`) runs them top-down and
`init_coro + 2` (`$run_end`) bottom-up; the VM drains `$run_start` before a
root action's entry and `$run_end` after it (`VM.root_frame`). In a simulation
(one run, many action calls on a shared tree, `VM.root_frame(cobj=, hooks=)`)
`$run_start` runs before the first action only, and `$run_end` when the host
calls the reserved entry `$run_end` ([host-abi.md](host-abi.md)). Both are always
written when the flag is set, even if one is empty. The native engine does not
run them yet and refuses the flag with `ZBC_ERR_COMP_INIT` before anything runs.

A `YIELD` with `INSTR_F_SPIN` (a channel's blocking wait) runs as the
oracle's: once every ready thread has come back to its SPIN yield with no
progress, the run advances to the next timed thread, and with none it halts
with `ZBC_ERR_DEADLOCK` (-27, `halted_op` = `YIELD`) where the oracle raises.
