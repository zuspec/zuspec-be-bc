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
| `LD_COMP` (0x16) | `arg0` = rd, `arg1` = slot | rd ← component object slot `cbase + arg1` |
| `ST_COMP` (0x17) | `arg0` = rs, `arg1` = slot | component object slot `cbase + arg1` ← rs |

`arg1` is static: a path below the frame's instance (`comp.a.sub.k`, or
`self.k` in a component function inlined at a call through `comp.a.sub`) is a
fixed slot offset. An element of a component array is reached only through a
constant index; a computed one is refused at lowering.

`LD_COMP`/`ST_COMP` in a frame whose instance is not chosen yet (an action's
`pre_solve`, when its solve chooses its instance) is a run-time error.

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

`zbc_run` refuses an image with `ZBC_HDR_COMP_INIT` set (`ZBC_ERR_COMP_INIT`),
so construction is never skipped, even when its blocks only call imports. It
refuses an image containing `LD_COMP` or `ST_COMP` with `ZBC_ERR_UNSUPPORTED_OP`,
`halted_op` set to the opcode. Both happen before anything runs (P1-D6). P8
ports them.
