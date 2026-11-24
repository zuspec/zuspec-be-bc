# Calls: `ARG`, `LD_ARG`, `CALL`, `LD_RES`

A native PSS function is inlined at each call site. A call to a function that
is already being inlined, which is recursion (direct or mutual), cannot be, so
it is a `CALL` of the function's **called form** (bc procedural gaps B-D5).
That is a coroutine of its own, lowered once per function and component
instance. A model with no recursion has no `CALL` and keeps its bytecode.

| Op | Operands | Meaning |
|---|---|---|
| `ARG` (0x18) | `arg0` = rs, `arg1` = i | stage argument `i` of the frame's next `CALL` (or of an `IMPORT` with `imm` > 0, host-abi.md "Import arity") |
| `LD_ARG` (0x19) | `arg0` = rd, `arg1` = i | rd ← argument `i` of this frame's call (0 if none was staged) |
| `LD_RES` (0x1A) | `arg0` = rd, `arg1` = i | rd ← result `i` of this frame's last `CALL`: what the callee staged by `ARG` before its `RET` (0 if none) |
| `CALL` (0x4C) | `arg0` = coroutine, `arg1` = rd | call the coroutine with the staged arguments; rd ← its `RET` value (`0xFFFFFFFF`: none) |

**The call.** The callee runs at once, before any other ready frame. It runs
in the caller's action object, base, node and component instance. Its `RET`
completes it, and the caller resumes at once with the value in `rd`. A callee
may suspend (a blocking import, `yield`), and the caller waits for it. A call
is therefore no scheduling point of its own, as an inlined body is none.

**Arguments.** At most 16 (`CALL_MAX_ARGS`). Lowering evaluates every
argument before staging any, since evaluating one may itself `CALL`.

**Structs.** A struct argument is one argument per leaf, in declaration order.
A struct parameter is a handle to the caller's instance (LRM 20.3.2), so a
called form passes it copy-in, copy-out: before each `RET` the callee stages
by `ARG` first a struct result's leaves (a struct result's `RET` has no
operand), then the leaves of each struct parameter the body may write
(assigns into it, passes it whole, or calls a method of it; a read-only one is
not handed back); the caller reads them with `LD_RES` right after the `CALL`
and stores the parameters' leaves back to the argument. Values in and out are
each at most 16. A function with no struct
parameter or result stages nothing, so its bytecode is unchanged.

**Depth.** A `CALL` from a frame already 1024 calls deep (`CALL_MAX_DEPTH`) is
a run-time error in both engines: `VMError` in the oracle, and
`ZBC_ERR_CALL_DEPTH` (-24) with `halted_op` `CALL` in the native engine.

**Seeds.** A `CALL` forks no seed stream (`determinism.md` §2).

**Native engine.** It implements all three. A `CALL` is nested on the
caller's thread (`zsp_timebase_call`), as a blocking `INVOKE` is. An error in
any frame, a nested callee's included, is the run's: the first one wins, and
a frame resuming after it stops.
