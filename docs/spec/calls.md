# Calls: `ARG`, `LD_ARG`, `CALL`

A native PSS function is inlined at each call site. A call to a function that
is already being inlined, which is recursion (direct or mutual), cannot be, so
it is a `CALL` of the function's **called form** (bc procedural gaps B-D5).
That is a coroutine of its own, lowered once per function and component
instance. A model with no recursion has no `CALL` and keeps its bytecode.

| Op | Operands | Meaning |
|---|---|---|
| `ARG` (0x18) | `arg0` = rs, `arg1` = i | stage argument `i` of the frame's next `CALL` |
| `LD_ARG` (0x19) | `arg0` = rd, `arg1` = i | rd ← argument `i` of this frame's call (0 if none was staged) |
| `CALL` (0x4C) | `arg0` = coroutine, `arg1` = rd | call the coroutine with the staged arguments; rd ← its `RET` value (`0xFFFFFFFF`: none) |

**The call.** The callee runs at once, before any other ready frame. It runs
in the caller's action object, base, node and component instance. Its `RET`
completes it, and the caller resumes at once with the value in `rd`. A callee
may suspend (a blocking import, `yield`), and the caller waits for it. A call
is therefore no scheduling point of its own, as an inlined body is none.

**Arguments.** At most 16 (`CALL_MAX_ARGS`). Lowering evaluates every
argument before staging any, since evaluating one may itself `CALL`. A struct
parameter or result on a called function is refused at lowering for now.

**Depth.** A `CALL` from a frame already 1024 calls deep (`CALL_MAX_DEPTH`) is
a run-time error in both engines: `VMError` in the oracle, and
`ZBC_ERR_CALL_DEPTH` (-24) with `halted_op` `CALL` in the native engine.

**Seeds.** A `CALL` forks no seed stream (`determinism.md` §2).

**Native engine.** It implements all three. A `CALL` is nested on the
caller's thread (`zsp_timebase_call`), as a blocking `INVOKE` is. An error in
any frame, a nested callee's included, is the run's: the first one wins, and
a frame resuming after it stops.
