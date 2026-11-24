# Engine debug stub

Status: **pssc P2.1** · Version `1` · Authority: `zuspec.be.bc.interp.debug`
(pyvm) · Companion to the pssc design (`pss-test-debug-env-design.md` §12.2).

An engine (pyvm, or the native engine) implements this small protocol so that
a debugger never has to be built into the engine. In the spirit of gdbserver's
RSP, the engine only knows images, coroutines, pcs, frame slots, component
instances and addresses. Everything with a source-level meaning (line →
pc, register paths, PSS expressions, DAP) lives once in the **front end**
(pssc daemon, P2.2) and is lowered to these messages.

Messages are JSON objects. The transport is the same in-process (pyvm:
`Stub.request(msg) -> reply`, events to a callback) and over a socket (one
JSON object per line, P3.9 / P4.5), so one conformance suite (T2.1) runs
against every engine.

Socket transport: `zuspec.be.bc.interp.stub_socket`. `serve_stub(stub, sock)`
serves any stub object on a connected socket (requests one at a time; an
event only between replies, so a request's events follow its reply; each
line the server writes adds `stub_state`, the stub's state then, which the
client removes). `SocketStub(sock)` is the client, with the stub interface
(`request`, `next_event`, `join`, `state`); a closed socket gives one
`exited` event `{detached: true, closed: true}`, and the server detaches a
stub still attached when its client goes away.

## Messages

Three shapes:

```json
{"seq": 7, "cmd": "set_pc_breaks", "args": {"breaks": [{"id": 1, "coro": 4, "pc": 12}]}}
```

```json
{"seq": 7, "ok": true, "body": {"breaks": [{"id": 1, "verified": true}]}}
```

```json
{"event": "stopped", "body": {"reason": "breakpoint", "thread": 3, "frame": 9, "coro": 4, "pc": 12, "src_ref": 41, "hits": [1], "detail": {}}}
```

- **Request** (front end → engine): `seq` (int, chosen by the front end,
  echoed in the reply), `cmd`, `args` (object; may be omitted when empty).
- **Reply** (engine → front end): exactly one per request, in request order.
  `ok: true` with `body`, or `ok: false` with
  `error: {"code": str, "message": str}`.
- **Event** (engine → front end, unsolicited): `event`, `body`. An event
  caused by a request (the stop that ends a `step`) is sent **after** that
  request's reply.

JSON is written with sorted keys and compact separators (as the trace,
`trace-schema.md`), so a scripted session's transcript is byte-comparable.

Error codes: `bad_request` (unknown `cmd`, missing or ill-typed args),
`not_stopped` (an inspection or resume command while running), `not_running`
(`stop` while stopped), `no_thread`, `no_frame`, `no_comp`, `no_coro`, `bad_handle`,
`unsupported` (a capability the engine lacks), `snippet` (an `exec_snippet`
fault; `message` says which).

## Execution model

The engine is **stopped** or **running**. It starts stopped (after
`attach`, before the first instruction of the root frame) and reports
nothing until resumed. With `"entry": false` it starts running instead: for
a run its host drives (a simulator calling into the engine), which stops
only at breakpoints, watches and the stop set, on the host's thread; the
host's call blocks while stopped. **All-stop:** when one thread stops, every thread is
stopped; a resume resumes every thread (a step steps one thread and lets the
others run as the scheduler would).

A stop happens only at an **instruction boundary**: before the instruction
at `pc` executes (pc breaks, steps, `stop`), or just after the instruction
that caused it (watches and the other stop events below). pyvm runs the VM on
its own host thread and blocks it in the event loop (`VM._run_frame_ev`, the
P1.6 hooks) while stopped; a run with no stub attached is unchanged (T-G1,
T-G3).

While running, the engine accepts only `stop`, `set_pc_breaks`,
`set_access_watch` and `set_stop_on` (they take effect at the next
instruction boundary). Everything else replies `not_stopped`.

### Threads and frames

- A **thread** is a coroutine activation the scheduler runs on its own: the
  root, and each frame started by `SPAWN`/`INVOKE` (an action, a parallel
  branch). Its id is that frame's id (`Frame.id`).
- A **frame** is an activation on a thread's stack: the thread's frame and
  each called function frame (`CALL`, `Frame.called`) above it. Frame ids
  are `Frame.id`; they stay valid while the frame is live.
- A thread is `running`, `ready`, `waiting` (on a join, channel, time or a
  blocking import) or `done`. Done and cancelled threads are not listed.

## Commands

### Session

| `cmd` | `args` | `body` |
|---|---|---|
| `attach` | `{"version": 1, "entry"?: bool}` | `{"version": 1, "engine": str, "capabilities": [str], "image": {"coros": int, "sha256": str}}` |
| `detach` | `{}` | `{}`; clears every break and watch and resumes; the run goes on unobserved |
| `terminate` | `{}` | `{}`; ends the run; an `exited` event follows with `{"status": "terminated"}` |

`capabilities` names the optional commands the engine has: `exec_snippet`,
`backdoor_read`, `reload`, `access_watch`. pyvm v1 has all but `reload`.

### Control

| `cmd` | `args` | `body` |
|---|---|---|
| `continue` | `{}` | `{}` |
| `stop` | `{}` | `{}`; a `stopped` event with reason `pause` follows |
| `step` | `{"thread": int, "kind": "stmt"\|"into"\|"out"}` | `{}`; a `stopped` event with reason `step` follows (or another reason, if something else stops first) |
| `set_pc_breaks` | `{"breaks": [{"id": int, "coro": int, "pc": int}]}` | `{"breaks": [{"id": int, "verified": bool, "message"?: str}]}` |
| `set_access_watch` | `{"watches": [{"id": int, "addr": int, "size": int, "rw": "r"\|"w"\|"rw", "value"?: int, "mask"?: int}]}` | `{"watches": [{"id": int, "verified": bool}]}` |
| `set_stop_on` | `{"kinds": [str]}` | `{"kinds": [str]}` (the kinds the engine accepted) |

- **`step`** works on statement markers (debug lowering's LINE table,
  `CoroDescriptor.stmts`):
  - `stmt`: run until this thread reaches the start of a statement in the
    same frame or a caller of it (a called function runs to its return);
  - `into`: until the next statement start in any frame of this thread;
  - `out`: until the current frame returns, then to the next statement start
    in its caller.

  A step whose thread ends stops at the next statement start of any thread.
  A frame with no statement markers (no debug lowering) is stepped over.
- **`set_pc_breaks`** replaces the whole set. `coro` is an index into the
  image's coroutines (`model.coros`), the same in every engine. A break is
  `verified` when `coro` exists and `pc` is in range; it hits **before** the
  instruction at `pc`, in every activation of that coroutine. The front end
  normally sets pcs that are statement starts.
- **`set_access_watch`** replaces the whole set. A watch matches a memory
  access (the `access` event: `read8..64`, `write8..64`, register reads and
  writes) that overlaps `[addr, addr+size)`, has a matching direction, and,
  when `value` is given, `(data & mask) == (value & mask)` (`mask` defaults
  to all ones). The stop is after the access, so a read's `data` is the value
  read. The front end resolves a register path (`ch[0].regs.csr`) to its
  address and size, and a field to a `mask`.
- **`set_stop_on`** replaces the set of event kinds that stop the run:
  `check` (a failing check), `sanitizer` (a failing sanitizer), `error` (an
  `error()`/`fatal()` message, or an engine error), `unimplemented` (a call
  to an import nothing provides), `mocked` (a call a double answered). The
  default is `["error"]`.
  *pyvm (v1):* accepts `check`, `error` and `sanitizer`; `check` and
  `sanitizer` stop only when the host turns those events on (the stub tees
  `vm.events`). `unimplemented` and `mocked` are not accepted in v1 (left
  out of the reply). A watch or stop-event stop is reported at the next
  instruction boundary (possibly in another frame), with the `src_ref` of
  the instruction that caused it. An engine error's detail is
  `{"error": message}` (`error()`/`fatal()` too).

### Inspection (stopped only)

| `cmd` | `args` | `body` |
|---|---|---|
| `threads` | `{}` | `{"threads": [{"id": int, "coro": int, "name": str, "state": str}]}` |
| `frames` | `{"thread": int}` | `{"frames": [{"id": int, "coro": int, "name": str, "pc": int, "src_ref": int, "depth": int}]}`, innermost first |
| `read_frame` | `{"frame": int}` | `{"locals": [Var], "args": [int], "this": Comp\|null, "regs"?: {str: int}}` |
| `read_comp` | `{"comp": int}` or `{"path": str}` | `{"comp": int, "path": str, "fields": [Var]}` |
| `read_heap` | `{"handle": int, "kind"?: "string"}` | `{"kind": str, "items": [Cell], "text"?: str}` |
| `backdoor_read` | `{"addr": int, "width": 8\|16\|32\|64}` | `{"value": int}` |
| `exec_snippet` | `{"frame": int, "code": [Ins], "nregs": int}` | `{"value": int\|null}` |
| `reload` | `{"image": str}` | `unsupported` in v1 (hot reload, P3.6) |

```
Var  = {"name": str, "value": int, "width": int, "signed": bool, "kind": str}
Comp = {"comp": int, "path": str}
Cell = {"value": int, "kind": str, "text"?: str}   (a map's also key_value, key_kind, key_text?)
Ins  = {"op": str, "args": [int], "imm": int, "flags": int}
```

- `frames[].pc` is the pc of the next instruction to execute in that frame
  (for a caller, the instruction after its `CALL`); `src_ref` is the
  statement containing it (0 when there are no markers).
- `read_frame.locals` are the PSS locals in scope at the frame's pc
  (`CoroDescriptor.locals`, `[pc_start, pc_end)`); with no debug lowering
  the list is empty and `regs` holds the raw register file. `args` are the
  called function's arguments; `this` is the frame's component instance.
- `read_comp` takes an index into `model.components.paths`, or that path;
  `fields` are the instance's fields, each with its width, signedness and
  kind from the image's slot types (LAYOUT): `int`, `bool`, `enum`, ... or
  `list`/`map`/`set`/`string`, whose value is a heap handle (`read_heap`).
- `read_heap` reads heap object `handle` (a `Var` of kind `list`, `map`,
  `set` or `string` holds one; heap.md): `kind` is `list`/`map`/`set`/
  `string`, or `empty` for handle 0. `items` are the elements in order (a
  map's in insertion order, each with its key); an item's `kind` is `int`,
  `string` (with its `text`), a nested collection's kind, or `empty`. With
  `kind: "string"` the handle is any string value, a model-string index too.
  An unknown handle is `bad_handle`. The C engine has no heap (it refuses
  heap images), so it has no `read_heap`.
- `backdoor_read` reads the platform's memory or register mirror **without**
  a bus access: no `access` event, no watch, no bus-log entry, no read side
  effect (a read-to-clear field stays set).
  pyvm calls the platform memory's `peek(addr, n)` when it has one, else
  reads `extern.Memory`'s bytes.
- `exec_snippet` runs `code` (procedural ops only: arithmetic, logic,
  moves, loads of frame slots and component fields, branches, `RET`) on a
  fresh register file of `nregs` registers, reading the given frame's
  locals and component; a memory read goes through the back door. It may not
  write frame slots, fields or memory; an orchestration op or a write is a
  `snippet` error. The value is the `RET` operand (null for a bare `RET`).
  The front end compiles PSS expressions to it (P2.2).

## Events

| `event` | `body` |
|---|---|
| `stopped` | `{"reason": str, "thread": int, "frame": int, "coro": int, "pc": int, "src_ref": int, "hits"?: [int], "detail": {}}` |
| `exited` | `{"status": "ok"\|"error"\|"terminated", "message"?: str}` |

Stop reasons and their `detail`:

| `reason` | When | `detail` |
|---|---|---|
| `entry` | never sent: the engine starts stopped | |
| `pause` | `stop` | `{}` |
| `step` | a `step` finished | `{"kind": "stmt"\|"into"\|"out"}` |
| `breakpoint` | a pc break; `hits` = the break ids | `{}` |
| `watch` | an access watch; `hits` = the watch ids | the `access` event's `{rw, width, addr, data}` (+ `reg_id` when known) |
| `check` | a failing check (`set_stop_on` `check`) | the `check` event's detail |
| `sanitizer` | a failing sanitizer | the `sanitizer` event's detail |
| `error` | `error()`/`fatal()`, or an engine error | `{"severity", "text"}`, or `{"error": message}` |
| `unimplemented` | an import with no provider was called | `{"name", "fn_id", "args"}` |
| `mocked` | a call a double answered | the `import` / `call` event's detail |

`thread`, `frame`, `coro`, `pc` and `src_ref` locate the stop: the frame
that was executing, the pc of its next instruction, and the statement (or,
for a watch or a stop event, the instruction's `src_ref`). When an engine
error stops the run, inspection still works; any resume then ends the run
(`exited` with `status: "error"`).

## Versioning

`attach` carries the protocol version. A new command, argument, reply field,
reason or capability is a compatible change; the front end ignores fields it
does not know and checks `capabilities` before using optional commands.
Renaming or removing anything is version 2.
