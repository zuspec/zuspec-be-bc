# Engine host ABI

Status: **pssc P3.1a (spec; pyvm seam P3.1b)** · Version `PSSVM_ABI_VERSION = 1`
· Authority: this file, then `zuspec.be.bc.interp` (pyvm) and `zuspec-rt-eng`
(native) · Companion to the pssc design (`pss-test-debug-env-design.md` §8.4,
§10.3). Values: [value-abi.md](value-abi.md). Images: [zbc-format.md](zbc-format.md).
Debugging: [debug-stub.md](debug-stub.md).

A host (a SystemVerilog testbench through DPI, a C test, or Python) runs an
image through this ABI. The engine never owns a thread or a clock: whenever
it needs the outside world (a bus access, an import, time, an event) it
**yields** a request and returns to the host, which does the work in its own
time and then completes the request. A time-consuming access is therefore an
ordinary SV task on the host side; no DPI call ever blocks and no callback
crosses the boundary.

The ABI is plain C so that DPI can call it directly. pyvm implements the same
calls in Python (P3.1b), and its synchronous API (`VM.run`, pssc
`bc_session`) is a loop over them (P3.1c). Under pyhdl-if the SV package
`pssvm_pkg` calls the same names through a shim, so **switching engines
changes no SV**.

## Handles

| Handle | What | Lifetime |
|---|---|---|
| `pssvm_t` | an engine: one loaded image (one generation at a time), its component tree, its scheduler and time | `pssvm_load` … `pssvm_free` |
| `pssvm_call_t` | one call into the image: an entry (action or component function) and every thread it spawns | `pssvm_call_start` … `pssvm_call_end` |

All calls on one `pssvm_t` come from one host thread (DPI is single-threaded).
Several calls may be live at once; each is driven by its own SV thread and
they share the model (§10.3: parallel operations are SV threads). Within a
call the engine's own scheduler interleaves the call's threads exactly as
pyvm does today (FIFO ready order, [determinism.md](determinism.md)).

## Calls

```c
int           pssvm_abi_version(void);                     /* 1 */
pssvm_t      *pssvm_load(const uint8_t *img, size_t len, const char *opts);
int           pssvm_construct(pssvm_t *e, const char *root_type,
                              const uint8_t *args, size_t len);
int           pssvm_fresh(pssvm_t *e, const char *root_type);
pssvm_call_t *pssvm_call_start(pssvm_t *e, const char *entry, const char *comp_path,
                               const uint8_t *args, size_t len, uint64_t seed);
pssvm_call_t *pssvm_call_start_obj(pssvm_t *e, const char *entry, const char *comp_path,
                                   const uint8_t *args, size_t len, uint64_t seed,
                                   uint64_t *slots, uint32_t n_slots);
int           pssvm_run(pssvm_call_t *h);                  /* → status below */
const pssvm_req_t *pssvm_request(pssvm_call_t *h);         /* the pending request */
int           pssvm_complete(pssvm_call_t *h, const uint8_t *data, size_t len);
size_t        pssvm_result(pssvm_call_t *h, uint8_t *buf, size_t cap);
void          pssvm_call_end(pssvm_call_t *h);
int           pssvm_notify(pssvm_t *e, uint32_t kind, uint32_t id, uint64_t value);
int           pssvm_set_irq(pssvm_t *e, const char *entry); /* the IRQ handler */
int           pssvm_reload(pssvm_t *e, const uint8_t *img, size_t len, uint32_t flags);
const char   *pssvm_error(pssvm_t *e, pssvm_call_t *h);    /* JSON, or NULL */
void          pssvm_free(pssvm_t *e);
```

- **`pssvm_load`** checks the header and integrity ([zbc-format.md](zbc-format.md)
  §Integrity) and reads `IDENT` (`api_hash`, `generation`) and `ENTRY`. `opts` is
  `key=value` pairs separated by `;` (`events=0x..`, `stub=1`; rt-eng reads
  `verbosity=N`: the highest `message()` level a run reports when no host is set), or NULL. NULL on
  failure; `pssvm_error(NULL, NULL)` says why.
- **`pssvm_construct`** builds the component tree under `root_type` (NULL: the
  image's root) and runs each `initialize()` with the constructor arguments
  (value ABI, one leaf per parameter as in `ENTRY.sig`). Construction must not
  yield: an access, import or wait there is an error (`construct-yield`); bind
  handles only. Returns 0, or -1 with `pssvm_error`.
- **`pssvm_fresh`** drops the component tree and constructs it again (or for
  the first time): a clean model between tests. No call may be running.
  The C engine also ships `pssvm_dpi.c`, these calls in the shapes SV DPI-C
  imports take (chandles, strings, the request field by field, call
  arguments staged one at a time); pssc's `pssvm_pkg.sv` imports it.
- **`pssvm_call_start`** looks `entry` up in `ENTRY` by name (`dev_c::put`, or an
  action's coroutine name), runs it in the instance at `comp_path` (`""` = the
  root; must be of `comp_type` or a type derived from it), with `args` encoded by
  the value ABI in `sig` order, and `seed` for its solves. It does not run
  anything: the first `pssvm_run` does. **`pssvm_call_start_obj`** also gives
  an action entry its action object: `n_slots` fields (the action's layout,
  be-bc `Model.obj_layouts`) in `slots`, which the host owns and the call reads
  and writes (`LD_FIELD`/`ST_FIELD`, a solve's write-back) until `pssvm_call_end`.
- **`pssvm_run`** runs the call's threads until the call needs the host, then
  returns a status. It is the only call that executes instructions.
- **`pssvm_complete`** answers the pending request. `data` is the read value or
  the import's result (value ABI; `len` 0 when there is none). The next
  `pssvm_run` resumes the thread that asked, at the instruction after the yield.
  Completing when nothing is pending, or with the wrong size, is
  `PSSVM_E_STATE`.
- **`pssvm_result`** copies the entry's return value (value ABI) after DONE;
  returns its size (0: none). **`pssvm_call_end`** frees the call; ending a call
  that is not DONE cancels its threads (as a cancelled branch: they never run
  again).

## Run status and requests

`pssvm_run` returns one of:

| Status | Value | Pending request | The host then |
|---|---|---|---|
| `PSSVM_DONE` | 0 | none | `pssvm_result`, `pssvm_call_end` |
| `PSSVM_ACCESS` | 1 | `kind` r/w, `addr`, `size` (bytes), `data` (write), `reg_id`, `src_ref` | does the bus access (consumes sim time), `pssvm_complete(h, read data or none)` |
| `PSSVM_IMPORT` | 2 | `fn_id`, `name`, `args` (value ABI), `blocking` | calls the import (a blocking one may consume time), `pssvm_complete(h, ret)` |
| `PSSVM_WAIT` | 3 | `on` = `TIME` with `delay`, or `EVENT` | `TIME`: waits `delay` engine units, then `pssvm_run`; `EVENT`: waits for something it will `pssvm_notify`, then `pssvm_run` |
| `PSSVM_STOP` | 4 | `src_ref`; `pssvm_dbg_stop` gives reason, thread, frame, coro, pc, depth, breakpoint ids | serves the debug stub until it resumes (`pssvm_dbg_resume`: continue or step), then `pssvm_run` (run again unresumed = continue) |
| `PSSVM_ERROR` | -1 | none | `pssvm_error(e, h)`; the call is over (`pssvm_call_end`) |

`pssvm_req_t` is a plain struct (`status`, the fields above, with `args` a
pointer and length into engine memory valid until the next call on `h`).
Accesses of 1, 2, 4 and 8 bytes carry `data` as a little-endian `uint64_t`;
this matches pyvm's `Memory.read(addr, n)` / `write(addr, n, value)`.

Rules:

- **One request per call at a time.** A call that yields is suspended whole;
  its other threads do not run until it is completed and run again. (Two
  operations that overlap on the bus are two calls.) A later version may add
  request ids for overlapping requests within one call.
- **Every host-visible effect yields.** Non-blocking imports (`target`
  functions, solve-time functions) yield `IMPORT` too, with `blocking = 0`: the
  host must complete them without consuming time. Builtin imports that do not
  touch the outside (`print`, `error`, string formatting) do not yield.
- **WAIT TIME** is returned when no thread of the call is ready and the earliest
  is on the timed queue: `delay` is that time minus the engine's `now`. On the
  next `pssvm_run` the engine sets `now` to that time. A host with its own clock
  maps units to sim time; a host without one (pyvm's sync loop) runs at once,
  which is today's virtual time.
- **WAIT EVENT** is returned when every thread of the call waits on something
  only the host can change (a channel token an interrupt handler puts, a
  `wake`). Today pyvm raises `deadlock` there; with a host that can
  `pssvm_notify`, it waits instead. The sync loop keeps the `deadlock` error
  when the host has nothing to notify.
- **Errors are located failures.** `pssvm_error` returns
  `{"message", "src_ref", "file", "line", "col", "call", "thread"}` (the
  `VMError` fields, stamped at the innermost frame). An engine error never takes
  the host down; the engine stays usable for other calls.

## Notify

`pssvm_notify(e, kind, id, value)` tells the engine about a host event. It may be
called at any time between engine calls, from any SV thread; it takes effect
at the next `pssvm_run` of any call (an interrupt is taken between
instructions).

| `kind` | Meaning | pyvm today |
|---|---|---|
| `PSSVM_N_IRQ` (1) | interrupt `id` raised; starts the image's interrupt handler as a thread of its own in the root component | `Ctx.raise_irq(irq)` (pssc `testing/vm.py`) |
| `PSSVM_N_CHAN` (2) | put `value` on the host-fed channel `id` (by path id from the image) | — (P3.6a: channels by path) |

The interrupt handler is a function entry the host names once with
`pssvm_set_irq(e, entry)` (pyvm: `HostCall(irq_entry=coro)`, pssc
`Session.irq_entry`). The C engine queues notifications (at most
`PSSVM_NOTES_MAX`, 64; more is `PSSVM_E_STATE`) and the first `pssvm_run` of
any call takes them all: an IRQ starts one handler thread in that call. A kind
with no handler (an IRQ before `pssvm_set_irq`; `CHAN` until host-fed channels
exist) ends that call in `ERROR` ("notify kind K has no handler"), as pyvm's
`VMError` does.

Each call waiting on `EVENT` sees the notification on its next `pssvm_run`;
one that is still not runnable returns `WAIT EVENT` again.

Each notification taken raises a `notify` event (when the events sink watches
`EV_NOTIFY` in every coroutine), before its handler starts:
`{irq|chan: id, value, delivered}` (trace-schema.md "The platform side"), so
a recorded run says where each one arrived, on both engines.

## Reload

`pssvm_reload(e, img, len, flags)` loads the image as generation N+1 beside N.
It applies at the next **quiescent boundary**: no call live, or (flag
`PSSVM_R_AT_CALL`) before the next `pssvm_call_start`; calls already running
finish in N. It is refused (`PSSVM_E_API`) when `IDENT.api_hash` differs from
N's: a signature changed, and the SV facade would no longer match. Channel
tokens carry over by path. Details: [generations.md](generations.md).

Until the C engine does this alone (P4.9, a simulator without Python), a
Python host reloads it (P4.6a, rt-eng `zuspec.rt.eng.generations.reload_engine`):
the checks and migration of `interp/generations.py` on the models, a new engine
for N+1 constructed with `pssvm_construct` (a construct-yield refuses, reason
`bus`), and N's state copied in with `pssvm_comp_write(e, buf, n)`.

The C engine's own `pssvm_reload` (P4.9e1, rt-eng `pssvm.h`) applies at once,
in place (the host keeps its `pssvm_t`; call it with no call live), with flag
`PSSVM_R_STRICT` only. It refuses with `PSSVM_E_API`, the error message starting
with the reason (`api`, `structure`, `bus`, `construct`). So far it keeps state
only when the component tree's layout is unchanged; any layout change is
refused as `structure`. Migration by instance path comes in P4.9e2.
`pssvm_generation(e)` counts from 1.

## Return codes

`0` ok; `PSSVM_E_STATE` (-2) call made in the wrong state; `PSSVM_E_ARGS` (-3)
bad entry, path or argument size; `PSSVM_E_API` (-4) reload refused;
`PSSVM_E_IMAGE` (-5) bad image. Every failure sets `pssvm_error`.

## How pyvm maps onto it (P3.1b)

| Host ABI | pyvm today (synchronous) |
|---|---|
| `ACCESS` | `ops_orch._builtin` → `vm.memory.read/write` (`_MEM_READ`/`_MEM_WRITE`) |
| `IMPORT` | `ops_orch._op_import` → `vm.import_provider.call(fn_id, args)` |
| `WAIT TIME` | `VM._drain` → `sched.next_timed()` (virtual `now`) |
| `WAIT EVENT` | `VM._drain` → `VMError("deadlock: ...")` |
| `STOP` | the stub blocks the VM thread in `VM._run_frame_ev` (debug-stub.md) |
| `DONE` | `_drain` returns (`sched.has_work()` false) |
| `pssvm_construct` | `VM.construct_components` |
| `pssvm_call_start` | `VM.root_frame(coro, seed, obj)` |

In pyvm the seam is a Python object per call (`run()` returns a request object,
`complete(value)` answers it); the existing `Memory` and `ImportProvider`
become the sync loop's default answers, so a run with no host is unchanged
(T-G1).

That object is `interp/host.py` `HostCall` (`run`, `complete`, `notify`,
`result`, `end`); `run_sync(call)` is the sync loop. The call runs on a thread
of its own that blocks at each yield; the VM consults `vm.host` (None: the old
in-place answers). `STOP` is not mapped yet: the debug stub still blocks the
VM thread itself (P3.7).

`run_sync` answers each request from `vm.memory` / `vm.import_provider` as
they are at that request (a debugger swaps in a watching memory mid-run). A
platform callback that raises is handed back with `HostCall.fail(exc)`: the
exception is raised in the VM thread at the access or import, so the run
sees it exactly as it would from a direct call (the old path). The callbacks
run on the host's thread while the VM thread waits, so a callback that
readies a thread (pssc's `Ctx.raise_irq`) never races the VM.

`HostCall(..., construct=False)` runs a frame on the tree as it stands: a
function called on a tree built before, or a setup/teardown function between
runs. pssc's synchronous API (`pssc.testing.bc_session`: `run`,
`Session.call`, `Session.run_entry`, setup/teardown) runs every frame this
way, one call per frame, through `run_sync` (P3.1c): there is no path in
pssc that drains the VM without a host.

## Python host for the C engine (P4.3d)

zuspec-rt-eng implements this ABI in C (`pssvm.h`) and drives it from Python
with `zuspec.rt.eng.host`, in the same shape as pyvm's `HostCall`:

| pyvm (`interp/host.py`) | C engine (`zuspec.rt.eng.host`) |
|---|---|
| `VM(model, ...)` + `vm.root_frame(...)` | `Engine(lib, model.to_bytes())` (`pssvm_load`) + `.call(entry, args, seed=...)` |
| `HostCall.run/complete/fail/result/end` | `NativeCall.run/complete/fail/result/end` |
| `run_sync(call)` | `run_sync(call, memory, import_provider)` |
| `vm.cobj` | `Engine.comp_slots()` (`pssvm_comp_read`, interim until P4.4) |

So one Python `Memory` / `ImportProvider` mock answers both engines.
Differences: a mock's exception ends the C call with `ERROR` and `run_sync`
re-raises it (C cannot raise inside the model); an engine error is a
`NativeError` carrying the error JSON; `WAIT TIME` runs on at once (no clock);
`WAIT EVENT` there is a `deadlock` `NativeError`; a host loop of its own
serves it with `Engine.set_irq(entry)` and `NativeCall.notify(N_IRQ, id)`
(P4.4e). pssc runs a test on it
with `Session(engine="c")` or `PSSC_ENGINE=c` (`pssc.testing.native`).

The engine refers to libdv_solve weakly: a library built without it loads,
and a `SOLVE` then fails (`ZBC_ERR_SOLVE_FAIL`). A host links libdv_solve only
for an image that solves; rt-eng's `build_shared_lib` puts
`-Wl,--no-as-needed` before the link arguments so the weak-only reference
still records the dependency.

## Versioning

`pssvm_abi_version()` returns 1. A new status, request field or notify kind
adds a minor feature the host checks for; a change to an existing call's
meaning raises the version. `pssvm_pkg` refuses an engine whose version it does
not know.

## Debug stops in the C engine (P4.5a)

`pssvm_dbg_enable(e, on)` turns on a check before every instruction of later
calls (off: none, at no cost). `pssvm_dbg_set_breaks(e, id_coro_pc, n, verified)`
replaces the breakpoints with (id, coro, pc) triples, pc in the coroutine;
`verified[i]` says whether triple i names an instruction. `pssvm_dbg_pause(e)`
stops the next instruction of any thread. A step (`pssvm_dbg_resume(h, kind)`,
kind 1 statement, 2 into, 3 out) stops the stopped thread at its next
statement start (LINE table) at the same depth or above, any depth, or above;
when that thread ends, any thread's next statement. A stop blocks its thread
before the instruction and the run returns `PSSVM_STOP` with no thread
running (all-stop); a resumed thread does not stop again at that pc, nor does
a thread re-entering where it already passed the check (an answered IMPORT, a
wait). `pssvm_dbg_stop(h)` (`pssvm_stop_t`, zbc_interp.h `zbc_dbg_stop_t`)
says where: reason (1 pause, 2 step, 3 breakpoint, 4 watch, 5 error), thread and frame ids
(a thread's id is its bottom frame's), coro, pc, src_ref, depth, hit ids.
Python: rt-eng host.py `Engine.dbg_*`, `NativeCall.dbg_stop`/`dbg_resume`.

While stopped, the host can inspect the run (P4.5b): `pssvm_dbg_threads(h, ids,
cap)` gives the live threads' ids, ascending; `pssvm_dbg_frames(h, thread, out,
cap)` a thread's frames innermost first; `pssvm_dbg_frame(h, frame, out)` one
frame. A frame (`pssvm_frame_t`, zbc_interp.h `zbc_dbg_frame_t`) has its id,
thread id, coro, pc (a caller's is past its CALL), depth on the thread, CALL
depth, its thread's state (0 running = the stopped one, 1 ready, 2 waiting),
object base, and pointers to its registers (64), locals (32) and call
arguments (16), valid until the run goes on. The counts are full counts (up
to `cap` filled) and 0 when the call is not stopped. Names (the LOCALS
table, component paths) are the image's, read host-side; component fields
are `pssvm_comp_read`; memory is the host's, so a back-door read is too.
Python: `NativeCall.dbg_threads`/`dbg_frames`/`dbg_read_frame`.

Access watches and error stops (P4.5b2): `pssvm_dbg_set_watches(e, w, n)`
replaces the watches (`pssvm_watch_t`: id, rw mask 1 read / 2 write, addr,
size > 0, and an optional value compared under mask). A bus access that
overlaps a watch, once done (a read has its value), stops the run before the
next instruction: reason 4 (watch), the hit ids, the access's src_ref, and
the access in the stop record (is_write, size in bytes, addr, data).
`pssvm_dbg_set_stop_on(e, PSSVM_STOP_ON_ERROR)` (default off) makes the run's
first error stop where it happened, reason 5 (error) with status, halted op
and message, frames still live; any resume ends the call with that error.
Python: `Engine.dbg_set_watches` (the stub's `set_access_watch` dicts),
`Engine.dbg_set_stop_on`, `NativeCall.dbg_stop()["detail"]`.
