# Engine host ABI

Status: **pssc P3.1a (spec; pyvm seam P3.1b); request model revised in pssc
full-PSS plan F5.1** · Version `PSSVM_ABI_VERSION = 2` (engines move to it in
F5.2 (pyvm) and F5.2b (C); until then they return 1 and keep one request per
call, `ACCESS`/`IMPORT` statuses, and `pssvm_request`) · Authority: this file, then `zuspec.be.bc.interp` (pyvm) and `zuspec-rt-eng`
(native) · Companion to the pssc design (`pss-test-debug-env-design.md` §8.4,
§10.3). Values: [value-abi.md](value-abi.md). Images: [zbc-format.md](zbc-format.md).
Debugging: [debug-stub.md](debug-stub.md).

A host (a SystemVerilog testbench through DPI, a C test, or Python) runs an
image through this ABI. The engine never owns a thread or a clock: whenever
it needs the outside world (a memory access, an import, time, an event) it
**yields** a request and returns to the host, which does the work in its own
time and then completes the request. A time-consuming access is therefore an
ordinary SV task on the host side; no DPI call ever blocks and no callback
crosses the boundary.

The ABI is plain C so that DPI can call it directly. pyvm implements the same
calls in Python (P3.1b), and its synchronous API (`VM.run`, pssc
`bc_session`) is a loop over them (P3.1c). In SV both run behind one
package, `zsp_rt_bc_pkg` ([SystemVerilog](#systemverilog-zsp_rt_bc_pkg)): the
variant a build compiles chooses the engine, so **switching engines changes
no SV**.

## Handles

| Handle | What | Lifetime |
|---|---|---|
| `pssvm_t` | an engine: one loaded image (one generation at a time), its component tree, its scheduler and time | `pssvm_load` … `pssvm_free` |
| `pssvm_call_t` | one call into the image: an entry (action or component function) and every thread it spawns; also the call's completion object ([Completion](#completion)) | `pssvm_call_start` / `pssvm_call` … `pssvm_call_end` |
| `pssvm_obj_t` | an object the host names: the root component (its tree), an instance under it, an action object ([Objects](#objects)) | `pssvm_new` / `pssvm_sub` … `pssvm_release` |

All calls on one `pssvm_t` come from one host thread (DPI is single-threaded).
Several calls may be live at once; each is driven by its own SV thread and
they share the model (§10.3: parallel operations are SV threads). At most
one of them is an action traversal; the rest are function calls
([Concurrent calls](#concurrent-calls)). Within a
call the engine's own scheduler interleaves the call's threads exactly as
pyvm does today (FIFO ready order, [determinism.md](determinism.md)).

## Calls

```c
int           pssvm_abi_version(void);                     /* 2 */
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
const pssvm_req_t *pssvm_take(pssvm_call_t *h);            /* next new request, or NULL */
int           pssvm_complete(pssvm_call_t *h, uint32_t req_id,
                             const uint8_t *data, size_t len);
int           pssvm_set_time(pssvm_t *e, uint64_t now);    /* host clock, in ticks */
size_t        pssvm_result(pssvm_call_t *h, uint8_t *buf, size_t cap);
int           pssvm_call_field(pssvm_call_t *h, const char *name, uint64_t *out);
void          pssvm_call_end(pssvm_call_t *h);
int           pssvm_notify(pssvm_t *e, uint32_t kind, uint32_t id, uint64_t value);
int           pssvm_set_irq(pssvm_t *e, const char *entry); /* the IRQ handler */
int           pssvm_reload(pssvm_t *e, const uint8_t *img, size_t len, uint32_t flags);
const char   *pssvm_error(pssvm_t *e, pssvm_call_t *h);    /* JSON, or NULL */
void          pssvm_free(pssvm_t *e);

/* step 6 (pssc pssvm-plan): pyvm implements these; the C engine does not
 * yet (below, and "Linking" to "Completion") */
pssvm_obj_t   pssvm_new(pssvm_t *e, const char *type, pssvm_obj_t parent,
                        const uint8_t *args, size_t len, uint64_t seed);
pssvm_obj_t   pssvm_sub(pssvm_t *e, pssvm_obj_t parent, const char *path);
int           pssvm_release(pssvm_t *e, pssvm_obj_t obj);
pssvm_call_t *pssvm_call(pssvm_t *e, pssvm_obj_t obj, const char *method,
                         const uint8_t *args, size_t len, uint64_t seed);
int           pssvm_obj_field(pssvm_t *e, pssvm_obj_t obj, const char *name,
                              uint64_t *out);
int           pssvm_link(pssvm_t *e, int32_t api, int64_t ctx,
                         pssvm_handler_fn fn, void *ud);
int           pssvm_link_check(pssvm_t *e);                /* names what is missing */
int           pssvm_add_listener(pssvm_t *e, pssvm_listener_fn fn, void *ud);
int           pssvm_step(pssvm_t *e, uint32_t budget);    /* slices run */
uint32_t      pssvm_pending(pssvm_t *e, uint64_t *next_time);  /* PSSVM_PEND_* */
int           pssvm_done(pssvm_call_t *h);                 /* 1: DONE or ERROR */
int           pssvm_status(pssvm_call_t *h);               /* DONE / ERROR */
void          pssvm_on_done(pssvm_call_t *h, pssvm_done_fn fn, void *ud);
```

(Step 7 renames the C calls `zbrt_*`; the shapes stay.)

- **`pssvm_load`** checks the header and integrity ([zbc-format.md](zbc-format.md)
  §Integrity) and reads `IDENT` (`api_hash`, `generation`) and `ENTRY`. `opts` is
  `key=value` pairs separated by `;` (`events=0x..`, `stub=1`; rt-eng reads
  `verbosity=N`: the highest `message()` level a run reports when no host is set), or NULL. NULL on
  failure; `pssvm_error(NULL, NULL)` says why.
- **`pssvm_construct`** builds the component tree under `root_type` (NULL: the
  image's root) and runs each `initialize()` with the constructor arguments
  (value ABI, one leaf per parameter as in `ENTRY.sig`). Construction must not
  yield: an access, a blocking import or a wait there is an error
  (`construct-yield`); a non-blocking import is called through its linked
  handler ([Construction](#construction)); bind handles only. Returns 0, or -1 with `pssvm_error`.
- **`pssvm_fresh`** drops the component tree and constructs it again (or for
  the first time): a clean model between tests. No call may be running.
  The C engine also ships `pssvm_dpi.c`, these calls in the shapes SV DPI-C
  imports take (chandles, strings, the request field by field, call
  arguments staged one at a time); rt-eng's `zsp_rt_bc_pkg.sv` imports it.
- **`pssvm_call_start`** looks `entry` up in `ENTRY` by name (`dev_c::put`, or an
  action's coroutine name), runs it in the instance at `comp_path` (`""` = the
  root; must be of `comp_type` or a type derived from it), with `args` encoded by
  the value ABI in `sig` order, and `seed` for its solves. It does not run
  anything: the first `pssvm_run` does. **`pssvm_call_start_obj`** also gives
  an action entry its action object: `n_slots` fields (the action's layout,
  be-bc `Model.obj_layouts`) in `slots`, which the host owns and the call reads
  and writes (`LD_FIELD`/`ST_FIELD`, a solve's write-back) until `pssvm_call_end`.
  The entry name `$run_end` is reserved: it is no `ENTRY` but ends the run,
  running the tree's `exec run_end` blocks once (components.md) after the last
  action a simulation calls. It does nothing when the image has no run hooks or
  no action has run; the C engine, which refuses images with hooks, treats it
  as a call that returns at once.
- **`pssvm_run`** runs the call's threads until the call needs the host, then
  returns a status. It is the only call that executes instructions.
- **`pssvm_complete`** answers outstanding request `req_id`. `data` is the read
  value or the function's result (value ABI; `len` 0 when there is none). The
  next `pssvm_run` resumes the thread that asked, at the instruction after the
  yield. Completing an unknown or dropped `req_id`, or with the wrong size, is
  `PSSVM_E_STATE`.
- **`pssvm_call_field`** (F5.6): after DONE of a call on an action entry, and
  before `pssvm_call_end`, field `name` of its action object (the entry's `obj`
  layout names the fields; the field's index there is its slot) in `*out`.
  `PSSVM_E_STATE`: the call is not DONE; `PSSVM_E_ARGS`: no such field (or no
  object). DPI `pssvm_dpi_obj_field(h, name)`, py shim `Engine.obj_field(h,
  name)` and SV `engine.obj_field(h, name)` return the value, or 0 with the
  call's error set; a bench reads it in a `zsp_rt_bc_ctxt` subclass's
  `done(h, entry)` hook, which `call` runs at DONE before it ends the call.
- **`pssvm_result`** copies the entry's return value (value ABI) after DONE;
  returns its size (0: none). **`pssvm_call_end`** frees the call; ending a call
  that is not DONE cancels its threads (as a cancelled branch: they never run
  again).

## The import interface

Everything the engine needs from outside goes through the **import
interface**, which carries calls to **APIs** the host implements. An API is a
named set of functions; the image lists them (`API` section, below) and a
request names the API and the function. Built-in APIs:

| API (id) | Functions | Blocking | What |
|---|---|---|---|
| `mem` (0) | `read`, `write` | yes | memory and register access: `addr`, `size`, `data`, `reg_id` |
| `config` (1) | `get` | no | a solve-time configuration value by name (reserved: M6/M8) |
| `seq` (2) | `start` | yes | start a named host sequence with arguments (reserved) |
| user APIs (≥ 16) | the PSS `import` functions, grouped by the package or component that declares them | per function | `args` in the value ABI |

A **blocking** function may consume host time; a **non-blocking** one
(`target` functions, solve-time functions) must be completed without
consuming time, but still yields: every host-visible effect yields. Builtins
that touch nothing outside (`print`, `error`, string formatting) do not
yield. The std_pkg file functions (`file_open`, `file_close`, `file_exists`,
`file_write`; LRM 24.2) are not an API either: the engine serves them itself,
on the machine it runs on, with names relative to its working directory, and
they never yield. A handle is 1-based in the engine's table (0 is
`nullfilehandle`, also what a failed open returns); a write to a closed or
read-only handle stops the run with a run-time error. `file_read(h, size)`
(at most `size` characters from where the last read stopped; -1: to the end),
`file_read_lines(name)` and `file_write_lines(name, lines, option)` are served
the same way (builtin ids +19..+21); what they read comes back as new heap
strings and lists ([heap.md](heap.md)), so the C engine refuses an image that
calls them, as one with heap ops, until it has a heap.
`urandom()` and `urandom_range(min, max)` (LRM 24.4) are served the same way:
one draw from the calling frame's seed stream (`urandom`: the draw's upper 32
bits; `urandom_range`: `min + draw % (max - min + 1)`, inclusive, min > max
swapped), so a seed gives the same numbers on every engine. `mem` requests keep their fixed fields (`addr`, `size`, `data` as a
little-endian `uint64_t` for 1, 2, 4 and 8 bytes, `reg_id`) so the hot path
decodes no value ABI; this matches pyvm's `Memory.read(addr, n)` /
`write(addr, n, value)`. Reserved ids are listed so hosts can refuse them
clearly until they exist.

**Import arity (F6.5b).** The `IMPORT` op carries `fn_id`, the result
register and up to two argument registers inline (`arg2`, `arg3`). A call
with more arguments (at most 16, `CALL_MAX_ARGS`) evaluates them all, stages
each with `ARG` as for `CALL` (spec/calls.md), and emits `IMPORT fn_id, ret`
with `imm` = the argument count; the engine reads the staged values and
consumes them when the call completes (a parked or yielded `IMPORT` runs
again with them still staged). Calls of 0–2 arguments keep the inline form,
so their images do not change. The request carries all the arguments the
same way either way (`args`, `args_len`; `pssvm_dpi_req_nargs` /
`pssvm_dpi_req_arg` on the SV side).

**Struct arguments (F6.4b).** An import parameter of a plain-data struct
type is passed as one argument per leaf field, nested structs flattened in
declaration order, each a scalar like any other; the arity limit counts the
leaves. The host gets copies: what it does with them does not change the
caller's struct. The image's `imports` metadata lists the leaves'
`(width, signed)` in `arg_types`, so a host sees only scalars, and the SV
side reads them with `pssvm_dpi_req_arg` as for any call.

**Struct results (F6.4b2).** An import returning a plain-data struct is
answered with one value per leaf, in the same order. The `IMPORT` carries
the leaf count in `imm` bits 8–15 (bits 0–7 stay the staged argument count)
and no result register; the engine stages the host's values as a called
function's results, and the caller reads them back with `LD_RES`
(spec/calls.md "Structs"). The request has `has_ret` = 1 and `nret` = the
leaf count; `pssvm_complete` takes `8 * nret` bytes, one little-endian u64
per leaf (the Python hosts' `complete(req_id, value)` takes a sequence). The
image's `imports` metadata lists the leaves' `(width, signed)` in
`ret_types`. Scalar imports are unchanged (`nret` = 1, or 0 when void).

On the SV side (`zsp_rt_bc_pkg`), `zsp_rt_bc_req.nret` is the field count when it is
more than 1 (else 0: a scalar, or a one-field struct answered through the
usual `ret`). The platform's `call_import` pushes the fields on
`req.rets`; `zsp_rt_bc_ctxt` stages each with the engine's `set_ret(h, rid, i,
v)` (DPI `pssvm_dpi_set_ret`, read as `pssvm_dpi_req_int(h, rid, "nret")`;
pyhdl-if `ZspRtBcPy.set_ret`) before `complete`, whose `data` a struct
result ignores. Fields are staged one call each because Verilator 5.052
cannot compile an open array in a DPI import.

**String arguments (F6.3a).** bc makes no string at run time: every string
value is an index into the image's model-string table (the message string
table's first `n_model_strings` entries). A string argument of an import
crosses as that index, a scalar like any other; the lowered model's
`imports` (`Model.imports`, in the API hash, and in the image's `IMPORT` rows) lists
its position in `string_at`, and the host turns it into text
with `pssvm_string(e, idx)` (NULL out of range; valid while `e` is). The
Python hosts have `HostCall.string(idx)` (pyvm: `Model.strings[idx]`) and
`Engine.string(idx)` (C engine through ctypes), both None out of range. In
SV, `zsp_rt_bc::string_at(idx)` gives it ("" out of range): DPI
`pssvm_dpi_string(e, idx)` on the C engine, the shim's `string_at` on pyvm.

**Heap strings (F6.9a; heap.md).** With run-time strings a string value is
either that table index (bit 63 clear) or a heap string handle (bit 63 set;
value-abi.md §3.1). An argument crosses as the slot value either way, and the
same calls resolve both forms: `pssvm_string(e, v)`, `HostCall.string(v)`,
`string_at(v)`. The text of a heap string is valid until the request
completes; a host that keeps it copies it.

**String results (F6.3b2).** An import declared to return a `string`
answers with text, not a value: the host calls `pssvm_string_ret(e, req_id,
i, text, len)` (`i` = result leaf, 0 for a scalar) before `pssvm_complete` (whose `data` is then ignored); the
engine makes a heap string from the bytes and the caller's result slot gets
its handle. The Python hosts answer with `HostCall.ret_string(req_id, text, i)` (or
`complete(req_id, text)` with a `str`, a struct's leaf as a `str` in its
tuple; an `ImportProvider` returns the `str` itself); in SV, the platform's
`call_import` sets `zsp_rt_bc_req.ret_s[i]` (keyed by leaf, so an empty string
is still an answer) and `zsp_rt_bc_ctxt` stages each with
`zsp_rt_bc::string_ret` before `complete` (pyhdl-if
`ZspRtBcPy.string_ret`; the C engine's variant has none until the C engine
has a heap, M14, and keeps the base's refusal). The image's `imports` metadata marks
such a result `ret_type` = `[64, false, "string"]`. A string field of a
struct result is staged the same way, with its leaf index. The C engine has
no heap until M14, so it refuses images with string results like any other
heap op (heap.md section 9); pssc refuses such an image for engine `c`
before the run (`check_model` in `pssc.testing.native`).

**Chandle arguments and results (F6.7).** A `chandle` is an opaque 64-bit
value: it crosses an import, either way, as an unsigned u64 like any scalar
(also as a struct leaf), all 64 bits intact, and bc never looks inside it;
`null` is 0. On the SV side it is a `longint unsigned` in `req.args` / `ret`;
a platform that keeps a real SV `chandle` (a C pointer from its own DPI code)
maps it to and from that value itself.

**Float fields (F6.6f).** A `float32` or `float64` struct field crosses an
import, either way, as the IEEE binary64 bits of its value in the u64 slot
(value-abi.md §1.1; a `float32` too, already rounded to float32 precision).
On the SV side it is a `longint unsigned`: `$bitstoreal` gives the `real`,
`$realtobits` the bits of a result. A float parameter or result of an import
itself crosses the same way (F6.6f2): the argument slot holds its binary64
bits, and the host returns binary64 bits; the signature's `ret_type` is then
`[64, true, "float"]`. PSS 3.0 20.4.1 c lists no float type for an import;
pssc allows it (an extension, like the float fields above).

## Run status and requests

`pssvm_run(h)` runs every ready thread of the call until none is ready. A
thread that calls an API is suspended with a **request** added to the call's
outstanding set; the other threads keep running. A thread has at most one
request outstanding, so one thread's requests reach the host in program
order. An engine may choose to run fewer threads at once (the C engine keeps
at most one request outstanding per call until M14); a host must not depend
on either.

`pssvm_run` returns, in this order of precedence:

| Status | Value | When | The host then |
|---|---|---|---|
| `PSSVM_ERROR` | -1 | a thread failed; the call is over | `pssvm_error(e, h)`, `pssvm_call_end` |
| `PSSVM_STOP` | 4 | a debug stop (all-stop); `pssvm_dbg_stop` gives reason, thread, frame, coro, pc, depth, breakpoint ids | serves the debug stub until it resumes (`pssvm_dbg_resume`: continue or step), then `pssvm_run` (run again unresumed = continue) |
| `PSSVM_REQS` | 1 | at least one new request since the last run | `pssvm_take` until NULL; starts each (in SV, a process each); runs again when any completes |
| `PSSVM_DONE` | 0 | the entry returned and nothing is outstanding | `pssvm_result`, `pssvm_call_end` |
| `PSSVM_WAIT` `TIME` | 3 | nothing ready; a timer is the earliest wake (`delay` ticks) | waits `delay` ticks **or** for a completion or notify, whichever is first; `pssvm_set_time`; runs again |
| `PSSVM_WAIT` `REQS` | 3 | nothing ready, no timer, requests outstanding | waits for a completion, runs again |
| `PSSVM_WAIT` `EVENT` | 3 | nothing ready or outstanding; threads wait on the host (or on another call) | waits for something it will `pssvm_notify`, or for another call to run, then `pssvm_run` |
| `PSSVM_WAIT` `TURN` | 3 | a traversal not started: another traversal is active, or queued before it ([Concurrent calls](#concurrent-calls)); nothing ran | waits for a traversal to end (DONE or ERROR), then `pssvm_run` |

Value 2 (the old `IMPORT`) is unused.

`pssvm_req_t` is a plain struct:

| Field | What |
|---|---|
| `req_id` | unique among the call's outstanding requests (reused after completion) |
| `api`, `fn` | API id (`mem` 0, or the import's own API, its `IMPORT` row's `api`: 16 up; 16 when the image has no row) and function id (an import's `fn_id`); `name` the function's PSS name (NULL when the image has no `IMPORT` entry for it) |
| `blocking` | 1: may consume host time |
| `addr` `size` `data` `kind` `reg_id` | `mem` only: `kind` r/w, `size` in bytes, `data` the value written |
| `args` `args_len` `has_ret` `nret` `bound` | other APIs: `args` (value ABI), `has_ret` (the answer is `nret` u64s, else empty; `nret` > 1: a struct result), `bound` (`pssvm_bind`, below) |
| `thread` | engine thread id (the debug stub's thread id) |
| `executor` | index into the image's `EXEC` table; `0xFFFFFFFF`: none (the default) |
| `stim_id` | the action run that made the request (F2.8) |
| `src_ref` | the source location of the call |

`reg_id` and `src_ref` are `0xFFFFFFFF` when unknown. A request's pointers
stay valid until it is completed or the call ends.

Rules:

- **Out of order.** Outstanding requests may be completed in any order; each
  completion readies only its thread.
- **Order the host may rely on**: per thread, program order; per executor,
  issue order (the order `pssvm_take` gives). Across threads and executors,
  none. Whether an executor serves its requests one at a time (a FIFO) or
  overlaps them (a pipelined interface) is the host's choice; the engine sets
  no limit.
- **Determinism.** Threads become ready in completion order, then FIFO
  ([determinism.md](determinism.md)). The trace records each completion
  (`req_done {req, thread, t}`), so a run replays from its trace, and a
  scripted host that completes in a fixed order reproduces exactly.
- **Time.** `delay` is in ticks (`IDENT.time_exp`, below). A host with a clock
  calls `pssvm_set_time(e, now)` before each `pssvm_run` (never backwards:
  `PSSVM_E_STATE`); timers due by `now` fire. A host without one (pyvm's sync
  loop) never calls it; time then moves only by `WAIT TIME`, at once, which is
  virtual time.
- **WAIT EVENT** is returned when every thread of the call waits on something
  only the host can change (a channel token an interrupt handler puts, a
  `wake`). The sync loop raises `deadlock` there when the host has nothing to
  notify.
- **Cancellation.** A thread cancelled while its request is outstanding (a
  `select` or failed branch) leaves the request outstanding: the host is not
  asked to abort an API call (PSS has no abort; a later version may add a
  cancel notice the host may ignore). Its completion is accepted and
  discarded. `pssvm_call_end` drops outstanding requests; completing one
  afterwards is `PSSVM_E_STATE`.
- **Debug stops.** While stopped, completions are accepted and queued; the
  threads they ready run after resume. All-stop is unchanged.
- **Errors are located failures.** One thread's error ends the call;
  outstanding requests are dropped as in cancellation, and the host lets
  calls in flight finish. `pssvm_error` returns
  `{"message", "src_ref", "file", "line", "col", "call", "thread"}` (the
  `VMError` fields, stamped at the innermost frame). An engine error never
  takes the host down; the engine stays usable for other calls.
- **Several calls** run at once; each has its own outstanding set. One
  traversal at a time; see [Concurrent calls](#concurrent-calls).

## Hosts

- **SV (`zsp_rt_bc_pkg`).** One process per taken request, dispatched by `api`
  to the handler `link`ed for (`api`, `executor`), else the platform: `mem`
  to the agent or sequencer bound to the executor path, user APIs to the
  import class, `seq` to a sequence starter. An executor multitasks: its
  imports are not ordered, so a thread blocked in one (an interrupt wait)
  never holds up another thread's requests. Its memory accesses (`mem`),
  the inferred default executor's included, enter their handler in issue
  order; whether one finishes before the next starts is the handler's
  choice (a simple bus serializes, a pipelined one overlaps). Accesses with
  no executor (-1) are unordered.
  Requests of different calls or executors that reach a shared bus in one
  time step may reach it in either order.
- **Python sync loop (`run_sync`).** Takes every request and completes them in
  issue order, non-blocking ones first. For one thread this is the old order.
  Function calls started meanwhile (an interrupt handler a platform calls)
  run alongside and are driven to their ends. `run_calls(calls)` is the
  same loop over several calls: each round runs every call that can run,
  then serves their requests; when none can, the earliest timer goes on
  (what the host did meanwhile runs first), else `wait_event`, else a
  deadlock ([Concurrent calls](#concurrent-calls)).
- **Python host for the C engine** wraps the same calls (below).

## SystemVerilog: zsp_rt_bc_pkg

One package name, `zsp_rt_bc_pkg` ("runtime for bytecode"), in two source
variants; **the one a build compiles chooses the interpreter**, and user
code names neither (pssc design `pssvm-in-practice.md` §2.1; pssvm-plan
step 7):

| Variant | File | Engine | Links |
|---|---|---|---|
| pyvm | be-bc `share/sv/zsp_rt_bc_pkg.sv` (generated) | pyvm in the simulator's Python, through pyhdl-if and the shim `share/py/pssvm_dpi_shim.py` | `libpyhdl_if` |
| C | rt-eng `share/sv/zsp_rt_bc_pkg.sv` (hand-written) | the C engine through `pssvm_dpi.c` | `libpssvm` |

Both `` `include `` be-bc's `share/sv/zsp_rt_bc_common.svh` (so a build has
be-bc's `share/sv` on +incdir): the run status and request record
(`zsp_rt_bc_status_e`, `zsp_rt_bc_wait_on_e`, `zsp_rt_bc_req`), the
import-API base class `zsp_rt_bc_platform_if` (bus accesses, imports, time,
reset), the engine interface `zsp_rt_bc_if`, the base class
`zsp_rt_bc_base`, and the run loop `zsp_rt_bc_ctxt`. Each variant defines
`class zsp_rt_bc extends zsp_rt_bc_base` and
`static function zsp_rt_bc load(string img, string opts = "")`, which
returns the engine (null, and why on stdout, on failure). There is no
process-wide default engine: whoever loads one passes it on.

**Load options** (`opts`, comma-separated `key=value`) carry what only one
implementation understands; an implementation ignores an option it does not
know and reports it (`PSSVM-WARNING: zsp_rt_bc::load: option 'k' is not
known to ...`). pyvm: `root=`, `platform=` (default `fake` under SV: the
register fake keeps what the bus returned, for back-door checks), `seed=`.
The C engine: `root=`, `seed=`, `seed_mode=`, `trace=`, `trace_mask=`,
`verbosity=`, `step_limit=`. `+pss_seed` and `+pss_seed_mode` are put in
front, so they win.

**One method list.** Every method of `zsp_rt_bc_if` comes from one table,
be-bc `share/rt_bc_api.py`, which generates the interface
(`zsp_rt_bc_if.svh`), `zsp_rt_bc_base`'s default for each engine method
(`zsp_rt_bc_defaults.svh`), the pyvm variant (a forwarder per method over
the pyhdl-if class `ZspRtBcPy`, itself built by the shim's `make_api` from
the same table). A method a variant does not serve is the base's
**refusal**: it returns its failure value, `error(0)` says so, and the
first refusal of each method is printed (`PSSVM-ERROR: zsp_rt_bc::step is
not served by the C engine (zuspec-rt-eng)`); it is never a missing symbol.
A few older queries instead have a documented answer for an engine that
cannot tell (`api_check` "", `progress` -1, `fatal` ""). Regenerate with
`python -m zuspec.be.bc.share.rt_bc_api`; be-bc's `test_rt_bc_api.py` and
rt-eng's `test_zsp_rt_bc_sv.py` check the files and both variants'
signatures against the table.

| Method | Host ABI | C variant |
|---|---|---|
| `call_start(entry, comp_path, args)`, `run`, `take`, `request`, `complete`, `set_ret`, `result`, `obj_field`, `call_end`, `notify`, `error` | [Calls](#calls), [Run status](#run-status-and-requests) | served |
| `string_ret` | String results | refused (no heap) |
| `new_obj(type, parent)` (`new`: `parent` 0 for the root), `sub(parent, path)`, `call_obj(obj, method, args)`, `release_obj(obj)`, `field(obj, name)` | [Objects](#objects) | refused |
| `api_id(name)`, `missing_links(linked)` | [Linking](#linking) | refused |
| `link(api, ctx, handler)`, `link_check()`, `handler(api, executor)`, `add_listener(l)` | [Linking](#linking), [Listeners](#listeners) | in `zsp_rt_bc_base` (SV) for both; `link_check` asks `missing_links` |
| `step(budget)`, `pending()`, `next_time()` | [step and pending](#driving-every-call-step-and-pending) | refused |
| `fatal()`, `progress()` | [Listeners](#listeners), [Concurrent calls](#concurrent-calls) | "" and -1 |
| `call_inside(h, outer, rid)` | self-wait | refused |

(`release` is an SV keyword, hence `release_obj`; `new` is the
constructor, hence `new_obj`.)

**Links are SV objects.** `link(api, ctx, handler)` keeps `handler` (a
`zsp_rt_bc_platform_if`) for API `api` (`"mem"`, an import API's name, or
`"*"`) and executor `ctx` (an instance path, `""` for none, `"*"`), with
the lookup order of [Linking](#linking). The run loop serves each request
with its handler, else with the context's platform. The engine is not told:
every import, blocking or not, still reaches SV as a request (a pyvm
handler is called synchronously; an SV one cannot be). `link_check()` gives
the engine the linked keys (`missing_links`, "`api:ctx;`" with -1 for any and
-2 for no executor) and fails with what is missing in `error(0)`.

**Listeners and the fatal.** `add_listener(l)` registers a
`zsp_rt_bc_listener` (`on_error`, `on_fatal`, functions taking a
`zsp_rt_bc_event`: `sev`, `msg`, `call_id`, `entry`). When a call ends in
ERROR, the run loop asks `fatal()`: none, and the listeners hear the
call's error; else they hear the fatal, once however many calls it ends
(the calls it released have no event of their own). **With no listener a
fatal ends the simulation** (`$fatal(1, "PSSVM FATAL: ...")`); a listener
replaces that default. pyvm's shim no longer prints the fatal itself.

**Waking a waiting call.** A call waiting on EVENT or TURN runs again when
another call ends, when the testbench notifies, and when `progress()` has
moved since the run loop last looked: the run loop asks after every `run`,
so a function call that puts on a channel and then blocks on an import
wakes the traversal waiting in `get()` at the put, not at its own end.
pyvm counts a run that ends parked on a request as progress (`VM.progress`).

**Self-wait.** The run loop records the request each serving process is
inside. A call started from that process (an import task calling
`ctxt.call`) is named to the engine with `call_inside`, so a traversal that
would queue behind the traversal whose request it serves ends in ERROR
(`self-wait: ...`), as in the [sync loops](#concurrent-calls), instead of
waiting for ever. A call started from a process the import task forked is
not recognised.

**The PSS face** (`pssvm_pkg`, be-bc `share/sv`, moving to pssc) imports
`zsp_rt_bc_pkg`: `pssvm_report`/`pssvm_bench_fail` (to every engine
loaded, `zsp_rt_bc_engines`), `pssvm_dbg_wait`, `pssvm_runner`, module
`pssvm_dbg`. The C ABI keeps its `pssvm_*` names (`pssvm.h`,
`libpssvm.so`, `pssvm_dpi_*`) until the C engine is picked up again.

## Image sections for hosts

- `API` (35; codegen/debug profiles): `{id, name, kind, n_fns}` per API, the
  built-in `mem` (0), `config` (1) and `seq` (2) always, then one user API per
  scope declaring imports (16 up). `IMPORT` rows carry `api`, `flags`
  (blocking, target, solve), the qualified name and the signature (`sig`:
  argument and result leaves, string positions), so a host links against the
  image alone (zbc-format.md). A request carries its import's API id
  (`api`) and its `fn_id` (`fn`); a host routes by `api`, and links a
  handler per API ([Linking](#linking)).
- `ROOT` (37) and `ROOTIMG` (38): the root candidates, each with its API hash
  and constructor; row 0 is the image's own (zbc-format.md). The C engine runs
  row 0's tables only.
- `EXEC` (codegen/debug profiles): `zbc_exectab`, then `{path, type, trait}` per
  executor in instance order (row index = the id requests carry), `{path, trait,
  first, count}` per executor group and the u32 member ids (an executor may be
  in several groups). `pssvm_exec_id(e, path)`, `pssvm_exec_path(e, id)`. An
  action's frame takes the executor its component's static `set_executor` put in
  force (`EXEC_SET` at the body's start); branches and callees inherit it.
  **Every model has an executor** (owner ruling, 2026-11-26). The
  *default executor* is the one the root component's `set_executor` puts in
  force (21.7.2.6); with none, bc infers one: a row after the instances',
  path `""` (the root's context; type `executor_base_c`, empty trait). Its
  row + 1 is the `zbc_exectab` header's `default_exec` (0: none, an image
  from before). Every frame with no parent starts on it, in both engines,
  and an action whose component has no executor of its own sets it
  (`EXEC_SET`), as if the component called `set_executor(executor())`. So
  every request carries an executor, a memory access reaches the platform
  through it, and `executor()` is never null, except in `pre_solve` and
  `post_solve`, which run on none. A function the host calls (`call_start`,
  `call_obj`) is called on the root's context, the handle `create()`
  returns (also the engine API for exported actions), and runs on it.
  Other executors' contexts (`get_context()`, LRM 21.7.1.1.1) are to
  come.
- `EVENT` (33): one `zbc_evtent` per host-fed source, `{path, kind, width,
  flags, count, head, buf, depth}`: every scalar `channel_c` instance, in slot
  order, `kind` 2 (`chan`, the notify kind that feeds it), `width`/`flags` bit
  0 its element's width and sign, and the component slots of its count, ring
  head and `depth` ring elements. The row index is the channel's id:
  `pssvm_event_id(e, path)` (-1: none; py shim `Engine.event_id`, SV
  `engine.event_id`). PSS has no event type, so there is no `event` kind;
  interrupts have no rows (the bench numbers them). A notify (below) for an
  unlisted id is `PSSVM_E_ARGS` at once (F5.5a).
- `IDENT` has `time_exp` (int32 in a u32): one tick is `10^time_exp` s
  (-12, ps, when the image has no IDENT or a 16-byte one; the debug profile
  writes IDENT when `time_exp` is not -12). `pssvm_time_exp(e)` (py shim and
  SV `engine.time_exp()`). `WAIT TIME` is in ticks; the SV host converts
  with its `$timeprecision`; the DPI `set_time` that stamps trace events
  stays in fs. A run under two timescales gives the same trace in ticks
  (F5.7). No PSS construct emits `WAIT TIME` yet (PSS has no time
  statement); the unit is there for hand-built IR and later imports.

After `DONE` of a `pssvm_call_start_obj` call, the host's `slots` hold the
action's solved values, and `pssvm_obj_field(e, entry, name)` gives a field's
slot index (-1: none) from the layout the image carries (F5.6).

## Conformance tests

A **scripted host** (be-bc `tests/unit/interp/abi_script.py`) drives a call
from a script: the requests to expect (api, fn, args or addr, executor), the
order to complete them, the times. `test_host_abi_v2.py` runs every script on
pyvm, and on the C engine through the same calls:

- a two-branch `parallel` with a `mem` access in each: both outstanding at
  once; completing the second first resumes its branch first; per-thread and
  per-executor order hold under every completion order;
- the same with a blocking user import and a `mem` access in parallel;
- a non-blocking import completed at the same time it is taken;
- `DONE` waits for outstanding requests; cancellation discards a late
  completion; completing an unknown or dropped `req_id` is `E_STATE`;
- `WAIT TIME` raced by a completion; `pssvm_set_time` backwards refused;
- events, executor ids, `time_exp` once F5.4, F5.5, F5.7 land.

## Concurrent calls

(Ruled 2026-11-25; pssc `docs/design/pssvm-in-practice.md` §4.1. pyvm
implements it; the C engine refuses what it does not, below.)

- **One traversal at a time.** A call on an action entry is a
  **traversal**: the root activation, owning the pools, resources, claims
  and the cone solve. At most one is active per engine. A traversal started
  while another is active **queues**: its `pssvm_run` returns `WAIT` `TURN`
  and runs nothing until the active one ends (DONE, ERROR or
  `pssvm_call_end`); traversals then start in FIFO order of their first
  `pssvm_run`. No host sees "busy" as an error. Ending a queued call drops
  it from the queue.
- **Function calls run alongside.** Any number of exported function calls
  may be live with the traversal, or with none. A function call never
  queues, and claims, locks, shares, produces or consumes nothing: it does
  not touch the traversal's pools, resources or solve. Calls interact only
  through component state and channels.
- **Interrupts are function calls.** The host signals an interrupt by
  calling the handler, an exported function, which may make blocking
  imports while the traversal waits (pyvm `CallTable.start_function`,
  `HostCall.start_function`). There is no interrupt notification on pyvm:
  `pssvm_notify` kind 1 (the old `PSSVM_N_IRQ`) is an error there. The C
  engine still has `pssvm_set_irq` and kind 1 until it catches up.
- **Calls interleave only at yields.** Calls are cooperative: what one call
  does between two of its yields is atomic with respect to the others. A
  call waiting on `EVENT` may be able to go on once another call has run (a
  channel put, a component field written): a host runs it again then.
  pyvm counts progress (`VM.progress`: every frame run that was not stuck,
  one that ended parked on a request included, since it may have put on a
  channel first; and host channel puts), and `HostCall.runnable()` says
  whether a waiting call can go on. SV's run loop wakes its waiting calls
  when the count moves ([SystemVerilog](#systemverilog-zsp_rt_bc_pkg)).
- **A deadlock is across calls.** A wait is dead only when no live call can
  run, no timer is due and the host has nothing outstanding or to notify.
  Within one call that is still `WAIT EVENT` (the host may yet act); a host
  loop driving several calls decides it. pyvm's `run_calls(calls, ...)`
  (and `run_sync`, which adopts the function calls started meanwhile) ends
  every live call in ERROR with a message naming what each waits on:
  `deadlock: ...: T (waits on an event), w (waits on an event), T2 (queued
  behind traversal T)`. A deadlock is a **fatal** ([Listeners](#listeners)):
  the listeners hear it first, every call is released, and with no
  listener the loop raises it.
- **Self-wait.** A traversal started from inside a request of the active
  traversal (an import's handler that starts a traversal) would wait for a
  traversal that waits for that handler. pyvm reports it, rather than
  hanging, when the host loop serving the request is a pyvm sync loop on the
  same host thread (`run_sync`/`run_calls` record what they serve): the
  new call ends in ERROR, `self-wait: traversal B was started inside request
  N (import) of the active traversal A; ...`, which the error listeners
  hear (only that call failed; the VM goes on). A host serving requests
  asynchronously (SV) cannot be checked by the engine; its facade must.
- **A traversal cannot start from inside a function call.** PSS functions
  cannot traverse actions; a host that wants a recovery scenario after an
  interrupt starts it after the handler returns (it then queues as usual).
- **The C engine** does not queue yet: `pssvm_call_start` of an action entry
  while another action call is live (not DONE or ERROR) fails with
  `PSSVM_E_STATE`, "cannot start traversal 'B': traversal 'A' is still
  active, and the C engine does not queue a second traversal yet". Function
  calls alongside a traversal run as before.

## Objects

(pssc `docs/design/pssvm-in-practice.md` §3.2. pyvm: `CallTable.new`,
`sub`, `release`, `call`, `field`, in `interp/handles.py`.)

A host names what it calls by **handle**, not by path. Handles are
`pssvm_obj_t` values from 1 (0 is none), never reused in an engine's life.

- **`pssvm_new(e, type, 0, args, len, seed)`** constructs the tree of root
  component `type` with its constructor arguments and returns the root's
  handle. It is LRM 20.1.2's construct-then-initialize in one call:
  initial values, `init_down`, `init_up`, the constructor. One root per
  engine: a second `new` while one is live is `PSSVM_E_STATE`. `type` is
  the image's own root (ROOT row 0); another root candidate is an image of
  its own (ROOTIMG), which the host loads. bc's root takes no constructor
  arguments yet (`len` 0).
- **`pssvm_new(e, action, parent, NULL, 0, 0)`** under a component makes an
  **action object**: the object a traversal of `action` runs on (its
  layout, `obj`), which stays readable after it
  (`pssvm_obj_field`). A traversal runs from the root for now: `parent` is
  the root's handle (a traversal in another instance is pssc pssvm-plan
  step 8's `comp=`).
- **`pssvm_sub(e, parent, path)`** gives the handle of the instance at
  `path` under component `parent` (`ch[0]`). Inside, an instance is
  `(tree, base)`; the handle replaces `pssvm_call_start`'s `comp_path`.
- **`pssvm_call(e, obj, method, args, len, seed)`** is a call keyed by
  object: on an action object, a traversal of it (`method` NULL, "" or its
  action; `args` an exported action's parameters, which pin its fields);
  on a component, its function `method`; `$run_end` on the root runs the
  tree's `exec run_end` blocks. The first traversal after `new` runs the
  tree's `exec run_start` blocks first. The call runs at its first
  `pssvm_run` or `pssvm_step`, as any.
- **`pssvm_release(e, obj)`** frees a handle explicitly (no reference
  counts). Releasing the root frees every handle under it and the tree;
  it is refused (`PSSVM_E_STATE`) while any call is live, as releasing an
  action object is while a call runs on it. A released handle is refused
  wherever it is used, naming what it was: `handle 2 (component sub: S)
  was released`. After the root's release, `new` builds a new tree.

`pssvm_construct`/`pssvm_fresh`/`pssvm_call_start` stay for hosts that
name by path; pyvm's sync API (`Session`) keeps them.

## Construction

(Ruled 2026-11-25, pssc `pssvm-in-practice.md` §4.3.) Construction is a
plain function call with no request loop, so it cannot yield. It may call
**non-blocking (solve) imports**, through their linked handlers, which the
engine calls synchronously ([Linking](#linking)). Anything else that
would yield is `construct-yield` and the construction fails: an access, a
blocking (target) import (`construct-yield: blocking import p::ext during
construction`), a non-blocking import with no handler linked (`... import
q::cfg with no linked handler ...`), a wait. The pssc gate refuses a
target import reachable from construction at compile time; the engine's
check is the backstop. A construction that fails is a **fatal**: there is
no tree, no handle, and the listeners hear it.

## Linking

`pssvm_link(e, api, ctx, fn, ud)` registers the handler of API `api` (an
id; `PSSVM_API_ANY`, -1: every API with no handler of its own, a
backstop) for requests whose executor is `ctx` (an `EXEC` id or
`0xFFFFFFFF`, none; `PSSVM_CTX_ANY`: every context). The lookup is (api,
executor), (api, any), then the backstop's. Linking is optional sugar
over the request loop: a host that serves requests by hand still works.

- **A non-blocking import with a handler is called synchronously**: the
  engine calls the handler and goes on, with no request and no host
  round trip (§3.2). That is what lets construction and solve-time code
  call one. Its error is raised at the import, as an error of the model.
- **A blocking request with a handler is still a request.** The engine's
  loops (`pssvm_step`; pyvm's `run_sync`/`run_calls` too) hand it to the
  handler as they take it. The handler answers at once, or keeps the
  request as a token and completes it later from the host's own timing
  (`pssvm_complete`; pyvm returns `DEFER`), which is also how a handler
  waits on hardware. Requests with no handler wait for `pssvm_take`.
- **`pssvm_link_check(e)`** fails, before time 0, naming each API and its
  imports that a coroutine reachable from an entry, the image's entry,
  construction or the run hooks calls with no handler for every context:
  `no handler linked for mem: read; q: q::cfg`. A backstop satisfies it.

pyvm: `CallTable.link(api, handler, ctx=CTX_ANY)` takes an API's id or
name (`"mem"`, the declaring scope, `"*"`), and a handler is a callable
`handler(request)`, an object with `call(fn_id, args)` (an
`ImportProvider`) or, for `mem`, a `Memory`. `link(api, None)` unlinks;
`missing_links()` is the list `link_check()` raises with.

## Listeners

(pssc `pssvm-in-practice.md` §3.0.6 note 3.) A listener hears the VM's
failures, in two severities:

| Severity | When | The VM afterwards |
|---|---|---|
| error (`PSSVM_SEV_ERROR`) | a call ended in ERROR: an unsat solve, an import that failed, a PSS `error()`, a self-wait | goes on; the next call runs normally |
| fatal (`PSSVM_SEV_FATAL`) | the VM cannot go on: a deadlock across calls, a construction that failed, an engine error marked fatal | refuses every later call, naming the fatal (`refused: the engine stopped on a fatal error: ...`), until `pssvm_new` builds a new tree |

- **A fatal releases everything blocked.** Every other call, started
  (parked on a request, an event, a timer) or queued for its turn, ends in
  ERROR with the fatal's message, so no host task waits for ever. A
  released call raises no event of its own, and `pssvm_run` on it gives
  its ERROR again.
- **Listeners are functions, called synchronously** at the failure, before
  any call it releases returns, so a listener cannot wait. The event
  (`pssvm_event_t`; pyvm `HostEvent`) names the call (`call_id`; 0 for a
  fatal that is no one call's, as a deadlock), its entry, the instance it
  ran in, the PSS location (`file`, `line`, `col`, `src_ref`) and the
  message.
- **By default a fatal ends the simulation.** With no listener the default
  one does: in SV it prints the located message and calls `$fatal` (the
  SV facade's, pssc pssvm-plan steps 7-8); in Python the host loop that met
  the fatal raises it, after the release. A registered listener replaces
  the default: one that only records lets the host decide, and one that
  raises (`uvm_fatal`) ends the run there.
- Host-ABI misuse (completing an unknown request, using a released
  handle) stays a return code (`PSSVM_E_STATE`) to its caller, not an
  event: the caller already has it, and the engine is unharmed.

pyvm: `CallTable.add_listener(l)` (`l.on_error(ev)`, `l.on_fatal(ev)`;
`interp/host.py` `Listener` has both as no-ops), `remove_listener`,
`fatal_exc`. The pyhdl-if shim listens on each VM it runs: it prints
`PSSVM FATAL: <located message>` and keeps it for `Engine.fatal()`.

## Driving every call: step and pending

For a host with no coroutines (C) or a foreign scheduler (cocotb),
`pssvm_step(e, budget)` runs every call that can run, one `pssvm_run`
slice each, in the order the calls were made, round after round, until
none can run or `budget` slices have run (0: no limit); a call's own
`pssvm_run` is the same for one call. New requests a linked handler serves
are handed to it then; the rest wait for `pssvm_take`. It returns the
slices run. A call queued for the traversal slot does not run before its
turn.

`pssvm_pending(e, &next_time)` says what is left across every call:
`PSSVM_PEND_READY` (1) a call can run, `PSSVM_PEND_REQS` (2) requests are
outstanding (issued and not completed), `PSSVM_PEND_TIME` (4) a timer is
set, the earliest in `next_time` (ticks). A call waiting on a timer runs
once the host's clock (`pssvm_set_time`) reaches it. Nothing pending with
calls live is for the host to break (a notify, a completion); if it has
nothing, that is a deadlock, which pyvm's `run_calls` decides.

pyvm: `CallTable.step(budget=0)`, `pending()` (a `Pending`: `ready`,
`reqs`, `next_time`, `flags`), `set_time(now)` for every call.

## Completion

A call is its own completion object. `pssvm_done(h)` is 1 once it ended in
DONE or ERROR, `pssvm_status(h)` which; `pssvm_on_done(h, fn, ud)`
registers a callback the engine calls once, at that end, on the host
thread that saw it, before that `pssvm_run`/`pssvm_step` returns (at once
if the call is over). Polling suits a firmware loop, the callback an
event-driven host (§3.5):

```c
pssvm_call_t *c = pssvm_call(e, act, NULL, args, len, seed);
pssvm_on_done(c, my_done, ud);            /* optional */
while (!pssvm_done(c)) pssvm_step(e, 0);  /* or the host's own loop */
if (pssvm_status(c) != PSSVM_DONE) fprintf(stderr, "%s\n", pssvm_error(e, c));
pssvm_call_end(c);
```

pyvm: `HostCall.status` (None until over), `HostCall.on_done(fn)` (`fn(call,
request)`), `HostCall.id`.

**The C engine** implements none of [Objects](#objects) to here yet (pssc
pssvm-plan: the C engine is on hold): no handles, links, listeners,
`step`/`pending` or completion callbacks. What it would run wrongly it
refuses: its DPI staging (`g_seed`, the argument stage, the message
queue) is process-global, so `pssvm_dpi_load` refuses a second engine in
one simulation.

## Notify

`pssvm_notify(e, kind, id, value)` tells the engine about a host event. It may be
called at any time between engine calls, from any SV thread; it takes effect
at the next `pssvm_run` of any call (an interrupt is taken between
instructions).

| `kind` | Meaning | pyvm today |
|---|---|---|
| `PSSVM_N_IRQ` (1) | C engine only, until it catches up: interrupt `id` raised; starts the image's interrupt handler as a thread of its own in the root component. pyvm has no such kind: an interrupt is a call of the handler ([Concurrent calls](#concurrent-calls)), and kind 1 is an error | pssc `Ctx.raise_irq(irq)` starts the handler as a function call |
| `PSSVM_N_CHAN` (2) | put `value` on the host-fed channel `id` (its `EVENT` row; `pssvm_event_id`), masked to the channel's width | `HostCall.notify(2, id, value)` (a try_put on the channel) |

On the C engine the interrupt handler is a function entry the host names
once with `pssvm_set_irq(e, entry)`. The C engine queues notifications (at most
`PSSVM_NOTES_MAX`, 64; more is `PSSVM_E_STATE`) and the first `pssvm_run` of
any call takes them all: an IRQ starts one handler thread in that call. A kind
with no handler (an IRQ before `pssvm_set_irq`) ends that call in `ERROR`
("notify kind K has no handler"), as pyvm's `VMError` does. A `CHAN` id with
no `EVENT` row is refused at `pssvm_notify` (`PSSVM_E_ARGS`); a put on a full
channel ends the call that takes it in `ERROR` ("host put on full channel
<path>"). A channel put goes to one call: a shim serving several calls gives
it to one of them (and queues it while no call exists).

Each call waiting on `EVENT` sees the notification on its next `pssvm_run`;
one that is still not runnable returns `WAIT EVENT` again.

Each notification taken raises a `notify` event (when the events sink watches
`EV_NOTIFY` in every coroutine), before its handler starts:
`{irq|chan: id, value, delivered}` (trace-schema.md "The platform side"), so
a recorded run says where each one arrived, on both engines.

## Bench reports (F2.7)

`pssvm_report(h, kind, t_fs, json)` tells the engine the bench saw something:
`kind` names the source (`uvm` from `pssvm_uvm_pkg::pssvm_report_catcher`,
`bench` from `pssvm_bench_fail`), `json` is `{severity, id, msg, keys}`, `t_fs`
the sim time. It needs no response and may come at any time (`h` 0: from no
call). The engine snapshots the action runs in flight, ranks suspects and
attributes the report to the first (F2.8): a run the bench names (`keys.stim`,
else a `pss#<id>` token in `msg`; evidence `tag`), then the runs in flight,
newest first (`temporal`). It raises a `bench` event (trace-schema.md); an
`error` or `fatal` fails the test.
pyvm's shim also prints one line per failing report to the simulation log
(F2.10c), e.g.

    PSSVM: uvm error SB: action ch_c::xfer_a pss#t#0.#2#0 (tag,temporal); in flight: ...

An action bc inferred (M9b1b) adds the action it was inferred for, after the
evidence and after its in-flight entry:
`action src pss#... (temporal), inferred for pss#t#0.#0#0; in flight: ... pss#... src (inferred for pss#t#0.#0#0)`.

The bench learns which run made a request from its `stim_id` (F2.8; pyvm's
shim `req_str(h, "stim_id")`, SV `zsp_rt_bc_req.stim_id`, and
`zsp_rt_bc_platform_if.stim_id`, set before each request is served, so the
memory and import tasks can stamp it on their transaction). The C engine gives "" (M14).
`bench_failed()` counts those, and the SV runner fails a test when it grows.
pyvm's shim (`Engine.report`) has it; the C engine does not yet (its SV
`report` is the base class's no-op).

## Reload

`pssvm_reload(e, img, len, flags)` loads the image as generation N+1 in place
of N. It is refused (`PSSVM_E_API`) when `IDENT.api_hash` differs from N's: a
signature changed, and the SV facade would no longer match. Details:
[generations.md](generations.md). (Applying it at a later quiescent boundary
while calls finish in N is not implemented: call it with no call live.)

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

## Other calls

`pssvm.h` also has calls a host needs only for one feature:

| Call | What |
|---|---|
| `pssvm_meta(e, name)` | the text of the image's host metadata document `name` (`META` section; pssc writes them in debug images), or NULL; valid while `e` is |
| `pssvm_bind(e, coro, on)` | from the next call on, each `CALL` of coroutine `coro` is a request of a user API with `bound = 1`, `fn` the coroutine and `args` its arguments, which the host answers with the function's result in place of its body; `PSSVM_E_ARGS`: no such coroutine |
| `pssvm_set_events(e, sink)` | raise debug events ([trace-schema.md](trace-schema.md), rt-eng `zbc_events.h`) into `sink` in every later call; which events per coroutine is the sink's watch mask |
| `pssvm_set_host(e, host)` | interim: the `zbc_host_t` whose message callback, verbosity and import log every call uses (its `mem_access` and `import_returns` are not: imports and memory yield); kept only to set the verbosity |
| `pssvm_call_status(h, &halted_op, &now)` | the engine-side `ZBC_*` status and halting opcode of a call that ended in `ERROR`, and its end time in ticks (tests compare it with `zbc_run`) |
| `pssvm_image(e)` | the loaded image's parsed sections (`zbc_image_t`) |
| `pssvm_stub_*` | the C debug stub over these calls ([debug-stub.md](debug-stub.md); rt-eng `pssvm_stub.c`) |

## Return codes

`0` ok; `PSSVM_E_STATE` (-2) call made in the wrong state; `PSSVM_E_ARGS` (-3)
bad entry, path or argument size; `PSSVM_E_API` (-4) reload refused;
`PSSVM_E_IMAGE` (-5) bad image. Every failure sets `pssvm_error`.

## How pyvm maps onto it (P3.1b)

| Host ABI | pyvm today (synchronous) |
|---|---|
| `REQS`, API `mem` | `ops_orch._builtin` → `vm.memory.read/write` (`_MEM_READ`/`_MEM_WRITE`) |
| `REQS`, a user API | `ops_orch._op_import` → `vm.import_provider.call(fn_id, args)` |
| `WAIT TIME` | `VM._drain` → `sched.next_timed()` (virtual `now`) |
| `WAIT EVENT` | `VM._drain` → `VMError("deadlock: ...")` |
| `WAIT TURN` | a queued traversal's `HostCall.run` (`CallTable`) |
| `STOP` | the stub blocks the VM thread in `VM._run_frame_ev` (debug-stub.md) |
| `DONE` | `_drain` returns (`sched.has_work()` false) |
| `pssvm_construct` | `VM.construct_components` |
| `pssvm_call_start` | `VM.root_frame(coro, seed, obj)` |
| `pssvm_new` / `sub` / `release` / `call` / `obj_field` | `CallTable.new` / `sub` / `release` / `call` / `field` (`interp/handles.py`) |
| `pssvm_link` / `link_check` | `CallTable.link` / `link_check` |
| `pssvm_add_listener` | `CallTable.add_listener` |
| `pssvm_step` / `pending` | `CallTable.step` / `pending` |
| `pssvm_done` / `status` / `on_done` | `HostCall.over` / `status` / `on_done` |

In pyvm the seam is a Python object per call: `run()` returns the status
(`REQS` and the others above), `take() -> Optional[Request]` gives each new
request and `complete(req_id, value=None)` answers one; `Request` carries
`req_id`, `api`, `fn`, `thread` and `executor`. Memory providers and import
providers are handlers of the `mem` API and the user APIs; the existing `Memory` and `ImportProvider`
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
| `HostCall.run/take/complete(req_id)/fail(req_id)/result/end` | `NativeCall.run/take/complete(req_id)/fail(req_id)/result/end` (`pssvm_take`, `pssvm_complete(h, req_id, data, len)`) |
| `run_sync(call)` | `run_sync(call, memory, import_provider)` |
| `vm.cobj` | `Engine.comp_slots()` (`pssvm_comp_read`, interim until P4.4) |

The C engine keeps at most one request outstanding (so `pssvm_take` gives
at most one per REQS) and keeps `pssvm_request(h)` for what a run status
alone does not say: `on` and `delay` after `WAIT` (`on` = REQS while a taken
request is unanswered) and `src_ref` after `STOP`. `NativeCall.run` takes
every new request at once and queues it for `take`.

SV sees both engines the same way (`pssvm_dpi.c` for C, the pyvm shim for
Python): `run(h)` gives -1 ERROR, 0 DONE, 1 REQS, 3 WAIT or 4 STOP (pyvm
only; the C DPI serves STOP itself); `take(h)` gives the next taken
request's `req_id`, -1 when none is left; `req_int/req_u64/req_str/
req_nargs/req_arg(h, rid, ...)` read request `rid`'s fields (`req_id`,
`api`, `executor`, `thread`, `has_ret`, ...), rid -1 the run status's own
(`on`, `delay`, `reason`, `src_ref`); `complete(h, rid, data)` answers it.
`on` is 0 TIME, 1 EVENT, 2 REQS (the DPI maps C's values). The pyvm shim
still answers imports a double mocks itself; SV never sees them. The C
DPI keeps a small table of (call, req_id) -> taken request, dropped on
complete or call end.

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

`pssvm_abi_version()` returns 2 (version 2 replaced the one-request-per-call
model of version 1, with no version-1 mode). A new status, request field or notify kind
adds a minor feature the host checks for; a change to an existing call's
meaning raises the version. `zsp_rt_bc_pkg` refuses an engine of any other
version.

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
a thread re-entering where it already passed the check (a completed request, a
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
size > 0, and an optional value compared under mask). A `mem` request that
overlaps a watch, once done (a read has its value), stops the run before the
next instruction: reason 4 (watch), the hit ids, the access's src_ref, and
the access in the stop record (is_write, size in bytes, addr, data).
`pssvm_dbg_set_stop_on(e, PSSVM_STOP_ON_ERROR)` (default off) makes the run's
first error stop where it happened, reason 5 (error) with status, halted op
and message, frames still live; any resume ends the call with that error.
Python: `Engine.dbg_set_watches` (the stub's `set_access_watch` dicts),
`Engine.dbg_set_stop_on`, `NativeCall.dbg_stop()["detail"]`.
