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
  call's error set; a bench reads it in a `pssvm_ctxt` subclass's
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

On the SV side (`pssvm_pkg`), `pssvm_req.nret` is the field count when it is
more than 1 (else 0: a scalar, or a one-field struct answered through the
usual `ret`). The platform's `call_import` pushes the fields on
`req.rets`; `pssvm_ctxt` stages each with the engine's `set_ret(h, rid, i,
v)` (DPI `pssvm_dpi_set_ret`, read as `pssvm_dpi_req_int(h, rid, "nret")`;
pyhdl-if `PssvmEngine.set_ret`) before `complete`, whose `data` a struct
result ignores. Fields are staged one call each because Verilator 5.052
cannot compile an open array in a DPI import.

**String arguments (F6.3a).** bc makes no string at run time: every string
value is an index into the image's model-string table (the message string
table's first `n_model_strings` entries). A string argument of an import
crosses as that index, a scalar like any other; the lowered model's
`imports` (`Model.imports`, in the API hash, not in the image bytes) lists
its position in `string_at`, and the host turns it into text
with `pssvm_string(e, idx)` (NULL out of range; valid while `e` is). The
Python hosts have `HostCall.string(idx)` (pyvm: `Model.strings[idx]`) and
`Engine.string(idx)` (C engine through ctypes), both None out of range. In
SV, `pssvm_engine::string_at(idx)` gives it ("" out of range): DPI
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
`call_import` sets `pssvm_req.ret_s[i]` (keyed by leaf, so an empty string
is still an answer) and `pssvm_ctxt` stages each with
`pssvm_engine::string_ret` before `complete` (pyhdl-if
`PssvmEngine.string_ret`; `pssvm_engine_dpi` has none until the C engine
has a heap, M14, and keeps the default that refuses). The image's `imports` metadata marks
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
| `PSSVM_WAIT` `EVENT` | 3 | nothing ready or outstanding; threads wait on the host | waits for something it will `pssvm_notify`, then `pssvm_run` |

Value 2 (the old `IMPORT`) is unused.

`pssvm_req_t` is a plain struct:

| Field | What |
|---|---|
| `req_id` | unique among the call's outstanding requests (reused after completion) |
| `api`, `fn` | API id and function id; `name` the function's PSS name (NULL when the image has no `IMPORT` entry for it) |
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
- **Several calls** run at once; each has its own outstanding set.

## Hosts

- **SV (`pssvm_pkg`).** One process per taken request, dispatched by `api`
  to the handler registered for (`api`, `executor`): `mem` to the agent or
  sequencer bound to the executor path, user APIs to the import class, `seq`
  to a sequence starter. Requests of one executor (>= 0) are served FIFO in
  issue order, none overlapping; requests with no executor (-1) run freely.
  Requests of different calls or executors that reach a shared bus in one
  time step may reach it in either order.
- **Python sync loop (`run_sync`).** Takes every request and completes them in
  issue order, non-blocking ones first. For one thread this is the old order.
- **Python host for the C engine** wraps the same calls (below).

## Image sections for hosts

- `API`: `{id, name}` per API the image calls; `IMPORT` rows gain `api`.
  `pssvm_api_id(e, name)` looks one up.
- `EXEC` (codegen/debug profiles): `zbc_exectab`, then `{path, type, trait}` per
  executor in instance order (row index = the id requests carry), `{path, trait,
  first, count}` per executor group and the u32 member ids (an executor may be
  in several groups). `pssvm_exec_id(e, path)`, `pssvm_exec_path(e, id)`. An
  action's frame takes the executor its component's static `set_executor` put in
  force (`EXEC_SET` at the body's start); branches and callees inherit it.
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

## Notify

`pssvm_notify(e, kind, id, value)` tells the engine about a host event. It may be
called at any time between engine calls, from any SV thread; it takes effect
at the next `pssvm_run` of any call (an interrupt is taken between
instructions).

| `kind` | Meaning | pyvm today |
|---|---|---|
| `PSSVM_N_IRQ` (1) | interrupt `id` raised; starts the image's interrupt handler as a thread of its own in the root component | `Ctx.raise_irq(irq)` (pssc `testing/vm.py`) |
| `PSSVM_N_CHAN` (2) | put `value` on the host-fed channel `id` (its `EVENT` row; `pssvm_event_id`), masked to the channel's width | `HostCall.notify(2, id, value)` (a try_put on the channel) |

The interrupt handler is a function entry the host names once with
`pssvm_set_irq(e, entry)` (pyvm: `HostCall(irq_entry=coro)`, pssc
`Session.irq_entry`). The C engine queues notifications (at most
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
shim `req_str(h, "stim_id")`, SV `pssvm_req.stim_id`, and
`pssvm_platform_if.stim_id`, set before each request is served, so the
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
| `STOP` | the stub blocks the VM thread in `VM._run_frame_ev` (debug-stub.md) |
| `DONE` | `_drain` returns (`sched.has_work()` false) |
| `pssvm_construct` | `VM.construct_components` |
| `pssvm_call_start` | `VM.root_frame(coro, seed, obj)` |

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
meaning raises the version. `pssvm_pkg` refuses an engine of any other
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
