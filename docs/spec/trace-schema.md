# Trace / event schema

Status: **P0 contract (M1)**; debug events pssc P1.6 · Version `2` · Authority:
`zuspec.be.bc.trace.schema` · Companion to design `[D§10]`.

An execution emits an **event-sourced** trace of the orchestration ops. The
serialized form is canonical so two runs with the same seed produce a
**byte-identical** trace and oracle-vs-engine traces are byte-diffable.
The interactive debug stub that stops on these events is specified in
[debug-stub.md](debug-stub.md).

## Event record

```
TraceEvent { seq: int, kind: str, coro: int, src_ref: int, detail: {..} }
```

- `seq` — monotonic, assigned by the sink.
- `kind` — an orchestration op, one of `SPAWN INVOKE PAR JOIN WAIT SELECT SOLVE
  BIND IMPORT YIELD`, or a debug event kind (below, lower case).
- `coro` — emitting frame/coroutine id.
- `src_ref` — provenance back-pointer (`zbc_prov` index; 0 = none).
- `detail` — kind-specific payload (e.g. `SOLVE` → `{"vars": {...}, "seed": N}`,
  `SELECT` → `{"choice": i}`, `IMPORT` → `{"fn": id}`).

### Stimulus ids

`INVOKE` carries `stim`, the **stimulus id** of the run it starts, stable for
a given (image, seed) and equal across seeds while the activity takes the
same shape. A root run is `name#k` (its action's or function's last name
component; k counts that name's roots in the VM from 0). A traversal of an
activation node is the root's id, `.`, the node's path, then `#n`, how many
traversals of that node the root has started before it:
`top_a#0.p#2`, `top_a#0.p.l1#2`, `top_a#0.p.#0#1` (an anonymous `do`'s key is
`#0`, so the occurrence is always the last `#n`). A traversal outside an
activation is its parent's id, `.`, its action's name and `#n`. A call, or a
branch that runs on its parent's node, has its parent's id. The C engine
records no orchestration events and so no ids (yet).

## Serialized form

JSONL with a one-line versioned header:

```
{"zbc_trace":2}
{"coro":0,"detail":{"child":1},"kind":"SPAWN","seq":0,"src_ref":3}
...
```

JSON is emitted with **sorted keys** and **compact separators**, so the byte
stream is a pure function of event contents — `detail` key order does not affect
the bytes. This is what makes trace equality a reliable determinism check (T1-D).

## Debug events (v2)

A second stream, separate from the orchestration trace: the VM writes debug
events to its `events` sink (`VM(events=...)`), never to its trace `sink`, so
the orchestration trace is byte-identical with events on or off (apart from the
version header). The record shape is the same `TraceEvent`. Which events a
coroutine raises is a bit mask, set with `VM.watch(mask, coros=None)`
(authority: `zuspec.be.bc.trace.events`); the VM tests it once per frame entry,
so a run with no events on pays one test per frame, not per instruction.

| Kind | Bit | Raised by | `detail` |
|---|---|---|---|
| `stmt` | `EV_STMT` | VM, at a statement's first pc (debug lowering's LINE table) | `{pc}` (+ `path`, and `scope` in a function: the statement's coverage item; + `defs`/`uses`: the values it stores and loads, `l:<name>`/`l#<slot>`, `a#<i>`, `f#<slot>`, `c#<abs slot>`, first-seen order, `interp.defuse`); `src_ref` = the statement |
| `call` / `return` | `EV_CALL` | VM, a called form's frame starts / ends | `{fn, depth}`, + `args` on `call`, + `retval` on `return`; a host-bound call (`VM.bind`): + `mocked: true` |
| `test.start` / `test.end` | `EV_TEST` | host, `VM.host_event` | the host's (name, seed, verdict, ...) |
| `access` | `EV_ACCESS` | VM, a memory builtin (`read32`, `write32`, ...) | `{rw: "r"\|"w", width, addr, data}` |
| `import` | `EV_IMPORT` | VM, a user import returned | `{name, fn_id, args, blocking, mocked}` (+ `ret`) |
| `message` | `EV_MESSAGE` | VM, `message()` / `print()` (info, level 0) / `error()` / `fatal()` | `{severity: "info"\|"error"\|"fatal", text}` (+ `level` / `status`) |
| `check` | `EV_CHECK` | VM, a function the host named a check (`VM.on_check`) starts | the host hook's (`{verdict, what, check}` in pssc) |
| `chan` | `EV_CHAN` | VM, a channel operation ended (debug lowering) | `{channel, op, value}`; a try adds `ok`, and has no `value` when it failed |
| `notify` | `EV_NOTIFY` | host (the platform), `VM.host_event`; a host-ABI notification as `run` takes it (pyvm `HostCall`, C `pssvm_notify`) | pssc `raise_irq`: `{irq, delivered}`; host ABI: `{chan: id, value, delivered}` (`{kind, ...}` for an unknown kind; the C engine, which keeps `N_IRQ` for now, also `{irq: id, ...}`) |
| `action` | `EV_ACTION` | Python VM only (the C engine ignores the bit), an action run starts (its frame's first run) / ends (done or cancelled) | `{stim, type, phase: "start"\|"end"}` (+ `cancelled: true`); `stim` = its stimulus id (below), `type` = its coroutine. `VM.in_flight()` lists the stims of action runs not yet ended, oldest first |
| `bench` | `EV_BENCH` | host, `VM.bench_report(kind, severity, msg, id, time, keys)` (F2.7; Python VM only): the bench reported a failure (`uvm_error` via the report catcher, an assertion, `pssvm_bench_fail`) | `{kind, severity, id, msg, time, keys, in_flight: [{stim, type}], stim, suspects: [{stim, evidence}], verdict}`; `in_flight` = action runs not yet ended, oldest first; `suspects` (F2.8) = the run the bench names (`keys.stim`, else a `pss#<id>` token in `msg`; evidence `tag`, plus `temporal` if in flight), then the others in flight, newest first (`temporal`); `stim` = the first suspect (None if none); `verdict` = `fail` for `error`/`fatal` (the run's verdict becomes `bench_fail`, `RunResult.bench` lists them), else `record` |
| `solve` | `EV_SOLVE` | VM (both engines; the C engine sends raw keys a host fills, [solve.md](solve.md)), a solve chose values: a traversal's (SOLVE_NODE) or a SOLVE / procedural `randomize` | `{site, node, type, seed, values: {name: value}, softs, dropped: [{text, file, line}]}`; `softs` = soft constraints in force, `dropped` = those the solution breaks (dv-solve drops a soft only on a conflict, so these are exactly the dropped ones; checked only while watched); a problem loaded from an image (no source) lists none |
| `cov` | `EV_COV` | VM (Python engine; C engine from F4.6b), `COV_SAMPLE`: a covergroup instance was sampled | `{covg, values: [int]}`; `covg` = index in the image's COVG section, `values` in COVG-document order (per coverpoint its value, then its iff flag if any; then one iff flag per cross with one); the host bins them (`cov/covergroup.py`), so a trace carries the run's coverage in and out of simulation |
| `sanitizer` | `EV_SANITIZER` | host (a sanitizer hook), `VM.host_event` | `{name, verdict, what}`; `src_ref` = the location blamed |

`stmt` and `chan` need a model lowered with `debug=True`; `chan` events come
from the `NOP` that ends each channel operation (`CoroDescriptor.chans`, in
memory only). The host kinds are on when a `watch` of every coroutine sets
their bits. `coro` is the raising frame's id (0 for host events).

**The platform side** of a run is what the host answered and when it
notified: `access` with `rw: "r"` (its `data` is the value read), `import`
(its `ret`) and `notify`, in stream order. `EV_PLATFORM` (`"platform"` in
`mask_of`) watches just those; `events.platform_side(events)` lists them as
`("read", addr, width, data)`, `("import", fn_id, ret)` and
`("notify", detail)`, equal for two runs that took the same answers on
either engine (rt-eng `tests/test_native_host.py`). A replay (pssc P5.1)
feeds them back in that order. Notifications are ordered by their place in
the stream; there is no platform time yet.

Not yet: platform time on each event, the register path and decoded fields on
`access`, and a live socket (pssc plan P1.7, P3).

## Binary form (`.pstrace`)

The same records in fewer bytes, written by the native engine's sink and by
`BinarySink` (authority: `zuspec.be.bc.trace.binary`, pssc P4.4c).
Conversion is lossless both ways: `to_ndjson(from_ndjson(t)) == t` for a
canonical NDJSON trace `t`, so readers may convert and use the NDJSON code.

```
file   := "PSTRACE\0" uvarint(version) value(header map) record*
record := uvarint(len) body            -- body is len bytes
body   := zigzag(seq - prev_seq - 1)   -- prev_seq of the first record = -1
          value(kind: str)  uvarint(coro)  uvarint(src_ref)
          value(detail: map)  value(extras: map)
```

The header map is the NDJSON header without `zbc_trace` (a test's name,
seed, ...). `extras` holds a line's keys other than the five fixed ones
(`file`, `line`, `t`, ...); empty when there are none. Integers are LEB128
(`uvarint`; `zigzag` maps n to 2n, -n to 2n-1), of any size.

| Tag | Value | Payload |
|---|---|---|
| 0 / 1 / 2 | null / false / true | none |
| 3 | int | zigzag uvarint |
| 4 | float | 8 bytes, binary64 little endian |
| 5 | str | uvarint byte length, UTF-8; gets the next string id |
| 6 | strref | uvarint id of a string earlier in the file |
| 7 | list | uvarint count, values |
| 8 | map | uvarint count, (key: tag 5 or 6, value) pairs, keys sorted |

String ids count from 0 in first-appearance order across the file (header
included). Maps are written with sorted keys, so the bytes are a pure function
of the events. A reader rejects a bad magic, a newer version, a record whose
length does not match its fields, and a truncated last record.

## Sinks

`MemorySink` (tests / differential comparison), `FileSink` (streams JSONL),
`NullSink` (runtime hot path — still advances `seq` so callers behave
identically), `CallbackSink` (each event to a callable; debug events),
`BinarySink` (streams the binary form).

## Versioning

`TRACE_SCHEMA_VERSION` bumps on any incompatible change to the record shape; the
header carries it so a reader can reject or adapt. v2 added the debug kinds; a
v1 orchestration trace and a v2 one differ only in the header.
