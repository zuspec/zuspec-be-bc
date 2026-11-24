# Run API

Status: **full-PSS plan F1.2a** (memory and imports; events, time and
observation are F1.2b) · Authority: this file, then `zuspec.be.bc.run` ·
Engine seam underneath: [host-abi.md](host-abi.md).

Every host that executes an image (a test runner, a compliance adapter, a
benchmark, a simulator shim) needs the same sequence. `zuspec.be.bc.run` is
that sequence, with nothing host-specific in it (it never imports pssc):

```python
from zuspec.be.bc import run as bcrun
from zuspec.be.bc.interp import NativeBlobBackend

res = bcrun.run("model.zbc", "pss_top::check_regs",
                bcrun.SimplePlatform(memory=my_mem, imports=my_imports),
                seed=7,
                opts=bcrun.RunOpts(setup=["pss_top::init"],
                                   solve_backend=NativeBlobBackend()))
if not res.ok:
    print(res.verdict, res.error)
```

## `run(image, entry=None, platform=None, seed=0, opts=None) -> RunResult`

1. `image`: `.zbc` bytes, a path, or a `ZbcModel` (`load`).
2. `entry`: a symbol of the image's entry table (an action by name, a
   function by `<comp type>::<fn>`), a coroutine name, a coroutine index, or
   None for the image's own entry (`resolve_entry`).
3. A VM is built on the platform; `opts.bindings` replace functions by host
   callables; with `opts.name` set, `test.start` is raised.
4. The entry's root frame is made (its arguments: `opts.args`), the
   component tree is constructed, each `opts.setup` entry runs in the root
   component, then the entry runs to quiescence.
5. Each `opts.teardown` entry runs whatever the entry did. A teardown that
   fails after the entry failed does not replace the entry's error.
6. `test.end` (with `opts.name`); the run's own solve cache is released.

## `Platform`

What the run's target answers. Both attributes may be None.

| Attribute | Has | None means |
|---|---|---|
| `memory` | `read(addr, size)`, `write(addr, size, value)` (`extern.Memory`) | a fresh sparse `extern.Memory` |
| `imports` | `call(fn_id, args)` (`extern.ImportProvider`) | no user imports |

`SimplePlatform(memory, imports)` builds one. A platform may also have
these (each optional, looked up with `getattr`). Any of the first four runs
every entry over the host ABI (`HostCall` + `run_sync`):

| Attribute | Called | Without it |
|---|---|---|
| `wait_time(host, delay)` | every thread waits on time; the platform's clock moves `delay` units and may `host.notify(kind, id, value)` | time runs on at once |
| `wait_event(host) -> bool` | every thread waits on an event; may notify; True goes on | deadlock (`runtime_error`) |
| `on_notify(vm, kind, id, value)` | a notification is taken (`HostCall` `on_notify`) | `N_IRQ` starts `irq_entry`; any other kind is an error |
| `irq_entry` | entry (symbol or index) an `N_IRQ` starts as a thread in the root component | — |
| `observe(event)` | each debug event of the kinds in `watch` | — |
| `watch` | the `trace.events` mask `observe` sees (default `EV_PLATFORM`: access, import, notify) | — |

## `RunOpts`

`args`, `setup`, `teardown`, `bindings`, `obj` / `field_names`, `name`,
`host_abi` (run each entry over `HostCall` + `run_sync`, as a simulator host
does, instead of `VM._drain`), `on_root(frame)` (called once the entry's
frame is the VM's entry, before anything runs), and
what `oracle.run_model` took: `solve_backend` (default `FixedSolveBackend`;
a PSS image wants `NativeBlobBackend()`), `solve_cache` / `max_restarts`,
`solve_bounds`, `on_solve`, `out`, `verbosity`, `sink`, `events`, `watch`.
`coverage` watches statements and branch arms and counts them in
`RunResult.coverage`. When `watch`, `coverage` or an observing platform
needs debug events and `events` is None, `run` keeps them in a `MemorySink`.

## `run_on(vm, entry=None, seed=0, opts=None, platform=None) -> RunResult`

`run` on a VM the host built (a debugger already holds it, or the host's
platform made it): steps 2-6 above on `vm`. Only the per-run options apply
(`args`, `setup`, `teardown`, `bindings`, `obj`/`field_names`, `name`,
`host_abi`, `on_root`); the backend, cache, sinks, watch and bounds are the
VM's; of `platform` only the time and notification hooks are used. pssc's `bc_session.run` and `Session.run_entry` use it.

## `RunResult`

`oracle.RunResult` (`obj`, `retval`, `events`, `now`, `frames`, `solved`)
plus:

| Field | What |
|---|---|
| `cobj` | the component object the run left behind (a ledger's counts live there) |
| `verdict` | `ok`, `solve_fail`, `runtime_error` or `unfinished` |
| `error` | the `VMError` that ended the run, unless `ok` |
| `debug_events` | the debug events the run raised, when its events sink keeps them (`events` is the orchestration trace) |
| `coverage` | with `RunOpts.coverage`: `{"stmts": {id: hits}, "arms": {id: hits}}`, IDs `<scope>/<path>` (an arm `<scope>/<branch>/<arm>`, as pssc's report names them); points never reached are absent |

A failure in the image (any `VMError`) is a verdict, not an exception;
`RunResult.check()` raises it. A bad entry or an exception from the host
raises. `oracle.run_model` is `run(...).check()` on the image's own entry.

| Verdict | When |
|---|---|
| `ok` | the entry and every setup and teardown entry finished |
| `solve_fail` | `SolveUnsatError`, `SolveBudgetError`, `ScopeUnsatError` |
| `runtime_error` | any other `VMError` (a failed assert, a bad access, a deadlock) |
| `unfinished` | the run went quiet with the entry blocked |

Judging a test (pass/fail counts, divergence, checks) stays with the host.

## Command line: `zuspec-bc run`

`zuspec-bc run IMAGE [options]` (or `python -m zuspec.be.bc run ...`) runs one
entry of a `.zbc` file through `run`, solving with dv-solve. It needs only
be-bc and its dependencies, not pssc.

| Option | What |
|---|---|
| `--entry E` | an action name, `<comp type>::<fn>`, a coroutine name or index (default: the image's own entry) |
| `--seed N` | solver seed; default random, and always printed |
| `--arg V` | a function entry's argument, in order; repeat it (`0x` prefixes allowed) |
| `--verbosity V` | `message()` verbosity shown: 0 NONE … 4 FULL (default 2 MEDIUM) |
| `--platform py:<module>[:<attr>]` | imports `<module>` (current directory first) and takes `<attr>` (default `platform`): a `Platform`, or a callable returning one. Without it the image runs on the VM's own memory with no user imports. |

What the image prints goes to stdout, then one verdict line:
`PASS <entry> seed=N` or `FAIL <entry> seed=N: <verdict>: <error>`.

| Exit | When |
|---|---|
| 0 | the run finished (`ok`) |
| 1 | it did not (any other verdict) |
| 2 | no such image or entry |
| 3 | the platform did not load |
