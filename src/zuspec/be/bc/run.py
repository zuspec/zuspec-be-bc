"""
run.py -- the one way to run a ZBC image (full-PSS plan F1.2).

Every host that executes an image (a test runner, a compliance adapter, a
benchmark, a simulator shim) wants the same sequence: load the image, pick an
entry, build a VM on a platform, construct the component tree, run the setup
entries, the entry, the teardown entries, and say how it went. This module is
that sequence, with no host-specific knowledge: a host supplies a
:class:`Platform` (memory and imports) and :class:`RunOpts`, and reads a
:class:`RunResult` whose ``verdict`` says whether the run finished.

Public surface:

* :class:`Platform`  -- what a run's target answers: memory accesses, imports
  and, optionally, time, notifications and observation (served over the host
  ABI, ``interp.host.HostCall``).
* :class:`SimplePlatform` -- a platform from a memory and an import provider.
* :class:`RunOpts`   -- everything else a run may be given.
* :class:`RunResult` -- the oracle's result plus the component object, the
  verdict and the error.
* :func:`run`        -- run one entry of an image.
* :func:`run_on`     -- the same on a VM the host built.
* :func:`load`       -- an image from bytes, a path or a model.
* :func:`resolve_entry` -- an entry's coroutine index from its symbol.
"""

import dataclasses as dc
import functools
import os
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Union, \
    runtime_checkable

from .model import ZbcModel
from .trace import events as ev
from .trace.sink import MemorySink, TraceSink
from .interp.vm import VM, VERBOSITY_MEDIUM
from .interp.ops_proc import VMError
from .interp.heap import HEAP_CAP
from .interp.extern import Obj, SolveBackend, ImportProvider
from .interp.solve_cache import SolveCache, SolveBudgetError, SolveUnsatError
from .interp.activation import ScopeUnsatError
from .interp import oracle

#: The entry and every setup and teardown entry finished.
OK = "ok"
#: A solve found no values, or ran out of budget.
SOLVE_FAIL = "solve_fail"
#: The image raised an error (a failed assert, a bad access, a deadlock...).
RUNTIME_ERROR = "runtime_error"
#: The run went quiet with the entry not finished: every thread is blocked.
UNFINISHED = "unfinished"

#: The run finished, but the bench reported a failure (``VM.bench_report``
#: with an error or fatal) while it ran (F2.7).
BENCH_FAIL = "bench_fail"

VERDICTS = (OK, SOLVE_FAIL, RUNTIME_ERROR, UNFINISHED, BENCH_FAIL)

_SOLVE_ERRORS = (SolveUnsatError, SolveBudgetError, ScopeUnsatError)


@runtime_checkable
class Platform(Protocol):
    """What a run's target answers. Both are optional (None: the VM's own
    sparse ``extern.Memory``; no user imports).

    ``memory`` has ``read(addr, size)`` and ``write(addr, size, value)``
    (``extern.Memory``); ``imports`` has ``call(fn_id, args)``
    (``extern.ImportProvider``).

    A platform may also have any of these (looked up with ``getattr``); any
    of the first four runs every entry over the host ABI (``host_abi``):

    * ``wait_time(host, delay)`` -- every thread waits on time: the
      platform's clock moves ``delay`` engine units (it may
      ``host.notify(kind, id, value)``); without it time runs on at once.
    * ``wait_event(host) -> bool`` -- every thread waits on an event: the
      platform may notify and returns True to go on; False is a deadlock.
    * ``on_notify(vm, kind, id, value)`` -- takes a notification
      (``HostCall`` ``on_notify``).
    * ``irq_entry`` -- the entry (symbol or index) an ``N_IRQ`` notification
      starts as a thread, when there is no ``on_notify``.
    * ``observe(event)`` -- sees every debug event the run raises of the
      kinds in ``watch`` (``trace.events`` mask, default ``EV_PLATFORM``:
      accesses, imports, notifications)."""

    memory: Any
    imports: Optional[ImportProvider]


@dc.dataclass
class SimplePlatform:
    """A :class:`Platform` from its two parts."""

    memory: Any = None
    imports: Optional[ImportProvider] = None


@dc.dataclass
class RunOpts:
    """What a run may be given besides its image, entry, platform and seed.

    ``args`` are a function entry's arguments, in parameter order, or an
    exported action's (each pins the field its parameter names). ``setup``
    and ``teardown`` are function entries of the root component (symbols or
    coroutine indices), run in order on the same tree: setup after it is
    constructed, teardown after the entry whatever it did. ``bindings``
    (``{"<comp type>::<fn>": callable}``) run host callables in place of
    those functions (``VM.bind``). ``field_names`` (or ``obj``) give the
    entry's action object; by default it comes from the entry's layout.
    ``name`` names the run in the ``test.start``/``test.end`` host events
    (raised only when ``name`` is set; they reach ``events`` when ``watch``
    has ``EV_TEST``). ``host_abi`` runs each entry over the host ABI
    (``HostCall`` + ``run_sync``, as a simulator host does) instead of
    ``VM._drain``. ``on_root(frame)`` sees the entry's frame once it is the
    VM's entry, before anything runs (a debugger reads ``vm.solved`` from
    then on).

    The rest are :func:`oracle.run_model`'s: ``solve_backend`` (default:
    the VM's, ``FixedSolveBackend``; a PSS image wants
    ``interp.NativeBlobBackend()``), ``solve_cache``/``max_restarts``,
    ``solve_bounds``, ``seed_mode``, ``on_solve``, ``out``, ``verbosity``, ``sink``,
    ``events`` and ``watch``. ``coverage`` watches statements and branch
    arms and counts them in ``RunResult.coverage`` (a debug-lowered image
    has them)."""

    args: Sequence[int] = ()
    setup: Sequence[Union[int, str]] = ()
    teardown: Sequence[Union[int, str]] = ()
    bindings: Optional[Dict[str, Callable]] = None
    obj: Optional[Obj] = None
    field_names: Sequence[str] = ()
    name: str = ""
    host_abi: bool = False
    on_root: Optional[Callable[[Any], None]] = None
    solve_backend: Optional[SolveBackend] = None
    solve_cache: Optional[SolveCache] = None
    max_restarts: Optional[int] = None
    solve_bounds: Optional[Dict[str, tuple]] = None
    on_solve: Optional[Callable[[Any], None]] = None
    out: Optional[Callable[[str], None]] = None
    #: the heap's byte cap (heap.md section 5; "heap exhausted" past it)
    heap_cap: int = HEAP_CAP
    verbosity: int = VERBOSITY_MEDIUM
    sink: Optional[TraceSink] = None
    events: Optional[TraceSink] = None
    watch: int = 0
    coverage: bool = False
    seed_mode: str = "keyed"
    #: M8b6: an ``interp.activation.PlanLog`` the run records its realised
    #: plan in (what ran, its values, each select's branch); None: none
    plan: Optional[Any] = None
    #: M8d5: stim ids (``Frame.stim``) whose action runs have no effects:
    #: still solved and scheduled, but no import or memory access and no
    #: counted failed check (``pssc test --drop``/``--reduce``)
    drop: Sequence[str] = ()


@dc.dataclass
class RunResult(oracle.RunResult):
    """:class:`oracle.RunResult` plus what a host needs to judge the run:
    the component object the run left behind (where a ledger's counts
    live), the verdict (one of :data:`VERDICTS`) and, unless it is
    :data:`OK`, the error that ended the run. ``events`` is the
    orchestration trace; ``debug_events`` the debug events the run raised
    (when its events sink keeps them: :func:`run` makes one if none is
    given); ``coverage`` (``RunOpts.coverage``) the statement and branch-arm
    hits, ``{"stmts": {id: hits}, "arms": {id: hits}}`` with IDs
    ``<scope>/<path>`` (an arm: ``<scope>/<branch>/<arm>``); ``bench`` the
    bench reports (``VM.bench_report``), attributed to actions."""

    cobj: Optional[Obj] = None
    verdict: str = OK
    error: Optional[BaseException] = None
    debug_events: list = dc.field(default_factory=list)
    coverage: Optional[Dict[str, Dict[str, int]]] = None
    bench: list = dc.field(default_factory=list)
    #: (covg id, register values) of each COV_SAMPLE (F4.4)
    cov_samples: list = dc.field(default_factory=list)
    #: the samples binned against the image's COVG docs (F4.5,
    #: ``cov.covergroup.bin_samples``); None if the image has no covergroup
    covergroups: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.verdict == OK

    def check(self) -> "RunResult":
        """Raise the run's error unless it finished; else return self."""
        if self.error is not None:
            raise self.error
        return self


def load(image: Union[bytes, bytearray, str, os.PathLike, ZbcModel]) -> ZbcModel:
    """A model from ``.zbc`` bytes, the path of a ``.zbc`` file, or a model
    (returned as it is)."""
    if isinstance(image, ZbcModel):
        return image
    if isinstance(image, (bytes, bytearray)):
        return ZbcModel.from_bytes(bytes(image))
    with open(image, "rb") as f:
        return ZbcModel.from_bytes(f.read())


def resolve_entry(model: ZbcModel, entry: Union[int, str, None]) -> int:
    """The coroutine index of ``entry``: an index as it is; a symbol from
    the image's entry table (an action by name or qualified name, a
    function by ``<comp type>::<fn>``), else a coroutine of that name;
    None: the image's own entry."""
    if entry is None:
        return model.entry_coro
    if isinstance(entry, int):
        if not 0 <= entry < len(model.coros):
            raise ValueError(f"entry {entry}: the image has "
                             f"{len(model.coros)} coroutines")
        return entry
    table = getattr(model, "entries", None) or {}
    if entry in table:
        return table[entry].coro
    for e in table.values():
        if e.kind == "action" and getattr(e, "qname", None) == entry:
            return e.coro
    for i, c in enumerate(model.coros):
        if c.name == entry:
            return i
    raise ValueError(f"no entry '{entry}' in the image; it has: "
                     + ", ".join(sorted(table) or [c.name for c in model.coros]))


def _verdict(e: BaseException) -> str:
    return SOLVE_FAIL if isinstance(e, _SOLVE_ERRORS) else RUNTIME_ERROR


def run(image: Union[bytes, str, os.PathLike, ZbcModel],
        entry: Union[int, str, None] = None,
        platform: Optional[Platform] = None, seed: int = 0,
        opts: Optional[RunOpts] = None) -> RunResult:
    """Run ``entry`` of ``image`` on ``platform`` with ``seed``.

    A run that fails in the image (any ``VMError``: a solve failure, an
    error the image raises, a deadlock) does not raise: the result says so
    in ``verdict`` and ``error`` (``RunResult.check`` raises it). Anything
    else (a bad entry, an exception from the host) does raise.
    """
    opts = opts if opts is not None else RunOpts()
    model = load(image)
    coro = resolve_entry(model, entry)
    sink = opts.sink if opts.sink is not None else MemorySink()
    cache = opts.solve_cache
    own = cache is None
    if own:
        cache = SolveCache(**({} if opts.max_restarts is None
                              else {"max_restarts": opts.max_restarts}))
    elif opts.max_restarts is not None:
        raise ValueError("max_restarts belongs to the solve_cache passed in")
    watch = opts.watch
    sinks = []
    if opts.coverage:
        sinks.append(_CovSink())
        watch |= ev.EV_STMT | ev.EV_BRANCH
    observe = getattr(platform, "observe", None)
    if observe is not None:
        sinks.append(_FnSink(observe))
        watch |= getattr(platform, "watch", ev.EV_PLATFORM)
    events = opts.events
    if events is None and watch:
        events = MemorySink()
    if sinks:
        events = _Tee([events] + sinks)
    vm = VM(model, solve_backend=opts.solve_backend,
            import_provider=getattr(platform, "imports", None), sink=sink,
            out=opts.out, verbosity=opts.verbosity,
            memory=getattr(platform, "memory", None), solve_cache=cache,
            on_solve=opts.on_solve, events=events)
    if watch:
        vm.watch(watch)
    vm.solve_bounds = opts.solve_bounds
    vm.seed_mode = opts.seed_mode
    vm.plan = opts.plan
    vm.drop = frozenset(opts.drop)
    vm.heap.cap = opts.heap_cap
    try:
        return run_on(vm, coro, seed, opts, platform)
    finally:
        if own:
            cache.clear()


def run_on(vm: VM, entry: Union[int, str, None] = None, seed: int = 0,
           opts: Optional[RunOpts] = None,
           platform: Optional[Platform] = None) -> RunResult:
    """:func:`run` on a VM the host built (a debugger holds it, or the
    host's platform made it): ``entry`` on a fresh tree of ``vm``, between
    ``opts.setup`` and ``opts.teardown``. Of ``opts`` only what is per run
    applies: ``args``, ``setup``, ``teardown``, ``bindings``, ``obj``,
    ``field_names``, ``name``, ``host_abi`` and ``on_root``; the rest
    (backend, cache, sinks, watch, bounds) are the VM's already; so are
    ``platform``'s memory and imports: only its time and notification
    hooks are used here."""
    opts = opts if opts is not None else RunOpts()
    model = vm.model
    coro = resolve_entry(model, entry)
    setup = [resolve_entry(model, e) for e in opts.setup]
    teardown = [resolve_entry(model, e) for e in opts.teardown]
    hooks = {k: getattr(platform, k, None) for k in _HOOKS}
    if opts.host_abi or any(v is not None for v in hooks.values()):
        if hooks["irq_entry"] is not None:
            hooks["irq_entry"] = resolve_entry(model, hooks["irq_entry"])
        drain = functools.partial(_host_drain, seed=seed, **hooks)
    else:
        drain = _vm_drain
    for qname, fn in (opts.bindings or {}).items():
        vm.bind(qname, fn)
    obj = opts.obj
    if obj is None and opts.field_names:
        # the image's slot layout (struct attributes flattened) when it has one
        obj = vm._new_obj(coro) or Obj(field_names=list(opts.field_names))
    if opts.name:
        vm.host_event("test.start", {"name": opts.name, "seed": seed})
    root = error = None
    try:
        try:
            # an exported action's arguments pin its parameters' fields
            exported = any(e.coro == coro and e.kind == "action" and e.params
                           for e in (getattr(model, "entries", None) or {}).values())
            root = vm.root_frame(coro, seed=seed, obj=obj,
                                 args=opts.args if exported else ())
            root.args = [] if exported else list(opts.args)
            vm.set_entry(root)
            if opts.on_root is not None:
                opts.on_root(root)
            if root.cobj is not None:
                vm.construct_components()
            for c in setup:
                _call_root(vm, c, seed, drain)
            drain(vm, root)
        except VMError as e:
            error = e
        for c in teardown:
            try:
                _call_root(vm, c, seed, drain)
            except VMError as e:
                # a teardown that fails after the entry failed does not
                # replace the entry's error
                error = error or e
    finally:
        if opts.name:
            vm.host_event("test.end", {"name": opts.name,
                                       "done": error is None and root is not None
                                       and root.done})
    if error is None and not root.done:
        error = VMError("the run ended with the entry unfinished: every "
                        "thread left is blocked")
        verdict = UNFINISHED
    else:
        verdict = OK if error is None else _verdict(error)
    bench = list(getattr(vm, "bench_reports", ()))
    if verdict == OK and any(b["verdict"] == "fail" for b in bench):
        verdict = BENCH_FAIL
    covergroups = _covergroups(vm)
    if verdict == OK and covergroups:
        from zuspec.be.bc.cov.covergroup import illegal_hits
        bad = illegal_hits(vm.model.covgs, covergroups)
        if bad:
            error, verdict = IllegalBinError(bad), RUNTIME_ERROR
    return RunResult(
        obj=root.obj if root is not None else None,
        retval=root.retval if root is not None else None,
        events=list(getattr(vm.sink, "events", [])),
        now=vm.sched.now,
        frames=vm.sched._next_id,
        solved=dict(vm.solved),
        cobj=getattr(vm, "cobj", None),
        verdict=verdict,
        error=error,
        debug_events=list(getattr(_kept(vm.events), "events", [])),
        coverage=_coverage(vm.events),
        bench=bench,
        cov_samples=list(getattr(vm, "cov_samples", ())),
        covergroups=covergroups,
    )


_HOOKS = ("wait_time", "wait_event", "on_notify", "irq_entry")


class _Tee(TraceSink):
    """Passes each event, sequenced once, to every sink."""

    def __init__(self, sinks):
        super().__init__()
        self.sinks = [s for s in sinks if s is not None]

    def emit(self, event) -> None:
        for s in self.sinks:
            s.emit(event)


class _FnSink(TraceSink):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def emit(self, event) -> None:
        self.fn(event)


class _CovSink(TraceSink):
    """Statement and branch-arm hits by point ID (``<scope>/<path>``, as
    pssc's coverage report names them)."""

    def __init__(self):
        super().__init__()
        self.stmts: Dict[str, int] = {}
        self.arms: Dict[str, int] = {}

    def emit(self, event) -> None:
        d = event.detail
        if event.kind == ev.STMT:
            table, path = self.stmts, d.get("path")
        elif event.kind == ev.BRANCH:
            table = self.arms
            path = d.get("branch", "") + "/" + str(d.get("arm", ""))
        else:
            return
        if path is None:                # lowered without coverage
            return
        scope = d.get("scope", "")
        pid = f"{scope}/{path}" if scope else path
        table[pid] = table.get(pid, 0) + 1


def _kept(sink):
    """The sink that keeps the run's debug events (the first of a tee)."""
    return sink.sinks[0] if isinstance(sink, _Tee) and sink.sinks else sink


class IllegalBinError(VMError):
    """A run that hit an illegal covergroup bin (LRM 19.3.4): reported at
    the first such bin's declaration; ``hits`` lists them all
    (``cov.covergroup.illegal_hits``)."""

    def __init__(self, hits) -> None:
        super().__init__("; ".join(h["message"] for h in hits))
        self.hits = list(hits)
        h = hits[0]
        if h.get("line"):
            from .model import Prov
            self.prov = Prov(name=h["bin"], file=h.get("file", ""), line=h["line"])


def _covergroups(vm) -> Optional[Dict[str, Any]]:
    covgs = getattr(vm.model, "covgs", None)
    if not covgs:
        return None
    from zuspec.be.bc.cov.covergroup import bin_samples
    return bin_samples(covgs, getattr(vm, "cov_samples", ()))


def _coverage(sink):
    for s in getattr(sink, "sinks", ()):
        if isinstance(s, _CovSink):
            return {"stmts": dict(s.stmts), "arms": dict(s.arms)}
    return None


def _vm_drain(vm: VM, frame) -> None:
    vm._drain(frame)


def _host_drain(vm: VM, frame, seed: int = 0, wait_time=None,
                wait_event=None, on_notify=None, irq_entry=None) -> None:
    """``VM._drain`` over the host ABI: one ``HostCall`` on the tree as it
    stands, served by ``run_sync`` from the VM's own memory and imports and
    the platform's time and notification hooks (:class:`Platform`). A
    frame left blocked returns normally; the caller checks ``frame.done``."""
    from .interp.host import HostCall, run_sync

    call = HostCall(vm, frame, on_notify=on_notify, irq_entry=irq_entry,
                    seed=seed, construct=False)
    try:
        run_sync(call,
                 wait_time=None if wait_time is None
                 else (lambda d: wait_time(call, d)),
                 wait_event=None if wait_event is None
                 else (lambda: wait_event(call)))
    finally:
        call.end()


def _call_root(vm: VM, coro: int, seed: int, drain=_vm_drain) -> None:
    """Run function entry ``coro`` in the root component, to completion."""
    frame = vm.sched.new_frame(vm.model.coros[coro], seed=seed,
                               obj=vm._new_obj(coro))
    frame.cobj = getattr(vm, "cobj", None)
    vm.set_comp(frame, 0)
    drain(vm, frame)
    if not frame.done:
        raise VMError(f"{vm.model.coros[coro].name} did not finish: every "
                      f"thread left is blocked")
