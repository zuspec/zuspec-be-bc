"""The host ABI seam (``docs/spec/host-abi.md``, P3.1b).

A :class:`HostCall` runs every ready thread of one call of a
:class:`~.vm.VM` until none is ready. A thread that calls an API (a memory
access, a user import) is suspended with a :class:`Request` in the call's
outstanding set while the others run on; ``run`` then says ``REQS``, the host
takes each new request with :meth:`HostCall.take` and answers each, in any
order, with :meth:`HostCall.complete`::

    call = HostCall(vm, vm.root_frame(coro, seed))
    while (r := call.run()).status != DONE:
        if r.status == REQS:
            while (q := call.take()) is not None:
                start(q)          # later: call.complete(q.req_id, value)
        ...

:func:`run_sync` is that loop with the VM's own ``Memory`` and
``ImportProvider`` as the answers: a run through it is the same as
``VM.run``.

Host ABI version 2 (host-abi.md "Run status and requests"). The call runs
on a thread of its own that blocks at each yield, so the
interpreter keeps its plain call stack (the debug stub does the same,
``debug.py``). Only one of the two threads runs at a time.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .ops_proc import VMError

#: run status (``pssvm_run``; value 2, the old IMPORT, is unused)
DONE, REQS, WAIT, STOP, ERROR = 0, 1, 3, 4, -1
STATUS_NAMES = {DONE: "DONE", REQS: "REQS", WAIT: "WAIT", STOP: "STOP",
                ERROR: "ERROR"}

#: built-in API ids (host-abi.md "The import interface"); user APIs are >= 16
API_MEM, API_CONFIG, API_SEQ, API_USER = 0, 1, 2, 16
#: ``mem`` functions
FN_READ, FN_WRITE = 0, 1
#: no executor (``Request.executor``)
NO_EXECUTOR = 0xFFFFFFFF

#: what ``access``/``import_`` return when the thread is suspended on its
#: request (``ops_orch`` re-runs the instruction once it is completed)
PARKED = object()

#: notify kinds (``pssvm_notify``)
N_IRQ, N_CHAN = 1, 2


class HostStateError(RuntimeError):
    """A call made in the wrong state (``PSSVM_E_STATE``)."""


@dataclass
class Request:
    """What :meth:`HostCall.run` returned (``status`` and, for WAIT, ``on``
    and ``delay``), or a request :meth:`HostCall.take` gave (``req_id`` and
    the fields of its API)."""
    status: int
    #: a request: its id, API and function, engine thread and executor
    req_id: int = 0
    api: int = API_MEM
    fn: int = 0
    thread: int = 0
    executor: int = NO_EXECUTOR
    #: API mem: "r" or "w", byte address, size in bytes, write data, register
    kind: str = ""
    addr: int = 0
    size: int = 0
    data: Optional[int] = None
    reg_id: Optional[int] = None
    src_ref: int = 0
    #: the stimulus id of the action run that made it
    stim_id: str = ""
    #: user APIs (``fn`` is the import's fn_id)
    name: Optional[str] = None
    args: List[int] = field(default_factory=list)
    blocking: bool = True
    #: a struct result's field count (its answer is that many values); 0
    #: for a scalar or no result
    nret: int = 0
    #: WAIT: "TIME" (with ``delay`` engine units), "REQS" or "EVENT"
    on: str = ""
    delay: int = 0
    #: ERROR: the ``pssvm_error`` fields, and the exception itself
    error: Optional[dict] = None
    exc: Optional[BaseException] = None

    @property
    def status_name(self) -> str:
        return STATUS_NAMES[self.status]

    @property
    def fn_id(self) -> int:
        return self.fn


class _Cancelled(BaseException):
    """Unwinds the VM thread of a call ended before it was DONE."""


_RESUME = object()
_CANCEL = object()


class _Raise:
    """An answer that raises ``exc`` in the VM thread (:meth:`HostCall.fail`)."""

    def __init__(self, exc: BaseException):
        self.exc = exc


class HostCall:
    """One call run against a host. ``root`` is the call's frame
    (``VM.root_frame``); a root with a component tree is constructed first,
    as ``VM.run`` does, and construction must not yield.

    ``on_notify(vm, kind, id, value)`` takes a :meth:`notify` on the VM
    thread; without one, ``N_IRQ`` starts ``irq_entry`` (a coroutine index)
    as a thread of its own in the root component, as pssc's
    ``Ctx.raise_irq`` does, and any other kind is an error.

    ``construct=False`` runs ``root`` on the tree as it stands: a function
    called on a tree built before (no construction, even with a ``cobj``).

    Several calls may be live on one VM (a host thread per call, e.g. an
    SV task blocked in a transfer while an irq hook calls ``notify_irq``).
    The first runs on the VM's scheduler; one started while another is
    live gets a ready queue of its own, so each call runs only its own
    threads (they share the component tree: a channel put by one wakes the
    other at its next :meth:`run`)."""

    def __init__(self, vm, root, on_notify: Optional[Callable] = None,
                 irq_entry: Optional[int] = None, seed: int = 0,
                 construct: bool = True):
        self.vm = vm
        self.construct = construct
        self.root = root
        self.on_notify = on_notify
        self.irq_entry = irq_entry
        self.seed = seed
        self._to_vm: queue.Queue = queue.Queue()
        self._to_host: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._next_id = 0
        self._new: List[Request] = []       # issued, not taken
        self._out: dict = {}                # req_id -> (Request, frame)
        self._answers: dict = {}            # id(frame) -> completed answer
        self._strs: dict = {}               # req_id -> {leaf: text} (ret_string)
        self._woken: List[Any] = []         # frames completed, in order
        #: (scheduler, its ready length) when the call last yielded: a thread
        #: readied while the host served (an interrupt raised by an import)
        #: runs after the completed requests' threads resume
        self._mark = None
        self._time: Optional[int] = None    # the host's clock (set_time)
        self._clocked = False
        self._last: Optional[Request] = None
        self._notes: List[tuple] = []
        self._lock = threading.Lock()
        self._constructing = False
        self.over = False
        self.sched = None          # set when the call starts (_main)

    # -- the host's side -------------------------------------------------- #

    def run(self) -> Request:
        """Run the call until it needs the host; the request says why."""
        if self.over:
            raise HostStateError("the call is over")
        if self._thread is None:
            self._thread = threading.Thread(target=self._main, daemon=True,
                                            name="pssvm-call")
            self._thread.start()
        else:
            self._to_vm.put(_RESUME)
        r = self._to_host.get()
        self._last = r
        if r.status in (DONE, ERROR):
            self.over = True
            self._thread.join()
        return r

    def take(self) -> Optional[Request]:
        """The next new request, in issue order (None: none)."""
        return self._new.pop(0) if self._new else None

    def complete(self, req_id: int, value: Optional[int] = None) -> None:
        """Answer outstanding request ``req_id``: the read value, the
        import's result (None: none), nothing for a write. Only its thread is
        readied; one of a cancelled thread is accepted and discarded."""
        r, frame = self._req(req_id)
        if r.api == API_MEM:
            if r.kind == "r":
                if value is None:
                    raise HostStateError("a read completes with its value")
                value = int(value) & ((1 << (8 * r.size)) - 1)
            elif value is not None:
                raise HostStateError("a write completes with no value")
        strs = self._strs.pop(req_id, None)
        if strs:
            if r.nret:
                vals = list(value) if isinstance(value, (list, tuple)) else [0] * r.nret
                vals += [0] * (r.nret - len(vals))
                for i, text in strs.items():
                    vals[i] = text
                value = vals
            else:
                value = strs[0]
        self._answer(req_id, frame, value)

    def ret_string(self, req_id: int, text, i: int = 0) -> None:
        """Stage the text of string result leaf ``i`` (0 for a scalar) of
        import request ``req_id``, before :meth:`complete`; the engine makes
        a heap string of it (``pssvm_string_ret``; host-abi.md "String
        results"). ``complete(req_id, text)`` with a ``str`` does the same
        for a scalar."""
        r, _ = self._req(req_id)
        if r.api == API_MEM or not 0 <= i < max(r.nret, 1):
            raise HostStateError(f"request {req_id} has no string result {i}")
        self._strs.setdefault(req_id, {})[i] = (
            text if isinstance(text, (str, bytes, bytearray)) else str(text))

    def fail(self, req_id: int, exc: BaseException) -> None:
        """Answer request ``req_id`` with an error: ``exc`` is raised where
        the VM made it, as a platform callback's own error is when the VM
        calls it directly."""
        _, frame = self._req(req_id)
        self._strs.pop(req_id, None)
        self._answer(req_id, frame, _Raise(exc))

    def set_time(self, now: int) -> None:
        """The host's clock, in ticks, before a :meth:`run`: timers due by
        ``now`` fire. Never backwards."""
        cur = self._time if self._time is not None else 0
        if self.sched is not None:
            cur = max(cur, self.sched.now)
        if now < cur:
            raise HostStateError(f"set_time({now}) is before {cur}")
        self._time = now
        self._clocked = True

    def _req(self, req_id: int):
        try:
            return self._out[req_id]
        except KeyError:
            raise HostStateError(f"request {req_id} is not outstanding") from None

    def _answer(self, req_id: int, frame, value) -> None:
        del self._out[req_id]
        if frame.done or frame.cancelled:
            return
        self._answers[id(frame)] = value
        self._woken.append(frame)

    def notify(self, kind: int, id: int = 0, value: int = 0) -> None:
        """A host event; it takes effect at the next :meth:`run`. ``N_CHAN``
        for an id with no EVENT row is refused at once (``PSSVM_E_ARGS``),
        unless an ``on_notify`` hook takes every notification."""
        if (kind == N_CHAN and self.on_notify is None
                and not (0 <= id < len(self._events()))):
            raise HostStateError(f"no host-fed channel {id}")
        with self._lock:
            self._notes.append((kind, id, value))

    @property
    def request(self) -> Optional[Request]:
        """The last request :meth:`run` returned."""
        return self._last

    @property
    def result(self):
        """The entry's return value after DONE (None: none)."""
        if self._last is None or self._last.status != DONE:
            raise HostStateError("the call is not DONE")
        return self.root.retval

    def string(self, idx: int) -> Optional[str]:
        """The text of model string ``idx`` (an import's string argument
        arrives as this index; host-abi.md "String arguments"), or None out
        of range (``pssvm_string``)."""
        strings = getattr(self.vm.model, "strings", None) or []
        return strings[idx] if 0 <= idx < len(strings) else None

    def end(self) -> None:
        """Free the call; one not DONE is cancelled (its threads never run
        again)."""
        if self._thread is not None and not self.over:
            self._to_vm.put(_CANCEL)
            self._thread.join()
        self.over = True
        self._new.clear()
        self._out.clear()
        self._answers.clear()
        self._woken.clear()
        if self.vm.host is self:
            self.vm.host = None
        self._leave(done=True)

    # -- the VM's side (on the call's thread) ----------------------------- #

    def _start(self) -> None:
        """Pick the call's scheduler: the VM's own if no other call is live,
        else a fresh one sharing the VM's clock and frame ids."""
        vm = self.vm
        home = vm.__dict__.setdefault("_home_sched", vm.sched)
        live = vm.__dict__.setdefault("_live_calls", [])
        if live:
            s = type(home)()
            s.now = home.now
            s.new_frame = home.new_frame
            self.sched = s
        else:
            self.sched = home
        live.append(self)
        self._enter()

    def _enter(self) -> None:
        """The call's thread runs: the VM is this call's."""
        vm = self.vm
        vm.host = self
        if self.sched is not vm._home_sched:
            self.sched.now = max(self.sched.now, vm._home_sched.now)
        if self._time is not None:
            self.sched.now = max(self.sched.now, self._time)
        vm.sched = self.sched
        woken, self._woken = self._woken, []
        ready = self.sched._ready
        at = (self._mark[1] if self._mark is not None
              and self._mark[0] is self.sched else len(ready))
        self._mark = None
        for f in woken:               # completion order (determinism.md)
            if not (f.done or f.cancelled):
                ready.insert(min(at, len(ready)), f)
                at += 1

    def _leave(self, done: bool = False) -> None:
        """The call's thread stops (a yield, or the end): the VM's own
        scheduler is back for the host (and for the next call)."""
        vm = self.vm
        home = getattr(vm, "_home_sched", None)
        if home is None or self.sched is None:
            return
        home.now = max(home.now, self.sched.now)
        vm.sched = home
        self._mark = (home, len(home._ready))
        if done and self in vm._live_calls:
            vm._live_calls.remove(self)

    def _main(self) -> None:
        vm = self.vm
        self._start()
        try:
            self._take_notes()
            if self.construct and self.root.cobj is not None:
                self._constructing = True
                try:
                    vm.construct_components()
                finally:
                    self._constructing = False
            vm._drain(self.root)
            r = Request(DONE)
        except _Cancelled:
            vm.host = None
            return
        except BaseException as e:   # an engine error never takes the host down
            r = Request(ERROR, error=self._error(e), exc=e)
        vm.host = None
        self._leave(done=True)
        self._to_host.put(r)

    def _error(self, e: BaseException) -> dict:
        p = getattr(e, "prov", None)
        return {"message": str(e), "src_ref": getattr(e, "src_ref", 0),
                "file": getattr(p, "file", None) if p else None,
                "line": getattr(p, "line", None) if p else None,
                "col": getattr(p, "col", None) if p else None,
                "call": self.root.coro.name, "thread": None}

    def _yield(self, r: Request):
        self._leave()
        self._to_host.put(r)
        a = self._to_vm.get()
        if a is _CANCEL:
            raise _Cancelled()
        self._enter()
        self._take_notes()
        if isinstance(a, _Raise):
            raise a.exc
        return a

    def _take_notes(self) -> None:
        with self._lock:
            notes, self._notes = self._notes, []
        for kind, id_, value in notes:
            self._ev_notify(kind, id_, value)
            if self.on_notify is not None:
                self.on_notify(self.vm, kind, id_, value)
            elif kind == N_IRQ and self.irq_entry is not None:
                self._irq()
            elif self._chan_ok(kind, id_):
                self._chan_put(id_, value)
            else:
                raise VMError(f"notify kind {kind} has no handler")

    def _ev_notify(self, kind: int, id_: int, value: int) -> None:
        """The ``notify`` event of a notification taken: the platform side a
        replay feeds back (the native engine raises the same)."""
        delivered = self.on_notify is not None or (
            kind == N_IRQ and self.irq_entry is not None) or self._chan_ok(kind, id_)
        key = {N_IRQ: "irq", N_CHAN: "chan"}.get(kind)
        detail = {key: id_} if key else {"kind": kind}
        detail.update(value=value, delivered=delivered)
        self.vm.host_event("notify", detail)

    def _events(self) -> list:
        return getattr(self.vm.model, "events", None) or []

    def _chan_ok(self, kind: int, id_: int) -> bool:
        return (kind == N_CHAN and getattr(self.vm, "cobj", None) is not None
                and 0 <= id_ < len(self._events()))

    def _chan_put(self, id_: int, value: int) -> None:
        """A host put on host-fed channel ``id_`` (its EVENT row): try_put,
        the ring as lower/procedural.py ``_chan_put`` has it; spinning
        get()s see it when they re-test."""
        ev = self._events()[id_]
        c = self.vm.cobj
        n, h, depth = c.get_field(ev["count"]), c.get_field(ev["head"]), ev["depth"]
        if n >= depth:
            raise VMError(f"host put on full channel {ev['path']}")
        w = ev.get("width", 0)
        if 0 < w < 64:
            value &= (1 << w) - 1
            if ev.get("signed") and value >> (w - 1):
                value -= 1 << w
        c.set_field(ev["buf"] + (h + n) % depth, value)
        c.set_field(ev["count"], n + 1)

    def _irq(self) -> None:
        vm = self.vm
        frame = vm.sched.new_frame(vm.model.coros[self.irq_entry], seed=self.seed,
                                   obj=vm._new_obj(self.irq_entry))
        frame.cobj = getattr(vm, "cobj", None)
        if frame.cobj is not None:
            vm.set_comp(frame, 0)
        vm.sched.ready(frame)

    def _no_yield(self, what: str) -> None:
        if self._constructing:
            raise VMError(f"construct-yield: {what} during construction")

    def _issue(self, frame, r: Request):
        """Suspend ``frame`` on request ``r``; or, re-run once ``r`` is
        completed, its answer."""
        a = self._answers.pop(id(frame), PARKED)
        if a is not PARKED:
            if isinstance(a, _Raise):
                raise a.exc
            return a
        self._no_yield("an access" if r.api == API_MEM else "an import")
        ids = self._out
        rid = self._next_id
        while rid in ids:                 # reused after completion
            rid = (rid + 1) & 0xFFFFFFFF
        self._next_id = (rid + 1) & 0xFFFFFFFF
        r.req_id = rid
        r.thread = frame.id
        r.executor = getattr(frame, "executor", NO_EXECUTOR)
        r.stim_id = getattr(frame, "stim", "") or ""
        ids[rid] = (r, frame)
        self._new.append(r)
        return PARKED

    def access(self, frame, ins, kind: str, addr: int, size: int,
               data: Optional[int] = None):
        if id(frame) in self._answers:
            return self._issue(frame, None)
        return self._issue(frame, Request(
            REQS, api=API_MEM, fn=FN_READ if kind == "r" else FN_WRITE,
            kind=kind, addr=addr, size=size, data=data,
            reg_id=self.vm.reg_id(frame),
            src_ref=ins.src_ref if ins is not None else 0))

    def import_(self, frame, ins, fn_id: int, args, blocking: bool):
        if id(frame) in self._answers:
            return self._issue(frame, None)
        return self._issue(frame, Request(
            REQS, api=API_USER, fn=fn_id, name=self.vm.import_name(fn_id),
            args=list(args), blocking=blocking, src_ref=ins.src_ref,
            nret=(ins.imm >> 8) & 0xFF))

    def answered(self, frame) -> bool:
        """``frame`` is parked on a request the host has completed."""
        return id(frame) in self._answers

    def flush(self) -> bool:
        """Nothing ready: hand new requests to the host (``REQS``). False:
        none."""
        if not self._new:
            return False
        self._yield(Request(REQS))
        return True

    def live(self) -> bool:
        """A thread still waits on an outstanding request."""
        return any(not (f.done or f.cancelled) for _, f in self._out.values())

    def wait_reqs(self) -> None:
        self._yield(Request(WAIT, on="REQS"))

    def wait_time(self, delay: int) -> bool:
        """WAIT TIME; True when the host set its clock meanwhile (time then
        runs only as far as it says)."""
        self._clocked = False
        self._yield(Request(WAIT, on="TIME", delay=delay))
        return self._clocked

    def wait_event(self) -> None:
        self._yield(Request(WAIT, on="EVENT"))


def _note(mem, q: Request) -> bool:
    """Tell a memory that asks (a register fake) which register access ``q``
    is for, from the request itself: several may be out at once."""
    note = getattr(mem, "note_access", None)
    if note is not None:
        note(q.reg_id, q.src_ref)
    return True


def run_sync(call: HostCall, memory=None, import_provider=None,
             wait_time: Optional[Callable[[int], None]] = None,
             wait_event: Optional[Callable[[], bool]] = None):
    """Run ``call`` to DONE, answering from ``memory`` and ``import_provider``
    (default: the VM's own): at each REQS it takes every request and
    completes them in issue order, non-blocking ones first (one thread: the
    old order). A WAIT TIME calls ``wait_time(delay)`` (the host's clock; it
    may :meth:`HostCall.notify`), then runs on; without one it runs on at
    once. A WAIT EVENT calls ``wait_event()``, which may notify and returns
    True to run on; without one, or when it returns False, it is the
    ``deadlock`` error.
    Returns the entry's return value; an engine error is re-raised. An
    error in ``memory`` or ``import_provider`` is raised in the VM thread
    (:meth:`HostCall.fail`), so the VM sees it as it would a direct call."""
    vm = call.vm
    while True:
        r = call.run()
        if r.status == DONE:
            return call.result
        if r.status == REQS:
            reqs = []
            while (q := call.take()) is not None:
                reqs.append(q)
            reqs.sort(key=lambda q: q.blocking)      # stable: issue order
            for q in reqs:
                # the VM's own, looked up per request: a debugger may swap them
                mem = memory if memory is not None else vm.memory
                provider = (import_provider if import_provider is not None
                            else vm.import_provider)
                try:
                    if q.api != API_MEM:
                        # the requesting thread, for a provider keeping
                        # per-thread state across imports
                        vm.serving_thread = q.thread
                        try:
                            v = provider.call(q.fn, q.args)
                        finally:
                            vm.serving_thread = None
                    elif _note(mem, q) and q.kind == "r":
                        v = mem.read(q.addr, q.size)
                    else:
                        mem.write(q.addr, q.size, q.data)
                        v = None
                except BaseException as e:
                    call.fail(q.req_id, e)
                    continue
                call.complete(q.req_id, v)
        elif r.status == WAIT and r.on != "REQS":
            if r.on == "TIME" and wait_time is not None:
                wait_time(r.delay)
            elif r.on == "EVENT" and not (wait_event is not None
                                          and wait_event()):
                call.end()
                raise VMError("deadlock: every running thread waits on a "
                              "channel that nothing can put to or get from")
        elif r.status == ERROR:
            raise r.exc
        else:
            call.end()
            raise HostStateError(f"run_sync cannot serve {r.status_name}"
                                 + (f" {r.on}" if r.on else ""))
