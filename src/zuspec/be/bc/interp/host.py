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

Several calls may be live on one VM (host-abi.md "Concurrent calls"):
at most one action **traversal** is active, a second one waits in a FIFO
queue (``run`` says ``WAIT TURN``) until it ends, and any number of
exported **function** calls run alongside it. An interrupt is such a
function call (:meth:`CallTable.start_function`). :class:`CallTable` keeps
them; :func:`run_calls` drives several from one host thread, and finds a
deadlock across them; :meth:`CallTable.step` runs every call that can run
once, for a host with a loop of its own (``pssvm_step``/``pssvm_pending``).

The engine's host state is the :class:`CallTable` (host-abi.md "Objects",
"Linking", "Listeners"): a handle table (``new``/``release``: the root
component, its instances, action objects, and ``call`` keyed by one),
handlers linked per API and executor (a non-blocking import with one is
called synchronously, so construction may call it), and listeners that
hear each call's error and the VM's fatal, which ends every other call.
"""
from __future__ import annotations

import queue
import threading
from collections import deque
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

#: notify kinds (``pssvm_notify``). Kind 1, the old ``N_IRQ``, is gone:
#: an interrupt is an exported function call (``CallTable.start_function``;
#: host-abi.md "Concurrent calls")
N_CHAN = 2

#: ``CallTable.link``: the API ``api`` that is every API with no handler of
#: its own (a backstop), and the context that is every executor and none
API_ANY = -1
CTX_ANY = None
#: a linked handler's answer to a blocking request it completes later
#: (``HostCall.complete``); only :meth:`CallTable.step` serves such
DEFER = object()

#: ``CallTable.pending`` flags: a call can run; requests are outstanding; a
#: timer is set
PEND_READY, PEND_REQS, PEND_TIME = 1, 2, 4

#: listener severities: a call failed (the VM goes on); the VM cannot go on
SEV_ERROR, SEV_FATAL = 1, 2

#: the VM state a traversal's root frame sets (``VM.root_frame``,
#: ``set_entry``), saved with its call and put back each time the call runs:
#: a traversal queued behind it, or a function call made with
#: ``root_frame``, must not change the active one's pins
_TRAVERSAL_STATE = ("arg_bounds", "arg_types", "solved", "bound_frame",
                    "_bounded")
_MISSING = object()


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


@dataclass
class HostEvent:
    """What a listener hears (host-abi.md "Listeners"): ``sev``
    (``SEV_ERROR``: call ``call_id`` failed, the VM goes on; ``SEV_FATAL``:
    the VM cannot go on), the call's entry and the instance it ran in, the
    PSS location of the failure and its message. A fatal that is no one
    call's (a deadlock across calls) has ``call_id`` 0."""
    sev: int
    msg: str
    call_id: int = 0
    entry: str = ""
    comp_path: str = ""
    file: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None
    src_ref: int = 0
    exc: Optional[BaseException] = None

    @property
    def located(self) -> str:
        """The message, ``file:line:`` first when it has a place."""
        if self.file and self.line:
            return f"{self.file}:{self.line}: {self.msg}"
        return self.msg


class Listener:
    """A listener's two functions (``CallTable.add_listener``). They are
    called synchronously at the failure, before any call it releases
    returns, so they cannot wait. A fatal listener that returns lets the
    host decide; one that raises ends the host's loop with that error."""

    def on_error(self, ev: HostEvent) -> None:
        pass

    def on_fatal(self, ev: HostEvent) -> None:
        pass


@dataclass
class Pending:
    """``CallTable.pending``: calls that can run now, requests outstanding
    (issued and not completed), and the earliest timer (engine ticks;
    None: none). ``flags`` is the C ABI's ``PEND_*`` mask; false when
    nothing is pending."""
    ready: int = 0
    reqs: int = 0
    next_time: Optional[int] = None

    @property
    def flags(self) -> int:
        return ((PEND_READY if self.ready else 0)
                | (PEND_REQS if self.reqs else 0)
                | (PEND_TIME if self.next_time is not None else 0))

    def __bool__(self) -> bool:
        return bool(self.flags)


class HostCall:
    """One call run against a host. ``root`` is the call's frame
    (``VM.root_frame``); a root with a component tree is constructed first,
    as ``VM.run`` does, and construction must not yield.

    ``on_notify(vm, kind, id, value)`` takes a :meth:`notify` on the VM
    thread; without one, ``N_CHAN`` puts on a host-fed channel and any
    other kind is an error. An interrupt is not a notification: the host
    calls the handler, an exported function (:meth:`start_function`).

    ``construct=False`` runs ``root`` on the tree as it stands: a function
    called on a tree built before (no construction, even with a ``cobj``).

    Several calls may be live on one VM (a host thread per call, e.g. an
    SV task blocked in a transfer while an irq monitor calls
    ``notify_irq``). The first runs on the VM's scheduler; one started
    while another is live gets a ready queue of its own, so each call runs
    only its own threads (they share the component tree: a channel put by
    one wakes the other at its next :meth:`run`). A call whose root is an
    action (``traversal``) takes the VM's one traversal slot when it first
    runs; while another traversal holds it, :meth:`run` says ``WAIT TURN``
    and the call waits, in FIFO order, until the slot is free
    (:class:`CallTable`)."""

    def __init__(self, vm, root, on_notify: Optional[Callable] = None,
                 seed: int = 0, construct: bool = True):
        self.vm = vm
        self.construct = construct
        self.root = root
        self.on_notify = on_notify
        self.seed = seed
        #: the root is an action run: the call needs the traversal slot
        self.traversal = root is not None and (
            bool(getattr(root, "action", False))
            or getattr(root, "act", None) is not None)
        #: what the root frame set on the VM (``_TRAVERSAL_STATE``)
        self._state = ({k: getattr(vm, k, _MISSING) for k in _TRAVERSAL_STATE}
                       if self.traversal else None)
        t = self.table = calls_of(vm)
        t.owner = None              # a root frame may have just set the VM's
        #: the call's id in events (``HostEvent.call_id``), from 1
        self.id = t.next_call
        t.next_call += 1
        t.calls.append(self)
        #: ``vm.progress`` when the call last waited on EVENT or TIME: it
        #: may run on once another call has made progress since
        self._idle_at: Optional[int] = None
        #: WAIT TIME: the VM time its timer is due
        self._due = 0
        #: run again although nothing it can see changed (a host's
        #: ``wait_event`` said to go on)
        self._poke = False
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
        #: its errors are fatal (a construction: ``CallTable.new``)
        self.escalate = False
        #: ended by a fatal elsewhere: no event of its own, and ``run``
        #: gives its ERROR again
        self.released = False
        self._done_fns: List[Callable] = []
        self.over = False
        self.sched = None          # set when the call starts (_main)

    # -- the host's side -------------------------------------------------- #

    def run(self) -> Request:
        """Run the call until it needs the host; the request says why. A
        traversal that cannot have the traversal slot yet says ``WAIT``
        ``TURN`` and runs nothing; one started from inside a request of the
        active traversal, which would wait on itself, ends in ``ERROR``
        (self-wait). After a fatal, a call not started ends in ERROR naming
        it, and one the fatal released gives its ERROR again."""
        if self.over:
            if self.released:
                return self._last
            raise HostStateError("the call is over")
        if self._thread is None:
            t = self.table
            if t.fatal_exc is not None:
                return self._fail_unstarted(t.refusal(), quiet=True)
            if not t.admit(self):
                exc = t.self_wait(self)
                if exc is not None:
                    return self._fail_unstarted(exc)
                self._last = Request(WAIT, on="TURN")
                return self._last
            self._thread = threading.Thread(target=self._main, daemon=True,
                                            name="pssvm-call")
            self._thread.start()
        else:
            self._to_vm.put(_RESUME)
        return self._host_wait()

    def _host_wait(self) -> Request:
        r = self._to_host.get()
        self._last = r
        if r.status in (DONE, ERROR):
            self.over = True
            self._thread.join()
            self.table.ended(self, r)
        return r

    def abort(self, exc: BaseException) -> Request:
        """End the call in ``ERROR`` with ``exc``, raised where its threads
        wait as an engine error would be: a deadlock across calls, or a
        traversal queued behind one. Returns the ERROR request (the last
        one, for a call already over)."""
        if self.over:
            return self._last
        if self._thread is None:
            return self._fail_unstarted(exc)
        self._to_vm.put(_Raise(exc))
        return self._host_wait()

    def _fail_unstarted(self, exc: BaseException, quiet: bool = False) -> Request:
        r = Request(ERROR, error=self._error(exc), exc=exc)
        self._last = r
        self.over = True
        self.released = self.released or quiet
        self.table.drop(self)
        self.table.ended(self, r)
        return r

    @property
    def status(self) -> Optional[int]:
        """DONE or ERROR once the call is over; None before (C:
        ``pssvm_done``/``pssvm_status`` of the completion object)."""
        r = self._last
        return r.status if self.over and r is not None else None

    def on_done(self, fn: Callable) -> None:
        """Call ``fn(call, request)`` once, when the call ends in DONE or
        ERROR (at once if it has): the completion object's callback
        (host-abi.md "Completion"). It runs on the host thread that saw
        the end, before that thread's ``run`` returns."""
        if self.over and self._last is not None \
                and self._last.status in (DONE, ERROR):
            fn(self, self._last)
        else:
            self._done_fns.append(fn)

    @property
    def name(self) -> str:
        """The call's name in a message: its root's coroutine."""
        coro = getattr(self.root, "coro", None)
        return getattr(coro, "name", None) or "<call>"

    def queued(self) -> bool:
        """A traversal waiting for the traversal slot (never started)."""
        return (self._thread is None and not self.over
                and self in self.table.queue)

    def runnable(self) -> bool:
        """A run now could do something: not started (and, a traversal,
        its turn), requests completed, or idle on EVENT or TIME while
        another call made progress or a notification arrived. A call
        waiting on TIME is runnable through :func:`run_calls`' clock
        otherwise."""
        if self.over:
            return False
        if self._thread is None:
            return self.table.admissible(self)
        r = self._last
        if r is None or r.status not in (REQS, WAIT):
            return True
        if r.status == REQS or r.on == "REQS":
            return bool(self._woken) or not self._out
        if r.on == "TIME" and self.table.now is not None \
                and self.table.now >= self._due:
            return True                 # the engine clock reached its timer
        return (self._poke or bool(self._notes)
                or self.vm.progress != self._idle_at)

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
        self.table.drop(self)

    def start_function(self, entry, args=(), comp: int = 0):
        """Start exported function ``entry`` (a symbol or coroutine index)
        on this call's tree, as a call of its own: an interrupt handler the
        host calls (host-abi.md "Concurrent calls"). :func:`run_sync` and
        :func:`run_calls` driving this call drive it too."""
        return calls_of(self.vm).start_function(entry, args, comp=comp,
                                                seed=self.seed)

    # -- the VM's side (on the call's thread) ----------------------------- #

    def _start(self) -> None:
        """Pick the call's scheduler: the VM's own if no other call is live,
        else a fresh one sharing the VM's clock and frame ids."""
        t = calls_of(self.vm)
        home = t.home
        if t.live:
            s = type(home)()
            s.now = home.now
            s.new_frame = home.new_frame
            self.sched = s
        else:
            self.sched = home
        t.live.append(self)
        self._enter()

    def _enter(self) -> None:
        """The call's thread runs: the VM is this call's."""
        vm = self.vm
        t = calls_of(vm)
        home = t.home
        vm.host = self
        if self._state is not None and t.owner is not self:
            for k, v in self._state.items():
                if v is not _MISSING:
                    setattr(vm, k, v)
            t.owner = self
        if self.sched is not home:
            self.sched.now = max(self.sched.now, home.now)
        if self._time is not None:
            self.sched.now = max(self.sched.now, self._time)
        vm.sched = self.sched
        self._poke = False
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
        t = vm.__dict__.get("_calls")
        if t is None or self.sched is None:
            return
        home = t.home
        home.now = max(home.now, self.sched.now)
        vm.sched = home
        self._mark = (home, len(home._ready))
        if done:
            t.drop(self)

    def _main(self) -> None:
        vm = self.vm
        self._start()
        try:
            self._take_notes()
            if self.construct and self.root.cobj is not None:
                self._constructing = True
                try:
                    vm.construct_components()
                except VMError as e:
                    e.fatal = True       # the tree is half built
                    raise
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
            elif self._chan_ok(kind, id_):
                self._chan_put(id_, value)
            elif kind == 1:
                raise VMError("notify kind 1 has no handler: an interrupt is "
                              "an exported function call, which the host "
                              "makes (host-abi.md \"Concurrent calls\")")
            else:
                raise VMError(f"notify kind {kind} has no handler")

    def _ev_notify(self, kind: int, id_: int, value: int) -> None:
        """The ``notify`` event of a notification taken: the platform side a
        replay feeds back (the native engine raises the same)."""
        delivered = self.on_notify is not None or self._chan_ok(kind, id_)
        key = {N_CHAN: "chan"}.get(kind)
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
        self.vm.progress += 1          # a call waiting on it may run on

    def _no_yield(self, what: str) -> None:
        if self._constructing:
            raise VMError(f"construct-yield: {what} during construction")

    def _construct_yield(self, r: Request) -> None:
        """Construction calls a non-blocking import only through a linked
        handler (host-abi.md "Construction"); anything else would yield."""
        if not self._constructing:
            return
        if r.api == API_MEM:
            self._no_yield("an access")
        name = r.name or f"fn {r.fn}"
        if r.blocking:
            self._no_yield(f"blocking import {name}")
        self._no_yield(f"import {name} with no linked handler (a non-blocking "
                       f"import is called synchronously when its API has "
                       f"one: CallTable.link)")

    def _issue(self, frame, r: Request):
        """Suspend ``frame`` on request ``r``; or, re-run once ``r`` is
        completed, its answer."""
        a = self._answers.pop(id(frame), PARKED)
        if a is not PARKED:
            if isinstance(a, _Raise):
                raise a.exc
            return a
        self._construct_yield(r)
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
        t = self.table
        r = Request(
            REQS, api=t.api_of(fn_id), fn=fn_id, name=self.vm.import_name(fn_id),
            args=list(args), blocking=blocking, src_ref=ins.src_ref,
            nret=(ins.imm >> 8) & 0xFF)
        if not blocking and t.links:
            r.executor = getattr(frame, "executor", NO_EXECUTOR)
            h = t.handler(r.api, r.executor)
            if h is not None:
                # linked: called here and now, no request (host-abi.md
                # "Linking"); its error is raised at the import
                r.thread = frame.id
                r.stim_id = getattr(frame, "stim", "") or ""
                v = t.invoke(h, r)
                if v is DEFER:
                    raise VMError(f"the handler linked for import {r.name} "
                                  f"deferred a non-blocking call")
                return v
        return self._issue(frame, r)

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
        self._idle_at = self.vm.progress
        self._due = self.sched.now + delay
        self._yield(Request(WAIT, on="TIME", delay=delay))
        return self._clocked

    def wait_event(self) -> None:
        self._idle_at = self.vm.progress
        self._yield(Request(WAIT, on="EVENT"))


class CallTable:
    """The engine's host state, for one VM (host-abi.md): every call
    ("Concurrent calls": the live calls, the one traversal slot and its
    FIFO queue, the function calls started for a host loop to adopt
    (:meth:`start_function`), the requests a sync loop is serving now
    (self-wait)); the object handles ("Objects": :meth:`new`,
    :meth:`release`, :meth:`call`); the linked handlers ("Linking"); the
    listeners and the fatal ("Listeners"); and :meth:`step` /
    :meth:`pending` over every call. :func:`calls_of` gives a VM's."""

    def __init__(self, vm):
        self.vm = vm
        #: the VM's own scheduler: the first live call's
        self.home = getattr(vm, "sched", None)
        #: calls started and not over, in start order
        self.live: List[HostCall] = []
        #: the traversal holding the slot (None: free)
        self.active: Optional[HostCall] = None
        #: traversals waiting for it, oldest first
        self.queue: deque = deque()
        #: function calls :meth:`start_function` made, not yet adopted
        self.adopt: List[HostCall] = []
        #: (call, request, host thread) a sync loop is serving, innermost last
        self.serving: List[tuple] = []
        #: the traversal whose root-frame state (``_TRAVERSAL_STATE``) the
        #: VM holds now (None: unknown; a call's construction clears it)
        self.owner: Optional[HostCall] = None
        #: every call made and not over, in the order made (``step``)
        self.calls: List[HostCall] = []
        self.next_call = 1
        #: (api, context) -> handler (:meth:`link`)
        self.links: dict = {}
        self._apis: Optional[dict] = None
        self.listeners: List[Any] = []
        #: the fatal the VM stopped on (None: none): later calls are refused
        self.fatal_exc: Optional[BaseException] = None
        #: the engine clock (:meth:`set_time`; None: never set)
        self.now: Optional[int] = None
        #: the object handles (:meth:`new`; ``handles.Handles``)
        self._handles = None

    # -- the traversal slot -------------------------------------------- #

    def admissible(self, call: HostCall) -> bool:
        """``call`` may start now: a function call always; a traversal
        when the slot is free and it is first in the queue (it joins the
        queue here if it is not in it)."""
        if not call.traversal:
            return True
        if self.active is call:
            return True
        if call not in self.queue:
            self.queue.append(call)
        return self.active is None and self.queue[0] is call

    def admit(self, call: HostCall) -> bool:
        """:meth:`admissible`, and a traversal admitted takes the slot."""
        if not self.admissible(call):
            return False
        if call.traversal and self.active is not call:
            self.queue.popleft()
            self.active = call
        return True

    def drop(self, call: HostCall) -> None:
        """``call`` is over (or ended): it leaves the live set, the slot or
        the queue."""
        if call in self.calls:
            self.calls.remove(call)
        if call in self.live:
            self.live.remove(call)
        if self.active is call:
            self.active = None
        if call in self.queue:
            self.queue.remove(call)

    def self_wait(self, call: HostCall) -> Optional[VMError]:
        """The error for traversal ``call``, queued, when this host thread
        is inside a request of the active traversal: that traversal cannot
        end before the request returns, and the request waits for
        ``call``. None: the wait is an ordinary one."""
        a, me = self.active, threading.get_ident()
        for c, q, tid in reversed(self.serving):
            if c is a and tid == me:
                if q.api != API_MEM:
                    what = q.name or f"fn {q.fn}"
                else:
                    what = (f"{'read' if q.kind == 'r' else 'write'} "
                            f"of 0x{q.addr:x}")
                return VMError(
                    f"self-wait: traversal {call.name} was started inside "
                    f"request {q.req_id} ({what}) of the active traversal "
                    f"{a.name}; it would wait for {a.name} to end, and "
                    f"{a.name} waits for that request. One traversal runs at "
                    f"a time: start the second after the first ends, or call "
                    f"a function instead")
        return None

    # -- function calls ------------------------------------------------- #

    def start_function(self, entry, args=(), comp: int = 0, seed: int = 0,
                       on_notify: Optional[Callable] = None) -> HostCall:
        """Start exported function ``entry`` (a function ENTRY's symbol, its
        unqualified name, or a coroutine index) in component instance
        ``comp`` of the VM's tree, as a call of its own: an interrupt
        handler, which may make blocking imports while a traversal waits.
        It never queues. The next :func:`run_sync` or :func:`run_calls` loop
        on this VM adopts and drives it; another host runs it itself."""
        vm = self.vm
        coro = _function_coro(vm.model, entry)
        frame = self.home.new_frame(vm.model.coros[coro], seed=seed,
                                    obj=vm._new_obj(coro))
        frame.cobj = getattr(vm, "cobj", None)
        if frame.cobj is not None:
            vm.set_comp(frame, comp)
        frame.args = list(args)
        call = HostCall(vm, frame, on_notify=on_notify, seed=seed,
                        construct=False)
        self.adopt.append(call)
        return call

    def take_adopted(self) -> List[HostCall]:
        out, self.adopt = self.adopt, []
        return out

    # -- objects (handles.py) -------------------------------------------- #

    @property
    def handles(self):
        if self._handles is None:
            from .handles import Handles
            self._handles = Handles(self)
        return self._handles

    def new(self, type_: str, parent: int = 0, args=(), seed: int = 0) -> int:
        """``pssvm_new``: with no ``parent``, construct the tree of root
        component ``type_`` (constructor ``args``) and return its handle;
        under a component, an action object of action ``type_`` (a
        traversal's ``call`` target, whose fields stay readable after it).
        A construction that fails is fatal (host-abi.md "Objects")."""
        return self.handles.new(type_, parent, args, seed)

    def sub(self, parent: int, path: str) -> int:
        """A handle of the instance at ``path`` under component ``parent``
        (``ch[0]``): what a function call names in place of a path."""
        return self.handles.sub(parent, path)

    def release(self, h: int) -> None:
        """``pssvm_release``: free handle ``h``. The root's release frees
        every handle under it and the tree; refused while a call runs on
        what it frees. A released handle is refused wherever it is used."""
        self.handles.release(h)

    def call(self, obj: int, method: str = "", args=(), seed: int = 0) -> HostCall:
        """``pssvm_call``: a call keyed by object: a traversal of action
        object ``obj`` (``method`` "" or its action), or function
        ``method`` of component ``obj``; ``$run_end`` on the root runs the
        tree's ``exec run_end`` blocks. The call runs at its first
        ``run`` (or :meth:`step`)."""
        return self.handles.call(obj, method, args, seed)

    def field(self, obj: int, name: str) -> int:
        """Field ``name`` of action object ``obj``, once no call on it runs
        (``pssvm_call_field``)."""
        return self.handles.field(obj, name)

    def pin(self, obj: int, path: str, value: str) -> None:
        """A field override on action object ``obj``: ``path`` fixed to the
        value text ``value`` names, in its later traversals
        (``Handles.pin``)."""
        self.handles.pin(obj, path, value)

    # -- linking --------------------------------------------------------- #

    def api_id(self, api) -> int:
        """An API's id from its id or name (``mem``, ``pss_top``); API_ANY
        for "*"."""
        if isinstance(api, int):
            return api
        if api == "*":
            return API_ANY
        for a in getattr(self.vm.model, "apis", None) or ():
            if a["name"] == api:
                return a["id"]
        builtin = {"mem": API_MEM, "config": API_CONFIG, "seq": API_SEQ}
        if api in builtin:
            return builtin[api]
        raise HostStateError(f"no API named '{api}' in the image")

    def api_of(self, fn_id: int) -> int:
        """The API of user import ``fn_id`` (its IMPORT row); API_USER when
        the image does not say."""
        apis = self._apis
        if apis is None:
            apis = self._apis = {
                sig.get("fn_id"): sig.get("api", API_USER)
                for sig in (getattr(self.vm.model, "imports", None) or {}).values()}
        return apis.get(fn_id, API_USER)

    def link(self, api, handler, ctx: Optional[int] = CTX_ANY) -> None:
        """``pssvm_link``: ``handler`` serves API ``api`` (id or name; "*"
        or API_ANY: every API with no handler of its own) for requests of
        executor ``ctx`` (an EXEC id or NO_EXECUTOR; CTX_ANY: all). A
        handler is a callable ``handler(request) -> value``, an object
        with ``call(fn_id, args)`` (an ImportProvider) or, for ``mem``,
        ``read(addr, size)``/``write(addr, size, data)`` (a Memory). A
        non-blocking import whose API has one is called synchronously,
        with no request; a blocking one is still a request, which the
        sync loops and :meth:`step` serve from it (``step`` lets it answer
        DEFER and complete later). None unlinks."""
        key = (self.api_id(api), ctx)
        if handler is None:
            self.links.pop(key, None)
        else:
            self.links[key] = handler

    def handler(self, api: int, executor: int = NO_EXECUTOR):
        """The handler linked for ``api`` and ``executor`` (None: none)."""
        links = self.links
        for key in ((api, executor), (api, CTX_ANY), (API_ANY, executor),
                    (API_ANY, CTX_ANY)):
            h = links.get(key)
            if h is not None:
                return h
        return None

    def invoke(self, h, q: Request):
        """Ask handler ``h`` for request ``q``'s answer."""
        if q.api == API_MEM and hasattr(h, "read"):
            if q.kind == "r":
                return h.read(q.addr, q.size)
            h.write(q.addr, q.size, q.data)
            return None
        call = getattr(h, "call", None)
        if call is not None and q.api != API_MEM:
            return call(q.fn, q.args)
        return h(q)

    def missing_links(self) -> List[str]:
        """What :meth:`link_check` names: each API a coroutine reachable
        from an entry (or construction, or the run hooks) uses, with no
        handler for every context, as ``api: import, ...``."""
        from .handles import reachable_uses
        uses = reachable_uses(self.vm.model)
        ctxs = [NO_EXECUTOR] + list(range(len(
            getattr(self.vm.model, "executors", None) or ())))
        out = []
        for api in sorted(uses):
            if any(self.links.get(k) is not None for k in
                   ((api, CTX_ANY), (API_ANY, CTX_ANY))):
                continue
            if all(self.handler(api, c) is not None for c in ctxs):
                continue
            name = ("mem" if api == API_MEM else
                    next((a["name"] for a in getattr(self.vm.model, "apis", None)
                          or () if a["id"] == api), f"api {api}"))
            out.append(f"{name}: {', '.join(sorted(uses[api]))}")
        return out

    def link_check(self) -> None:
        """``pssvm_link_check``: raise HostStateError naming every API
        (and its imports) that a reachable coroutine calls with no linked
        handler, before time 0 rather than at the first call. A host that
        serves requests itself links a backstop ("*") or skips the check."""
        missing = self.missing_links()
        if missing:
            raise HostStateError("no handler linked for " + "; ".join(missing))

    # -- listeners and the fatal ---------------------------------------- #

    def add_listener(self, listener) -> None:
        """``pssvm_add_listener``: ``listener.on_error(ev)`` hears each
        call that ends in ERROR, ``on_fatal(ev)`` the fatal (``HostEvent``).
        With none, an error is the call's status only, and a fatal is
        raised out of the host loop that met it (Python's end of the
        simulation; SV's default listener calls ``$fatal``)."""
        self.listeners.append(listener)

    def remove_listener(self, listener) -> None:
        if listener in self.listeners:
            self.listeners.remove(listener)

    def event(self, sev: int, exc: BaseException,
              call: Optional[HostCall] = None) -> HostEvent:
        p = getattr(exc, "prov", None)
        ev = HostEvent(sev=sev, msg=str(exc), exc=exc,
                       src_ref=getattr(exc, "src_ref", 0) or 0,
                       file=getattr(p, "file", None) if p else None,
                       line=getattr(p, "line", None) if p else None,
                       col=getattr(p, "col", None) if p else None)
        if call is not None:
            ev.call_id, ev.entry = call.id, call.name
            comp = getattr(call.root, "comp", None)
            comps = getattr(self.vm.model, "components", None)
            if comps is not None and comp is not None and 0 <= comp < len(comps.paths):
                ev.comp_path = comps.paths[comp]
        return ev

    def ended(self, call: HostCall, r: Request) -> None:
        """``call`` is over: its completion callbacks, then, for an ERROR
        the call did not get from a fatal elsewhere, the listeners (a
        fatal error: :meth:`fatal`)."""
        fns, call._done_fns = call._done_fns, []
        for fn in fns:
            fn(call, r)
        if r.status != ERROR or call.released:
            return
        exc = r.exc if r.exc is not None else VMError(
            (r.error or {}).get("message", "error"))
        if getattr(exc, "fatal", False) or call.escalate:
            self.fatal(exc, call)
            return
        ev = None
        for l in list(self.listeners):
            f = getattr(l, "on_error", None)
            if f is not None:
                f(ev if ev is not None else (ev := self.event(SEV_ERROR, exc, call)))

    def fatal(self, exc: BaseException, call: Optional[HostCall] = None) -> None:
        """The VM cannot go on (``exc``; ``call`` the one it happened in, if
        one): the listeners hear it, then every other call ends in ERROR
        with it, started or queued, so none waits for ever; later calls are
        refused until a new tree is built (:meth:`new`). With no listener,
        ``exc`` is raised here once they are released."""
        if self.fatal_exc is None:
            self.fatal_exc = exc
        try:
            ev = self.event(SEV_FATAL, exc, call)
            for l in list(self.listeners):
                f = getattr(l, "on_fatal", None)
                if f is not None:
                    f(ev)
        finally:
            self.release_all(exc)
        if not self.listeners:
            raise exc

    def release_all(self, exc: BaseException) -> None:
        """End every call not over in ERROR with ``exc`` (a fatal)."""
        for c in list(self.calls) + self.take_adopted():
            if not c.over:
                c.released = True
                c.abort(exc)

    def refusal(self) -> VMError:
        e = VMError(f"refused: the engine stopped on a fatal error: "
                    f"{self.fatal_exc}")
        e.fatal = True
        return e

    # -- step and pending ------------------------------------------------ #

    def set_time(self, now: int) -> None:
        """``pssvm_set_time``: the host's clock, in ticks, for every call;
        a call waiting on a timer due by ``now`` can run (:meth:`step`)."""
        if self.now is not None and now < self.now:
            raise HostStateError(f"set_time({now}) is before {self.now}")
        self.now = now
        for c in self.calls:
            if not c.over:
                c.set_time(now)

    def pending(self) -> Pending:
        """``pssvm_pending``: what is left to do across every call."""
        p = Pending()
        for c in self.calls:
            if c.over:
                continue
            if c.runnable():
                p.ready += 1
            p.reqs += len(c._out)
            r = c._last
            if (c._thread is not None and r is not None and r.status == WAIT
                    and r.on == "TIME"):
                p.next_time = c._due if p.next_time is None else min(p.next_time, c._due)
        return p

    def step(self, budget: int = 0) -> int:
        """``pssvm_step``: run calls that can run, each ``run`` one slice,
        in the order they were made, round after round until none can run
        or ``budget`` slices have run (0: no limit). The new requests a
        linked handler serves are answered at once (DEFER: it completes
        them later); the rest wait for the host's ``take``. Returns the
        slices run: 0 with :meth:`pending` false and calls live is for the
        host to break (a notify, a completion), else a deadlock."""
        n = 0
        while True:
            ran = False
            for c in list(self.calls):
                if budget and n >= budget:
                    return n
                if c.over or not c.runnable():
                    continue
                r = c.run()
                n += 1
                ran = True
                if r.status == REQS and self.links:
                    self._serve_linked(c)
            if not ran:
                return n

    def _serve_linked(self, c: HostCall) -> None:
        keep, me = [], threading.get_ident()
        for q in c._new:
            h = self.handler(q.api, q.executor)
            if h is None:
                keep.append(q)
                continue
            self.serving.append((c, q, me))
            try:
                v = self.invoke(h, q)
            except BaseException as e:
                c.fail(q.req_id, e)
                continue
            finally:
                self.serving.pop()
            if v is not DEFER:
                c.complete(q.req_id, v)
        c._new = keep

    # -- deadlock ------------------------------------------------------- #

    def waits(self, calls) -> str:
        """What each of ``calls`` waits on, for a deadlock message."""
        out = []
        for c in calls:
            if c.over:
                continue
            if c.queued() or (c._thread is None and c.traversal):
                act = self.active.name if self.active is not None else "?"
                out.append(f"{c.name} (queued behind traversal {act})")
            elif c._last is not None and c._last.status == WAIT:
                out.append(f"{c.name} (waits on "
                           f"{'an event' if c._last.on == 'EVENT' else c._last.on})")
            else:
                out.append(c.name)
        return ", ".join(out)


def calls_of(vm) -> CallTable:
    """The :class:`CallTable` of ``vm`` (made on first use)."""
    t = vm.__dict__.get("_calls")
    if t is None:
        t = vm.__dict__["_calls"] = CallTable(vm)
    return t


def _function_coro(model, entry) -> int:
    """The coroutine of function entry ``entry`` (symbol, unqualified name
    or index)."""
    if isinstance(entry, int):
        return entry
    table = getattr(model, "entries", None) or {}
    e = table.get(entry)
    if e is None and "::" not in entry:
        found = [x for sym, x in table.items()
                 if sym.rpartition("::")[2] == entry and x.kind == "function"]
        e = found[0] if len(found) == 1 else None
    if e is None or e.kind != "function":
        raise HostStateError(f"no function entry named '{entry}' (a traversal "
                             f"cannot be started as a function call)")
    return e.coro


def _note(mem, q: Request) -> bool:
    """Tell a memory that asks (a register fake) which register access ``q``
    is for, from the request itself: several may be out at once."""
    note = getattr(mem, "note_access", None)
    if note is not None:
        note(q.reg_id, q.src_ref)
    return True


DEADLOCK = ("deadlock: every running thread waits on a channel that nothing "
            "can put to or get from")


class _Loop:
    """One host thread driving several calls of one VM: each round runs
    every call that can run, then serves the requests they made, from
    ``memory`` and ``import_provider`` (default: the VM's own), in issue
    order, non-blocking ones first. When none can run, the earliest WAIT
    TIME goes on (``wait_time(delay)`` first, the host's clock), else
    ``wait_event()`` may notify and return True to go on, else it is a
    deadlock across every live call. Calls started meanwhile with
    :meth:`CallTable.start_function` are adopted."""

    def __init__(self, calls, memory, import_provider, wait_time, wait_event):
        self.calls = list(calls)
        self.vm = self.calls[0].vm
        self.table = calls_of(self.vm)
        self.memory, self.provider = memory, import_provider
        self.wait_time, self.wait_event = wait_time, wait_event

    def live(self) -> List[HostCall]:
        if self.table.adopt:
            self.calls += [c for c in self.table.take_adopted()
                           if c not in self.calls]
        calls = self.calls
        if len(calls) == 1:
            return [] if calls[0].over else calls
        return [c for c in calls if not c.over]

    def round(self) -> Optional[str]:
        """One round. Returns None after progress; "idle" when nothing
        could run and nothing was waited on; "done" when no call is live."""
        live = self.live()
        if not live:
            return "done"
        reqs = []
        ran = self.run_runnable(live, reqs)
        self.serve(reqs)
        if ran or reqs:
            return None
        timed = [c for c in live if c._last is not None
                 and c._last.status == WAIT and c._last.on == "TIME"]
        if timed:
            c = min(timed, key=lambda c: c._due)
            if self.wait_time is not None:
                self.wait_time(max(0, c._due - self.table.home.now))
            # what the host did meanwhile (an interrupt handler it called, a
            # notification) runs first, at the time the wait ended
            self.run_runnable([o for o in self.live() if o is not c], reqs)
            if not c.over:
                self.ran(c, c.run(), reqs)
            self.serve(reqs)
            return None
        if self.wait_event is not None and self.wait_event():
            for c in live:
                c._poke = True
            return None
        return "idle"

    def run_runnable(self, calls, reqs: list) -> bool:
        """Run each of ``calls`` that can run, once, in order; their new
        requests go on ``reqs``. True: one ran (or ended)."""
        ran = False
        for c in calls:
            if c.over:
                continue
            if not c.runnable():
                exc = (self.table.self_wait(c)
                       if c._thread is None and c.traversal else None)
                if exc is not None:
                    c._fail_unstarted(exc)
                    ran = True
                continue
            if (c._last is not None and c._last.status == WAIT
                    and c._last.on == "TIME"):
                # run on without moving time to its timer: something
                # another call did may let it go on now
                c.set_time(max(self.table.home.now, c.sched.now))
            ran = True
            self.ran(c, c.run(), reqs)
        return ran

    def ran(self, c: HostCall, r: Request, reqs: list) -> None:
        if r.status == REQS:
            while (q := c.take()) is not None:
                reqs.append((c, q))
        elif r.status == STOP:
            raise HostStateError(f"{c.name}: cannot serve STOP")

    def serve(self, reqs) -> None:
        vm, t = self.vm, self.table
        if len(reqs) > 1:
            reqs.sort(key=lambda cq: cq[1].blocking)   # stable: issue order
        me = threading.get_ident()
        for c, q in reqs:
            if c.over:
                continue
            # the VM's own, looked up per request: a debugger may swap them
            mem = self.memory if self.memory is not None else vm.memory
            provider = (self.provider if self.provider is not None
                        else vm.import_provider)
            t.serving.append((c, q, me))
            try:
                h = t.handler(q.api, q.executor) if t.links else None
                if h is not None:
                    v = t.invoke(h, q)
                    if v is DEFER:
                        raise HostStateError(
                            f"the handler linked for request {q.req_id} "
                            f"deferred it, which only step() serves")
                elif q.api != API_MEM:
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
                c.fail(q.req_id, e)
                continue
            finally:
                t.serving.pop()
            c.complete(q.req_id, v)

    def deadlock(self) -> str:
        live = [c for c in self.calls if not c.over]
        if len(live) <= 1 and not any(c._thread is None for c in live):
            return DEADLOCK
        return (f"{DEADLOCK}, and no other call can run or put to it: "
                f"{self.table.waits(live)}")

    def dead(self) -> None:
        """A deadlock: a fatal (``CallTable.fatal``), which ends every live
        call; raised here when no listener takes it."""
        exc = VMError(self.deadlock())
        exc.fatal = True
        self.table.fatal(exc)


def run_calls(calls, memory=None, import_provider=None,
              wait_time: Optional[Callable[[int], None]] = None,
              wait_event: Optional[Callable[[], bool]] = None) -> List[Request]:
    """Run ``calls`` (of one VM) to their ends from one host thread, as
    the host ABI allows several live calls (host-abi.md "Concurrent
    calls"): a traversal queued behind another starts when it ends, in
    FIFO order; function calls run alongside, and those started meanwhile
    (``start_function``: an interrupt) are adopted. Requests are served
    as :func:`run_sync` serves them; ``wait_time``/``wait_event`` are its
    hooks. A deadlock -- no live call can run, no timer is due and the host
    has nothing to notify -- is a fatal: the listeners hear it, every live
    call ends in ERROR with a message naming what each waits on, and with
    no listener it is raised. Returns each call's last request (DONE or
    ERROR), adopted calls after the rest."""
    calls = list(calls)
    if not calls:
        return []
    loop = _Loop(calls, memory, import_provider, wait_time, wait_event)
    while (st := loop.round()) != "done":
        if st == "idle":
            loop.dead()
    return [c.request for c in loop.calls]


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
    ``deadlock`` error, a fatal (``run_calls``). Function calls started meanwhile on the VM
    (``start_function``: an interrupt handler) run alongside and are driven
    to their ends too, as :func:`run_calls` does.
    Returns the entry's return value; an engine error is re-raised. An
    error in ``memory`` or ``import_provider`` is raised in the VM thread
    (:meth:`HostCall.fail`), so the VM sees it as it would a direct call."""
    loop = _Loop([call], memory, import_provider, wait_time, wait_event)
    try:
        while True:
            st = loop.round()
            for c in loop.calls:
                r = c.request
                if r is not None and r.status == ERROR and c.over:
                    raise r.exc
            if st == "done":
                return call.result
            if st == "idle":
                if call._thread is None and call.traversal and not call.over:
                    raise HostStateError(
                        f"traversal {call.name} is queued behind "
                        f"{loop.table.active.name if loop.table.active else '?'}"
                        f", which this loop does not drive")
                loop.dead()
    except BaseException:
        for c in loop.calls:
            if not c.over:
                c.end()
        raise
