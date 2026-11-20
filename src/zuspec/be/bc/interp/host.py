"""The host ABI seam (``docs/spec/host-abi.md``, P3.1b).

A :class:`HostCall` runs one call of a :class:`~.vm.VM` until the call needs
its host, then hands the host a :class:`Request`: a memory access, a user
import, or a wait. The host answers with :meth:`HostCall.complete` (for an
access or import) and runs the call on with :meth:`HostCall.run`::

    call = HostCall(vm, vm.root_frame(coro, seed))
    while (r := call.run()).status != DONE:
        if r.status == ACCESS:
            call.complete(bus.read(r.addr, r.size) if r.kind == "r"
                          else bus.write(r.addr, r.size, r.data))
        ...

:func:`run_sync` is that loop with the VM's own ``Memory`` and
``ImportProvider`` as the answers: a run through it is the same as
``VM.run``.

The call runs on a thread of its own that blocks at each yield, so the
interpreter keeps its plain call stack (the debug stub does the same,
``debug.py``). Only one of the two threads runs at a time.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .ops_proc import VMError

#: run status (``pssvm_run``)
DONE, ACCESS, IMPORT, WAIT, STOP, ERROR = 0, 1, 2, 3, 4, -1
STATUS_NAMES = {DONE: "DONE", ACCESS: "ACCESS", IMPORT: "IMPORT", WAIT: "WAIT",
                STOP: "STOP", ERROR: "ERROR"}

#: notify kinds (``pssvm_notify``)
N_IRQ, N_CHAN = 1, 2


class HostStateError(RuntimeError):
    """A call made in the wrong state (``PSSVM_E_STATE``)."""


@dataclass
class Request:
    """What :meth:`HostCall.run` returned: ``status`` and its fields."""
    status: int
    #: ACCESS: "r" or "w", byte address, size in bytes, write data, register
    kind: str = ""
    addr: int = 0
    size: int = 0
    data: Optional[int] = None
    reg_id: Optional[int] = None
    src_ref: int = 0
    #: IMPORT
    fn_id: int = 0
    name: Optional[str] = None
    args: List[int] = field(default_factory=list)
    blocking: bool = False
    #: WAIT: "TIME" (with ``delay`` engine units) or "EVENT"
    on: str = ""
    delay: int = 0
    #: ERROR: the ``pssvm_error`` fields, and the exception itself
    error: Optional[dict] = None
    exc: Optional[BaseException] = None

    @property
    def status_name(self) -> str:
        return STATUS_NAMES[self.status]


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
        self._pending: Optional[Request] = None   # ACCESS/IMPORT not completed
        self._answer: Any = _RESUME
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
        if self._pending is not None:
            raise HostStateError(f"{self._pending.status_name} not completed")
        if self._thread is None:
            self._thread = threading.Thread(target=self._main, daemon=True,
                                            name="pssvm-call")
            self._thread.start()
        else:
            answer, self._answer = self._answer, _RESUME
            self._to_vm.put(answer)
        r = self._to_host.get()
        self._last = r
        if r.status in (ACCESS, IMPORT):
            self._pending = r
        elif r.status in (DONE, ERROR):
            self.over = True
            self._thread.join()
        return r

    def complete(self, value: Optional[int] = None) -> None:
        """Answer the pending access or import: the read value, the import's
        result (None: none), nothing for a write."""
        r = self._pending
        if r is None:
            raise HostStateError("nothing to complete")
        if r.status == ACCESS:
            if r.kind == "r":
                if value is None:
                    raise HostStateError("a read completes with its value")
                value = int(value) & ((1 << (8 * r.size)) - 1)
            elif value is not None:
                raise HostStateError("a write completes with no value")
        self._pending = None
        self._answer = value

    def fail(self, exc: BaseException) -> None:
        """Answer the pending access or import with an error: ``exc`` is
        raised where the VM made it, as a platform callback's own error is
        when the VM calls it directly."""
        if self._pending is None:
            raise HostStateError("nothing to fail")
        self._pending = None
        self._answer = _Raise(exc)

    def notify(self, kind: int, id: int = 0, value: int = 0) -> None:
        """A host event; it takes effect at the next :meth:`run`."""
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

    def end(self) -> None:
        """Free the call; one not DONE is cancelled (its threads never run
        again)."""
        if self._thread is not None and not self.over:
            self._to_vm.put(_CANCEL)
            self._thread.join()
        self.over = True
        self._pending = None
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
        vm.sched = self.sched

    def _leave(self, done: bool = False) -> None:
        """The call's thread stops (a yield, or the end): the VM's own
        scheduler is back for the host (and for the next call)."""
        vm = self.vm
        home = getattr(vm, "_home_sched", None)
        if home is None or self.sched is None:
            return
        home.now = max(home.now, self.sched.now)
        vm.sched = home
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
            else:
                raise VMError(f"notify kind {kind} has no handler")

    def _ev_notify(self, kind: int, id_: int, value: int) -> None:
        """The ``notify`` event of a notification taken: the platform side a
        replay feeds back (the native engine raises the same)."""
        delivered = self.on_notify is not None or (
            kind == N_IRQ and self.irq_entry is not None)
        key = {N_IRQ: "irq", N_CHAN: "chan"}.get(kind)
        detail = {key: id_} if key else {"kind": kind}
        detail.update(value=value, delivered=delivered)
        self.vm.host_event("notify", detail)

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

    def access(self, frame, ins, kind: str, addr: int, size: int,
               data: Optional[int] = None):
        self._no_yield("an access")
        return self._yield(Request(ACCESS, kind=kind, addr=addr, size=size,
                                   data=data, reg_id=self.vm.reg_id(frame),
                                   src_ref=ins.src_ref if ins is not None else 0))

    def import_(self, frame, ins, fn_id: int, args, blocking: bool):
        self._no_yield("an import")
        return self._yield(Request(IMPORT, fn_id=fn_id, name=self.vm.import_name(fn_id),
                                   args=list(args), blocking=blocking,
                                   src_ref=ins.src_ref))

    def wait_time(self, delay: int) -> None:
        self._yield(Request(WAIT, on="TIME", delay=delay))

    def wait_event(self) -> None:
        self._yield(Request(WAIT, on="EVENT"))


def run_sync(call: HostCall, memory=None, import_provider=None):
    """Run ``call`` to DONE, answering from ``memory`` and ``import_provider``
    (default: the VM's own), with no clock: a WAIT TIME runs on at once and
    a WAIT EVENT, which nothing here can notify, is the ``deadlock`` error.
    Returns the entry's return value; an engine error is re-raised. An
    error in ``memory`` or ``import_provider`` is raised in the VM thread
    (:meth:`HostCall.fail`), so the VM sees it as it would a direct call."""
    vm = call.vm
    while True:
        r = call.run()
        if r.status == DONE:
            return call.result
        if r.status in (ACCESS, IMPORT):
            # the VM's own, looked up per request: a debugger may swap them
            mem = memory if memory is not None else vm.memory
            provider = (import_provider if import_provider is not None
                        else vm.import_provider)
            try:
                if r.status == IMPORT:
                    v = provider.call(r.fn_id, r.args)
                elif r.kind == "r":
                    v = mem.read(r.addr, r.size)
                else:
                    mem.write(r.addr, r.size, r.data)
                    v = None
            except BaseException as e:
                call.fail(e)
                continue
            call.complete(v)
        elif r.status == WAIT:
            if r.on == "EVENT":
                call.end()
                raise VMError("deadlock: every running thread waits on a "
                              "channel that nothing can put to or get from")
        elif r.status == ERROR:
            raise r.exc
        else:
            call.end()
            raise HostStateError(f"run_sync cannot serve {r.status_name}")
