"""The engine debug stub (pssc P2.1): ``docs/spec/debug-stub.md``, v1.

A :class:`Stub` drives one run of a :class:`~.vm.VM` on its own host
thread. The host sends requests (``Stub.request(msg) -> reply``); events
(``stopped``, ``exited``) queue up for ``Stub.next_event`` and, if given,
go to ``on_event`` too. The VM calls :meth:`Stub._before` before each
instruction (``VM._run_frame_ev``) only while a stub is attached
(``VM._dbg``); a run without one is unchanged.
"""

import bisect
import hashlib
import queue
import threading
from typing import Any, Callable, Optional

from . import ops_proc
from ..model import (BUILTIN_READ, HEAP_CELL_INT, HEAP_CELL_STR, HEAPK_LIST,
                     HEAPK_SET, HEAPK_STR, INSTR_F_HAS_RET, Instr, Op)
from .heap import _KIND_NAMES
from .scheduler import Frame
from .vm import VMError

VERSION = 1
ENGINE = "pyvm"
CAPABILITIES = ["access_watch", "backdoor_read", "exec_snippet", "run_to"]

#: commands accepted while running (the rest reply ``not_stopped``)
_WHILE_RUNNING = {"stop", "set_pc_breaks", "set_access_watch", "set_stop_on",
                  "detach", "terminate"}


class Terminated(BaseException):
    """Raised in the VM thread by ``terminate`` to end the run (a
    BaseException, so no ``except Exception`` on the way swallows it)."""


class StubError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def thread_of(frame: Frame) -> Frame:
    """The thread a frame runs on: itself, or the frame its call chain
    (``Frame.called``) started from."""
    while frame.called and frame.parent is not None:
        frame = frame.parent
    return frame


def backdoor_read(memory, addr: int, n: int) -> int:
    """``n`` bytes at ``addr`` with no bus access (no log, no side effect):
    the memory's ``peek(addr, n)`` if it has one (a platform with a
    register mirror), else ``extern.Memory``'s bytes."""
    peek = getattr(memory, "peek", None)
    if peek is not None:
        return peek(addr, n)
    data = getattr(memory, "bytes", None)
    if not isinstance(data, dict):
        raise StubError("unsupported", "the platform memory has no back door")
    v = 0
    for i in range(n):
        v |= data.get((addr + i) & ((1 << 64) - 1), 0) << (8 * i)
    return v


#: ops a snippet may not run: writes (frame slots, fields, CALL staging)
_SNIPPET_WRITES = {Op.ST_LOCAL, Op.ST_FIELD, Op.ST_COMP, Op.ARG}
_SNIPPET_READ = {v: k for k, v in BUILTIN_READ.items()}
#: instructions a snippet may execute before it is a ``snippet`` error
SNIPPET_MAX_STEPS = 100000


class _Scratch:
    """A snippet's frame: a fresh register file over a copy of the
    stopped frame's locals, object and component instance."""

    def __init__(self, f: Frame):
        self.regs = {}
        self.locals = list(f.locals)
        self.obj, self.base = f.obj, f.base
        self.cobj, self.comp, self.cbase = f.cobj, f.comp, f.cbase
        self.args, self.staged = list(f.args or ()), []


def run_snippet(frame: Frame, code, nregs: int, model, memory):
    """Run ``code`` (``Instr``s, procedural ops only) read-only against
    ``frame``; the ``RET`` operand, or None. A memory read (``IMPORT`` of
    a ``read<N>`` builtin) goes through :func:`backdoor_read`."""
    sf = _Scratch(frame)
    pc, steps = 0, 0
    while True:
        if not 0 <= pc < len(code):
            raise StubError("snippet", f"pc {pc} out of the snippet (no RET)")
        steps += 1
        if steps > SNIPPET_MAX_STEPS:
            raise StubError("snippet", f"more than {SNIPPET_MAX_STEPS} instructions")
        ins = code[pc]
        if ins.op == Op.RET:
            return ops_proc._get(sf, ins.args[0]) if ins.args else None
        if ins.op in _SNIPPET_WRITES:
            raise StubError("snippet", f"{ins.op.name} at {pc} writes the frame")
        if ins.op == Op.IMPORT and ins.args and ins.args[0] in _SNIPPET_READ:
            w = _SNIPPET_READ[ins.args[0]]
            if len(ins.args) < 3:
                raise StubError("snippet", f"IMPORT at {pc} has no address")
            v = backdoor_read(memory, ops_proc._get(sf, ins.args[2]), w // 8)
            if ins.flags & INSTR_F_HAS_RET and ins.args[1] < nregs:
                ops_proc._set(sf, ins.args[1], v)
            pc += 1
            continue
        if ins.op not in ops_proc.PROC_OPS:
            raise StubError("snippet", f"{ins.op.name} at {pc} is not procedural")
        if ins.op in (Op.BR, Op.BRZ):
            tgt = ops_proc.exec_branch(sf, ins, model)
            pc = pc + 1 if tgt is None else tgt
        else:
            ops_proc.exec_proc(sf, ins, model)
            pc += 1
        bad = [r for r in sf.regs if not 0 <= r < nregs]
        if bad:
            raise StubError("snippet", f"register r{bad[0]} is outside nregs {nregs}")


#: the ``set_stop_on`` kinds pyvm can stop on (``check``/``sanitizer``
#: need the host to turn those events on: the stub sees the VM's events)
STOP_KINDS = ("check", "error", "sanitizer")


class _WatchedMemory:
    """The platform memory while access watches are set: each read/write
    goes on to it, then to :meth:`Stub._access`."""

    def __init__(self, mem, stub):
        self._mem, self._stub = mem, stub

    def read(self, addr, n):
        v = self._mem.read(addr, n)
        self._stub._access("r", addr, n, v)
        return v

    def write(self, addr, n, data):
        self._mem.write(addr, n, data)
        self._stub._access("w", addr, n, data)

    def __getattr__(self, name):
        return getattr(self._mem, name)


class _Tee:
    """The VM's events sink while attached: each event goes to
    :meth:`Stub._event`, then on to the sink."""

    def __init__(self, sink, stub):
        self._sink, self._stub = sink, stub

    def event(self, kind, coro=0, src_ref=0, detail=None):
        self._stub._event(kind, src_ref, detail)
        return self._sink.event(kind, coro=coro, src_ref=src_ref, detail=detail)

    def __getattr__(self, name):
        return getattr(self._sink, name)


def _stack_depth(frame: Frame) -> int:
    d = 0
    while frame.called and frame.parent is not None:
        frame, d = frame.parent, d + 1
    return d


class Stub:
    """A debug session on one run. ``run`` runs it (default
    ``vm.run(root)``); it starts on ``attach`` and stops before its first
    instruction."""

    def __init__(self, vm, root: Optional[Frame] = None,
                 run: Optional[Callable[[], Any]] = None,
                 on_event: Optional[Callable[[dict], None]] = None):
        if run is None:
            if root is None:
                raise ValueError("Stub needs a root frame or a run callable")
            run = lambda: vm.run(root)
        self.vm = vm
        self._run = run
        self.on_event = on_event
        self.events: "queue.Queue[dict]" = queue.Queue()
        coros = vm.model.coros
        #: id(coroutine) -> its index in ``model.coros``
        self._coro_index = {id(c): i for i, c in enumerate(coros)}
        #: id(coroutine) -> {pc: first src_ref} of its statement starts
        self._stmts = {}
        #: id(coroutine) -> its statement start pcs, ascending
        self._stmt_pcs = {}
        for c in coros:
            pcs = {}
            for pc, sr in getattr(c, "stmts", None) or ():
                pcs.setdefault(pc, sr)
            self._stmts[id(c)] = pcs
            self._stmt_pcs[id(c)] = sorted(pcs)
        self._root = root
        #: thread id -> the innermost frame it last ran (``_before``)
        self._top = {}
        self._cv = threading.Condition()
        self.state = "detached"     # detached|stopped|running|exited
        self._thread: Optional[threading.Thread] = None
        self._breaks = {}           # (coro index, pc) -> [break ids]
        self._step = None           # (kind, thread frame, depth) while stepping
        self._pause = False
        self._terminate = False
        self._resumed = 0           # bumped by every resume, for _before's wait
        #: where the run is stopped: (frame, pc), or None
        self.where = None
        #: the instruction the VM thread runs now (``_before``)
        self._cur = None
        self._watches = []          # [(id, addr, size, rw, value, mask)]
        self._stop_on = {"error"}
        #: a stop an access or event asked for: (reason, hits, detail, src_ref)
        self._pending = None
        self._mem0 = self._sink0 = None
        self._hosted = False
        #: statement starts run so far, in any thread (a run on a
        #: deterministic platform reaches the same tick at the same place)
        self.tick = 0
        #: (tick, thread id, stack depth) of each statement start
        self.history = []
        self._goto = None           # run_to: stop when tick reaches it
        self._quiet = False         # run_to with breaks false: no user stops

    # -- the requests ----------------------------------------------------- #

    def request(self, msg: dict) -> dict:
        """Run one request; returns its reply."""
        seq = msg.get("seq") if isinstance(msg, dict) else None
        try:
            if not isinstance(msg, dict) or not isinstance(msg.get("cmd"), str):
                raise StubError("bad_request", "a request needs a cmd")
            args = msg.get("args") or {}
            if not isinstance(args, dict):
                raise StubError("bad_request", "args must be an object")
            fn = getattr(self, "_cmd_" + msg["cmd"], None)
            if fn is None:
                raise StubError("bad_request", f"unknown cmd {msg['cmd']!r}")
            with self._cv:
                if msg["cmd"] != "attach" and self.state == "detached":
                    raise StubError("bad_request", "not attached")
                if self.state == "exited" and msg["cmd"] not in ("detach", "terminate"):
                    raise StubError("not_stopped", "the run has ended")
                if self.state == "running" and msg["cmd"] not in _WHILE_RUNNING:
                    raise StubError("not_stopped", f"{msg['cmd']} needs a stopped run")
                body = fn(args)
            if msg["cmd"] == "attach":
                self._wait_stopped()
            return {"seq": seq, "ok": True, "body": body}
        except StubError as e:
            return {"seq": seq, "ok": False,
                    "error": {"code": e.code, "message": str(e)}}

    def next_event(self, timeout: Optional[float] = 5.0) -> Optional[dict]:
        """The next event, or None if none comes within ``timeout``."""
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def join(self, timeout: Optional[float] = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _cmd_attach(self, args):
        if self.state != "detached":
            raise StubError("bad_request", "already attached")
        if args.get("version", VERSION) != VERSION:
            raise StubError("bad_request", f"protocol version {VERSION} only")
        model = self.vm.model
        try:
            sha = hashlib.sha256(model.to_bytes()).hexdigest()
        except Exception:       # a hand-built model may not encode
            sha = ""
        # entry false: a run driven by its host (a simulator calling into the
        # VM): no stop before the first instruction, the run is running
        self._hosted = not args.get("entry", True)
        self.state = "running" if self._hosted else "starting"
        self.vm._dbg = self
        if self.vm.events is not None and not isinstance(self.vm.events, _Tee):
            self._sink0 = self.vm.events
            self.vm.events = _Tee(self._sink0, self)
        self._thread = threading.Thread(target=self._main, name="pyvm-debug",
                                        daemon=True)
        self._thread.start()
        return {"version": VERSION, "engine": ENGINE,
                "capabilities": list(CAPABILITIES),
                "image": {"coros": len(model.coros), "sha256": sha}}

    def _wait_stopped(self):
        with self._cv:
            while self.state == "starting":
                self._cv.wait()

    def _unhook(self):
        if self._mem0 is not None:
            self.vm.memory, self._mem0 = self._mem0, None
        if self._sink0 is not None:
            self.vm.events, self._sink0 = self._sink0, None

    def rebind(self, vm) -> None:
        """Follow ``vm``, a new VM on the same model (its host built the
        tree again between runs): breakpoints and watches stay set."""
        old, attached = self.vm, self.vm._dbg is self
        mem = self._mem0 is not None
        self._unhook()
        if attached:
            old._dbg = None
        self.vm = vm
        if not attached:
            return
        vm._dbg = self
        if vm.events is not None and not isinstance(vm.events, _Tee):
            self._sink0 = vm.events
            vm.events = _Tee(self._sink0, self)
        if mem:
            self._mem0 = vm.memory
            vm.memory = _WatchedMemory(self._mem0, self)

    def _cmd_detach(self, args):
        self._breaks, self._step, self._pause = {}, None, False
        self._watches, self._stop_on, self._pending = [], set(), None
        self._unhook()
        self.vm._dbg = None
        self.on_event = None
        if self.state == "stopped":
            self._resume()
        self.state = "detached" if self.state != "exited" else "exited"
        return {}

    def _cmd_terminate(self, args):
        if self.state == "exited":
            return {}
        self._terminate = True
        if self.state == "stopped":
            self._resume()
        return {}

    def _cmd_continue(self, args):
        self._step = None
        self._resume()
        return {}

    def _cmd_run_to(self, args):
        """Run until ``tick`` statement starts have run (stop reason
        "goto"); ``breaks`` false: breakpoints, watches and failing checks
        do not stop the run on the way."""
        tick = args.get("tick")
        if not isinstance(tick, int) or isinstance(tick, bool) or tick <= self.tick:
            raise StubError("bad_request",
                            f"run_to needs an int tick after the current {self.tick}")
        self._step = None
        self._goto = tick
        self._quiet = args.get("breaks", True) is False
        self._resume()
        return {}

    def _cmd_history(self, args):
        """The statement starts run so far (``since``: after that tick)."""
        since = args.get("since", 0)
        if not isinstance(since, int):
            raise StubError("bad_request", "since must be an int")
        return {"tick": self.tick,
                "history": [list(h) for h in self.history[max(since, 0):]]}

    def _cmd_stop(self, args):
        if self.state != "running":
            raise StubError("not_running", "the run is already stopped")
        self._pause = True
        return {}

    def _cmd_step(self, args):
        kind = args.get("kind", "stmt")
        if kind not in ("stmt", "into", "out"):
            raise StubError("bad_request", f"unknown step kind {kind!r}")
        tid = args.get("thread")
        if not isinstance(tid, int):
            raise StubError("bad_request", "step needs a thread")
        thread = self._thread_by_id(tid)
        self._step = (kind, thread, _stack_depth(self._innermost(thread)))
        self._resume()
        return {}

    # -- inspection (stopped only) ----------------------------------------- #

    def _threads(self):
        """The live threads, by id: the root, the frames ``SPAWN``/``INVOKE``
        started under it, and those started inside a called function (found
        through the frames ``_before`` has seen)."""
        seen, todo = {}, []
        if self._root is not None:
            todo.append(self._root)
        for f in self._top.values():
            while f is not None:
                todo.append(f)
                f = f.parent if f.called else None
        while todo:
            f = todo.pop()
            if f.id in seen:
                continue
            seen[f.id] = f
            todo.extend(f.children)
        return {i: f for i, f in sorted(seen.items())
                if not f.called and not f.done and not f.cancelled}

    def _thread_by_id(self, tid):
        if not isinstance(tid, int):
            raise StubError("bad_request", "needs an int thread")
        t = self._threads().get(tid)
        if t is None:
            raise StubError("no_thread", f"no live thread {tid}")
        return t

    def _innermost(self, thread):
        if self.where is not None and thread_of(self.where[0]) is thread:
            return self.where[0]
        f = self._top.get(thread.id, thread)
        while f.done and f.called and f.parent is not None:
            f = f.parent
        return f

    def _stack(self, thread):
        """The thread's frames, innermost first."""
        f, out = self._innermost(thread), []
        while True:
            out.append(f)
            if not f.called or f.parent is None:
                return out
            f = f.parent

    def _pc(self, frame):
        if self.where is not None and self.where[0] is frame:
            return self.where[1]
        return frame.pc

    def _src_ref_at(self, coro, pc):
        """The statement containing ``pc``: the last start at or before it."""
        pcs = self._stmt_pcs.get(id(coro)) or ()
        i = bisect.bisect_right(pcs, pc)
        return self._stmts[id(coro)][pcs[i - 1]] if i else 0

    def _frame_by_id(self, fid):
        if not isinstance(fid, int):
            raise StubError("bad_request", "needs an int frame")
        for t in self._threads().values():
            for f in self._stack(t):
                if f.id == fid:
                    return f
        raise StubError("no_frame", f"no live frame {fid}")

    def _cmd_threads(self, args):
        ready = {thread_of(f).id for f in self.vm.sched._ready}
        here = thread_of(self.where[0]).id
        return {"threads": [
            {"id": i, "coro": self._coro_index.get(id(t.coro), -1),
             "name": t.coro.name,
             "state": ("running" if i == here else
                       "ready" if i in ready else "waiting"),
             **({"inferred_for": t.inferred_for}
                if t.inferred_for is not None else {})}
            for i, t in self._threads().items()]}

    def _cmd_frames(self, args):
        thread = self._thread_by_id(args.get("thread"))
        out = []
        for f in self._stack(thread):
            pc = self._pc(f)
            out.append({"id": f.id, "coro": self._coro_index.get(id(f.coro), -1),
                        "name": f.coro.name, "pc": pc,
                        "src_ref": self._src_ref_at(f.coro, pc), "depth": f.depth})
        return {"frames": out}

    def _cmd_read_frame(self, args):
        f = self._frame_by_id(args.get("frame"))
        pc = self._pc(f)
        lvars = getattr(f.coro, "locals", None) or ()
        locs = []
        for v in lvars:
            if v.pc_start <= pc < v.pc_end and v.slot < len(f.locals):
                locs.append({"name": v.name, "value": f.locals[v.slot],
                             "width": v.width, "signed": v.signed, "kind": v.kind})
        body = {"locals": locs, "args": list(f.args or ()), "this": None}
        comps = getattr(self.vm.model, "components", None)
        if comps is not None and f.comp is not None and f.comp < len(comps.paths):
            body["this"] = {"comp": f.comp, "path": comps.paths[f.comp]}
        if not lvars:
            body["regs"] = {str(k): v for k, v in sorted(f.regs.items())
                            if isinstance(v, int)}
        return body

    def _cell(self, v, ck, pre=""):
        """One heap cell as ``{value, kind[, text]}`` (keys prefixed *pre*):
        kind ``int``, ``string`` (with its text), a nested collection's
        ``list``/``map``/``set``, or ``empty`` (collection handle 0)."""
        heap = self.vm.heap
        if ck == HEAP_CELL_INT:
            return {pre + "value": v, pre + "kind": "int"}
        if ck == HEAP_CELL_STR:
            return {pre + "value": v, pre + "kind": "string",
                    pre + "text": heap.text(v).decode(errors="replace")}
        kind = _KIND_NAMES[heap.obj(v).kind] if v else "empty"
        return {pre + "value": v, pre + "kind": kind}

    def _cmd_read_heap(self, args):
        """The contents of heap object ``handle`` (a list/map/set local's
        value, or a string's): ``{kind, items: [{value, kind[, text]}]}``;
        a map item also has ``key``/``key_kind``/``key_text``; a string has
        ``text``. Handle 0 is an empty collection; with ``kind: "string"``
        the handle is any string value (a model-string index too)."""
        h = args.get("handle")
        if not isinstance(h, int):
            raise StubError("bad_request", "read_heap needs an int handle")
        try:
            if args.get("kind") == "string":
                return {"kind": "string", "items": [],
                        "text": self.vm.heap.text(h).decode(errors="replace")}
            if not h:
                return {"kind": "empty", "items": []}
            o = self.vm.heap.obj(h)
            if o.kind == HEAPK_STR:
                return {"kind": "string", "items": [],
                        "text": o.data.decode(errors="replace")}
            if o.kind == HEAPK_LIST:
                items = [self._cell(c, o.cell) for c in o.data]
            elif o.kind == HEAPK_SET:
                items = [self._cell(c, o.cell) for c in o.data.values()]
            else:
                items = [dict(self._cell(k, o.key, "key_"), **self._cell(c, o.cell))
                         for k, c in o.data.values()]
        except VMError as e:
            raise StubError("bad_handle", str(e))
        return {"kind": _KIND_NAMES[o.kind], "items": items}

    def _cmd_read_comp(self, args):
        comps = getattr(self.vm.model, "components", None)
        if comps is None:
            raise StubError("no_comp", "the image has no component tree")
        comp = args.get("comp")
        if comp is None and isinstance(args.get("path"), str):
            comp = comps.paths.index(args["path"]) if args["path"] in comps.paths else -1
        if not isinstance(comp, int) or not 0 <= comp < len(comps.paths):
            raise StubError("no_comp", f"no component {args.get('comp', args.get('path'))!r}")
        # An instance's fields are its slots up to the next instance's base
        # (guess: a child's range follows its parent's own fields).
        lo = comps.bases[comp]
        hi = min([b for b in comps.bases if b > lo], default=len(comps.names))
        cobj = getattr(self.vm, "cobj", None) or self.where[0].cobj
        # each slot's type (LAYOUT): a list/map/set/string field's value is
        # a heap handle, read with read_heap
        types = getattr(comps, "slot_types", None) or []
        fields = []
        for s in range(lo, hi):
            v = cobj.values[s] if cobj is not None and s < len(cobj.values) else 0
            t = types[s] if s < len(types) else {}
            fields.append({"name": comps.names[s], "value": v,
                           "width": t.get("width") or 64,
                           "signed": bool(t.get("signed")),
                           "kind": t.get("kind") or "int"})
        return {"comp": comp, "path": comps.paths[comp], "fields": fields}

    def _cmd_backdoor_read(self, args):
        addr, width = args.get("addr"), args.get("width")
        if not isinstance(addr, int) or width not in (8, 16, 32, 64):
            raise StubError("bad_request", "backdoor_read needs int addr, width 8|16|32|64")
        return {"value": backdoor_read(self.vm.memory, addr, width // 8)}

    def _cmd_exec_snippet(self, args):
        f = self._frame_by_id(args.get("frame"))
        code, nregs = args.get("code"), args.get("nregs")
        if not isinstance(code, list) or not isinstance(nregs, int) or nregs < 0:
            raise StubError("bad_request", "exec_snippet needs code [Ins], int nregs")
        try:
            ins = [Instr(Op[c["op"]], c.get("args", ()), c.get("imm", 0), c.get("flags", 0))
                   for c in code]
        except (KeyError, TypeError, ValueError) as e:
            raise StubError("bad_request", f"bad instruction: {e}")
        try:
            v = run_snippet(f, ins, nregs, self.vm.model, self.vm.memory)
        except VMError as e:
            raise StubError("snippet", str(e))
        return {"value": v}

    def _cmd_set_access_watch(self, args):
        ws = args.get("watches")
        if not isinstance(ws, list):
            raise StubError("bad_request", "set_access_watch needs watches")
        watches, out = [], []
        for w in ws:
            if not isinstance(w, dict) or not isinstance(w.get("id"), int):
                raise StubError("bad_request", "a watch needs an int id")
            addr, size, rw = w.get("addr"), w.get("size"), w.get("rw", "rw")
            ok = (isinstance(addr, int) and isinstance(size, int) and size > 0
                  and rw in ("r", "w", "rw"))
            if ok:
                mask = w.get("mask", (1 << 64) - 1)
                watches.append((w["id"], addr, size, rw, w.get("value"), mask))
            out.append({"id": w["id"], "verified": ok})
        self._watches = watches
        if watches and self._mem0 is None:
            self._mem0 = self.vm.memory
            self.vm.memory = _WatchedMemory(self._mem0, self)
        elif not watches and self._mem0 is not None:
            self.vm.memory, self._mem0 = self._mem0, None
        return {"watches": out}

    def _cmd_set_stop_on(self, args):
        kinds = args.get("kinds")
        if not isinstance(kinds, list) or not all(isinstance(k, str) for k in kinds):
            raise StubError("bad_request", "set_stop_on needs kinds [str]")
        self._stop_on = {k for k in kinds if k in STOP_KINDS}
        return {"kinds": sorted(self._stop_on)}

    def _cmd_set_pc_breaks(self, args):
        breaks = args.get("breaks")
        if not isinstance(breaks, list):
            raise StubError("bad_request", "set_pc_breaks needs breaks")
        coros = self.vm.model.coros
        table, out = {}, []
        for b in breaks:
            if not isinstance(b, dict) or not all(
                    isinstance(b.get(k), int) for k in ("id", "coro", "pc")):
                raise StubError("bad_request", "a break needs int id, coro, pc")
            c, pc = b["coro"], b["pc"]
            if not 0 <= c < len(coros):
                out.append({"id": b["id"], "verified": False,
                            "message": f"no coroutine {c}"})
            elif not 0 <= pc < len(coros[c].code):
                out.append({"id": b["id"], "verified": False,
                            "message": f"pc {pc} out of range"})
            else:
                table.setdefault((c, pc), []).append(b["id"])
                out.append({"id": b["id"], "verified": True})
        self._breaks = table
        return {"breaks": out}

    def _resume(self):
        self.state = "running"
        self.where = None
        self._resumed += 1
        self._cv.notify_all()

    # -- the VM thread ---------------------------------------------------- #

    def _emit(self, event: str, body: dict) -> None:
        msg = {"event": event, "body": body}
        self.events.put(msg)
        if self.on_event is not None:
            self.on_event(msg)

    def _main(self):
        try:
            self._run()
        except Terminated:
            self._exit({"status": "terminated"})
        except VMError as e:
            self._error_stop(str(e))
            self._exit({"status": "error", "message": str(e)})
        except Exception as e:      # an engine bug still ends the session
            msg = f"{type(e).__name__}: {e}"
            self._error_stop(msg)
            self._exit({"status": "error", "message": msg})
        else:
            if self._terminate:
                self._exit({"status": "terminated"})
            else:
                self._exit({"status": "ok"})

    def _error_stop(self, message):
        """An engine error with ``error`` in the stop set: stop where the
        run was, for inspection; any resume then ends the run."""
        with self._cv:
            if (self.vm._dbg is not self or "error" not in self._stop_on
                    or self._cur is None or self._terminate):
                return
            frame, pc = self._cur
            code = frame.coro.code
            sr = (code[pc].src_ref if pc < len(code) else 0) or 0
            self._stop(frame, pc, ("error", None, {"error": message}, sr))
            mark = self._resumed
            while self._resumed == mark:
                self._cv.wait()

    def _access(self, rw, addr, n, data):
        """A memory access (``_WatchedMemory``) on the VM thread."""
        with self._cv:
            hits = [i for i, a, size, wrw, value, mask in self._watches
                    if a < addr + n and addr < a + size and rw in wrw
                    and (value is None or (data & mask) == (value & mask))]
            if hits and self._pending is None and self._cur is not None:
                frame, pc = self._cur
                code = frame.coro.code
                sr = (code[pc].src_ref if pc < len(code) else 0) or 0
                self._pending = ("watch", hits, {"rw": rw, "width": 8 * n,
                                                 "addr": addr, "data": data}, sr)

    def _event(self, kind, src_ref, detail):
        """A VM event (``_Tee``): a failing check or sanitizer stops the
        run when its kind is in the stop set."""
        if kind not in ("check", "sanitizer") or kind not in self._stop_on:
            return
        if isinstance(detail, dict) and detail.get("verdict", "fail") != "fail":
            return
        with self._cv:
            if self._pending is None:
                self._pending = (kind, None, dict(detail or {}), src_ref or 0)

    def _exit(self, body):
        with self._cv:
            observed = self.vm._dbg is self
            self.vm._dbg = None
            self._unhook()
            self.state = "exited"
            self.where = None
            self._cv.notify_all()
        if observed:
            self._emit("exited", body)

    def _before(self, frame: Frame, pc: int) -> None:
        """Called by the VM before the instruction at ``pc`` of ``frame``;
        returns when the run may go on."""
        with self._cv:
            if self._terminate:
                raise Terminated()
            self._top[thread_of(frame).id] = frame
            self._cur = (frame, pc)
            if pc in self._stmts.get(id(frame.coro), ()):
                self.tick += 1
                self.history.append((self.tick, thread_of(frame).id,
                                     _stack_depth(frame)))
            if self.state == "starting":
                if self._root is None:
                    root = frame
                    while root.parent is not None:
                        root = root.parent
                    self._root = root
                self._stop(frame, pc, None)
            else:
                why = self._why(frame, pc)
                if why[0] is None:
                    return
                if self._hosted:            # a host's calls each have a root
                    root = frame
                    while root.parent is not None:
                        root = root.parent
                    self._root = root
                self._stop(frame, pc, why)
            mark = self._resumed
            while self._resumed == mark:
                self._cv.wait()
            if self._terminate:
                raise Terminated()

    def _why(self, frame, pc):
        if self._goto is not None and self.tick >= self._goto:
            return "goto", None
        if self._quiet:
            self._pending = None
            if self._pause:
                self._pause = False
                return "pause", None
            return None, None
        if self._pending is not None and not (
                self._pending[0] == "watch" and frame.parked):
            # a watched access the host served stops once its parked
            # instruction has re-run, as a direct access stops after it
            why, self._pending = self._pending, None
            return why
        if self._pause:
            self._pause = False
            return "pause", None
        if self._breaks:
            hits = self._breaks.get((self._coro_index.get(id(frame.coro)), pc))
            if hits:
                return "breakpoint", hits
        if self._step is not None and pc in self._stmts.get(id(frame.coro), ()):
            kind, thread, depth = self._step
            if thread.done or thread.cancelled:
                return "step", None
            if thread_of(frame) is thread:
                d = _stack_depth(frame)
                if (kind == "into" or (kind == "stmt" and d <= depth)
                        or (kind == "out" and d < depth)):
                    return "step", None
        return None, None

    def _stop(self, frame, pc, why):
        """Stop at (frame, pc) under the lock; ``why`` None: the start."""
        self.state = "stopped"
        self.where = (frame, pc)
        self._goto, self._quiet = None, False
        self._cv.notify_all()
        if why is None:
            return
        reason, hits = why[:2]
        detail = why[2] if len(why) > 2 else {}
        if reason == "step":
            detail = {"kind": self._step[0]}
            self._step = None
        code = frame.coro.code
        sr = self._stmts.get(id(frame.coro), {}).get(pc)
        if sr is None:
            sr = (code[pc].src_ref if pc < len(code) else 0) or 0
        if len(why) > 3:
            sr = why[3]
        body = {"reason": reason, "thread": thread_of(frame).id,
                "frame": frame.id, "coro": self._coro_index.get(id(frame.coro), -1),
                "pc": pc, "src_ref": sr, "detail": detail, "tick": self.tick}
        if hits:
            body["hits"] = list(hits)
        self._emit("stopped", body)
