"""
vm.py -- the dispatch loop (P1-9).

The ``switch``-style interpreter over the flat CODE stream. It resumes one frame
and runs it until the next suspend op (or completion), mirroring the native
``while zsp_timebase_run`` shape `[D§5]`: procedural ops execute in-line via
:mod:`.ops_proc`; orchestration ops go through :mod:`.ops_orch`, which drives the
:class:`.scheduler.Scheduler` and decides whether the frame keeps running or
suspends. ``RET`` (and falling off the end of a coroutine) completes the frame and
wakes anyone JOINing / blocking-INVOKING on it.

The VM owns the run-scope state the handlers need -- the scheduler, the model
(code + const pool + in-memory problem table), the SOLVE/IMPORT externs, and the
trace sink -- and exposes :meth:`spawn_child` so orchestration handlers can fork
child frames with correctly forked seeds (determinism spec fork rule).
"""

from typing import Callable, List, Optional

from ..model import Op
from ..trace import events as ev
from ..trace.sink import NullSink, TraceSink
from . import ops_proc, ops_orch
from .defuse import abs_comp as _abs_comp, defuse
from .ops_proc import VMError
from .scheduler import Frame, Scheduler

#: message() verbosity levels (std_pkg::message_verbosity_e, 21.1.3).
VERBOSITY_NONE, VERBOSITY_LOW, VERBOSITY_MEDIUM, VERBOSITY_HIGH, VERBOSITY_FULL = range(5)
from .extern import Obj, SolveBackend, ImportProvider, FixedSolveBackend, Memory, \
    NativeBlobBackend, RecordingImportProvider
from .solve_cache import SolveCache


class VM:
    """Executes a :class:`..model.ZbcModel` under a :class:`Scheduler`."""

    def __init__(self, model, solve_backend: Optional[SolveBackend] = None,
                 import_provider: Optional[ImportProvider] = None,
                 sink: Optional[TraceSink] = None,
                 out: Optional[Callable[[str], None]] = None,
                 verbosity: int = VERBOSITY_MEDIUM,
                 memory=None, solve_cache=None, on_solve=None,
                 events: Optional[TraceSink] = None):
        self.model = model
        #: where debug events go (``trace/events.py``); never the trace sink,
        #: so the orchestration trace is the same with events on or off
        self.events = events
        #: id(coroutine) -> its event mask; empty while no events are on,
        #: which is all ``run_frame`` tests on a frame's entry
        self._ev = {}
        #: the attached debug stub (``debug.Stub``), or None; while one is,
        #: every frame runs in ``_run_frame_ev``, which calls it per instruction
        self._dbg = None
        #: id(coroutine) -> {pc: (src_ref, ...)} of its statement starts
        self._ev_stmts = {}
        #: id(coroutine) -> {pc: stmt event detail} where the coverage table
        #: (``CoroDescriptor.cov``) names the statement
        self._ev_paths = {}
        #: id(coroutine) -> {pc: (src_ref, branch event detail)} of its arms
        self._ev_arms = {}
        #: id(coroutine) -> {pc: (src_ref, channel, op, value reg, ok reg)}
        #: of its channel operations (``CoroDescriptor.chans``)
        self._ev_chans = {}
        #: id(coro) -> {IMPORT pc: register instance, from the frame's}
        #: (``CoroDescriptor.regs``)
        self._ev_regs = {}
        #: id(coro) -> the pcs of its register reads whose value is dropped
        #: (``CoroDescriptor.discards``)
        self._ev_discards = {}
        #: the host events (``test.start/end``) on
        self._ev_host = 0
        #: id(coroutine) -> hook(vm, frame) -> (detail, src_ref) for each
        #: function the host named a check (``on_check``)
        self._ev_checks = {}
        #: fn_id -> import name, built on the first ``import`` event
        self._import_names = None
        #: called-form coroutine index -> (function qname, host callable)
        #: (``bind``); empty, which is all a CALL tests, in a normal run
        self.bindings = {}
        #: called with the frame after each solve has written its values
        #: back (telemetry; None in a normal run)
        self.on_solve = on_solve
        #: rand variable name -> (lo, hi) every activation's solves keep to
        #: (a property test's shrink; ``Activation.bounds``); None in a normal run
        self.solve_bounds = None
        #: what the entry action's solves chose, by rand variable name
        #: (``set_entry``; its activation's ``solved`` when it has one)
        self.solved = {}
        #: the entry frame (``set_entry``): its plain SOLVEs keep to
        #: ``solve_bounds`` and are recorded in ``solved``
        self.bound_frame = None
        self._bounded = {}
        #: compiled solve problems, reused across the run's solves
        #: (``solve_cache.SolveCache``)
        self.solves = solve_cache if solve_cache is not None else SolveCache()
        #: the solver for a SOLVE whose problem carries a dv-solve blob
        self.blob_backend = NativeBlobBackend(self.solves)
        #: the platform memory a memory builtin reaches (``extern.Memory``)
        self.memory = memory if memory is not None else Memory()
        self.sched = Scheduler()
        #: set by a SPIN yield that made no progress (see _drain)
        self.stuck = False
        self.solve_backend = solve_backend or FixedSolveBackend()
        self.import_provider = import_provider or RecordingImportProvider()
        self.sink = sink or NullSink()
        #: where message() lines go: one call per line, without the newline
        self.out = out or (lambda line: print(line, flush=True))
        #: the run's message verbosity (21.1.3): NONE=0 LOW=1 MEDIUM=2 HIGH=3 FULL=4
        self.verbosity = verbosity
        #: the host a call yields to (``host.HostCall``; docs/spec/host-abi.md),
        #: or None: accesses, imports and waits are answered in place
        self.host = None

    def _new_obj(self, coro_index: int) -> Optional[Obj]:
        """A fresh action object for a traversal of an action coroutine."""
        layout = getattr(self.model, "obj_layouts", {}).get(coro_index)
        return Obj(field_names=layout) if layout is not None else None

    # -- frame construction ----------------------------------------------- #

    def call_child(self, parent: Frame, coro_index: int) -> Frame:
        """A CALL's callee: the caller's object, node, base and instance, and
        its seed stream (a function draws nothing, so a call forks none and
        leaves the caller's child numbering as an inlined body would)."""
        child = self.sched.new_frame(self.model.coros[coro_index], seed=parent.seed,
                                     obj=parent.obj, parent=parent)
        child.base, child.act, child.node, child.site = (
            parent.base, parent.act, parent.node, parent.site)
        child.cobj, child.comp, child.cbase = parent.cobj, parent.comp, parent.cbase
        child.depth, child.called = parent.depth + 1, True
        return child

    def spawn_child(self, parent: Frame, coro_index: int, obj: Optional[Obj] = None,
                    node: Optional[tuple] = None, start_pc: int = 0) -> Frame:
        """Create a child frame with a seed forked from the parent (D§15.1).

        *node* = ``(child_base, site)`` for a traversal of a node of the
        parent's activation: the child runs on the parent's object at its
        node's base (P1-D1).
        """
        coro = self.model.coros[coro_index]
        child_seed = parent.seed.fork(parent.next_child_index())
        if node is not None:
            child = self.sched.new_frame(coro, seed=child_seed, obj=parent.obj,
                                         parent=parent)
            child.pc = start_pc
            child.cobj = parent.cobj
            child_base, local_site = node
            act = parent.act
            if act is not None:
                site = act.t.site(parent.node, local_site)
                child.act, child.site = act, site
                child.node = act.t.sites[site][1]
                child.base = act.t.nodes[child.node].base
                rel = act.t.nodes[child.node].comp_rel
                self.set_comp(child, None if rel is None or parent.comp is None
                              else parent.comp + rel)
                act.enter_node(child.node, site)
                child.is_node = True
            else:
                child.base = parent.base + child_base
                self.set_comp(child, parent.comp)
            parent.children.append(child)
            return child
        inherit = False
        if obj is None:
            # An action traversal outside an activation gets its own object; a
            # synthesized PAR/SELECT branch (no layout) runs on its parent's,
            # at its parent's node.
            obj = self._new_obj(coro_index)
            inherit = obj is None
        child = self.sched.new_frame(
            coro, seed=child_seed,
            obj=obj if obj is not None else parent.obj,
            parent=parent,
        )
        if inherit:
            child.base, child.act = parent.base, parent.act
            child.node, child.site = parent.node, parent.site
        child.cobj = parent.cobj
        child.comp, child.cbase = parent.comp, parent.cbase
        parent.children.append(child)
        return child

    def set_comp(self, frame: Frame, comp: Optional[int]) -> None:
        """Run *frame* in component instance *comp* (None: not chosen yet)."""
        frame.comp = comp
        comps = getattr(self.model, "components", None)
        frame.cbase = comps.base(comp) if comps is not None else 0

    def root_frame(self, coro_index: int, seed: int, obj: Optional[Obj] = None) -> Frame:
        table = getattr(self.model, "activations", {}).get(coro_index)
        if obj is None:
            obj = (Obj(field_names=table.names) if table is not None
                   else self._new_obj(coro_index))
        frame = self.sched.new_frame(self.model.coros[coro_index], seed=seed, obj=obj)
        comps = getattr(self.model, "components", None)
        if comps is not None:
            frame.cobj = self.cobj = comps.new_obj()
            self._seed = seed
        if table is not None:
            from .activation import Activation
            frame.act = Activation(table, obj, self.solves, frame.cobj,
                                   self.solve_bounds)
            self.set_comp(frame, table.nodes[0].comp_rel)
            frame.act.enter_node(0, None)
            frame.is_node = True
        return frame

    def set_entry(self, frame: Frame) -> None:
        """Make *frame* (from :meth:`root_frame`) the run's entry action: its
        own SOLVEs (outside any cone) keep to ``solve_bounds`` too, and
        ``solved`` gathers what they and its activation chose (a property
        test's shrink, P5.5)."""
        self.solved = frame.act.solved if frame.act is not None else {}
        self._bounded, self.bound_frame = {}, frame

    def construct_components(self) -> None:
        """Run the coroutine that constructs the component tree, to
        completion (LRM 20.1.3: before the root action's pre_solve)."""
        comps = getattr(self.model, "components", None)
        if comps is None or comps.init_coro is None:
            return
        frame = self.sched.new_frame(self.model.coros[comps.init_coro],
                                     seed=getattr(self, "_seed", 0))
        frame.cobj = self.cobj
        self.set_comp(frame, 0)
        self._drain(frame)

    # -- debug events ----------------------------------------------------- #

    def bind(self, qname: str, fn: Optional[Callable]) -> None:
        """Run host callable ``fn`` in place of function ``qname``
        (``<comp type>::<fn>``) on every CALL of it from now on; None
        unbinds it. ``fn`` gets the raw argument values and returns the
        result (None: 0). Only a called form can be bound: a debug lowering
        calls each function through one; a release one inlines most. The
        bindings are this VM's, so they last one run."""
        coros = self.model.coros
        called = {ins.args[0] for c in coros for ins in c.code if ins.op == Op.CALL}
        idx = [i for i in called if getattr(coros[i], "fn_qname", "") == qname]
        if not idx:
            raise KeyError(f"{qname} is never called through a CALL "
                           f"(bindings need a debug lowering)")
        for i in idx:
            if fn is None:
                self.bindings.pop(i, None)
            else:
                self.bindings[i] = (qname, fn)

    def watch(self, mask: int, coros=None) -> None:
        """Raise the events in ``mask`` (``trace.events.EV_*``) for
        ``coros`` (indices or names; None: every coroutine) from now on; a
        mask of 0 turns them off. Events go to the ``events`` sink."""
        if mask and self.events is None:
            raise ValueError("events need an events sink (VM(events=...))")
        all_coros = self.model.coros
        if coros is None:
            picked = list(all_coros)
        else:
            by_name = {c.name: c for c in all_coros}
            picked = [all_coros[c] if isinstance(c, int) else by_name[c]
                      for c in coros]
        for c in picked:
            if mask:
                self._ev[id(c)] = mask
                if mask & ev.EV_STMT and id(c) not in self._ev_stmts:
                    pcs = {}
                    for pc, sr in getattr(c, "stmts", None) or ():
                        pcs[pc] = pcs.get(pc, ()) + (sr,)
                    self._ev_stmts[id(c)] = pcs
                    paths = {
                        pc: ({"pc": pc, "path": path, "scope": scope} if scope
                             else {"pc": pc, "path": path})
                        for k, pc, scope, path, _, _ in getattr(c, "cov", None) or ()
                        if k == "stmt"}
                    for pc, (defs, uses) in defuse(c).items():
                        d = paths.setdefault(pc, {"pc": pc})
                        if defs:
                            d["defs"] = defs
                        if uses:
                            d["uses"] = uses
                    self._ev_paths[id(c)] = paths
                if mask & ev.EV_BRANCH and id(c) not in self._ev_arms:
                    arms = {}
                    for k, pc, scope, path, sr, implicit in getattr(c, "cov", None) or ():
                        if k == "arm":
                            branch, _, arm = path.rpartition("/")
                            d = {"pc": pc, "branch": branch, "arm": arm}
                            if scope:
                                d["scope"] = scope
                            if implicit:
                                d["implicit"] = True
                            arms[pc] = (sr, d)
                    self._ev_arms[id(c)] = arms
                if mask & ev.EV_ACCESS and id(c) not in self._ev_regs:
                    self._ev_regs[id(c)] = dict(getattr(c, "regs", None) or ())
                    self._ev_discards[id(c)] = set(getattr(c, "discards", None) or ())
                if mask & ev.EV_CHAN and id(c) not in self._ev_chans:
                    self._ev_chans[id(c)] = {
                        pc: rest for pc, *rest in getattr(c, "chans", None) or ()}
            else:
                self._ev.pop(id(c), None)
        self._ev_host = (mask & ev.EV_HOST) if coros is None else (
            self._ev_host | (mask & ev.EV_HOST))

    def on_check(self, coro, hook) -> None:
        """Name the function ``coro`` (index or name) a check: when a watched
        frame of it starts with ``EV_CHECK`` on, ``hook(vm, frame)`` returns
        ``(detail, src_ref)`` for a ``check`` event (``detail`` None: none)."""
        c = (self.model.coros[coro] if isinstance(coro, int)
             else {c.name: c for c in self.model.coros}[coro])
        self._ev_checks[id(c)] = hook

    def import_name(self, fn_id: int) -> Optional[str]:
        """The user import ``fn_id`` names (``model.imports``), or None."""
        if self._import_names is None:
            self._import_names = {sig.get("fn_id"): n for n, sig in
                                  (getattr(self.model, "imports", None) or {}).items()}
        return self._import_names.get(fn_id)

    def host_event(self, kind: str, detail=None, src_ref: int = 0) -> None:
        """An event the host raises (``test.start/end``, ``notify``,
        ``sanitizer``), if its bit is on."""
        if self._ev_host & ev.BITS[kind] and self.events is not None:
            self.events.event(kind, src_ref=src_ref, detail=detail)

    def reg_id(self, frame: Frame) -> Optional[int]:
        """The register instance (``model.components.paths`` index) the
        ``IMPORT`` *frame* is executing accesses, or None (no debug table,
        or not a register access)."""
        rel = self._ev_regs.get(id(frame.coro), {}).get(frame.pc - 1)
        if rel is None or frame.comp is None:
            return None
        return frame.comp + rel

    def discarded(self, frame: Frame) -> bool:
        """Whether the register read the ``IMPORT`` *frame* is executing
        drops its value (debug lowering)."""
        return frame.pc - 1 in self._ev_discards.get(id(frame.coro), ())

    def _ev_chan(self, frame: Frame, c) -> None:
        """The ``chan`` event of a channel operation that just ended."""
        sr, name, op, val, ok = c
        d = {"channel": name, "op": op}
        if ok >= 0:
            d["ok"] = bool(frame.regs[ok])
        if ok < 0 or d["ok"]:
            d["value"] = frame.regs[val]
        self.events.event(ev.CHAN, coro=frame.id, src_ref=sr, detail=d)

    def _run_frame_ev(self, frame: Frame, m: int) -> None:
        """``run_frame`` for a coroutine with events on: the same loop, plus
        a statement test per instruction and the call/return events."""
        code = frame.coro.code
        n = len(code)
        frame.resume_pc = frame.pc
        sink = self.events
        stmts = self._ev_stmts.get(id(frame.coro)) if m & ev.EV_STMT else None
        chans = self._ev_chans.get(id(frame.coro)) if m & ev.EV_CHAN else None
        arms = self._ev_arms.get(id(frame.coro)) if m & ev.EV_BRANCH else None
        paths = self._ev_paths.get(id(frame.coro)) if stmts else None
        calls = bool(m & ev.EV_CALL) and frame.called
        if calls and frame.pc == 0:
            sink.event(ev.CALL, coro=frame.id, src_ref=frame.coro.src_ref,
                       detail={"fn": frame.coro.name, "depth": frame.depth,
                               "args": list(frame.args or ())})
        if m & ev.EV_CHECK and frame.pc == 0 and frame.called:
            hook = self._ev_checks.get(id(frame.coro))
            if hook is not None:
                detail, sr = hook(self, frame)
                if detail is not None:
                    sink.event(ev.CHECK, coro=frame.id, src_ref=sr, detail=detail)
        ins = None
        try:
            while frame.pc < n:
                pc = frame.pc
                if self._dbg is not None:
                    self._dbg._before(frame, pc)
                if stmts:
                    srs = stmts.get(pc)
                    if srs is not None:
                        d = paths.get(pc) if paths else None
                        d = dict(d) if d else {"pc": pc}
                        if frame.cbase and ("defs" in d or "uses" in d):
                            _abs_comp(d, frame.cbase)
                        for sr in srs:
                            sink.event(ev.STMT, coro=frame.id, src_ref=sr,
                                       detail=dict(d))
                if arms:
                    a = arms.get(pc)
                    if a is not None:
                        sink.event(ev.BRANCH, coro=frame.id, src_ref=a[0],
                                   detail=dict(a[1]))
                if chans:
                    c = chans.get(pc)
                    if c is not None:
                        self._ev_chan(frame, c)
                ins = code[pc]
                op = ins.op

                if op == Op.RET:
                    if ins.args:
                        frame.retval = ops_proc._get(frame, ins.args[0])
                    frame.pc += 1
                    if calls:
                        self._ev_return(frame)
                    self._complete(frame)
                    return

                if op in ops_proc.PROC_OPS:
                    if op in (Op.BR, Op.BRZ):
                        target = ops_proc.exec_branch(frame, ins, self.model)
                        frame.pc = target if target is not None else frame.pc + 1
                    else:
                        ops_proc.exec_proc(frame, ins, self.model)
                        frame.pc += 1
                    continue

                if op in ops_orch.ORCH_OPS:
                    frame.pc += 1
                    action = ops_orch.exec_orch(self, frame, ins)
                    if action.suspend:
                        return
                    continue

                raise VMError(f"unhandled opcode {op.name} at pc={frame.pc}")
        except VMError as e:
            if not e.src_ref:
                self._locate(e, frame, ins)
            raise

        if calls:
            self._ev_return(frame)
        self._complete(frame)

    def _ev_return(self, frame: Frame) -> None:
        detail = {"fn": frame.coro.name, "depth": frame.depth}
        if isinstance(frame.retval, int):
            detail["retval"] = frame.retval
        self.events.event(ev.RETURN, coro=frame.id, src_ref=frame.coro.src_ref,
                          detail=detail)

    # -- the dispatch loop ------------------------------------------------ #

    def run_frame(self, frame: Frame) -> None:
        """Run ``frame`` from its pc until it suspends or completes."""
        if self._ev:
            m = self._ev.get(id(frame.coro))
            if m:
                return self._run_frame_ev(frame, m)
        if self._dbg is not None:
            return self._run_frame_ev(frame, 0)
        code = frame.coro.code
        n = len(code)
        frame.resume_pc = frame.pc
        ins = None
        try:
            while frame.pc < n:
                ins = code[frame.pc]
                op = ins.op

                if op == Op.RET:
                    if ins.args:
                        frame.retval = ops_proc._get(frame, ins.args[0])
                    frame.pc += 1
                    self._complete(frame)
                    return

                if op in ops_proc.PROC_OPS:
                    if op in (Op.BR, Op.BRZ):
                        target = ops_proc.exec_branch(frame, ins, self.model)
                        frame.pc = target if target is not None else frame.pc + 1
                    else:
                        ops_proc.exec_proc(frame, ins, self.model)
                        frame.pc += 1
                    continue

                if op in ops_orch.ORCH_OPS:
                    frame.pc += 1  # resume past the suspend point
                    action = ops_orch.exec_orch(self, frame, ins)
                    if action.suspend:
                        return
                    continue

                raise VMError(f"unhandled opcode {op.name} at pc={frame.pc}")
        except VMError as e:
            # Stamp it once, at the innermost frame (a CALL nests frames).
            if not e.src_ref:
                self._locate(e, frame, ins)
            raise

        # Fell off the end without an explicit RET.
        self._complete(frame)

    def _locate(self, e: VMError, frame: Frame, ins) -> None:
        """Stamp ``e`` with the source of the instruction that raised it."""
        sr = (ins.src_ref if ins is not None else 0) or frame.coro.src_ref
        if not sr:
            return
        e.src_ref = sr
        entries = self.model.prov.entries
        if sr < len(entries):
            e.prov = entries[sr]

    def _complete(self, frame: Frame) -> None:
        """Mark ``frame`` done and wake anyone waiting on it."""
        frame.done = True
        if frame.is_node and frame.act is not None:
            frame.act.exit_node(frame.node, frame.comp)
        # Deliver a return value to a blocking INVOKE caller.
        if frame.ret_target is not None:
            waiter, reg = frame.ret_target
            if reg is not None and frame.retval is not None:
                ops_proc._set(waiter, reg, frame.retval)
        # Count this completion toward the parent's active JOIN/INVOKE wait. Only
        # children a JOIN (or blocking INVOKE) marked ``counted`` participate, so a
        # detached / not-yet-joined child never perturbs the count.
        waiter = frame.parent
        if waiter is not None:
            # Only live children are ever consulted (JOIN, cancel), so a
            # done one leaves the list: a long loop does not keep its frames.
            # (By identity: a Frame compares by value. A call's callee
            # is not in the list.)
            kids = waiter.children
            for i, c in enumerate(kids):
                if c is frame:
                    del kids[i]
                    break
        if waiter is not None and frame.counted:
            frame.counted = False
            if waiter.pending > 0:
                waiter.pending -= 1
                if waiter.pending == 0:
                    if frame.called:
                        self.sched.ready_first(waiter)    # a call returns at once
                        return
                    # A FIRST(n) join is satisfied: cancel siblings still counted
                    # (not yet complete), matching the legacy runtime which cancels
                    # the branches it did not join.
                    for c in waiter.children:
                        if c.counted:
                            c.counted = False
                            self._cancel(c)
                    self.sched.ready(waiter)

    def _cancel(self, frame: Frame) -> None:
        """Cancel a branch and its live descendants; they never run again."""
        if frame.cancelled or frame.done:
            return
        frame.cancelled = True
        if frame.is_node and frame.act is not None:
            frame.act.exit_node(frame.node, frame.comp, completed=False)
        for c in frame.children:
            self._cancel(c)

    # -- the drain -------------------------------------------------------- #

    def run(self, root: Frame) -> None:
        """Construct the component tree, then run ``root`` and everything it
        spawns to quiescence."""
        if root.cobj is not None:
            self.construct_components()
        self._drain(root)

    def _drain(self, root: Frame) -> None:
        """Run to quiescence. ``stuck`` counts the runs in a row that ended
        in a SPIN yield with no progress (INSTR_F_SPIN); once it exceeds the
        ready queue, every ready frame is waiting on a condition no ready
        frame will change, so time must advance, or nothing can."""
        self.sched.ready(root)
        stuck = 0
        sched = self.sched
        while sched.has_work():
            host = self.host
            if sched._ready and stuck > len(sched._ready):
                if host is not None:
                    t = sched.peek_time()
                    if t is None:
                        host.wait_event()     # WAIT EVENT: a notify may help
                        stuck = 0
                        continue
                    if t > sched.now:
                        host.wait_time(t - sched.now)
                frame = sched.next_timed()
                if frame is None:
                    raise VMError("deadlock: every running thread waits on a "
                                  "channel that nothing can put to or get from")
                stuck = 0
            else:
                if host is not None and not sched._ready:
                    t = sched.peek_time()
                    if t is not None and t > sched.now:
                        host.wait_time(t - sched.now)
                        if sched._ready:      # a notify made a thread ready
                            continue
                frame = sched.next_frame()
            if frame is None:
                break
            if not frame.done and not frame.cancelled:
                self.stuck = False
                self.run_frame(frame)
                stuck = stuck + 1 if self.stuck else 0
