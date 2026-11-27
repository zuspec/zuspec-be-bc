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

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..model import Op
from ..trace import events as ev
from ..trace.sink import NullSink, TraceSink
from .heap import Heap
from . import ops_proc, ops_orch
from .defuse import abs_comp as _abs_comp, defuse
from .ops_proc import VMError
from .scheduler import Frame, Scheduler

#: message() verbosity levels (std_pkg::message_verbosity_e, 21.1.3).
VERBOSITY_NONE, VERBOSITY_LOW, VERBOSITY_MEDIUM, VERBOSITY_HIGH, VERBOSITY_FULL = range(5)
from .extern import Obj, SolveBackend, ImportProvider, FixedSolveBackend, Memory, \
    NativeBlobBackend, RecordingImportProvider
from .solve_cache import SolveCache
from ..determinism import SeedStream, coro_seed_keys, fork_seed, keyed_index

#: A stimulus id a bench put in a report message: ``pss#top_a#0.l2#0``.
_STIM_TOKEN = re.compile(r"\bpss#([^\s,;:()\[\]'\"<>]+)")


def _key_int(v) -> Optional[int]:
    """A bench report key as an integer: an int, or a string in decimal,
    ``0x..`` or SV ``'h..`` form; None when absent or not a number."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    t = str(v).strip().lower().replace("_", "")
    try:
        if t.startswith("'h"):
            return int(t[2:], 16)
        return int(t, 0)
    except ValueError:
        return None


def stim_tag(msg: str) -> Optional[str]:
    """The stimulus id in the first ``pss#<id>`` token of *msg*, or None."""
    m = _STIM_TOKEN.search(msg or "")
    return m.group(1).rstrip(".") if m else None


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
        #: the instance's heap (F6.9c; ``RunOpts.heap_cap`` sets its cap)
        self.heap = Heap(getattr(model, "strings", ()) or ())
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
        #: how a child's seed forks (F3.9): "stream" by its child index, "keyed"
        #: (the default) by what it traverses and how many such siblings
        #: came before (``determinism.keyed_index``): adding an unrelated
        #: traversal leaves the others' seeds unchanged
        self.seed_mode = "keyed"
        #: M8b6: the :class:`~.activation.PlanLog` activations record the
        #: run's plan in (``RunOpts.plan``); None: nothing is recorded
        self.plan = None
        #: M8d5: stim ids whose action runs have no effects (``RunOpts.drop``)
        self.drop = frozenset()
        #: M8d5: coroutine indices a dropped frame does not call (a host's
        #: check ledger: ``skip_when_dropped``)
        self.drop_calls = set()
        self._seed_keys = None
        #: an exported action's arguments (``root_frame``'s ``args``), as
        #: bounds that pin each parameter's field: {object slot: (v, v)}
        #: (by slot: a loaded image's solve problems do not name variables)
        self.arg_bounds = {}
        self.arg_types = {}
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
        d = getattr(model, "default_executor", None)
        if d is not None:
            self.sched.default_executor = d
        #: set by a SPIN yield that made no progress (see _drain)
        self.stuck = False
        #: frame runs that made progress (not stuck; a run that parked on
        #: a request counts too: what it did first, a channel put, may wake
        #: another call), and host channel puts: a call idle on EVENT or
        #: TIME since a smaller count may run on (host.HostCall.runnable;
        #: calls share the tree)
        self.progress = 0
        self.solve_backend = solve_backend or FixedSolveBackend()
        self.import_provider = import_provider or RecordingImportProvider()
        self.sink = sink or NullSink()
        #: (covg id, register values) of each COV_SAMPLE, in order (F4.4)
        self.cov_samples = []
        #: where message() lines go: one call per line, without the newline
        self.out = out or (lambda line: print(line, flush=True))
        #: the run's message verbosity (21.1.3): NONE=0 LOW=1 MEDIUM=2 HIGH=3 FULL=4
        self.verbosity = verbosity
        #: the host a call yields to (``host.HostCall``; docs/spec/host-abi.md),
        #: or None: accesses, imports and waits are answered in place
        self.host = None
        #: the thread whose import ``run_sync`` is serving (None: none)
        self.serving_thread = None
        #: F2.6: (prefix, name) -> stimulus ids handed out (``_stim_id``)
        self._stim_counts: Dict[str, int] = {}
        #: F2.6b: live action frames, in start order (``in_flight``)
        self._live: Dict[int, Frame] = {}
        #: F7.7a: the latest activations run, newest last, whose address
        #: claims name a region's owner (``owners``)
        self._acts: List[Any] = []
        #: F2.7: the bench reports (``bench_report``) of this run, in order
        self.bench_reports: List[dict] = []

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
        if parent.key_counts is None:
            parent.key_counts = {}
        child.origin, child.key_counts = parent.origin, parent.key_counts
        child.stim = parent.stim
        child.dropped = parent.dropped
        return child

    def _keyed_index(self, parent: Frame, coro_index: int) -> int:
        if self._seed_keys is None:
            keys = getattr(self.model, "coro_seed_keys", None)
            self._seed_keys = (keys() if keys else coro_seed_keys(
                self.model.coros, getattr(self.model, "obj_layouts", {}) or {}))
        key = self._seed_keys[coro_index]
        if parent.key_counts is None:
            parent.key_counts = {}
        n = parent.key_counts.get(key, 0)
        parent.key_counts[key] = n + 1
        return keyed_index(key, n)

    def spawn_child(self, parent: Frame, coro_index: int, obj: Optional[Obj] = None,
                    node: Optional[tuple] = None, start_pc: int = 0,
                    abs_site: Optional[int] = None) -> Frame:
        """Create a child frame with a seed forked from the parent (D§15.1).

        *node* = ``(child_base, site)`` for a traversal of a node of the
        parent's activation: the child runs on the parent's object at its
        node's base (P1-D1). *abs_site*: the traversal's site in the whole
        tree, not the parent's (an inferred node's, M9a3).
        """
        coro = self.model.coros[coro_index]
        if self.seed_mode == "keyed":
            child_seed = SeedStream(fork_seed(parent.origin,
                                              self._keyed_index(parent, coro_index)))
        else:
            child_seed = parent.seed.fork(parent.next_child_index())
        if node is not None:
            child = self.sched.new_frame(coro, seed=child_seed, obj=parent.obj,
                                         parent=parent)
            child.pc = start_pc
            child.cobj = parent.cobj
            child_base, local_site = node
            act = parent.act
            if act is not None:
                site = (abs_site if abs_site is not None
                        else act.t.site(parent.node, local_site))
                child.act, child.site = act, site
                child.node = act.t.sites[site][1]
                child.base = act.t.nodes[child.node].base
                rel = act.t.nodes[child.node].comp_rel
                self.set_comp(child, None if rel is None or parent.comp is None
                              else parent.comp + rel)
                act.enter_node(child.node, site)
                child.is_node = True
                k = act.visits[child.node]
                act.visits[child.node] = k + 1
                if k and child.node in getattr(act.t, "node_uid", ()):
                    # traversed again (a loop): a new action, a new uid
                    self.uid_next = getattr(self, "uid_next", 0) + 1
                    act.obj.set_field(act.t.node_uid[child.node], self.uid_next)
                child.stim = f"{act.stim}.{act.t.nodes[child.node].path}#{k}"
                child.dropped = parent.dropped or child.stim in self.drop
                self._action(child)
            else:
                child.base = parent.base + child_base
                self.set_comp(child, parent.comp)
                child.dropped = parent.dropped
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
            child.stim = parent.stim
            child.dropped = parent.dropped
        else:
            child.stim = self._stim_id(coro.name, parent.stim)
            child.dropped = parent.dropped or child.stim in self.drop
            self._action(child)
        child.cobj = parent.cobj
        child.comp, child.cbase = parent.comp, parent.cbase
        parent.children.append(child)
        return child

    def _stim_id(self, name: str, prefix: str = "") -> str:
        """F2.6: the next stimulus id for a run of *name* (its last
        ``::``/``.`` component) under *prefix*: ``[prefix.]name#k``, k
        counting from 0 per (prefix, name) in this VM. A root function entry
        is ``fn#k``; a traversal of an activation node is named by its node
        (``spawn_child``)."""
        short = name.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
        key = f"{prefix}.{short}" if prefix else short
        k = self._stim_counts.get(key, 0)
        self._stim_counts[key] = k + 1
        return f"{key}#{k}"

    def _action(self, frame: Frame) -> None:
        """F2.6b: *frame* is an action run: it is in flight until it ends."""
        frame.action = True
        self._live[id(frame)] = frame

    def in_flight(self) -> List[str]:
        """F2.6b: the stimulus ids of the action runs spawned and not yet
        ended (done or cancelled), oldest first."""
        return [f.stim for f in self._live.values()]

    def owners(self, addr: int, size: int = 1) -> List[dict]:
        """F7.7a: the address claims of the latest activations whose region
        overlaps ``[addr, addr+size)``, live ones first (``Activation.owners``)."""
        live, gone = [], []
        for act in reversed(self._acts):
            for o in act.owners(addr, size):
                (live if o["live"] else gone).append(o)
        return live + gone

    def bench_report(self, kind: str = "uvm", severity: str = "error",
                     msg: str = "", id: str = "", time=None,
                     keys: Optional[Dict[str, Any]] = None) -> dict:
        """F2.7: the bench reported a failure (a ``uvm_error`` through the
        report catcher, an assertion, ``pssvm_bench_fail``). Snapshots the
        action runs in flight, ranks suspects and attributes the report to the
        first (F2.8): an action the bench names (``keys["stim"]``, else a
        ``pss#<id>`` token in *msg*: evidence ``tag``) before those in flight,
        newest first (``temporal``). Applies the verdict rule (an error or
        fatal fails the test; anything else is recorded) and raises a
        ``bench`` event. Returns the record, which is also kept in
        ``bench_reports``."""
        keys = dict(keys or {})
        flight = [{"stim": f.stim, "type": f.coro.name if f.coro else "",
                   **({"inferred_for": f.inferred_for}
                      if f.inferred_for is not None else {})}
                  for f in self._live.values()]
        tag = keys.get("stim") or stim_tag(msg)
        # F7.7a: an address the bench names is owned by the action whose
        # claim holds it (evidence ``ownership``), ranked after a tag
        addr = _key_int(keys.get("addr"))
        owners = (self.owners(addr, _key_int(keys.get("size")) or 1)
                  if addr is not None else [])
        ranked: Dict[str, List[str]] = {}
        if tag:
            ranked[tag] = ["tag"]
        for o in owners:
            e = ranked.setdefault(o["stim"], [])
            if "ownership" not in e:
                e.append("ownership")
        for f in reversed(flight):
            e = ranked.setdefault(f["stim"], [])
            if "temporal" not in e:
                e.append("temporal")
        suspects = [{"stim": k, "evidence": v} for k, v in ranked.items()]
        stim = suspects[0]["stim"] if suspects else None
        d = {"kind": kind, "severity": severity, "id": id, "msg": msg,
             "time": self.sched.now if time is None else time, "keys": keys,
             "in_flight": flight, "stim": stim, "suspects": suspects,
             **({"owners": owners} if owners else {}),
             "verdict": "fail" if severity in ("error", "fatal") else "record"}
        self.bench_reports.append(d)
        self.host_event(ev.BENCH, d)
        return d

    def _action_end(self, frame: Frame, cancelled: bool = False) -> None:
        """*frame*, an action run, ended: out of flight, and its ``action``
        end event when its start went out."""
        self._live.pop(id(frame), None)
        if not frame.started or self.events is None:
            return
        if not self._ev.get(id(frame.coro), 0) & ev.EV_ACTION:
            return
        d = {"stim": frame.stim, "type": frame.coro.name, "phase": "end"}
        if cancelled:
            d["cancelled"] = True
        self.events.event(ev.ACTION, coro=frame.id, src_ref=frame.coro.src_ref,
                          detail=d)

    def set_comp(self, frame: Frame, comp: Optional[int]) -> None:
        """Run *frame* in component instance *comp* (None: not chosen yet)."""
        frame.comp = comp
        comps = getattr(self.model, "components", None)
        frame.cbase = comps.base(comp) if comps is not None else 0

    def root_frame(self, coro_index: int, seed: int, obj: Optional[Obj] = None,
                   args: Sequence[int] = (), cobj: Optional[Obj] = None,
                   hooks=True, pins: Optional[dict] = None,
                   comp: int = 0) -> Frame:
        """A frame that runs action (or coroutine) *coro_index* as the root.
        *args*: an exported action's arguments (LRM 20.10, ``export T(bit m)``),
        one per entry parameter (or none: they are solved as usual): each pins the field it names in every solve
        (``arg_bounds``), and the frame becomes the entry (``set_entry``).
        *cobj*: the component tree to run in (None: a new one, constructed
        when the frame runs). *hooks*: which of the tree's ``exec run_start``
        / ``run_end`` blocks run around a root action: True both, "start"
        or "end" that one, False none (a simulation runs them once per
        run, not once per action, F2.5). *pins*: a host's field overrides,
        ``{slot: (value, (width, signed))}``, pinned like *args* (a slot
        both name is refused). *comp*: the instance the root action runs in
        (its one candidate's; with several, *pins* holds its ``comp``
        slot)."""
        table = getattr(self.model, "activations", {}).get(coro_index)
        self.arg_types = {}     #: pinned slot -> (width, signed)
        self.arg_bounds = self._arg_bounds(coro_index, args)
        for slot, (v, wt) in (pins or {}).items():
            if slot in self.arg_bounds:
                name = table.names[slot] if table is not None else slot
                raise TypeError(f"field {name} is both an argument and an "
                                f"override")
            self.arg_bounds[slot] = (v, v)
            self.arg_types[slot] = wt
        if obj is None:
            obj = (Obj(field_names=table.names) if table is not None
                   else self._new_obj(coro_index))
        elif table is not None and isinstance(obj, Obj) and \
                len(obj.values) < len(table.names):
            # An object laid out from the entry's fields lacks the slots the
            # table adds past them (a loop index a ``with`` reads, G6).
            for name in table.names[len(obj.values):]:
                if name:
                    obj._by_name.setdefault(name, len(obj.values))
                obj.field_names.append(name)
                obj.values.append(0)
        frame = self.sched.new_frame(self.model.coros[coro_index], seed=seed, obj=obj)
        frame.stim = self._stim_id(frame.coro.name)
        frame.dropped = frame.stim in self.drop
        comps = getattr(self.model, "components", None)
        if comps is not None:
            frame.cobj = self.cobj = cobj if cobj is not None else comps.new_obj()
            self._seed = seed
        if table is not None:
            from .activation import Activation
            # Each node's uid: unique in the run (component uids start at
            # 0x40000000, ir-core comp_tree COMP_UID_BASE).
            for slot in getattr(table, "uid_slots", ()):
                self.uid_next = getattr(self, "uid_next", 0) + 1
                obj.set_field(slot, self.uid_next)
            frame.act = Activation(table, obj, self.solves, frame.cobj,
                                   self.entry_bounds(), heap=self.heap,
                                   plan=self.plan)
            frame.act.stim = frame.stim
            self._acts = self._acts[-7:] + [frame.act]
            self.set_comp(frame, table.nodes[0].comp_rel)
            frame.act.enter_node(0, None)
            frame.is_node = True
        elif comp and comps is not None:
            self.set_comp(frame, comp)
        if table is not None or self._is_action_entry(coro_index):
            self._action(frame)
        if hooks and comps is not None and getattr(comps, "run_hooks", False) \
                and self._is_action_entry(coro_index):
            # drained, a root action runs between $run_start and $run_end
            frame.run_hooks = hooks
        if self.arg_bounds:
            self.set_entry(frame)
        return frame

    def _arg_bounds(self, coro_index: int, args: Sequence[int]) -> dict:
        """*args* of the action entry at *coro_index* as ``{slot: (v, v)}``;
        each a raw (u64) value, cut to its parameter's width (and sign).
        No args: none pinned (a standalone run randomizes them). TypeError
        if their number is not the entry's parameter count."""
        if not args:
            return {}
        entries = getattr(self.model, "entries", None) or {}
        e = next((x for x in entries.values()
                  if x.coro == coro_index and x.kind == "action"), None)
        params = e.params if e is not None else []
        if len(args) != len(params):
            name = e.qname if e is not None else self.model.coros[coro_index].name
            raise TypeError(f"action {name} takes {len(params)} arguments, "
                            f"not {len(args)}")
        table = getattr(self.model, "activations", {}).get(coro_index)
        names = table.names if table is not None else [n for n, _ in e.obj]
        pins = {}
        for (name, d), v in zip(params, args):
            w = d.get("width") or 64
            v = int(v) & ((1 << w) - 1)
            if d.get("signed") and v >> (w - 1):
                v -= 1 << w
            pins[names.index(name)] = (v, v)
            self.arg_types[names.index(name)] = (w, bool(d.get("signed")))
        return pins

    def entry_bounds(self) -> Optional[dict]:
        """``solve_bounds`` with the entry's ``arg_bounds`` (which win), or
        None when there are neither."""
        if not self.arg_bounds:
            return self.solve_bounds
        return {**(self.solve_bounds or {}), **self.arg_bounds}

    def _is_action_entry(self, coro_index: int) -> bool:
        entries = getattr(self.model, "entries", None) or {}
        for e in entries.values():
            if e.coro == coro_index:
                return e.kind == "action"
        return not getattr(self.model.coros[coro_index], "fn_qname", None)

    def _run_hook(self, coro_index: int) -> None:
        """Run component-tree coroutine ``$run_start`` or ``$run_end`` to
        completion in instance 0, as construction does, but with the host:
        its blocks may access memory and call imports."""
        frame = self.sched.new_frame(self.model.coros[coro_index],
                                     seed=getattr(self, "_seed", 0))
        frame.cobj = self.cobj
        self.set_comp(frame, 0)
        self._drain_one(frame)
        if not frame.done:
            raise VMError(f"{frame.coro.name} did not finish: every thread "
                          f"left is blocked")

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

    def skip_when_dropped(self, coro) -> None:
        """M8d5: a dropped frame (``drop``) does not call the function
        ``coro`` (index or name), such as the host's check-recording one,
        so a dropped action's failed checks are not counted; the call
        reads 0."""
        self.drop_calls.add(coro if isinstance(coro, int) else
                            [c.name for c in self.model.coros].index(coro))

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
        if not frame.parked:
            frame.resume_pc = frame.pc
        sink = self.events
        stmts = self._ev_stmts.get(id(frame.coro)) if m & ev.EV_STMT else None
        chans = self._ev_chans.get(id(frame.coro)) if m & ev.EV_CHAN else None
        arms = self._ev_arms.get(id(frame.coro)) if m & ev.EV_BRANCH else None
        paths = self._ev_paths.get(id(frame.coro)) if stmts else None
        calls = bool(m & ev.EV_CALL) and frame.called
        if frame.action and not frame.started:
            frame.started = True
            if m & ev.EV_ACTION:
                d = {"stim": frame.stim, "type": frame.coro.name, "phase": "start"}
                if frame.inferred_for is not None:
                    d["inferred_for"] = frame.inferred_for
                sink.event(ev.ACTION, coro=frame.id, src_ref=frame.coro.src_ref,
                           detail=d)
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
                    if frame.partners and self._await_partners(frame):
                        return
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
        if not frame.parked:
            frame.resume_pc = frame.pc
        ins = None
        try:
            while frame.pc < n:
                ins = code[frame.pc]
                op = ins.op

                if op == Op.RET:
                    if frame.partners and self._await_partners(frame):
                        return
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

    def _await_partners(self, frame: Frame) -> bool:
        """M9a6b: *frame* is at its RET with inferred stream partners still
        running: wait for them (RET runs again then). False: none left."""
        live = [c for c in frame.partners if not c.done and not c.cancelled]
        frame.partners = []
        for c in live:
            c.counted = True
        frame.pending += len(live)
        return bool(live)

    def _complete(self, frame: Frame) -> None:
        """Mark ``frame`` done and wake anyone waiting on it."""
        frame.done = True
        if frame.is_node and frame.act is not None:
            frame.act.exit_node(frame.node, frame.comp)
            for w in frame.act.waking:
                self.sched.ready(w)
            frame.act.waking.clear()
        if frame.action:
            self._action_end(frame)
        # Deliver a return value to a blocking INVOKE caller.
        if frame.ret_target is not None:
            waiter, reg = frame.ret_target
            if reg is not None and frame.retval is not None:
                ops_proc._set(waiter, reg, frame.retval)
            if frame.called:
                # LD_RES: a struct result's leaves and the struct parameters'
                # copy-out, staged by ARG before RET (calls.md)
                waiter.results, frame.staged = frame.staged, []
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
            for w in frame.act.waking:
                self.sched.ready(w)
            frame.act.waking.clear()
        if frame.action:
            self._action_end(frame, cancelled=True)
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
        """Run ``root`` to quiescence; a root action marked by
        :meth:`root_frame` runs after the tree's ``exec run_start`` blocks,
        and once done, before its ``exec run_end`` blocks."""
        if not getattr(root, "run_hooks", False):
            self._drain_one(root)
            return
        hooks, root.run_hooks = root.run_hooks, False
        comps = self.model.components
        if hooks != "end":
            self._run_hook(comps.run_start_coro)
        self._drain_one(root)
        if root.done and hooks != "start":
            self._run_hook(comps.run_end_coro)

    def _drain_one(self, root: Frame) -> None:
        """Run to quiescence. ``stuck`` counts the runs in a row that ended
        in a SPIN yield with no progress (INSTR_F_SPIN); once it exceeds the
        ready queue, every ready frame is waiting on a condition no ready
        frame will change, so time must advance, or nothing can."""
        self.sched.ready(root)
        stuck = 0
        sched = self.sched
        while sched.has_work() or (self.host is not None and self.host.live()):
            host = self.host
            if sched._ready and stuck > len(sched._ready):
                answered = None if host is None else next(
                    (f for f in sched._ready if f.parked and host.answered(f)),
                    None)
                if answered is not None:
                    # its request was completed: the re-run is progress
                    # (it counts once it yields), not a wait
                    sched._ready.remove(answered)
                    frame = answered
                elif host is not None:
                    if host.flush():          # REQS: a completion may help
                        continue
                    t = sched.peek_time()
                    if t is None:
                        if host.live():
                            host.wait_reqs()  # WAIT REQS
                        else:
                            host.wait_event() # WAIT EVENT: a notify may help
                        stuck = 0
                        continue
                    if t > sched.now:
                        if host.wait_time(t - sched.now) and sched.now < t:
                            stuck = 0
                            continue
                if answered is None:
                    frame = sched.next_timed()
                    if frame is None:
                        raise VMError("deadlock: every running thread waits on "
                                      "a channel that nothing can put to or "
                                      "get from")
                    stuck = 0
            else:
                if host is not None and not sched._ready:
                    if host.flush():          # REQS: new requests
                        continue
                    t = sched.peek_time()
                    if t is not None and t > sched.now:
                        # a notify or completion made a thread ready, or the
                        # host's clock is not yet at t
                        if (host.wait_time(t - sched.now) and sched.now < t
                                or sched._ready):
                            continue
                    elif t is None and host.live():
                        host.wait_reqs()      # WAIT REQS
                        continue
                frame = sched.next_frame()
            if frame is None:
                break
            if not frame.done and not frame.cancelled:
                self.stuck = False
                self.run_frame(frame)
                if not frame.parked:    # a parked run goes on at its re-run
                    if self.stuck:
                        stuck += 1
                    else:
                        stuck = 0
                        self.progress += 1
                else:
                    # it ran up to a request: a put before it may wake a
                    # call waiting on EVENT (a function call that puts, then
                    # blocks on an import; pssvm-plan step 7)
                    self.progress += 1
