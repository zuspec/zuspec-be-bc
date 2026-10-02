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
from ..trace.sink import NullSink, TraceSink
from . import ops_proc, ops_orch
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
                 memory=None, solve_cache=None, on_solve=None):
        self.model = model
        #: called with the frame after each solve has written its values
        #: back (telemetry; None in a normal run)
        self.on_solve = on_solve
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
            frame.act = Activation(table, obj, self.solves, frame.cobj)
            self.set_comp(frame, table.nodes[0].comp_rel)
            frame.act.enter_node(0, None)
            frame.is_node = True
        return frame

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

    # -- the dispatch loop ------------------------------------------------ #

    def run_frame(self, frame: Frame) -> None:
        """Run ``frame`` from its pc until it suspends or completes."""
        code = frame.coro.code
        n = len(code)
        frame.resume_pc = frame.pc
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

        # Fell off the end without an explicit RET.
        self._complete(frame)

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
        while self.sched.has_work():
            if self.sched._ready and stuck > len(self.sched._ready):
                frame = self.sched.next_timed()
                if frame is None:
                    raise VMError("deadlock: every running thread waits on a "
                                  "channel that nothing can put to or get from")
                stuck = 0
            else:
                frame = self.sched.next_frame()
            if frame is None:
                break
            if not frame.done and not frame.cancelled:
                self.stuck = False
                self.run_frame(frame)
                stuck = stuck + 1 if self.stuck else 0
