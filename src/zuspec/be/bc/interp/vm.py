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
from .extern import Obj, SolveBackend, ImportProvider, FixedSolveBackend, \
    RecordingImportProvider


class VM:
    """Executes a :class:`..model.ZbcModel` under a :class:`Scheduler`."""

    def __init__(self, model, solve_backend: Optional[SolveBackend] = None,
                 import_provider: Optional[ImportProvider] = None,
                 sink: Optional[TraceSink] = None,
                 out: Optional[Callable[[str], None]] = None,
                 verbosity: int = VERBOSITY_MEDIUM):
        self.model = model
        self.sched = Scheduler()
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

    def spawn_child(self, parent: Frame, coro_index: int, obj: Optional[Obj] = None) -> Frame:
        """Create a child frame with a seed forked from the parent (D§15.1)."""
        coro = self.model.coros[coro_index]
        child_seed = parent.seed.fork(parent.next_child_index())
        if obj is None:
            # An action traversal gets its own object; a synthesized PAR/SELECT
            # branch (no layout) runs on its parent's.
            obj = self._new_obj(coro_index)
        child = self.sched.new_frame(
            coro, seed=child_seed,
            obj=obj if obj is not None else parent.obj,
            parent=parent,
        )
        parent.children.append(child)
        return child

    def root_frame(self, coro_index: int, seed: int, obj: Optional[Obj] = None) -> Frame:
        if obj is None:
            obj = self._new_obj(coro_index)
        return self.sched.new_frame(self.model.coros[coro_index], seed=seed, obj=obj)

    # -- the dispatch loop ------------------------------------------------ #

    def run_frame(self, frame: Frame) -> None:
        """Run ``frame`` from its pc until it suspends or completes."""
        code = frame.coro.code
        n = len(code)
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
        # Deliver a return value to a blocking INVOKE caller.
        if frame.ret_target is not None:
            waiter, reg = frame.ret_target
            if reg is not None and frame.retval is not None:
                ops_proc._set(waiter, reg, frame.retval)
        # Count this completion toward the parent's active JOIN/INVOKE wait. Only
        # children a JOIN (or blocking INVOKE) marked ``counted`` participate, so a
        # detached / not-yet-joined child never perturbs the count.
        waiter = frame.parent
        if waiter is not None and frame.counted:
            frame.counted = False
            if waiter.pending > 0:
                waiter.pending -= 1
                if waiter.pending == 0:
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
        for c in frame.children:
            self._cancel(c)

    # -- the drain -------------------------------------------------------- #

    def run(self, root: Frame) -> None:
        """Run ``root`` and everything it spawns to quiescence."""
        self.sched.ready(root)
        while self.sched.has_work():
            frame = self.sched.next_frame()
            if frame is None:
                break
            if not frame.done and not frame.cancelled:
                self.run_frame(frame)
