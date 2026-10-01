"""
scheduler.py -- frames + the run loop (P1-8).

A Python analogue of the native ``zsp_timebase``: a FIFO **ready queue** plus a
**timed-event min-heap**, driving pool-free Python :class:`Frame` objects. This
models *ordering*, not speed -- the contract is that the sequence of resumptions
(and therefore the emitted trace) matches what the eventual C scheduler will
produce, per the determinism spec `[D§5, §15.1]`.

Concurrency in M1 is cooperative and coarse-grained:

* **WAIT** re-queues a frame on the timed heap at ``now + delay``.
* **SPAWN** creates a child frame (ready immediately); the parent keeps running.
* **JOIN** suspends the parent until every live child it spawned has completed.
* **blocking INVOKE** suspends the parent until the single callee returns, then
  delivers the callee's return value into the caller's result register.
* **YIELD** re-queues the frame at the back of the ready queue (same ``now``).

Determinism anchors: ready order is FIFO (spawn/enqueue order); ties on the timed
heap break by an insertion sequence number; child seeds fork from the parent via
:func:`..determinism.fork_seed` using a per-parent child index. Nothing here draws
on wall-clock time.
"""

import dataclasses as dc
import heapq
from typing import Any, List, Optional, Tuple

from ..determinism import SeedStream
from .extern import Obj


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #

@dc.dataclass
class Frame:
    """One coroutine activation: its code cursor, register file, and frame state."""

    id: int
    coro: Any                       # CoroDescriptor
    seed: SeedStream
    obj: Optional[Obj] = None
    pc: int = 0
    regs: dict = dc.field(default_factory=dict)
    locals: List[int] = dc.field(default_factory=list)
    retval: Optional[int] = None
    done: bool = False

    # -- child / join / call bookkeeping ---------------------------------- #
    parent: Optional["Frame"] = None
    children: List["Frame"] = dc.field(default_factory=list)
    pending: int = 0                # live children this frame is JOINing on
    #: True while this child counts toward its parent's active JOIN. A JOIN marks
    #: its members; a blocking INVOKE marks its callee. Cleared on completion, or
    #: when a FIRST(n)/SELECT join cancels the branches it did not wait for.
    counted: bool = False
    #: Cancelled branches (FIRST(n)/SELECT surplus, and their descendants) never run
    #: again and are excluded from any JOIN's live set. Matches the legacy runtime,
    #: which cancels the tasks it does not join.
    cancelled: bool = False
    #: (waiter_frame, result_reg) to satisfy when *this* frame completes; set by a
    #: blocking INVOKE so the callee delivers its return value to the caller.
    ret_target: Optional[Tuple["Frame", int]] = None
    _child_index: int = 0           # next fork index for SPAWN/INVOKE
    #: P1-D1: the slot of ``obj`` this frame's field 0 is (its node's base)
    base: int = 0
    #: the activation this frame runs in (``activation.Activation``), its node
    #: and the traversal site that reached it; None outside an activation
    act: Any = None
    node: int = 0
    site: Optional[int] = None
    #: P1.5: the component object, the frame's instance in it (None: not
    #: chosen yet -- the node's solve chooses it), and that instance's base
    cobj: Optional[Obj] = None
    comp: Optional[int] = 0
    cbase: int = 0
    #: the pc this frame's current run started at (a SPIN yield's progress test)
    resume_pc: int = 0
    #: CALL: the arguments staged for the next call, this frame's own
    #: arguments, its call depth, and whether it is a called function
    staged: List[int] = dc.field(default_factory=list)
    args: List[int] = dc.field(default_factory=list)
    depth: int = 0
    called: bool = False

    def next_child_index(self) -> int:
        i = self._child_index
        self._child_index += 1
        return i


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #

class Scheduler:
    """Ready queue + timed-event heap. The VM asks it to run frames to quiescence."""

    def __init__(self):
        self.now: int = 0
        self._ready: List[Frame] = []
        self._timed: List[Tuple[int, int, Frame]] = []  # (time, seq, frame) min-heap
        self._seq: int = 0
        self._next_id: int = 0

    # -- frame construction ----------------------------------------------- #

    def new_frame(self, coro, seed: int, obj: Optional[Obj] = None,
                  parent: Optional[Frame] = None) -> Frame:
        f = Frame(
            id=self._next_id,
            coro=coro,
            seed=SeedStream(seed) if not isinstance(seed, SeedStream) else seed,
            obj=obj,
            locals=[0] * len(coro.frame_locals),
            parent=parent,
        )
        self._next_id += 1
        return f

    # -- queueing --------------------------------------------------------- #

    def ready(self, frame: Frame) -> None:
        """Enqueue a frame to run at the current time (FIFO)."""
        self._ready.append(frame)

    def ready_first(self, frame: Frame) -> None:
        """Run *frame* next: a CALL's callee, and its caller when it returns,
        so a call is no scheduling point (an inlined body is none)."""
        self._ready.insert(0, frame)

    def schedule_at(self, frame: Frame, time: int) -> None:
        """Enqueue a frame to resume at absolute ``time`` (timed heap)."""
        heapq.heappush(self._timed, (time, self._seq, frame))
        self._seq += 1

    def wait(self, frame: Frame, delay: int) -> None:
        """WAIT: resume ``frame`` at ``now + delay`` (delay 0 ⇒ same instant)."""
        self.schedule_at(frame, self.now + max(0, int(delay)))

    # -- the drain -------------------------------------------------------- #

    def has_work(self) -> bool:
        return bool(self._ready or self._timed)

    def next_timed(self) -> Optional[Frame]:
        """Pop the next timed frame, advancing ``now`` to it: what runs when
        every ready frame is waiting on a condition only time can change."""
        while self._timed:
            time, _, frame = heapq.heappop(self._timed)
            if frame.done or frame.cancelled:
                continue
            self.now = max(self.now, time)
            return frame
        return None

    def next_frame(self) -> Optional[Frame]:
        """Pop the next frame to run, advancing ``now`` across idle gaps.

        Cancelled/done frames still parked on the timed heap are dropped without
        advancing the clock, so a cancelled WAIT never influences ``now``.
        """
        if self._ready:
            return self._ready.pop(0)
        while self._timed:
            time, _, frame = heapq.heappop(self._timed)
            if frame.done or frame.cancelled:
                continue
            self.now = max(self.now, time)
            return frame
        return None
