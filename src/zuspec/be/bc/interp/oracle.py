"""
oracle.py -- the canonical oracle run configuration (P1-13).

The oracle is the differential reference: lower Scenario IR to ZBC, then execute
it. To keep the *serialize* path honest, the default run configuration is
IR → in-memory model → **serialize → deserialize** → execute, so a writer/reader
bug surfaces as an execution divergence instead of hiding off the tested path
`[D§12.3]`. A cheap ``model == deserialize(serialize(model))`` self-check runs
first.

The in-memory SOLVE problem table is an M1 side channel (no SEC_SOLVE bytes yet),
so it is *reattached* to the deserialized model rather than round-tripped through
bytes -- exactly how the native path will treat it until P4.

Public surface:

* :class:`RunResult`     -- final object state, return value, trace events, clock.
* :func:`run_model`      -- execute an already-built :class:`ZbcModel`.
* :func:`run_scenario`   -- lower a set of ``ScCoroutine`` and execute (round-trip
  by default -- the P1-13 discipline).
"""

import dataclasses as dc
from typing import Any, Callable, Dict, List, Optional

from ..model import ZbcModel
from ..trace.sink import MemorySink, TraceSink
from ..trace.schema import TraceEvent
from ..lower import lower_scenario, lower_module
from .vm import VM, VERBOSITY_MEDIUM
from .extern import Obj, SolveBackend, ImportProvider


@dc.dataclass
class RunResult:
    """The observable output of an oracle run (the differential comparison unit)."""

    obj: Optional[Obj]
    retval: Optional[int]
    events: List[TraceEvent]
    now: int
    frames: int

    @property
    def fields(self) -> Dict[str, int]:
        return self.obj.as_dict() if self.obj is not None else {}

    def trace_json(self) -> str:
        from ..trace.schema import serialize
        return serialize(self.events)


class RoundTripError(RuntimeError):
    """Raised when serialize→deserialize does not reproduce the model (P1-13)."""


def _reattach_side_channel(dst: ZbcModel, src: ZbcModel) -> ZbcModel:
    """Copy the in-memory side channels (not serialized in M1) onto ``dst``."""
    dst.problems = src.problems
    dst.selects = src.selects
    dst.messages = src.messages
    dst.strings = src.strings
    dst.obj_layouts = src.obj_layouts
    dst.activations = src.activations
    dst.components = src.components
    return dst


def roundtrip(model: ZbcModel) -> ZbcModel:
    """serialize → deserialize ``model`` and reattach the M1 side channel.

    Verifies the (serialized) content is byte-stable and self-consistent, then
    hands back a model equivalent to the input but reconstructed from bytes.
    """
    data = model.to_bytes()
    restored = ZbcModel.from_bytes(data)
    if restored != model:
        raise RoundTripError("deserialized model differs from the in-memory model")
    if restored.to_bytes() != data:
        raise RoundTripError("re-serialization is not byte-stable")
    return _reattach_side_channel(restored, model)


def run_model(model: ZbcModel, obj: Optional[Obj] = None, seed: int = 0,
              solve_backend: Optional[SolveBackend] = None,
              import_provider: Optional[ImportProvider] = None,
              sink: Optional[TraceSink] = None,
              out: Optional[Callable[[str], None]] = None,
              verbosity: int = VERBOSITY_MEDIUM) -> RunResult:
    """Execute an already-built model from its entry coroutine.

    ``out`` receives each ``message()`` line (default: stdout). With no ``obj``,
    the entry action gets a fresh object from its layout, when it has one.
    """
    sink = sink if sink is not None else MemorySink()
    vm = VM(model, solve_backend=solve_backend,
            import_provider=import_provider, sink=sink, out=out, verbosity=verbosity)
    root = vm.root_frame(model.entry_coro, seed=seed, obj=obj)
    vm.run(root)
    events = list(getattr(sink, "events", []))
    return RunResult(
        obj=root.obj,
        retval=root.retval,
        events=events,
        now=vm.sched.now,
        frames=vm.sched._next_id,
    )


def run_scenario(coros, entry: int = 0, obj: Optional[Obj] = None, seed: int = 0,
                 blocking_targets=None,
                 solve_backend: Optional[SolveBackend] = None,
                 import_provider: Optional[ImportProvider] = None,
                 sink: Optional[TraceSink] = None,
                 round_trip: bool = True) -> RunResult:
    """Lower ``coros`` to ZBC and execute (round-trip serialize path by default).

    This is the entry point the differential suite (T1-A) and golden/round-trip
    tests call. ``round_trip=True`` makes serialize→deserialize→execute the
    default CI path (P1-13 / T1-B); pass ``False`` to execute the in-memory model
    directly (useful for isolating a lowering vs. serialization fault).
    """
    model = lower_scenario(coros, entry=entry, blocking_targets=blocking_targets)
    if round_trip:
        model = roundtrip(model)
    return run_model(model, obj=obj, seed=seed,
                     solve_backend=solve_backend,
                     import_provider=import_provider, sink=sink)


def run_module(module, entry_action: Optional[str] = None,
               obj: Optional[Obj] = None, seed: int = 0,
               solve_backend: Optional[SolveBackend] = None,
               import_provider: Optional[ImportProvider] = None,
               sink: Optional[TraceSink] = None,
               round_trip: bool = True) -> RunResult:
    """Lower a ``ScenarioModule`` and execute it -- the successor **engine** entry.

    This is the public "run a lowered PSS scenario through the ZBC oracle" surface
    that replaces the legacy ``ScenarioRunner`` as the execution engine (W8 / P1-14).
    A frontend (or ``PSSToScenarioPass``) produces the ``ScenarioModule``; this runs
    its entry action (or ``entry_action``) with the value ABI's solve/import seams,
    through the round-trip serialize path by default.
    """
    model = lower_module(module, entry_action=entry_action)
    if round_trip:
        model = roundtrip(model)
    return run_model(model, obj=obj, seed=seed,
                     solve_backend=solve_backend,
                     import_provider=import_provider, sink=sink)
