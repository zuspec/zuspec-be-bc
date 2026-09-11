"""
interp -- the Python ZBC interpreter (the M1 oracle).

The differential reference implementation of ZBC execution (roadmap P1 / W7). It
executes the same lowered ZBC the native engine will run and reproduces the legacy
runtime's results, so it can validate simulation semantics in pure Python before
any native build exists `[D§3a #1, §5]`.

Layers:

* :mod:`.scheduler`  -- frames + ready queue / timed-event heap (P1-8).
* :mod:`.vm`         -- the dispatch loop (P1-9).
* :mod:`.ops_proc` / :mod:`.ops_orch` -- opcode handlers (P1-10) + trace (P1-12).
* :mod:`.extern`     -- the SOLVE / IMPORT seam (P1-11).
* :mod:`.oracle`     -- the canonical round-trip run configuration (P1-13).
"""

from .scheduler import Frame, Scheduler
from .vm import VM
from .ops_proc import VMError
from .extern import (
    Obj,
    SolveBackend, CallbackSolveBackend, FixedSolveBackend, NativeBlobBackend,
    ImportProvider, RecordingImportProvider,
)
from .oracle import (
    RunResult, RoundTripError, run_model, run_scenario, run_module, roundtrip,
)

__all__ = [
    "Frame", "Scheduler", "VM", "VMError",
    "Obj",
    "SolveBackend", "CallbackSolveBackend", "FixedSolveBackend", "NativeBlobBackend",
    "ImportProvider", "RecordingImportProvider",
    "RunResult", "RoundTripError", "run_model", "run_scenario", "run_module",
    "roundtrip",
]
