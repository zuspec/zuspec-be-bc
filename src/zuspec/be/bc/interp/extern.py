"""
extern.py -- the SOLVE / IMPORT boundary of the oracle (P1-11).

The interpreter reaches the outside world through exactly two seams, both narrow
Python protocols so the differential adapter (T1-A) can bridge them to the legacy
runtime and the unit tests can stub them:

* :class:`SolveBackend` -- ``SOLVE`` calls ``randomize(obj, problem, seed)`` and
  the interpreter writes the results back into ``obj`` per the value ABI's
  sorted-by-name / ``var_id`` mapping (the ``problem.writeback`` map).
* :class:`ImportProvider` -- ``IMPORT`` dispatches ``call(fn_id, args)`` through a
  Python provider table and the result lands in the instruction's result register.

M1 keeps both synchronous and pure-Python (roadmap Q1): no SEC_SOLVE bytes, no DPI
marshaling. Seeds derive from the determinism spec's fork rule (see
:mod:`..determinism`), applied by the scheduler before it hands a seed here.

:class:`Obj` is a minimal field-addressable action object: fields are reachable
both by integer slot (``LD_FIELD``/``ST_FIELD`` use ``expr.index``) and by name
(SOLVE write-back is keyed by field name). The differential adapter substitutes a
wrapper over the legacy dataclass instance; the unit tests use ``Obj`` directly.
"""

from typing import Any, Dict, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Action object
# --------------------------------------------------------------------------- #

class Obj:
    """A field-addressable value object (M1 test/adapter default).

    Fields are 64-bit integer slots addressable both by index (procedural
    ``LD_FIELD``/``ST_FIELD``) and by name (SOLVE write-back). ``field_names`` is
    the *slot order*; the name<->slot map is derived from it.
    """

    def __init__(self, field_names: Sequence[str] = (), values: Sequence[int] = ()):
        self.field_names: List[str] = list(field_names)
        self._by_name: Dict[str, int] = {n: i for i, n in enumerate(self.field_names)}
        self.values: List[int] = list(values) if values else [0] * len(self.field_names)
        if len(self.values) != len(self.field_names):
            raise ValueError("values length must match field_names length")

    # -- procedural (slot) access ----------------------------------------- #

    def get_field(self, index: int) -> int:
        return self.values[index]

    def set_field(self, index: int, value: int) -> None:
        self.values[index] = int(value)

    # -- named access (SOLVE write-back) ---------------------------------- #

    def get_field_name(self, name: str) -> int:
        return self.values[self._by_name[name]]

    def set_field_name(self, name: str, value: int) -> None:
        self.values[self._by_name[name]] = int(value)

    def has_field(self, name: str) -> bool:
        return name in self._by_name

    def as_dict(self) -> Dict[str, int]:
        return {n: self.values[i] for n, i in self._by_name.items()}

    def __repr__(self) -> str:
        return f"Obj({self.as_dict()!r})"


# --------------------------------------------------------------------------- #
# SOLVE backend
# --------------------------------------------------------------------------- #

class SolveBackend:
    """SOLVE seam. Return ``{var_name: value}`` for the problem's variables."""

    def randomize(self, obj: Any, problem, seed: int) -> Dict[str, int]:  # pragma: no cover - abstract
        raise NotImplementedError


class CallbackSolveBackend(SolveBackend):
    """Adapt a plain ``callable(obj, problem, seed) -> {name: value}``.

    The differential adapter passes a callback that drives the real
    ``NativeSolverBackend.randomize`` and reads the solved fields back off the
    live object; unit tests pass a deterministic stub.
    """

    def __init__(self, fn):
        self._fn = fn

    def randomize(self, obj, problem, seed) -> Dict[str, int]:
        return dict(self._fn(obj, problem, seed))


class FixedSolveBackend(SolveBackend):
    """Deterministic stub: each variable gets ``base + seed`` mixed by its index.

    Not a real solver -- just a reproducible mapping so oracle/scheduler tests can
    assert exact written-back values without pulling in ``zuspec-solver``.
    """

    def __init__(self, base: int = 0):
        self._base = base

    def randomize(self, obj, problem, seed) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i, name in enumerate(problem.var_names):
            out[name] = (self._base + seed + i) & ((1 << 64) - 1)
        return out


# --------------------------------------------------------------------------- #
# IMPORT provider
# --------------------------------------------------------------------------- #

class ImportProvider:
    """IMPORT seam. ``call(fn_id, args)`` returns the (optional) result value."""

    def call(self, fn_id: int, args: Sequence[int]) -> Optional[int]:  # pragma: no cover - abstract
        raise NotImplementedError


class RecordingImportProvider(ImportProvider):
    """Records every call and returns a configured (or zero) result.

    ``returns`` maps ``fn_id -> value`` (or ``fn_id -> callable(args) -> value``).
    The recorded ``calls`` list is the import-sequence the differential suite
    compares against the legacy runtime.
    """

    def __init__(self, returns: Optional[Dict[int, Any]] = None):
        self.returns: Dict[int, Any] = dict(returns or {})
        self.calls: List[Dict[str, Any]] = []

    def call(self, fn_id: int, args: Sequence[int]) -> Optional[int]:
        args = list(args)
        self.calls.append({"fn_id": fn_id, "args": args})
        r = self.returns.get(fn_id, 0)
        if callable(r):
            return r(args)
        return r
