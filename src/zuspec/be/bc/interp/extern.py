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


class NativeBlobBackend(SolveBackend):
    """Solve a ``SolveProblem.problem_bytes`` blob with the real dv-solve solver.

    Solves the relocatable blob with ``seed`` and returns ``{var_name: value}``
    for every var named in ``problem.writeback``. Drives the *same* solver +
    bytes the native engine does, so oracle and engine agree exactly. Each
    distinct blob is compiled once, in *cache* (a ``solve_cache.SolveCache``;
    a private one when None), and every solve is held to its budget: running
    out raises ``solve_cache.SolveBudgetError``. dv-solve is imported lazily;
    used only when a problem carries a blob.
    """

    _MASK64 = (1 << 64) - 1

    def __init__(self, cache=None):
        if cache is None:
            from .solve_cache import SolveCache
            cache = SolveCache()
        self.cache = cache

    def randomize(self, obj, problem, seed, pins=None, arrays=None) -> Dict[str, int]:
        """*pins*: ``{var_id: 64-bit value}`` fixed before the solve (the
        inputs of a procedural ``randomize``, F3.7b); a value outside its
        variable's domain is unsat. *arrays*: each input array's elements,
        by array id (a problem that reads lists, F6.10d)."""
        from dv_solve.ctx import SOLVE_OK, SOLVE_TIMEOUT, CompileUnsatError
        from .solve_cache import SolveBudgetError

        if problem.input_arrays:
            return self._randomize_arrays(problem, seed, pins, arrays)

        try:
            with self.cache.session(problem.problem_bytes) as ctx:
                src = problem.source
                for vid, val in sorted((pins or {}).items()):
                    # An image-loaded problem has no source: pin the raw value.
                    if not ctx.pin(vid, _domain_value(val, src.vars[vid]) if src is not None
                                   else int(val)):
                        raise _unsat(problem, pins)
                rc = self.cache.solve(ctx, seed & self._MASK64)
                if rc == SOLVE_TIMEOUT:
                    raise SolveBudgetError(self.cache.budget_message())
                if rc != SOLVE_OK:
                    raise _unsat(problem, pins)
                vids = (range(len(problem.var_names))
                        if getattr(problem, "procedural", False)
                        else sorted(set(problem.writeback.values())))
                # Mask to 64-bit so a signed get_value matches the engine's uint64 store.
                return {problem.var_names[v]: ctx.get_value(v) & self._MASK64
                        for v in vids}
        except CompileUnsatError:
            raise _unsat(problem, pins) from None

    def _randomize_arrays(self, problem, seed, pins, arrays) -> Dict[str, int]:
        """``randomize`` of a problem that reads lists: through dv-solve's
        handle (one per problem), the pins as state."""
        from dv_solve.handle import RANDOMIZE_OK
        handles = self.__dict__.setdefault("_handles", {})
        h = handles.get(problem.problem_bytes)
        src = problem.source
        if src is not None:
            specs = [None] * len(problem.var_names)
            for v in src.vars:
                specs[v.var_id] = (v.width, v.signed)
        else:
            # An image-loaded problem: the specs its blob declares.
            specs = problem.var_specs
        if h is None:
            from dv_solve.handle import SolveHandle
            h = handles[problem.problem_bytes] = SolveHandle(
                problem.problem_bytes, specs, problem.array_specs)
        state = {vid: _signed_value(val, *specs[vid]) for vid, val in (pins or {}).items()}
        rc, got, elems = h.randomize(seed & self._MASK64, state, arrays or [])
        if rc != RANDOMIZE_OK:
            raise _unsat(problem, pins)
        # The elements of the rand lists, by array id, for the caller to
        # write back (F6.10e).
        self.solved_arrays = elems
        vids = (range(len(problem.var_names))
                if getattr(problem, "procedural", False)
                else sorted(set(problem.writeback.values())))
        return {problem.var_names[v]: got.get(v, 0) & self._MASK64 for v in vids}


def _unsat(problem, pins):
    """The SolveUnsatError for *problem*: with its source at hand, it names
    the constraints that conflict (``.core``, a list of their ``Loc``)."""
    from .solve_cache import SolveUnsatError
    from .unsat_core import describe, unsat_core

    core = []
    src = problem.source
    if src is not None and not getattr(problem, "input_arrays", None):
        core = unsat_core(src, {vid: _domain_value(val, src.vars[vid])
                                for vid, val in (pins or {}).items()})
    msg = "no values satisfy its constraints"
    if core:
        msg += "; " + describe(core)
    e = SolveUnsatError(msg)
    e.core = [c.loc for c in core]
    return e


def _signed_value(val: int, width: int, signed: bool) -> int:
    """A 64-bit register pattern as a *width*-bit value of that signedness."""
    w = width if width and width > 0 else 32
    x = int(val) & ((1 << w) - 1)
    if signed and x >> (w - 1):
        x -= 1 << w
    return x


def _domain_value(val: int, v) -> int:
    """A 64-bit register pattern as a value of solve var *v*'s domain."""
    return _signed_value(val, v.width, v.signed)


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


class Memory:
    """The platform's memory, for an access no executor overrides (LRM
    21.13.9): sparse, a byte per address, 0 where nothing was written. A
    value of N bytes is little-endian: the byte at the address in bits [7:0]
    (21.13.9.1). A platform with real memory passes its own object with the
    same ``read`` / ``write``. With the ``access`` event on, a memory that
    has ``note_access(reg_id, src_ref)`` is told, just before each access,
    which register it is for (None: no register's)."""

    def __init__(self):
        self.bytes: Dict[int, int] = {}
        #: every access: (kind "r"/"w", address, bytes, value)
        self.log: List[tuple] = []

    def read(self, addr: int, n: int) -> int:
        v = 0
        for i in range(n):
            v |= self.bytes.get((addr + i) & ((1 << 64) - 1), 0) << (8 * i)
        self.log.append(("r", addr, n, v))
        return v

    def write(self, addr: int, n: int, value: int) -> None:
        for i in range(n):
            self.bytes[(addr + i) & ((1 << 64) - 1)] = (value >> (8 * i)) & 0xFF
        self.log.append(("w", addr, n, value & ((1 << (8 * n)) - 1)))
