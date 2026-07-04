"""
adapter.py -- the T1-A differential harness.

Drives **one** hand-built ``zuspec.ir.core`` fixture through *both* runtimes and
returns comparable results:

* the **legacy** ``zuspec-be-py`` runtime (``build_registry`` -> ``create(seed)`` ->
  ``await ep.<Action>()``); and
* the **ZBC oracle** (``PSSToScenarioPass`` -> ``lower_module`` -> ``run_model``).

What is compared, and why (M1 reality):

* **Import call sequence + args** -- fully deterministic on both sides; the
  load-bearing check that the oracle's lowering + orchestration reproduce the
  legacy executor's foreign-call behavior.
* **Solved field set + values** -- the *solver* is a pinned external dependency
  the M1 oracle shares (roadmap Q1: SOLVE calls the solver backend), so the oracle
  is seeded from the legacy-produced values and we verify it routes them to the
  right output fields per the value ABI. Bit-exact *independent* solving is **not**
  asserted: the legacy per-action seed is ``run_seed ^ id(action_type)`` (see
  ``activity_runner``), which is not reproducible across registry rebuilds -- an
  ``id()`` wart, not part of the determinism contract the oracle implements.

The adapter deliberately lives in the test tree: the differential reference is a
test-only dependency on ``zuspec-be-py`` (guarded by ``importorskip``), not a
runtime edge of ``zuspec-be-bc``.
"""

import asyncio
import dataclasses as dc
from typing import Any, Dict, List, Optional, Tuple

import zuspec.ir.core as ir
from zuspec.ir.core.data_type import Function, DataTypeInt
from zuspec.ir.core.stmt import Arguments, Arg
from zuspec.ir.core.xf import PSSToScenarioPass

from zuspec.be.bc.lower import lower_module
from zuspec.be.bc.interp import run_model, Obj, ImportProvider, SolveBackend


# --------------------------------------------------------------------------- #
# Import-function synthesis (bridge be-py ImportSpec -> ctx.import_functions)
# --------------------------------------------------------------------------- #

def synth_import_functions(import_specs: Dict[str, Any], arity: int = 1) -> List[Function]:
    """Build the ``ctx.import_functions`` ``Function`` list ``PSSToScenarioPass``
    reads, from the be-py ``{name: ImportSpec}`` map the legacy runtime uses.

    A ``target`` import is void/blocking; a ``solve`` import returns a value.
    """
    fns: List[Function] = []
    for name, spec in import_specs.items():
        is_target = bool(getattr(spec, "is_target", False))
        returns = None if is_target else DataTypeInt(bits=32, signed=False)
        args = Arguments(args=[Arg(arg=f"a{i}", annotation=DataTypeInt(bits=32))
                               for i in range(arity)])
        fns.append(Function(name=name, is_import=True, is_target=is_target,
                            is_solve=bool(getattr(spec, "is_solve", False)),
                            returns=returns, args=args))
    return fns


# --------------------------------------------------------------------------- #
# Recording import implementations
# --------------------------------------------------------------------------- #

class _RecordingImpl:
    """Wraps a user import object, recording (name, args) on every call."""

    def __init__(self, impl: Any, names: List[str]):
        self._impl = impl
        self.calls: List[Tuple[str, list]] = []
        for name in names:
            self._install(name)

    def _install(self, name: str) -> None:
        target = getattr(self._impl, name)

        def wrapper(*args):
            self.calls.append((name, list(args)))
            return target(*args)

        setattr(self, name, wrapper)


class _OracleImportProvider(ImportProvider):
    """Dispatches ``IMPORT`` (by fn_id) to the user impl by name, recording calls."""

    def __init__(self, impl: Any, id2name: Dict[int, str]):
        self._impl = impl
        self._id2name = id2name
        self.calls: List[Tuple[str, list]] = []

    def call(self, fn_id: int, args) -> Optional[int]:
        name = self._id2name[fn_id]
        args = list(args)
        self.calls.append((name, args))
        result = getattr(self._impl, name)(*args)
        return result if result is not None else 0


class _LegacyValueSolveBackend(SolveBackend):
    """Return solved values captured from the legacy run (pinned solver)."""

    def __init__(self, values: Dict[str, int]):
        self._values = dict(values)

    def randomize(self, obj, problem, seed) -> Dict[str, int]:
        return {name: self._values[name]
                for name in problem.var_names if name in self._values}


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #

@dc.dataclass
class DiffResult:
    legacy_fields: Dict[str, int]
    oracle_fields: Dict[str, int]
    legacy_calls: List[Tuple[str, list]]
    oracle_calls: List[Tuple[str, list]]


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #

def _rand_field_names(ctx: ir.Context, action_qname: str) -> List[str]:
    dt = ctx.type_m[action_qname]
    return [f.name for f in getattr(dt, "fields", [])
            if getattr(f, "rand_kind", None) is not None]


def _ensure_lowerable(ctx: ir.Context, action_qname: str) -> None:
    """Give a rand-only atomic action an empty ``body`` so it lowers.

    ``PSSToScenarioPass._is_action`` recognizes an action by its exec body or
    activity; a randomize-only action (no body) is valid to the legacy runtime but
    invisible to the pass. An empty body is a semantic no-op and leaves the solve
    problem (from the rand fields) intact.
    """
    dt = ctx.type_m[action_qname]
    fns = list(getattr(dt, "functions", []) or [])
    has_body = any(getattr(f, "name", None) == "body" for f in fns)
    if getattr(dt, "activity_ir", None) is None and not has_body:
        dt.functions = fns + [Function(name="body", body=[])]


def run_legacy(ctx, action_qname: str, *, seed: Optional[int] = None,
               imports_impl: Any = None,
               import_specs: Optional[Dict[str, Any]] = None
               ) -> Tuple[Dict[str, int], List[Tuple[str, list]]]:
    """Run the legacy be-py runtime; return (solved fields, import call log).

    The legacy runtime is used here strictly as the **differential reference**
    (W8 / P1-14): its role is declared in ``zuspec.be.py.rt.legacy_status`` and
    gated by ``legacy_reference_enabled()``.
    """
    from zuspec.be.py import build_registry
    from zuspec.be.py.rt.legacy_status import legacy_reference_enabled

    if not legacy_reference_enabled():
        raise RuntimeError(
            "legacy runtime disabled as a differential reference "
            "(ZUSPEC_LEGACY_REFERENCE=0); nothing to compare against")

    simple = action_qname.rsplit("::", 1)[-1]
    reg = build_registry(ctx, export_actions=[action_qname],
                         import_specs=import_specs or None)

    rec = None
    if imports_impl is not None:
        rec = _RecordingImpl(imports_impl, list((import_specs or {}).keys()))
        ep = reg.create(rec, seed=seed)
    else:
        ep = reg.create(seed=seed)

    async def _go():
        return await getattr(ep, simple)()

    result = asyncio.run(_go())

    fields = {n: int(getattr(result, n)) for n in _rand_field_names(ctx, action_qname)}
    calls = rec.calls if rec is not None else []
    return fields, calls


def run_oracle(ctx, action_qname: str, *, seed: Optional[int] = None,
               imports_impl: Any = None,
               import_specs: Optional[Dict[str, Any]] = None,
               solve_values: Optional[Dict[str, int]] = None
               ) -> Tuple[Dict[str, int], List[Tuple[str, list]]]:
    """Run the ZBC oracle over the same fixture; return (fields, import call log)."""
    simple = action_qname.rsplit("::", 1)[-1]

    if import_specs:
        ctx.import_functions = synth_import_functions(import_specs)
    _ensure_lowerable(ctx, action_qname)

    module = PSSToScenarioPass(exports=[simple]).lower(ctx)
    model = lower_module(module, entry_action=simple)

    id2name = {d.fn_id: d.name for d in module.imports}
    provider = (_OracleImportProvider(_RecordingImpl(imports_impl,
                                                     list((import_specs or {}).keys())),
                                      id2name)
                if imports_impl is not None else _OracleImportProvider(object(), id2name))

    rand_names = _rand_field_names(ctx, action_qname)
    obj = Obj(field_names=rand_names) if rand_names else None
    backend = _LegacyValueSolveBackend(solve_values or {})

    res = run_model(model, obj=obj, seed=seed or 0,
                    solve_backend=backend, import_provider=provider)
    return res.fields, provider.calls


def differential(ctx, action_qname: str, *, seed: Optional[int] = None,
                 imports_impl: Any = None,
                 import_specs: Optional[Dict[str, Any]] = None) -> DiffResult:
    """Run both runtimes over ``ctx`` and return their results side by side.

    The oracle's SOLVE is seeded from the legacy-produced field values (pinned
    solver), so ``legacy_fields == oracle_fields`` proves write-back routing.
    """
    legacy_fields, legacy_calls = run_legacy(
        ctx, action_qname, seed=seed, imports_impl=imports_impl,
        import_specs=import_specs)
    oracle_fields, oracle_calls = run_oracle(
        ctx, action_qname, seed=seed, imports_impl=imports_impl,
        import_specs=import_specs, solve_values=legacy_fields)
    return DiffResult(legacy_fields, oracle_fields, legacy_calls, oracle_calls)
