"""
driver.py -- the lowering entry points (P1-6).

``lower_coroutine`` runs the shared :class:`CoroutineFSMPass`, walks its blocks,
and builds a :class:`CoroDescriptor`. ``lower_scenario`` lowers a set of
coroutines into a complete :class:`ZbcModel` (code + provenance + const pool +
in-memory problem table).
"""

from typing import Iterable, List, Optional

from zuspec.ir.core.xf.coro_fsm import CoroutineFSMPass
from zuspec.ir.core.xf.validate import UnsupportedConstructError

from ..model import ZbcModel, CoroDescriptor, Block, Op
from .context import Lowerer, CoroCtx
from .orchestration import lower_orch_stmt
from .procedural import init_proc_state
from .errors import LoweringError


def lower_coroutine(coro, lowerer: Lowerer,
                    blocking_targets: Optional[Iterable[str]] = None) -> CoroDescriptor:
    """Lower one ``ScCoroutine`` into a :class:`CoroDescriptor`.

    Uses the FSM pass to split at suspend points; each block's statements lower to
    procedural/orchestration ops and its trailing suspend to an orchestration op.
    A loop/branch that *contains* a suspend is kept opaque (``allow_nested_suspend``)
    and lowered to a flat, resumable code stream -- the oracle's VM saves the pc
    across a suspend, so the back-edge + interior suspend execute directly.
    """
    try:
        form = CoroutineFSMPass(blocking_targets=list(blocking_targets or []),
                                allow_nested_suspend=True).run(coro)
    except UnsupportedConstructError as e:
        raise LoweringError(
            f"coroutine {coro.name!r} is out of M1 scope: {e}",
            loc=getattr(e, "loc", None),
        ) from e

    ctx = CoroCtx.create(lowerer, list(form.frame_locals), coro_name=coro.name)
    ctx.src_fields = list(getattr(coro, "fields", []) or [])
    ctx.action_type = getattr(coro, "action_type", None)
    init_proc_state(ctx, coro)
    coro_sr = lowerer.prov.src_ref(coro, name=coro.name)

    blocks: List[Block] = []
    for blk in form.blocks:
        pc_start = len(ctx.code)
        for st in blk.stmts:
            lower_orch_stmt(ctx, st, is_suspend=False)
        suspend_op = 0
        if blk.suspend is not None:
            op = lower_orch_stmt(ctx, blk.suspend, is_suspend=True)
            suspend_op = int(op) if op is not None else 0
        blocks.append(Block(
            idx=blk.idx,
            pc_start=pc_start,
            pc_end=len(ctx.code),
            suspend_op=suspend_op,
        ))

    return CoroDescriptor(
        name=coro.name,
        code=ctx.code,
        blocks=blocks,
        frame_locals=ctx.frame_locals,
        src_ref=coro_sr,
    )


def _imports_table(imports) -> dict:
    """Build the name -> {fn_id, blocking, ret_type} map from ``ScImportDecl``s."""
    table = {}
    for d in (imports or []):
        table[d.name] = {
            "fn_id": d.fn_id,
            "blocking": bool(getattr(d, "blocking", False)),
            "ret_type": getattr(d, "ret_type", None),
        }
    return table


def lower_scenario(coros, entry: int = 0,
                   blocking_targets: Optional[Iterable[str]] = None,
                   imports=None,
                   profile: str = "codegen",
                   functions=None,
                   solve_unconstrained: bool = False,
                   types=None) -> ZbcModel:
    """Lower a set of ``ScCoroutine`` into a complete :class:`ZbcModel`.

    ``imports`` is an optional list of ``ScImportDecl`` so procedural code can
    resolve value-returning import calls (e.g. ``getval(7)``) to ``IMPORT`` ops.
    """
    coros = list(coros)
    lowerer = Lowerer()
    lowerer.imports = _imports_table(imports)
    lowerer.blocking_targets = list(blocking_targets or [])
    lowerer.n_toplevel = len(coros)
    lowerer.functions = dict(functions or {})
    lowerer.types = dict(types or {})
    lowerer.solve_unconstrained = solve_unconstrained
    # Pre-register coroutine names so INVOKE/SPAWN can resolve forward references.
    for i, c in enumerate(coros):
        lowerer._coro_index[c.name] = i

    descs = [lower_coroutine(c, lowerer, blocking_targets=blocking_targets) for c in coros]

    # Synthesized PAR/SELECT branch sub-coroutines follow the top-level coros, so
    # their global indices (assigned at creation) match their list position here.
    return ZbcModel(
        coros=descs + lowerer.branch_coros,
        entry_coro=entry,
        consts=lowerer.consts,
        prov=lowerer.prov.table,
        problems=lowerer.problems,
        selects=lowerer.selects,
        profile=profile,
        messages=lowerer.messages,
        strings=lowerer.strings,
        obj_layouts={i: [f.name for f in sorted(c.fields, key=lambda f: f.slot)]
                     for i, c in enumerate(coros) if getattr(c, "fields", None)},
    )


def lower_module(module, entry_action: Optional[str] = None,
                 profile: str = "codegen",
                 solve_unconstrained: bool = False) -> ZbcModel:
    """Lower a ``ScenarioModule`` (from ``PSSToScenarioPass``) into a ``ZbcModel``.

    Coroutines are lowered in the module's declaration order; the entry coroutine
    is ``entry_action`` (or the module's first export). The module's ``imports``
    (``ScImportDecl``) drive import-call resolution and blocking-import splitting.

    ``solve_unconstrained`` sends a solve problem with rand variables but no
    constraints to dv-solve too, so each variable is drawn from its domain. Off,
    such a problem goes to the run's configured backend -- which is what a test
    pinning solver values through a stub needs, and what leaves every
    unconstrained rand field at 0 under ``NativeBlobBackend``.
    """
    coros = list(module.coroutines.values())
    names = [c.name for c in coros]

    entry = entry_action
    if entry is None:
        exports = list(getattr(module, "export_actions", []) or [])
        entry = exports[0] if exports else (names[0] if names else None)
    entry_idx = names.index(entry) if entry in names else 0

    # Suspend targets for the FSM split / blocking-INVOKE derivation:
    #   * blocking (target/void) imports, and
    #   * every action coroutine -- a `do Sub` traversal is sequential in PSS, so
    #     its INVOKE must block until the sub-action completes. Parallelism comes
    #     solely from ScPar's SPAWN, never from a bare traversal.
    blocking = [d.name for d in getattr(module, "imports", [])
                if getattr(d, "blocking", False)]
    blocking += names

    return lower_scenario(coros, entry=entry_idx, blocking_targets=blocking,
                          imports=getattr(module, "imports", None), profile=profile,
                          functions=getattr(module, "functions", None),
                          solve_unconstrained=solve_unconstrained,
                          types=getattr(module, "types", None))
