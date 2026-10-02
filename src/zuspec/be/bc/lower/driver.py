"""
driver.py -- the lowering entry points (P1-6).

``lower_coroutine`` runs the shared :class:`CoroutineFSMPass`, walks its blocks,
and builds a :class:`CoroDescriptor`. ``lower_scenario`` lowers a set of
coroutines into a complete :class:`ZbcModel` (code + provenance + const pool +
in-memory problem table).
"""

import dataclasses as dc
from typing import Iterable, List, Optional

from zuspec.ir.core import scenario as SC

from zuspec.ir.core.xf.coro_fsm import CoroutineFSMPass
from zuspec.ir.core.xf.validate import UnsupportedConstructError

from ..model import ZbcModel, CoroDescriptor, Block, Op
from .context import Lowerer, CoroCtx, branch_inherit
from .orchestration import lower_orch_stmt
from .procedural import init_proc_state, lower_comp_init, proc_state
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


def new_ctx_like(ctx: CoroCtx, name: str) -> CoroCtx:
    """A context for a coroutine bc synthesizes from *ctx*'s (a called
    function's form): the same object layout, action type and component."""
    sub = CoroCtx.create(ctx.lowerer, [], coro_name=name)
    sub.src_fields = ctx.src_fields
    sub.action_type = ctx.action_type
    init_proc_state(sub, SC.ScCoroutine(name=name, body=[], **branch_inherit(ctx)))
    st, src = proc_state(sub), proc_state(ctx)
    st.comp_self, st.exec_kind = src.comp_self, src.exec_kind
    return sub


def _imports_table(imports) -> dict:
    """Build the name -> {fn_id, blocking, ret_type} map from ``ScImportDecl``s."""
    table = {}
    for d in (imports or []):
        table[d.name] = {
            "fn_id": d.fn_id,
            "blocking": bool(getattr(d, "blocking", False)),
            "ret_type": getattr(d, "ret_type", None),
            "string_at": list(getattr(d, "string_at", None) or []),
        }
    return table


def _with_solve_point(coro):
    """*coro* with a solve point: a cone member solves at its traversal even
    when its type has nothing of its own to solve (after its initial values
    and ``pre_solve``, as a solve would be)."""
    if any(isinstance(s, SC.ScSolveProblem) for s in coro.body):
        return coro
    at = 0
    while (at < len(coro.body) and isinstance(coro.body[at], SC.ScExecBlock)
           and coro.body[at].kind in ("init", "pre_solve")):
        at += 1
    body = list(coro.body)
    body.insert(at, SC.ScSolveProblem())
    return dc.replace(coro, body=body)


def _activation_table(coro, tree, types):
    """The run-time table of *tree*, rooted at *coro*'s action."""
    from zuspec.ir.core import expr as E
    from zuspec.ir.core.xf.pss_lower.action_tree import Layouts
    from ..interp.activation import ActivationTable
    layout = coro.subtree or coro.fields
    names = [f.name for f in sorted(layout, key=lambda f: f.slot)]
    # A node choosing its component instance holds the choice past the
    # action subtrees (P1-D4).
    names += [""] * (tree.size - len(names))
    for n in tree.nodes:
        if n.comp_slot is not None:
            names[n.comp_slot] = (n.path + "." if n.path else "") + "comp"
    # ... and so does a component attribute a constraint reads (comp.f).
    for c in tree.cones:
        for v in c.vars:
            if v.slot >= len(layout) and not names[v.slot]:
                names[v.slot] = v.name
    # ... and so does each state pool's current object (B5).
    for p in getattr(tree, "pools", ()):
        for leaf, slot in zip(p.leaves, p.slots):
            names[slot] = "%s@%d.%s" % (p.name, p.inst, leaf)
    init = {}
    for name, slot, leaf in Layouts(types or {}).subtree(tree.type_qname):
        if leaf.rand:
            continue
        iv = getattr(leaf.field, "initial_value", None)
        if iv is None:
            init[slot] = 0
        elif isinstance(iv, E.ExprConstant) and isinstance(iv.value, (int, bool)):
            init[slot] = int(iv.value)
        else:
            init[slot] = None            # computed at traversal: not known ahead
    table = ActivationTable(tree, names, init)
    table.check()
    return table


def lower_scenario(coros, entry: int = 0,
                   blocking_targets: Optional[Iterable[str]] = None,
                   imports=None,
                   profile: str = "codegen",
                   functions=None,
                   solve_unconstrained: bool = False,
                   types=None,
                   trees=None,
                   comp_tree=None) -> ZbcModel:
    """Lower a set of ``ScCoroutine`` into a complete :class:`ZbcModel`.

    ``imports`` is an optional list of ``ScImportDecl`` so procedural code can
    resolve value-returning import calls (e.g. ``getval(7)``) to ``IMPORT`` ops.

    ``trees`` maps a coroutine name to the ``ScActionTree`` of an activation
    rooted at it (P1.4): running that coroutine as the root runs the tree,
    every traversal on one object, its cones solved with lookahead.

    ``comp_tree`` is the ``ScComponentTree`` (P1.5): component attributes and
    functions resolve through it, and the coroutine constructing it runs
    before the entry.
    """
    coros = list(coros)
    trees = dict(trees or {})
    lowerer = Lowerer()
    lowerer.imports = _imports_table(imports)
    lowerer.blocking_targets = list(blocking_targets or [])
    lowerer.functions = dict(functions or {})
    lowerer.types = dict(types or {})
    if comp_tree is not None:
        from zuspec.ir.core.xf.pss_lower.comp_tree import CompLayouts
        lowerer.comps = CompLayouts(lowerer.types)
        lowerer.comp_tree = comp_tree
    # The coroutine constructing the component tree follows the top-level
    # ones (and precedes the synthesized branches).
    init_idx = len(coros) if comp_tree is not None and comp_tree.init else None
    lowerer.n_toplevel = len(coros) + (1 if init_idx is not None else 0)
    lowerer.solve_unconstrained = solve_unconstrained
    # A type with a node in some cone solves through SOLVE_NODE; the others
    # keep SOLVE, and a model with no cone keeps its bytecode (P1-D3).
    lowerer.cone_types = {t.nodes[n].type_qname for t in trees.values()
                          for c in t.cones for n in c.nodes}
    lowerer.scoped = any(t.cones for t in trees.values())
    coros = [_with_solve_point(c) if c.action_type in lowerer.cone_types else c
             for c in coros]
    # Pre-register coroutine names so INVOKE/SPAWN can resolve forward references.
    for i, c in enumerate(coros):
        lowerer._coro_index[c.name] = i

    descs = [lower_coroutine(c, lowerer, blocking_targets=blocking_targets) for c in coros]
    if init_idx is not None:
        descs.append(lower_comp_init(comp_tree, lowerer))
    for ins, target in lowerer.inited_invokes:
        args = list(ins.args)
        args[3] = lowerer.init_end.get(target, 0)
        ins.args = tuple(args)

    activations = {}
    for i, c in enumerate(coros):
        if c.name in trees:
            try:
                activations[i] = _activation_table(c, trees[c.name], types)
            except UnsupportedConstructError as e:
                raise LoweringError(str(e), loc=getattr(e, "loc", None)) from e

    from ..interp.components import CompTable
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
        obj_layouts={i: [f.name for f in sorted(getattr(c, "subtree", None) or c.fields,
                                                key=lambda f: f.slot)]
                     for i, c in enumerate(coros) if getattr(c, "fields", None)},
        activations=activations,
        components=(CompTable(comp_tree, init_idx) if comp_tree is not None else None),
        comp_init=init_idx is not None,
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
    elif entry not in names:
        from zuspec.ir.core.xf.pss_lower import coro_key
        try:
            entry = coro_key(module, entry) or entry
        except UnsupportedConstructError as e:
            raise LoweringError(str(e)) from e
    entry_idx = names.index(entry) if entry in names else 0

    # Suspend targets for the FSM split / blocking-INVOKE derivation:
    #   * blocking (target/void) imports, and
    #   * every action coroutine -- a `do Sub` traversal is sequential in PSS, so
    #     its INVOKE must block until the sub-action completes. Parallelism comes
    #     solely from ScPar's SPAWN, never from a bare traversal.
    blocking = [d.name for d in getattr(module, "imports", [])
                if getattr(d, "blocking", False)]
    blocking += names

    # The activation an entry runs: an export's tree, or (an entry nothing
    # exports) one built for it.
    trees = dict(getattr(module, "trees", None) or {})
    types = getattr(module, "types", None)
    comp_tree = getattr(module, "comp_tree", None)
    ec = module.coroutines.get(entry) if entry is not None else None
    if ec is not None and entry not in trees and ec.action_type and types:
        from zuspec.ir.core.xf.pss_lower.action_tree import Layouts, build_tree
        from zuspec.ir.core.xf.pss_lower.comp_tree import CompLayouts
        layouts = Layouts(types)
        if layouts.try_get(ec.action_type) is not None:
            try:
                trees[entry] = build_tree(
                    layouts, entry, ec.action_type,
                    comps=CompLayouts(types) if comp_tree is not None else None,
                    root_comp=comp_tree.root if comp_tree is not None else None)
            except UnsupportedConstructError as e:
                raise LoweringError(str(e), loc=getattr(e, "loc", None)) from e

    return lower_scenario(coros, entry=entry_idx, blocking_targets=blocking,
                          imports=getattr(module, "imports", None), profile=profile,
                          functions=getattr(module, "functions", None),
                          solve_unconstrained=solve_unconstrained,
                          types=types, trees=trees, comp_tree=comp_tree)
