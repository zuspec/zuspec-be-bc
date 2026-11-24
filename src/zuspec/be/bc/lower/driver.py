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

from ..model import ZbcModel, CoroDescriptor, Block, EntryPoint, Op, api_hash_of
from ..model import HEAP_STR_EQ, HEAP_NEG, HEAPK_STR, heap_imm
from .context import Lowerer, CoroCtx, branch_inherit
from .orchestration import lower_orch_stmt
from .coverage import with_cov_sample
from .procedural import (entry_function_symbols, init_proc_state, lower_comp_init,
                         lower_entry_function, proc_state)
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
        stmts=ctx.stmts,
        chans=ctx.chans,
        # an action's exec blocks are scoped by the action's coroutine
        cov=[(k, pc, scope or coro.name, path, sr, imp)
             for k, pc, scope, path, sr, imp in ctx.cov],
        regs=ctx.regs,
        discards=ctx.discards,
        locals=ctx.debug_locals(),
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
            "struct_at": dict(getattr(d, "struct_at", None) or {}),
        }
    return table


def _import_arg_types(d, types) -> list:
    """The ``(width, signed)`` of each value an IMPORT of *d* passes: a
    struct argument passes one per leaf (F6.4b)."""
    from .types import value_type, StructT
    out = []
    structs = getattr(d, "struct_at", None) or {}
    for i, t in enumerate(getattr(d, "arg_types", None) or []):
        st = value_type(structs[i], types) if i in structs else None
        if isinstance(st, StructT):
            out.extend([lt.width, lt.signed] for _, lt in st.leaves)
        else:
            out.append(list(t))
    return out


def _import_ret_types(d, types) -> dict:
    """``{"ret_types": [(width, signed), ...]}``, the leaves the host hands
    back, when *d* returns a struct (F6.4b2); else nothing. A string leaf
    is ``(64, False, "string")``: the host answers it with text (F6.3b2)."""
    from .types import value_type, StructT
    rt = (getattr(d, "struct_at", None) or {}).get("return")
    st = value_type(rt, types) if rt is not None else None
    if not isinstance(st, StructT):
        return {}
    return {"ret_types": [[lt.width, lt.signed] + (["string"] if lt.kind == "string" else [])
                          for _, lt in st.leaves]}


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


def _obj_layout(coro):
    """*coro*'s action-object slots in slot order: (name, type descriptor),
    the descriptor marked ``"rand": True`` for a rand field (an ENTRY's
    ``obj``)."""
    return [(f.name, _field_desc(f)) for f in sorted(
        getattr(coro, "subtree", None) or coro.fields or (), key=lambda f: f.slot)]


def _field_desc(f):
    """*f*'s type descriptor, ``"rand": True`` for a rand field; opaque
    when the type has none."""
    from .types import from_datatype
    try:
        d = from_datatype(f.datatype).descriptor() if f.datatype is not None else {}
    except LoweringError:
        d = {}
    if not d:
        d = {"kind": "opaque", "width": 0, "signed": False}
    if getattr(f, "rand", False):
        d["rand"] = True
    return d


_HEAP_TYPES = {"DataTypeList": "list", "DataTypeMap": "map", "DataTypeSet": "set"}


def _refuse_heap_fields(coros, comp_tree):
    """Refuse a ``rand`` action field of map or set type, at its line: bc
    solves the elements of a rand list only (F6.10e), and the field's slot
    would otherwise be dropped while the run goes on. A non-rand one is a
    heap handle the object holds (spec/heap.md)."""
    fields = [f for c in coros for f in (c.fields or ())]
    fields += list(getattr(comp_tree, "fields", None) or ())
    for f in fields:
        kind = _HEAP_TYPES.get(type(f.datatype).__name__)
        if kind not in (None, "list") and getattr(f, "rand", False):
            raise LoweringError(f"field '{f.name}': rand {kind} types are not supported "
                                f"by bc yet", loc=f.loc or getattr(f.datatype, "loc", None))


def _run_hooks(comp_tree) -> bool:
    """Has *comp_tree* any ``exec run_start`` / ``run_end`` block?"""
    return bool(getattr(comp_tree, "run_start", None)
                or getattr(comp_tree, "run_end", None))


def _exec_table(lowerer) -> tuple:
    """The EXEC section's executor rows and groups (F5.4a/b), when any action
    names an executor or a group has one."""
    from .procedural import exec_groups, exec_rows, exec_trait
    groups = [g for g in exec_groups(lowerer) if g["members"]]
    if not (lowerer.executors or groups):
        return [], []
    rows = exec_rows(lowerer)
    ids = {i.id: n for n, i in enumerate(rows)}
    return ([{"path": i.path, "type": i.type_qname,
              "trait": exec_trait(lowerer, i.type_qname)} for i in rows],
            [{"path": g["inst"].path, "trait": exec_trait(lowerer, g["inst"].type_qname),
              "members": [ids[m] for m in g["members"]]} for g in groups])


def _event_table(comp_tree) -> list:
    """The EVENT section's rows (F5.5a): every scalar ``channel_c`` instance
    (its ``.$count`` slot), in slot order, with the slots a host put fills."""
    if comp_tree is None:
        return []
    from .types import from_datatype
    by_name = {f.name: f for f in comp_tree.fields}
    out = []
    for f in sorted(comp_tree.fields, key=lambda f: f.slot):
        if not f.name.endswith(".$count"):
            continue
        path = f.name[:-len(".$count")]
        bufs = sorted(g.slot for n, g in by_name.items()
                      if n.startswith(path + ".$buf") and n[len(path) + 5:].isdigit())
        head = by_name.get(path + ".$head")
        if not bufs or head is None or bufs != list(range(bufs[0], bufs[0] + len(bufs))):
            continue
        try:
            d = from_datatype(by_name[path + ".$buf0"].datatype).descriptor()
        except LoweringError:
            continue
        out.append({"path": path, "kind": 2, "width": d.get("width", 0),
                    "signed": bool(d.get("signed")), "count": f.slot, "head": head.slot,
                    "buf": bufs[0], "depth": len(bufs)})
    return out


def _comp_table(comp_tree, init_idx):
    """The ``CompTable`` of *comp_tree*, its slots' type descriptors filled
    (the LAYOUT section)."""
    from ..interp.components import CompTable
    t = CompTable(comp_tree, init_idx)
    t.run_hooks = _run_hooks(comp_tree)
    t.slot_types = [_field_desc(f) for f in sorted(comp_tree.fields, key=lambda f: f.slot)]
    return t


def _activation_table(coro, tree, types):
    """The run-time table of *tree*, rooted at *coro*'s action."""
    from zuspec.ir.core import expr as E
    from zuspec.ir.core.xf.pss_lower.action_tree import Layouts
    from ..interp.activation import UID, ActivationTable
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
        if leaf.field.name == "uid" and len(leaf.path) == 1:
            init[slot] = UID             # a node's built-in uid: set per run
        elif iv is None:
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
                   comp_tree=None,
                   entry_functions: Optional[Iterable[str]] = None,
                   reg_offset=None,
                   debug: bool = False,
                   export_params=None,
                   regions=None) -> ZbcModel:
    """Lower a set of ``ScCoroutine`` into a complete :class:`ZbcModel`.

    ``imports`` is an optional list of ``ScImportDecl`` so procedural code can
    resolve value-returning import calls (e.g. ``getval(7)``) to ``IMPORT`` ops.

    ``trees`` maps a coroutine name to the ``ScActionTree`` of an activation
    rooted at it (P1.4): running that coroutine as the root runs the tree,
    every traversal on one object, its cones solved with lookahead.

    ``comp_tree`` is the ``ScComponentTree`` (P1.5): component attributes and
    functions resolve through it, and the coroutine constructing it runs
    before the entry.

    ``debug`` (P1.4) marks where each PSS statement starts (the LINE section,
    zbc-format.md "Statement markers"); without it the code is unchanged.

    ``entry_functions`` names functions (``<comp type>::<fn>``) the host may
    enter in an instance of the tree (pssc P1.3); ``"*"`` names every
    function of every component type in the tree. Each gets an entry in
    :attr:`ZbcModel.entries`, as does every top-level coroutine.

    ``reg_offset(group_type, name, index)`` gives the byte offset of
    register-group instance ``name`` (``name[index]`` for an array; index
    None otherwise): the group's own ``get_offset_of_instance[_array]``,
    which only a front end can evaluate. Without it a register is placed
    by the group type's dense ``offset_map`` and a sub-group cannot be.

    ``export_params`` maps a coroutine name to its export's parameters
    (``export A1(bit mode);``, LRM 20.10): the rand fields of the action a
    caller sets. They are the entry's ``params``, typed from its object.

    ``regions`` is the build's address regions, ``{path, k, addr, size,
    tag}`` each (``addr`` None for the k-th region of contiguous space
    ``path``), which ``get_tag()`` looks a handle up in.
    """
    coros = list(coros)
    trees = dict(trees or {})
    lowerer = Lowerer()
    lowerer.debug = debug
    lowerer.imports = _imports_table(imports)
    lowerer.blocking_targets = list(blocking_targets or [])
    lowerer.functions = dict(functions or {})
    lowerer.reg_offset = reg_offset
    lowerer.regions = regions
    lowerer.types = dict(types or {})
    if comp_tree is not None:
        from zuspec.ir.core.xf.pss_lower.comp_tree import CompLayouts
        lowerer.comps = CompLayouts(lowerer.types)
        lowerer.comp_tree = comp_tree
        # Executor claims (F5.4b3) are assigned in the trees' solves, which
        # need the groups add_executor fills here.
        if any(getattr(t, "template", None) == "executor_pkg::executor_claim_s"
               for t in lowerer.types.values()):
            from zuspec.ir.core.xf.pss_lower.action_tree import Layouts, build_tree
            from .procedural import exec_claim_members
            layouts = Layouts(lowerer.types)
            for name, t in list(trees.items()):
                try:
                    nt = build_tree(layouts, t.root, t.type_qname, comps=lowerer.comps,
                                    root_comp=comp_tree.root,
                                    claims=lambda *a: exec_claim_members(lowerer, *a))
                except UnsupportedConstructError as e:
                    raise LoweringError(str(e), loc=getattr(e, "loc", None)) from e
                nt.lookahead = t.lookahead
                trees[name] = nt
    # The coroutine constructing the component tree follows the top-level
    # ones (and precedes the synthesized branches).
    # With run_start/run_end blocks, it is always there, and $run_start and
    # $run_end follow it (ZBC_COMP_F_RUN_HOOKS).
    run_hooks = comp_tree is not None and _run_hooks(comp_tree)
    from .procedural import comp_ref_slots
    init_idx = (len(coros) if comp_tree is not None
                and (comp_tree.init or run_hooks or comp_ref_slots(comp_tree, lowerer))
                else None)
    lowerer.n_toplevel = (len(coros) + (1 if init_idx is not None else 0)
                          + (2 if run_hooks else 0))
    lowerer.solve_unconstrained = solve_unconstrained
    # A type with a node in some cone solves through SOLVE_NODE; the others
    # keep SOLVE, and a model with no cone keeps its bytecode (P1-D3).
    lowerer.cone_types = {t.nodes[n].type_qname for t in trees.values()
                          for c in t.cones for n in c.nodes}
    lowerer.scoped = any(t.cones for t in trees.values())
    coros = [_with_solve_point(c) if c.action_type in lowerer.cone_types else c
             for c in coros]
    coros = [with_cov_sample(c, lowerer.types) for c in coros]
    # Pre-register coroutine names so INVOKE/SPAWN can resolve forward references.
    for i, c in enumerate(coros):
        lowerer._coro_index[c.name] = i

    _refuse_heap_fields(coros, comp_tree)
    descs = [lower_coroutine(c, lowerer, blocking_targets=blocking_targets) for c in coros]
    if init_idx is not None:
        descs.append(lower_comp_init(comp_tree, lowerer))
    if run_hooks:
        descs.append(lower_comp_init(comp_tree, lowerer, comp_tree.run_start, "$run_start"))
        descs.append(lower_comp_init(comp_tree, lowerer, comp_tree.run_end, "$run_end"))
    entries = {c.name: EntryPoint(coro=i, qname=c.action_type, obj=_obj_layout(c))
               for i, c in enumerate(coros)}
    for name, pnames in (export_params or {}).items():
        e = entries.get(name)
        if e is None:
            continue
        slots = dict(e.obj)
        for p in pnames:
            d = slots.get(p)
            if d is None or not d.get("rand"):
                raise LoweringError(
                    f"export {e.qname}: parameter {p!r} must name a rand field "
                    f"of the action")
            e.params.append((p, {k: v for k, v in d.items() if k != "rand"}))
    syms: List[str] = []
    for sym in (entry_functions or ()):
        for s in (entry_function_symbols(lowerer) if sym == "*" else [sym]):
            if s not in syms:
                syms.append(s)
    for sym in syms:
        idx, params, ret, comp_type = lower_entry_function(sym, lowerer)
        entries[sym] = EntryPoint(coro=idx, kind="function", params=params,
                                  returns=ret, comp_type=comp_type)
    if lowerer.uses_heap:
        # A run-time string may equal a literal with another index (F6.3b1)
        for code, pc, neg in lowerer.string_cmps:
            ins = code[pc]
            ins.op = Op.HEAP
            ins.imm = heap_imm(HEAP_STR_EQ, HEAPK_STR) | (HEAP_NEG if neg else 0)
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

    imports_sig = {d.name: {"fn_id": d.fn_id, "blocking": bool(getattr(d, "blocking", False)),
                            "arg_types": _import_arg_types(d, types),
                            "ret_type": (list(d.ret_type)
                                         if getattr(d, "ret_type", None) is not None else None),
                            "string_at": list(getattr(d, "string_at", None) or []),
                            **_import_ret_types(d, types)}
                   for d in (imports or [])}
    # Synthesized PAR/SELECT branch sub-coroutines follow the top-level coros, so
    # their global indices (assigned at creation) match their list position here.
    exec_tab = _exec_table(lowerer)
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
        obj_layouts={e.coro: [n for n, _ in e.obj] for e in entries.values()
                     if e.kind == "action" and e.obj},
        activations=activations,
        components=(_comp_table(comp_tree, init_idx) if comp_tree is not None else None),
        comp_init=init_idx is not None,
        entries=entries,
        imports=imports_sig,
        api_hash=api_hash_of(entries, imports_sig),
        covgs=lowerer.covgs,
        executors=exec_tab[0],
        exec_groups=exec_tab[1],
        events=_event_table(comp_tree),
        opaque_regions=lowerer.opaque_regions,
    )


def lower_module(module, entry_action: Optional[str] = None,
                 profile: str = "codegen",
                 solve_unconstrained: bool = False,
                 entry_functions: Optional[Iterable[str]] = None,
                 reg_offset=None,
                 debug: bool = False,
                 regions=None, every_tree: bool = False) -> ZbcModel:
    """Lower a ``ScenarioModule`` (from ``PSSToScenarioPass``) into a ``ZbcModel``.

    Coroutines are lowered in the module's declaration order; the entry coroutine
    is ``entry_action`` (or the module's first export). The module's ``imports``
    (``ScImportDecl``) drive import-call resolution and blocking-import splitting.

    ``solve_unconstrained`` sends a solve problem with rand variables but no
    constraints to dv-solve too, so each variable is drawn from its domain. Off,
    such a problem goes to the run's configured backend -- which is what a test
    pinning solver values through a stub needs, and what leaves every
    unconstrained rand field at 0 under ``NativeBlobBackend``.

    ``entry_functions``: component functions the host may enter, and
    ``reg_offset``: register offsets, and ``regions``: address regions
    (see :func:`lower_scenario`).

    ``every_tree`` gives every action an activation table, not only the
    entry, the exports and actions whose constraints form cones, so any
    action run as the root records a plan (``RunOpts.plan``, M8b6c).
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
    if types:
        # Any action may be run as the root (`pssc run --root`), also one
        # another action traverses (so not an export): one whose
        # constraints form cones (`comp.f`, a list) gets its tree, as only
        # an activation solves those.
        from zuspec.ir.core.xf.pss_lower.action_tree import Layouts, build_tree
        from zuspec.ir.core.xf.pss_lower.comp_tree import CompLayouts
        layouts = Layouts(types)
        comps = CompLayouts(types) if comp_tree is not None else None
        for name, c in module.coroutines.items():
            if name in trees or not c.action_type \
                    or layouts.try_get(c.action_type) is None:
                continue
            try:
                nt = build_tree(layouts, name, c.action_type, comps=comps,
                                root_comp=comp_tree.root if comp_tree is not None else None)
            except UnsupportedConstructError:
                continue                # its traversing action's tree reports it
            if nt.cones or every_tree:
                trees[name] = nt

    return lower_scenario(coros, entry=entry_idx, blocking_targets=blocking,
                          imports=getattr(module, "imports", None), profile=profile,
                          functions=getattr(module, "functions", None),
                          solve_unconstrained=solve_unconstrained,
                          types=types, trees=trees, comp_tree=comp_tree,
                          entry_functions=entry_functions, reg_offset=reg_offset,
                          debug=debug, regions=regions,
                          export_params=getattr(module, "export_params", None))
