"""
activation.py -- one activation of an exported action, and its scope solve (P1.4).

An activation is ONE object (P1-D1): every action its activity can run is a node
of the action tree (``ScActionTree``, ir-core ``xf/pss_lower/action_tree.py``)
with its own slot range, and each frame runs at its node's base. This module holds
the run-time side:

* :class:`ActivationTable` -- the tree, flattened for the interpreter: nodes,
  scopes, traversal sites, cones, and what each block resets. Built once at
  lowering (``ZbcModel.activations``).
* :class:`Activation` -- the state of one run: which nodes hold committed values,
  which sites have run, which branch scopes were entered.

**Which constraints are in force** when node N is solved (P1-D2, LRM 13.4.7-10).
A node is *live* if it holds committed values or a traversal of it is still to
come in a live scope (a branch or loop body is live once entered; any other block
with its parent). A constraint is in force when every node it reads is live --
one through a handle that will never be traversed is vacuously satisfied (13.4.8)
-- and: a type constraint, or a node's choice of component instance (P1-D4),
when its owner is live; an activity ``constraint``, when its scope is (13.1.9
b.3); a ``with``, at its own traversal, before it as lookahead, and after it
while the values it chose stand (13.1.4).

With the tree's ``lookahead`` off (calibration only, P1.6), a constraint is in
force only when every node it reads is N or holds committed values: nothing
still to be traversed constrains N's choice.

**What is pinned.** Committed values, and every non-rand value: the object's
value for a node that has started (its initial values and ``pre_solve`` have
run), else the attribute's constant initial value; a component attribute
(``comp.f``), from the component object. N's own rand values are free;
so are the rand values of live nodes not yet traversed -- that freedom IS the
lookahead (13.4.9).

**Resets (13.4.8).** Entering a node resets its subtree: its handles are
uninitialized on entry to its activity. Entering a block resets the nodes
traversed in it (and their subtrees), which is what makes a loop iteration see
fresh handles (Ex 180).

The solve problem is built per set of constraints in force (cached), from the
cone's IR, with the same translator ``SOLVE`` blobs use, and compiled once per
run (``solve_cache.SolveCache``); each solve pins the committed values between
a checkpoint and a restore. A solve that exhausts its conflict budget is a
:class:`~.solve_cache.SolveBudgetError` naming the traversal. No solution is a :class:`ScopeUnsatError` naming the
traversal and the constraints in force, never a fallback.

**A solution serves the nodes it chose freely.** Each solve of a cone finds
values for all of it, and the cone keeps them as its *witness*. When a later
node of the cone is solved, and every value now pinned equals the witness's,
and the constraints now in force are among those the witness satisfied, the
witness is a solution of this solve's problem too: the node takes its values
from it, without solving. Only values the witness solve chose -- the node's
variables were free in it -- and only once per node, so a loop iteration
never repeats the last one's values (B6e).
"""

from __future__ import annotations

import dataclasses as dc
import types
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from zuspec.ir.core import constraint as C
from zuspec.ir.core import expr as E
from zuspec.ir.core import scenario as SC
from zuspec.ir.core.expr_phase2 import ExprIfExp as _IfExpP2

from . import faults
from .ops_proc import VMError
from .solve_cache import SolveBudgetError

_MASK64 = (1 << 64) - 1


class ScopeUnsatError(VMError):
    """No values satisfy the constraints in force at a traversal."""


@dc.dataclass
class _Node:
    path: str
    type_qname: str
    base: int
    size: int
    parent: Optional[int]
    site_base: int = 0          # its first site in ``sites``
    scope_base: int = 0         # its first scope in ``scopes``
    #: its component instance, as an offset from its parent's (the root's,
    #: from instance 0); None when its solve chooses one (P1-D4)
    comp_rel: Optional[int] = 0
    #: the slot holding that choice
    comp_slot: Optional[int] = None


@dc.dataclass
class _Branch:
    """One branch of a ``parallel``: its key below the parallel in a site's
    scope chain, the nodes in it that lock, and the node whose cone each
    probe of a footprint solves."""
    key: object
    lockers: List[int]
    probes: List[int]


@dc.dataclass
class _Par:
    """A ``parallel`` whose branches lock instances of the same pools."""
    branches: List[_Branch]
    shared: set


@dc.dataclass
class _Witness:
    """The last solution of a cone: the constraints it satisfied, every
    variable's value, the variables it chose (not pinned), and the nodes that
    have taken their values from it."""
    enabled: frozenset
    values: list
    free: frozenset
    taken: set


@dc.dataclass
class _Cone:
    id: int
    nodes: List[int]
    vars: List[SC.ScScopeVar]
    constraints: List[SC.ScScopeConstraint]


@dc.dataclass
class PlanLog:
    """The realised plan of a run (M8b6): what ran, with what values, and
    which way each choice went, recorded as the run decides it. Opt-in
    (``RunOpts.plan``); a run without one records nothing.

    ``occurrences``: one dict per traversal started, in start order (its
    ``id`` is its index): ``site`` (None: the root), ``path``, ``stim``,
    ``type``, ``comp`` (instance index; set when it ends), ``fields`` (name
    -> value, from its solve; empty for a node in no cone), ``start`` and
    ``end`` (this log's event counter; ``end`` None while it runs) and
    ``completed``. ``control``: one dict per choice, ``{"occ", "kind",
    "select", "value"}`` (``kind`` "select": the branch index, None when
    none was eligible); from scope entries (M8b6b), ``{"occ", "kind",
    "scope", "value"}``: ``kind`` "if" (value True: the then branch, False:
    the else branch) or "repeat" (value: the iterations whose body was
    entered; a loop that ran none has no record). Only a model with a cone
    enters scopes, so a model without one records neither.

    ``flow`` (M8b6b): one dict per flow object, ``{"id", "kind", "pool",
    "producer", "consumers", "fields"}``: ``kind`` "buffer" (an output
    picked from its pool, or a bound pair), "state" (an output, or the
    initial object, producer None) or "stream" (a bound or paired output);
    ``fields`` are the object's leaves. ``resources``: ``{"pool",
    "instance", "claims": [{"occ", "lock"}]}`` per instance claimed.
    :meth:`order` gives the realised order."""
    occurrences: List[dict] = dc.field(default_factory=list)
    control: List[dict] = dc.field(default_factory=list)
    flow: List[dict] = dc.field(default_factory=list)
    resources: List[dict] = dc.field(default_factory=list)
    tick: int = 0
    _res: Dict[tuple, dict] = dc.field(default_factory=dict, repr=False)

    def _next(self) -> int:
        self.tick += 1
        return self.tick

    def start(self, site, path: str, stim: str, type_qname: str) -> int:
        occ = len(self.occurrences)
        self.occurrences.append({
            "id": occ, "site": site, "path": path, "stim": stim,
            "type": type_qname, "comp": None, "fields": {},
            "start": self._next(), "end": None, "completed": None})
        return occ

    def end(self, occ: int, comp, completed: bool) -> None:
        o = self.occurrences[occ]
        o["comp"], o["completed"], o["end"] = comp, completed, self._next()

    def choice(self, occ, kind: str, select: int, value) -> None:
        self.control.append({"occ": occ, "kind": kind, "select": select,
                             "value": value})

    def flow_out(self, kind: str, pool: Optional[str], producer, fields: dict) -> int:
        """A new flow object; its id."""
        fid = len(self.flow)
        self.flow.append({"id": fid, "kind": kind, "pool": pool,
                          "producer": producer, "consumers": [],
                          "fields": fields})
        return fid

    def consume(self, fid: int, occ) -> None:
        if occ not in self.flow[fid]["consumers"]:
            self.flow[fid]["consumers"].append(occ)

    def claim(self, pool: str, instance: int, occ, lock: bool) -> None:
        r = self._res.get((pool, instance))
        if r is None:
            r = self._res[(pool, instance)] = {"pool": pool, "instance": instance,
                                               "claims": []}
            self.resources.append(r)
        r["claims"].append({"occ": occ, "lock": lock})

    def order(self) -> List[List[int]]:
        """The realised order: ``[a, b]`` where occurrence *a* ended before
        *b* started, transitively reduced (no *c* between them). The root
        and occurrences still running are left out."""
        done = sorted((o for o in self.occurrences
                       if o["site"] is not None and o["end"] is not None),
                      key=lambda o: o["start"])
        out = []
        for b in done:
            before = [a for a in done if a["end"] < b["start"]]
            if not before:
                continue
            # a pair is implied when some c started after a ended (and
            # ended before b started)
            last = max(a["start"] for a in before)
            out.extend([a["id"], b["id"]] for a in before if a["end"] > last)
        return sorted(out)

    def to_json(self, seed) -> dict:
        """The plan as ``pss-plan`` 1.0 (``pssc test --plan-out``): the
        records above with the run's ``seed`` and :meth:`order`."""
        return {"format": "pss-plan", "version": "1.0", "seed": seed,
                "occurrences": self.occurrences, "flow": self.flow,
                "resources": self.resources, "order": self.order(),
                "control": self.control}


#: An init value marking a node's built-in ``uid`` slot (PSS 3.1, clause 9).
UID = "uid"


class ActivationTable:
    """The action tree of one exported action, laid out for the interpreter."""

    def __init__(self, tree: SC.ScActionTree, names: List[str],
                 init_values: Dict[int, Optional[int]]):
        """*names*: the object's slot names (the root type's subtree
        layout). *init_values*: slot -> a non-rand attribute's constant
        initial value, or None where it is not a constant."""
        self.tree = tree
        self.size = tree.size
        self.nodes = [_Node(n.path, n.type_qname, n.base, n.size, n.parent,
                            comp_rel=(n.comp[0] if len(n.comp) == 1 else None),
                            comp_slot=n.comp_slot)
                      for n in tree.nodes]
        for st in reversed(tree.sites):
            self.nodes[st.owner].site_base = st.id
        for sc in reversed(tree.scopes):
            self.nodes[sc.node].scope_base = sc.id
        #: scope -> (commits on entry only, parent)
        self.scopes = [(sc.kind in SC.COMMITTING_SCOPES, sc.parent) for sc in tree.scopes]
        self.sites = [(st.owner, st.target, st.scope) for st in tree.sites]
        self.cones = [_Cone(c.id, list(c.nodes), list(c.vars), list(c.constraints))
                      for c in tree.cones]
        #: node -> the cone it is a member of
        self.cone_of: Dict[int, int] = {}
        for c in self.cones:
            for n in c.nodes:
                self.cone_of[n] = c.id
        #: the nodes' built-in ``uid`` slots (init value :data:`UID`): the
        #: VM numbers them when the activation starts
        self.uid_slots = sorted(k for k, v in init_values.items() if v is UID)
        #: node -> its own uid slot (the innermost node holding the slot);
        #: the VM renumbers it when the node is traversed again
        self.node_uid: Dict[int, int] = {}
        for slot in self.uid_slots:
            owners = [i for i, n in enumerate(self.nodes)
                      if n.base <= slot < n.base + n.size]
            if owners:
                self.node_uid[min(owners, key=lambda i: self.nodes[i].size)] = slot
        self.init_values = {k: v for k, v in init_values.items() if v is not UID}
        self.names = list(names)
        #: M9a3: node -> the inferred node whose subtree holds it (nodes
        #: are in pre-order, so a parent is seen before its children)
        self.inferred: Dict[int, int] = {}
        for nd in tree.nodes:
            if getattr(nd, "inferred_for", None) is not None:
                self.inferred[nd.id] = nd.id
            elif nd.parent in self.inferred:
                self.inferred[nd.id] = self.inferred[nd.parent]

        n = len(self.nodes)
        children: List[List[int]] = [[] for _ in range(n)]
        for i, node in enumerate(self.nodes):
            if node.parent is not None:
                children[node.parent].append(i)
        self.subtree: List[List[int]] = [self._closure(i, children) for i in range(n)]
        self.node_sites: List[List[int]] = [[] for _ in range(n)]
        for sid, (_, target, _) in enumerate(self.sites):
            self.node_sites[target].append(sid)
        # A block's own nested blocks (same node), for its reset.
        sub_scopes: List[List[int]] = [[] for _ in self.scopes]
        for sid, sc in enumerate(tree.scopes):
            p = sc.parent
            while p is not None and tree.scopes[p].node == sc.node:
                sub_scopes[p].append(sid)
                p = tree.scopes[p].parent
        #: scope -> the sites in it or in its nested blocks
        self.scope_sites: List[List[int]] = [
            [st for st, (_, _, ssc) in enumerate(self.sites)
             if ssc == sc or ssc in set(sub_scopes[sc])]
            for sc in range(len(self.scopes))]
        self.scope_nested = sub_scopes
        #: node -> the sites and scopes of its subtree's activities
        self.subtree_sites: List[List[int]] = []
        self.subtree_scopes: List[List[int]] = []
        for i in range(n):
            sub = set(self.subtree[i])
            self.subtree_sites.append(
                [st for st, (owner, _, _) in enumerate(self.sites) if owner in sub])
            self.subtree_scopes.append(
                [sc.id for sc in tree.scopes if sc.node in sub])
        #: cone id -> (base blob, one blob per constraint); filled on first
        #: use, or from the image (ACT) when the constraints are not kept
        self._blobs: Dict[int, Tuple[bytes, List[bytes], set]] = {}
        #: (cone id, constraints in force) -> one blob, when a soft or dist
        #: constraint is in force (see :meth:`blob`)
        self._whole: Dict[tuple, bytes] = {}
        #: cone id -> constraint text, from the image (see :meth:`con_text`)
        self._texts: Dict[int, List[str]] = {}

        # -- flow objects and resources (B5) --
        self.pools = list(tree.pools)
        self.claims = [list(nd.claims) for nd in tree.nodes]
        #: node -> its address claims, (space, addr, size, permanent slot)
        self.addr_claims = [list(getattr(nd, "addr_claims", None) or ())
                            for nd in tree.nodes]
        #: node -> its contiguous address claims, (index in addr_claims,
        #: alignment slot, ((trait field, slot), ...)) (F7.3e)
        self.addr_contig = [list(getattr(nd, "addr_contig", None) or ())
                            for nd in tree.nodes]
        #: space -> its allocatable regions, (lo, size, trait), where the run
        #: places contiguous claims (:meth:`set_regions`)
        self.addr_regions: Dict[str, List[tuple]] = {}
        self.state_writes = [list(nd.state_writes) for nd in tree.nodes]
        self.buffer_writes = [list(nd.buffer_writes) for nd in tree.nodes]
        self.nodes_picks = [list(nd.picks) for nd in tree.nodes]
        #: node -> its ancestors (a claim of one does not exclude its own)
        self.ancestors: List[set] = []
        for i, node in enumerate(self.nodes):
            up, p = set(), node.parent
            while p is not None:
                up.add(p)
                p = self.nodes[p].parent
            self.ancestors.append(up)
        #: scope -> runs its members concurrently (parallel, schedule)
        self.concurrent_scope = [sc.kind in (SC.ScopeKind.PARALLEL, SC.ScopeKind.SCHEDULE)
                                 for sc in tree.scopes]
        #: site -> the scopes it is in, innermost first
        self.site_chain: List[List[int]] = []
        for _, _, scope in self.sites:
            chain = []
            while scope is not None:
                chain.append(scope)
                scope = tree.scopes[scope].parent
            self.site_chain.append(chain)
        #: M9a7: the atomic blocks: nothing is inferred inside one (11.3.7)
        self.atomic = {i for i, sc in enumerate(tree.scopes)
                       if sc.kind == SC.ScopeKind.ATOMIC}
        self._conc: Dict[Tuple[int, int], Optional[int]] = {}
        self.depth = [0] * n
        for i, node in enumerate(self.nodes):
            self.depth[i] = len(self.ancestors[i])
        #: parallel scope -> its branches' footprints to choose on entry (B5e)
        self.par: Dict[int, _Par] = {}
        #: a node may take its values from its cone's last solution (B6e);
        #: off only to compare against solving every traversal
        self.reuse = True
        #: cone id -> the arrays a bulk read of its values fills
        self._readback: Dict[int, tuple] = {}
        #: (cone id, enabled, pinned) -> its dv-solve handle (``handle``)
        self._handles: Dict[tuple, object] = {}
        #: cone ids whose constraints read a list (F6.10d)
        self.array_cones = {c.id for c in self.cones if any(v.array for v in c.vars)}
        #: F8.6e: (pool id, instance) -> the slots of its rand attributes
        self.attr_slots: Dict[tuple, List[int]] = {}
        for c in self.cones:
            for v in c.vars:
                if v.attr_pool is not None:
                    sl = self.attr_slots.setdefault((v.attr_pool, v.attr_inst), [])
                    if v.slot not in sl:
                        sl.append(v.slot)
        kids: Dict[Optional[int], List[int]] = {}
        for sc in tree.scopes:
            kids.setdefault(sc.parent, []).append(sc.id)
        for par in tree.scopes:
            if par.kind != SC.ScopeKind.PARALLEL:
                continue
            branches = []
            keys = [c for c in kids.get(par.id, ()) if tree.scopes[c].node == par.node]
            keys += [("site", st.id) for st in tree.sites if st.scope == par.id]
            for key in keys:
                if isinstance(key, tuple):
                    sites = [key[1]]
                else:
                    inside, todo = set(), [key]
                    while todo:
                        x = todo.pop()
                        inside.add(x)
                        todo.extend(kids.get(x, ()))
                    sites = [st.id for st in tree.sites if st.scope in inside]
                nodes = set()
                for st in sites:
                    nodes.update(self.subtree[self.sites[st][1]])
                lockers = sorted(x for x in nodes if any(c.lock for c in self.claims[x]))
                probes = {}
                for x in sorted(nodes, key=lambda x: self.depth[x]):
                    cid = self.cone_of.get(x)
                    if cid is not None and cid not in probes and any(
                            self.cone_of.get(y) == cid for y in lockers):
                        probes[cid] = x
                branches.append(_Branch(key, lockers, list(probes.values())))
            used: Dict[int, int] = {}
            for br in branches:
                for pid in {pid for x in br.lockers for c in self.claims[x]
                            if c.lock for _, pid in c.pools}:
                    used[pid] = used.get(pid, 0) + 1
            shared = {pid for pid, k in used.items() if k > 1}
            if shared:
                self.par[par.id] = _Par(branches, shared)

    def concurrent(self, a: int, b: int) -> Optional[int]:
        """The parallel (or schedule) scope in which sites *a* and *b* are
        in different branches, so every action reached through one is
        concurrent with every action reached through the other (LRM 11.3.4);
        None if there is none."""
        key = (a, b) if a <= b else (b, a)
        if key not in self._conc:
            ca, cb = self.site_chain[a], self.site_chain[b]
            sb = set(cb)
            out = None
            for i, sc in enumerate(ca):
                if sc in sb:
                    j = cb.index(sc)
                    below_a = ca[i - 1] if i else ("site", a)
                    below_b = cb[j - 1] if j else ("site", b)
                    if self.concurrent_scope[sc] and below_a != below_b:
                        out = sc
                    break
            self._conc[key] = out
        return self._conc[key]

    @staticmethod
    def _closure(i, children) -> List[int]:
        out, todo = [], [i]
        while todo:
            x = todo.pop()
            out.append(x)
            todo.extend(children[x])
        return sorted(out)

    def own_fields(self, node: int) -> List[Tuple[int, str]]:
        """*node*'s own named slots, not its children's, as (slot, name
        without the node's path): what a plan records as its values."""
        cache = self.__dict__.setdefault("_own", {})     # a table from an image too
        own = cache.get(node)
        if own is None:
            n = self.nodes[node]
            kids = [(c.base, c.base + c.size) for c in self.nodes if c.parent == node]
            pre = n.path + "." if n.path else ""
            own = [(s, self.names[s][len(pre):] if self.names[s].startswith(pre)
                    else self.names[s])
                   for s in range(n.base, min(n.base + n.size, len(self.names)))
                   if self.names[s] and not any(lo <= s < hi for lo, hi in kids)]
            cache[node] = own
        return own

    def site(self, node: int, local: int) -> int:
        return self.nodes[node].site_base + local

    def values(self, ctx, cone: _Cone) -> list:
        """Every variable of *cone* in the last solution of *ctx*, in one
        read."""
        arrs = self._readback.get(cone.id)
        if arrs is None:
            import ctypes
            n = len(cone.vars)
            ids = (ctypes.c_uint32 * n)(*range(n))
            arrs = self._readback[cone.id] = (ids, (ctypes.c_int64 * n)(), n)
        ids, out, n = arrs
        ctx.get_values(ids, out, n)
        return list(out)

    # -- the problem of one set of constraints in force ----------------------

    def blob(self, cone: _Cone, enabled: Tuple[int, ...]) -> Tuple[bytes, ...]:
        """The problem of *cone* with the constraints *enabled* in force:
        the cone's base blob (its variables) and the blob of each enabled
        constraint, which ``SolveCache`` adds to the base (dv-solve
        ``add_constraint``). One base and one blob per constraint per cone,
        so the image can carry them and any engine compose the subset.

        Except a soft or dist constraint: dv-solve ``add_constraint`` keeps
        only the hard constraints of an added blob, so a set in force with
        one of them is one blob of the whole problem (F3.2b; not from an
        image, which keeps no constraints)."""
        parts = self._blobs.get(cone.id)
        if parts is None:
            from ..lower.constraints import build_solve_blob, solve_blob_vars
            # A constraint with selectors (an `||` over `&&`s, as an
            # executor claim's match) declares them past the base's
            # variables, so two such would number theirs alike (-4). The
            # base declares a spare 1-bit variable per selector of every
            # constraint, and each constraint's blob the spares of those
            # before it: its selectors land on spares of its own, which
            # any engine adds in any order (F8.6c2).
            n = solve_blob_vars(self._problem(cone, ()))
            k = [solve_blob_vars(self._problem(cone, (i,))) - n
                 for i in range(len(cone.constraints))]
            base, _ = build_solve_blob(self._problem(cone, (), spare=sum(k)))
            parts = self._blobs[cone.id] = (base, [
                build_solve_blob(self._problem(cone, (i,), spare=sum(k[:i])))[0]
                for i in range(len(cone.constraints))], {
                i for i in range(len(cone.constraints)) if k[i]})
        if cone.constraints and any(
                isinstance(cone.constraints[i].constraint,
                           (C.ConstraintSoft, C.ConstraintDist))
                for i in enabled):
            whole = self._whole.get((cone.id, enabled))
            if whole is None:
                from ..lower.constraints import build_solve_blob
                whole = self._whole[cone.id, enabled] = build_solve_blob(
                    self._problem(cone, enabled))[0]
            return (whole,)
        base, cons, _ = parts
        return (base,) + tuple(cons[i] for i in enabled)

    def bound_blob(self, cone: _Cone, bounds: Dict[str, tuple]) -> Optional[bytes]:
        """A blob that keeps each rand variable of *cone* named in *bounds*
        within its ``(lo, hi)`` (either None: unbounded), or None if *bounds*
        names none of them. (A property test's shrink: P5.5.)"""
        cons = []
        for v in cone.vars:
            b = (bounds.get(v.name, bounds.get(v.slot))  # name, or slot
                 if v.rand else None)
            if b is None:
                continue
            ref = E.ExprRefField(base=E.TypeExprRefSelf(), index=v.slot)
            for op, k in ((E.BinOp.GtE, b[0]), (E.BinOp.LtE, b[1])):
                if k is not None:
                    cons.append(C.ConstraintExpr(expr=E.ExprBin(
                        lhs=ref, op=op, rhs=E.ExprConstant(value=k))))
        if not cons:
            return None
        from ..lower.constraints import build_solve_blob
        p = self._problem(cone, ())
        return build_solve_blob(dc.replace(p, constraints=cons))[0]

    def set_regions(self, memmap: dict) -> None:
        """Take the allocatable regions of each space from *memmap* (pssc
        ``_memmap``): a transparent region from its ``addr``, a contiguous
        one from its opaque ``base`` (F7.3e)."""
        spaces = memmap.get("spaces") or []
        self.addr_regions = {}
        for r in memmap.get("regions") or ():
            lo = r.get("addr") if r.get("addr") is not None else r.get("base")
            if not r.get("allocatable") or lo is None:
                continue
            self.addr_regions.setdefault(spaces[r["space"]]["path"], []).append(
                (lo, r["size"], dict(r.get("trait") or {})))

    def apart_blob(self, cone: _Cone, apart) -> bytes:
        """A blob that keeps each address claim in *apart*, ``(addr slot,
        size slot, lo, hi)``, out of the live interval ``[lo, hi)``: ``addr
        >= hi || lo >= addr + size``. (No ``&&`` inside the ``||``: that
        needs selector variables, which an added blob cannot declare.)"""
        def ref(slot):
            return E.ExprRefField(base=E.TypeExprRefSelf(), index=slot)
        cons = []
        for a, sz, lo, hi in apart:
            cons.append(C.ConstraintExpr(expr=E.ExprBin(
                lhs=E.ExprBin(lhs=ref(a), op=E.BinOp.GtE,
                              rhs=E.ExprConstant(value=hi)),
                op=E.BinOp.Or,
                rhs=E.ExprBin(lhs=E.ExprConstant(value=lo), op=E.BinOp.GtE,
                              rhs=E.ExprBin(lhs=ref(a), op=E.BinOp.Add,
                                            rhs=ref(sz))))))
        from ..lower.constraints import build_solve_blob
        p = self._problem(cone, ())
        return build_solve_blob(dc.replace(p, constraints=cons))[0]

    def cone_image(self, cone_id: int) -> Tuple[bytes, List[bytes], List[str]]:
        """Cone *cone_id*'s base blob, blob per constraint and constraint
        text, as the image (ACT) stores them."""
        cone = next(c for c in self.cones if c.id == cone_id)
        self.blob(cone, ())
        base, cons, _ = self._blobs[cone_id]
        return base, list(cons), [self.con_text(cone, i) for i in range(len(cone.constraints))]

    def load_cone(self, cone_id: int, base: bytes, cons: List[bytes], texts: List[str]) -> None:
        """Set a cone's blobs and constraint text read from the image."""
        # (no selector sets: an image keeps no constraints to rebuild one blob)
        self._blobs[cone_id] = (base, list(cons), set())
        self._texts[cone_id] = list(texts)

    def con_text(self, cone: _Cone, i: int) -> str:
        """Constraint *i* of *cone* as written, for an unsat message."""
        texts = self._texts.get(cone.id)
        if texts is not None:
            return texts[i]
        return describe(cone.constraints[i].constraint, self.names)

    def handle(self, cone: _Cone, enabled, pinned: frozenset):
        """dv-solve's handle of *cone*'s problem with the constraints
        *enabled* in force and the variables *pinned* (cone.vars indices)
        given as state: a cone that reads a list (F6.10d), whose elements
        are passed per solve. Its var ids skip the list variables."""
        key = (cone.id, enabled, pinned)
        h = self._handles.get(key)
        if h is None:
            if any(c.constraint is None for c in cone.constraints):
                # Read from an image, which keeps per-constraint blobs only;
                # the handle needs the whole problem (F6.10f4b).
                raise VMError("an activity constraint reads or solves a list: "
                              "the .zbc image cannot carry that yet; run the "
                              "model in memory (the Python engine)")
            from dv_solve.handle import SolveHandle
            from ..lower.constraints import build_solve_blob
            buf, _ = build_solve_blob(self._problem(cone, enabled, pinned))
            h = self._handles[key] = SolveHandle(
                buf, [(v.width, v.signed) for v in cone.vars if not v.array],
                [(v.width, v.signed) for v in cone.vars if v.array])
        return h

    def _problem(self, cone: _Cone, enabled, pinned=frozenset(),
                 spare: int = 0) -> SC.ScSolveProblem:
        # A list a constraint reads is an input array (F6.10d), so the
        # cone's scalars are solved by dv-solve's handle, the inputs (and
        # the variables *pinned*) as state. *spare*: 1-bit variables past
        # the cone's, as a selector is declared (see ``blob``); their slots
        # are no field's.
        arrays = [v for v in cone.vars if v.array]
        vs = [SC.ScSolveVar(name=v.name, var_id=i, slot=v.slot,
                            width=v.width, signed=v.signed,
                            input=bool(arrays) and (not v.rand or k in pinned))
              for i, (k, v) in enumerate((k, v) for k, v in enumerate(cone.vars)
                                         if not v.array)]
        vs += [SC.ScSolveVar(name=f"$sel{j}", var_id=len(vs) + j, slot=(1 << 30) + j,
                             width=1, signed=False) for j in range(spare)]
        return SC.ScSolveProblem(
            vars=vs,
            input_arrays=[SC.ScSolveArray(name=v.name, array_id=i, slot=v.slot,
                                          width=v.width, signed=v.signed, rand=v.rand)
                          for i, v in enumerate(arrays)],
            constraints=[cone.constraints[i].constraint for i in enabled])

    def check(self) -> None:
        """Build each cone with every constraint enabled, so a constraint the
        solver cannot take is a lowering error, not a run-time one."""
        for c in self.cones:
            self.blob(c, tuple(range(len(c.constraints))))


class Activation:
    """The state of one run of an :class:`ActivationTable`."""

    def __init__(self, table: ActivationTable, obj, cache=None, cobj=None,
                 bounds: Optional[Dict[str, tuple]] = None, heap=None,
                 plan: Optional[PlanLog] = None):
        """*cache*: the run's :class:`~.solve_cache.SolveCache` (a private
        one when None). *cobj*: the component object, which a constraint
        reading a component attribute is pinned from. *bounds*: rand
        variable name (as the cone names it), or object slot, -> ``(lo, hi)``, held as extra
        constraints in every solve (a property test's shrink). *plan*: the
        :class:`PlanLog` the run's traversals and values are recorded in."""
        self.t = table
        self.plan = plan
        #: node -> its current traversal's occurrence in :attr:`plan`
        self.occ: List[Optional[int]] = [None] * len(table.nodes)
        #: state pool -> its current object's flow id in :attr:`plan`
        self.state_cur: Dict[int, int] = {}
        #: flow bind index -> the flow id one side of it opened, until the
        #: other side commits
        self.bind_open: Dict[int, int] = {}
        #: loop scope -> its open repeat record in :attr:`plan`
        self.loop_rec: Dict[int, dict] = {}
        self.obj = obj
        self.cobj = cobj
        self.bounds = bounds or {}
        #: the run's heap, which a list a constraint reads is in (F6.10d)
        self.heap = heap
        self._bound_blobs: Dict[int, Optional[bytes]] = {}
        #: rand variable name -> the value its node's solve chose
        self.solved: Dict[str, int] = {}
        if cache is None:
            from .solve_cache import SolveCache
            cache = SolveCache()
        self.cache = cache
        n = len(table.nodes)
        self.committed = [False] * n
        #: M9a3: the inferred nodes bc runs now; the others' constraints
        #: are off
        self.infer_on: set = set()
        #: consumer -> the inferred nodes run for its pending solve, in order
        self.infer_tried: Dict[int, List[int]] = {}
        #: M9b1c: inferred node -> the component instance it runs in, when
        #: the consumer's solve chooses one among several
        self.infer_comp: Dict[int, int] = {}
        #: M9a6b: producer -> the inferred stream partners its solve picked,
        #: started with it once it commits
        self.partners: Dict[int, List[int]] = {}
        #: M9a9d: node -> the (pool, writes) state references of its
        #: traversal in progress; frames waiting for one to end (13.3.1)
        self.state_live: Dict[int, List[tuple]] = {}
        self.state_waiters: List = []
        #: waiters an ended traversal releases; the VM readies them
        self.waking: List = []
        self.started = [False] * n
        #: F2.6: node -> how many traversals of it have started (never
        #: reset; a stimulus id's ``#n``)
        self.visits = [0] * n
        #: the root's stimulus id, which its nodes' ids start with
        self.stim = ""
        self.by_site: List[Optional[int]] = [None] * n
        self.fired = [False] * len(table.sites)
        self.entered = [False] * len(table.scopes)
        # -- the claim table (B5, D-B12) --
        #: an event counter: when a scope was entered, when a claim was made
        self.clock = 0
        self.entered_at = [0] * len(table.scopes)
        #: pool id -> the claims held now: [node, site, instance_id, lock, at]
        self.holds: Dict[int, List[list]] = {}
        #: node -> its held claims, as (pool id, claim)
        self.held_by: Dict[int, List[tuple]] = {}
        #: node -> its live address claims, as (space, ledger entry)
        self.addr_held_by: Dict[int, List[tuple]] = {}
        #: pool id -> released claims, (node, site, instance_id, lock) -> the
        #: latest made. A released claim matters only to a claim concurrent
        #: with it, and only if made since their parallel was entered, so the
        #: latest of each is all that is kept: the table stays the size of
        #: the tree, however long the run.
        self.released: Dict[int, Dict[tuple, int]] = {}
        #: F7.3d: space -> the address claims live now: [node, site, addr,
        #: size, permanent, at]. Released when the node completes (LRM
        #: Table 29), unless permanent (the end of the test, 24.8.5)
        self.addr_live: Dict[str, List[list]] = {}
        #: space -> released address claims, (node, site, addr, size) -> the
        #: latest made (as :attr:`released`)
        self.addr_released: Dict[str, Dict[tuple, int]] = {}
        #: F7.7a: the latest released address claims, newest last, as
        #: (space, addr, size, stim, type): who owned a region a late write
        #: hits. Bounded: only for attribution (:meth:`owners`)
        self.addr_history: Deque[tuple] = deque(maxlen=64)
        #: buffer pool id -> times picked -> the objects completed actions
        #: output to it that were picked that many times: [values, times
        #: picked, index in its list]
        self.outputs: Dict[int, Dict[int, List[list]]] = {}
        #: parallel scope -> (entered at, per branch: pool id -> footprint)
        self.footprint: Dict[int, Tuple[int, List[Dict[int, int]]]] = {}
        #: M9a8c: select branch scope -> its parallel's footprint epoch when chosen
        self.chosen: Dict[int, int] = {}
        #: cone id -> its last solution (B6e)
        self.witness: Dict[int, _Witness] = {}
        #: the constraints in force at the last solve (``explain``)
        self.last_enabled: Tuple[int, ...] = ()
        #: F8.6e: the slots of resource attributes that have a value: every
        #: solve of a cone reading one writes and holds it; entering a
        #: committing scope (a branch chosen) frees those no claim has fixed
        #: (``attr_claimed``), so a branch's own constraints may choose them
        self.attr_held: set = set()
        self.attr_claimed: set = set()

    # -- events --------------------------------------------------------------

    def _reset_node(self, node: int) -> None:
        """*node*'s handles become uninitialized: it and its subtree hold no
        values, none of their traversals has run, no branch was entered."""
        t = self.t
        for d in t.subtree[node]:
            self.committed[d] = False
            self.started[d] = False
            self.by_site[d] = None
        for st in t.subtree_sites[node]:
            self.fired[st] = False
        for sc in t.subtree_scopes[node]:
            self.entered[sc] = False

    def enter_node(self, node: int, site: Optional[int]) -> None:
        """A traversal of *node* starts (through *site*; None: the root)."""
        self._reset_node(node)
        if node == 0:
            self.attr_held.clear()
            self.attr_claimed.clear()
        self.started[node] = True
        if site is not None:
            self.fired[site] = True
        if self.plan is not None:
            n = self.t.nodes[node]
            stim = (f"{self.stim}.{n.path}#{self.visits[node]}"
                    if node else self.stim)
            self.occ[node] = self.plan.start(site, n.path, stim, n.type_qname)

    def enter_scope(self, scope: int, seed=None) -> None:
        """Entry to an activity block: its traversed handles are reset. A
        ``parallel`` whose branches lock the same pools chooses each
        branch's footprint (B5e); *seed* is drawn for it only then (a
        callable returning one)."""
        t = self.t
        for st in t.scope_sites[scope]:
            self.fired[st] = False
            self._reset_node(t.sites[st][1])
        for sc in t.scope_nested[scope]:
            self.entered[sc] = False
        self.entered[scope] = True
        if self.plan is not None:
            self._plan_scope(scope)
        self.clock += 1
        self.entered_at[scope] = self.clock
        if t.scopes[scope][0] and self.attr_held:
            self.attr_held &= self.attr_claimed
        if scope in t.par:
            self._footprints(scope, seed() if seed is not None else 0)

    def _plan_scope(self, scope: int) -> None:
        """An if branch taken, or one more iteration of a loop (M8b6b)."""
        t = self.t
        for sc in t.scope_nested[scope]:
            self.loop_rec.pop(sc, None)
        sc = t.tree.scopes[scope]
        occ = self.occ[sc.node]
        K = SC.ScopeKind
        if sc.kind in (K.IF_THEN, K.IF_ELSE):
            self.plan.control.append({"occ": occ, "kind": "if", "scope": scope,
                                      "value": sc.kind == K.IF_THEN})
        elif sc.kind in (K.LOOP_BODY, K.LOOP_BODY_CERTAIN):
            r = self.loop_rec.get(scope)
            if r is None or r["occ"] != occ:
                r = self.loop_rec[scope] = {"occ": occ, "kind": "repeat",
                                            "scope": scope, "value": 0}
                self.plan.control.append(r)
            r["value"] += 1

    # -- footprints (B5e) ------------------------------------------------------

    def _footprints(self, scope: int, seed: int) -> None:
        """Choose each branch's footprint: per pool the branches share, the
        instances its locks may take, disjoint from every other branch's
        (LRM 11.3.4 a: every lock in one branch is concurrent with every lock
        in another).

        Each branch's footprint holds what one traversal of it needs: a
        probe solves the cone of its first locking node as if it were being
        traversed, with the instances earlier branches reserved excluded --
        one iteration's projection, which is every iteration's, since a site
        in a loop is one node. The instances no probe took are then dealt
        out at random among the branches that use their pool (M9a7b: also
        to one whose only locker of it is an inferred producer, which is not
        probed: it runs only if picked, and a lock its branch cannot get
        keeps it from being picked). A branch whose
        probe finds nothing locks more than is left: an error, not a wait
        (LRM 11.3.4, the resource rules for concurrent actions)."""
        import random
        t = self.t
        par = t.par[scope]
        rng = random.Random(seed)
        reserved: Dict[int, int] = {}
        fps: List[Dict[int, int]] = []
        branch_of = {x: bi for bi, br in enumerate(par.branches) for x in br.lockers}
        for bi, br in enumerate(par.branches):
            mine: Dict[int, int] = {}
            for member in br.probes:
                if self._optional_locker(member, scope):
                    continue     # M9a7b/M9a8c: run only if picked; reserves nothing
                vals = self._probe(member, bi, branch_of, fps, reserved, par.shared,
                                   seed + bi)
                if vals is None:
                    n = t.nodes[member]
                    pools = ", ".join(sorted("%s@%d" % (t.pools[p].name, t.pools[p].inst)
                                             for p in par.shared))
                    raise ScopeUnsatError(
                        "branch %d of a parallel cannot lock: %r (%s) has no "
                        "instance of %s left that the other branches do not lock "
                        "(LRM 11.3.4: concurrent claims are not serialized)"
                        % (bi, n.path or "<root>", n.type_qname, pools))
                cone = t.cones[t.cone_of[member]]
                for x in br.lockers:
                    if t.cone_of.get(x) != cone.id:
                        continue
                    inst = self._instance_of(x, vals)
                    for c in t.claims[x]:
                        pid = _pool_for(c.pools, inst)
                        if c.lock and pid in par.shared and c.iid_slot in vals:
                            bit = 1 << vals[c.iid_slot]
                            mine[pid] = mine.get(pid, 0) | bit
                            reserved[pid] = reserved.get(pid, 0) | bit
            fps.append(mine)
        for pid in sorted(par.shared):
            users = [i for i, br in enumerate(par.branches)
                     if pid in fps[i] or any(c.lock and pid in {q for _, q in c.pools}
                                             for x in br.lockers
                                             if self._optional_locker(x, scope)
                                             for c in t.claims[x])]
            if not users:
                continue
            free = ((1 << t.pools[pid].size) - 1) & ~reserved.get(pid, 0)
            for i in range(t.pools[pid].size):
                if free >> i & 1:
                    u = rng.choice(users)
                    fps[u][pid] = fps[u].get(pid, 0) | 1 << i
        self.footprint[scope] = (self.entered_at[scope], fps, reserved)

    def _optional_locker(self, x: int, scope: int) -> bool:
        """*x* may not run in a traversal of parallel *scope*: an inferred
        producer (M9a7b), or a node under a select inside the parallel
        (M9a8c: its branch may never be chosen). Not probed; it locks from
        the instances dealt out at random."""
        t = self.t
        if t.tree.nodes[x].inferred_for is not None:
            return True
        for st in t.node_sites[x]:
            for sc in t.site_chain[st]:
                if sc == scope:
                    break
                if t.tree.scopes[sc].kind == SC.ScopeKind.SELECT_BRANCH:
                    return True
        return False

    def _probe(self, member: int, bi: int, branch_of, fps, reserved, shared,
               seed) -> Optional[Dict[int, int]]:
        """Solve *member*'s cone as branch *bi*'s traversal of it would,
        committing nothing, and return the values by slot (None if there are
        none). A lock of a branch probed already (*fps*) takes an instance it
        reserved; any other lock, none that the probed branches reserved
        (*reserved*)."""
        t = self.t
        cone = t.cones[t.cone_of[member]]
        # The traversal the branch will make: the blocks on its way there
        # (a loop body, entered only when the loop runs) are as if entered.
        site = t.node_sites[member][0]
        saved = [(sc, self.entered[sc]) for sc in t.site_chain[site]]
        for sc, _ in saved:
            self.entered[sc] = True
        try:
            enabled = tuple(i for i, c in enumerate(cone.constraints)
                            if self._in_force(c, member, site))
        finally:
            for sc, was in saved:
                self.entered[sc] = was
        pins = []
        for i, v in enumerate(cone.vars):
            if v.busy_pool is not None:
                if not self.committed[v.node]:
                    pid, j = v.busy_pool, branch_of.get(v.node)
                    if pid not in shared:
                        mask = 0
                    elif j is not None and j < bi:
                        mask = ((1 << t.pools[pid].size) - 1) & ~fps[j].get(pid, 0)
                    else:
                        mask = reserved.get(pid, 0)
                    pins.append((i, v, _busy_word(v, mask | self._busy(pid, v.node, v.busy_lock))))
                continue
            pin = self._pin(v, member, None, probe=True)
            if pin is not None:
                pins.append((i, v, pin))
        # More locks of one pool than it has instances left: no solution, and
        # a pigeonhole the solver may take its whole budget to refute
        need: Dict[int, int] = {}
        for c in t.claims[member]:
            pids = {pid for _, pid in c.pools}
            if c.lock and len(pids) == 1 and pids <= set(shared):
                pid = next(iter(pids))
                need[pid] = need.get(pid, 0) + 1
        for pid, n in need.items():
            if n > t.pools[pid].size - bin(reserved.get(pid, 0)).count("1"):
                return None
        vals = self._attempt(member, cone, t.blob(cone, enabled), pins, seed & _MASK64,
                             write=False)
        return vals

    def _instance_of(self, node: int, vals: Dict[int, int]) -> int:
        """The component instance *node* runs in, given solved *vals* for
        the slots that choose one."""
        off, x = 0, node
        while x is not None:
            nd = self.t.nodes[x]
            if nd.comp_slot is not None:
                v = vals.get(nd.comp_slot)
                return (v if v is not None else self.obj.get_field(nd.comp_slot)) + off
            off += nd.comp_rel or 0
            x = nd.parent
        return off

    def _fp_mask(self, pid: int, node: int, site: Optional[int],
                 lock: bool = True) -> int:
        """The instances of pool *pid* outside the footprint of the branch
        *node* (reached through *site*) is in, for each parallel it is in.
        A share (*lock* False) is kept only from the instances other
        branches' probes reserved (M9a8c): one dealt out at random may never
        be locked (its locker optional), and a locker that finds it shared
        is not chosen (``select_blocked``)."""
        t = self.t
        if site is None:
            return 0
        mask = 0
        chain = t.site_chain[site]
        for i, sc in enumerate(chain):
            fp = self.footprint.get(sc)
            if fp is None or fp[0] != self.entered_at[sc] or pid not in t.par[sc].shared:
                continue
            below = chain[i - 1] if i else ("site", site)
            bi = next((j for j, br in enumerate(t.par[sc].branches) if br.key == below), None)
            if bi is not None:
                full = (1 << t.pools[pid].size) - 1
                if not lock:
                    full &= fp[2].get(pid, 0) | self._chosen_fp(sc, bi, pid, fp[1])
                mask |= full & ~fp[1][bi].get(pid, 0)
        return mask

    def _on_its_way(self, x: int, sc: int) -> bool:
        """Locker *x* under a select in parallel *sc* is not yet solved and
        every select branch on its way down from *sc* is entered (chosen)."""
        t = self.t
        if self.committed[x] or t.tree.nodes[x].inferred_for is not None:
            return False
        for st in t.node_sites[x]:
            sel = []
            for s2 in t.site_chain[st]:
                if s2 == sc:
                    break
                if t.tree.scopes[s2].kind == SC.ScopeKind.SELECT_BRANCH:
                    sel.append(s2)
            if sel and all(self._chosen(s2, sc) for s2 in sel):
                return True
        return False

    def _chosen_fp(self, sc: int, bi: int, pid: int, fps) -> int:
        """M9a8c: the instances of pool *pid* dealt to the other branches of
        parallel *sc* whose optional locker of it is now on its way (its
        select branch chosen, its traversal not yet made)."""
        t = self.t
        mask = 0
        for j, br in enumerate(t.par[sc].branches):
            if j == bi or not fps[j].get(pid):
                continue
            if any(c.lock and pid in {q for _, q in c.pools}
                   and self._on_its_way(x, sc)
                   for x in br.lockers for c in t.claims[x]):
                mask |= fps[j][pid]
        return mask

    def exit_node(self, node: int, comp: Optional[int], completed: bool = True) -> None:
        """A traversal of *node* ends (*completed*: it ran to its end, rather
        than being cancelled). Its claims are released; a completed node's
        state outputs become their pools' current objects (12.5)."""
        t = self.t
        if self.plan is not None and self.occ[node] is not None:
            self.plan.end(self.occ[node], comp, completed)
        for pid, h in self.held_by.pop(node, ()):
            self.holds[pid].remove(h)
            key = (h[0], h[1], h[2], h[3])
            rel = self.released.setdefault(pid, {})
            if rel.get(key, -1) < h[4]:
                rel[key] = h[4]
        for sp, h in self.addr_held_by.pop(node, ()):
            if h[4]:
                continue                     # permanent: never released
            self.addr_live[sp].remove(h)
            self.addr_history.append((sp, h[2], h[3], h[6], h[7]))
            key = (h[0], h[1], h[2], h[3])
            rel = self.addr_released.setdefault(sp, {})
            if rel.get(key, -1) < h[5]:
                rel[key] = h[5]
        if self.state_live.pop(node, None) is not None and self.state_waiters:
            self.waking += self.state_waiters
            self.state_waiters = []
        if not completed:
            return
        if not faults.on("skip-buffer-wait"):
            self.publish_buffers(node, comp)
        if not faults.on("race-state"):
            self.publish_states(node, comp)

    def _state_refs(self, node: int, comp) -> List[tuple]:
        """*node*'s state references as ``(pool, writes)``: its state
        outputs (each pool a candidate when *comp* is not yet known) and
        the state pools its cone reads."""
        t = self.t
        refs = []
        for w in t.state_writes[node]:
            pids = [_pool_for(w.pools, comp)] if comp is not None or len(w.pools) == 1 \
                else [pid for _, pid in w.pools]
            refs += [(pid, True) for pid in pids]
        ci = t.cone_of.get(node)
        if ci is not None:
            state = self.__dict__.get("_state_slot")
            if state is None:
                state = self._state_slot = {s: p.id for p in t.pools
                                            if p.kind == "state" for s in p.slots}
            written = {pid for pid, _ in refs}
            for v in t.cones[ci].vars:
                pid = state.get(v.live_from) if v.node == node else None
                if pid is not None and pid not in written:
                    written.add(pid)
                    refs.append((pid, False))
        return refs

    def state_wait(self, node: int, comp, frame) -> bool:
        """M9a9d: *node* may not start yet: a traversal in progress outside
        its ancestors writes a state pool it references, or references a
        pool it writes (LRM 13.3.1). *frame* then waits for one to end."""
        if faults.on("race-state") or faults.on("skip-state-wait") or not self.state_live:
            return False
        refs = self._state_refs(node, comp)
        if not refs:
            return False
        anc = self.t.ancestors[node]
        for n, live in self.state_live.items():
            if n == node or n in anc:
                continue
            for pid, w in refs:
                if any(q == pid and (w or lw) for q, lw in live):
                    self.state_waiters.append(frame)
                    return True
        return False

    def state_enter(self, node: int, comp) -> None:
        """*node*'s traversal, just solved, holds its state pools until it
        ends."""
        refs = self._state_refs(node, comp)
        if refs:
            self.state_live[node] = refs

    def publish_states(self, node: int, comp: Optional[int]) -> None:
        """*node*'s state outputs become their pools' current objects (at
        its end; at its solve under fault ``race-state``)."""
        t = self.t
        plan = self.plan if self.occ[node] is not None else None
        for w in t.state_writes[node]:
            pid = _pool_for(w.pools, comp)
            dst = t.pools[pid].slots
            vals = [self.obj.get_field(src) for src in w.src]
            rec = plan.flow_out("state", self._pool_name(pid), self.occ[node],
                                self._ref_fields(node, w.ref)) if plan is not None else None
            if faults.on("stale-prev"):
                # one write behind: this write lands when the next one does
                held = self.__dict__.setdefault("_stale", {})
                prior = held.get(pid)
                held[pid] = (vals, rec)
                if prior is None:
                    continue
                vals, rec = prior
            for v, d in zip(vals, dst):
                self.obj.set_field(d, v)
            if rec is not None:
                self.state_cur[pid] = rec

    def publish_buffers(self, node: int, comp: Optional[int]) -> None:
        """*node*'s buffer outputs join their pools' completed objects (at
        its end; at its solve under fault ``skip-buffer-wait``)."""
        t = self.t
        plan = self.plan if self.occ[node] is not None else None
        for w in t.buffer_writes[node]:
            pid = _pool_for(w.pools, comp)
            fresh = self.outputs.setdefault(pid, {}).setdefault(0, [])
            fresh.append([tuple(self.obj.get_field(s) for s in w.src), 0, len(fresh)])
            if plan is not None:
                fresh[-1].append(plan.flow_out(
                    "buffer", self._pool_name(pid), self.occ[node],
                    self._ref_fields(node, w.ref)))

    def _pool_name(self, pid: int) -> str:
        return "%s@%d" % (self.t.pools[pid].name, self.t.pools[pid].inst)

    def _ref_fields(self, node: int, ref: str) -> dict:
        """The current values of *node*'s flow-object reference *ref*'s
        leaves, by leaf name (for the plan)."""
        pre = ref + "."
        return {name[len(pre):]: self.obj.get_field(s)
                for s, name in self.t.own_fields(node) if name.startswith(pre)}

    def _plan_flow_in(self, node: int) -> None:
        """Record *node*'s flow inputs in the plan as its solve commits: the
        state objects it reads and the bound buffer/stream objects it shares
        (picked buffers are recorded where they are picked)."""
        t, plan, occ = self.t, self.plan, self.occ[node]
        ci = t.cone_of.get(node)
        if ci is not None:
            state = self.__dict__.get("_state_slot")
            if state is None:
                state = self._state_slot = {s: p.id for p in t.pools
                                            if p.kind == "state" for s in p.slots}
            seen = set()
            for v in t.cones[ci].vars:
                pid = state.get(v.live_from) if v.node == node else None
                if pid is None or pid in seen:
                    continue
                seen.add(pid)
                if pid not in self.state_cur:
                    p = t.pools[pid]
                    self.state_cur[pid] = plan.flow_out(
                        "state", self._pool_name(pid), None,
                        {leaf: self.obj.get_field(s) for leaf, s in zip(p.leaves, p.slots)})
                plan.consume(self.state_cur[pid], occ)
        for i, (onode, oref, inode, iref, kind) in enumerate(
                getattr(t.tree, "flow_binds", ())):
            if node not in (onode, inode):
                continue
            fid = self.bind_open.get(i)
            out = node == onode
            if fid is None or (plan.flow[fid]["producer"] is not None) == out:
                fid = plan.flow_out(kind, None, None,
                                    self._ref_fields(node, oref if out else iref))
                self.bind_open[i] = fid
            else:
                del self.bind_open[i]
            if out:
                plan.flow[fid]["producer"] = occ
            else:
                plan.consume(fid, occ)

    def _busy(self, pid: int, node: int, lock: bool) -> int:
        """The instances of pool *pid* a claim of *node* may not take: those a
        conflicting claim holds now (other than its own ancestors'), and those
        a conflicting claim concurrent with it took (LRM 9.4, 11.3.4)."""
        if faults.on("ignore-lock"):
            return 0
        t = self.t
        site = self.by_site[node]
        mask = 0
        for hnode, hsite, iid, hlock, at in self.holds.get(pid, ()):
            if not (lock or hlock) or hnode == node:
                continue
            if hnode not in t.ancestors[node]:
                mask |= 1 << iid
            elif self._concurrent_since(site, hsite, at):
                mask |= 1 << iid
        for (hnode, hsite, iid, hlock), at in self.released.get(pid, {}).items():
            if (lock or hlock) and hnode != node and self._concurrent_since(site, hsite, at):
                mask |= 1 << iid
        if t.par:
            mask |= self._fp_mask(pid, node, site if site is not None
                                  else self._pending_site(node), lock)
        return mask

    def _concurrent_since(self, site, hsite, at: int) -> bool:
        """A claim made at *at* through *hsite* is concurrent with one
        through *site*: they are in different branches of a parallel entered
        no later than the claim was made."""
        if site is None or hsite is None:
            return False
        sc = self.t.concurrent(site, hsite)
        return sc is not None and at >= self.entered_at[sc]

    def _claim(self, node: int, comp: Optional[int]) -> None:
        """Record *node*'s claims, now that its solve chose them."""
        self.clock += 1
        for c in self.t.claims[node]:
            pid = _pool_for(c.pools, comp)
            iid = self.obj.get_field(c.iid_slot)
            h = [node, self.by_site[node], iid, c.lock, self.clock]
            self.holds.setdefault(pid, []).append(h)
            if self.plan is not None and self.occ[node] is not None:
                self.plan.claim(self._pool_name(pid), iid, self.occ[node], c.lock)
            self.held_by.setdefault(node, []).append((pid, h))
            # The instance's attributes, as this claim's solve chose them,
            # are fixed from now on
            self.attr_claimed.update(self.t.attr_slots.get((pid, iid), ()))

    def _addr_claim(self, node: int, stim: str = "") -> None:
        """Record *node*'s address claims in the ledger, now that its solve
        placed them (F7.3d), with *stim*, the stimulus id of the action
        that owns them (F7.7a)."""
        self.clock += 1
        g = self.obj.get_field
        claims = self.t.addr_claims[node]
        # transparent claims first: the solve placed them; then contiguous
        # ones, first-fit around everything live (F7.3e)
        order = [i for i, c in enumerate(claims) if c[1] is not None]
        contig = {i: (al, tr, ad) for i, al, tr, ad in self.t.addr_contig[node]}
        for i in order + [i for i, c in enumerate(claims) if c[1] is None]:
            sp, a, sz, perm = claims[i]
            if a is None:
                al, tr, ad = contig.get(i, (None, (), None))
                addr = self._addr_place(node, sp, g(sz),
                                        g(al) if al is not None else 1,
                                        {k: g(s) for k, s in tr})
                if ad is not None:
                    # where make_handle_from_claim finds it (F7.4b)
                    self.obj.set_field(ad, addr)
            else:
                addr = g(a)
            h = [node, self.by_site[node], addr, g(sz),
                 bool(g(perm)) if perm is not None else False, self.clock,
                 stim, self.t.nodes[node].type_qname]
            self.addr_live.setdefault(sp, []).append(h)
            self.addr_held_by.setdefault(node, []).append((sp, h))

    def owners(self, addr: int, size: int = 1) -> List[dict]:
        """F7.7a: the address claims whose region overlaps ``[addr,
        addr+size)``: those live now (``live``), then released ones, newest
        first. Each ``{stim, type, space, addr, size, live}``."""
        out = []
        for sp, hs in self.addr_live.items():
            for h in hs:
                if h[2] < addr + size and addr < h[2] + h[3]:
                    out.append({"stim": h[6], "type": h[7], "space": sp,
                                "addr": h[2], "size": h[3], "live": True})
        for sp, a, sz, stim, ty in reversed(self.addr_history):
            if a < addr + size and addr < a + sz:
                out.append({"stim": stim, "type": ty, "space": sp,
                            "addr": a, "size": sz, "live": False})
        return out

    def _addr_busy(self, node: int, sp: str) -> List[tuple]:
        """The intervals ``[lo, hi)`` of space *sp* a claim of *node* must
        avoid (as :meth:`_apart`)."""
        site = self.by_site[node]
        out = [(h[2], h[2] + h[3]) for h in self.addr_live.get(sp, ())]
        for (hnode, hsite, addr, size), at in self.addr_released.get(sp, {}).items():
            if hnode != node and self._concurrent_since(site, hsite, at):
                out.append((addr, addr + size))
        return sorted(out)

    def _addr_place(self, node: int, sp: str, size: int, align: int,
                    trait: dict) -> int:
        """First fit of a contiguous claim of *size* bytes, aligned to
        *align*, in an allocatable region of space *sp* whose trait it
        matches, around the ledger's intervals (F7.3e; design
        address-space-bc-design.md section 7). A run error if none fits."""
        align = max(int(align), 1)
        busy = self._addr_busy(node, sp)

        def up(x):
            return -(-x // align) * align
        for lo, rsize, rtrait in self.t.addr_regions.get(sp, ()):
            if any(trait.get(k) != v for k, v in rtrait.items()):
                continue
            cand = up(lo)
            for b0, b1 in busy:
                if b0 < cand + size and cand < b1:
                    cand = up(b1)
            if cand + size <= lo + rsize:
                return cand
        n = self.t.nodes[node]
        raise VMError(
            "contiguous address claim of %d bytes (alignment %d) in traversal "
            "%r (%s): no room in address space %r; live claims there: %s"
            % (size, align, n.path or "<root>", n.type_qname, sp,
               ", ".join("[0x%x, 0x%x)" % b for b in busy) or "none"))

    def _apart(self, node: int) -> list:
        """For each address claim of *node*, the ledger's intervals it must
        avoid, as ``(addr slot, size slot, lo, hi)``: those live now (a
        permanent one, an ancestor's, another branch's), and released ones
        made through a branch concurrent with *node*'s (as :meth:`_busy`)."""
        out = []
        site = self.by_site[node]
        for sp, a, sz, _ in self.t.addr_claims[node]:
            if a is None:
                continue                     # contiguous: placed after
            for h in self.addr_live.get(sp, ()):
                out.append((a, sz, h[2], h[2] + h[3]))
            for (hnode, hsite, addr, size), at in self.addr_released.get(sp, {}).items():
                if hnode != node and self._concurrent_since(site, hsite, at):
                    out.append((a, sz, addr, addr + size))
        return out

    def commit(self, node: int, site: Optional[int]) -> None:
        self.committed[node] = True
        self.by_site[node] = site
        if self.plan is not None and self.occ[node] is not None:
            self.plan.occurrences[self.occ[node]]["fields"] = {
                name: self.obj.get_field(s) for s, name in self.t.own_fields(node)}
            self._plan_flow_in(node)

    # -- liveness ------------------------------------------------------------

    def scope_live(self, scope: Optional[int]) -> bool:
        while scope is not None:
            commits_on_entry, parent = self.t.scopes[scope]
            if commits_on_entry and not self.entered[scope]:
                return False
            scope = parent
        return True

    def _pending_site(self, node: int) -> Optional[int]:
        for st in self.t.node_sites[node]:
            if not self.fired[st] and self.scope_live(self.t.sites[st][2]):
                return st
        return None

    def node_live(self, node: int) -> bool:
        return (node == 0 or self.committed[node] or self.started[node]
                or self._pending_site(node) is not None)

    def _in_force(self, c: SC.ScScopeConstraint, cur: int, cur_site) -> bool:
        inf = self.t.inferred
        if inf and any(inf.get(n, -1) not in self.infer_on
                       for n in (c.owner, *c.nodes) if n in inf):
            return False                       # an inferred node not run (M9a3)
        if not all(self.node_live(n) for n in c.nodes):
            return False                       # vacuously satisfied (13.4.8)
        if (not self.t.tree.lookahead or faults.on("no-lookahead")) and not all(
                n == cur or self.committed[n] for n in c.nodes):
            return False                       # calibration: no lookahead
        K = SC.ScopeConstraintKind
        if c.kind in (K.TYPE, K.COMP):
            return self.node_live(c.owner)
        if c.kind == K.ACTIVITY:
            return self.scope_live(c.scope)
        if c.kind == K.INFER:
            return True                        # its node runs (checked above)
        # WITH: at its traversal, before it (lookahead), and after it while
        # the values it chose stand.
        _, target, scope = self.t.sites[c.site]
        if not self.scope_live(scope):
            return False
        if c.site == cur_site:
            return True
        if self.committed[target] or self.started[target]:
            return self.by_site[target] == c.site and target != cur
        return self._pending_site(target) == c.site

    # -- the solve -------------------------------------------------------------

    def solve(self, node: int, site: Optional[int], seed: int, frame=None) -> Optional[int]:
        """Choose *node*'s values in its cone, with lookahead, and commit them.
        *frame* is the node's: a loop index a ``with`` reads is read from the
        frame running the loop, above it. Returns an inferred node to run
        before *node* is solved again (M9a3), nothing committed; else None."""
        t = self.t
        probs = getattr(t.tree, "flow_problems", None)
        if probs:
            # no schedule of this entry can run (lowering's FlowChecks, M8a6)
            from zuspec.ir.core.xf.validate import UnsupportedConstructError
            raise ScopeUnsatError(str(UnsupportedConstructError(
                probs[0].message, loc=probs[0].loc)))
        cone = t.cones[t.cone_of[node]]
        enabled = tuple(i for i, c in enumerate(cone.constraints)
                        if self._in_force(c, node, site))
        self.by_site[node] = site
        self.last_enabled = enabled
        pins = []
        for i, v in enumerate(cone.vars):
            pin = self._pin(v, node, frame)
            if pin is not None:
                pins.append((i, v, pin))
        picks = t.nodes_picks[node]
        # Address claims still live elsewhere (F7.3d): kept apart by an added
        # blob, which also rules out reusing the last solution
        apart = self._apart(node) if t.addr_claims[node] else ()
        blob = t.blob(cone, enabled)
        if apart:
            blob = blob + (t.apart_blob(cone, apart),)
        if getattr(t.tree.nodes[node], "stream_infer", None):
            if self._in_atomic(node, site):
                raise ScopeUnsatError(
                    "stream of traversal %r (%s) has no partner in its atomic "
                    "block, and none is inferred inside one (11.3.7)" % (
                        t.nodes[node].path or "<root>", t.nodes[node].type_qname))
            self.partners[node] = self._stream_pick(node, site, seed, frame, cone, pins)
            enabled = tuple(i for i, c in enumerate(cone.constraints)
                            if self._in_force(c, node, site))
            self.last_enabled = enabled
            blob = t.blob(cone, enabled)
            if apart:
                blob = blob + (t.apart_blob(cone, apart),)
        if not picks and not apart and t.reuse and \
                self._reuse(node, cone, enabled, pins):
            pass
        elif not picks:
            if not self._attempt(node, cone, blob, pins, seed, enabled=enabled):
                si = [x for _, by in getattr(t.tree.nodes[node], "state_infer", ()) for x in by]
                p = self._infer_next(node, site, si, seed, frame, cone, pins) if si else None
                if p is not None:
                    return p                 # a state writer first (M9a5)
                self._unsat(node, cone, enabled, pins)
        else:
            # A buffer input no bind connects: one of its pool's completed
            # objects, tried in a seeded order (D-B5); the first the cone
            # accepts is picked.
            pk = picks[0]
            var_of = {v.slot: i for i, v in enumerate(cone.vars) if v.node == node}
            tried = 0
            for pid, obj in self._candidates(pk, seed):
                tried += 1
                extra = [(var_of[pk.sel_slot], cone.vars[var_of[pk.sel_slot]], pid)]
                extra += [(var_of[s], cone.vars[var_of[s]], val)
                          for s, val in zip(pk.slots, obj[0]) if s in var_of]
                if self._attempt(node, cone, blob, pins + extra, seed,
                                 enabled=enabled):
                    self._picked(pid, obj)
                    if len(obj) > 3 and self.occ[node] is not None:
                        self.plan.consume(obj[3], self.occ[node])
                    break
            else:
                p = self._infer_next(node, site, pk.infer_nodes, seed, frame, cone, pins)
                if p is not None:
                    return p
                n = t.nodes[node]
                pools = ", ".join(sorted({"%s@%d" % (t.pools[p].name, t.pools[p].inst)
                                          for _, p in pk.pools}))
                inferred = self.infer_tried.get(node, [])
                raise ScopeUnsatError(
                    "no object output to %s so far satisfies buffer input %r of "
                    "traversal %r (%s) (%d tried), and %s" % (
                        pools, pk.ref, n.path or "<root>", n.type_qname, tried,
                        ("no inferred producer made one that does (tried %s)"
                         % ", ".join("%s (%s)" % (t.nodes[q].path, t.nodes[q].type_qname)
                                     for q in inferred)) if inferred
                        else "no action is inferred inside an atomic block (11.3.7)"
                        if self._in_atomic(node, site)
                        else "no action can be inferred to produce one"))
        for q in self.infer_tried.pop(node, ()):
            self.infer_on.discard(q)
        self.commit(node, site)
        if node == 0:
            # The root's solve chose each state pool's initial object.
            for p in t.pools:
                for src, dst in zip(p.init_slots, p.slots):
                    self.obj.set_field(dst, self.obj.get_field(src))
        if t.claims[node]:
            comp = (self.obj.get_field(t.nodes[node].comp_slot)
                    if t.nodes[node].comp_slot is not None
                    else (frame.comp if frame is not None else 0))
            self._claim(node, comp)
        if t.addr_claims[node]:
            self._addr_claim(node, getattr(frame, "stim", "") or "")

    def explain(self, node: int, seed: int) -> dict:
        """The ``solve`` debug event's detail for *node*'s last solve
        (F3.10): its rand values, and the softs in force the solution
        breaks (dropped), by source line."""
        from .unsat_core import dropped_softs, soft_detail
        t = self.t
        n = t.nodes[node]
        cone = t.cones[t.cone_of[node]]
        enabled = self.last_enabled
        w = self.witness.get(cone.id)
        softs = [i for i in enabled
                 if isinstance(cone.constraints[i].constraint, C.ConstraintSoft)]
        dropped = []
        if softs and w is not None:
            vals = {i: _domain_value(w.values[i], v) for i, v in enumerate(cone.vars)}
            src = t._problem(cone, tuple(softs))
            gone = {id(c) for c in dropped_softs(src, vals)}
            dropped = [soft_detail(c, t.con_text(cone, i))
                       for i, c in zip(softs, src.constraints) if id(c) in gone]
        return {"site": n.path or "<root>", "node": node, "type": n.type_qname,
                "seed": seed & _MASK64,
                "values": {v.name: _domain_value(self.obj.get_field(v.slot), v)
                           for v in cone.vars if v.node == node and v.rand},
                "softs": len(softs), "dropped": dropped}

    def _pin(self, v, node: int, frame, probe: bool = False) -> Optional[int]:
        """The value cone variable *v* is pinned to when *node* is solved,
        or None if it is free. (*probe*: a footprint probe of *node*, which
        has no frame yet and commits nothing.)"""
        t = self.t
        if v.node == node and v.live_from is not None:
            # A state input: its pool's current object (12.5).
            return self.obj.get_field(v.live_from)
        if v.busy_pool is not None:
            if v.node == node:
                return _busy_word(v, self._busy(v.busy_pool, node, v.busy_lock))
            if not self.committed[v.node] and t.par:
                # Not solved yet: what a footprint keeps from it already
                # holds (B5e), so the lookahead sees it.
                return _busy_word(v, self._fp_mask(
                    v.busy_pool, v.node, self.by_site[v.node] or self._pending_site(v.node),
                    v.busy_lock))
            return None                      # free before; nothing after
        if v.live_from is not None and not self.committed[v.node]:
            return None                      # not read yet: free (lookahead)
        if v.attr_pool is not None:
            # Held from the last solve that chose it, until a branch frees
            # it; fixed once a claim takes the instance (F8.6e)
            return self.obj.get_field(v.slot) if v.slot in self.attr_held else None
        if v.comp_read is not None:
            # A component attribute: fixed once the tree is constructed.
            return self.cobj.get_field(v.comp_read)
        if v.loop_local is not None and v.node == node:
            # A loop's index, at the traversal it constrains: the counter's
            # value in this iteration. (Before, it is free; after, committed
            # with the node's values.)
            return None if probe else _loop_counter(frame, v)
        if v.node == node and v.rand:
            return None
        if self.started[v.node] or self.committed[v.node]:
            if v.rand and not self.committed[v.node]:
                return None                  # started, not yet solved: free
            return self.obj.get_field(v.slot)
        if not v.rand:
            if v.slot in getattr(t, "uid_slots", ()):
                return self.obj.get_field(v.slot)    # fixed at activation
            return t.init_values.get(v.slot, 0)
        return None

    def _keep_attrs(self, cone, vals) -> None:
        """Write and hold the values a solve chose for the resource
        attributes of *cone* that no value holds yet (F8.6e)."""
        for i, v in enumerate(cone.vars):
            if v.attr_pool is not None and v.slot not in self.attr_held:
                self.obj.set_field(v.slot, vals[i] & _MASK64)
                self.attr_held.add(v.slot)

    def _reuse(self, node, cone, enabled, pins) -> bool:
        """Take *node*'s values from its cone's last solution, if that is a
        solution of this solve and chose them (see the module doc)."""
        w = self.witness.get(cone.id)
        if w is None or node in w.taken or not w.enabled.issuperset(enabled):
            return False
        vals = w.values
        pinned = set()
        for i, v, val in pins:
            if (vals[i] - _domain_value(val, v)) & _MASK64:
                return False
            pinned.add(i)
        mine = [(i, v) for i, v in enumerate(cone.vars) if v.node == node]
        if any(i not in w.free for i, _ in mine if i not in pinned):
            return False
        # What a solve would write: a state input's pinned value too.
        self._keep_attrs(cone, vals)
        for i, v in mine:
            if (v.rand or v.live_from is not None) and v.busy_pool is None \
                    and v.attr_pool is None:
                self.obj.set_field(v.slot, vals[i] & _MASK64)
                if v.rand:
                    self.solved[v.name] = vals[i] & _MASK64
        w.taken.add(node)
        self.cache.reused += 1
        return True

    def _attempt(self, node, cone, blob, pins, seed, write: bool = True,
                 enabled=None):
        """Solve *cone* with *pins*; on success write *node*'s values (not
        *write*: return every variable's value by slot). False, or None
        when not *write*, if there is no solution. With *enabled* (the
        constraints in force), the solution becomes the cone's witness."""
        from dv_solve.ctx import SOLVE_OK, SOLVE_TIMEOUT, CompileUnsatError
        if cone.id in self.t.array_cones:
            return self._attempt_arrays(node, cone, pins, seed, write, enabled)
        if self.bounds:
            if cone.id not in self._bound_blobs:
                self._bound_blobs[cone.id] = self.t.bound_blob(cone, self.bounds)
            bb = self._bound_blobs[cone.id]
            if bb is not None:
                blob = (blob if isinstance(blob, tuple) else (blob,)) + (bb,)
        try:
            with self.cache.session(blob) as ctx:
                for i, v, val in pins:
                    if not ctx.pin(i, _domain_value(val, v)):
                        return False if write else None
                rc = self.cache.solve(ctx, seed & _MASK64)
                if rc == SOLVE_TIMEOUT:
                    n = self.t.nodes[node]
                    raise SolveBudgetError(
                        "choosing values for traversal %r (%s): %s"
                        % (n.path or "<root>", n.type_qname,
                           self.cache.budget_message()))
                if rc != SOLVE_OK:
                    return False if write else None
                if not write:
                    return {v.slot: ctx.get_value(i) & _MASK64
                            for i, v in enumerate(cone.vars)}
                vals = self.t.values(ctx, cone)
                self._keep_attrs(cone, vals)
                for i, v in enumerate(cone.vars):
                    if v.node == node and (v.rand or v.live_from is not None) \
                            and v.busy_pool is None and v.attr_pool is None:
                        self.obj.set_field(v.slot, vals[i] & _MASK64)
                        if v.rand:
                            self.solved[v.name] = vals[i] & _MASK64
                if enabled is not None:
                    pinned = {i for i, _, _ in pins}
                    self.witness[cone.id] = _Witness(
                        frozenset(enabled), vals,
                        frozenset(i for i in range(len(vals)) if i not in pinned),
                        {node})
        except CompileUnsatError:
            return False if write else None
        return True

    def _attempt_arrays(self, node, cone, pins, seed, write, enabled):
        """``_attempt`` of a cone that reads a list (F6.10d): solved by
        dv-solve's handle, each list's current elements passed as an array
        and the pinned variables as state. (A property test's bounds are
        not applied to such a cone.)"""
        from dv_solve.handle import RANDOMIZE_OK
        if self.heap is None:
            raise VMError("a constraint reads a list, but this run has no heap")
        arrays, state, pinned = [], {}, set()
        vid = {}
        for i, v in enumerate(cone.vars):
            if not v.array:
                vid[i] = len(vid)
        given = {i: val for i, _, val in pins}
        for i, v in enumerate(cone.vars):
            if v.array:
                # the list's handle: the pin (comp attribute, committed
                # field or initial value), else the field as it is now
                h = given.get(i)
                if h is None:
                    h = self.obj.get_field(v.slot)
                arrays.append([x & _MASK64 for x in self.heap.items(h)])
        for i, v, val in pins:
            if not v.array:
                pinned.add(i)
                state[vid[i]] = _domain_value(val, v)
        h = self.t.handle(cone, tuple(enabled or ()), frozenset(pinned))
        rc, got, elems = h.randomize(seed & _MASK64, state, arrays)
        if rc != RANDOMIZE_OK:
            return False if write else None
        vals = [got.get(vid[i], 0) if i in vid else 0 for i in range(len(cone.vars))]
        if not write:
            return {v.slot: vals[i] & _MASK64 for i, v in enumerate(cone.vars)
                    if not v.array}
        aid = 0
        for i, v in enumerate(cone.vars):
            if not v.array:
                continue
            if v.node == node and v.rand:
                # A rand list's elements (F6.10e), in place: it keeps the
                # length it had when solved.
                lh = given.get(i) or self.obj.get_field(v.slot)
                if lh:
                    self.heap.obj(lh).data[:] = [x & _MASK64 for x in elems[aid]]
            aid += 1
        self._keep_attrs(cone, vals)
        for i, v in enumerate(cone.vars):
            if v.node == node and not v.array and (v.rand or v.live_from is not None) \
                    and v.busy_pool is None and v.attr_pool is None:
                self.obj.set_field(v.slot, vals[i] & _MASK64)
                if v.rand:
                    self.solved[v.name] = vals[i] & _MASK64
        # No witness: a list's elements may change while its handle, the
        # pin a reuse compares, does not.
        return True

    def _candidates(self, pk, seed):
        """The objects buffer input *pk* may pick: ``(pool, [values,
        consumers, _])`` for each object output to one of its pools by a
        completed action -- those consumed least first, in an order drawn
        from *seed* among equals. (Any is legal: a buffer object may have
        many consumers. Preferring a fresh one is what a test that reads
        back what it wrote expects.) The order is drawn as it is consumed,
        so a pick costs what it tries, not what the pools hold."""
        import random
        rng = random.Random(seed)
        pools = [(pid, self.outputs.get(pid, {}))
                 for pid in sorted({p for _, p in pk.pools})]
        for times in sorted({k for _, by in pools for k, objs in by.items() if objs}):
            groups = [(pid, by.get(times, ())) for pid, by in pools]
            n = sum(len(objs) for _, objs in groups)
            swap: Dict[int, int] = {}
            for k in range(n):
                # a Fisher-Yates shuffle of 0..n-1, one position at a time
                r = rng.randrange(k, n)
                x = swap.get(r, r)
                swap[r] = swap.get(k, k)
                for pid, objs in groups:
                    if x < len(objs):
                        yield pid, objs[x]
                        break
                    x -= len(objs)

    def _in_atomic(self, node, site) -> bool:
        """*node*, traversed through *site*, is in an atomic block, and not
        itself inferred (an inferred action and its own chain run before
        the block)."""
        t = self.t
        if faults.on("atomic-infer"):
            return False
        return bool(t.atomic) and site is not None \
            and t.tree.nodes[node].inferred_for is None \
            and any(sc in t.atomic for sc in t.site_chain[site])

    def atomic_infer(self, scope: int, seed: int, comp) -> Optional[int]:
        """M9a7: entry to *scope*. When it is an atomic block (11.3.7), an
        action inferred for one of its traversals runs now, before the
        block, never inside it: the first traversal of the block whose
        buffer input no completed object fits (or whose state input the
        current object does not satisfy) gets its next inferred node, as
        its own solve would (``_infer_next``), and that node is returned to
        run before the block is entered again. A traversal an explicit
        action earlier in the block may feed is left to its own solve.
        (*comp*: the entering frame's component instance.)"""
        t = self.t
        if scope not in t.atomic or faults.on("atomic-infer"):
            return None
        for need in self._unfed(scope, seed, comp):
            p = self._infer_next(*need, pre=True)
            if p is not None:
                return p
        return None

    def _unfed(self, scope: int, seed: int, comp, all_picks: bool = False):
        """The traversals of *scope* (in site order) whose buffer input no
        completed object fits (or whose state input the current object does
        not satisfy), each as the arguments ``_infer_next`` takes for it.
        A traversal an explicit action earlier in the block may feed is
        left out. Only inputs with inferred candidates count, unless
        *all_picks* (then ``infer_nodes`` may be empty). (*comp*: the
        entering frame's component instance.)"""
        t = self.t
        sites = [st for st in t.scope_sites[scope]
                 if t.tree.nodes[t.sites[st][1]].inferred_for is None]
        for k, st in enumerate(sites):
            c = t.sites[st][1]
            if c not in t.cone_of:
                continue
            pk = next((x for x in t.nodes_picks[c] if all_picks or x.infer_nodes), None)
            si = [x for _, by in getattr(t.tree.nodes[c], "state_infer", ()) for x in by]
            if pk is None and not si:
                continue
            if pk is not None:
                mine = {p for _, p in pk.pools}
                if any(mine & {p for w in t.buffer_writes[t.sites[o][1]] for _, p in w.pools}
                       for o in sites[:k]):
                    continue
            cone = t.cones[t.cone_of[c]]
            enabled = tuple(i for i, x in enumerate(cone.constraints)
                            if self._in_force(x, c, st))
            pins = []
            for i, v in enumerate(cone.vars):
                pin = self._pin(v, c, None, probe=True)
                if pin is not None:
                    pins.append((i, v, pin))
            blob = t.blob(cone, enabled)
            if pk is not None:
                var_of = {v.slot: i for i, v in enumerate(cone.vars) if v.node == c}
                fits = False
                for pid, obj in self._candidates(pk, seed):
                    extra = [(var_of[pk.sel_slot], cone.vars[var_of[pk.sel_slot]], pid)]
                    extra += [(var_of[s], cone.vars[var_of[s]], val)
                              for s, val in zip(pk.slots, obj[0]) if s in var_of]
                    if self._attempt(c, cone, blob, pins + extra, seed, write=False,
                                     enabled=enabled) is not None:
                        fits = True
                        break
                infer_nodes = pk.infer_nodes
            else:
                fits = self._attempt(c, cone, blob, pins, seed, write=False,
                                     enabled=enabled) is not None
                infer_nodes = si
            if fits:
                continue
            rel = t.nodes[c].comp_rel
            at = types.SimpleNamespace(
                comp=None if rel is None or comp is None else comp + rel)
            yield c, st, infer_nodes, seed, at, cone, pins

    def _fed_by(self, c, st, p, cone, pins, seed) -> bool:
        """Whether what inferred node *p* outputs (as a probe solve with it
        on picks) can be the object *c*'s buffer input takes. *p* runs and
        then *c* picks, so its values are not chosen to fit *c*'s. Without
        a buffer input (a state writer) or with *p*'s output fields free,
        what the probe in ``_infer_next`` found stands."""
        t = self.t
        pk = next((x for x in t.nodes_picks[c]), None)
        if pk is None:
            return True
        enabled = tuple(i for i, x in enumerate(cone.constraints)
                        if self._in_force(x, c, st))
        blob = t.blob(cone, enabled)
        vals = self._attempt(c, cone, blob, self._infer_pins(c, p, cone, pins), seed,
                             write=False, enabled=enabled)
        mine = {q for _, q in pk.pools}
        var_of = {v.slot: i for i, v in enumerate(cone.vars) if v.node == c}
        for w in t.buffer_writes[p]:
            pid = next((q for _, q in w.pools if q in mine), None)
            if pid is None or vals is None:
                continue
            extra = [(var_of[pk.sel_slot], cone.vars[var_of[pk.sel_slot]], pid)]
            extra += [(var_of[s], cone.vars[var_of[s]], vals[src])
                      for s, src in zip(pk.slots, w.src) if s in var_of and src in vals]
            return self._attempt(c, cone, blob, pins + extra, seed, write=False,
                                 enabled=enabled) is not None
        return True

    def select_blocked(self, scope: int, seed: int, comp) -> Optional[str]:
        """M9a7c (M8b5b): a select branch whose block is *scope* cannot run
        now: a traversal in it has a flow input no completed object fits
        and for which no action can be inferred (none is a candidate, or
        none can hold with it; 11.3.7: none inside an enclosing atomic
        block). The reason, or None when the branch may be chosen. Nothing
        is turned on: the branch's own solves infer as usual."""
        t = self.t
        if t.tree.scopes[scope].kind != SC.ScopeKind.SELECT_BRANCH \
                or faults.on("select-no-prune"):
            return None
        # probed as if the block were entered (its constraints in force)
        entered = [(sc, self.entered[sc]) for sc in [scope] + t.scope_nested[scope]]
        fired = [(st, self.fired[st]) for st in t.scope_sites[scope]]
        for st, _ in fired:
            self.fired[st] = False
        for sc, _ in entered[1:]:
            self.entered[sc] = False
        self.entered[scope] = True
        try:
            return self._blocked(scope, seed, comp)
        finally:
            for sc, v in entered:
                self.entered[sc] = v
            for st, v in fired:
                self.fired[st] = v

    def _blocked(self, scope: int, seed: int, comp) -> Optional[str]:
        t = self.t
        for need in self._unfed(scope, seed, comp, all_picks=True):
            c, st = need[0], need[1]
            chain = t.site_chain[st]
            outer = chain[chain.index(scope) + 1:] if scope in chain else chain
            on, tried = set(self.infer_on), self.infer_tried.get(c)
            saved = None if tried is None else list(tried)
            self.infer_tried[c] = []
            try:
                pre = not any(sc in t.atomic for sc in outer)
                while True:
                    p = self._infer_next(*need, pre=pre)
                    if p is None or self._fed_by(c, st, p, need[5], need[6], seed):
                        break
            finally:
                self.infer_on.clear()
                self.infer_on.update(on)
                if saved is None:
                    self.infer_tried.pop(c, None)
                else:
                    self.infer_tried[c] = saved
            if p is None:
                n = t.nodes[c]
                return "traversal %r (%s) has no input object and none can be inferred" % (
                    n.path or "<root>", n.type_qname)
        return self._lock_blocked(scope, comp)

    def _lock_blocked(self, scope: int, comp) -> Optional[str]:
        """M9a8c: a lock in select branch *scope*, of a pool the branches of
        an enclosing parallel share, has no instance left in its branch's
        footprint that no concurrent claim holds (its footprint reserved
        nothing for it: ``_optional_locker``)."""
        t = self.t
        if not t.par:
            return None
        for st in t.scope_sites[scope]:
            shared = set()
            for sc in t.site_chain[st]:
                if sc in t.par:
                    shared |= t.par[sc].shared
            if not shared:
                continue
            for x in t.subtree[t.sites[st][1]]:
                for c in t.claims[x]:
                    pid = _pool_for(c.pools, comp)
                    if pid not in shared:
                        continue
                    full = (1 << t.pools[pid].size) - 1
                    free = full & ~self._busy(pid, x, c.lock)
                    free &= ~self._fp_mask(pid, x, st, c.lock)
                    if free and c.lock and self._sharer_waiting(pid, st, comp):
                        # keep one instance for a concurrent share on its way
                        avail = full & ~self._busy(pid, x, False)
                        avail &= ~self._fp_mask(pid, x, st, False)
                        free &= ~(avail if avail & (avail - 1) == 0 else 0)
                    if not free:
                        n = t.nodes[x]
                        return "traversal %r (%s) cannot %s: no instance of %s@%d left" % (
                            n.path or "<root>", n.type_qname,
                            "lock" if c.lock else "share", t.pools[pid].name,
                            t.pools[pid].inst)
        return None

    def _chosen(self, branch: int, par: int) -> bool:
        """Select branch scope *branch* was chosen in this traversal of
        parallel *par*."""
        fp = self.footprint.get(par)
        return fp is not None and self.chosen.get(branch) == fp[0]

    def select_chosen(self, scope: int, comp) -> None:
        """M9a8c: select branch *scope* is chosen. Each enclosing parallel's
        footprint for it narrows to the instances its locks take (one per
        lock, the lowest free), so a concurrent share is kept only from
        those (``_chosen_fp``)."""
        t = self.t
        if not t.par or t.tree.scopes[scope].kind != SC.ScopeKind.SELECT_BRANCH:
            return
        for st in t.scope_sites[scope]:
            chain = t.site_chain[st]
            for i, sc in enumerate(chain):
                fp = self.footprint.get(sc)
                if fp is None or fp[0] != self.entered_at[sc]:
                    continue
                if scope in chain[:i]:
                    self.chosen[scope] = fp[0]
                below = chain[i - 1] if i else ("site", st)
                bi = next((j for j, br in enumerate(t.par[sc].branches) if br.key == below), None)
                if bi is None:
                    continue
                need: Dict[int, List[int]] = {}
                for x in t.subtree[t.sites[st][1]]:
                    for c in t.claims[x]:
                        pid = _pool_for(c.pools, comp)
                        if c.lock and pid in t.par[sc].shared:
                            need.setdefault(pid, []).append(x)
                for pid, xs in need.items():
                    free = fp[1][bi].get(pid, 0) & ~self._busy(pid, xs[0], True)
                    keep = 0
                    for k in range(t.pools[pid].size):
                        if free >> k & 1 and bin(keep).count("1") < len(xs):
                            keep |= 1 << k
                    fp[1][bi][pid] = keep

    def _sharer_waiting(self, pid: int, site: int, comp) -> bool:
        """A share of pool *pid* in another branch of a parallel *site* is
        in is not yet solved and will be (unconditional, or its select
        branches chosen)."""
        t = self.t
        mine = t.site_chain[site]
        for y, claims in enumerate(t.claims):
            if self.committed[y] or t.tree.nodes[y].inferred_for is not None:
                continue
            if not any(not c.lock and _pool_for(c.pools, comp) == pid for c in claims):
                continue
            for sty in t.node_sites[y]:
                chain = t.site_chain[sty]
                for k, sc in enumerate(chain):
                    if sc not in t.par or sc not in mine:
                        continue
                    ky = chain[k - 1] if k else ("site", sty)
                    i = mine.index(sc)
                    kx = mine[i - 1] if i else ("site", site)
                    if ky != kx and all(self._chosen(s2, sc) for s2 in chain[:k]
                                        if t.tree.scopes[s2].kind == SC.ScopeKind.SELECT_BRANCH):
                        return True
                    break
        return False

    def _infer_next(self, node, site, infer_nodes, seed, frame, cone, pins,
                    pre: bool = False) -> Optional[int]:
        """M9a3: no completed object fits a buffer input of *node* (M9a5: or
        its state input's current object does not): the next of
        *infer_nodes* (``(instance, [node, ...])``) to run first (its
        constraints turned on), in an order drawn from *seed* among those
        not tried yet for this solve whose constraints and *node*'s can
        hold together (a probe solve) -- None when none is left. A
        producer whose object did not fit is turned off again
        (chronological retry). Inside an atomic block none is, unless
        *pre*: the block's entry choosing them for it (M9a7)."""
        import random
        if not infer_nodes or (not pre and self._in_atomic(node, site)):
            return None
        comp = getattr(frame, "comp", None)
        insts = [nids for k, nids in infer_nodes if k == comp]
        if not insts:
            # M9b1c: the solve chooses *node*'s instance; each instance's
            # producers' INFER constraints pin it to theirs
            insts = [nids for _, nids in infer_nodes]
            for k, nids in infer_nodes:
                for p in nids:
                    self.infer_comp[p] = k
        tried = self.infer_tried.setdefault(node, [])
        left = [p for nids in insts for p in nids if p not in tried]
        for q in tried:
            self.infer_on.discard(q)
        random.Random(seed).shuffle(left)
        for p in left:
            tried.append(p)
            self.infer_on.add(p)
            enabled = tuple(i for i, c in enumerate(cone.constraints)
                            if self._in_force(c, node, site))
            if self._attempt(node, cone, self.t.blob(cone, enabled),
                             self._infer_pins(node, p, cone, pins), seed,
                             write=False) is not None \
                    or self._chain_probe(node, site, [p], seed, cone, pins):
                return p
            self.infer_on.discard(p)
        return None

    def _stream_pick(self, node, site, seed, frame, cone, pins) -> List[int]:
        """M9a6b: a stream reference of *node* with no partner in the
        activity: one of its inferred partners (``stream_infer``) is turned
        on, in an order drawn from *seed* among those whose constraints and
        *node*'s can hold together (a probe solve). The partners are
        started with *node* once it commits."""
        import random
        t = self.t
        comp = getattr(frame, "comp", None)
        picked = []
        for field, by in t.tree.nodes[node].stream_infer:
            for _, nids in by:
                for q in nids:
                    self.infer_on.discard(q)   # last traversal's partner
            insts = [nids for k, nids in by if k == comp]
            if not insts and len(by) == 1:
                insts = [by[0][1]]
            left = [q for nids in insts for q in nids]
            random.Random(seed).shuffle(left)
            for q in left:
                self.infer_on.add(q)
                enabled = tuple(i for i, c in enumerate(cone.constraints)
                                if self._in_force(c, node, site))
                if self._attempt(node, cone, t.blob(cone, enabled), pins, seed,
                                 write=False) is not None:
                    picked.append(q)
                    break
                self.infer_on.discard(q)
            else:
                n = t.nodes[node]
                raise ScopeUnsatError(
                    "stream %r of traversal %r (%s) has no partner in the activity, "
                    "and no inferred partner can take its object (tried %s)" % (
                        field, n.path or "<root>", n.type_qname,
                        ", ".join("%s (%s)" % (t.nodes[q].path, t.nodes[q].type_qname)
                                  for q in left) or "none"))
        return picked

    def _chain_probe(self, node, site, chain, seed, cone, pins) -> bool:
        """M9a5c: whether the last writer of *chain* (inferred nodes, run
        last to first before *node*) can read an object a writer of its
        own outputs, at most MAX_DEPTH deep: a probe with that writer on
        too, reading the current object. The writer is picked again when
        the last of *chain* is solved; all turned off again here."""
        from zuspec.ir.core.xf.pss_lower.infer import MAX_DEPTH
        t = self.t
        si = [q for _, by in getattr(t.tree.nodes[chain[-1]], "state_infer", ())
              for _, nids in by for q in nids]
        if not si or len(chain) >= MAX_DEPTH:
            return False
        for q in si:
            self.infer_on.add(q)
            run = chain + [q]
            enabled = tuple(i for i, c in enumerate(cone.constraints)
                            if self._in_force(c, node, site))
            ok = self._attempt(node, cone, t.blob(cone, enabled),
                               self._infer_pins(node, run, cone, pins), seed,
                               write=False) is not None \
                or self._chain_probe(node, site, run, seed, cone, pins)
            self.infer_on.discard(q)
            if ok:
                return True
        return False

    def _infer_pins(self, node, p, cone, pins) -> list:
        """*pins* for a probe of inferred node *p* before *node*: *node*'s
        state inputs from a pool *p* writes are what *p* outputs, not the
        current object (M9a5); *p*'s own subtree runs next, so its state
        inputs and ``prev`` read the current object now."""
        t = self.t
        chain = p if isinstance(p, list) else [p]
        # M9a5c: in a chain, each writer reads what the next one writes
        # and only the last reads the current object
        readers = [node] + chain[:-1]
        out = list(pins)
        for r, w in zip(readers, chain):
            written = {s for x in t.state_writes[w] for _, pid in x.pools
                       for s in t.pools[pid].slots}
            out = [x for x in out if not (x[1].node == r and x[1].live_from in written)]
        for i, v in enumerate(cone.vars):
            if v.live_from is not None and t.inferred.get(v.node) == chain[-1] \
                    and not self.committed[v.node]:
                out.append((i, v, self.obj.get_field(v.live_from)))
        return out

    def _picked(self, pid: int, obj: list) -> None:
        """*obj*, of pool *pid*, has one more consumer."""
        by = self.outputs[pid]
        objs = by[obj[1]]
        last = objs.pop()
        if last is not obj:
            objs[obj[2]] = last
            last[2] = obj[2]
        obj[1] += 1
        nxt = by.setdefault(obj[1], [])
        obj[2] = len(nxt)
        nxt.append(obj)

    def _unsat(self, node, cone, enabled, pins):
        t = self.t
        n = t.nodes[node]
        cons = "; ".join(t.con_text(cone, i) for i in enabled) or "(none)"
        fixed = ", ".join(
            "%s=%s" % (v.name, self._list_text(val) if v.array else _domain_value(val, v))
            for _, v, val in pins) or "(none)"
        # F3.8b: the constraints in force that conflict, by source line
        from .unsat_core import describe, unsat_core
        core = [] if cone.id in t.array_cones else unsat_core(
            t._problem(cone, tuple(enabled)),
            {i: _domain_value(val, v) for i, v, val in pins})
        msg = ("no values for traversal %r (%s) satisfy the constraints in force: "
               "%s; with %s" % (n.path or "<root>", n.type_qname, cons, fixed))
        # the host's fixed values: an export's arguments, a field override
        host = []
        for v in cone.vars:
            b = self.bounds.get(v.name, self.bounds.get(v.slot)) if v.rand else None
            if b is not None and b[0] is not None and b[0] == b[1]:
                host.append("%s=%s" % (v.name, b[0]))
        if host:
            msg += "; fixed by the caller: " + ", ".join(host)
        if core:
            msg += "; " + describe(core)
        e = ScopeUnsatError(msg)
        e.core = [c.loc for c in core]
        raise e

    def _list_text(self, h) -> str:
        if self.heap is None:
            return "list"
        return "{%s}" % ", ".join(str(x) for x in self.heap.items(h))


def _pool_for(pools, comp) -> int:
    """The pool of ``(instance, pool)`` pairs *pools* for instance *comp*."""
    if len(pools) == 1:
        return pools[0][1]
    for k, pid in pools:
        if k == comp:
            return pid
    raise VMError("no pool for component instance %r" % comp)


def _loop_counter(frame, v) -> int:
    """The value of loop index *v* in the nearest frame above *frame* that
    runs node ``v.loop_node`` and holds the counter."""
    f = frame.parent if frame is not None else None
    while f is not None:
        if f.node == v.loop_node and v.loop_local in f.coro.frame_locals:
            return f.locals[f.coro.frame_locals.index(v.loop_local)]
        f = f.parent
    raise VMError("loop index %r is read by a `with` outside its loop" % v.loop_local)


def _busy_word(v, mask: int) -> int:
    """Busy variable *v*'s word of a pool's busy *mask* (busy_word)."""
    return (mask >> (SC.BUSY_WORD_BITS * v.busy_word)) & ((1 << v.width) - 1)


def _domain_value(val: int, v) -> int:
    """An object slot's 64-bit pattern as a value of *v*'s domain."""
    w = v.width if v.width and v.width > 0 else 32
    x = int(val) & ((1 << w) - 1)
    if v.signed and x >> (w - 1):
        x -= 1 << w
    return x


_OPS = {E.BinOp.Lt: "<", E.BinOp.LtE: "<=", E.BinOp.Gt: ">", E.BinOp.GtE: ">=",
        E.BinOp.Eq: "==", E.BinOp.NotEq: "!=", E.BinOp.Add: "+", E.BinOp.Sub: "-",
        E.BinOp.Mult: "*", E.BinOp.Div: "/", E.BinOp.Mod: "%", E.BinOp.And: "&&",
        E.BinOp.Or: "||", E.BinOp.BitAnd: "&", E.BinOp.BitOr: "|",
        E.BinOp.BitXor: "^", E.BinOp.LShift: "<<", E.BinOp.RShift: ">>"}


def describe(c, names: List[str]) -> str:
    """A constraint as PSS-like text, slots named by their tree path."""
    def ex(e) -> str:
        if isinstance(e, E.ExprRefField):
            return names[e.index] if 0 <= e.index < len(names) else "$%d" % e.index
        if isinstance(e, E.ExprConstant):
            return str(e.value)
        if isinstance(e, E.ExprBin):
            return "%s %s %s" % (ex(e.lhs), _OPS.get(e.op, e.op.name), ex(e.rhs))
        if isinstance(e, E.ExprUnary):
            return "%s%s" % ({"Not": "!", "USub": "-", "Invert": "~"}.get(
                e.op.name, e.op.name), ex(e.operand))
        if isinstance(e, E.ExprBool):
            j = " && " if e.op is E.BoolOp.And else " || "
            return "(" + j.join(ex(v) for v in e.values) + ")"
        if isinstance(e, (E.ExprIfExp, _IfExpP2)):
            return "(%s ? %s : %s)" % (ex(e.test), ex(e.body), ex(e.orelse))
        if isinstance(e, E.ExprCast):
            dt = e.target_type
            tn = getattr(dt, "name", None) or type(dt).__name__
            if type(dt).__name__ == "DataTypeInt" and tn != "bool":
                tn = "%s[%d]" % ("int" if dt.signed else "bit", dt.bits or 32)
            return "(%s)%s" % (tn, ex(e.value))
        if type(e).__name__ == "ExprIn":
            return "%s in %s" % (ex(e.value), ex(e.container))
        if isinstance(e, E.ExprCall) and isinstance(e.func, E.ExprAttribute):
            return "%s.%s()" % (ex(e.func.value), e.func.attr)   # l.size()
        return type(e).__name__
    if isinstance(c, C.ConstraintExpr):
        return ex(c.expr)
    if isinstance(c, C.ConstraintSoft):
        return "soft " + ex(c.expr)
    return type(c).__name__
