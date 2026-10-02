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
"""

from __future__ import annotations

import dataclasses as dc
from typing import Dict, List, Optional, Tuple

from zuspec.ir.core import constraint as C
from zuspec.ir.core import expr as E
from zuspec.ir.core import scenario as SC

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
class _Cone:
    id: int
    nodes: List[int]
    vars: List[SC.ScScopeVar]
    constraints: List[SC.ScScopeConstraint]


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
        self.init_values = dict(init_values)
        self.names = list(names)

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
        self._blobs: Dict[Tuple[int, Tuple[int, ...]], bytes] = {}

        # -- flow objects and resources (B5) --
        self.pools = list(tree.pools)
        self.claims = [list(nd.claims) for nd in tree.nodes]
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
        self._conc: Dict[Tuple[int, int], Optional[int]] = {}
        self.depth = [0] * n
        for i, node in enumerate(self.nodes):
            self.depth[i] = len(self.ancestors[i])
        #: parallel scope -> its branches' footprints to choose on entry (B5e)
        self.par: Dict[int, _Par] = {}
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

    def site(self, node: int, local: int) -> int:
        return self.nodes[node].site_base + local

    # -- the problem of one set of constraints in force ----------------------

    def blob(self, cone: _Cone, enabled: Tuple[int, ...]) -> bytes:
        key = (cone.id, enabled)
        b = self._blobs.get(key)
        if b is None:
            from ..lower.constraints import build_solve_blob
            problem = SC.ScSolveProblem(
                vars=[SC.ScSolveVar(name=v.name, var_id=i, slot=v.slot,
                                    width=v.width, signed=v.signed)
                      for i, v in enumerate(cone.vars)],
                constraints=[cone.constraints[i].constraint for i in enabled])
            b, _ = build_solve_blob(problem)
            self._blobs[key] = b
        return b

    def check(self) -> None:
        """Build each cone with every constraint enabled, so a constraint the
        solver cannot take is a lowering error, not a run-time one."""
        for c in self.cones:
            self.blob(c, tuple(range(len(c.constraints))))


class Activation:
    """The state of one run of an :class:`ActivationTable`."""

    def __init__(self, table: ActivationTable, obj, cache=None, cobj=None):
        """*cache*: the run's :class:`~.solve_cache.SolveCache` (a private
        one when None). *cobj*: the component object, which a constraint
        reading a component attribute is pinned from."""
        self.t = table
        self.obj = obj
        self.cobj = cobj
        if cache is None:
            from .solve_cache import SolveCache
            cache = SolveCache()
        self.cache = cache
        n = len(table.nodes)
        self.committed = [False] * n
        self.started = [False] * n
        self.by_site: List[Optional[int]] = [None] * n
        self.fired = [False] * len(table.sites)
        self.entered = [False] * len(table.scopes)
        # -- the claim table (B5, D-B12) --
        #: an event counter: when a scope was entered, when a claim was made
        self.clock = 0
        self.entered_at = [0] * len(table.scopes)
        #: pool id -> claims made: [node, site, instance_id, lock, at, held]
        self.holds: Dict[int, List[list]] = {}
        #: buffer pool id -> the objects completed actions output to it:
        #: [values, times picked]
        self.outputs: Dict[int, List[list]] = {}
        #: parallel scope -> (entered at, per branch: pool id -> footprint)
        self.footprint: Dict[int, Tuple[int, List[Dict[int, int]]]] = {}

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
        self.started[node] = True
        if site is not None:
            self.fired[site] = True

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
        self.clock += 1
        self.entered_at[scope] = self.clock
        if scope in t.par:
            self._footprints(scope, seed() if seed is not None else 0)

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
        out at random among the branches that use their pool. A branch whose
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
            users = [i for i, m in enumerate(fps) if pid in m]
            if not users:
                continue
            free = ((1 << t.pools[pid].size) - 1) & ~reserved.get(pid, 0)
            for i in range(t.pools[pid].size):
                if free >> i & 1:
                    fps[rng.choice(users)][pid] |= 1 << i
        self.footprint[scope] = (self.entered_at[scope], fps)

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
                    pins.append((i, v, mask | self._busy(pid, v.node, v.busy_lock)))
                continue
            pin = self._pin(v, member, None, probe=True)
            if pin is not None:
                pins.append((i, v, pin))
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

    def _fp_mask(self, pid: int, node: int, site: Optional[int]) -> int:
        """The instances of pool *pid* outside the footprint of the branch
        *node* (reached through *site*) is in, for each parallel it is in."""
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
                mask |= full & ~fp[1][bi].get(pid, 0)
        return mask

    def exit_node(self, node: int, comp: Optional[int], completed: bool = True) -> None:
        """A traversal of *node* ends (*completed*: it ran to its end, rather
        than being cancelled). Its claims are released; a completed node's
        state outputs become their pools' current objects (12.5)."""
        t = self.t
        for pool in self.holds.values():
            for h in pool:
                if h[0] == node and h[5]:
                    h[5] = False
        if not completed:
            return
        for w in t.buffer_writes[node]:
            self.outputs.setdefault(_pool_for(w.pools, comp), []).append(
                [tuple(self.obj.get_field(s) for s in w.src), 0])
        for w in t.state_writes[node]:
            pid = _pool_for(w.pools, comp)
            dst = t.pools[pid].slots
            for src, d in zip(w.src, dst):
                self.obj.set_field(d, self.obj.get_field(src))

    def _busy(self, pid: int, node: int, lock: bool) -> int:
        """The instances of pool *pid* a claim of *node* may not take: those a
        conflicting claim holds now (other than its own ancestors'), and those
        a conflicting claim concurrent with it took (LRM 9.4, 11.3.4)."""
        t = self.t
        site = self.by_site[node]
        mask = 0
        for h in self.holds.get(pid, ()):
            hnode, hsite, iid, hlock, at, held = h
            if not (lock or hlock) or hnode == node:
                continue
            if held and hnode not in t.ancestors[node]:
                mask |= 1 << iid
                continue
            if site is not None and hsite is not None:
                sc = t.concurrent(site, hsite)
                if sc is not None and at >= self.entered_at[sc]:
                    mask |= 1 << iid
        if t.par:
            mask |= self._fp_mask(pid, node, site if site is not None
                                  else self._pending_site(node))
        return mask

    def _claim(self, node: int, comp: Optional[int]) -> None:
        """Record *node*'s claims, now that its solve chose them."""
        self.clock += 1
        for c in self.t.claims[node]:
            pid = _pool_for(c.pools, comp)
            iid = self.obj.get_field(c.iid_slot)
            self.holds.setdefault(pid, []).append(
                [node, self.by_site[node], iid, c.lock, self.clock, True])

    def commit(self, node: int, site: Optional[int]) -> None:
        self.committed[node] = True
        self.by_site[node] = site

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
        if not all(self.node_live(n) for n in c.nodes):
            return False                       # vacuously satisfied (13.4.8)
        if not self.t.tree.lookahead and not all(
                n == cur or self.committed[n] for n in c.nodes):
            return False                       # calibration: no lookahead
        K = SC.ScopeConstraintKind
        if c.kind in (K.TYPE, K.COMP):
            return self.node_live(c.owner)
        if c.kind == K.ACTIVITY:
            return self.scope_live(c.scope)
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

    def solve(self, node: int, site: Optional[int], seed: int, frame=None) -> None:
        """Choose *node*'s values in its cone, with lookahead, and commit them.
        *frame* is the node's: a loop index a ``with`` reads is read from the
        frame running the loop, above it."""
        t = self.t
        cone = t.cones[t.cone_of[node]]
        enabled = tuple(i for i, c in enumerate(cone.constraints)
                        if self._in_force(c, node, site))
        self.by_site[node] = site
        pins = []
        for i, v in enumerate(cone.vars):
            pin = self._pin(v, node, frame)
            if pin is not None:
                pins.append((i, v, pin))
        blob = t.blob(cone, enabled)
        picks = t.nodes_picks[node]
        if not picks:
            if not self._attempt(node, cone, blob, pins, seed):
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
                if self._attempt(node, cone, blob, pins + extra, seed):
                    obj[1] += 1
                    break
            else:
                n = t.nodes[node]
                pools = ", ".join(sorted({"%s@%d" % (t.pools[p].name, t.pools[p].inst)
                                          for _, p in pk.pools}))
                raise ScopeUnsatError(
                    "no object output to %s so far satisfies buffer input %r of "
                    "traversal %r (%s) (%d tried); inferring a producer is not "
                    "supported yet" % (pools, pk.ref, n.path or "<root>",
                                       n.type_qname, tried))
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
                return self._busy(v.busy_pool, node, v.busy_lock)
            if not self.committed[v.node] and t.par:
                # Not solved yet: what a footprint keeps from it already
                # holds (B5e), so the lookahead sees it.
                return self._fp_mask(v.busy_pool, v.node,
                                     self.by_site[v.node] or self._pending_site(v.node))
            return None                      # free before; nothing after
        if v.live_from is not None and not self.committed[v.node]:
            return None                      # not read yet: free (lookahead)
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
            return t.init_values.get(v.slot, 0)
        return None

    def _attempt(self, node, cone, blob, pins, seed, write: bool = True):
        """Solve *cone* with *pins*; on success write *node*'s values (not
        *write*: return every variable's value by slot). False, or None
        when not *write*, if there is no solution."""
        from dv_solve.ctx import SOLVE_OK, SOLVE_TIMEOUT, CompileUnsatError
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
                for i, v in enumerate(cone.vars):
                    if v.node == node and (v.rand or v.live_from is not None) \
                            and v.busy_pool is None:
                        self.obj.set_field(v.slot, ctx.get_value(i) & _MASK64)
        except CompileUnsatError:
            return False if write else None
        return True

    def _candidates(self, pk, seed):
        """The objects buffer input *pk* may pick: ``(pool, [values,
        consumers])`` for each object output to one of its pools by a
        completed action -- those consumed least first, in an order drawn
        from *seed* among equals. (Any is legal: a buffer object may have
        many consumers. Preferring a fresh one is what a test that reads
        back what it wrote expects.)"""
        import random
        out = []
        for pid in sorted({p for _, p in pk.pools}):
            out.extend((pid, o) for o in self.outputs.get(pid, ()))
        random.Random(seed).shuffle(out)
        out.sort(key=lambda c: c[1][1])
        return out

    def _unsat(self, node, cone, enabled, pins):
        t = self.t
        n = t.nodes[node]
        cons = "; ".join(describe(cone.constraints[i].constraint, t.names)
                         for i in enabled) or "(none)"
        fixed = ", ".join("%s=%d" % (v.name, _domain_value(val, v))
                          for _, v, val in pins) or "(none)"
        raise ScopeUnsatError(
            "no values for traversal %r (%s) satisfy the constraints in force: "
            "%s; with %s" % (n.path or "<root>", n.type_qname, cons, fixed))


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
        return type(e).__name__
    if isinstance(c, C.ConstraintExpr):
        return ex(c.expr)
    return type(c).__name__
