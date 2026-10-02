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

    def enter_scope(self, scope: int) -> None:
        """Entry to an activity block: its traversed handles are reset."""
        t = self.t
        for st in t.scope_sites[scope]:
            self.fired[st] = False
            self._reset_node(t.sites[st][1])
        for sc in t.scope_nested[scope]:
            self.entered[sc] = False
        self.entered[scope] = True

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
        pins = []
        for i, v in enumerate(cone.vars):
            if v.comp_read is not None:
                # A component attribute: fixed once the tree is constructed.
                pins.append((i, v, self.cobj.get_field(v.comp_read)))
                continue
            if v.loop_local is not None and v.node == node:
                # A loop's index, at the traversal it constrains: the
                # counter's value in this iteration. (Before, it is free; after,
                # committed with the node's values.)
                pins.append((i, v, _loop_counter(frame, v)))
                continue
            if v.node == node and v.rand:
                continue
            if self.started[v.node] or self.committed[v.node]:
                if v.rand and not self.committed[v.node]:
                    continue                 # started, not yet solved: free
                pins.append((i, v, self.obj.get_field(v.slot)))
            elif not v.rand:
                init = t.init_values.get(v.slot, 0)
                if init is not None:
                    pins.append((i, v, init))
        from dv_solve.ctx import SOLVE_OK, SOLVE_TIMEOUT, CompileUnsatError
        blob = t.blob(cone, enabled)
        try:
            with self.cache.session(blob) as ctx:
                for i, v, val in pins:
                    if not ctx.pin(i, _domain_value(val, v)):
                        self._unsat(node, cone, enabled, pins)
                rc = self.cache.solve(ctx, seed & _MASK64)
                if rc == SOLVE_TIMEOUT:
                    n = t.nodes[node]
                    raise SolveBudgetError(
                        "choosing values for traversal %r (%s): %s"
                        % (n.path or "<root>", n.type_qname,
                           self.cache.budget_message()))
                if rc != SOLVE_OK:
                    self._unsat(node, cone, enabled, pins)
                for i, v in enumerate(cone.vars):
                    if v.node == node and v.rand:
                        self.obj.set_field(v.slot, ctx.get_value(i) & _MASK64)
        except CompileUnsatError:
            self._unsat(node, cone, enabled, pins)
        self.commit(node, site)

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
