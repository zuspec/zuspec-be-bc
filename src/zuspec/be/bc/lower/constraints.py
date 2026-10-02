"""
constraints.py -- lower an ``ScSolveProblem``'s constraints to a dv-solve blob.

The scenario IR carries constraints as backend-neutral :class:`~zuspec.ir.core.
constraint.Constraint` items wrapping Layer-0 :class:`~zuspec.ir.core.expr.Expr`
trees (D3). This module translates them into a *relocatable dv-solve
``SolveProblem`` blob* (`problem_bytes`) plus the slot write-back map, so the
native engine (and the oracle, via the same blob) can solve the real constraint
system instead of the minimal ``slot = seed + var_id`` randomizer.

Building the blob uses the native ``dv-solve`` *growable* builder
(``SolveProblemBuilder``): it grows in blocks and ``finalize()``s to an exact-sized,
relocatable ``SolveProblem`` buffer -- so there is no fixed 64 KiB cap and no
manual byte-slicing. It is imported lazily so plain lowering stays dv-solve-free.

**Var/slot mapping.** A constraint names a field by its object storage *slot* (its
full-field index -- ``ExprRefField.index``, the same index procedural ``LD_FIELD``/
``ST_FIELD`` use), while the solver problem numbers variables by ``var_id`` (the
rand-only enumeration). These differ when non-rand fields are interleaved among the
rand fields. ``ScSolveVar.slot`` carries the object slot (``-1`` => the rand-only
layout where slot == var_id); we ``add_var(var_id)``, resolve a field reference via
``slot -> var_id``, and emit write-back ``{slot: var_id}``. A field reference whose
slot is not a declared rand var is rejected.

**Scope (M1).** ``ConstraintExpr`` over: arithmetic/bitwise/shift ``ExprBin``,
relational + logical ops, ``ExprUnary``, ``ExprBool``, chained ``ExprCompare``,
and ``ExprIn`` over ``ExprRange``/``ExprRangeList`` (including *disjoint* range
unions, via the native ``EXPR_IN_RANGES`` primitive). Disjunctions (``||``) work
too: an OR of simple comparisons folds to a native OR, and an OR that contains a
bounded range is reduced to one variable's interval union (``EXPR_IN_RANGES``),
since the solver does not propagate OR-of-ranges directly. A range disjunction
that spans *several* variables (``f0 in [..] || f1 in [..]``) is encoded with one
boolean *selector* per disjunct: ``(∨ s_i)`` plus ``s_i -> D_i`` clauses (a sound
half-reification using only OR-of-comparisons). ``ConstraintImplies``
(``A -> { body }``) and ``ConstraintIfElse`` (``if (c) {..} else {..}``) lower via
the clausal encoding ``!A || consequent`` -- one OR-of-comparisons clause per
consequent atom, a range splitting into two -- because ``expr_ite`` is not reliably
propagated by this solver. A conditional VALUE (``ExprIfExp``, ``c ? a : b``)
is another matter: it lowers to ``expr_ite``, which the solver bounds by both
branches while ``c`` is open. ``ConstraintDist`` lowers to the native ``add_dist``
(weighted value distribution). ``ConstraintSoft`` lowers to the native
``add_soft_constraint``: the solver relaxes (drops) a soft only when it conflicts
with the hard system, never violating a hard constraint; softs are assigned
increasing priority in declaration order so earlier softs are dropped last.
``ConstraintForeach`` (``foreach (arr[i]) { body }``) is unrolled over the rand
array's elements -- each element is its own solver var (the problem's ``arrays``
map records base slot -> element slots), and every ``arr[<const>]`` subscript
resolves to that element's field. ``ConstraintUnique`` (``unique { a, b, c }``)
lowers to the native ``add_all_different`` when the items are distinct vars that
fit the propagator's 16-var watch limit, else to pairwise ``!=``.
``ConstraintSolveBefore`` (``solve a before b``) is accepted and dropped: it is a
distribution-only ordering hint and dv-solve has no ordering input, so honouring it
is impossible and skipping it is sound (all hard constraints still hold).
Unsupported nodes are rejected cleanly.
"""
from __future__ import annotations

import dataclasses as _dc
from typing import Dict, Tuple

from zuspec.ir.core import expr as E
from zuspec.ir.core import constraint as C
from zuspec.ir.core import scenario as SC
from zuspec.ir.core.expr_phase2 import ExprIfExp as _IfExpP2

from .errors import LoweringError

_MASK64 = (1 << 64) - 1
_INT64_MAX = (1 << 63) - 1
_INT64_MIN = -(1 << 63)

# ir-core BinOp -> dv-solve BIN_* code.
#
# `NotEq` used to be missing here, expanded by `_binary` into `(a<b) | (a>b)`
# under the comment "BIN_NEQ is buggy in the native solver". No such bug is
# reachable: dv-solve's own copy of the same workaround was removed after
# verifying BIN_NEQ against the pre-change library (its G12), and the shapes
# THIS file can emit it in -- var-var, against a constant, against an arithmetic
# result, under AND, as an OR leaf, and pairwise for an all-different -- were
# each checked here, including the narrowed cases where a NEQ that propagated
# nothing would surface as a false unsat rather than a wrong value.
_BINOP: Dict[E.BinOp, int] = {
    E.BinOp.Add: 0, E.BinOp.Sub: 1, E.BinOp.Mult: 2, E.BinOp.Div: 3,
    E.BinOp.Mod: 4, E.BinOp.FloorDiv: 3,
    E.BinOp.BitAnd: 5, E.BinOp.BitOr: 6, E.BinOp.BitXor: 7,
    E.BinOp.LShift: 8, E.BinOp.RShift: 9,
    E.BinOp.Eq: 10, E.BinOp.NotEq: 11,
    E.BinOp.Lt: 12, E.BinOp.LtE: 13, E.BinOp.Gt: 14, E.BinOp.GtE: 15,
    E.BinOp.And: 16, E.BinOp.Or: 17,
}
# ir-core CmpOp -> ir-core BinOp (so ExprCompare chains reuse the ExprBin path).
_CMP_TO_BIN: Dict[E.CmpOp, E.BinOp] = {
    E.CmpOp.Eq: E.BinOp.Eq, E.CmpOp.NotEq: E.BinOp.NotEq, E.CmpOp.Lt: E.BinOp.Lt,
    E.CmpOp.LtE: E.BinOp.LtE, E.CmpOp.Gt: E.BinOp.Gt, E.CmpOp.GtE: E.BinOp.GtE,
}
_UNOP: Dict[E.UnaryOp, int] = {
    E.UnaryOp.USub: 0,   # UN_NEG
    E.UnaryOp.Not: 1,    # UN_NOT
    E.UnaryOp.Invert: 2, # UN_INVERT
}

# dv-solve BIN_* used directly (kept in sync with dv_solve.problem).
_BIN_LT, _BIN_GT, _BIN_AND, _BIN_OR = 12, 14, 16, 17


_ZERO = E.ExprConstant(value=0)


def _is_value(e) -> bool:
    """Is *e* a value rather than a condition built of comparisons and
    logical operators? As a condition, a value (a `bool` attribute) holds
    when it is nonzero."""
    if isinstance(e, E.ExprBin):
        return e.op not in _CMP_BINOPS and e.op not in (E.BinOp.And, E.BinOp.Or)
    if isinstance(e, E.ExprUnary):
        return e.op is not E.UnaryOp.Not
    return not isinstance(e, (E.ExprBool, E.ExprCompare, E.ExprIn, E.ExprCall))


def _through_comp(e) -> bool:
    """Is *e* a path through an action's ``comp`` (``comp.a.f``)?"""
    while isinstance(e, (E.ExprAttribute, E.ExprSubscript)):
        if (isinstance(e, E.ExprAttribute) and e.attr == "comp"
                and isinstance(e.value, (E.TypeExprRefSelf, E.TypeExprRefTraversed))):
            return True
        e = e.value
    return False


def _var_domain(v: SC.ScSolveVar) -> Tuple[int, int]:
    """[lo, hi] for a rand var from its width/signedness, clamped to int64."""
    w = v.width if v.width and v.width > 0 else 32
    if v.signed:
        lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
    else:
        lo, hi = 0, (1 << w) - 1
    return max(lo, _INT64_MIN), min(hi, _INT64_MAX)


def _slot_of(v: SC.ScSolveVar) -> int:
    """A rand var's object storage slot: its explicit ``slot`` or (unset) var_id.

    Constraints (and procedural code) address a field by its *full-field* index; the
    frontend records that as ``ScSolveVar.slot``. ``slot == -1`` means the rand-only
    layout where slot coincides with ``var_id``.
    """
    return v.slot if getattr(v, "slot", -1) is not None and v.slot >= 0 else v.var_id


# Comparison BinOps + the op you get by swapping the operands (`c < x` == `x > c`).
_CMP_BINOPS = frozenset((E.BinOp.Eq, E.BinOp.NotEq, E.BinOp.Lt, E.BinOp.LtE,
                         E.BinOp.Gt, E.BinOp.GtE))
_MIRROR = {E.BinOp.Lt: E.BinOp.Gt, E.BinOp.Gt: E.BinOp.Lt,
           E.BinOp.LtE: E.BinOp.GtE, E.BinOp.GtE: E.BinOp.LtE,
           E.BinOp.Eq: E.BinOp.Eq, E.BinOp.NotEq: E.BinOp.NotEq}


def _const_int(e):
    """The integer value of an ``ExprConstant`` node, else ``None``."""
    if isinstance(e, E.ExprConstant) and isinstance(e.value, (int, bool)):
        return int(e.value)
    return None


def _eval_const_int(e):
    """Fold a constant integer expression (``ExprConstant`` or ``+``/``-``/``*`` of
    constants), else ``None``. Used to resolve array subscript indices such as
    ``i + 1`` after the foreach index variable has been substituted."""
    v = _const_int(e)
    if v is not None:
        return v
    if isinstance(e, E.ExprBin):
        l = _eval_const_int(e.lhs)
        r = _eval_const_int(e.rhs)
        if l is None or r is None:
            return None
        if e.op is E.BinOp.Add:
            return l + r
        if e.op is E.BinOp.Sub:
            return l - r
        if e.op is E.BinOp.Mult:
            return l * r
    if isinstance(e, E.ExprUnary) and e.op is E.UnaryOp.USub:
        v = _eval_const_int(e.operand)
        return None if v is None else -v
    return None


def _iv_intersect(a, b):
    """Intersection of two interval lists (inclusive integer [lo, hi])."""
    out = []
    for l1, h1 in a:
        for l2, h2 in b:
            lo, hi = max(l1, l2), min(h1, h2)
            if lo <= hi:
                out.append((lo, hi))
    return out


def _is_or(e: E.Expr) -> bool:
    """True for a boolean-OR node (either the binary or the n-ary form)."""
    return ((isinstance(e, E.ExprBin) and e.op is E.BinOp.Or)
            or (isinstance(e, E.ExprBool) and e.op is E.BoolOp.Or))


def _cmp_intervals(op: E.BinOp, c: int, lo: int, hi: int):
    """Intervals within domain [lo, hi] satisfying ``var <op> c``."""
    if op is E.BinOp.Eq:
        return [(c, c)]
    if op is E.BinOp.NotEq:
        return [(lo, c - 1), (c + 1, hi)]
    if op is E.BinOp.Lt:
        return [(lo, c - 1)]
    if op is E.BinOp.LtE:
        return [(lo, c)]
    if op is E.BinOp.Gt:
        return [(c + 1, hi)]
    if op is E.BinOp.GtE:
        return [(c, hi)]
    return []


class _Translator:
    """Fills one dv-solve growable builder from a scenario ``ScSolveProblem``."""

    def __init__(self, problem: SC.ScSolveProblem):
        try:
            from dv_solve.builder import SolveProblemBuilder
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise LoweringError(
                "dv-solve is required to lower constraints to a solve blob "
                "(cannot import dv_solve.builder): %s" % exc)
        self._sp = SolveProblemBuilder()
        # A constraint names a field by its object slot (ExprRefField.index); map that
        # to the solver var_id we declare it under. add_var uses var_id.
        self._slot_to_vid = {}
        self._var_by_id = {}
        # Soft constraints are relaxed lowest-preference-first. dv-solve drops the
        # assumption with the *highest* priority value first, so to honour PSS's
        # "an earlier soft has higher priority" we assign increasing priority values
        # in declaration order: the first soft (priority 0) is dropped last.
        self._soft_priority = 0
        # array field slot -> ordered element object slots (for foreach / subscript).
        self._arrays = {int(base): list(elems)
                        for base, elems in (problem.arrays or {}).items()}
        for v in problem.vars:
            lo, hi = _var_domain(v)
            self._sp.add_var(v.var_id, v.width or 32, bool(v.signed), lo, hi)
            self._slot_to_vid[_slot_of(v)] = v.var_id
            self._var_by_id[v.var_id] = v
        # Auxiliary selector vars (for cross-variable range disjunction) are declared
        # with var_ids beyond every problem var so they never collide and are never
        # written back (writeback covers only problem.vars slots).
        self._next_aux_vid = max((v.var_id for v in problem.vars), default=-1) + 1

    def close(self) -> None:
        self._sp.destroy()

    # -- top level --------------------------------------------------------- #

    def add(self, expr: E.Expr) -> None:
        """Add a boolean constraint, splitting a top-level conjunction."""
        if isinstance(expr, E.ExprBin) and expr.op is E.BinOp.And:
            self.add(expr.lhs)
            self.add(expr.rhs)
            return
        if isinstance(expr, E.ExprBool) and expr.op is E.BoolOp.And:
            for v in expr.values:
                self.add(v)
            return
        # A disjunction whose ranges span more than one variable can't be a single
        # native OR (dv-solve doesn't propagate OR-of-ranges) nor one variable's
        # interval union. Encode it with per-disjunct boolean selectors instead.
        if _is_or(expr):
            disjuncts = self._disjuncts(expr)
            if self._needs_selector_disjunction(disjuncts):
                self._add_selector_disjunction(disjuncts)
                return
        self._sp.add_constraint(self._expr(expr))

    def bytes(self) -> bytes:
        return self._sp.finalize_bytes()

    # -- expression translation ------------------------------------------- #

    def _expr(self, e: E.Expr) -> int:
        sp = self._sp
        if isinstance(e, E.ExprConstant):
            v = e.value
            if not isinstance(v, (int, bool)):
                raise LoweringError("non-integer constant %r in a constraint" % (v,))
            return sp.expr_const(int(v) & _MASK64 if int(v) >= 0 else int(v))
        if isinstance(e, E.ExprRefField):
            # e.index is the object field slot; map it to the solver var_id.
            if e.index not in self._slot_to_vid:
                raise LoweringError(
                    "constraint references field slot %d, which is not a declared "
                    "rand var" % e.index)
            return sp.expr_var(self._slot_to_vid[e.index])
        if isinstance(e, E.ExprBin):
            if e.op is E.BinOp.Or:
                return self._or(self._disjuncts(e))
            return self._binary(e.op, e.lhs, e.rhs)
        if isinstance(e, E.ExprUnary):
            if e.op is E.UnaryOp.UAdd:
                return self._expr(e.operand)
            if e.op not in _UNOP:
                raise LoweringError("unsupported unary op %s" % e.op.name)
            return sp.expr_unary(_UNOP[e.op], self._expr(e.operand))
        if isinstance(e, E.ExprBool):
            if e.op is E.BoolOp.Or:
                return self._or(self._disjuncts(e))
            return self._fold(_BIN_AND, [self._expr(v) for v in e.values])
        if isinstance(e, E.ExprCompare):
            return self._compare(e)
        if isinstance(e, E.ExprIn):
            return self._in(e)
        if isinstance(e, E.ExprSubscript):
            # A constant-index array access -> the element's own solver var.
            return self._expr(self._resolve_subscript(e.value, e.slice))
        if isinstance(e, (E.ExprIfExp, _IfExpP2)):
            # A conditional VALUE (`c ? a : b`). The solver bounds it by both
            # branches while c is open and narrows c from it, which the
            # clausal encoding used for implications does not.
            return sp.expr_ite(self._expr(e.test), self._expr(e.body),
                               self._expr(e.orelse))
        if _through_comp(e):
            raise LoweringError(
                "a constraint reading a component attribute (comp.%s) is not "
                "supported by bc yet" % e.attr)
        raise LoweringError("unsupported constraint expression %s" % type(e).__name__)

    def _binary(self, op: E.BinOp, lhs: E.Expr, rhs: E.Expr) -> int:
        sp = self._sp
        if op not in _BINOP:
            raise LoweringError("unsupported binary op %s" % op.name)
        return sp.expr_binary(_BINOP[op], self._expr(lhs), self._expr(rhs))

    def _compare(self, e: E.ExprCompare) -> int:
        # a op0 b op1 c ... == (a op0 b) AND (b op1 c) AND ...
        operands = [e.left] + list(e.comparators)
        terms = []
        for i, op in enumerate(e.ops):
            if op not in _CMP_TO_BIN:
                raise LoweringError("unsupported comparison op %s" % op.name)
            terms.append(self._binary(_CMP_TO_BIN[op], operands[i], operands[i + 1]))
        return self._fold(_BIN_AND, terms)

    def _in(self, e: E.ExprIn) -> int:
        # Set membership. Encodings, chosen for what the native solver actually
        # propagates (verified): a single range -> expr_in_range; a single value ->
        # equality; a disjunction of pure values -> OR of equalities; and any union
        # that includes a true range -> the native EXPR_IN_RANGES primitive (OR of
        # expr_in_range / expr_in_set are NOT enforced, so we do not use them here).
        val = self._expr(e.value)
        ranges = self._ranges(e.container)
        if not ranges:
            raise LoweringError("empty membership set in a constraint")

        if len(ranges) == 1:
            lo, hi = ranges[0]
            if hi is None:
                return self._sp.expr_binary(_BINOP[E.BinOp.Eq], val, self._expr(lo))
            return self._sp.expr_in_range(val, self._expr(lo), self._expr(hi))

        if not any(hi is not None for _, hi in ranges):
            terms = [self._sp.expr_binary(_BINOP[E.BinOp.Eq], val, self._expr(lo))
                     for lo, _ in ranges]
            return self._fold(_BIN_OR, terms)

        # Union of ranges (and/or values): each value folds to a degenerate [v, v].
        pairs = []
        for lo, hi in ranges:
            lo_ref = self._expr(lo)
            pairs.append((lo_ref, lo_ref if hi is None else self._expr(hi)))
        return self._sp.expr_in_ranges(val, pairs)

    def _ranges(self, container: E.Expr):
        if isinstance(container, E.ExprRange):
            return [(container.lower, container.upper)]
        if isinstance(container, E.ExprRangeList):
            return [(r.lower, r.upper) for r in container.ranges]
        raise LoweringError(
            "unsupported membership container %s (want ExprRange/ExprRangeList)"
            % type(container).__name__)

    def _fold(self, code: int, terms) -> int:
        if not terms:
            raise LoweringError("empty operand list for a boolean fold")
        acc = terms[0]
        for t in terms[1:]:
            acc = self._sp.expr_binary(code, acc, t)
        return acc

    # -- disjunction ------------------------------------------------------- #

    def _disjuncts(self, e: E.Expr):
        """Flatten a (possibly nested) OR into its list of disjunct exprs."""
        out = []
        stack = [e]
        while stack:
            x = stack.pop()
            if isinstance(x, E.ExprBin) and x.op is E.BinOp.Or:
                stack.extend((x.rhs, x.lhs))
            elif isinstance(x, E.ExprBool) and x.op is E.BoolOp.Or:
                stack.extend(reversed(x.values))
            else:
                out.append(x)
        return out

    def _or(self, disjuncts) -> int:
        # The native solver enforces an OR of *simple comparisons* but NOT an OR that
        # contains a bounded range (a range is `x>=lo && x<=hi`, and OR-of-AND /
        # OR-of-in_range are not propagated -- verified). So if any disjunct is a
        # bounded range, reduce the whole disjunction to a single variable's interval
        # union and emit the native EXPR_IN_RANGES primitive; otherwise fold a plain OR.
        if any(self._is_bounded_range(d) for d in disjuncts):
            vid = None
            intervals = []
            for d in disjuncts:
                r = self._single_var_ranges(d)
                if r is None or (vid is not None and r[0] != vid):
                    # A cross-variable range disjunction is handled at the top level
                    # via boolean selectors (see add()/_add_selector_disjunction); it
                    # cannot be reduced to a single expr ref here, so nesting one
                    # inside a larger expression is unsupported.
                    raise LoweringError(
                        "cross-variable range disjunction nested inside another "
                        "expression is unsupported; lift it to a top-level constraint")
                vid = r[0]
                intervals += r[1]
            if not intervals:
                raise LoweringError("disjunction reduces to an empty value set")
            sp = self._sp
            pairs = [(sp.expr_const(lo), sp.expr_const(hi)) for lo, hi in intervals]
            return sp.expr_in_ranges(sp.expr_var(vid), pairs)
        return self._fold(_BIN_OR, [self._expr(d) for d in disjuncts])

    # -- cross-variable range disjunction (boolean selectors) -------------- #

    def _needs_selector_disjunction(self, disjuncts) -> bool:
        """True when a disjunction carries a bounded range that spans more than one
        variable (or an irreducible predicate) -- the case a native OR / single-var
        ``expr_in_ranges`` cannot express, requiring the selector encoding."""
        if not any(self._is_bounded_range(d) for d in disjuncts):
            return False                       # plain OR of comparisons: native fold
        vid = None
        for d in disjuncts:
            r = self._single_var_ranges(d)
            if r is None:
                return True                    # not a const-range predicate
            if vid is not None and r[0] != vid:
                return True                    # spans a second variable
            vid = r[0]
        return False                           # one variable: _or's expr_in_ranges

    def _add_selector_disjunction(self, disjuncts) -> None:
        """Encode ``D_0 || D_1 || ... || D_k`` over multiple variables with one
        boolean selector ``s_i in [0,1]`` per disjunct:

            (s_0>=1) || (s_1>=1) || ... || (s_k>=1)      -- at least one chosen
            s_i -> D_i         for each i                -- (s_i<=0) || <D_i clauses>

        Satisfiable iff some ``D_i`` holds, and every model satisfies at least one.
        Only OR-of-simple-comparisons clauses are emitted (dv-solve propagates those
        soundly -- verified). Selectors are auxiliary vars, never written back.
        """
        sp = self._sp
        sels = [self._new_selector_var() for _ in disjuncts]
        at_least_one = [sp.expr_binary(_BINOP[E.BinOp.GtE], sp.expr_var(s),
                                       sp.expr_const(1)) for s in sels]
        sp.add_constraint(self._fold(_BIN_OR, at_least_one))
        for s, d in zip(sels, disjuncts):
            not_s = sp.expr_binary(_BINOP[E.BinOp.LtE], sp.expr_var(s),
                                   sp.expr_const(0))          # !s_i  ==  (s_i <= 0)
            self._imply([not_s], [C.ConstraintExpr(expr=d)])

    def _new_selector_var(self) -> int:
        """Declare a fresh 1-bit [0,1] selector var and return its var_id."""
        vid = self._next_aux_vid
        self._next_aux_vid += 1
        self._sp.add_var(vid, 1, False, 0, 1)
        return vid

    def _is_bounded_range(self, d: E.Expr) -> bool:
        """True for disjunct forms that break a native OR (they carry a conjunction)."""
        if isinstance(d, E.ExprIn):
            return any(hi is not None for _, hi in self._ranges(d.container))
        if isinstance(d, E.ExprBin) and d.op is E.BinOp.And:
            return True
        if isinstance(d, E.ExprBool) and d.op is E.BoolOp.And:
            return True
        if isinstance(d, E.ExprCompare) and len(d.ops) >= 2:
            return True
        return False

    # -- single-variable interval analysis (for range disjunctions) -------- #

    def _single_var_ranges(self, e: E.Expr):
        """(var_id, [(lo, hi), ...]) if ``e`` is a single-var, const-bounded predicate.

        Handles comparisons, ``in`` ranges, ``&&``/``||`` and comparison chains over
        one variable (intersecting for AND, unioning for OR). Returns ``None`` when
        ``e`` spans another variable or uses a non-constant bound.
        """
        if isinstance(e, E.ExprBin):
            if e.op is E.BinOp.And:
                return self._combine([e.lhs, e.rhs], intersect=True)
            if e.op is E.BinOp.Or:
                return self._combine([e.lhs, e.rhs], intersect=False)
            if e.op in _CMP_BINOPS:
                return self._cmp_ranges(e.op, e.lhs, e.rhs)
            return None
        if isinstance(e, E.ExprBool):
            return self._combine(e.values, intersect=(e.op is E.BoolOp.And))
        if isinstance(e, E.ExprCompare):
            operands = [e.left] + list(e.comparators)
            parts = []
            for i, op in enumerate(e.ops):
                b = _CMP_TO_BIN.get(op)
                if b is None:
                    return None
                parts.append(self._cmp_ranges(b, operands[i], operands[i + 1]))
            return self._merge(parts, intersect=True)
        if isinstance(e, E.ExprIn):
            vid = self._field_vid(e.value)
            if vid is None:
                return None
            ivs = self._const_intervals(e.container)
            return None if ivs is None else self._clamp(vid, ivs)
        return None

    def _combine(self, exprs, intersect: bool):
        return self._merge([self._single_var_ranges(x) for x in exprs], intersect)

    def _merge(self, parts, intersect: bool):
        if any(p is None for p in parts):
            return None
        vid = parts[0][0]
        if any(p[0] != vid for p in parts):
            return None
        ivs = parts[0][1]
        for p in parts[1:]:
            ivs = _iv_intersect(ivs, p[1]) if intersect else ivs + p[1]
        return (vid, ivs)

    def _cmp_ranges(self, op: E.BinOp, lhs: E.Expr, rhs: E.Expr):
        lv, lc = self._field_vid(lhs), _const_int(lhs)
        rv, rc = self._field_vid(rhs), _const_int(rhs)
        if lv is not None and rc is not None:
            lo, hi = _var_domain(self._var_by_id[lv])
            return self._clamp(lv, _cmp_intervals(op, rc, lo, hi))
        if rv is not None and lc is not None:
            lo, hi = _var_domain(self._var_by_id[rv])
            return self._clamp(rv, _cmp_intervals(_MIRROR[op], lc, lo, hi))
        return None

    def _clamp(self, vid: int, ivs):
        lo, hi = _var_domain(self._var_by_id[vid])
        return (vid, _iv_intersect(ivs, [(lo, hi)]))

    def _field_vid(self, e: E.Expr):
        if isinstance(e, E.ExprRefField):
            return self._slot_to_vid.get(e.index)
        return None

    def _const_intervals(self, container: E.Expr):
        out = []
        for lo, hi in self._ranges(container):
            lc = _const_int(lo)
            if lc is None:
                return None
            hc = lc if hi is None else _const_int(hi)
            if hc is None:
                return None
            out.append((lc, hc))
        return out

    # -- constraint-item dispatch (implies / if-else) ---------------------- #

    def add_constraint(self, con) -> None:
        """Lower one structured constraint item into the problem."""
        if isinstance(con, C.ConstraintExpr):
            self.add(con.expr)
        elif isinstance(con, C.ConstraintImplies):
            self._imply(self._neg_literals(con.antecedent), con.body)
        elif isinstance(con, C.ConstraintIfElse):
            # if (cond) { then } else { else } == (cond -> then) && (!cond -> else).
            self._imply(self._neg_literals(con.cond), con.then_body)
            if con.else_body:
                # else antecedent is !cond, so its negation is cond -> the clause
                # prefix is the literals whose disjunction equals cond.
                self._imply(self._pos_literals(con.cond), con.else_body)
        elif isinstance(con, C.ConstraintForeach):
            self._add_foreach(con)
        elif isinstance(con, C.ConstraintUnique):
            self._add_unique(con)
        elif isinstance(con, C.ConstraintDist):
            self._add_dist(con)
        elif isinstance(con, C.ConstraintSoft):
            self._add_soft(con)
        elif isinstance(con, C.ConstraintSolveBefore):
            self._add_solve_before(con)
        else:
            raise LoweringError(
                "unsupported constraint form %s" % type(con).__name__)

    # dv-solve's AllDifferent propagator watches at most this many vars.
    _MAX_ALLDIFF = 16

    def _add_unique(self, con) -> None:
        """``unique { a, b, c };`` -- all listed members take distinct values.

        Preferred lowering is dv-solve's native ``add_all_different`` (global Hall-
        interval propagation), used when every item resolves to a distinct solver
        var and the set fits the propagator's watch limit. Otherwise -- more than
        16 items, a repeated var, or a non-variable item -- it falls back to
        **pairwise ``!=``** (``a!=b && a!=c && ...``), each ``x!=y`` emitted as the
        verified-sound ``(x<y) | (x>y)``. Both encodings are correct at every width
        (the historical width-32 ``add_all_different`` unsoundness -- it read tier-1
        bounds as raw int32 -- was fixed in ``_fire_all_different_32``).
        """
        items = list(con.items)
        if len(items) < 2:
            return                      # unique over 0 or 1 items is vacuously true
        vids = [self._item_vid(it) for it in items]
        if (all(v is not None for v in vids)
                and len(set(vids)) == len(vids)
                and len(vids) <= self._MAX_ALLDIFF):
            self._sp.add_all_different(vids)
            return
        # Fallback: pairwise inequality.
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                self._sp.add_constraint(
                    self._binary(E.BinOp.NotEq, items[i], items[j]))

    def _item_vid(self, e: E.Expr):
        """The solver var_id an item refers to (field or constant-index subscript),
        else ``None`` for anything that is not a plain variable reference."""
        if isinstance(e, E.ExprSubscript):
            e = self._resolve_subscript(e.value, e.slice)
        if isinstance(e, E.ExprRefField):
            return self._slot_to_vid.get(e.index)
        return None

    def _add_foreach(self, con) -> None:
        """Iterative constraint ``foreach (arr[i]) { body }`` -- unrolled per element.

        A rand array is flattened into one solver var per element (recorded in the
        problem's ``arrays`` map: array base slot -> ordered element slots). For each
        index ``i`` in ``[0, size)`` the body is rewritten with ``i`` substituted for
        the index variable and every ``arr[<const>]`` resolved to that element's own
        ``ExprRefField``, then lowered through the normal constraint machinery. This
        mirrors the compile-time (`sw_solve`) and Python-runtime unrolling.
        """
        if not isinstance(con.array, E.ExprRefField):
            raise LoweringError("foreach array must be a field reference")
        elems = self._arrays.get(con.array.index)
        if elems is None:
            raise LoweringError(
                "foreach iterates slot %d, which is not a declared rand array"
                % con.array.index)
        if not con.index_var:
            raise LoweringError("foreach constraint has no index variable")
        for i in range(len(elems)):
            for item in con.body:
                self.add_constraint(self._foreach_rewrite(item, con.index_var, i))

    def _foreach_rewrite(self, node, idx_name: str, idx_val: int):
        """Copy a constraint/expr subtree, resolving one foreach iteration.

        Replaces the loop variable (``ExprRefLocal(idx_name)``) with the concrete
        integer ``idx_val`` and resolves any ``array[<const>]`` subscript to the
        element's ``ExprRefField``. Recurses structurally over dataclass nodes so it
        reaches nested constraint bodies (implies/if/inner foreach) too.
        """
        if isinstance(node, E.ExprRefLocal) and node.name == idx_name:
            return E.ExprConstant(value=idx_val)
        if isinstance(node, E.ExprSubscript):
            value = self._foreach_rewrite(node.value, idx_name, idx_val)
            sl = self._foreach_rewrite(node.slice, idx_name, idx_val)
            return self._resolve_subscript(value, sl)
        if _dc.is_dataclass(node) and not isinstance(node, type):
            repl = {}
            for f in _dc.fields(node):
                val = getattr(node, f.name)
                if isinstance(val, (E.Expr, C.Constraint)):
                    repl[f.name] = self._foreach_rewrite(val, idx_name, idx_val)
                elif isinstance(val, list) and any(
                        isinstance(x, (E.Expr, C.Constraint)) for x in val):
                    repl[f.name] = [
                        self._foreach_rewrite(x, idx_name, idx_val)
                        if isinstance(x, (E.Expr, C.Constraint)) else x
                        for x in val]
            new_node = _dc.replace(node, **repl) if repl else node
            # Fold a now-constant arithmetic subtree (e.g. `i * 10` once `i` is
            # substituted) so the solver sees a literal rather than const-op-const,
            # which its compiler does not reduce.
            if isinstance(new_node, E.Expr):
                folded = _eval_const_int(new_node)
                if folded is not None:
                    return E.ExprConstant(value=folded)
            return new_node
        return node

    def _resolve_subscript(self, value: E.Expr, slice_expr: E.Expr) -> E.Expr:
        """``arr[<const>]`` -> the element's ``ExprRefField`` (via the arrays map)."""
        if not isinstance(value, E.ExprRefField):
            raise LoweringError(
                "array subscript base must be a rand array field reference")
        elems = self._arrays.get(value.index)
        if elems is None:
            raise LoweringError(
                "subscript over slot %d, which is not a declared rand array"
                % value.index)
        k = _eval_const_int(slice_expr)
        if k is None:
            raise LoweringError(
                "array subscript index must be constant after foreach unrolling")
        if k < 0 or k >= len(elems):
            raise LoweringError(
                "array index %d out of range [0, %d)" % (k, len(elems)))
        return E.ExprRefField(base=value.base, index=elems[k])

    def _add_solve_before(self, con) -> None:
        """``solve <before> before <after>;`` -- a solver *ordering* hint.

        Per PSS, ``solve...before`` affects only the probability distribution of the
        chosen solution, never the set of legal solutions. dv-solve exposes no
        variable-ordering / decision-priority input (neither in the problem blob nor
        in ``SolveOpts`` -- decision order is an internal MRV/fair-pick heuristic), so
        the hint cannot be honoured and is dropped. This is sound: every hard
        constraint is still enforced, and because the oracle and the native engine
        solve the identical blob they remain in lock-step. We accept it (rather than
        reject) so scenarios that use ``solve...before`` lower and run. The lifted
        form (``ScSolveProblem.solve_before``) is likewise ignored by
        :func:`build_solve_blob`.
        """
        return

    def _add_soft(self, con) -> None:
        """Soft (relaxable) constraint: ``soft expr;``.

        Lowered to dv-solve's ``add_soft_constraint``, which gates the constraint
        behind a pinned assumption var and relaxes it (drops the whole soft) when it
        conflicts with the hard constraint system -- never violating a hard one. A
        soft is *honoured* when satisfiable for simple ``field <op> const`` forms;
        richer bodies (ranges, disjunctions) are safely dropped on conflict but may
        be ignored when satisfiable (a missed bias, never an unsound result). Both
        the oracle and the native engine solve the identical lowered blob, so the
        relaxation decision is deterministic and identical on both sides.
        """
        priority = self._soft_priority
        self._soft_priority += 1
        self._sp.add_soft_constraint(self._expr(con.expr), priority)

    def _add_dist(self, con) -> None:
        """Weighted value distribution: ``target dist { rng := w, ... }``."""
        vid = self._field_vid(con.target)
        if vid is None:
            raise LoweringError("dist target must be a declared rand field")
        entries = []
        for dw in con.weights:
            lo, hi = self._dist_range(dw.rng)
            w = 1 if dw.weight is None else _const_int(dw.weight)
            if w is None:
                raise LoweringError("dist weight must be a constant")
            # DistWeight.per_value selects `:/` (split across the range) vs `:=` (each
            # value); passed through to the native DistEntry (affects the weighting
            # distribution only, not the value set).
            entries.append({"lo": lo, "hi": hi, "weight": w,
                            "is_per_value": bool(dw.per_value)})
        if not entries:
            raise LoweringError("dist constraint has no weighted entries")
        self._sp.add_dist(vid, entries)

    def _dist_range(self, rng: E.Expr):
        if isinstance(rng, E.ExprRange):
            lo = _const_int(rng.lower)
            hi = lo if rng.upper is None else _const_int(rng.upper)
        elif isinstance(rng, E.ExprConstant):
            lo = hi = _const_int(rng)
        else:
            lo = hi = None
        if lo is None or hi is None:
            raise LoweringError("dist range bounds must be constants")
        return lo, hi

    def _imply(self, neg_lits, body) -> None:
        """Emit clauses for ``A -> body`` where ``neg_lits`` are the literals of !A.

        Each clause is ``(!A) OR consequent_literal``, which the solver enforces as
        an OR of simple comparisons (expr_ite is unreliable here). A range in the
        consequent (a conjunction ``x>=lo && x<=hi``) splits into two such clauses.
        """
        for item in body:
            if not isinstance(item, C.ConstraintExpr):
                raise LoweringError(
                    "implication/if body must be plain boolean constraints (got %s)"
                    % type(item).__name__)
            for clause in self._consequent_clauses(item.expr):
                self._sp.add_constraint(self._fold(_BIN_OR, neg_lits + clause))

    def _neg_comparison(self, op: E.BinOp, lhs: E.Expr, rhs: E.Expr):
        """Literals whose disjunction equals ``!(lhs op rhs)``."""
        sp = self._sp
        l, r = self._expr(lhs), self._expr(rhs)
        if op is E.BinOp.Eq:
            return [sp.expr_binary(_BIN_LT, l, r), sp.expr_binary(_BIN_GT, l, r)]
        if op is E.BinOp.NotEq:
            return [sp.expr_binary(_BINOP[E.BinOp.Eq], l, r)]
        neg = {E.BinOp.Lt: E.BinOp.GtE, E.BinOp.LtE: E.BinOp.Gt,
               E.BinOp.Gt: E.BinOp.LtE, E.BinOp.GtE: E.BinOp.Lt}[op]
        return [sp.expr_binary(_BINOP[neg], l, r)]

    def _neg_literals(self, a: E.Expr):
        """Literals whose disjunction equals ``!a`` (a = a conjunction of atoms)."""
        if isinstance(a, E.ExprBin):
            if a.op is E.BinOp.And:
                return self._neg_literals(a.lhs) + self._neg_literals(a.rhs)
            if a.op in _CMP_BINOPS:
                return self._neg_comparison(a.op, a.lhs, a.rhs)
        if isinstance(a, E.ExprBool) and a.op is E.BoolOp.And:
            out = []
            for v in a.values:
                out += self._neg_literals(v)
            return out
        if isinstance(a, E.ExprCompare):
            out = []
            operands = [a.left] + list(a.comparators)
            for i, op in enumerate(a.ops):
                b = _CMP_TO_BIN.get(op)
                if b is None:
                    raise LoweringError("unsupported comparison op %s" % op.name)
                out += self._neg_comparison(b, operands[i], operands[i + 1])
            return out
        if isinstance(a, E.ExprUnary) and a.op is E.UnaryOp.Not:
            return self._one_clause(a.operand)          # !!e is e
        if _is_value(a):
            # A bare boolean holds when nonzero: its negation is `a == 0`.
            return self._neg_comparison(E.BinOp.NotEq, a, _ZERO)
        if isinstance(a, E.ExprIn):
            rngs = self._ranges(a.container)
            if len(rngs) != 1 or rngs[0][1] is None:
                raise LoweringError("antecedent 'in' must be a single range")
            lo, hi = rngs[0]
            x = self._expr(a.value)
            return [self._sp.expr_binary(_BIN_LT, x, self._expr(lo)),
                    self._sp.expr_binary(_BIN_GT, x, self._expr(hi))]
        raise LoweringError(
            "unsupported antecedent %s (want a conjunction of comparisons/ranges)"
            % type(a).__name__)

    def _pos_literals(self, a: E.Expr):
        """Literals whose disjunction equals ``a`` (a = a disjunction of comparisons)."""
        if isinstance(a, E.ExprBin):
            if a.op is E.BinOp.Or:
                return self._pos_literals(a.lhs) + self._pos_literals(a.rhs)
            if a.op in _CMP_BINOPS:
                return self._one_clause(a)
        if isinstance(a, E.ExprBool) and a.op is E.BoolOp.Or:
            out = []
            for v in a.values:
                out += self._pos_literals(v)
            return out
        if isinstance(a, E.ExprCompare) and len(a.ops) == 1:
            return self._one_clause(a)
        if (isinstance(a, E.ExprUnary) and a.op is E.UnaryOp.Not) or _is_value(a):
            return self._one_clause(a)
        raise LoweringError(
            "if/else condition %s is unsupported with a non-empty else (want a "
            "comparison or a disjunction of comparisons)" % type(a).__name__)

    def _consequent_clauses(self, e: E.Expr):
        """A conjunction of clauses (each a list of literals) for a consequent expr."""
        if isinstance(e, E.ExprBin) and e.op is E.BinOp.And:
            return self._consequent_clauses(e.lhs) + self._consequent_clauses(e.rhs)
        if isinstance(e, E.ExprBool) and e.op is E.BoolOp.And:
            out = []
            for v in e.values:
                out += self._consequent_clauses(v)
            return out
        if isinstance(e, E.ExprIn):
            rngs = self._ranges(e.container)
            x = self._expr(e.value)
            if len(rngs) == 1:
                lo, hi = rngs[0]
                if hi is None:
                    return [[self._sp.expr_binary(_BINOP[E.BinOp.Eq], x, self._expr(lo))]]
                return [[self._sp.expr_binary(_BINOP[E.BinOp.GtE], x, self._expr(lo))],
                        [self._sp.expr_binary(_BINOP[E.BinOp.LtE], x, self._expr(hi))]]
            # A pure discrete union `x in {a, b, c}` is one OR-of-equalities clause.
            if all(hi is None for _, hi in rngs):
                return [[self._sp.expr_binary(_BINOP[E.BinOp.Eq], x, self._expr(lo))
                         for lo, _ in rngs]]
            raise LoweringError(
                "consequent 'in' with a union of ranges is unsupported (a disjunction "
                "of conjunctions has no CNF-clause form here)")
        if isinstance(e, E.ExprCompare):
            operands = [e.left] + list(e.comparators)
            out = []
            for i, op in enumerate(e.ops):
                b = _CMP_TO_BIN.get(op)
                if b is None:
                    raise LoweringError("unsupported comparison op %s" % op.name)
                out.append(self._one_clause(E.ExprBin(lhs=operands[i], op=b,
                                                      rhs=operands[i + 1])))
            return out
        # A single clause (a comparison or a disjunction of comparisons).
        return [self._one_clause(e)]

    def _one_clause(self, e: E.Expr):
        """Literals of a single clause: a comparison or a disjunction of comparisons."""
        if isinstance(e, E.ExprBin):
            if e.op is E.BinOp.Or:
                return self._one_clause(e.lhs) + self._one_clause(e.rhs)
            if e.op is E.BinOp.NotEq:
                # Two literals, not one BIN_NEQ -- and NOT the old "BIN_NEQ is
                # buggy" workaround, which is gone. A clause here is a list that
                # the caller OR-folds with the antecedent's literals, and
                # splitting `!=` into its two halves at that level is what lets
                # the whole clause stay a flat disjunction of simple
                # comparisons, which is the form `_imply` documents dv-solve
                # propagates soundly.
                l, r = self._expr(e.lhs), self._expr(e.rhs)
                return [self._sp.expr_binary(_BIN_LT, l, r),
                        self._sp.expr_binary(_BIN_GT, l, r)]
            if e.op in _CMP_BINOPS:
                return [self._sp.expr_binary(_BINOP[e.op], self._expr(e.lhs),
                                             self._expr(e.rhs))]
        if isinstance(e, E.ExprBool) and e.op is E.BoolOp.Or:
            out = []
            for v in e.values:
                out += self._one_clause(v)
            return out
        if isinstance(e, E.ExprUnary) and e.op is E.UnaryOp.Not:
            return self._neg_literals(e.operand)
        if _is_value(e):
            return self._neg_comparison(E.BinOp.Eq, e, _ZERO)   # e != 0
        raise LoweringError(
            "cannot place %s in a disjunctive clause (want comparisons)"
            % type(e).__name__)


def build_solve_blob(problem: SC.ScSolveProblem) -> Tuple[bytes, Dict[int, int]]:
    """Translate *problem*'s constraints to a (problem_bytes, writeback_slots) pair.

    ``writeback_slots`` maps object field slot -> solver var_id (M1: slot == var_id).
    Raises :class:`LoweringError` on an unsupported constraint form.

    ``problem.solve_before`` (the lifted ``solve...before`` ordering pairs) is
    intentionally ignored: it is a distribution-only hint and dv-solve exposes no
    ordering input, so it cannot affect the solution set (see ``_add_solve_before``).
    """
    tr = _Translator(problem)
    try:
        for con in problem.constraints:
            tr.add_constraint(con)
        blob = tr.bytes()
    finally:
        tr.close()
    # Native write-back: object field slot -> solver var_id (== var_id in the
    # rand-only layout, but distinct when non-rand fields are interleaved).
    writeback_slots = {_slot_of(v): v.var_id for v in problem.vars}
    return blob, writeback_slots
