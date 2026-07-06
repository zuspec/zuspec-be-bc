"""Constraint lowering: ScSolveProblem constraints -> dv-solve blob.

Each test builds a scenario ``ScSolveProblem`` (rand vars + a Layer-0 constraint
expr), lowers it with :func:`build_solve_blob`, then solves the produced blob with
the real dv-solve ``SolveCtx`` and asserts the solution satisfies the constraint.
That validates the translator end-to-end against the actual solver (the native
engine feeds the identical blob to the identical solver).
"""

import ctypes

import pytest

pytest.importorskip("dv_solve.problem")

from zuspec.ir.core import expr as E
from zuspec.ir.core import constraint as C
from zuspec.ir.core import scenario as SC

from zuspec.be.bc.lower.constraints import build_solve_blob
from zuspec.be.bc.lower.errors import LoweringError

from dv_solve.ctx import SolveCtx, SOLVE_OK


# --- IR builders ---------------------------------------------------------- #

def V(i, width=32, signed=False):
    return SC.ScSolveVar(name="f%d" % i, var_id=i, width=width, signed=signed)

def FIELD(i):   return E.ExprRefField(base=E.TypeExprRefSelf(), index=i)
def K(n):       return E.ExprConstant(value=n)
def BIN(l, op, r): return E.ExprBin(lhs=l, op=op, rhs=r)
def CE(expr):   return C.ConstraintExpr(expr=expr)

def _problem(vars_, *constraints):
    return SC.ScSolveProblem(vars=list(vars_),
                             constraints=[CE(c) for c in constraints],
                             writeback={v.name: v.var_id for v in vars_})


def _problem_c(vars_, *items):
    # Like _problem but items are already Constraint objects (implies/if-else).
    return SC.ScSolveProblem(vars=list(vars_), constraints=list(items),
                             writeback={v.name: v.var_id for v in vars_})


def _solve(problem, seed=7):
    """Lower + solve; return {var_id: value} and the writeback_slots map."""
    blob, wb_slots = build_solve_blob(problem)
    raw = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
    ctx = SolveCtx(raw)
    try:
        assert ctx.solve(seed=seed) == SOLVE_OK
        vals = {v.var_id: ctx.get_value(v.var_id) for v in problem.vars}
    finally:
        ctx.destroy()
    return vals, wb_slots


# --- tests ---------------------------------------------------------------- #

def test_sum_and_lower_bound():
    # width 8 so x+y stays under 256 (bit-width arithmetic wraps mod 2^w, which is
    # correct PSS semantics but would make x+y == 42 hold only mod 2^32 at width 32).
    p = _problem([V(0, width=8), V(1, width=8)],
                 BIN(BIN(FIELD(0), E.BinOp.Add, FIELD(1)), E.BinOp.Eq, K(42)),
                 BIN(FIELD(0), E.BinOp.GtE, K(10)))
    vals, wb = _solve(p)
    assert vals[0] + vals[1] == 42 and vals[0] >= 10
    assert wb == {0: 0, 1: 1}                    # slot == var_id


def test_top_level_and_is_split():
    # A single conjunction constraint must be honored as two constraints.
    p = _problem([V(0)],
                 BIN(BIN(FIELD(0), E.BinOp.GtE, K(5)),
                     E.BinOp.And,
                     BIN(FIELD(0), E.BinOp.LtE, K(8))))
    vals, _ = _solve(p)
    assert 5 <= vals[0] <= 8


def test_range_membership():
    # f0 in [10..20]
    rng = E.ExprIn(value=FIELD(0),
                   container=E.ExprRange(lower=K(10), upper=K(20)))
    vals, _ = _solve(_problem([V(0)], rng))
    assert 10 <= vals[0] <= 20


def test_discrete_value_set_membership():
    # f0 in {3, 7, 42} (single values) -> OR of equalities.
    rl = E.ExprRangeList(ranges=[E.ExprRange(lower=K(3), upper=None),
                                 E.ExprRange(lower=K(7), upper=None),
                                 E.ExprRange(lower=K(42), upper=None)])
    for seed in (1, 2, 5, 9):
        vals, _ = _solve(_problem([V(0)], E.ExprIn(value=FIELD(0), container=rl)), seed=seed)
        assert vals[0] in (3, 7, 42)


def test_disjoint_range_union():
    # f0 in {[0..2], [100..102]} -- native EXPR_IN_RANGES enforces the union.
    rl = E.ExprRangeList(ranges=[E.ExprRange(lower=K(0), upper=K(2)),
                                 E.ExprRange(lower=K(100), upper=K(102))])
    for seed in (1, 2, 5, 9, 17, 23):
        vals, _ = _solve(_problem([V(0)], E.ExprIn(value=FIELD(0), container=rl)), seed=seed)
        assert vals[0] in (0, 1, 2, 100, 101, 102)


def test_mixed_value_and_range_union():
    # f0 in {5, [10..20], 100} -- values fold to degenerate [v, v] in the union.
    rl = E.ExprRangeList(ranges=[E.ExprRange(lower=K(5), upper=None),
                                 E.ExprRange(lower=K(10), upper=K(20)),
                                 E.ExprRange(lower=K(100), upper=None)])
    allowed = {5, 100} | set(range(10, 21))
    for seed in (1, 2, 5, 9, 17, 23, 31):
        vals, _ = _solve(_problem([V(0)], E.ExprIn(value=FIELD(0), container=rl)), seed=seed)
        assert vals[0] in allowed


def test_not_equal_uses_lt_or_gt():
    # f0 in [4..6] and f0 != 5  ->  f0 in {4, 6}
    p = _problem([V(0, width=8)],
                 E.ExprIn(value=FIELD(0), container=E.ExprRange(lower=K(4), upper=K(6))),
                 BIN(FIELD(0), E.BinOp.NotEq, K(5)))
    for seed in (1, 2, 3, 9, 17):
        vals, _ = _solve(p, seed=seed)
        assert vals[0] in (4, 6)


def test_bool_or():
    # (f0 == 1) OR (f0 == 2)
    e = E.ExprBool(op=E.BoolOp.Or, values=[
        BIN(FIELD(0), E.BinOp.Eq, K(1)),
        BIN(FIELD(0), E.BinOp.Eq, K(2))])
    vals, _ = _solve(_problem([V(0, width=8)], e))
    assert vals[0] in (1, 2)


def test_chained_compare():
    # 10 <= f0 <= 20  as a single ExprCompare
    cmp = E.ExprCompare(left=K(10), ops=[E.CmpOp.LtE, E.CmpOp.LtE],
                        comparators=[FIELD(0), K(20)])
    vals, _ = _solve(_problem([V(0)], cmp))
    assert 10 <= vals[0] <= 20


def test_arithmetic_op():
    # f0 * 2 == f1, f0 in [1..10]
    p = _problem([V(0), V(1)],
                 BIN(BIN(FIELD(0), E.BinOp.Mult, K(2)), E.BinOp.Eq, FIELD(1)),
                 E.ExprIn(value=FIELD(0), container=E.ExprRange(lower=K(1), upper=K(10))))
    vals, _ = _solve(p)
    assert vals[1] == vals[0] * 2 and 1 <= vals[0] <= 10


def test_slot_differs_from_var_id():
    # Non-rand field at slot 0; rand x at slot 1 (var_id 0), y at slot 2 (var_id 1).
    # The constraint addresses fields by slot (1, 2); the solver problem uses var_ids.
    x = SC.ScSolveVar(name="x", var_id=0, slot=1, width=8)
    y = SC.ScSolveVar(name="y", var_id=1, slot=2, width=8)
    p = SC.ScSolveProblem(
        vars=[x, y],
        constraints=[CE(BIN(BIN(FIELD(1), E.BinOp.Add, FIELD(2)), E.BinOp.Eq, K(30))),
                     CE(BIN(FIELD(1), E.BinOp.GtE, K(5)))],
        writeback={"x": 0, "y": 1})
    blob, wb = build_solve_blob(p)
    assert wb == {1: 0, 2: 1}                     # {slot: var_id}
    raw = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
    ctx = SolveCtx(raw)
    try:
        assert ctx.solve(seed=4) == SOLVE_OK
        vx, vy = ctx.get_value(0), ctx.get_value(1)   # get_value is by var_id
    finally:
        ctx.destroy()
    assert vx + vy == 30 and vx >= 5


def _IN(i, lo, hi):
    return E.ExprIn(value=FIELD(i), container=E.ExprRange(lower=K(lo), upper=K(hi)))


def test_or_of_in_range_same_var():
    # f0 in [0..10] || f0 in [20..30]  -> reduced to EXPR_IN_RANGES over f0.
    e = E.ExprBool(op=E.BoolOp.Or, values=[_IN(0, 0, 10), _IN(0, 20, 30)])
    for seed in (1, 2, 5, 9, 17, 23):
        vals, _ = _solve(_problem([V(0)], e), seed=seed)
        assert 0 <= vals[0] <= 10 or 20 <= vals[0] <= 30


def test_or_of_and_ranges_same_var():
    # (f0 >= 0 && f0 <= 5) || (f0 >= 100 && f0 <= 105)
    d1 = BIN(BIN(FIELD(0), E.BinOp.GtE, K(0)), E.BinOp.And, BIN(FIELD(0), E.BinOp.LtE, K(5)))
    d2 = BIN(BIN(FIELD(0), E.BinOp.GtE, K(100)), E.BinOp.And, BIN(FIELD(0), E.BinOp.LtE, K(105)))
    e = BIN(d1, E.BinOp.Or, d2)
    for seed in (1, 3, 7, 12, 25):
        vals, _ = _solve(_problem([V(0)], e), seed=seed)
        assert 0 <= vals[0] <= 5 or 100 <= vals[0] <= 105


def test_or_of_range_and_value_same_var():
    # f0 in [0..10] || f0 == 50
    e = E.ExprBool(op=E.BoolOp.Or, values=[
        _IN(0, 0, 10), BIN(FIELD(0), E.BinOp.Eq, K(50))])
    for seed in (1, 2, 5, 9):
        vals, _ = _solve(_problem([V(0)], e), seed=seed)
        assert 0 <= vals[0] <= 10 or vals[0] == 50


def test_or_of_open_comparisons_same_var():
    # (f0 < 2) || (f0 > 31) -- open comparisons fold to a native OR (no range).
    e = BIN(BIN(FIELD(0), E.BinOp.Lt, K(2)), E.BinOp.Or, BIN(FIELD(0), E.BinOp.Gt, K(31)))
    for seed in (1, 4, 9, 16):
        vals, _ = _solve(_problem([V(0, width=8)], e), seed=seed)
        assert vals[0] < 2 or vals[0] > 31


def test_or_of_comparisons_cross_var():
    # (f0 < 5) || (f1 > 100) -- cross-variable OR of comparisons via native OR.
    e = BIN(BIN(FIELD(0), E.BinOp.Lt, K(5)), E.BinOp.Or, BIN(FIELD(1), E.BinOp.Gt, K(100)))
    for seed in (1, 3, 8, 15):
        vals, _ = _solve(_problem([V(0), V(1)], e), seed=seed)
        assert vals[0] < 5 or vals[1] > 100


# --- cross-variable range disjunction (boolean selectors) ----------------- #

def test_cross_var_range_disjunction():
    # f0 in [10..12] || f1 in [50..52] -- ranges over two variables, encoded with
    # boolean selectors. Every solution satisfies at least one branch.
    e = E.ExprBool(op=E.BoolOp.Or, values=[_IN(0, 10, 12), _IN(1, 50, 52)])
    for seed in (1, 2, 3, 7, 11, 19):
        vals, _ = _solve(_problem([V(0, width=8), V(1, width=8)], e), seed=seed)
        assert (10 <= vals[0] <= 12) or (50 <= vals[1] <= 52)


def test_cross_var_disjunction_forces_other_branch():
    # f0 in [10..12] || f1 in [50..52], with f0 pinned outside its range -> the f1
    # branch must be taken.
    e = E.ExprBool(op=E.BoolOp.Or, values=[_IN(0, 10, 12), _IN(1, 50, 52)])
    p = _problem([V(0, width=8), V(1, width=8)], e, BIN(FIELD(0), E.BinOp.Gt, K(200)))
    for seed in (1, 4, 9):
        vals, _ = _solve(p, seed=seed)
        assert vals[0] > 200 and 50 <= vals[1] <= 52


def test_cross_var_disjunction_both_blocked_unsat():
    # both branches made impossible -> UNSAT.
    e = E.ExprBool(op=E.BoolOp.Or, values=[_IN(0, 10, 12), _IN(1, 50, 52)])
    p = _problem([V(0, width=8), V(1, width=8)], e,
                 BIN(FIELD(0), E.BinOp.Gt, K(200)), BIN(FIELD(1), E.BinOp.Gt, K(200)))
    assert _solve_status(p) != SOLVE_OK


def test_cross_var_disjunction_three_way():
    # f0 in [10..12] || f1 in [50..52] || f2 in [90..92] -- three vars.
    e = E.ExprBool(op=E.BoolOp.Or,
                   values=[_IN(0, 10, 12), _IN(1, 50, 52), _IN(2, 90, 92)])
    vs = [V(0, width=8), V(1, width=8), V(2, width=8)]
    for seed in (1, 5, 13):
        vals, _ = _solve(_problem(vs, e), seed=seed)
        assert (10 <= vals[0] <= 12 or 50 <= vals[1] <= 52 or 90 <= vals[2] <= 92)


def test_cross_var_disjunction_selectors_not_written_back():
    # The auxiliary selector vars must not appear in the writeback (only real fields).
    e = E.ExprBool(op=E.BoolOp.Or, values=[_IN(0, 10, 12), _IN(1, 50, 52)])
    _, wb = build_solve_blob(_problem([V(0, width=8), V(1, width=8)], e))
    assert wb == {0: 0, 1: 1}                # only f0, f1; no selector slots


def test_cross_var_disjunction_nested_rejected():
    # A cross-var range disjunction used as an operand (not a top-level boolean
    # constraint) can't be a single expr ref, so it is rejected.
    inner = E.ExprBool(op=E.BoolOp.Or, values=[_IN(0, 10, 12), _IN(1, 50, 52)])
    e = BIN(inner, E.BinOp.Eq, K(1))          # disjunction in a comparison operand
    with pytest.raises(LoweringError):
        build_solve_blob(_problem([V(0, width=8), V(1, width=8)], e))


# --- implies / if-else ---------------------------------------------------- #

def _EQ(i, v):  return BIN(FIELD(i), E.BinOp.Eq, K(v))
def _LT(i, v):  return BIN(FIELD(i), E.BinOp.Lt, K(v))
def _GT(i, v):  return BIN(FIELD(i), E.BinOp.Gt, K(v))


def test_implies_comparison_consequent():
    # mode==1 -> x<10, with mode forced to 1: x must be < 10.
    p = _problem_c([V(0, 8), V(1, 8)],
                   CE(_EQ(0, 1)),
                   C.ConstraintImplies(antecedent=_EQ(0, 1), body=[CE(_LT(1, 10))]))
    for seed in (1, 2, 5, 9, 17):
        vals, _ = _solve(p, seed=seed)
        assert vals[0] == 1 and vals[1] < 10


def test_implies_is_conditional():
    # mode==0 (antecedent false) leaves x unconstrained -> some x >= 10 is reachable.
    p = _problem_c([V(0, 8), V(1, 8)],
                   CE(_EQ(0, 0)),
                   C.ConstraintImplies(antecedent=_EQ(0, 1), body=[CE(_LT(1, 10))]))
    xs = {_solve(p, seed=s)[0][1] for s in range(1, 50)}
    assert max(xs) >= 10                # not clamped -> the implication is conditional


def test_implies_range_consequent():
    # mode==1 -> x in [50..55]  (range splits into two clauses).
    p = _problem_c([V(0, 8), V(1, 8)],
                   CE(_EQ(0, 1)),
                   C.ConstraintImplies(antecedent=_EQ(0, 1),
                                       body=[CE(_IN(1, 50, 55))]))
    for seed in (1, 2, 5, 9, 17):
        vals, _ = _solve(p, seed=seed)
        assert 50 <= vals[1] <= 55


def test_implies_antecedent_conjunction():
    # (a>5 && b<3) -> x==7, with a=10, b=1 forced.
    ante = BIN(_GT(0, 5), E.BinOp.And, _LT(1, 3))
    p = _problem_c([V(0, 8), V(1, 8), V(2, 8)],
                   CE(_EQ(0, 10)), CE(_EQ(1, 1)),
                   C.ConstraintImplies(antecedent=ante, body=[CE(_EQ(2, 7))]))
    for seed in (1, 3, 8):
        vals, _ = _solve(p, seed=seed)
        assert vals[2] == 7


def test_if_else_both_branches():
    def prob(force):
        return _problem_c([V(0, 8), V(1, 8)],
                          CE(_EQ(0, force)),
                          C.ConstraintIfElse(cond=_EQ(0, 1),
                                             then_body=[CE(_LT(1, 10))],
                                             else_body=[CE(_GT(1, 100))]))
    for seed in (1, 4, 9, 16):
        assert _solve(prob(1), seed=seed)[0][1] < 10        # then branch
        assert _solve(prob(0), seed=seed)[0][1] > 100       # else branch


def test_if_else_conjunction_cond_with_else_rejected():
    # A conjunction condition can't express !cond as a clause prefix for the else.
    cond = BIN(_GT(0, 1), E.BinOp.And, _LT(0, 9))
    p = _problem_c([V(0, 8), V(1, 8)],
                   C.ConstraintIfElse(cond=cond, then_body=[CE(_EQ(1, 1))],
                                      else_body=[CE(_EQ(1, 2))]))
    with pytest.raises(LoweringError):
        build_solve_blob(p)


# --- dist ----------------------------------------------------------------- #

def _dw(lo, hi, weight=None, per_value=False):
    rng = E.ExprRange(lower=K(lo), upper=(None if hi is None else K(hi)))
    return C.DistWeight(rng=rng, weight=(None if weight is None else K(weight)),
                        per_value=per_value)


def test_dist_values_in_union():
    # f0 dist { [0..2] := 1, [100..102] := 5 } -- solved values land in the union.
    p = _problem_c([V(0)],
                   C.ConstraintDist(target=FIELD(0), weights=[_dw(0, 2, 1), _dw(100, 102, 5)]))
    vals = [_solve(p, seed=s)[0][0] for s in range(1, 30)]
    assert all(v in {0, 1, 2, 100, 101, 102} for v in vals)


def test_dist_weight_bias():
    # The heavily-weighted range should dominate the samples.
    p = _problem_c([V(0)],
                   C.ConstraintDist(target=FIELD(0), weights=[_dw(0, 2, 1), _dw(100, 102, 50)]))
    hi = sum(1 for s in range(1, 40) if _solve(p, seed=s)[0][0] >= 100)
    assert hi > 20                      # far more than half fall in the weight-50 range


def test_dist_single_values():
    # f0 dist { 7 := 1, 42 := 1 } -- degenerate [v, v] ranges.
    p = _problem_c([V(0, width=8)],
                   C.ConstraintDist(target=FIELD(0), weights=[_dw(7, None), _dw(42, None)]))
    vals = {_solve(p, seed=s)[0][0] for s in range(1, 20)}
    assert vals <= {7, 42}


# --- foreach (array unroll) ----------------------------------------------- #

def LOCAL(name):        return E.ExprRefLocal(name=name)
def SUB(base, idx):     return E.ExprSubscript(value=FIELD(base), slice=idx)

def _problem_arr(vars_, arrays, *items):
    # `arrays` maps array base slot -> ordered element slots. `items` are Constraints.
    return SC.ScSolveProblem(vars=list(vars_), constraints=list(items),
                             writeback={v.name: v.var_id for v in vars_},
                             arrays=dict(arrays))


def test_foreach_each_element_lower_bound():
    # arr[4] elements at slots 0..3 (array base slot 10); foreach (arr[i]) arr[i] > 50.
    vs = [V(0), V(1), V(2), V(3)]
    fe = C.ConstraintForeach(array=FIELD(10), index_var="i",
                             body=[CE(BIN(SUB(10, LOCAL("i")), E.BinOp.Gt, K(50)))])
    vals, _ = _solve(_problem_arr(vs, {10: [0, 1, 2, 3]}, fe))
    assert all(vals[i] > 50 for i in range(4))


def test_foreach_index_used_in_body():
    # foreach (arr[i]) arr[i] == i * 10  -> element i pinned to 10*i.
    vs = [V(0, width=16), V(1, width=16), V(2, width=16)]
    fe = C.ConstraintForeach(
        array=FIELD(10), index_var="i",
        body=[CE(BIN(SUB(10, LOCAL("i")), E.BinOp.Eq,
                     BIN(LOCAL("i"), E.BinOp.Mult, K(10))))])
    vals, _ = _solve(_problem_arr(vs, {10: [0, 1, 2]}, fe))
    assert [vals[0], vals[1], vals[2]] == [0, 10, 20]


def test_foreach_range_body_two_constraints():
    # foreach (arr[i]) { arr[i] >= 10; arr[i] <= 20; } -- multi-item body.
    vs = [V(0), V(1), V(2)]
    fe = C.ConstraintForeach(
        array=FIELD(10), index_var="i",
        body=[CE(BIN(SUB(10, LOCAL("i")), E.BinOp.GtE, K(10))),
              CE(BIN(SUB(10, LOCAL("i")), E.BinOp.LtE, K(20)))])
    vals, _ = _solve(_problem_arr(vs, {10: [0, 1, 2]}, fe))
    assert all(10 <= vals[i] <= 20 for i in range(3))


def test_top_level_subscript_constant_index():
    # arr[1+1] == 99 -- constant-folded subscript at top level (no foreach).
    vs = [V(0), V(1), V(2), V(3)]
    con = CE(BIN(SUB(10, BIN(K(1), E.BinOp.Add, K(1))), E.BinOp.Eq, K(99)))
    vals, _ = _solve(_problem_arr(vs, {10: [0, 1, 2, 3]}, con))
    assert vals[2] == 99


def test_foreach_element_slots_differ_from_var_ids():
    # Array elements at interleaved slots 1,3,5 (var_ids 0,1,2): foreach must resolve
    # arr[i] to the element's *slot* and lower it under the right var_id.
    vs = [SC.ScSolveVar(name="e0", var_id=0, slot=1, width=8),
          SC.ScSolveVar(name="e1", var_id=1, slot=3, width=8),
          SC.ScSolveVar(name="e2", var_id=2, slot=5, width=8)]
    fe = C.ConstraintForeach(array=FIELD(10), index_var="i",
                             body=[CE(BIN(SUB(10, LOCAL("i")), E.BinOp.Eq, K(7)))])
    blob, wb = build_solve_blob(_problem_arr(vs, {10: [1, 3, 5]}, fe))
    assert wb == {1: 0, 3: 1, 5: 2}          # slot -> var_id
    vals, _ = _solve(_problem_arr(vs, {10: [1, 3, 5]}, fe))
    assert vals == {0: 7, 1: 7, 2: 7}


def test_foreach_over_non_array_rejected():
    fe = C.ConstraintForeach(array=FIELD(99), index_var="i",
                             body=[CE(BIN(SUB(99, LOCAL("i")), E.BinOp.Gt, K(0)))])
    with pytest.raises(LoweringError):
        build_solve_blob(_problem_arr([V(0)], {10: [0]}, fe))


def test_foreach_subscript_out_of_range_rejected():
    # body indexes arr[i+1]; at the last element i+1 == size -> out of range.
    vs = [V(0), V(1)]
    fe = C.ConstraintForeach(
        array=FIELD(10), index_var="i",
        body=[CE(BIN(SUB(10, BIN(LOCAL("i"), E.BinOp.Add, K(1))), E.BinOp.Gt, K(0)))])
    with pytest.raises(LoweringError):
        build_solve_blob(_problem_arr(vs, {10: [0, 1]}, fe))


# --- unique --------------------------------------------------------------- #

def _solve_status(problem, seed=7):
    """Compile+solve; return the solver rc, or 'compile-unsat' if it fails compile."""
    blob, _ = build_solve_blob(problem)
    raw = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
    try:
        ctx = SolveCtx(raw)
    except Exception:
        return "compile-unsat"
    try:
        return ctx.solve(seed=seed)
    finally:
        ctx.destroy()


def test_unique_all_distinct_width32():
    # unique { f0, f1, f2 } over default-width (32-bit) vars -- lowered to the native
    # add_all_different, whose historical width-32 (tier-1) unsoundness is now fixed.
    vs = [V(0), V(1), V(2)]
    p = _problem_c(vs, C.ConstraintUnique(items=[FIELD(0), FIELD(1), FIELD(2)]))
    for seed in (1, 7, 42):
        vals, _ = _solve(p, seed=seed)
        assert len({vals[0], vals[1], vals[2]}) == 3


def test_unique_two_items():
    # unique { f0, f1 } == f0 != f1.
    vs = [V(0, width=8), V(1, width=8)]
    p = _problem_c(vs, C.ConstraintUnique(items=[FIELD(0), FIELD(1)]))
    vals, _ = _solve(p)
    assert vals[0] != vals[1]


def test_unique_respects_other_constraints():
    # unique alongside a bound: all distinct and each in [10, 12] -> a permutation.
    vs = [V(0), V(1), V(2)]
    p = _problem_c(
        vs,
        CE(BIN(FIELD(0), E.BinOp.GtE, K(10))), CE(BIN(FIELD(0), E.BinOp.LtE, K(12))),
        CE(BIN(FIELD(1), E.BinOp.GtE, K(10))), CE(BIN(FIELD(1), E.BinOp.LtE, K(12))),
        CE(BIN(FIELD(2), E.BinOp.GtE, K(10))), CE(BIN(FIELD(2), E.BinOp.LtE, K(12))),
        C.ConstraintUnique(items=[FIELD(0), FIELD(1), FIELD(2)]))
    vals, _ = _solve(p)
    assert sorted([vals[0], vals[1], vals[2]]) == [10, 11, 12]


def test_unique_pigeonhole_unsat():
    # 3 one-bit vars (domain {0, 1}) cannot all be distinct -> UNSAT.
    vs = [V(0, width=1), V(1, width=1), V(2, width=1)]
    p = _problem_c(vs, C.ConstraintUnique(items=[FIELD(0), FIELD(1), FIELD(2)]))
    assert _solve_status(p) != SOLVE_OK        # UNSAT (rc != 0) or compile-unsat


def test_unique_repeated_item_unsat():
    # unique { f0, f0 } -- a repeated var can't feed add_all_different, so the
    # pairwise fallback applies: f0 != f0 is unsatisfiable.
    vs = [V(0, width=8)]
    p = _problem_c(vs, C.ConstraintUnique(items=[FIELD(0), FIELD(0)]))
    assert _solve_status(p) != SOLVE_OK


def test_unique_single_item_vacuous():
    # unique over a single item imposes nothing; the problem still solves.
    vs = [V(0, width=8)]
    p = _problem_c(vs, C.ConstraintUnique(items=[FIELD(0)]))
    assert _solve_status(p) == SOLVE_OK


def test_unique_over_array_elements():
    # unique { arr[0], arr[1], arr[2] } via subscripts (arrays map) -> distinct.
    vs = [V(0, width=8), V(1, width=8), V(2, width=8)]
    items = [SUB(10, K(0)), SUB(10, K(1)), SUB(10, K(2))]
    p = _problem_arr(vs, {10: [0, 1, 2]}, C.ConstraintUnique(items=items))
    vals, _ = _solve(p)
    assert len({vals[0], vals[1], vals[2]}) == 3


# --- solve...before ------------------------------------------------------- #

def test_solve_before_is_a_no_op():
    # `solve f0 before f1` is a distribution-only hint dv-solve can't honour, so it
    # must lower cleanly and change nothing: the blob is byte-identical to the same
    # problem without the hint, so the solution is unchanged.
    hard = BIN(BIN(FIELD(0), E.BinOp.Add, FIELD(1)), E.BinOp.Eq, K(42))
    base = _problem_c([V(0, width=8), V(1, width=8)], CE(hard))
    withsb = _problem_c([V(0, width=8), V(1, width=8)], CE(hard),
                        C.ConstraintSolveBefore(before=[FIELD(0)], after=[FIELD(1)]))
    blob_a, _ = build_solve_blob(base)
    blob_b, _ = build_solve_blob(withsb)
    assert blob_a == blob_b                    # the hint contributes nothing
    vals, _ = _solve(withsb)
    assert vals[0] + vals[1] == 42             # hard constraint still enforced


def test_solve_before_lifted_field_ignored():
    # The lifted form on ScSolveProblem.solve_before is likewise ignored (lowers fine).
    p = SC.ScSolveProblem(
        vars=[V(0, width=8)],
        constraints=[CE(BIN(FIELD(0), E.BinOp.GtE, K(5)))],
        writeback={"f0": 0},
        solve_before=[([FIELD(0)], [FIELD(0)])])
    vals, _ = _solve(p)
    assert vals[0] >= 5


# --- soft ----------------------------------------------------------------- #

def test_soft_honored_when_satisfiable():
    # soft f0 == 12345 with no conflicting hard constraint -> honored exactly.
    p = _problem_c([V(0)],
                   C.ConstraintSoft(expr=BIN(FIELD(0), E.BinOp.Eq, K(12345))))
    for seed in (1, 7, 99):
        assert _solve(p, seed=seed)[0][0] == 12345


def test_soft_dropped_when_conflicting():
    # hard f0 >= 100 wins; the conflicting soft f0 == 5 is relaxed (never violated).
    p = _problem_c([V(0)],
                   C.ConstraintExpr(expr=BIN(FIELD(0), E.BinOp.GtE, K(100))),
                   C.ConstraintSoft(expr=BIN(FIELD(0), E.BinOp.Eq, K(5))))
    for seed in (1, 7, 99):
        assert _solve(p, seed=seed)[0][0] >= 100


def test_soft_never_violates_hard_wide_var():
    # Regression for the historical width-32 soft *hang*: a conflicting soft on a
    # full-width int must relax quickly and leave the hard constraint intact.
    p = _problem_c([V(0, width=32)],
                   C.ConstraintExpr(expr=BIN(FIELD(0), E.BinOp.GtE, K(1_000_000))),
                   C.ConstraintSoft(expr=BIN(FIELD(0), E.BinOp.Eq, K(5))))
    assert _solve(p, seed=1)[0][0] >= 1_000_000


def test_soft_priority_partial_relaxation():
    # hard f0 >= 100; soft(f0 <= 10) conflicts (dropped), soft(f0 <= 200) is
    # satisfiable (kept) -> result in [100, 200]. The solver keeps the maximal
    # satisfiable soft set, never violating the hard constraint.
    p = _problem_c([V(0)],
                   C.ConstraintExpr(expr=BIN(FIELD(0), E.BinOp.GtE, K(100))),
                   C.ConstraintSoft(expr=BIN(FIELD(0), E.BinOp.LtE, K(10))),
                   C.ConstraintSoft(expr=BIN(FIELD(0), E.BinOp.LtE, K(200))))
    v = _solve(p, seed=1)[0][0]
    assert 100 <= v <= 200


# --- rejection ------------------------------------------------------------ #


def test_dist_non_const_bound_rejected():
    p = _problem_c([V(0), V(1)],
                   C.ConstraintDist(target=FIELD(0),
                                    weights=[C.DistWeight(rng=E.ExprRange(lower=FIELD(1), upper=K(9)))]))
    with pytest.raises(LoweringError):
        build_solve_blob(p)


def test_unsupported_block_form_rejected():
    # An unknown Constraint form (the abstract base, no matching dispatch) is rejected.
    p = SC.ScSolveProblem(vars=[V(0), V(1)], constraints=[C.Constraint()])
    with pytest.raises(LoweringError):
        build_solve_blob(p)


def test_unknown_field_slot_rejected():
    # constraint references slot 3, but only var 0 is declared.
    with pytest.raises(LoweringError):
        build_solve_blob(_problem([V(0)], BIN(FIELD(3), E.BinOp.Eq, K(1))))


def test_unsupported_binop_rejected():
    with pytest.raises(LoweringError):
        build_solve_blob(_problem([V(0)], BIN(FIELD(0), E.BinOp.Exp, K(2))))
