"""M8b6a: the opt-in plan log (``activation.PlanLog``).

A run given a log records each select's branch in it; a run without one
records nothing. Action-tree occurrences and their values are covered in
pssc's tests/unit/integration/test_plan_log.py (they need a PSS front end).
"""

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import stmt as S
from zuspec.ir.core import expr as E

from zuspec.be.bc.interp import Obj, run_model
from zuspec.be.bc.interp.activation import PlanLog
from zuspec.be.bc.lower import lower_scenario


def _branch(sentinel):
    return SC.ScSelectBranch(guard=None, weight=E.ExprConstant(value=1), body=[
        SC.ScExecBlock(kind="body", stmts=[
            S.StmtAssign(targets=[E.ExprRefField(base=E.TypeExprRefSelf(), index=0)],
                         value=E.ExprConstant(value=sentinel))])])


def _model():
    return lower_scenario([SC.ScCoroutine(name="root", body=[
        SC.ScSelect(branches=[_branch(10), _branch(11), _branch(12)])])])


def test_log_is_off_by_default():
    res = run_model(_model(), obj=Obj(field_names=["a"]), seed=3)
    assert res.fields["a"] in (10, 11, 12)


def test_select_branch_is_recorded():
    for seed in range(8):
        log = PlanLog()
        res = run_model(_model(), obj=Obj(field_names=["a"]), seed=seed, plan=log)
        assert len(log.control) == 1
        c = log.control[0]
        assert c["kind"] == "select" and c["select"] == 0
        # no action tree: no occurrence to hang the choice on
        assert c["occ"] is None and log.occurrences == []
        assert res.fields["a"] == 10 + c["value"]


def test_log_counts_its_own_events():
    log = PlanLog()
    a = log.start(None, "", "T#0", "p::T")
    b = log.start(0, "x", "T#0.x#0", "p::X")
    log.end(b, 0, True)
    log.end(a, 0, False)
    assert [(o["start"], o["end"]) for o in log.occurrences] == [(1, 4), (2, 3)]
    assert log.occurrences[0]["completed"] is False


def test_order_is_transitively_reduced():
    # root(a; (b || c); d): a < b, a < c, b < d, c < d; a < d is implied
    log = PlanLog()
    r = log.start(None, "", "T#0", "p::T")
    occ = {}
    def run(name):
        occ[name] = log.start(0, name, "", "p::A")
        return occ[name]
    log.end(run("a"), 0, True)
    b, c = run("b"), run("c")
    log.end(b, 0, True)
    log.end(c, 0, True)
    log.end(run("d"), 0, True)
    log.end(r, 0, True)
    a, d = occ["a"], occ["d"]
    assert log.order() == sorted([[a, b], [a, c], [b, d], [c, d]])


def test_flow_and_claims_are_recorded():
    log = PlanLog()
    f = log.flow_out("buffer", "buf_p@0", 1, {"v": 3})
    log.consume(f, 2)
    log.consume(f, 2)
    log.claim("ch_p@0", 1, 1, True)
    log.claim("ch_p@0", 1, 2, False)
    log.claim("ch_p@0", 0, 3, True)
    assert log.flow == [{"id": 0, "kind": "buffer", "pool": "buf_p@0",
                         "producer": 1, "consumers": [2], "fields": {"v": 3}}]
    assert log.resources == [
        {"pool": "ch_p@0", "instance": 1,
         "claims": [{"occ": 1, "lock": True}, {"occ": 2, "lock": False}]},
        {"pool": "ch_p@0", "instance": 0, "claims": [{"occ": 3, "lock": True}]}]
