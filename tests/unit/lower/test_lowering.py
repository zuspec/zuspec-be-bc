"""P1-W6 -- IR->ZBC lowering on hand-built Scenario IR."""

import pytest

from zuspec.ir.core.base import Loc
from zuspec.ir.core import scenario as SC
from zuspec.ir.core import stmt as S
from zuspec.ir.core import expr as E

from zuspec.ir.core.expr import ExprCall, ExprAttribute, TypeExprRefSelf

from zuspec.be.bc.lower import lower_scenario, lower_coroutine, Lowerer, LoweringError
from zuspec.be.bc.model import Op, ZbcModel, INSTR_F_BLOCKING, INSTR_F_HAS_RET


def _root_coro():
    body = [
        SC.ScExecBlock(kind="body", stmts=[
            S.StmtAssign(
                targets=[E.ExprRefLocal(name="x")],
                value=E.ExprConstant(value=5),
                comment="init x",
                loc=Loc(file="a.pss", line=3, pos=2, ref=None),
            ),
        ]),
        SC.ScWait(time=E.ExprConstant(value=10),
                  loc=Loc(file="a.pss", line=4, pos=2, ref=None)),
        SC.ScSolveProblem(
            vars=[SC.ScSolveVar(name="b", var_id=0), SC.ScSolveVar(name="a", var_id=1)],
            writeback={"a": 1, "b": 0},
        ),
        SC.ScImport(fn="send", fn_id=42, blocking=True,
                    args=[E.ExprConstant(value=7)], ret_var="r"),
    ]
    return SC.ScCoroutine(name="root", body=body, frame_locals=["x"])


def test_fsm_block_split():
    m = lower_scenario([_root_coro()])
    d = m.coros[0]
    # WAIT ends block 0, blocking IMPORT ends block 1, terminal block 2.
    assert [b.suspend_op for b in d.blocks] == [int(Op.WAIT), int(Op.IMPORT), 0]
    assert d.blocks[0].pc_start == 0
    assert d.blocks[-1].pc_start == d.blocks[-1].pc_end or d.blocks[-1].pc_end >= d.blocks[-1].pc_start


def test_exec_block_lowers_to_procedural():
    d = lower_scenario([_root_coro()]).coros[0]
    ops = [ins.op for ins in d.code]
    assert Op.CONST in ops and Op.ST_LOCAL in ops
    # x = 5
    const = next(i for i in d.code if i.op == Op.CONST)
    assert const.imm == 5


def test_wait_uses_immediate_for_constant_time():
    d = lower_scenario([_root_coro()]).coros[0]
    wait = next(i for i in d.code if i.op == Op.WAIT)
    assert wait.imm == 10 and wait.args == ()


def test_import_flags_and_writeback():
    d = lower_scenario([_root_coro()]).coros[0]
    imp = next(i for i in d.code if i.op == Op.IMPORT)
    assert imp.flags & INSTR_F_BLOCKING
    assert imp.flags & INSTR_F_HAS_RET
    assert imp.args[0] == 42  # fn_id
    # ret_var 'r' became a frame local and gets a store after the import
    assert "r" in d.frame_locals


def test_solve_problem_recorded_with_sorted_var_names():
    m = lower_scenario([_root_coro()])
    assert len(m.problems) == 1
    p = m.problems[0]
    assert p.var_names == ["a", "b"]  # sorted-by-name -> var_id order
    assert p.writeback == {"a": 1, "b": 0}
    solve = next(i for i in m.coros[0].code if i.op == Op.SOLVE)
    assert solve.args == (0,)  # problem id


def test_provenance_comment_survives():
    m = lower_scenario([_root_coro()])
    texts = [c.text for p in m.prov.entries for c in p.comments]
    assert "init x" in texts
    # the assign instruction carries a non-zero src_ref pointing at that prov
    d = m.coros[0]
    st = next(i for i in d.code if i.op == Op.ST_LOCAL)
    assert st.src_ref != 0
    assert m.prov.entries[st.src_ref].file == "a.pss"


def test_lowered_model_roundtrips():
    m = lower_scenario([_root_coro()])
    assert ZbcModel.from_bytes(m.to_bytes()) == m


def test_suspend_in_loop_lowers_to_flat_stream():
    # A WAIT inside a repeat loop now lowers (the oracle runs the flat, resumable
    # stream directly): a back-edge BR and the interior WAIT coexist in one coro.
    loop = SC.ScLoop(kind="repeat", count=E.ExprConstant(value=3),
                     body=[SC.ScWait(time=E.ExprConstant(value=1))])
    coro = SC.ScCoroutine(name="ok", body=[loop])
    code = lower_scenario([coro]).coros[0].code
    assert any(i.op == Op.BR for i in code)
    assert any(i.op == Op.WAIT for i in code)


def test_unknown_statement_rejected_cleanly():
    # The "no silent miscompile" guarantee (T1-F): an unsupported statement type
    # raises a clear LoweringError rather than being dropped.
    class _Bogus(SC.ScStmt):
        pass
    coro = SC.ScCoroutine(name="bad", body=[_Bogus()])
    with pytest.raises(LoweringError):
        lower_scenario([coro])


def test_select_rejected_cleanly():
    # ScSelect (weighted-choice) lowering is still deferred; reject, don't miscompile.
    coro = SC.ScCoroutine(name="p", body=[SC.ScSelect(branches=[])])
    with pytest.raises(LoweringError):
        lower_scenario([coro])


def test_non_all_par_join_rejected_cleanly():
    from zuspec.ir.core.activity import JoinSpec, JoinKind
    coro = SC.ScCoroutine(name="p", body=[
        SC.ScPar(branches=[SC.ScExecBlock(kind="body", stmts=[])],
                 join_spec=JoinSpec(kind=JoinKind.FIRST)),
    ])
    with pytest.raises(LoweringError):
        lower_scenario([coro])


def test_no_suspend_coroutine_is_single_block():
    coro = SC.ScCoroutine(name="pure", body=[
        SC.ScExecBlock(kind="body", stmts=[S.StmtReturn(value=None)]),
    ])
    d = lower_scenario([coro]).coros[0]
    assert len(d.blocks) == 1
    assert d.blocks[0].suspend_op == 0
    assert d.code[-1].op == Op.RET


def test_value_import_call_lowers_to_import_with_return():
    # A value-returning import call `getval(7)` inside procedural code becomes a
    # non-blocking IMPORT into a fresh register (the T1-A import-arg path).
    from zuspec.ir.core.scenario import ScImportDecl

    call = ExprCall(func=ExprAttribute(value=TypeExprRefSelf(), attr="getval"),
                    args=[E.ExprConstant(value=7)])
    coro = SC.ScCoroutine(name="c", frame_locals=["x"], body=[
        SC.ScExecBlock(kind="body", stmts=[
            S.StmtAssign(targets=[E.ExprRefLocal(name="x")], value=call)]),
    ])
    decl = ScImportDecl(name="getval", fn_id=5, blocking=False, ret_type=(32, False))
    d = lower_scenario([coro], imports=[decl]).coros[0]

    imp = next(i for i in d.code if i.op == Op.IMPORT)
    assert imp.args[0] == 5                      # fn_id
    assert imp.flags & INSTR_F_HAS_RET
    assert not (imp.flags & INSTR_F_BLOCKING)    # solve import is non-blocking
    # the imported value is stored into the frame local
    assert any(i.op == Op.ST_LOCAL for i in d.code)


def test_call_to_unknown_import_rejected():
    call = ExprCall(func=ExprAttribute(value=TypeExprRefSelf(), attr="mystery"), args=[])
    coro = SC.ScCoroutine(name="c", body=[
        SC.ScExecBlock(kind="body", stmts=[S.StmtExpr(expr=call)]),
    ])
    with pytest.raises(LoweringError):
        lower_scenario([coro])


def test_unsupported_statement_rejected():
    coro = SC.ScCoroutine(name="u", body=[
        SC.ScExecBlock(kind="body", stmts=[
            S.StmtBreak(),  # not supported in M1 procedural lowering
        ]),
    ])
    with pytest.raises(LoweringError):
        lower_scenario([coro])
