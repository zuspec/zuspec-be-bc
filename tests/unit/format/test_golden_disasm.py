"""T1-C -- golden ZBC disassembly for representative lowered activities.

Each fixture lowers a hand-built Scenario IR to a ``ZbcModel`` and compares its
opcode-level disassembly (:func:`zuspec.be.bc.format.disasm.disassemble`) to a
checked-in golden, so a lowering change shows up as a reviewable diff.

Regenerate after an intentional lowering/disasm change with::

    ZBC_REGEN_GOLDEN=1 python -m pytest tests/unit/format/test_golden_disasm.py

M1 covers the activities that lower today: **seq** (exec + WAIT + SOLVE + blocking
IMPORT), a standalone **solve**, **imports** (nested value + blocking call), **loop**
(a counted repeat with a nested if -> BR/BRZ), **parallel** (ScPar/ALL desugared to
branch sub-coroutines + SPAWN/JOIN), and **select** (weighted-choice SELECT op +
branch table). Degenerate/deferred forms (empty select, non-ALL joins, guards) still
reject cleanly (see ``test_degenerate_activities_rejected``).
"""

import os

import pytest

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import stmt as S
from zuspec.ir.core import expr as E
from zuspec.ir.core.base import Loc
from zuspec.ir.core.scenario import ScImportDecl
from zuspec.ir.core.expr import ExprCall, ExprAttribute, TypeExprRefSelf

from zuspec.be.bc.lower import lower_scenario, LoweringError
from zuspec.be.bc.format.disasm import disassemble

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
_REGEN = os.environ.get("ZBC_REGEN_GOLDEN") == "1"


def _check_golden(name: str, text: str) -> None:
    path = os.path.join(_GOLDEN_DIR, name)
    if _REGEN:
        os.makedirs(_GOLDEN_DIR, exist_ok=True)
        with open(path, "w") as fp:
            fp.write(text)
        pytest.skip(f"regenerated golden {name}")
    assert os.path.exists(path), (
        f"missing golden {path}; regenerate with "
        f"ZBC_REGEN_GOLDEN=1 python -m pytest {__file__}"
    )
    with open(path) as fp:
        expected = fp.read()
    assert text == expected, (
        f"disassembly drifted from golden {name}; if intentional regenerate with "
        f"ZBC_REGEN_GOLDEN=1"
    )


# --------------------------------------------------------------------------- #
# Fixtures -> models
# --------------------------------------------------------------------------- #

def _seq_model():
    body = [
        SC.ScExecBlock(kind="body", stmts=[
            S.StmtAssign(targets=[E.ExprRefLocal(name="x")],
                         value=E.ExprConstant(value=5), comment="init x",
                         loc=Loc(file="a.pss", line=3, pos=2, ref=None))]),
        SC.ScWait(time=E.ExprConstant(value=10),
                  loc=Loc(file="a.pss", line=4, pos=2, ref=None)),
        SC.ScSolveProblem(
            vars=[SC.ScSolveVar(name="b", var_id=0), SC.ScSolveVar(name="a", var_id=1)],
            writeback={"a": 0, "b": 1}, seed=E.ExprConstant(value=42)),
        SC.ScImport(fn="send", fn_id=7, blocking=True,
                    args=[E.ExprConstant(value=3)], ret_var="r"),
    ]
    return lower_scenario([SC.ScCoroutine(name="root", body=body, frame_locals=["x"])])


def _solve_model():
    coro = SC.ScCoroutine(name="Go", frame_locals=["addr"], body=[
        SC.ScSolveProblem(vars=[SC.ScSolveVar(name="addr", var_id=0, width=8)],
                          writeback={"addr": 0}),
        SC.ScExecBlock(kind="body", stmts=[]),
    ])
    return lower_scenario([coro])


def _imports_model():
    getval = ExprCall(func=ExprAttribute(value=TypeExprRefSelf(), attr="getval"),
                      args=[E.ExprConstant(value=7)])
    coro = SC.ScCoroutine(name="Go", body=[
        SC.ScImport(fn="doit", fn_id=1, blocking=True, args=[getval])])
    decls = [ScImportDecl(name="getval", fn_id=0, blocking=False, ret_type=(32, False)),
             ScImportDecl(name="doit", fn_id=1, blocking=True)]
    return lower_scenario([coro], imports=decls, blocking_targets=["doit"])


def _match_model():
    # r = 0; match (field0) { 1: r=10; default: r=99 }; return r
    def setr(v):
        return SC.ScExecBlock(kind="body", stmts=[
            S.StmtAssign(targets=[E.ExprRefLocal(name="r")],
                         value=E.ExprConstant(value=v))])
    coro = SC.ScCoroutine(name="Go", frame_locals=["r"], body=[
        setr(0),
        SC.ScMatch(subject=E.ExprRefField(base=E.TypeExprRefSelf(), index=0), cases=[
            SC.ScMatchCase(pattern=E.ExprConstant(value=1), body=[setr(10)]),
            SC.ScMatchCase(pattern=None, body=[setr(99)]),
        ]),
        SC.ScExecBlock(kind="body", stmts=[
            S.StmtReturn(value=E.ExprRefLocal(name="r"))]),
    ])
    return lower_scenario([coro])


def _parallel_model():
    # parallel { field0 = 1; field1 = 2 }
    def setf(idx, v):
        return SC.ScExecBlock(kind="body", stmts=[
            S.StmtAssign(targets=[E.ExprRefField(base=E.TypeExprRefSelf(), index=idx)],
                         value=E.ExprConstant(value=v))])
    coro = SC.ScCoroutine(name="Go", body=[
        SC.ScPar(branches=[setf(0, 1), setf(1, 2)])])
    return lower_scenario([coro])


def _select_model():
    # select { [2]: field0 = 1; [3]: field0 = 2 }
    def branch(w, v):
        return SC.ScSelectBranch(
            weight=E.ExprConstant(value=w),
            body=[SC.ScExecBlock(kind="body", stmts=[
                S.StmtAssign(targets=[E.ExprRefField(base=E.TypeExprRefSelf(), index=0)],
                             value=E.ExprConstant(value=v))])])
    coro = SC.ScCoroutine(name="Go", body=[
        SC.ScSelect(branches=[branch(2, 1), branch(3, 2)])])
    return lower_scenario([coro])


def _suspend_loop_model():
    # repeat 2 { do act } where `act` blocks -> a back-edge BR + a BLOCKING INVOKE
    # coexist in the root coroutine's flat stream (no FSM split of the loop body).
    act = SC.ScCoroutine(name="act", body=[SC.ScWait(time=E.ExprConstant(value=2))])
    root = SC.ScCoroutine(name="root", body=[
        SC.ScLoop(kind="repeat", count=E.ExprConstant(value=2),
                  body=[SC.ScInvoke(target="act")])])
    return lower_scenario([root, act], blocking_targets=["act"])


def _guarded_select_model():
    # select { [1] (field1): field0 = 1; [2] (field2): field0 = 2 } allow_none
    def branch(w, gidx, v):
        return SC.ScSelectBranch(
            guard=E.ExprRefField(base=E.TypeExprRefSelf(), index=gidx),
            weight=E.ExprConstant(value=w),
            body=[SC.ScExecBlock(kind="body", stmts=[
                S.StmtAssign(targets=[E.ExprRefField(base=E.TypeExprRefSelf(), index=0)],
                             value=E.ExprConstant(value=v))])])
    coro = SC.ScCoroutine(name="Go", body=[
        SC.ScSelect(branches=[branch(1, 1, 1), branch(2, 2, 2)], allow_none=True)])
    return lower_scenario([coro])


def _loop_model():
    # acc = 0; repeat 3: if field0: acc = acc + 1; return acc
    add1 = E.ExprBin(lhs=E.ExprRefLocal(name="acc"), op=E.BinOp.Add,
                     rhs=E.ExprConstant(value=1))
    coro = SC.ScCoroutine(name="Go", frame_locals=["acc"], body=[
        SC.ScExecBlock(kind="body", stmts=[
            S.StmtAssign(targets=[E.ExprRefLocal(name="acc")],
                         value=E.ExprConstant(value=0))]),
        SC.ScLoop(kind="repeat", count=E.ExprConstant(value=3), body=[
            SC.ScIf(cond=E.ExprRefField(base=E.TypeExprRefSelf(), index=0),
                    then_body=[SC.ScExecBlock(kind="body", stmts=[
                        S.StmtAssign(targets=[E.ExprRefLocal(name="acc")],
                                     value=add1)])]),
        ]),
        SC.ScExecBlock(kind="body", stmts=[
            S.StmtReturn(value=E.ExprRefLocal(name="acc"))]),
    ])
    return lower_scenario([coro])


# --------------------------------------------------------------------------- #
# Golden tests
# --------------------------------------------------------------------------- #

def test_golden_seq():
    _check_golden("seq.txt", disassemble(_seq_model()))


def test_golden_solve():
    _check_golden("solve.txt", disassemble(_solve_model()))


def test_golden_imports():
    _check_golden("imports.txt", disassemble(_imports_model()))


def test_golden_loop():
    _check_golden("loop.txt", disassemble(_loop_model()))


def test_golden_match():
    _check_golden("match.txt", disassemble(_match_model()))


def test_golden_suspend_loop():
    _check_golden("suspend_loop.txt", disassemble(_suspend_loop_model()))


def test_golden_parallel():
    _check_golden("parallel.txt", disassemble(_parallel_model()))


def test_golden_select():
    _check_golden("select.txt", disassemble(_select_model()))


def test_golden_guarded_select():
    _check_golden("guarded_select.txt", disassemble(_guarded_select_model()))


def test_disasm_survives_roundtrip_shape():
    # The serialized model reproduces the same block/opcode structure (minus the
    # in-memory problem side channel, which is not serialized in M1).
    from zuspec.be.bc.model import ZbcModel
    m = _seq_model()
    m2 = ZbcModel.from_bytes(m.to_bytes())
    # Code + blocks + frame locals disassemble identically; only the `problems:`
    # header block differs (side channel), so compare from the first coro on.
    a = disassemble(m).split("coro #0", 1)[1]
    b = disassemble(m2).split("coro #0", 1)[1]
    assert a == b


@pytest.mark.parametrize("act", [
    SC.ScSelect(branches=[]),          # empty select: nothing to choose
])
def test_degenerate_activities_rejected(act):
    # Degenerate forms must reject cleanly so a silent miscompile can't slip a
    # bogus golden in.
    with pytest.raises(LoweringError):
        lower_scenario([SC.ScCoroutine(name="d", body=[act])])
