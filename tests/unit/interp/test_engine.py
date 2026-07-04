"""W8 -- the ``run_module`` engine entry: PSS Context -> oracle execution.

Exercises the successor execution engine surface end-to-end without the legacy
runtime: a hand-built ``zuspec.ir.core`` Context is lowered by ``PSSToScenarioPass``
and executed by ``run_module`` (through the default round-trip serialize path).
"""

import zuspec.ir.core as ir
from zuspec.ir.core.data_type import Function, DataTypeInt
from zuspec.ir.core.stmt import Arguments, Arg, StmtExpr
from zuspec.ir.core.expr import ExprConstant, ExprAttribute, TypeExprRefSelf, ExprCall
from zuspec.ir.core.xf import PSSToScenarioPass

from zuspec.be.bc.interp import run_module, Obj, ImportProvider, FixedSolveBackend


def _self_call(name, *args):
    return ExprCall(func=ExprAttribute(value=TypeExprRefSelf(), attr=name),
                    args=list(args))


def _imports_ctx():
    body = ir.Function(name="body", body=[
        StmtExpr(expr=_self_call("doit", _self_call("getval", ExprConstant(value=7))))])
    go = ir.DataTypeClass(name="Go", super=None, functions=[body])
    top = ir.DataTypeComponent(name="Top", super=None)
    ctx = ir.Context(type_m={"Top": top, "Top::Go": go})
    ctx.import_functions = [
        Function(name="getval", is_import=True, is_solve=True,
                 returns=DataTypeInt(bits=32, signed=False),
                 args=Arguments(args=[Arg(arg="i", annotation=DataTypeInt(bits=32))])),
        Function(name="doit", is_import=True, is_target=True, returns=None,
                 args=Arguments(args=[Arg(arg="i", annotation=DataTypeInt(bits=32))])),
    ]
    return ctx


class _Provider(ImportProvider):
    def __init__(self, id2name):
        self.id2name = id2name
        self.calls = []

    def call(self, fn_id, args):
        name = self.id2name[fn_id]
        self.calls.append((name, list(args)))
        return (args[0] + 5) if name == "getval" else 0


def test_run_module_executes_import_scenario():
    module = PSSToScenarioPass(exports=["Go"]).lower(_imports_ctx())
    prov = _Provider({d.fn_id: d.name for d in module.imports})
    res = run_module(module, entry_action="Go", import_provider=prov)
    # getval(7) -> 12 threads into the blocking doit(12).
    assert prov.calls == [("getval", [7]), ("doit", [12])]
    assert res.now == 0


def test_run_module_roundtrip_default_matches_inmemory():
    def go(round_trip):
        module = PSSToScenarioPass(exports=["Go"]).lower(_imports_ctx())
        prov = _Provider({d.fn_id: d.name for d in module.imports})
        run_module(module, entry_action="Go", import_provider=prov,
                   round_trip=round_trip)
        return prov.calls

    assert go(True) == go(False) == [("getval", [7]), ("doit", [12])]


def test_run_module_solve_writeback():
    # An action with a rand field + empty body lowers to a leading SOLVE.
    go = ir.DataTypeClass(
        name="Go", super=None,
        fields=[ir.Field(name="addr", datatype=DataTypeInt(bits=8, signed=False),
                         rand_kind=ir.RandKind.RAND)],
        functions=[Function(name="body", body=[])])
    top = ir.DataTypeComponent(name="Top", super=None)
    ctx = ir.Context(type_m={"Top": top, "Top::Go": go})

    module = PSSToScenarioPass(exports=["Go"]).lower(ctx)
    obj = Obj(field_names=["addr"])
    run_module(module, entry_action="Go", obj=obj, seed=5,
               solve_backend=FixedSolveBackend(base=100))
    assert set(obj.as_dict()) == {"addr"}
