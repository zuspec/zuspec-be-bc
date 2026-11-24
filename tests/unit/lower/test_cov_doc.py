"""covg_doc: a covergroup as COVG JSON data (F4.4); what it cannot describe
is a lowering error, never dropped."""
import pytest

from zuspec.ir.core import expr as E
from zuspec.ir.core.coverage import (BinDef, BinKind, CovergroupDef,
                                     CoverpointDef, CrossBinDef, CrossDef)

from zuspec.be.bc.lower import LoweringError
from zuspec.be.bc.lower.coverage import covg_doc


def _c(v):
    return E.ExprConstant(value=v)


def _cp_ref(name):
    return E.ExprAttribute(value=E.TypeExprRefSelf(), attr=name)


def _cg(bins, crosses=()):
    cp = CoverpointDef(name="cp_x", target=_cp_ref("x"), bins=bins)
    return CovergroupDef(name="cg", instance_name="cs", coverpoints=[cp],
                         crosses=list(crosses))


def test_bins_values_ranges_and_with():
    lo = BinDef(name="lo", values=[E.ExprRange(lower=_c(0), upper=_c(7))])
    two = BinDef(name="two", values=[_c(2)], kind=BinKind.IGNORE)
    odd = BinDef(name="odd", values=[E.ExprRange(lower=_c(0), upper=None)],
                 array=True, with_expr=E.ExprBin(lhs=_cp_ref("item"),
                                                 op=E.BinOp.Mod, rhs=_c(2)))
    dflt = BinDef(name="other", is_default=True)
    doc = covg_doc(_cg([lo, two, odd, dflt]), "pss_top::A::cs", [(4, False, None)])
    assert doc["name"] == "pss_top::A::cs" and doc["instance"] == "cs"
    (cp,) = doc["coverpoints"]
    assert (cp["width"], cp["signed"], cp["iff"]) == (4, False, False)
    assert cp["bins"] == [
        {"name": "lo", "kind": "bins", "values": [[0, 7]], "array": False},
        {"name": "two", "kind": "ignore", "values": [2], "array": False},
        {"name": "odd", "kind": "bins", "values": [[0, None]], "array": True,
         "with": {"op": "Mod", "args": [{"ref": "item"}, {"c": 2}]}},
        {"name": "other", "kind": "default", "values": [], "array": False},
    ]


def test_cross_select_doc():
    cr = CrossDef(name="ab", coverpoints=["cp_x", "cp_x"],
                  bins=[CrossBinDef(name="ig", kind=BinKind.IGNORE,
                                    select=E.ExprBin(lhs=_cp_ref("cp_x"),
                                                     op=E.BinOp.Eq, rhs=_c(1)))])
    doc = covg_doc(_cg([], [cr]), "A::cs", [(4, False, None)])
    assert doc["crosses"] == [{"name": "ab", "coverpoints": ["cp_x", "cp_x"],
                               "iff": False,
                               "bins": [{"name": "ig", "kind": "ignore",
                                         "select": {"op": "Eq", "args": [
                                             {"ref": "cp_x"}, {"c": 1}]}}]}]


def test_non_constant_bin_value_is_an_error():
    b = BinDef(name="v", values=[_cp_ref("lim")])
    with pytest.raises(LoweringError, match="bin 'v'"):
        covg_doc(_cg([b]), "A::cs", [(4, False, None)])


def test_unsupported_with_expr_is_an_error():
    b = BinDef(name="v", values=[_c(1)], array=True,
               with_expr=E.ExprIn(value=_cp_ref("item"), container=_c(1)))
    with pytest.raises(LoweringError, match="ExprIn"):
        covg_doc(_cg([b]), "A::cs", [(4, False, None)])
