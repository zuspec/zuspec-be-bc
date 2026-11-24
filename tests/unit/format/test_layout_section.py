"""The LAYOUT section: component slot names and types and instance paths
and types survive write and read (F1.7a)."""

import types

from zuspec.ir.core import data_type as DT

from zuspec.be.bc.format import spec
from zuspec.be.bc.interp.components import CompTable
from zuspec.be.bc.lower.driver import _comp_table
from zuspec.be.bc.model import CoroDescriptor, ZbcModel

ns = types.SimpleNamespace
U8 = {"kind": "int", "width": 8, "signed": False}
OPAQUE = {"kind": "opaque", "width": 0, "signed": False}


def _model(profile="codegen"):
    m = ZbcModel(coros=[CoroDescriptor(name="t_a")], profile=profile)
    m.components = CompTable(ns(
        root="top_c",
        fields=[ns(name=n, slot=i) for i, n in enumerate(["x", "sub.y", "sub.z"])],
        instances=[ns(base=0, path="", type_qname="top_c"),
                   ns(base=1, path="sub", type_qname="sub_c")]), None)
    m.components.slot_types = [U8, U8, OPAQUE]
    return m


def test_layout_round_trip():
    m = _model()
    back = ZbcModel.from_bytes(m.to_bytes()).components
    assert back.names == ["x", "sub.y", "sub.z"]
    assert back.slot_types == [U8, U8, OPAQUE]
    assert back.paths == ["", "sub"] and back.types == ["top_c", "sub_c"]
    assert back.root == "top_c" and back.bases == [0, 1]


def test_the_image_has_one_layout_record():
    img = _model().to_container()
    sec = [s for s in img.sections if s.kind == spec.sec_kind("ZBC_SEC_LAYOUT")]
    assert len(sec) == 1 and sec[0].count == 1
    assert sec[0].elem_size == spec.RECORDS_BY_NAME["zbc_layout"].size() == 12


def test_runtime_image_has_no_layout():
    m = _model(profile="runtime")
    m.coros[0].name = ""
    img = m.to_container()
    assert spec.sec_kind("ZBC_SEC_LAYOUT") not in [s.kind for s in img.sections]
    assert ZbcModel.from_bytes(m.to_bytes()).components.names == ["$c0", "$c1", "$c2"]


def test_lowering_fills_slot_types():
    tree = ns(root="top_c", fields=[
        ns(name="n", slot=0, datatype=DT.DataTypeInt(bits=8, signed=False)),
        ns(name="h", slot=1, datatype=None)],
        instances=[ns(base=0, path="", type_qname="top_c")])
    assert _comp_table(tree, None).slot_types == [U8, OPAQUE]
