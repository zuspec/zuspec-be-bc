"""sv_entries: a SystemVerilog task per action entry of an image (F2.3b)."""
from types import SimpleNamespace

import pytest

from zuspec.be.bc import sv_entries
from zuspec.be.bc.model import EntryPoint


def _model(**entries):
    return SimpleNamespace(entries=entries)


def test_tasks_per_action():
    m = _model(
        X=EntryPoint(coro=0, qname="pss_top::X", params=[
            ("mode", {"kind": "int", "width": 8, "signed": False}),
            ("off", {"kind": "int", "width": 16, "signed": True}),
            ("output", {"kind": "bool", "width": 1, "signed": False})]),
        T=EntryPoint(coro=1, qname="pss_top::T"),
        f=EntryPoint(coro=2, kind="function", comp_type="pss_top"))
    text = sv_entries.emit(m, "my_pkg")
    assert "package my_pkg;" in text and "endpackage : my_pkg" in text
    assert ("task automatic pss_top_X(zsp_rt_bc_ctxt ctxt, input bit [7:0] mode, "
            "input bit signed [15:0] off, input bit output_, output bit ok);") in text
    assert "pss_args.push_back(64'(mode));" in text
    assert "pss_args.push_back(64'(longint'(off)));" in text
    assert 'ctxt.call("pss_top::X", "", pss_args, pss_ret, ok);' in text
    assert "task automatic pss_top_T(zsp_rt_bc_ctxt ctxt, output bit ok);" in text
    assert text.count("task automatic") == 2        # functions get none


def test_param_without_sv_type():
    m = _model(X=EntryPoint(coro=0, qname="pss_top::X", params=[
        ("h", {"kind": "opaque", "width": 0, "signed": False})]))
    with pytest.raises(ValueError, match="parameter 'h'"):
        sv_entries.emit(m)
