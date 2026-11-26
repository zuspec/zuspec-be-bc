"""zsp_rt_bc's method list (``share/rt_bc_api.py``; pssc pssvm-plan step 7).

Both variants of ``zsp_rt_bc_pkg`` take every method from one list: the
interface class, ``zsp_rt_bc_base``'s defaults and the pyvm variant are
generated from it, and the shim builds its pyhdl-if class from it. The
generated files must be current, and each declaration must say what the
list says. (zuspec-rt-eng checks its C variant against the same list.)
"""
import re
from pathlib import Path

import pytest

from zuspec.be.bc.share import rt_bc_api, sv_dir


def sv_methods(text: str, cls: str = None) -> dict:
    """``name -> normalized signature`` of the functions declared in
    ``text`` (in class ``cls``'s body when given)."""
    if cls is not None:
        m = re.search(rf"\bclass {cls}\b.*?\bendclass\b", text, re.S)
        assert m, cls
        text = m.group(0)
    out = {}
    for m in re.finditer(r"\b(?:pure\s+|protected\s+)?(?:virtual|static)\s+"
                         r"function\s+([^;(]*?)\s*\(([^)]*)\)\s*;", text, re.S):
        head = " ".join(m.group(1).split())
        name = head.split()[-1]
        params = ", ".join(" ".join(p.split()) for p in m.group(2).split(",")
                           if p.strip())
        out[name] = f"function {head}({params})"
    return out


def test_generated_files_are_current():
    pytest.importorskip("hdl_if")
    stale = [str(p) for p, text in rt_bc_api.generated().items()
             if not p.exists() or p.read_text() != text]
    assert not stale, ("regenerate: python -m zuspec.be.bc.share.rt_bc_api; "
                       f"stale: {stale}")


def test_interface_is_the_list():
    got = sv_methods((sv_dir() / "zsp_rt_bc_if.svh").read_text(), "zsp_rt_bc_if")
    want = {m.name: rt_bc_api.signature(m) for m in rt_bc_api.METHODS}
    assert got == want


def test_base_has_a_default_for_every_engine_method():
    got = sv_methods((sv_dir() / "zsp_rt_bc_defaults.svh").read_text())
    want = {m.name: rt_bc_api.signature(m) for m in rt_bc_api.METHODS
            if m.impl == "engine"}
    assert got == want
    common = sv_methods((sv_dir() / "zsp_rt_bc_common.svh").read_text(),
                        "zsp_rt_bc_base")
    for m in rt_bc_api.METHODS:
        if m.impl == "common":
            assert common.get(m.name) == rt_bc_api.signature(m), m.name


def test_pyvm_variant_serves_every_engine_method():
    got = sv_methods((sv_dir() / "zsp_rt_bc_pkg.sv").read_text(), "zsp_rt_bc")
    extra = {k: got.pop(k) for k in ("load", "impl")}
    assert extra["load"] == "function zsp_rt_bc load(string img, string opts = \"\")"
    want = {m.name: rt_bc_api.signature(m) for m in rt_bc_api.METHODS
            if m.impl == "engine" and m.py != ""}
    assert got == want


def test_the_shims_api_is_the_list():
    hif = pytest.importorskip("hdl_if")
    from zuspec.be.bc.share.py import pssvm_dpi_shim as shim
    api = shim.make_api(shim.Engine)
    names = [n for n, *_ in rt_bc_api.transport()]
    assert len(names) == len(set(names))
    for n in names:
        assert callable(getattr(api, n, None)), n
        assert callable(getattr(shim.Engine, n, None)), n


def test_no_method_is_an_sv_keyword():
    # `release` is one: the method is release_obj
    kw = {"release", "force", "assign", "new", "type", "event", "string",
          "process", "wait", "fork", "join", "final"}
    assert not kw & {m.name for m in rt_bc_api.METHODS}
