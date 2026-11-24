"""The simulation's Python host (``share/py/pssvm_dpi_shim``) is be-bc's
(full-PSS plan F1.13b): the SV runtime's ``PssvmEngine`` over ``Engine``,
loaded without pssc."""

import json
import os
import subprocess
import sys

import pytest

from zuspec.be.bc.share import py_dir
from zuspec.be.bc.share.py import pssvm_dpi_shim as shim


def test_host_ships_in_be_bc():
    assert (py_dir() / "pssvm_dpi_shim.py").is_file()


def test_generic_engine_reports_what_it_lacks():
    """A failed ``load`` is an error status, not a raise; with no image,
    calls fail likewise, and the generic engine has no daemon."""
    e = shim.Engine()
    assert e.load("x.zbc") == -1
    assert "x.zbc" in json.loads(e.error(0))["message"]
    assert e.call_start("f") == 0
    assert e.serve("s") == ""
    assert e.hook("setup", 0) == "" and e.test_name(0) == ""


def test_make_api_names_the_class_as_sv_expects():
    pytest.importorskip("hdl_if")

    class Mine(shim.Engine):
        pass

    api = shim.make_api(Mine)
    assert api.__qualname__ == "PssvmEngine"
    assert api.__module__ == __name__
    assert isinstance(api().e, Mine)


def test_loads_as_the_sv_module_without_pssc():
    """SV imports module ``pssvm_dpi_shim``: with ``py_dir()`` on the path
    it is this file, and pssc is not imported."""
    code = ("import sys, pssvm_dpi_shim as m; "
            "print(m.__file__, 'pssc' in sys.modules, "
            "m.hif is None or m.PssvmEngine.__qualname__)")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(py_dir())] + [p for p in sys.path if p]))
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=py_dir(),
                         capture_output=True, text=True, check=True).stdout.split()
    assert out[0] == str(py_dir() / "pssvm_dpi_shim.py")
    assert out[1] == "False"
    assert out[2] in ("True", "PssvmEngine")


# -- a .zbc run on its own (F1.13b2) ---------------------------------------- #

RUN_ONE = """
import sys
from zuspec.be.bc.interp import host as H
from zuspec.be.bc.share.py import pssvm_dpi_shim as shim
e = shim.Engine()
assert e.load(sys.argv[1]) == 0, e.error(0)
assert [e.test_name(i) for i in range(3)] == ["C::add", "C::init", ""]
h = e.call_start("C::init", "")
assert h and e.run(h) == H.REQS
rid = e.take(h)
assert rid >= 0 and e.take(h) == -1
assert e.req_int(h, rid, "api") == H.API_MEM
assert e.req_str(h, rid, "kind") == "w" and e.req_u64(h, rid, "addr") == 0x100
assert e.req_u64(h, rid, "data") == 0x41 and e.complete(h, rid, 0) == 0
assert e.run(h) == H.DONE
e.call_end(h)
assert e.fresh() == 0
e.push_arg(5); e.push_arg(7)
h = e.call_start("add", "")
assert h and e.run(h) == H.DONE and e.result(h) == 12, e.error(h)
print("pssc" in sys.modules)
"""


def _comp_image(tmp_path):
    """test_run's model in a one-instance tree (root ``C``), with ``add``
    and ``init`` as C's functions."""
    import types
    from zuspec.be.bc.interp.components import CompTable
    from zuspec.be.bc.model import EntryPoint
    from .test_run import _model

    ns = types.SimpleNamespace
    m = _model()
    u32 = {"kind": "int", "width": 32, "signed": False}
    m.entries["C::add"].params = [("a", u32), ("b", u32)]
    m.entries["C::init"] = EntryPoint(coro=1, kind="function", comp_type="C")
    m.components = CompTable(ns(root="C", fields=[ns(name="x", slot=0)],
                                instances=[ns(base=0, path="", type_qname="C")]),
                             None)
    path = tmp_path / "c.zbc"
    path.write_bytes(m.to_bytes())
    return str(path)


def test_generic_engine_runs_a_zbc_without_pssc(tmp_path):
    """The host loads any .zbc (its function entries are the tests) and
    runs one over the host ABI -- an access SV serves, then a value --
    with pssc never imported."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-c", RUN_ONE, _comp_image(tmp_path)],
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False"]


def test_generic_engine_traces_a_call(tmp_path):
    """``platform`` "" is the base platform; a trace of a
    call is written by be-bc's ``TraceWriter``."""
    e = shim.Engine()
    assert e.load(_comp_image(tmp_path), "", "") == 0, e.error(0)
    assert e.root == "C" and type(e.session.platform).__name__ == "Platform"
    path = e.trace(str(tmp_path / "t"), "C::add")
    assert path, e.error(0)
    e.push_arg(1); e.push_arg(2)
    h = e.call_start("C::add")
    assert e.run(h) == 0 and e.result(h) == 3
    e.call_end(h)
    assert e.trace("", "") == ""
    assert os.path.getsize(path) > 0
