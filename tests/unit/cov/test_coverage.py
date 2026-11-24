"""The coverage collector and its UCIS writer run from be-bc alone (full-PSS
plan F1.11).

The reports here are written by hand in the ``report()`` format; pssc's own
tests count them from lowered PSS runs."""

import subprocess
import sys

import pytest

from zuspec.be.bc.cov import VERSION, Coverage, point_id
from zuspec.be.bc.cov.coverage import _point_bin, _point_bins


def _report():
    return {
        "statements": {
            "c::f/0": {"kind": "stmt", "scope": "c::f", "path": "0",
                       "file": "a.pss", "line": 3, "hits": 2},
            "c::f/1": {"kind": "stmt", "scope": "c::f", "path": "1",
                       "file": "a.pss", "line": 4, "hits": 0},
            "t::g/0": {"kind": "stmt", "scope": "t::g", "path": "0",
                       "file": "t.pss", "line": 1, "hits": 1},
        },
        "arms": {
            "c::f/1/then": {"kind": "arm", "scope": "c::f", "path": "1",
                            "branch": "c::f/1", "arm": "then",
                            "file": "a.pss", "line": 4, "hits": 1},
            "c::f/1/else": {"kind": "arm", "scope": "c::f", "path": "1",
                            "branch": "c::f/1", "arm": "else",
                            "implicit": True, "hits": 0},
        },
        "registers": {
            "top.regs.csr": {"type": "csr_t", "comp": "dma_c", "reads": 1,
                             "writes": 2,
                             "fields": {"go": {"kind": "RW", "width": 1,
                                               "bins": {"bit0=1": 1}}}},
        },
        "envelope": {},
    }


def test_point_id():
    assert point_id("c::f", "2/then/0") == "c::f/2/then/0"
    assert point_id("c", "") == "c"


def test_merge_and_report():
    cov = Coverage()
    cov.merge(_report())
    cov.merge(_report())
    r = cov.report()
    assert r["version"] == VERSION
    assert r["statements"]["c::f/0"]["hits"] == 4
    assert r["registers"]["top.regs.csr"]["writes"] == 4
    assert r["registers"]["top.regs.csr"]["fields"]["go"]["bins"] == {"bit0=1": 2}
    assert r["summary"]["statements"] == {"total": 3, "hit": 2}
    assert r["summary"]["arms"] == {"total": 2, "hit": 1}
    # keep: the front end's boundary, a predicate on the scope
    kept = cov.report(keep=lambda s: not s.startswith("t::"))
    assert sorted(kept["statements"]) == ["c::f/0", "c::f/1"]


def test_envelope_bins():
    spec = {"kind": "int", "width": 2, "signed": False, "type": "bit[2]"}
    assert _point_bins(spec) == ["0", "1", "2", "3"]
    assert _point_bin(spec, 3) == "3"
    ops = {"c::f": {"params": [dict(spec, name="x", bins=True)],
                    "result": {"kind": "bool", "type": "bool"},
                    "crosses": [["x", "result"]]}}
    r = Coverage(ops=ops).report()
    e = r["envelope"]["c::f"]
    assert e["calls"] == 0
    assert list(e["params"]["x"]["bins"]) == ["0", "1", "2", "3"]
    assert r["summary"]["envelope"]["hit"] == 0


def test_cdb_round_trip(tmp_path):
    pytest.importorskip("covsight.core")
    from covsight.core.api import CoverTypeT, ScopeTypeT
    from covsight.core.ncdb.ncdb_reader import NcdbReader
    from zuspec.be.bc.cov.ucis import comp_of, write_cdb
    assert comp_of("c::f") == "c"
    out = tmp_path / "cov.cdb"
    write_cdb(_report(), str(out), tests=[
        {"name": "t1", "status": "pass", "seed": 1, "coverage": _report()}])
    db = NcdbReader().read(str(out))
    insts = {s.getScopeName(): s for s in db.scopes(ScopeTypeT.INSTANCE)}
    assert {"c", "t"} <= set(insts)
    blocks = {s.getScopeName(): s for s in insts["c"].scopes(ScopeTypeT.BLOCK)}
    stmts = {i.getName(): i.getCoverData().data
             for i in blocks["c::f"].coverItems(CoverTypeT.STMTBIN)}
    assert stmts == {"0": 2, "1": 0}


def test_from_traces(tmp_path):
    """Coverage replayed from recorded runs (F2.4): stmt and branch events
    count, others do not; a test per trace, named by its header or file."""
    import json
    from zuspec.be.bc.cov.coverage import from_traces

    def trace(name, header, events):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r) for r in
                               [dict(header, zbc_trace=2)] + events) + "\n")
        return str(p)
    stmt = {"kind": "stmt", "coro": 1, "detail": {"path": "body/0", "scope": "W"}}
    arm = {"kind": "branch", "detail": {"scope": "W", "branch": "body/1",
                                        "arm": "then"}}
    msg = {"kind": "message", "detail": {"text": "hi"}}
    a = trace("a.ndjson", {"test": "t::a", "seed": 5}, [stmt, msg, stmt, arm])
    b = trace("b.ndjson", {}, [stmt])
    cov, tests = from_traces([a, b])
    rep = cov.report()
    assert rep["statements"]["W/body/0"]["hits"] == 3
    assert rep["arms"]["W/body/1/then"]["hits"] == 1
    assert [(t["name"], t["seed"]) for t in tests] == [("t::a", 5), ("b", None)]
    assert tests[1]["coverage"]["statements"]["W/body/0"]["hits"] == 1


def test_from_traces_registers(tmp_path):
    """A traced access to a model's register (``reg_id``) counts as its
    read or write, with its field bins (F2.4b); other accesses do not."""
    import json, types
    from zuspec.be.bc.cov.coverage import from_traces
    reg = {"path": "top.regs.ctrl", "comp": "top.regs", "fields": [
        {"name": "en", "lsb": 0, "width": 1}]}
    model = types.SimpleNamespace(coros=[], registers={3: reg})
    acc = lambda rw, data, **k: {"kind": "access", "detail": dict(
        rw=rw, width=32, addr=0x10, data=data, **k)}
    p = tmp_path / "a.ndjson"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"zbc_trace": 2}, acc("w", 1, reg_id=3), acc("r", 1, reg_id=3),
        acc("r", 0, reg_id=3), acc("w", 7)]) + "\n")
    cov, _ = from_traces([str(p)], model)
    r = cov.report()["registers"]["top.regs.ctrl"]
    assert (r["reads"], r["writes"]) == (2, 1)
    assert r["fields"]["en"]["bins"]["bit0=1"] == 2
    assert r["fields"]["en"]["bins"]["bit0=0"] == 1


def test_from_traces_covergroups(tmp_path):
    """A trace's ``cov`` records are binned against the model's COVG
    descriptions (F4.7): per trace, and summed over the traces."""
    import json, types
    from zuspec.be.bc.cov.coverage import from_traces
    covg = {"name": "A::cs", "instance": "cs", "crosses": [], "coverpoints": [
        {"name": "cp_m", "width": 32, "signed": True, "iff": False, "bins": [],
         "items": [["SW", 0], ["HW", 1]]}]}
    model = types.SimpleNamespace(coros=[], registers={}, covgs=[covg])
    paths = []
    for name, vals in (("a", [1, 1]), ("b", [0])):
        p = tmp_path / f"{name}.ndjson"
        p.write_text("\n".join(json.dumps(r) for r in [{"zbc_trace": 2}] + [
            {"kind": "cov", "detail": {"covg": 0, "values": [v]}} for v in vals]) + "\n")
        paths.append(str(p))
    cov, tests = from_traces(paths, model)
    cg = cov.report()["covergroups"]["A::cs"]
    assert cg["samples"] == 3
    assert cg["coverpoints"]["cp_m"]["bins"] == {"auto[SW]": 1, "auto[HW]": 2}
    assert tests[1]["coverage"]["covergroups"]["A::cs"]["samples"] == 1


def test_no_pssc():
    """Importing the collector and the writer pulls in nothing of pssc."""
    code = ("import sys, zuspec.be.bc.cov, zuspec.be.bc.cov.coverage\n"
            "try:\n import zuspec.be.bc.cov.ucis\nexcept ImportError: pass\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'pssc'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]"


def test_cdb_covergroups(tmp_path):
    """F4.5c: PSS covergroups go under their enclosing type's component, a
    bin's kind picks its UCIS cover type, a cross keeps its points."""
    pytest.importorskip("covsight.core")
    from covsight.core.api import CoverTypeT, ScopeTypeT
    from covsight.core.ncdb.ncdb_reader import NcdbReader
    from zuspec.be.bc.cov.ucis import write_cdb
    cg = {"samples": 3,
          "coverpoints": {
              "a": {"bins": {"lo": 2, "hi": 1, "bad": 0}, "kinds": {"bad": "illegal"}},
              "b": {"bins": {"z": 3, "rest": 0}, "kinds": {"rest": "default"}}},
          "crosses": {"ab": {"coverpoints": ["a", "b"],
                             "bins": {"<lo,z>": 2, "<hi,z>": 1, "skip": 0},
                             "kinds": {"skip": "ignore"}}}}
    out = tmp_path / "cov.cdb"
    write_cdb({"covergroups": {"c::A::cs": cg}}, str(out))
    db = NcdbReader().read(str(out))
    inst = {s.getScopeName(): s for s in db.scopes(ScopeTypeT.INSTANCE)}["c"]
    g = {s.getScopeName(): s for s in inst.scopes(ScopeTypeT.COVERGROUP)}["A::cs"]
    assert g.getAttribute("pssunit.samples") == "3"

    def items(p, kind):
        return {i.getName(): i.getCoverData().data for i in p.coverItems(kind)}
    pts = {s.getScopeName(): s for s in g.scopes(ScopeTypeT.COVERPOINT)}
    assert items(pts["a"], CoverTypeT.CVGBIN) == {"lo": 2, "hi": 1}
    assert items(pts["a"], CoverTypeT.ILLEGALBIN) == {"bad": 0}
    assert items(pts["b"], CoverTypeT.DEFAULTBIN) == {"rest": 0}
    x = {s.getScopeName(): s for s in g.scopes(ScopeTypeT.CROSS)}["ab"]
    assert items(x, CoverTypeT.CVGBIN) == {"<lo,z>": 2, "<hi,z>": 1}
    assert items(x, CoverTypeT.IGNOREBIN) == {"skip": 0}
