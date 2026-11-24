"""Trace writing, replay steps, trace diff and replay run from be-bc alone
(full-PSS plan F1.12).

The traces here are written by ``TraceWriter`` from hand-made events; pssc's
own tests record them from lowered PSS runs."""

import io
import subprocess
import sys
from types import SimpleNamespace

import pytest

from zuspec.be.bc.trace import TraceEvent, diff
from zuspec.be.bc.trace.replay import ReplayDivergence, ReplayPlatform
from zuspec.be.bc.trace.sanitizers import latches
from zuspec.be.bc.trace.steps import load_trace, steps, trace_for
from zuspec.be.bc.trace.writer import TraceWriter, decode_access, trace_path

#: a register ``top.ctrl`` (id 1) with fields ``en`` [0] and ``mode`` [2:1]
MODEL = SimpleNamespace(registers={1: {"path": "top.ctrl", "fields": [
    {"name": "en", "lsb": 0, "width": 1},
    {"name": "mode", "lsb": 1, "width": 2}]}})


def _acc(seq, rw, addr, data, **kw):
    return TraceEvent(seq=seq, kind="access", detail=dict(
        rw=rw, width=32, addr=addr, data=data, **kw))


def _imp(seq, name, ret=0):
    return TraceEvent(seq=seq, kind="import", detail={"name": name, "ret": ret})


def _write(path, events, **header):
    with open(path, "w") as f:
        w = TraceWriter(f, MODEL, **header)
        for e in events:
            w.emit(e)
    return w


RUN = [_acc(0, "w", 0x10, 5, reg_id=1), _imp(1, "pkg::rt_log"),
       _imp(2, "pkg::kick", 3), _acc(3, "r", 0x14, 7)]


def test_writer_names_registers_and_writes_a_header(tmp_path):
    p = str(tmp_path / "a.ndjson")
    w = _write(p, RUN, test="c::t", seed=4)
    assert w.count == 4
    header, st = load_trace(p)
    assert header["test"] == "c::t" and header["seed"] == 4
    assert st[0]["reg"] == "top.ctrl" and st[0]["fields"] == {"en": 1, "mode": 2}
    assert decode_access(MODEL, {"addr": 0}) == {"addr": 0}


def test_steps_keep_only_user_imports(tmp_path):
    p = str(tmp_path / "a.ndjson")
    _write(p, RUN)
    assert [s["kind"] for s in load_trace(p)[1]] == ["access", "import",
                                                     "import", "access"]
    user = lambda n: not n.endswith("rt_log")  # noqa: E731
    assert [s.get("name") for s in load_trace(p, user)[1]] == [
        None, "pkg::kick", None]


def test_diff_of_a_trace_with_itself_is_none(tmp_path):
    p = str(tmp_path / "a.ndjson")
    _write(p, RUN)
    assert diff(p, p) is None


def test_diff_finds_the_first_different_step(tmp_path):
    a, b = str(tmp_path / "a.ndjson"), str(tmp_path / "b.ndjson")
    _write(a, RUN)
    _write(b, RUN[:3] + [_acc(3, "r", 0x14, 8)])  # the read returns 8, not 7
    d = diff(a, b)
    assert d["step"] == 4 and d["a"]["data"] == 7 and d["b"]["data"] == 8
    short = steps([e.to_dict() for e in RUN[:2]])
    d = diff(a, short)
    assert d["step"] == 3 and d["a"]["name"] == "pkg::kick" and d["b"] is None


def test_replay_answers_reads_and_imports_then_diverges(tmp_path):
    _write(trace_path(str(tmp_path), "c::t"), RUN, test="c::t")
    path = trace_for(str(tmp_path), "c::t")
    assert path.endswith("c.t.ndjson")
    p = ReplayPlatform(path, "c::t",
                       user_import=lambda n: not n.endswith("rt_log"))
    p.write32(0x10, 5)
    assert p.kick() == 3
    assert p.left() is not None
    assert p.read32(0x14) == 7 and p.left() is None
    q = ReplayPlatform(path, "c::t")
    with pytest.raises(ReplayDivergence, match="expected write32 0x10 = 0x5"):
        q.write32(0x10, 6)
    with pytest.raises(ValueError, match="not c::u"):
        ReplayPlatform(path, "c::u")


def test_latches_are_try_only_channels():
    coro = SimpleNamespace(chans=[(0, 0, "a", "try_put"), (1, 0, "a", "try_get"),
                                  (2, 0, "b", "put"), (3, 0, "b", "try_get")])
    assert latches(SimpleNamespace(coros=[coro])) == {"a"}


def test_no_pssc():
    """Importing the writer, replay and sanitizers pulls in nothing of pssc."""
    code = ("import sys, zuspec.be.bc.trace.writer, zuspec.be.bc.trace.steps\n"
            "import zuspec.be.bc.trace.replay, zuspec.be.bc.trace.sanitizers\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'pssc'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]"
