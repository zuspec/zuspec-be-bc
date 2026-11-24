"""Debug events (pssc P1.6a): the per-coroutine mask, ``stmt``,
``call``/``return``, host events, and that they leave the trace alone."""

import io
import json

import pytest

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import stmt as S
from zuspec.ir.core import expr as E

from zuspec.be.bc.interp import oracle
from zuspec.be.bc.interp.vm import VM
from zuspec.be.bc.lower import lower_scenario
from zuspec.be.bc.interp.defuse import defuse
from zuspec.be.bc.model import CoroDescriptor, Instr, LocalVar, Op, ZbcModel
from zuspec.be.bc.trace import CallbackSink, FileSink, MemorySink
from zuspec.be.bc.trace import events as ev


def _loop_model():
    # acc = 0; repeat 5: acc = acc + 1; return acc
    def assign(v):
        return SC.ScExecBlock(kind="body", stmts=[S.StmtAssign(
            targets=[E.ExprRefLocal(name="acc")], value=v)])
    coro = SC.ScCoroutine(name="root", frame_locals=["acc"], body=[
        assign(E.ExprConstant(value=0)),
        SC.ScLoop(kind="repeat", count=E.ExprConstant(value=5), body=[
            assign(E.ExprBin(lhs=E.ExprRefLocal(name="acc"), op=E.BinOp.Add,
                             rhs=E.ExprConstant(value=1)))]),
        SC.ScExecBlock(kind="body", stmts=[
            S.StmtReturn(value=E.ExprRefLocal(name="acc"))]),
    ])
    m = lower_scenario([coro])
    code = m.coros[0].code
    # Statement starts by hand (as debug lowering would): the first
    # instruction, the loop body's ADD, and the RET.
    add = next(i for i, x in enumerate(code) if x.op == Op.ADD)
    ret = next(i for i, x in enumerate(code) if x.op == Op.RET)
    m.coros[0].stmts = [(0, 1), (add, 2), (ret, 3)]
    return m


def _call_model():
    # f() { return 7; }   root: f(); return
    f = CoroDescriptor(name="f", src_ref=0, code=[
        Instr(Op.CONST, args=(0,), imm=7), Instr(Op.RET, args=(0,))],
        stmts=[(0, 5)])
    root = CoroDescriptor(name="root", code=[
        Instr(Op.CALL, args=(1, 0xFFFFFFFF)), Instr(Op.RET)])
    return ZbcModel(coros=[root, f], entry_coro=0)


def test_stmt_events_follow_execution():
    m = _loop_model()
    sink = MemorySink()
    res = oracle.run_model(m, events=sink, watch=ev.EV_STMT)
    assert res.retval == 5
    assert [e.src_ref for e in sink.events] == [1, 2, 2, 2, 2, 2, 3]
    assert {e.kind for e in sink.events} == {"stmt"}



def test_stmt_events_carry_defs_and_uses():
    # x = 3;  y = x + x;  return y   (slot 2 has no name: l#2)
    root = CoroDescriptor(name="root", code=[
        Instr(Op.CONST, args=(0,), imm=3), Instr(Op.ST_LOCAL, args=(0, 0)),
        Instr(Op.LD_LOCAL, args=(1, 0)), Instr(Op.ADD, args=(2, 1, 1)),
        Instr(Op.ST_LOCAL, args=(2, 1)), Instr(Op.ST_LOCAL, args=(2, 2)),
        Instr(Op.LD_LOCAL, args=(3, 1)), Instr(Op.RET, args=(3,))],
        frame_locals=["x", "y", "z"], stmts=[(0, 1), (2, 2), (6, 3)],
        locals=[LocalVar(0, "x", 0, 8, 32), LocalVar(1, "y", 2, 8, 32)])
    m = ZbcModel(coros=[root], entry_coro=0)
    assert defuse(root) == {0: (["l:x"], []), 2: (["l:y", "l#2"], ["l:x"]),
                            6: ([], ["l:y"])}
    sink = MemorySink()
    assert oracle.run_model(m, events=sink, watch=ev.EV_STMT).retval == 6
    assert [e.detail for e in sink.events] == [
        {"pc": 0, "defs": ["l:x"]},
        {"pc": 2, "defs": ["l:y", "l#2"], "uses": ["l:x"]},
        {"pc": 6, "uses": ["l:y"]}]


def test_events_leave_the_trace_unchanged():
    plain, on = MemorySink(), MemorySink()
    oracle.run_model(_loop_model(), sink=plain)
    oracle.run_model(_loop_model(), sink=on, events=MemorySink(),
                     watch=ev.EV_ALL)
    assert plain.serialize() == on.serialize()


def test_off_never_enters_the_event_loop(monkeypatch):
    m = _loop_model()
    vm = VM(m, events=MemorySink())
    entered = []
    monkeypatch.setattr(vm, "_run_frame_ev", lambda *a: entered.append(a))
    root = vm.root_frame(0, seed=0)
    vm.run(root)
    assert root.done and root.retval == 5 and not entered


def test_call_and_return():
    sink = MemorySink()
    res = oracle.run_model(_call_model(), events=sink,
                           watch=ev.EV_CALL | ev.EV_STMT)
    assert res.retval is None
    got = [(e.kind, e.detail) for e in sink.events]
    assert got == [("call", {"fn": "f", "depth": 1, "args": []}),
                   ("stmt", {"pc": 0}),
                   ("return", {"fn": "f", "depth": 1, "retval": 7})]


def test_a_bound_call_runs_the_host_and_is_mocked():
    # root: r = f(3); return r   with f bound to the host (pssc P1.8)
    f = CoroDescriptor(name="f", fn_qname="c::f", code=[
        Instr(Op.CONST, args=(0,), imm=7), Instr(Op.RET, args=(0,))])
    root = CoroDescriptor(name="root", code=[
        Instr(Op.CONST, args=(0,), imm=3), Instr(Op.ARG, args=(0, 0)),
        Instr(Op.CALL, args=(1, 1), src_ref=4), Instr(Op.RET, args=(1,))])
    m = ZbcModel(coros=[root, f], entry_coro=0)
    sink = MemorySink()
    vm = VM(m, events=sink)
    vm.watch(ev.EV_CALL | ev.EV_STMT)
    vm.bind("c::f", lambda x: x + 40)
    root_f = vm.root_frame(0, seed=0)
    vm.run(root_f)
    assert root_f.retval == 43
    assert [(e.kind, e.detail) for e in sink.events] == [
        ("call", {"fn": "f", "depth": 1, "mocked": True, "args": [3]}),
        ("return", {"fn": "f", "depth": 1, "mocked": True, "retval": 43})]
    vm = VM(m)                                  # a new run: unbound
    root_f = vm.root_frame(0, seed=0)
    vm.run(root_f)
    assert root_f.retval == 7
    with pytest.raises(KeyError):
        vm.bind("c::g", lambda: 0)


def test_mask_is_per_coroutine():
    m = _call_model()
    sink = MemorySink()
    vm = VM(m, events=sink)
    vm.watch(ev.EV_CALL, coros=["f"])
    vm.run(vm.root_frame(0, seed=0))
    assert [e.kind for e in sink.events] == ["call", "return"]
    sink2 = MemorySink()
    vm = VM(m, events=sink2)
    vm.watch(ev.EV_ALL, coros=[0])   # root only: no call of root
    vm.run(vm.root_frame(0, seed=0))
    # the root is an action run (F2.6b): its start and end only
    assert [(e.kind, e.detail["phase"]) for e in sink2.events] == [
        ("action", "start"), ("action", "end")]


def test_watch_off_again():
    sink = MemorySink()
    vm = VM(_loop_model(), events=sink)
    vm.watch(ev.EV_STMT)
    vm.watch(0)
    vm.run(vm.root_frame(0, seed=0))
    assert sink.events == [] and vm._ev == {}


def test_host_events_and_sinks():
    seen = []
    vm = VM(_loop_model(), events=CallbackSink(seen.append))
    vm.host_event(ev.TEST_START, {"name": "t"})      # off: dropped
    vm.watch(ev.EV_TEST)
    vm.host_event(ev.TEST_START, {"name": "t"})
    vm.run(vm.root_frame(0, seed=0))
    vm.host_event(ev.TEST_END, {"name": "t"})
    assert [(e.kind, e.seq) for e in seen] == [("test.start", 0), ("test.end", 1)]
    buf = io.StringIO()
    oracle.run_model(_loop_model(), events=FileSink(buf), watch=ev.EV_STMT)
    lines = buf.getvalue().splitlines()
    assert json.loads(lines[0]) == {"zbc_trace": 2}
    assert [json.loads(x)["kind"] for x in lines[1:]] == ["stmt"] * 7


def test_watch_needs_a_sink_and_known_names():
    with pytest.raises(ValueError):
        VM(_loop_model()).watch(ev.EV_STMT)
    assert ev.mask_of(["stmt", "return"]) == ev.EV_STMT | ev.EV_CALL
    with pytest.raises(ValueError):
        ev.mask_of(["nope"])


# --- P1.6b: access, import, message, check --------------------------------- #

from zuspec.be.bc.interp.extern import Memory, RecordingImportProvider
from zuspec.be.bc.model import (BUILTIN_MESSAGE, BUILTIN_READ, BUILTIN_WRITE,
                                BUILTIN_ERROR_MSG, INSTR_F_HAS_RET)
from zuspec.be.bc.interp.ops_proc import VMError

_NONE = 0xFFFFFFFF


def _io_model():
    # write32(0x100, 0x55); r = read32(0x100); x = ext(r); message(0, "r=%d", x)
    # (r4 = 0 is both the level and the message index)
    code = [
        Instr(Op.CONST, args=(0,), imm=0x100, src_ref=11),
        Instr(Op.CONST, args=(1,), imm=0x55, src_ref=11),
        Instr(Op.IMPORT, args=(BUILTIN_WRITE[32], _NONE, 0, 1), src_ref=11),
        Instr(Op.IMPORT, args=(BUILTIN_READ[32], 2, 0), flags=INSTR_F_HAS_RET,
              src_ref=12),
        Instr(Op.IMPORT, args=(0, 3, 2), flags=INSTR_F_HAS_RET, src_ref=13),
        Instr(Op.CONST, args=(4,), imm=0, src_ref=14),
        Instr(Op.ST_LOCAL, args=(4, 0)),
        Instr(Op.ST_LOCAL, args=(3, 1)),
        Instr(Op.IMPORT, args=(BUILTIN_MESSAGE, _NONE, 4), src_ref=14),
        Instr(Op.RET, args=(3,)),
    ]
    root = CoroDescriptor(name="root", code=code, frame_locals=["lvl", "x"])
    m = ZbcModel(coros=[root], entry_coro=0)
    m.messages = [{"fmt": "r=%d", "args": [{"kind": "int", "width": 32}],
                   "slots": [0, 1]}]
    m.imports = {"ext": {"fn_id": 0, "blocking": False}}
    return m


class _Mocks(RecordingImportProvider):
    def is_mocked(self, fn_id):
        return fn_id == 0


def _io_events(watch, provider=None):
    sink, out = MemorySink(), []
    res = oracle.run_model(_io_model(), events=sink, watch=watch, memory=Memory(),
                           import_provider=provider or RecordingImportProvider(
                               {0: lambda a: a[0] + 1}),
                           out=out.append)
    return res, sink.events, out


def test_access_import_message_events():
    res, evs, out = _io_events(ev.EV_ALL & ~ev.EV_ACTION)
    assert res.retval == 0x56 and out == ["r=86"]
    assert [(e.kind, e.src_ref) for e in evs] == [
        ("access", 11), ("access", 12), ("import", 13), ("message", 14)]
    assert evs[0].detail == {"rw": "w", "width": 32, "addr": 0x100, "data": 0x55}
    assert evs[1].detail == {"rw": "r", "width": 32, "addr": 0x100, "data": 0x55}
    assert evs[2].detail == {"name": "ext", "fn_id": 0, "args": [0x55],
                             "blocking": False, "mocked": False, "ret": 0x56}
    assert evs[3].detail == {"severity": "info", "level": 0, "text": "r=86"}


def test_each_kind_has_its_own_bit():
    for name in ("access", "import", "message"):
        _, evs, _ = _io_events(ev.mask_of([name]))
        assert {e.kind for e in evs} == {name}


def test_mocked_comes_from_the_provider():
    _, evs, _ = _io_events(ev.EV_IMPORT, _Mocks({0: 1}))
    assert evs[0].detail["mocked"] is True


def test_io_events_leave_the_trace_unchanged():
    plain, on = MemorySink(), MemorySink()
    for s, w in ((plain, 0), (on, ev.EV_ALL)):
        oracle.run_model(_io_model(), sink=s, memory=Memory(), out=lambda l: None,
                         import_provider=RecordingImportProvider(),
                         events=MemorySink() if w else None, watch=w)
    assert plain.serialize() == on.serialize()


def test_error_raises_a_message_event_before_it_stops():
    m = _io_model()
    m.coros[0].code[8] = Instr(Op.IMPORT, args=(BUILTIN_ERROR_MSG, _NONE, 4), src_ref=14)
    sink = MemorySink()
    with pytest.raises(VMError):
        oracle.run_model(m, events=sink, watch=ev.EV_MESSAGE, memory=Memory(),
                         import_provider=RecordingImportProvider())
    assert sink.events[-1].detail == {"severity": "error", "text": "r=0"}


def test_check_events_from_a_named_function():
    # check(a, b) { return; }   root: check(1, 1); check(1, 2)
    chk = CoroDescriptor(name="check", src_ref=0, code=[Instr(Op.RET)])
    root = CoroDescriptor(name="root", code=[
        Instr(Op.CONST, args=(0,), imm=1), Instr(Op.CONST, args=(1,), imm=2),
        Instr(Op.ARG, args=(0, 0)), Instr(Op.ARG, args=(0, 1)),
        Instr(Op.CALL, args=(1, _NONE), src_ref=21),
        Instr(Op.ARG, args=(0, 0)), Instr(Op.ARG, args=(1, 1)),
        Instr(Op.CALL, args=(1, _NONE), src_ref=22), Instr(Op.RET)])
    m = ZbcModel(coros=[root, chk], entry_coro=0)
    sink = MemorySink()
    vm = VM(m, events=sink)
    vm.watch(ev.EV_CHECK)

    def hook(vm, frame):
        caller = frame.parent
        a, b = frame.args
        return ({"ok": a == b}, caller.coro.code[caller.pc - 1].src_ref)
    vm.on_check("check", hook)
    vm.run(vm.root_frame(0, seed=0))
    assert [(e.kind, e.src_ref, e.detail) for e in sink.events] == [
        ("check", 21, {"ok": True}), ("check", 22, {"ok": False})]


# --- P1.6c: chan, notify, sanitizer; schema v2 ------------------------------ #

def test_host_kinds_each_have_their_own_bit():
    seen = []
    vm = VM(_loop_model(), events=CallbackSink(seen.append))
    vm.watch(ev.EV_SANITIZER)
    vm.host_event(ev.NOTIFY, {"source": "irq", "op": "raise"})      # off
    vm.host_event(ev.SANITIZER, {"name": "unmapped", "verdict": "fail",
                                 "what": "0x40"}, src_ref=3)
    vm.host_event(ev.TEST_START, {"name": "t"})                     # off
    assert [(e.kind, e.src_ref) for e in seen] == [("sanitizer", 3)]
    vm.watch(ev.EV_ALL)
    vm.host_event(ev.NOTIFY, {"source": "irq", "op": "raise"})
    assert seen[-1].kind == "notify"


def test_bits_are_distinct_and_every_kind_has_one():
    from zuspec.be.bc.trace.schema import EventKind
    bits = [ev.EV_STMT, ev.EV_CALL, ev.EV_TEST, ev.EV_ACCESS, ev.EV_IMPORT,
            ev.EV_MESSAGE, ev.EV_CHECK, ev.EV_CHAN, ev.EV_NOTIFY, ev.EV_SANITIZER,
            ev.EV_BRANCH, ev.EV_RELOAD, ev.EV_ACTION, ev.EV_BENCH, ev.EV_SOLVE,
            ev.EV_COV]
    assert len(set(bits)) == len(bits) and sum(bits) == ev.EV_ALL
    assert all(k in ev.BITS for k in EventKind.DEBUG)


def test_chan_events_without_markers_are_none():
    # a model lowered without debug has no chan table: EV_CHAN raises nothing
    sink = MemorySink()
    oracle.run_model(_loop_model(), events=sink, watch=ev.EV_CHAN)
    assert sink.events == []
