"""The engine debug stub (pssc P2.1b, T2.1): scripted request sequences on
small models, each step's reply and the events it causes. The cases are
data (``CASES``) so the native engine's conformance run (P4.5) can replay
them; an expected dict matches the keys it names (a subset match)."""

import json
import types

import pytest

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import stmt as S
from zuspec.ir.core import expr as E

from zuspec.be.bc.interp import Obj
from zuspec.be.bc.interp.components import CompTable
from zuspec.be.bc.interp.debug import Stub
from zuspec.be.bc.interp.stub_socket import SocketStub, serve_stub
from zuspec.be.bc.interp.vm import VM
from zuspec.be.bc.lower import lower_scenario
from zuspec.be.bc.model import (BUILTIN_READ, BUILTIN_WRITE, INSTR_F_HAS_RET, CoroDescriptor, Instr,
                                 LocalVar, Op, ZbcModel)
from zuspec.be.bc.trace import MemorySink


def loop_model():
    # acc = 0; repeat 5: acc = acc + 1; return acc
    # statements: pc 0 (src 1), the loop body's ADD pc 10 (src 2), RET pc 18 (src 3)
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
    assert code[10].op == Op.ADD and code[18].op == Op.RET
    m.coros[0].stmts = [(0, 1), (10, 2), (18, 3)]
    return m


def call_model():
    # f() { return 7; }  root: f(); return   (root stmts pc 0, 1; f's pc 0)
    f = CoroDescriptor(name="f", src_ref=0, code=[
        Instr(Op.CONST, args=(0,), imm=7), Instr(Op.RET, args=(0,))],
        stmts=[(0, 5)])
    root = CoroDescriptor(name="root", code=[
        Instr(Op.CALL, args=(1, 0xFFFFFFFF)), Instr(Op.RET)],
        stmts=[(0, 1), (1, 2)])
    return ZbcModel(coros=[root, f], entry_coro=0)


def par_model():
    # parallel { a = 1; b = 2 }; return 3
    # root: SPAWN 1, SPAWN 2, JOIN, CONST, RET (stmts pc 0, 3); branches stmt pc 0
    def set_field(idx, value):
        return SC.ScExecBlock(kind="body", stmts=[S.StmtAssign(
            targets=[E.ExprRefField(base=E.TypeExprRefSelf(), index=idx)],
            value=E.ExprConstant(value=value))])
    coro = SC.ScCoroutine(name="root", body=[
        SC.ScPar(branches=[set_field(0, 1), set_field(1, 2)]),
        SC.ScExecBlock(kind="body", stmts=[S.StmtReturn(value=E.ExprConstant(value=3))]),
    ])
    m = lower_scenario([coro])
    assert [i.op for i in m.coros[0].code] == [Op.SPAWN, Op.SPAWN, Op.JOIN, Op.CONST, Op.RET]
    m.coros[0].stmts = [(0, 1), (3, 4)]
    m.coros[1].stmts = [(0, 11)]
    m.coros[2].stmts = [(0, 12)]
    return m


def comp_model():
    # loop_model run in a component tree top (slots x, y) / top.sub (slot z),
    # with a debug local table and a word in memory
    m = loop_model()
    m.coros[0].locals = [LocalVar(slot=0, name="acc", pc_start=0, pc_end=19, width=32)]
    ns = types.SimpleNamespace
    m.components = CompTable(ns(
        fields=[ns(name=n, slot=i) for i, n in enumerate(["x", "y", "z"])],
        instances=[ns(base=0, path="top", type_qname="top_c"),
                   ns(base=2, path="top.sub", type_qname="sub_c")]), None)
    return m


def comp_setup(vm, root):
    root.cobj.values[:] = [1, 2, 3]
    root.comp, root.cbase = 1, 2
    vm.memory.write(0x100, 4, 0xDEADBEEF)


def mem_model():
    # write32(0x100, 0x55); r = read32(0x100); read8(0x200); return r
    # (each instruction's src_ref is 10 + its pc)
    V = 0xFFFFFFFF
    code = [Instr(Op.CONST, (0,), imm=0x100), Instr(Op.CONST, (1,), imm=0x55),
            Instr(Op.IMPORT, (BUILTIN_WRITE[32], V, 0, 1)),
            Instr(Op.IMPORT, (BUILTIN_READ[32], 2, 0), flags=INSTR_F_HAS_RET),
            Instr(Op.CONST, (3,), imm=0x200),
            Instr(Op.IMPORT, (BUILTIN_READ[8], 4, 3), flags=INSTR_F_HAS_RET),
            Instr(Op.RET, (2,))]
    for pc, ins in enumerate(code):
        ins.src_ref = 10 + pc
    return ZbcModel(coros=[CoroDescriptor(name="root", code=code)], entry_coro=0)


MODELS = {"loop": loop_model, "call": call_model, "par": par_model, "comp": comp_model,
          "mem": mem_model}
#: model key -> setup(vm, root frame), run before attach
SETUPS = {"comp": comp_setup}

# Thread/frame ids: the root frame is the first frame the run makes. Each
# case is (model, [(request, expected reply, [expected events])]); an
# expected reply is {"body": ...} or {"error": code}. "$root" in an event
# stands for the root thread's id, "$thread" for the last stopped thread.
ATTACH = ({"cmd": "attach", "args": {"version": 1}},
          {"body": {"version": 1, "engine": "pyvm", "image": {"coros": None}}}, [])
CONT_EXIT = ({"cmd": "continue"}, {"body": {}}, [{"event": "exited", "body": {"status": "ok"}}])


def _ins(op, *args, imm=0, flags=0):
    return {"op": op, "args": list(args), "imm": imm, "flags": flags}


def _stopped(reason, coro, pc, src_ref, **kw):
    body = {"reason": reason, "coro": coro, "pc": pc, "src_ref": src_ref}
    body.update(kw)
    return {"event": "stopped", "body": body}


CASES = {
    "run_to_end": ("loop", [ATTACH, CONT_EXIT]),
    "breakpoint_every_iteration": ("loop", [
        ATTACH,
        ({"cmd": "set_pc_breaks", "args": {"breaks": [
            {"id": 1, "coro": 0, "pc": 10}, {"id": 2, "coro": 0, "pc": 99},
            {"id": 3, "coro": 7, "pc": 0}]}},
         {"body": {"breaks": [{"id": 1, "verified": True},
                              {"id": 2, "verified": False, "message": "pc 99 out of range"},
                              {"id": 3, "verified": False, "message": "no coroutine 7"}]}}, []),
    ] + [({"cmd": "continue"}, {"body": {}},
          [_stopped("breakpoint", 0, 10, 2, hits=[1], detail={})])] * 5 + [CONT_EXIT]),
    "breaks_replace_the_set": ("loop", [
        ATTACH,
        ({"cmd": "set_pc_breaks", "args": {"breaks": [{"id": 1, "coro": 0, "pc": 10}]}},
         {"body": {}}, []),
        ({"cmd": "continue"}, {"body": {}}, [_stopped("breakpoint", 0, 10, 2, hits=[1])]),
        ({"cmd": "set_pc_breaks", "args": {"breaks": [{"id": 2, "coro": 0, "pc": 18}]}},
         {"body": {}}, []),
        ({"cmd": "continue"}, {"body": {}}, [_stopped("breakpoint", 0, 18, 3, hits=[2])]),
        CONT_EXIT]),
    "step_stmt_through_loop": ("loop", [
        ATTACH,
        ({"cmd": "step", "args": {"thread": "$root", "kind": "stmt"}}, {"body": {}},
         [_stopped("step", 0, 10, 2, detail={"kind": "stmt"}, thread="$root")]),
        ({"cmd": "step", "args": {"thread": "$root", "kind": "stmt"}}, {"body": {}},
         [_stopped("step", 0, 10, 2)]),
        CONT_EXIT]),
    "step_over_call": ("call", [
        ATTACH,
        ({"cmd": "step", "args": {"thread": "$root", "kind": "stmt"}}, {"body": {}},
         [_stopped("step", 0, 1, 2, frame="$root")]),
        CONT_EXIT]),
    "step_into_and_out": ("call", [
        ATTACH,
        ({"cmd": "step", "args": {"thread": "$root", "kind": "into"}}, {"body": {}},
         [_stopped("step", 1, 0, 5, thread="$root", detail={"kind": "into"})]),
        ({"cmd": "step", "args": {"thread": "$root", "kind": "out"}}, {"body": {}},
         [_stopped("step", 0, 1, 2, frame="$root", detail={"kind": "out"})]),
        CONT_EXIT]),
    "step_stmt_returns_to_caller": ("call", [
        ATTACH,
        ({"cmd": "step", "args": {"thread": "$root", "kind": "into"}}, {"body": {}},
         [_stopped("step", 1, 0, 5)]),
        ({"cmd": "step", "args": {"thread": "$root", "kind": "stmt"}}, {"body": {}},
         [_stopped("step", 0, 1, 2)]),
        CONT_EXIT]),
    "step_stays_on_thread": ("par", [
        ATTACH,
        ({"cmd": "step", "args": {"thread": "$root", "kind": "into"}}, {"body": {}},
         [_stopped("step", 0, 3, 4, thread="$root")]),
        CONT_EXIT]),
    "break_in_branch_thread": ("par", [
        ATTACH,
        ({"cmd": "set_pc_breaks", "args": {"breaks": [{"id": 1, "coro": 2, "pc": 0}]}},
         {"body": {}}, []),
        ({"cmd": "continue"}, {"body": {}}, [_stopped("breakpoint", 2, 0, 12, hits=[1])]),
        CONT_EXIT]),
    "step_off_ended_thread": ("par", [
        ATTACH,
        ({"cmd": "set_pc_breaks", "args": {"breaks": [{"id": 1, "coro": 1, "pc": 0}]}},
         {"body": {}}, []),
        ({"cmd": "continue"}, {"body": {}}, [_stopped("breakpoint", 1, 0, 11)]),
        ({"cmd": "set_pc_breaks", "args": {"breaks": []}}, {"body": {"breaks": []}}, []),
        ({"cmd": "step", "args": {"thread": "$thread", "kind": "stmt"}}, {"body": {}},
         [_stopped("step", 2, 0, 12)]),
        CONT_EXIT]),
    "errors": ("loop", [
        ({"cmd": "continue"}, {"error": "bad_request"}, []),
        ATTACH,
        ({"cmd": "attach", "args": {"version": 1}}, {"error": "bad_request"}, []),
        ({"cmd": "nope"}, {"error": "bad_request"}, []),
        ({"cmd": "stop"}, {"error": "not_running"}, []),
        ({"cmd": "step", "args": {"thread": "$root", "kind": "over"}}, {"error": "bad_request"}, []),
        ({"cmd": "step", "args": {"thread": 12345}}, {"error": "no_thread"}, []),
        ({"cmd": "set_pc_breaks", "args": {"breaks": [{"id": 1}]}}, {"error": "bad_request"}, []),
        CONT_EXIT,
        ({"cmd": "continue"}, {"error": "not_stopped"}, []),
        ({"cmd": "terminate"}, {"body": {}}, []),
    ]),
    "terminate_while_stopped": ("loop", [
        ATTACH,
        ({"cmd": "terminate"}, {"body": {}},
         [{"event": "exited", "body": {"status": "terminated"}}])]),
    "threads_and_frames_in_call": ("call", [
        ATTACH,
        ({"cmd": "step", "args": {"thread": "$root", "kind": "into"}}, {"body": {}},
         [_stopped("step", 1, 0, 5)]),
        ({"cmd": "threads"}, {"body": {"threads": [
            {"id": "$root", "coro": 0, "name": "root", "state": "running"}]}}, []),
        ({"cmd": "frames", "args": {"thread": "$root"}}, {"body": {"frames": [
            {"id": None, "coro": 1, "name": "f", "pc": 0, "src_ref": 5, "depth": 1},
            {"id": "$root", "coro": 0, "name": "root", "pc": 1, "src_ref": 2,
             "depth": 0}]}}, []),
        ({"cmd": "read_frame", "args": {"frame": "$root"}},
         {"body": {"locals": [], "args": [], "this": None, "regs": {}}}, []),
        ({"cmd": "frames", "args": {"thread": 999}}, {"error": "no_thread"}, []),
        ({"cmd": "read_frame", "args": {"frame": 999}}, {"error": "no_frame"}, []),
        ({"cmd": "read_comp", "args": {"comp": 0}}, {"error": "no_comp"}, []),
        CONT_EXIT]),
    "threads_of_a_parallel": ("par", [
        ATTACH,
        ({"cmd": "set_pc_breaks", "args": {"breaks": [{"id": 1, "coro": 2, "pc": 0}]}},
         {"body": {}}, []),
        ({"cmd": "continue"}, {"body": {}}, [_stopped("breakpoint", 2, 0, 12)]),
        ({"cmd": "threads"}, {"body": {"threads": [
            {"id": "$root", "coro": 0, "state": "waiting"},
            {"id": "$thread", "coro": 2, "state": "running"}]}}, []),
        ({"cmd": "frames", "args": {"thread": "$thread"}}, {"body": {"frames": [
            {"id": "$thread", "coro": 2, "pc": 0, "src_ref": 12, "depth": 0}]}}, []),
        ({"cmd": "frames", "args": {"thread": "$root"}}, {"body": {"frames": [
            {"id": "$root", "coro": 0, "pc": 3, "src_ref": 4}]}}, []),
        CONT_EXIT]),
    "read_locals_comp_memory": ("comp", [
        ATTACH,
        ({"cmd": "set_pc_breaks", "args": {"breaks": [{"id": 1, "coro": 0, "pc": 10}]}},
         {"body": {}}, []),
    ] + [({"cmd": "continue"}, {"body": {}}, [_stopped("breakpoint", 0, 10, 2)])] * 3 + [
        ({"cmd": "read_frame", "args": {"frame": "$root"}}, {"body": {
            "locals": [{"name": "acc", "value": 2, "width": 32, "signed": False,
                        "kind": "int"}],
            "args": [], "this": {"comp": 1, "path": "top.sub"}}}, []),
        ({"cmd": "read_comp", "args": {"comp": 0}}, {"body": {
            "comp": 0, "path": "top", "fields": [
                {"name": "x", "value": 1, "width": 64}, {"name": "y", "value": 2}]}}, []),
        ({"cmd": "read_comp", "args": {"path": "top.sub"}}, {"body": {
            "comp": 1, "fields": [{"name": "z", "value": 3}]}}, []),
        ({"cmd": "read_comp", "args": {"path": "top.nope"}}, {"error": "no_comp"}, []),
        ({"cmd": "backdoor_read", "args": {"addr": 0x100, "width": 32}},
         {"body": {"value": 0xDEADBEEF}}, []),
        ({"cmd": "backdoor_read", "args": {"addr": 0x102, "width": 8}},
         {"body": {"value": 0xAD}}, []),
        ({"cmd": "backdoor_read", "args": {"addr": 0x100, "width": 12}},
         {"error": "bad_request"}, []),
        ({"cmd": "set_pc_breaks", "args": {"breaks": []}}, {"body": {}}, []),
        CONT_EXIT,
        ({"cmd": "threads"}, {"error": "not_stopped"}, [])]),
    "exec_snippet": ("comp", [
        ATTACH,
        ({"cmd": "set_pc_breaks", "args": {"breaks": [{"id": 1, "coro": 0, "pc": 10}]}},
         {"body": {}}, []),
    ] + [({"cmd": "continue"}, {"body": {}}, [_stopped("breakpoint", 0, 10, 2)])] * 3 + [
        # acc + 1
        ({"cmd": "exec_snippet", "args": {"frame": "$root", "nregs": 3, "code": [
            _ins("LD_LOCAL", 0, 0), _ins("CONST", 1, imm=1), _ins("ADD", 2, 0, 1),
            _ins("RET", 2)]}}, {"body": {"value": 3}}, []),
        # this.z (top.sub's slot 0) and a back-door word
        ({"cmd": "exec_snippet", "args": {"frame": "$root", "nregs": 1, "code": [
            _ins("LD_COMP", 0, 0), _ins("RET", 0)]}}, {"body": {"value": 3}}, []),
        ({"cmd": "exec_snippet", "args": {"frame": "$root", "nregs": 2, "code": [
            _ins("CONST", 0, imm=0x100),
            _ins("IMPORT", BUILTIN_READ[32], 1, 0, flags=INSTR_F_HAS_RET),
            _ins("RET", 1)]}}, {"body": {"value": 0xDEADBEEF}}, []),
        # a branch, a bare RET
        ({"cmd": "exec_snippet", "args": {"frame": "$root", "nregs": 1, "code": [
            _ins("CONST", 0, imm=0), _ins("BRZ", 0, 3), _ins("RET", 0),
            _ins("RET")]}}, {"body": {"value": None}}, []),
        # writes, orchestration, bad registers / ops
        ({"cmd": "exec_snippet", "args": {"frame": "$root", "nregs": 1, "code": [
            _ins("CONST", 0, imm=7), _ins("ST_LOCAL", 0, 0), _ins("RET")]}},
         {"error": "snippet"}, []),
        ({"cmd": "exec_snippet", "args": {"frame": "$root", "nregs": 1, "code": [
            _ins("CONST", 0, imm=7), _ins("ST_COMP", 0, 0), _ins("RET")]}},
         {"error": "snippet"}, []),
        ({"cmd": "exec_snippet", "args": {"frame": "$root", "nregs": 1, "code": [
            _ins("YIELD"), _ins("RET")]}}, {"error": "snippet"}, []),
        ({"cmd": "exec_snippet", "args": {"frame": "$root", "nregs": 1, "code": [
            _ins("CONST", 5, imm=1), _ins("RET", 5)]}}, {"error": "snippet"}, []),
        ({"cmd": "exec_snippet", "args": {"frame": "$root", "nregs": 1, "code": [
            _ins("RET", 0)]}}, {"error": "snippet"}, []),
        ({"cmd": "exec_snippet", "args": {"frame": "$root", "nregs": 1, "code": [
            _ins("NOPE")]}}, {"error": "bad_request"}, []),
        # nothing changed: acc is still 2
        ({"cmd": "read_frame", "args": {"frame": "$root"}}, {"body": {
            "locals": [{"name": "acc", "value": 2}]}}, []),
        ({"cmd": "read_comp", "args": {"comp": 1}}, {"body": {
            "fields": [{"name": "z", "value": 3}]}}, []),
        ({"cmd": "set_pc_breaks", "args": {"breaks": []}}, {"body": {}}, []),
        CONT_EXIT]),
    "access_watch": ("mem", [
        ATTACH,
        ({"cmd": "set_access_watch", "args": {"watches": [
            {"id": 1, "addr": 0x100, "size": 4, "rw": "w"},
            {"id": 2, "addr": 0x103, "size": 1, "rw": "r"},
            {"id": 3, "addr": 0x200, "size": 1, "rw": "rw", "value": 1},
            {"id": 4, "addr": "x", "size": 1, "rw": "r"}]}},
         {"body": {"watches": [{"id": 1, "verified": True}, {"id": 2, "verified": True},
                               {"id": 3, "verified": True},
                               {"id": 4, "verified": False}]}}, []),
        ({"cmd": "set_stop_on", "args": {"kinds": ["check", "bogus", "error"]}},
         {"body": {"kinds": ["check", "error"]}}, []),
        ({"cmd": "set_stop_on", "args": {"kinds": "error"}}, {"error": "bad_request"}, []),
        ({"cmd": "continue"}, {"body": {}}, [_stopped("watch", 0, 3, 12, hits=[1], detail={
            "rw": "w", "width": 32, "addr": 0x100, "data": 0x55})]),
        ({"cmd": "continue"}, {"body": {}}, [_stopped("watch", 0, 4, 13, hits=[2], detail={
            "rw": "r", "width": 32, "addr": 0x100, "data": 0x55})]),
        # watch 3 wants the value 1; the byte at 0x200 is 0
        CONT_EXIT]),
    "detach_runs_unobserved": ("loop", [
        ATTACH,
        ({"cmd": "set_pc_breaks", "args": {"breaks": [{"id": 1, "coro": 0, "pc": 10}]}},
         {"body": {}}, []),
        ({"cmd": "detach"}, {"body": {}}, [])]),
}


def _subst(x, ids):
    if isinstance(x, str) and x.startswith("$"):
        return ids[x[1:]]
    if isinstance(x, dict):
        return {k: _subst(v, ids) for k, v in x.items()}
    if isinstance(x, list):
        return [_subst(v, ids) for v in x]
    return x


def _match(got, exp, where):
    if isinstance(exp, dict):
        assert isinstance(got, dict), (where, got)
        for k, v in exp.items():
            if v is None:
                assert k in got, (where, k, got)
            else:
                assert k in got, (where, k, got)
                _match(got[k], v, f"{where}.{k}")
    elif isinstance(exp, list):
        assert isinstance(got, list) and len(got) == len(exp), (where, got, exp)
        for i, (g, e) in enumerate(zip(got, exp)):
            _match(g, e, f"{where}[{i}]")
    else:
        assert got == exp, (where, got, exp)


def over_socket(stub):
    """``stub`` served as line JSON on one end of a socket pair; returns the
    client (a :class:`SocketStub`) and the server."""
    import socket
    a, b = socket.socketpair()
    return SocketStub(b, timeout=10.0), serve_stub(stub, a)


def run_case(model, script, setup=None, transport="direct"):
    """Play ``script`` against a fresh stub on ``model`` (``transport``
    "socket": through :func:`over_socket`); returns the stub, the root frame
    and the transcript (each reply and event, as sorted-key JSON lines)."""
    vm = VM(model)
    # par_model's branches write fields a, b of the root's object
    root = vm.root_frame(model.entry_coro, seed=0, obj=Obj(field_names=["a", "b"]))
    if setup is not None:
        setup(vm, root)
    real = Stub(vm, root)
    if transport == "socket":
        client, server = over_socket(real)
        try:
            lines = _play(client, script, {"root": root.id})
        finally:
            client.close()
            server.join(5.0)
        real.join()
        return real, root, lines
    return real, root, _play(real, script, {"root": root.id})


def _play(stub, script, ids):
    lines = []
    for seq, (req, exp, events) in enumerate(script):
        rep = stub.request(dict(_subst(req, ids), seq=seq))
        lines.append(json.dumps(rep, sort_keys=True, separators=(",", ":")))
        assert rep["seq"] == seq
        if "error" in exp:
            assert not rep["ok"] and rep["error"]["code"] == exp["error"], rep
        else:
            assert rep["ok"], rep
            _match(rep["body"], _subst(exp["body"], ids), f"reply {seq}")
        for e in events:
            got = stub.next_event()
            assert got is not None, f"no event after request {seq}"
            lines.append(json.dumps(got, sort_keys=True, separators=(",", ":")))
            _match(got, _subst(e, ids), f"event after {seq}")
            if got["event"] == "stopped":
                ids["thread"] = got["body"]["thread"]
    stub.join()
    assert stub.next_event(timeout=0.05) is None
    return lines


@pytest.mark.parametrize("transport", ["direct", "socket"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_scripted(name, transport):
    model, script = CASES[name]
    stub, root, lines = run_case(MODELS[model](), script, SETUPS.get(model), transport)
    if name != "terminate_while_stopped":
        assert root.done
    assert stub.vm._dbg is None


def test_run_result_is_unchanged():
    # A debugged run computes what an undebugged one does.
    stub, root, _ = run_case(loop_model(), CASES["breakpoint_every_iteration"][1])
    assert root.retval == 5


def test_transcript_is_deterministic():
    a = run_case(call_model(), CASES["step_into_and_out"][1])[2]
    b = run_case(call_model(), CASES["step_into_and_out"][1])[2]
    assert a == b


def test_socket_transcript_is_the_direct_one():
    # P4.5c2: line JSON on a socket changes nothing a debugger sees.
    for name in ("step_into_and_out", "breakpoint_every_iteration"):
        model, script = CASES[name]
        a = run_case(MODELS[model](), script, SETUPS.get(model))[2]
        b = run_case(MODELS[model](), script, SETUPS.get(model), "socket")[2]
        assert a == b, name


def test_socket_close_detaches_and_ends_the_client():
    # The debugger going away detaches the stub (the run goes on), and the
    # client sees one exited event; later requests fail as closed.
    vm = VM(loop_model())
    root = vm.root_frame(0, seed=0)
    real = Stub(vm, root)
    client, server = over_socket(real)
    assert client.request({"seq": 0, "cmd": "attach"})["ok"]
    assert client.state == "stopped"        # at entry (no event)
    server.close()          # the far end goes away
    ev = client.next_event()
    assert ev == {"event": "exited", "body": {"detached": True, "closed": True}}
    assert client.request({"seq": 1, "cmd": "threads"})["error"]["code"] == "closed"
    server.join(5.0)
    real.join()
    assert real.state in ("detached", "exited") and root.done and root.retval == 5


def test_stop_while_running():
    # A pause request lands on the next instruction boundary.
    m = loop_model()
    vm = VM(m)
    root = vm.root_frame(0, seed=0)
    stub = Stub(vm, root)
    stub.request({"seq": 0, "cmd": "attach"})
    with stub._cv:      # pause before resuming, so the stop is not racy
        stub._pause = True
    assert stub.request({"seq": 1, "cmd": "continue"})["ok"]
    ev = stub.next_event()
    assert ev["event"] == "stopped" and ev["body"]["reason"] == "pause"
    assert stub.request({"seq": 2, "cmd": "stop"})["error"]["code"] == "not_running"
    assert stub.request({"seq": 3, "cmd": "continue"})["ok"]
    assert stub.next_event()["body"] == {"status": "ok"}


def test_on_event_callback():
    seen = []
    vm = VM(loop_model())
    stub = Stub(vm, vm.root_frame(0, seed=0), on_event=seen.append)
    stub.request({"seq": 0, "cmd": "attach"})
    stub.request({"seq": 1, "cmd": "continue"})
    stub.join()
    assert seen == [{"event": "exited", "body": {"status": "ok"}}]


def test_engine_error_exits_with_error():
    bad = CoroDescriptor(name="root", code=[Instr(Op.CALL, args=(9, 0xFFFFFFFF))])
    vm = VM(ZbcModel(coros=[bad], entry_coro=0))
    stub = Stub(vm, vm.root_frame(0, seed=0))
    stub.request({"seq": 0, "cmd": "attach"})
    stub.request({"seq": 1, "cmd": "continue"})
    # "error" is in the default stop set: the run stops where it failed
    ev = stub.next_event()
    assert ev["event"] == "stopped" and ev["body"]["reason"] == "error", ev
    assert ev["body"]["pc"] == 0 and "error" in ev["body"]["detail"]
    assert stub.request({"seq": 2, "cmd": "frames", "args": {"thread": ev["body"]["thread"]}})["ok"]
    stub.request({"seq": 3, "cmd": "continue"})
    ev = stub.next_event()
    assert ev["event"] == "exited" and ev["body"]["status"] == "error"


def test_engine_error_without_stop_on_exits():
    bad = CoroDescriptor(name="root", code=[Instr(Op.CALL, args=(9, 0xFFFFFFFF))])
    vm = VM(ZbcModel(coros=[bad], entry_coro=0))
    stub = Stub(vm, vm.root_frame(0, seed=0))
    stub.request({"seq": 0, "cmd": "attach"})
    assert stub.request({"seq": 1, "cmd": "set_stop_on", "args": {"kinds": []}})["ok"]
    stub.request({"seq": 2, "cmd": "continue"})
    ev = stub.next_event()
    assert ev["event"] == "exited" and ev["body"]["status"] == "error"


def test_failing_sanitizer_stops_when_asked():
    from zuspec.be.bc.trace import events as tev
    sink = MemorySink()
    vm = VM(loop_model(), events=sink)
    vm.watch(tev.EV_SANITIZER)
    first, root = vm.root_frame(0, seed=0), vm.root_frame(0, seed=1)

    def run():
        # the run stops before first's first instruction; set_stop_on lands
        # then, so these events come after it
        vm.run(first)
        vm.host_event("sanitizer", {"name": "ok", "verdict": "pass"})
        vm.host_event("sanitizer", {"name": "w1c", "verdict": "fail"})
        return vm.run(root)
    stub = Stub(vm, first, run=run)
    stub.request({"seq": 0, "cmd": "attach"})
    stub.request({"seq": 1, "cmd": "set_stop_on", "args": {"kinds": ["sanitizer"]}})
    stub.request({"seq": 2, "cmd": "continue"})
    ev = stub.next_event()
    assert ev["event"] == "stopped" and ev["body"]["reason"] == "sanitizer", ev
    assert ev["body"]["detail"]["name"] == "w1c"
    stub.request({"seq": 3, "cmd": "continue"})
    assert stub.next_event()["body"]["status"] == "ok"
    # the events still reached the host's sink, and the sink is back
    assert [e.detail["name"] for e in sink.events if e.kind == "sanitizer"] == ["ok", "w1c"]
    assert vm.events is sink


def test_watch_unhooks_memory():
    vm = VM(mem_model())
    mem = vm.memory
    root = vm.root_frame(0, seed=0)
    stub = Stub(vm, root)
    stub.request({"seq": 0, "cmd": "attach"})
    stub.request({"seq": 1, "cmd": "set_access_watch", "args": {"watches": [
        {"id": 1, "addr": 0, "size": 1, "rw": "r"}]}})
    assert vm.memory is not mem
    stub.request({"seq": 2, "cmd": "set_access_watch", "args": {"watches": []}})
    assert vm.memory is mem
    stub.request({"seq": 3, "cmd": "continue"})
    stub.join()
    assert root.retval == 0x55


def test_no_stub_never_enters_the_event_loop(monkeypatch):
    vm = VM(call_model(), events=MemorySink())
    entered = []
    monkeypatch.setattr(vm, "_run_frame_ev", lambda *a: entered.append(a))
    root = vm.root_frame(0, seed=0)
    vm.run(root)
    assert root.done and not entered


def test_backdoor_read_is_not_an_access():
    # No bus-log entry; a platform's own ``peek`` wins over raw bytes.
    vm = VM(comp_model())
    root = vm.root_frame(0, seed=0)
    comp_setup(vm, root)
    stub = Stub(vm, root)
    stub.request({"seq": 0, "cmd": "attach"})
    n = len(vm.memory.log)
    rep = stub.request({"seq": 1, "cmd": "backdoor_read", "args": {"addr": 0x100, "width": 16}})
    assert rep["body"] == {"value": 0xBEEF} and len(vm.memory.log) == n
    vm.memory.peek = lambda addr, size: 42
    rep = stub.request({"seq": 2, "cmd": "backdoor_read", "args": {"addr": 0, "width": 64}})
    assert rep["body"] == {"value": 42}
    stub.request({"seq": 3, "cmd": "terminate"})
    stub.join()


def test_step_another_thread():
    # Stopped in one branch, step the root thread (listed by ``threads``).
    vm = VM(par_model())
    root = vm.root_frame(0, seed=0, obj=Obj(field_names=["a", "b"]))
    stub = Stub(vm, root)
    stub.request({"seq": 0, "cmd": "attach"})
    stub.request({"seq": 1, "cmd": "set_pc_breaks",
                  "args": {"breaks": [{"id": 1, "coro": 1, "pc": 0}]}})
    stub.request({"seq": 2, "cmd": "continue"})
    assert stub.next_event()["body"]["coro"] == 1
    stub.request({"seq": 3, "cmd": "set_pc_breaks", "args": {"breaks": []}})
    rep = stub.request({"seq": 4, "cmd": "step", "args": {"thread": root.id, "kind": "stmt"}})
    assert rep["ok"], rep
    ev = stub.next_event()["body"]
    assert (ev["reason"], ev["thread"], ev["pc"]) == ("step", root.id, 3)
    stub.request({"seq": 5, "cmd": "continue"})
    assert stub.next_event()["body"] == {"status": "ok"}


def _attached(model):
    vm = VM(model)
    root = vm.root_frame(model.entry_coro, seed=0, obj=Obj(field_names=["a", "b"]))
    stub = Stub(vm, root)
    assert stub.request({"seq": 0, "cmd": "attach"})["ok"]
    return stub, root


def _steps(model, n):
    """(tick, coro, pc) of the start and ``n`` statement steps."""
    stub, root = _attached(model)
    where = [(stub.tick, 0, stub.where[1])]
    for _ in range(n):
        stub.request({"cmd": "step", "args": {"thread": root.id, "kind": "into"}})
        ev = stub.next_event()
        if ev["event"] != "stopped":
            break
        where.append((ev["body"]["tick"], ev["body"]["coro"], ev["body"]["pc"]))
    assert len(where) > 1, where
    stub.request({"cmd": "terminate"})
    stub.join()
    return where


@pytest.mark.parametrize("model", ["loop", "call", "par"])
def test_ticks_count_statement_starts(model):
    where = _steps(MODELS[model](), 3)
    ticks = [w[0] for w in where]
    assert ticks[0] == 1 and ticks == sorted(set(ticks)), where


@pytest.mark.parametrize("model", ["loop", "call", "par"])
def test_run_to_stops_where_stepping_did(model):
    # A fresh run told to go to tick N stops where steps reached tick N.
    where = _steps(MODELS[model](), 3)
    for tick, coro, pc in where[1:]:
        stub, root = _attached(MODELS[model]())
        assert stub.request({"cmd": "run_to", "args": {"tick": tick}})["ok"]
        ev = stub.next_event()["body"]
        assert (ev["reason"], ev["tick"], ev["coro"], ev["pc"]) == ("goto", tick, coro, pc)
        h = stub.request({"cmd": "history"})["body"]
        assert h["tick"] == tick and [t for t, _, _ in h["history"]] == list(range(1, tick + 1))
        stub.request({"cmd": "terminate"})
        stub.join()


def test_run_to_without_breaks_passes_them():
    stub, root = _attached(loop_model())
    stub.request({"cmd": "set_pc_breaks", "args": {"breaks": [{"id": 1, "coro": 0, "pc": 10}]}})
    # ticks: 1 = pc 0, 2..6 = the loop body (pc 10), 7 = RET (pc 18)
    stub.request({"cmd": "run_to", "args": {"tick": 4, "breaks": False}})
    ev = stub.next_event()["body"]
    assert (ev["reason"], ev["tick"], ev["pc"]) == ("goto", 4, 10)
    stub.request({"cmd": "continue"})       # breaks live again
    ev = stub.next_event()["body"]
    assert (ev["reason"], ev["tick"], ev["hits"]) == ("breakpoint", 5, [1])
    # breaks live: the breakpoint at tick 6 comes before tick 7
    stub.request({"cmd": "run_to", "args": {"tick": 7}})
    ev = stub.next_event()["body"]
    assert (ev["reason"], ev["tick"]) == ("breakpoint", 6)
    rep = stub.request({"cmd": "run_to", "args": {"tick": 6}})
    assert rep["error"]["code"] == "bad_request"
    stub.request({"cmd": "set_pc_breaks", "args": {"breaks": []}})
    stub.request({"cmd": "continue"})
    assert stub.next_event()["body"] == {"status": "ok"}
    assert root.retval == 5


def test_snippet_value_and_runaway():
    from zuspec.be.bc.interp.debug import StubError, run_snippet
    from zuspec.be.bc.interp.scheduler import Frame
    f = Frame(id=0, coro=None, seed=None, locals=[5])
    code = [Instr(Op.LD_LOCAL, (0, 0)), Instr(Op.RET)]
    assert run_snippet(f, code, 1, None, None) is None
    with pytest.raises(StubError) as e:
        run_snippet(f, [Instr(Op.BR, (0,))], 1, None, None)
    assert e.value.code == "snippet"
    assert f.locals == [5] and f.regs == {}
