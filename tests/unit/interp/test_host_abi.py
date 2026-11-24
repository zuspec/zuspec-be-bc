"""The host ABI seam (docs/spec/host-abi.md, P3.1b / T3.1)."""
import threading

import pytest

from zuspec.be.bc.interp.extern import Memory, RecordingImportProvider
from zuspec.be.bc.interp.host import (DONE, ERROR, N_IRQ, REQS, WAIT,
                                      HostCall, HostStateError, run_sync)
from zuspec.be.bc.interp.ops_proc import VMError
from zuspec.be.bc.interp.vm import VM
from zuspec.be.bc.interp import oracle
from zuspec.be.bc.model import (BUILTIN_ERROR_MSG, BUILTIN_MESSAGE, BUILTIN_READ,
                                BUILTIN_WRITE, INSTR_F_BLOCKING, INSTR_F_HAS_RET,
                                INSTR_F_SPIN, CoroDescriptor, Instr, Op, Prov,
                                ZbcModel)

_NONE = 0xFFFFFFFF


def _io_model():
    # write32(0x100, 0x55); r = read32(0x100); x = ext(r); message("r=%d", x); return x
    code = [
        Instr(Op.CONST, args=(0,), imm=0x100, src_ref=11),
        Instr(Op.CONST, args=(1,), imm=0x55, src_ref=11),
        Instr(Op.IMPORT, args=(BUILTIN_WRITE[32], _NONE, 0, 1), src_ref=11),
        Instr(Op.IMPORT, args=(BUILTIN_READ[32], 2, 0), flags=INSTR_F_HAS_RET,
              src_ref=12),
        Instr(Op.IMPORT, args=(0, 3, 2), flags=INSTR_F_HAS_RET | INSTR_F_BLOCKING,
              src_ref=13),
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
    m.imports = {"ext": {"fn_id": 0, "blocking": True}}
    return m


_cur = {}


def _run(call):
    """``call.run()``; at REQS, the one request a single thread makes."""
    r = call.run()
    if r.status == REQS:
        r = _cur[call] = call.take()
        assert call.take() is None
    return r


def _done(call, value=None):
    """Complete the request :func:`_run` took last."""
    call.complete(_cur.pop(call).req_id, value)


def _call(model, out=None, **kw):
    vm = VM(model, out=(out.append if out is not None else (lambda l: None)))
    return HostCall(vm, vm.root_frame(model.entry_coro, 0), **kw)


def test_access_and_import_yield_and_complete():
    out = []
    call = _call(_io_model(), out)
    r = _run(call)
    assert (r.status, r.kind, r.addr, r.size, r.data, r.src_ref) == \
        (REQS, "w", 0x100, 4, 0x55, 11)
    req = _cur[call]
    r = call.run()                             # not completed yet
    assert (r.status, r.on) == (WAIT, "REQS")
    with pytest.raises(HostStateError):
        call.complete(req.req_id, 1)           # a write has no value
    _done(call)
    r = _run(call)
    assert (r.status, r.kind, r.addr, r.size) == (REQS, "r", 0x100, 4)
    _done(call, 0x1_0000_0077)               # masked to the access size
    r = _run(call)
    assert (r.status, r.fn_id, r.name, r.args, r.blocking) == \
        (REQS, 0, "ext", [0x77], True)
    _done(call, 9)
    assert out == []                           # message() has not run yet
    r = _run(call)
    assert r.status == DONE and call.result == 9 and out == ["r=9"]
    with pytest.raises(HostStateError):
        _run(call)
    with pytest.raises(HostStateError):
        call.complete(req.req_id)              # no longer outstanding


def test_sync_loop_matches_vm_run():
    def provider():
        return RecordingImportProvider({0: lambda a: a[0] + 1})
    out1, mem1, p1 = [], Memory(), provider()
    res = oracle.run_model(_io_model(), memory=mem1, import_provider=p1,
                           out=out1.append)
    out2, mem2, p2 = [], Memory(), provider()
    vm = VM(_io_model(), memory=mem2, import_provider=p2, out=out2.append)
    call = HostCall(vm, vm.root_frame(0, 0))
    assert run_sync(call) == res.retval == 0x56
    assert out2 == out1 and mem2.log == mem1.log and p2.calls == p1.calls
    assert vm.host is None


def _wait_model():
    # wait 7; write32(0x10, 1); wait 3; return 5
    root = CoroDescriptor(name="root", code=[
        Instr(Op.WAIT, (), imm=7),
        Instr(Op.CONST, args=(0,), imm=0x10),
        Instr(Op.CONST, args=(1,), imm=1),
        Instr(Op.IMPORT, args=(BUILTIN_WRITE[32], _NONE, 0, 1)),
        Instr(Op.WAIT, (), imm=3),
        Instr(Op.CONST, args=(2,), imm=5),
        Instr(Op.RET, (2,)),
    ])
    return ZbcModel(coros=[root], entry_coro=0)


def test_wait_time_yields_the_delay():
    call = _call(_wait_model())
    seen = []
    while (r := _run(call)).status != DONE:
        seen.append((r.status, r.on, r.delay, call.vm.sched.now))
        if r.status == REQS:
            _done(call)
    assert seen == [(WAIT, "TIME", 7, 0), (REQS, "", 0, 7), (WAIT, "TIME", 3, 7)]
    assert call.vm.sched.now == 10 and call.result == 5
    assert run_sync(_call(_wait_model())) == 5


def _irq_model():
    # root spins until read32(0x200) != 0, then returns it;
    # coro 1 (the interrupt handler) writes 7 there.
    root = CoroDescriptor(name="root", code=[
        Instr(Op.CONST, args=(0,), imm=0x200),
        Instr(Op.IMPORT, args=(BUILTIN_READ[32], 1, 0), flags=INSTR_F_HAS_RET),
        Instr(Op.BRZ, args=(1, 4)),
        Instr(Op.RET, (1,)),
        Instr(Op.YIELD, flags=INSTR_F_SPIN),
        Instr(Op.BR, args=(1,)),
    ])
    isr = CoroDescriptor(name="isr", code=[
        Instr(Op.CONST, args=(0,), imm=0x200),
        Instr(Op.CONST, args=(1,), imm=7),
        Instr(Op.IMPORT, args=(BUILTIN_WRITE[32], _NONE, 0, 1)),
        Instr(Op.RET),
    ])
    return ZbcModel(coros=[root, isr], entry_coro=0)


def _serve(call, mem):
    """Answer accesses from ``mem`` until something else is asked; after an
    interrupt, the handler's request and the spinning thread's are both out."""
    while (r := call.run()).status == REQS:
        while (q := call.take()) is not None:
            if q.kind == "r":
                call.complete(q.req_id, mem.read(q.addr, q.size))
            else:
                mem.write(q.addr, q.size, q.data)
                call.complete(q.req_id)
    return r


def test_wait_event_then_irq_resumes():
    mem = Memory()
    call = _call(_irq_model(), irq_entry=1)
    r = _serve(call, mem)
    assert (r.status, r.on) == (WAIT, "EVENT")
    r = _serve(call, mem)                      # nothing notified: still waiting
    assert (r.status, r.on) == (WAIT, "EVENT")
    call.notify(N_IRQ, 0)
    r = _serve(call, mem)
    assert r.status == DONE and call.result == 7
    assert ("w", 0x200, 4, 7) in mem.log


def test_a_second_live_call_runs_only_its_own_threads():
    # A spins on 0x200 (WAIT EVENT); B, started meanwhile on the same VM,
    # writes it. B must not run A's thread (A's read would be B's access);
    # A sees the write at its next run.
    mem = Memory()
    a = _call(_irq_model())
    vm = a.vm
    assert (_serve(a, mem).status, a.request.on) == (WAIT, "EVENT")
    home = vm.sched
    b = HostCall(vm, vm.sched.new_frame(vm.model.coros[1], seed=0,
                                        obj=vm._new_obj(1)), construct=False)
    seen = []
    while (r := _run(b)).status == REQS:
        seen.append((r.kind, r.addr))
        mem.write(r.addr, r.size, r.data)
        _done(b)
    assert r.status == DONE and seen == [("w", 0x200)]
    b.end()
    assert vm.sched is home
    r = _serve(a, mem)
    assert r.status == DONE and a.result == 7
    a.end()
    assert vm._live_calls == []


def test_notify_handler_and_sync_deadlock():
    seen = []

    def on_notify(vm, kind, id_, value):
        seen.append((kind, id_, value))
        vm.memory.write(0x200, 4, value)

    mem = Memory()
    vm = VM(_irq_model(), memory=mem)
    call = HostCall(vm, vm.root_frame(0, 0), on_notify=on_notify)
    while (r := _run(call)).status != WAIT:
        _done(call, mem.read(r.addr, r.size))
    call.notify(2, 3, 11)
    assert run_sync(call) == 11 and seen == [(2, 3, 11)]
    with pytest.raises(VMError, match="deadlock"):
        run_sync(_call(_irq_model()))


def test_error_is_located_and_engine_survives():
    model = ZbcModel(coros=[CoroDescriptor(name="root", code=[
        Instr(Op.CONST, args=(0,), imm=0x10, src_ref=1),
        Instr(Op.IMPORT, args=(BUILTIN_READ[32], 1, 0), flags=INSTR_F_HAS_RET,
              src_ref=1),
        Instr(Op.CONST, args=(2,), imm=0, src_ref=2),
        Instr(Op.ST_LOCAL, args=(2, 0)),
        Instr(Op.IMPORT, args=(BUILTIN_ERROR_MSG, _NONE, 2), src_ref=2),
        Instr(Op.RET),
    ], frame_locals=["m"])], entry_coro=0)
    model.messages = [{"fmt": "bad", "args": [], "slots": [0]}]
    model.prov.entries = [Prov(), Prov(file="t.pss", line=3, col=5),
                          Prov(file="t.pss", line=4, col=9)]
    call = _call(model)
    assert _run(call).status == REQS
    _done(call, 0)
    r = _run(call)
    assert r.status == ERROR and isinstance(r.exc, VMError)
    assert r.error["message"] == "error: bad"
    assert (r.error["src_ref"], r.error["file"], r.error["line"], r.error["col"],
            r.error["call"]) == (2, "t.pss", 4, 9, "root")
    assert call.vm.host is None
    with pytest.raises(VMError, match="error: bad"):
        run_sync(_call(model))


def test_end_cancels_a_suspended_call():
    call = _call(_io_model())
    assert _run(call).status == REQS
    n = threading.active_count()
    call.end()
    assert threading.active_count() == n - 1 and call.vm.host is None
    with pytest.raises(HostStateError):
        _run(call)


def test_platform_error_is_raised_in_the_vm():
    # run_sync hands a provider's error back to the VM thread (fail):
    # it comes out of the call as it would from a direct call
    class Boom(Exception):
        pass

    def ext(a):
        raise Boom("ext failed")
    vm = VM(_io_model(), import_provider=RecordingImportProvider({0: ext}),
            out=lambda l: None)
    call = HostCall(vm, vm.root_frame(0, 0))
    with pytest.raises(Boom, match="ext failed"):
        run_sync(call)
    assert call.over and vm.host is None
    with pytest.raises(HostStateError):
        call.fail(0, Boom())                   # nothing outstanding


def test_construct_false_keeps_the_tree():
    # a function called on a tree built before: no construction
    vm = VM(_io_model(), out=lambda l: None,
            import_provider=RecordingImportProvider({0: lambda a: a[0] + 1}))
    root = vm.root_frame(0, 0)
    root.cobj = object()                       # a tree that must not be rebuilt
    built = []
    vm.construct_components = lambda: built.append(1)
    call = HostCall(vm, root, construct=False)
    assert run_sync(call) == 0x56 and built == []
