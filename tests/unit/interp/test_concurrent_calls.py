"""Concurrent calls on one VM (docs/spec/host-abi.md "Concurrent calls";
pssc pssvm-plan step 5): one traversal at a time, later ones queued in FIFO
order; exported function calls alongside it and each other; an interrupt
is a function call, which may make blocking imports while the traversal
waits; a deadlock is found across calls; a traversal started inside the
active one's request is a self-wait, not a hang.
"""
from types import SimpleNamespace

import pytest

from zuspec.be.bc.interp.extern import Memory, RecordingImportProvider
from zuspec.be.bc.interp.host import (DONE, ERROR, REQS, WAIT, HostCall,
                                      calls_of, run_calls, run_sync)
from zuspec.be.bc.interp.ops_proc import VMError
from zuspec.be.bc.interp.vm import VM
from zuspec.be.bc.model import (BUILTIN_READ, BUILTIN_WRITE, INSTR_F_BLOCKING,
                                INSTR_F_HAS_RET, INSTR_F_SPIN, CoroDescriptor,
                                EntryPoint, Instr, Op, ZbcModel)

_NONE = 0xFFFFFFFF


def _write(addr, value, r=0):
    return [Instr(Op.CONST, args=(r,), imm=addr),
            Instr(Op.CONST, args=(r + 1,), imm=value),
            Instr(Op.IMPORT, args=(BUILTIN_WRITE[32], _NONE, r, r + 1))]


def _spin_on(addr, at=0):
    """Spin until read32(addr) != 0, then return it (a thread waiting on
    what only another call, or the host, can change); ``at``: the pc the
    code starts at."""
    return [Instr(Op.CONST, args=(0,), imm=addr),
            Instr(Op.IMPORT, args=(BUILTIN_READ[32], 1, 0), flags=INSTR_F_HAS_RET),
            Instr(Op.BRZ, args=(1, at + 4)),
            Instr(Op.RET, (1,)),
            Instr(Op.YIELD, flags=INSTR_F_SPIN),
            Instr(Op.BR, args=(1 + at,))]


def _import(fn, arg, rd=1, blocking=True):
    """``rd = import fn(arg)``"""
    return [Instr(Op.CONST, args=(0,), imm=arg),
            Instr(Op.IMPORT, args=(fn, rd, 0),
                  flags=INSTR_F_HAS_RET | (INSTR_F_BLOCKING if blocking else 0))]


def _model(*coros, functions=()):
    """Coroutines (name, code): actions, but those named in ``functions``,
    which are function entries."""
    m = ZbcModel(coros=[CoroDescriptor(name=n, code=c) for n, c in coros],
                 entry_coro=0)
    m.entries = {n: EntryPoint(coro=i, kind="function" if n in functions
                               else "action")
                 for i, (n, _) in enumerate(coros)}
    m.imports = {"ext": {"fn_id": 0, "blocking": True},
                 "wait_done": {"fn_id": 1, "blocking": True}}
    return m


def _traversal(vm, coro):
    return HostCall(vm, vm.root_frame(coro, 0))


def _take_all(call):
    out = []
    while (q := call.take()) is not None:
        out.append(q)
    return out


# -- one traversal slot, FIFO queue ---------------------------------------- #

def _three_writers():
    return _model(*[(f"A{k}", _write(0x100 + 4 * k, k + 1) + [Instr(Op.RET)])
                    for k in range(3)])


def test_a_second_traversal_waits_for_its_turn_in_fifo_order():
    vm = VM(_three_writers(), out=lambda l: None)
    a, b, c = (_traversal(vm, k) for k in range(3))
    assert all(x.traversal for x in (a, b, c))
    assert a.run().status == REQS                     # a holds the slot
    (wa,) = _take_all(a)
    for x in (c, b):                                  # c asks first: c, then b
        r = x.run()
        assert (r.status, r.on) == (WAIT, "TURN")
    t = calls_of(vm)
    assert t.active is a and list(t.queue) == [c, b]
    assert b.run().on == "TURN"                       # nothing changed
    a.complete(wa.req_id)
    assert a.run().status == DONE and t.active is None
    assert b.run().on == "TURN"                       # c is first in the queue
    assert c.run().status == REQS and t.active is c
    _, = _take_all(c)
    assert b.run().on == "TURN"
    c.end()                                           # an end frees it too
    assert b.run().status == REQS and t.active is b


def test_run_calls_runs_queued_traversals_one_after_another():
    mem = Memory()
    vm = VM(_three_writers(), memory=mem, out=lambda l: None)
    calls = [_traversal(vm, k) for k in range(3)]
    assert [r.status for r in run_calls(calls)] == [DONE] * 3
    assert [(a, v) for _, a, _, v in mem.log] == [
        (0x100, 1), (0x104, 2), (0x108, 3)]
    assert calls_of(vm).live == [] and calls_of(vm).active is None


def test_a_queued_traversal_keeps_its_own_pins():
    # VM.root_frame sets the entry's pins on the VM; the queued call's root
    # frame must not replace the active one's while it runs
    seen = []

    class _Mem(Memory):
        def write(self, addr, size, value):
            seen.append((addr, dict(vm.arg_bounds)))
            super().write(addr, size, value)

    vm = VM(_three_writers(), memory=_Mem(), out=lambda l: None)
    a = HostCall(vm, vm.root_frame(0, 0))
    a._state["arg_bounds"] = {0: (5, 5)}              # as an exported arg would
    b = _traversal(vm, 1)                             # resets vm.arg_bounds
    assert vm.arg_bounds == {}
    run_calls([a, b])
    assert seen == [(0x100, {0: (5, 5)}), (0x104, {})]


# -- function calls alongside ------------------------------------------- #

def test_function_calls_run_alongside_a_traversal_and_each_other():
    m = _model(("T", _import(1, 0) + [Instr(Op.RET, (1,))]),
               ("f", _write(0x200, 1) + [Instr(Op.RET)]),
               ("g", _write(0x204, 2) + [Instr(Op.RET)]),
               functions=("f", "g"))
    vm = VM(m, out=lambda l: None)
    t = _traversal(vm, 0)
    assert t.run().status == REQS
    (wait_done,) = _take_all(t)                       # held: the traversal waits
    f, g = (calls_of(vm).start_function(n) for n in ("f", "g"))
    assert not f.traversal and not g.traversal
    assert f.run().status == REQS and g.run().status == REQS
    (wf,), (wg,) = _take_all(f), _take_all(g)
    assert (wf.addr, wg.addr) == (0x200, 0x204)
    g.complete(wg.req_id)                             # in either order
    assert g.run().status == DONE
    f.complete(wf.req_id)
    assert f.run().status == DONE
    assert t.request.status == REQS and not t.over    # still parked
    t.complete(wait_done.req_id, 9)
    assert t.run().status == DONE and t.result == 9


def _irq_model():
    # T waits (blocking import wait_done), then returns read32(0x200);
    # isr, the interrupt handler, calls blocking import ext and writes its
    # result to 0x200.
    return _model(
        ("T", _import(1, 0, rd=2) + [
            Instr(Op.CONST, args=(0,), imm=0x200),
            Instr(Op.IMPORT, args=(BUILTIN_READ[32], 1, 0), flags=INSTR_F_HAS_RET),
            Instr(Op.RET, (1,))]),
        ("isr", _import(0, 3) + [
            Instr(Op.CONST, args=(0,), imm=0x200),
            Instr(Op.IMPORT, args=(BUILTIN_WRITE[32], _NONE, 0, 1)),
            Instr(Op.RET)]),
        functions=("isr",))


def test_an_irq_handler_makes_a_blocking_import_while_the_traversal_waits():
    mem = Memory()
    vm = VM(_irq_model(), memory=mem, out=lambda l: None)
    t = _traversal(vm, 0)
    assert t.run().status == REQS
    (wait_done,) = _take_all(t)
    isr = calls_of(vm).start_function("isr")
    assert isr.run().status == REQS
    (ext,) = _take_all(isr)
    assert (ext.name, ext.args, ext.blocking) == ("ext", [3], True)
    isr.complete(ext.req_id, 0x77)                    # the handler's import
    assert isr.run().status == REQS
    (w,) = _take_all(isr)
    mem.write(w.addr, w.size, w.data)
    isr.complete(w.req_id)
    assert isr.run().status == DONE
    t.complete(wait_done.req_id, 0)                   # the irq ended the wait
    assert t.run().status == REQS
    (r,) = _take_all(t)
    t.complete(r.req_id, mem.read(r.addr, r.size))
    assert t.run().status == DONE and t.result == 0x77


def test_run_sync_drives_an_irq_handler_the_host_calls():
    # a platform raising an interrupt from inside a request calls the
    # handler; run_sync adopts that call and runs it alongside, and T,
    # waiting on what the handler writes, wakes
    mem = Memory()
    calls = []

    def wait_done(args):
        calls.append(calls_of(vm).start_function("isr"))
        return 0

    m = _irq_model()
    m.coros[0].code = _import(1, 0, rd=2) + _spin_on(0x200, at=2)
    vm = VM(m, memory=mem, out=lambda l: None,
            import_provider=RecordingImportProvider({1: wait_done,
                                                     0: lambda a: a[0] + 1}))
    assert run_sync(_traversal(vm, 0)) == 4
    assert calls[0].request.status == DONE and calls_of(vm).live == []


# -- waits across calls ---------------------------------------------------- #

def _spinners():
    return _model(("T", _spin_on(0x200)), ("w", _spin_on(0x204)),
                  ("set", _write(0x200, 5) + [Instr(Op.RET)]),
                  ("T2", [Instr(Op.RET)]),
                  functions=("w", "set"))


def test_a_wait_another_call_ends_is_no_deadlock():
    mem = Memory()
    vm = VM(_spinners(), memory=mem, out=lambda l: None)
    t = _traversal(vm, 0)
    s = calls_of(vm).start_function("set")
    rs = run_calls([t, s])
    assert [r.status for r in rs] == [DONE, DONE] and t.result == 5


def test_a_deadlock_across_calls_ends_every_call_with_a_message():
    # a deadlock is a fatal (host-abi.md "Listeners"): a listener that only
    # records lets run_calls return every call's ERROR
    vm = VM(_spinners(), memory=Memory(), out=lambda l: None)
    fatals = []
    calls_of(vm).add_listener(SimpleNamespace(on_fatal=fatals.append))
    t = _traversal(vm, 0)
    w = calls_of(vm).start_function("w")
    t2 = _traversal(vm, 3)                            # queued behind t
    rs = run_calls([t, w, t2])
    assert [r.status for r in rs] == [ERROR] * 3
    assert len(fatals) == 1 and fatals[0].msg == rs[0].error["message"]
    msg = rs[0].error["message"]
    assert msg.startswith("deadlock:") and rs[1].error["message"] == msg
    assert "T (waits on an event)" in msg and "w (waits on an event)" in msg
    assert "T2 (queued behind traversal T)" in msg
    assert calls_of(vm).live == [] and calls_of(vm).active is None
    assert vm.host is None


def test_run_sync_reports_a_deadlock_across_calls():
    vm = VM(_spinners(), memory=Memory(), out=lambda l: None)
    calls_of(vm).start_function("w")                  # adopted, never wakes T
    with pytest.raises(VMError, match=r"deadlock: .*T \(waits on an event\)"):
        run_sync(_traversal(vm, 0))
    assert calls_of(vm).live == []


def test_a_traversal_started_inside_the_active_ones_request_is_a_self_wait():
    seen = []

    def wait_done(args):
        t2 = _traversal(vm, 0)
        r = t2.run()                                  # by hand: ERROR at once
        seen.append((r.status, r.error["message"]))
        run_sync(_traversal(vm, 0))                   # raises: the same
        return 0

    vm = VM(_irq_model(), memory=Memory(), out=lambda l: None,
            import_provider=RecordingImportProvider({1: wait_done}))
    with pytest.raises(VMError, match="self-wait: traversal T was started "
                       r"inside request \d+ \(wait_done\) of the active "
                       "traversal T"):
        run_sync(_traversal(vm, 0))
    assert seen[0][0] == ERROR and seen[0][1].startswith("self-wait")
    t = calls_of(vm)
    assert t.live == [] and t.active is None and not t.queue


def test_the_sv_shim_runs_a_handler_a_double_starts():
    # under SV (pyhdl-if shim) a double's ctx.raise_irq starts the handler;
    # the shim runs it at once, answering mocked imports; a request only SV
    # could serve fails it, with a message saying where to call it from
    from types import SimpleNamespace
    from zuspec.be.bc.share.py import pssvm_dpi_shim as shim

    class _Mocks(RecordingImportProvider):
        def is_mocked(self, fn):
            return fn == 0

    m = _irq_model()
    m.coros[1].code = _import(0, 3) + [Instr(Op.RET)]     # ext only
    vm = VM(m, memory=Memory(), out=lambda l: None,
            import_provider=_Mocks({0: lambda a: a[0] + 1}))
    eng = shim.Engine()
    eng.session = SimpleNamespace(vm=vm)
    isr = calls_of(vm).start_function("isr")
    eng._drive_adopted()
    assert isr.request.status == DONE and not calls_of(vm).adopt
    vm = VM(_irq_model(), memory=Memory(), out=lambda l: None,
            import_provider=_Mocks({0: lambda a: a[0] + 1}))
    eng.session = SimpleNamespace(vm=vm)
    calls_of(vm).start_function("isr")
    with pytest.raises(VMError, match="made a request SV serves under SV: "
                       "call the handler from the testbench"):
        eng._drive_adopted()
