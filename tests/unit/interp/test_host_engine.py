"""The engine-level host ABI on pyvm (docs/spec/host-abi.md "Objects",
"Linking", "Listeners", "Driving every call", "Completion"; pssc
pssvm-plan step 6): object handles, linked handlers and the synchronous
non-blocking import (construction may call one), listeners for a call's
error and the VM's fatal (which releases every blocked and queued call),
and step/pending over every call.
"""
import types

import pytest

from zuspec.be.bc.interp.components import CompTable
from zuspec.be.bc.interp.extern import Memory, RecordingImportProvider
from zuspec.be.bc.interp.host import (DEFER, DONE, ERROR, NO_EXECUTOR,
                                      PEND_READY, PEND_REQS, PEND_TIME, REQS,
                                      SEV_ERROR, SEV_FATAL, HostStateError,
                                      Listener, calls_of, run_calls, run_sync)
from zuspec.be.bc.interp.ops_proc import VMError
from zuspec.be.bc.interp.vm import VM
from zuspec.be.bc.model import (BUILTIN_READ, INSTR_F_BLOCKING,
                                INSTR_F_HAS_RET, INSTR_F_SPIN, CoroDescriptor,
                                EntryPoint, Instr, Op, ZbcModel)

ns = types.SimpleNamespace
EXT, CFG = 0, 2                 # blocking import (API p), non-blocking (API q)


def _import(fn, arg, rd=1, blocking=True):
    return [Instr(Op.CONST, args=(0,), imm=arg),
            Instr(Op.IMPORT, args=(fn, rd, 0),
                  flags=INSTR_F_HAS_RET | (INSTR_F_BLOCKING if blocking else 0))]


def _spin_on(addr, at=0):
    return [Instr(Op.CONST, args=(0,), imm=addr),
            Instr(Op.IMPORT, args=(BUILTIN_READ[32], 1, 0), flags=INSTR_F_HAS_RET),
            Instr(Op.BRZ, args=(1, at + 4)),
            Instr(Op.RET, (1,)),
            Instr(Op.YIELD, flags=INSTR_F_SPIN),
            Instr(Op.BR, args=(1 + at,))]


def _model(init=None):
    """A tree: root ``C`` (field ``x``, slot 0) with instance ``sub`` of
    ``S`` (field ``y``, slot 1). ``$comp_init`` is ``init`` (default: x =
    cfg(7), a non-blocking import). Entries: action ``A`` (a = ext(cfg(5)):
    its field ``a``), functions ``C::get`` (x), ``S::get`` (y) and
    ``C::spin`` (until read32(0x200) != 0)."""
    if init is None:
        init = _import(CFG, 7, blocking=False) + [
            Instr(Op.ST_COMP, args=(1, 0)), Instr(Op.RET)]
    act = (_import(CFG, 5, rd=2, blocking=False)
           + [Instr(Op.IMPORT, args=(EXT, 1, 2), flags=INSTR_F_HAS_RET | INSTR_F_BLOCKING),
              Instr(Op.ST_FIELD, args=(1, 0)), Instr(Op.RET)])
    get = [Instr(Op.LD_COMP, args=(0, 0)), Instr(Op.RET, (0,))]
    coros = [("A", act), ("C::get", get), ("S::get", get),
             ("C::spin", _spin_on(0x200)), ("$comp_init", init)]
    m = ZbcModel(coros=[CoroDescriptor(name=n, code=c) for n, c in coros],
                 entry_coro=0)
    m.entries = {
        "A": EntryPoint(coro=0, kind="action", qname="pss_top::A"),
        "C::get": EntryPoint(coro=1, kind="function", comp_type="C"),
        "S::get": EntryPoint(coro=2, kind="function", comp_type="S"),
        "C::spin": EntryPoint(coro=3, kind="function", comp_type="C"),
    }
    m.obj_layouts = {0: ["a"]}
    m.imports = {"p::ext": {"fn_id": EXT, "blocking": True, "api": 16},
                 "q::cfg": {"fn_id": CFG, "blocking": False, "api": 17}}
    m.apis = [{"id": 0, "name": "mem", "kind": "builtin"},
              {"id": 16, "name": "p", "kind": "user"},
              {"id": 17, "name": "q", "kind": "user"}]
    m.components = CompTable(ns(root="C", fields=[ns(name="x", slot=0),
                                                  ns(name="sub.y", slot=1)],
                                instances=[ns(base=0, path="", type_qname="C"),
                                           ns(base=1, path="sub", type_qname="S")]),
                             4)
    return m


def _vm(m=None, **kw):
    return VM(m or _model(), memory=Memory(), out=lambda l: None, **kw)


def _engine(cfg=lambda q: q.args[0] * 10):
    vm = _vm()
    t = calls_of(vm)
    t.link("q", cfg)
    return vm, t


class _Rec(Listener):
    def __init__(self):
        self.errors, self.fatals = [], []

    def on_error(self, ev):
        self.errors.append(ev)

    def on_fatal(self, ev):
        self.fatals.append(ev)


# -- handles ------------------------------------------------------------------ #

def test_new_constructs_the_root_and_calls_are_keyed_by_handle():
    vm, t = _engine()
    root = t.new("C")
    sub = t.sub(root, "sub")
    assert root and sub and root != sub
    assert vm.cobj.get_field(0) == 70              # x = cfg(7), at construction
    vm.cobj.set_field(1, 9)
    assert run_sync(t.call(root, "get")) == 70
    assert run_sync(t.call(sub, "get")) == 9       # S::get in instance sub
    a = t.new("pss_top::A", root)
    run_sync(t.call(a), import_provider=RecordingImportProvider({EXT: lambda x: x[0] + 1}))
    assert t.field(a, "a") == 51                    # ext(cfg(5)) = 50 + 1


def test_a_released_handle_is_refused_by_name():
    vm, t = _engine()
    root = t.new("C")
    sub, a = t.sub(root, "sub"), t.new("A", root)
    t.release(sub)
    with pytest.raises(HostStateError, match=r"handle 2 \(component sub: S\) "
                       "was released"):
        t.call(sub, "get")
    with pytest.raises(HostStateError, match="no handle 99"):
        t.release(99)
    t.release(root)                                 # frees everything under it
    for h in (root, a):
        with pytest.raises(HostStateError, match=f"handle {h} .* was released"):
            t.call(h)
    assert vm.cobj is None
    assert t.new("C") == 4                          # a new tree; ids never reused


def test_release_is_refused_while_a_call_runs_on_what_it_frees():
    vm, t = _engine()
    root = t.new("C")
    a = t.new("A", root)
    c = t.call(a)
    assert c.run().status == REQS                   # parked on ext
    with pytest.raises(HostStateError, match="a call runs on it"):
        t.release(a)
    with pytest.raises(HostStateError, match=r"calls are live \(A\)"):
        t.release(root)
    with pytest.raises(HostStateError, match="one root per engine"):
        t.new("C")
    c.end()
    t.release(root)


def test_new_names_what_it_cannot_build():
    vm, t = _engine()
    with pytest.raises(HostStateError, match="no root candidate 'D'.*has C"):
        t.new("D")
    root = t.new("C")
    with pytest.raises(HostStateError, match="no action entry 'B'"):
        t.new("B", root)
    with pytest.raises(HostStateError, match="S has no function entry 'put'"):
        t.call(t.sub(root, "sub"), "put")


# -- linking --------------------------------------------------------------- #

def test_link_check_names_each_import_with_no_handler():
    vm = _vm()
    t = calls_of(vm)
    with pytest.raises(HostStateError) as e:
        t.link_check()
    msg = str(e.value)
    assert "p: p::ext" in msg and "q: q::cfg" in msg and "mem: read32" in msg
    t.link("p", RecordingImportProvider({}))
    t.link("q", lambda q: 0)
    with pytest.raises(HostStateError, match=r"no handler linked for mem: read32$"):
        t.link_check()
    t.link("mem", Memory(), ctx=NO_EXECUTOR)        # every context the image has
    t.link_check()
    t.link("mem", None, ctx=NO_EXECUTOR)
    t.link("*", lambda q: 0)                         # a backstop covers the rest
    t.link_check()


def test_a_non_blocking_import_with_a_handler_is_synchronous():
    # no request for cfg: the handler answers at the import; ext (blocking)
    # is still a request, which the host completes
    seen = []
    vm, t = _engine(lambda q: seen.append((q.name, q.api, q.args)) or 4)
    root = t.new("C")
    c = t.call(t.new("A", root))
    assert c.run().status == REQS
    (q,) = [c.take()]
    assert c.take() is None
    assert (q.name, q.api, q.args) == ("p::ext", 16, [4])   # the API's own id
    c.complete(q.req_id, 8)
    assert c.run().status == DONE
    assert seen == [("q::cfg", 17, [7]), ("q::cfg", 17, [5])]   # init, then A


def test_construction_without_a_handler_is_a_located_fatal():
    vm = _vm()
    t = calls_of(vm)
    with pytest.raises(VMError, match="construct-yield: import q::cfg with no "
                       "linked handler"):
        t.new("C")                                 # no listener: raised
    assert vm.cobj is None and t.fatal_exc is not None
    blocking = _model(init=_import(EXT, 1) + [Instr(Op.RET)])
    vm = _vm(blocking)
    rec = _Rec()
    calls_of(vm).add_listener(rec)
    with pytest.raises(VMError, match="construct-yield: blocking import p::ext"):
        calls_of(vm).new("C")
    assert [e.sev for e in rec.fatals] == [SEV_FATAL] and not rec.errors


def test_a_linked_blocking_handler_serves_the_sync_loop():
    vm, t = _engine()
    t.link(16, RecordingImportProvider({EXT: lambda a: a[0] + 2}))
    root = t.new("C")
    a = t.new("A", root)
    run_sync(t.call(a), import_provider=RecordingImportProvider({}))
    assert t.field(a, "a") == 52


# -- listeners --------------------------------------------------------------- #

def test_a_listener_hears_a_calls_error_and_the_vm_goes_on():
    vm, t = _engine()
    rec = _Rec()
    t.add_listener(rec)
    root = t.new("C")
    c = t.call(t.new("A", root))
    assert c.run().status == REQS
    c.fail(c.take().req_id, VMError("bus error"))
    r = c.run()
    assert r.status == ERROR
    (ev,) = rec.errors
    assert (ev.sev, ev.call_id, ev.entry, ev.msg) == (SEV_ERROR, c.id, "A",
                                                      "bus error")
    assert not rec.fatals
    assert run_sync(t.call(root, "get")) == 70     # the next call runs


def test_a_listener_hears_a_self_wait_as_an_error():
    rec = _Rec()

    def ext(args):
        r = t.call(t.new("A", root)).run()        # the second traversal
        seen.append(r.status)
        return 0

    seen = []
    vm, t = _engine()
    t.add_listener(rec)
    root = t.new("C")
    run_sync(t.call(t.new("A", root)),
             import_provider=RecordingImportProvider({EXT: ext}))
    assert seen == [ERROR]
    (ev,) = rec.errors
    assert ev.msg.startswith("self-wait: traversal A was started inside "
                             "request") and ev.sev == SEV_ERROR
    assert not rec.fatals


def _deadlocked(t, root):
    """A traversal parked on ext, a function spinning on what nothing
    writes, and a traversal queued behind the first."""
    a = t.call(t.new("A", root))
    spin = t.call(root, "spin")
    b = t.call(t.new("A", root))
    return a, spin, b


def test_a_deadlock_is_a_fatal_the_default_listener_ends_the_run_with():
    vm, t = _engine()
    root = t.new("C")
    a, spin, b = _deadlocked(t, root)
    assert a.run().status == REQS                   # a holds the slot, parked
    with pytest.raises(VMError, match=r"deadlock: .*spin \(waits on an event\), "
                       r"A \(queued behind traversal A\)"):
        run_calls([spin, b])
    # released, not hanging: every call is over, started or queued
    assert all(c.over and c.status == ERROR for c in (a, spin, b))
    assert t.live == [] and t.active is None and not t.queue
    assert a.run().status == ERROR                  # a released call says so again


def test_a_listener_that_only_records_a_fatal_lets_the_host_decide():
    vm, t = _engine()
    rec = _Rec()
    t.add_listener(rec)
    root = t.new("C")
    a, spin, b = _deadlocked(t, root)
    assert a.run().status == REQS                   # a holds the slot, parked
    rs = run_calls([spin, b])                       # no raise
    assert [r.status for r in rs] == [ERROR, ERROR]
    (ev,) = rec.fatals
    assert ev.sev == SEV_FATAL and ev.call_id == 0 and "deadlock" in ev.msg
    assert not rec.errors                           # released calls: no events
    assert a.over and a.request.error["message"] == ev.msg
    # the VM refuses every later call, naming the fatal, until a new tree
    r = t.call(root, "get").run()
    assert r.status == ERROR and r.error["message"].startswith(
        "refused: the engine stopped on a fatal error: deadlock")
    t.release(root)
    assert run_sync(t.call(t.new("C"), "get")) == 70


def test_a_raising_listener_ends_the_loop_after_the_release():
    class _Stop(Exception):
        pass

    class _L(Listener):
        def on_fatal(self, ev):
            # synchronous, at the failure: no call has returned yet
            assert not (a.over or spin.over or b.over)
            raise _Stop(ev.msg)

    vm, t = _engine()
    t.add_listener(_L())
    root = t.new("C")
    a, spin, b = _deadlocked(t, root)
    assert a.run().status == REQS
    with pytest.raises(_Stop):
        run_calls([spin, b])
    assert a.over and spin.over and b.over          # released all the same


# -- step, pending, completion ------------------------------------------------- #

def test_step_runs_every_call_and_pending_says_what_is_left():
    vm, t = _engine()
    root = t.new("C")
    a = t.call(t.new("A", root))
    g = t.call(root, "get")
    b = t.call(t.new("A", root))                     # queued behind a
    # b cannot run before a has had the traversal slot
    assert t.pending().flags == PEND_READY and t.pending().ready == 2
    done = []
    g.on_done(lambda c, r: done.append((c.id, r.status)))
    assert t.step() == 2 and b.request is None       # a parks, g ends
    assert done == [(g.id, DONE)] and g.result == 70
    p = t.pending()
    assert (p.ready, p.reqs, p.flags) == (0, 1, PEND_REQS)
    assert t.step() == 0                             # nothing can run
    q = a.take()
    assert q.name == "p::ext" and a.take() is None
    a.complete(q.req_id, 3)
    assert t.pending().ready == 1
    assert t.step(budget=1) == 1 and a.status == DONE
    assert t.step() == 1 and b.request.status == REQS   # b's turn now
    assert not t.pending().ready and t.pending().reqs == 1


def test_step_serves_linked_requests_and_a_deferred_one_later():
    vm, t = _engine()
    later = []
    t.link("p", lambda q: later.append(q) or DEFER)
    root = t.new("C")
    a = t.new("A", root)
    c = t.call(a)
    assert t.step() == 1 and c.take() is None        # the handler took it
    assert t.pending().flags == PEND_REQS
    c.complete(later[0].req_id, 1)                   # the token, completed later
    assert t.step() == 1 and c.status == DONE
    assert t.field(a, "a") == 1


def test_step_moves_a_timer_only_when_the_host_clock_reaches_it():
    m = _model()
    m.coros.append(CoroDescriptor(name="C::nap", code=[
        Instr(Op.CONST, args=(0,), imm=10), Instr(Op.WAIT, args=(0,)),
        Instr(Op.RET)]))
    m.entries["C::nap"] = EntryPoint(coro=5, kind="function", comp_type="C")
    vm = _vm(m)
    t = calls_of(vm)
    t.link("q", lambda q: 0)
    c = t.call(t.new("C"), "nap")
    assert t.step() == 1 and c.request.on == "TIME"
    p = t.pending()
    assert p.flags == PEND_TIME and p.next_time == 10
    assert t.step() == 0
    t.set_time(10)
    assert t.step() == 1 and c.status == DONE


def test_on_done_on_a_call_already_over_runs_at_once():
    vm, t = _engine()
    c = t.call(t.new("C"), "get")
    run_sync(c)
    seen = []
    c.on_done(lambda call, r: seen.append(r.status))
    assert seen == [DONE]


def test_the_shim_reports_a_fatal_and_releases_the_other_calls(capsys):
    # under SV (pyhdl-if shim): the shim's listener keeps the fatal for
    # fatal(), which SV's run loop gives its listeners (or $fatal); the
    # call SV is serving ends too (-1), not hangs
    from zuspec.be.bc.share.py import pssvm_dpi_shim as shim

    class _Mocks(RecordingImportProvider):
        def is_mocked(self, fn):
            return fn == EXT

    def lost(args):
        e = VMError("engine lost")
        e.fatal = True
        raise e

    vm = _vm(import_provider=_Mocks({EXT: lost}))
    t = calls_of(vm)
    t.link("q", lambda q: 1)
    root = t.new("C")
    eng = shim.Engine()
    eng.session = ns(vm=vm)
    hs = eng._started(t.call(root, "spin"), "C::spin")
    ha = eng._started(t.call(t.new("A", root)), "A")
    assert eng.run(hs) == 1 and eng.take(hs) >= 0  # a read SV serves
    assert eng.run(ha) == -1
    assert eng.fatal() == "engine lost"
    assert "PSSVM FATAL" not in capsys.readouterr().out   # SV says it
    assert eng.run(hs) == -1 and "engine lost" in eng.error(hs)
