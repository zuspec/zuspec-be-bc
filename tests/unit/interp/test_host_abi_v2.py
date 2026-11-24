"""Host ABI version 2 conformance (docs/spec/host-abi.md "Run status and
requests", "Conformance tests"; pssc full-PSS plan F5.1c).

Each model is a few branches of one parallel, run by a scripted host that
completes outstanding requests in an order it chooses.
"""
import itertools

import pytest

from zuspec.be.bc.interp.extern import Memory, RecordingImportProvider
from zuspec.be.bc.interp.host import HostCall, HostStateError, run_sync
from zuspec.be.bc.interp.vm import VM

from .abi_script import (NONE, ScriptedHost, mem_read, mem_write, parallel,
                        user_import, wait)

EXT = {"ext": {"fn_id": 0, "blocking": True}}
EXT_NB = {"ext": {"fn_id": 0, "blocking": False}}


def _call(model):
    vm = VM(model, out=lambda l: None)
    return HostCall(vm, vm.root_frame(model.entry_coro, 0))


def _call2(model):
    return _call(model)


def _two_mem():
    # parallel { write32(0x100, 1); read32(0x104) || write32(0x200, 2); read32(0x204) }
    return parallel(mem_write(0x100, 1) + mem_read(0x104),
                    mem_write(0x200, 2) + mem_read(0x204))


@pytest.mark.parametrize("model", [
    _two_mem(),
    parallel(mem_write(0x100, 1), user_import(0, 5, True), imports=EXT),
    parallel(user_import(0, 5, False), imports=EXT_NB),
    parallel(wait(1), mem_read(0x100), first=1),
    parallel(wait(5), mem_read(0x100)),
], ids=["two_mem", "mem_import", "nonblocking", "first_cancel", "wait_race"])
def test_models_run_on_todays_engine(model):
    # each model is valid bytecode: the version-1 sync loop runs it to DONE
    run_sync(_call(model), memory=Memory(),
             import_provider=RecordingImportProvider({0: lambda a: a[0] + 1}))


def test_two_branches_both_outstanding():
    call = _call2(_two_mem())
    host = ScriptedHost(call)
    r = call.run()
    assert r.status_name == "REQS"
    first = host.take_all()
    assert [s.what for s in first] == ["w@0x100", "w@0x200"]
    assert len({s.req_id for s in first}) == 2
    assert len({s.thread for s in first}) == 2
    assert all(s.executor == NONE and s.blocking for s in first)
    r = call.run()
    assert (r.status_name, r.on) == ("WAIT", "REQS")


@pytest.mark.parametrize("newest_first", [False, True])
def test_completion_order_and_per_thread_order(newest_first):
    host = ScriptedHost(_call2(_two_mem()),
                        pick=(lambda out: out[-1]) if newest_first
                        else (lambda out: out[0]),
                        answer=lambda s: 7 if s.what.startswith("r") else None)
    host.run()
    assert sorted(host.per_thread().values()) == [["w@0x100", "r@0x104"],
                                                  ["w@0x200", "r@0x204"]]
    if newest_first:
        # completing branch 1 first resumes it first: its read is taken first
        reads = [s.what for s in host.taken if s.what.startswith("r")]
        assert reads == ["r@0x204", "r@0x104"]


@pytest.mark.parametrize("order", list(itertools.permutations(range(2))))
def test_mem_and_blocking_import_in_parallel(order):
    call = _call2(parallel(mem_write(0x100, 1), user_import(0, 5, True),
                          imports=EXT))
    host = ScriptedHost(call, answer=lambda s: 6 if s.what == "ext(5)" else None)
    assert call.run().status_name == "REQS"
    seen = host.take_all()
    assert sorted(s.what for s in seen) == ["ext(5)", "w@0x100"]
    for i in order:
        call.complete(seen[i].req_id, host.answer(seen[i]))
    assert call.run().status_name == "DONE"


def test_nonblocking_import_completed_when_taken():
    call = _call2(parallel(user_import(0, 5, False), imports=EXT_NB))
    assert call.run().status_name == "REQS"
    r = call.take()
    assert (r.name, r.args, r.blocking) == ("ext", [5], False)
    call.complete(r.req_id, 6)
    assert call.take() is None
    assert call.run().status_name == "DONE"


def test_done_waits_for_outstanding_and_bad_req_id():
    call = _call2(_two_mem())
    host = ScriptedHost(call)
    call.run()
    a, b = host.take_all()
    with pytest.raises(HostStateError):
        call.complete(max(a.req_id, b.req_id) + 100)        # unknown id
    call.complete(a.req_id)
    r = call.run()
    assert r.status_name != "DONE"
    with pytest.raises(HostStateError):
        call.complete(a.req_id)                              # already done
    call.end()
    with pytest.raises(HostStateError):
        call.complete(b.req_id)                              # dropped


def test_cancelled_branch_discards_late_completion():
    # parallel FIRST(1) { wait 1 || read32(0x100) }: the wait wins; the read's
    # request stays outstanding and its late completion is discarded
    call = _call2(parallel(wait(1), mem_read(0x100), first=1))
    assert call.run().status_name == "REQS"
    (rd,) = ScriptedHost(call).take_all()
    r = call.run()
    assert (r.status_name, r.on) == ("WAIT", "TIME")
    r = call.run()                                           # no clock: t=1
    assert r.status_name == "DONE"
    call.complete(rd.req_id, 9)                              # accepted, ignored
    call.end()


def test_wait_time_raced_by_completion_and_clock():
    # parallel { wait 5 || read32(0x100) }: the read completes before t=5
    call = _call2(parallel(wait(5), mem_read(0x100)))
    assert call.run().status_name == "REQS"
    (rd,) = ScriptedHost(call).take_all()
    r = call.run()
    assert (r.status_name, r.on, r.delay) == ("WAIT", "TIME", 5)
    call.set_time(2)
    call.complete(rd.req_id, 3)
    r = call.run()                                           # read branch ends
    assert (r.status_name, r.on, r.delay) == ("WAIT", "TIME", 3)
    with pytest.raises(HostStateError):
        call.set_time(1)                                     # backwards
    call.set_time(5)
    assert call.run().status_name == "DONE"
