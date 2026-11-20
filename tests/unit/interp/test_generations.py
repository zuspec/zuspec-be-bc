"""Generations (docs/spec/generations.md, P3.6a / T3.6)."""
import pytest

from zuspec.be.bc.interp.generations import (PSSVM_E_API, UNKNOWN_IMPORTS, ApiChange,
                                             ReloadRefused, api_diff, check_api)
from zuspec.be.bc.model import (CoroDescriptor, EntryPoint, Instr, Op, ZbcModel,
                                api_hash_of)

_U32 = {"kind": "int", "width": 32, "signed": False}


def _model(entries, imports=None, body=0, generation=0):
    # one coroutine returning `body`: a body-only edit changes it, not the API
    root = CoroDescriptor(name="root", code=[Instr(Op.CONST, args=(0,), imm=body),
                                             Instr(Op.RET, args=(0,))])
    imports = imports or {}
    return ZbcModel(coros=[root], entry_coro=0, profile="debug", entries=entries,
                    imports=imports, api_hash=api_hash_of(entries, imports),
                    generation=generation)


def _entries(**over):
    e = {"dma_c::start": EntryPoint(coro=0, kind="function", comp_type="dma_c",
                                    params=[("ch", _U32)], returns=None),
         "smoke_t": EntryPoint(coro=0, kind="action")}
    e.update(over)
    return {k: v for k, v in e.items() if v is not None}


_IMPORTS = {"ext": {"fn_id": 0, "blocking": True, "arg_types": [], "ret_type": None,
                    "string_at": []}}


def test_body_only_edit_is_accepted():
    old = _model(_entries(), _IMPORTS, body=1, generation=1)
    new = _model(_entries(), _IMPORTS, body=2, generation=2)
    assert old.api_hash == new.api_hash
    check_api(old, new)
    assert api_diff(old, new) == []


def test_changed_signature_is_refused_with_the_list():
    old = _model(_entries(), _IMPORTS)
    new = _model(_entries(**{"dma_c::start": EntryPoint(
        coro=0, kind="function", comp_type="dma_c",
        params=[("ch", _U32), ("len", _U32)])}), _IMPORTS)
    with pytest.raises(ReloadRefused) as ei:
        check_api(old, new)
    e = ei.value
    assert e.code == PSSVM_E_API and e.reason == "api"
    assert [(c.symbol, c.what) for c in e.changes] == [("entry:dma_c::start", "changed")]
    assert e.changes[0].new["params"][1][0] == "len"
    assert "entry:dma_c::start (changed)" in str(e)
    j = e.to_json()
    assert j["code"] == -4 and j["changes"][0]["symbol"] == "entry:dma_c::start"


def test_added_removed_and_import_changes():
    imports2 = {"ext": dict(_IMPORTS["ext"], blocking=False)}
    old = _model(_entries(), _IMPORTS)
    new = _model(_entries(smoke_t=None, new_t=EntryPoint(coro=0, kind="action")), imports2)
    with pytest.raises(ReloadRefused) as ei:
        check_api(old, new)
    assert [(c.symbol, c.what) for c in ei.value.changes] == [
        ("entry:new_t", "added"), ("entry:smoke_t", "removed"), ("import:ext", "changed")]


def test_read_back_images_keep_hash_and_entries():
    # imports are not serialized: a read-back image compares entries, and
    # an import-only difference is reported as UNKNOWN_IMPORTS
    old = _model(_entries(), _IMPORTS, generation=1)
    same = ZbcModel.from_bytes(_model(_entries(), _IMPORTS, body=5, generation=2).to_bytes())
    assert same.imports == {} and same.api_hash == old.api_hash
    check_api(old, same)

    imp_only = _model(_entries(), {"ext": dict(_IMPORTS["ext"], fn_id=3)})
    rb = ZbcModel.from_bytes(imp_only.to_bytes())
    with pytest.raises(ReloadRefused) as ei:
        check_api(old, rb)
    assert ei.value.changes == [ApiChange(UNKNOWN_IMPORTS, "changed")]


def test_unset_hash_compares_signatures():
    old = _model(_entries())
    new = _model(_entries())
    old.api_hash = new.api_hash = 0
    check_api(old, new)
    new.entries = _entries(smoke_t=None)
    with pytest.raises(ReloadRefused) as ei:
        check_api(old, new)
    assert ei.value.changes[0].symbol == "entry:smoke_t"


# -- the Generations holder (P3.6a2) ---------------------------------------- #

from zuspec.be.bc.interp.generations import Generations, changed_symbols
from zuspec.be.bc.interp.host import ACCESS, DONE, HostStateError, run_sync
from zuspec.be.bc.model import BUILTIN_READ, INSTR_F_HAS_RET


def _run_model(body, generation=0):
    # smoke_t (coro 0): read32(0x100), return body; dma_c::start (coro 1): return 10*body
    act = CoroDescriptor(name="smoke", code=[
        Instr(Op.CONST, args=(0,), imm=0x100),
        Instr(Op.IMPORT, args=(BUILTIN_READ[32], 1, 0), flags=INSTR_F_HAS_RET),
        Instr(Op.CONST, args=(2,), imm=body),
        Instr(Op.RET, args=(2,))])
    fn = CoroDescriptor(name="start", code=[Instr(Op.CONST, args=(0,), imm=10 * body),
                                            Instr(Op.RET, args=(0,))])
    entries = {"dma_c::start": EntryPoint(coro=1, kind="function", comp_type="dma_c",
                                          params=[("ch", _U32)]),
               "smoke_t": EntryPoint(coro=0, kind="action")}
    # N+1 may order its coroutines differently: the symbol, not the index, binds
    if body % 2 == 0:
        fn, act = act, fn
        entries["dma_c::start"].coro, entries["smoke_t"].coro = 0, 1
    return ZbcModel(coros=[act, fn], entry_coro=entries["smoke_t"].coro,
                    profile="debug", entries=entries,
                    api_hash=api_hash_of(entries, {}), generation=generation)


def _smoke(g):
    call = g.call("smoke_t")
    r = call.run()
    assert (r.status, r.kind, r.addr) == (ACCESS, "r", 0x100)
    return call


def test_calls_bind_by_symbol_and_reload_applies_when_quiet():
    g = Generations(_run_model(1))
    assert g.generation == 1
    c = g.call("dma_c::start", 0)
    assert run_sync(c) == 10
    c.end()
    r = g.reload(_run_model(2))
    assert r == {"old": 1, "new": 2, "changed": ["dma_c::start", "smoke_t"],
                 "structure": {"added": [], "removed": [], "changed": []},
                 "migrated": [], "cancelled": 0, "applied": True}
    assert g.generation == 2 and g.history == [r]
    c = g.call("dma_c::start", 0)              # coro 0 now, not 1
    assert run_sync(c) == 20
    c = _smoke(g)
    c.complete(0)
    assert c.run().status == DONE and c.result == 2
    with pytest.raises(KeyError):
        g.call("nope_t")
    with pytest.raises(TypeError):
        g.call("dma_c::start")


def test_running_call_finishes_on_n():
    g = Generations(_run_model(1))
    c = _smoke(g)
    r = g.reload(_run_model(3))
    assert r["applied"] is False and g.generation == 1 and g.pending is not None
    with pytest.raises(HostStateError):
        g.call("dma_c::start", 0)              # one call per VM; N+1 waits
    c.complete(0)
    assert c.run().status == DONE and c.result == 1    # N's code
    c2 = g.call("smoke_t")                     # the reload point: N+1 applies
    assert r["applied"] is True and g.generation == 2 and g.history == [r]
    assert run_sync(c2) == 3


def test_cancel_and_test_boundary():
    g = Generations(_run_model(1))
    c = _smoke(g)
    r = g.reload(_run_model(3), cancel=True)
    assert r["cancelled"] == 1 and r["applied"] is True and c.over
    assert g.at_test_boundary() is None        # nothing staged

    c = _smoke(g)
    r = g.reload(_run_model(5))
    with pytest.raises(HostStateError):
        g.at_test_boundary()
    c.complete(0)
    assert c.run().status == DONE
    assert g.at_test_boundary() is r and r["applied"] and g.generation == 3
    assert [h["new"] for h in g.history] == [2, 3]


def test_refused_reload_leaves_n_and_the_staged_image():
    g = Generations(_run_model(1))
    c = _smoke(g)
    staged = g.reload(_run_model(3))
    bad = _run_model(5)
    bad.entries["dma_c::start"].params = []
    bad.api_hash = api_hash_of(bad.entries, {})
    with pytest.raises(ReloadRefused) as ei:
        g.reload(bad)
    assert ei.value.reason == "api" and g.generation == 1
    assert g.pending is not None and not staged["applied"]
    c.end()
    g.at_test_boundary()
    assert g.generation == 2 and run_sync(g.call("smoke_t"), memory=_Zero()) == 3


def test_changed_symbols_body_only():
    assert changed_symbols(_run_model(1), _run_model(1)) == []
    assert changed_symbols(_run_model(1), _run_model(3)) == ["dma_c::start", "smoke_t"]


class _Zero:
    def read(self, addr, size):
        return 0

    def write(self, addr, size, data):
        pass


# -- migrate (P3.6a3) ------------------------------------------------------ #

import types

from zuspec.be.bc.interp.components import CompTable
from zuspec.be.bc.interp.generations import structure_diff


def _comp_model(insts, init=1, touch=None):
    # insts: [(path, type, [field names])]; construction stores ``init`` into
    # every slot; touch="read"/"import" makes it reach the bus too
    ns = types.SimpleNamespace
    fields, instances = [], []
    for path, typ, names in insts:
        instances.append(ns(base=len(fields), path=path, type_qname=typ))
        fields += names
    code = [Instr(Op.CONST, args=(0,), imm=init)]
    code += [Instr(Op.ST_COMP, args=(0, k)) for k in range(len(fields))]
    if touch == "read":
        code += [Instr(Op.CONST, args=(1,), imm=0x40),
                 Instr(Op.IMPORT, args=(BUILTIN_READ[32], 2, 1), flags=INSTR_F_HAS_RET)]
    elif touch == "import":
        code += [Instr(Op.IMPORT, args=(3, 0xFFFFFFFF))]
    code += [Instr(Op.RET, args=(0,))]
    coros = [CoroDescriptor(name="smoke", code=[Instr(Op.CONST, args=(0,), imm=0),
                                                Instr(Op.RET, args=(0,))]),
             CoroDescriptor(name="init", code=code)]
    entries = {"smoke_t": EntryPoint(coro=0, kind="action")}
    m = ZbcModel(coros=coros, entry_coro=0, profile="debug", entries=entries,
                 api_hash=api_hash_of(entries, {}))
    m.components = CompTable(ns(fields=[ns(name=n, slot=i) for i, n in enumerate(fields)],
                                instances=instances), 1)
    return m


_TREE = [("top", "top_c", ["x", "y"]), ("top.ch", "chan_c", ["wake", "inflight"])]


def test_state_migrates_by_path():
    g = Generations(_comp_model(_TREE))
    assert g.vm.cobj.values == [1, 1, 1, 1]
    g.vm.cobj.values[:] = [5, 6, 2, 3]          # N ran: fields and channel tokens moved
    r = g.reload(_comp_model(_TREE, init=9))
    assert r["applied"] and r["structure"] == {"added": [], "removed": [], "changed": []}
    assert r["migrated"] == ["top", "top.ch"] and g.vm.cobj.values == [5, 6, 2, 3]


def test_migration_takes_n_state_at_apply_time():
    g = Generations(_comp_model(_TREE))
    c = g.call("smoke_t")                        # live: N+1 waits
    r = g.reload(_comp_model(_TREE, init=9))
    assert not r["applied"] and g.pending.vm.cobj.values == [9, 9, 9, 9]
    assert run_sync(c) == 0                      # (an action builds its own tree)
    g.vm.cobj.values[:] = [4, 4, 4, 4]           # N's state when the call ended
    g.at_test_boundary()
    assert g.generation == 2 and g.vm.cobj.values == [4, 4, 4, 4]


def test_structural_change_is_reported_and_new_parts_start_fresh():
    new = [("top", "top_c", ["x", "y"]), ("top.ch", "chan_c", ["wake"]),
           ("top.dma", "dma_c", ["n"])]
    assert structure_diff(_comp_model(_TREE), _comp_model(new)) == {
        "added": ["top.dma"], "removed": [], "changed": ["top.ch"]}
    assert structure_diff(_comp_model(new), _comp_model(_TREE))["removed"] == ["top.dma"]
    g = Generations(_comp_model(_TREE))
    g.vm.cobj.values[:] = [5, 6, 2, 3]
    r = g.reload(_comp_model(new, init=9))
    assert r["migrated"] == ["top"] and g.vm.cobj.values == [5, 6, 9, 9]


def test_strict_refuses_a_structural_change():
    g = Generations(_comp_model(_TREE), strict=True)
    with pytest.raises(ReloadRefused) as ei:
        g.reload(_comp_model(_TREE[:1]))
    assert ei.value.reason == "structure" and "top.ch (removed)" in str(ei.value)
    assert g.generation == 1 and g.pending is None
    assert g.reload(_comp_model(_TREE, init=2))["applied"]     # same tree: fine


@pytest.mark.parametrize("touch,what", [("read", "memory read at 0x40"),
                                        ("import", "import 3")])
def test_bus_during_construction_refuses(touch, what):
    g = Generations(_comp_model(_TREE))
    g.vm.cobj.values[:] = [5, 6, 2, 3]
    with pytest.raises(ReloadRefused) as ei:
        g.reload(_comp_model(_TREE, touch=touch))
    assert ei.value.reason == "bus" and what in str(ei.value)
    assert g.generation == 1 and g.pending is None and g.vm.cobj.values == [5, 6, 2, 3]
    assert g.vm.host is None and run_sync(g.call("smoke_t")) == 0
