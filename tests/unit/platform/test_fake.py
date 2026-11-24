"""The register fake and field kinds run from be-bc alone (full-PSS plan F1.10).

The image is a stand-in with the debug image's ``registers`` table; pssc's own
tests drive the fake from lowered PSS."""

import json
import types

import pytest

from zuspec.be.bc.platform import Platform, RegisterFake, on_write, reg_names
from zuspec.be.bc.platform import reg_semantics as rs


def _model(**kw):
    regs = {
        0: {"path": "dma.regs.csr", "comp": "dma.regs", "offset": 0x0,
            "width": 32, "access": "READWRITE",
            "fields": [{"name": "go", "lsb": 0, "width": 1, "kind": "RW"},
                       {"name": "done", "lsb": 1, "width": 1, "kind": "W1C"},
                       {"name": "ver", "lsb": 8, "width": 8, "kind": "RO"}]},
        1: {"path": "dma.regs.id", "comp": "dma.regs", "offset": 0x4,
            "width": 32, "access": "READONLY", "fields": []},
    }
    return types.SimpleNamespace(registers=regs, prov=None, components=None,
                                 profile="debug", **kw)


def _fake(cls=RegisterFake):
    f = cls()
    f.attach(_model())
    return f


def test_platform_memory():
    p = Platform()
    p.write32(0x100, 0x11223344)
    assert p.read16(0x102) == 0x1122 and p.peek(0x100, 1) == 0x44
    assert p.log == [("write", 32, 0x100, 0x11223344), ("read", 16, 0x102, 0x1122)]


def test_reset_value_and_readonly():
    class F(RegisterFake):
        resets = {"dma.regs.id": 0xABCD}
    f = _fake(F)
    assert f.peek_reg("dma.regs.id") == 0xABCD
    f.note_access(1, 0)
    f.write32(0x1004, 0)
    assert f.peek_reg("dma.regs.id") == 0xABCD


def test_note_places_component_and_field_kinds():
    f = _fake()
    f.poke_reg("dma.regs.csr", 0x0302)       # done set, ver 3
    f.note_access(0, 0)
    f.write32(0x2000, 0x0001 | 0x0002 | 0xFF00)   # go, clear done, ver kept
    assert f.peek_field("dma.regs.csr", "go") == 1
    assert f.peek_field("dma.regs.csr", "done") == 0
    assert f.peek_field("dma.regs.csr", "ver") == 3
    # the other register of the component is now at base + 4, no note needed
    f.poke_reg("dma.regs.id", 7)
    assert f.read32(0x2004) == 7


def test_unknown_reset_refused():
    class F(RegisterFake):
        resets = {"nope": 1}
    with pytest.raises(ValueError, match="names no register"):
        _fake(F)


def test_on_write_hook():
    class F(RegisterFake):
        @on_write("dma.regs.csr", "go")
        def go(self, ctx, acc):
            self.poke_field("dma.regs.csr", "done", 1)
    f = _fake(F)
    f.note_access(0, 0)
    f.write32(0x3000, 1)
    assert f.peek_field("dma.regs.csr", "done") == 1


def test_reg_names():
    names = reg_names({0: {"path": "ch[1].regs.csr"}}, None)
    assert names["ch[1].regs.csr"] == {0} and names["ch.regs.csr"] == {0}


def test_reg_semantics_json(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"csr": {"done": "W1C"}}))
    assert rs.kinds_for(rs.load(str(p)), "dma.regs.csr") == {"done": "W1C"}
