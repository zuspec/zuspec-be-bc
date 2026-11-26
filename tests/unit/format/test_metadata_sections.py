"""Format 2.0 metadata (pssc docs/design/pssvm-plan.md step 4): extension
profiles and opcode ranges, ENTRY export flags, IMPORT signatures, the API,
CLASS and ROOT sections, and the normative native layout (value-abi.md 2.1)."""

import pytest

from zuspec.be.bc.abi.layout import native_layout
from zuspec.be.bc.format import spec
from zuspec.be.bc.format.reader import ZbcFormatError, read_image
from zuspec.be.bc.format.writer import write_image
from zuspec.be.bc.model import (CoroDescriptor, EntryPoint, Instr, Op, ZbcModel,
                                api_hash_of)

U8 = {"kind": "int", "width": 8, "signed": False}
U16 = {"kind": "int", "width": 16, "signed": False}
U32 = {"kind": "int", "width": 32, "signed": False}
U64 = {"kind": "int", "width": 64, "signed": False}
BOOL = {"kind": "bool", "width": 1, "signed": False}

PSS_OPS = {Op.SELECT, Op.SCOPE_ENTER, Op.SOLVE_NODE, Op.COV_SAMPLE,
           Op.EXEC_SET, Op.EXEC_GET, Op.LD_COMP, Op.ST_COMP}


# --------------------------------------------------------------- profiles

def test_every_opcode_is_in_its_profiles_range():
    for op in Op:
        ext = spec.op_extension(int(op))
        assert ext == ("pss" if op in PSS_OPS else None), op.name


def test_the_header_declares_the_pss_extension_when_the_code_uses_it():
    m = ZbcModel(coros=[CoroDescriptor(name="a", code=[
        Instr(Op.SCOPE_ENTER, imm=0), Instr(Op.RET)])])
    assert m.extensions() == spec.flag_value("zbc_hdr_flags", "ZBC_HDR_EXT_PSS")
    img = read_image(m.to_bytes())
    assert img.extensions == m.extensions()
    plain = ZbcModel(coros=[CoroDescriptor(name="a", code=[Instr(Op.RET)])])
    assert plain.extensions() == 0
    assert ZbcModel.from_bytes(plain.to_bytes()).coros[0].code[0].op == Op.RET


def test_an_extension_opcode_the_image_does_not_declare_is_refused():
    m = ZbcModel(coros=[CoroDescriptor(name="a", code=[
        Instr(Op.SOLVE_NODE, (0xFFFFFFFF,)), Instr(Op.RET)])])
    c = m.to_container()
    c.extensions = 0
    with pytest.raises(ZbcFormatError, match="SOLVE_NODE is in the pss"):
        ZbcModel.from_bytes(write_image(c))


def test_an_extension_the_engine_does_not_implement_is_refused():
    c = ZbcModel(coros=[CoroDescriptor(name="a", code=[Instr(Op.RET)])]).to_container()
    c.extensions = 0x0200
    with pytest.raises(ZbcFormatError, match="extension bits 0x0200"):
        ZbcModel.from_bytes(write_image(c))


def test_the_format_is_version_2():
    data = ZbcModel(coros=[CoroDescriptor(name="a")]).to_bytes()
    assert data[4] == 2 and spec.VERSION_MAJOR == 2
    old = bytearray(data)
    old[4] = 1
    with pytest.raises(ZbcFormatError, match="major version 1"):
        read_image(bytes(old), verify_hash=False)


# --------------------------------------------------------- native layout

def test_native_layout_is_c_layout():
    rows, size, align = native_layout([
        ("a", U8), ("s.b", U32), ("s.c", U8), ("d", U8), ("e", BOOL), ("f", U64)])
    # struct s {u32 b; u8 c;} is 8 bytes at 4; d follows its tail padding
    assert [(n, off, sz) for n, _, off, sz in rows] == [
        ("a", 0, 1), ("s.b", 4, 4), ("s.c", 8, 1), ("d", 12, 1), ("e", 13, 1),
        ("f", 16, 8)]
    assert (size, align) == (24, 8)


def test_native_layout_sizes():
    def one(d):
        rows, size, _ = native_layout([("x", d)])
        return rows[0][3]
    assert one({"kind": "int", "width": 1, "signed": False}) == 1
    assert one({"kind": "int", "width": 9, "signed": True}) == 2
    assert one({"kind": "enum", "width": 32, "signed": True}) == 4
    assert one({"kind": "int", "width": 33, "signed": False}) == 8
    assert one({"kind": "float", "width": 32, "signed": True}) == 4
    assert one({"kind": "string", "width": 64, "signed": False}) == 8
    assert one({"kind": "opaque", "width": 0, "signed": False}) == 8
    assert native_layout([]) == ([], 0, 1)


def test_an_array_of_structs_is_a_c_array():
    rows, size, _ = native_layout([("r.0.a", U16), ("r.0.b", U8),
                                   ("r.1.a", U16), ("r.1.b", U8)])
    assert [off for _, _, off, _ in rows] == [0, 2, 4, 6]
    assert size == 8


# ----------------------------------------------------- sections round trip

def _model():
    m = ZbcModel(coros=[CoroDescriptor(name="xfer_a"), CoroDescriptor(name="dma_c::f")])
    m.entries = {
        "xfer_a": EntryPoint(coro=0, qname="dma_c::xfer_a",
                             obj=[("sz", dict(U16, rand=True)), ("src", dict(U32, rand=True))],
                             exported=True, fixed=["sz"]),
        "dma_c::f": EntryPoint(coro=1, kind="function", comp_type="dma_c",
                               params=[("ch", U8)]),
    }
    m.imports = {
        "wait_done": {"fn_id": 0, "blocking": True, "target": True,
                      "arg_types": [[8, False]], "ret_type": None, "string_at": [],
                      "api": 16, "qname": "dma_pkg::wait_done"},
        "num_channels": {"fn_id": 1, "blocking": False, "solve": True,
                         "arg_types": [], "ret_type": [8, False], "string_at": [],
                         "api": 16, "qname": "dma_pkg::num_channels"},
    }
    m.apis = [{"id": 0, "name": "mem", "kind": "builtin"},
              {"id": 1, "name": "config", "kind": "builtin"},
              {"id": 2, "name": "seq", "kind": "builtin"},
              {"id": 16, "name": "dma_pkg", "kind": "user"}]
    rows, size, align = native_layout([("sz", U16), ("src", U32)])
    m.classes = [{"qname": "dma_c::xfer_a", "kind": "action", "base": None, "coro": 0,
                  "fields": [[n, d, o, s] for n, d, o, s in rows], "size": size,
                  "align": align, "methods": []},
                 {"qname": "dma_c", "kind": "component", "base": "base_c", "coro": None,
                  "fields": [["base", U32, 0, 4]], "size": 4, "align": 4,
                  "methods": [["f", "dma_c::f"]]}]
    m.api_hash = api_hash_of(m.entries, m.imports)
    m.roots = [{"type": "dma_c", "ctor": {"name": "initialize", "params": [
        {"name": "base", "type": "bit[32]"}]}, "api_hash": m.api_hash, "image": None}]
    return m


def test_entries_keep_the_export_flag_and_fixed_fields():
    back = ZbcModel.from_bytes(_model().to_bytes())
    assert back.entries["xfer_a"].exported and back.entries["xfer_a"].fixed == ["sz"]
    assert not back.entries["dma_c::f"].exported
    assert spec.RECORDS_BY_NAME["zbc_entry"].size() == 40


def test_exporting_changes_the_api_hash():
    m = _model()
    h = api_hash_of(m.entries, m.imports)
    m.entries["xfer_a"].exported = False
    assert api_hash_of(m.entries, m.imports) != h


def test_imports_carry_their_signatures_and_api():
    m = _model()
    back = ZbcModel.from_bytes(m.to_bytes())
    assert back.imports == m.imports
    assert back.import_names == {0: "wait_done", 1: "num_channels"}
    # so a read-back image hashes the same
    assert api_hash_of(back.entries, back.imports) == m.api_hash


def test_api_rows():
    back = ZbcModel.from_bytes(_model().to_bytes())
    assert back.apis == _model().apis


def test_class_rows():
    back = ZbcModel.from_bytes(_model().to_bytes())
    assert back.classes == _model().classes
    (act,) = [c for c in back.classes if c["kind"] == "action"]
    assert act["fields"] == [["sz", U16, 0, 2], ["src", U32, 4, 4]]


def test_root_rows_and_a_nested_candidate():
    m = _model()
    other = ZbcModel(coros=[CoroDescriptor(name="poke_a")])
    other.entries = {"poke_a": EntryPoint(coro=0, qname="uart_c::poke_a", exported=True)}
    other.api_hash = api_hash_of(other.entries, {})
    other.roots = [{"type": "uart_c", "ctor": None, "api_hash": other.api_hash,
                    "image": None}]
    m.roots.append({"type": "uart_c", "ctor": None, "api_hash": other.api_hash,
                    "image": other.to_bytes()})
    back = ZbcModel.from_bytes(m.to_bytes())
    assert back.root_types() == ["dma_c", "uart_c"]
    assert back.roots[0]["ctor"]["name"] == "initialize"
    assert back.roots[0]["api_hash"] == m.api_hash
    assert back.root_model("dma_c") is back
    u = back.root_model("uart_c")
    assert u.root_types() == ["uart_c"]
    assert u.entries["poke_a"].exported
    assert back.roots[1]["api_hash"] == u.api_hash
    with pytest.raises(KeyError):
        back.root_model("nope_c")


def test_the_runtime_profile_carries_no_metadata():
    m = _model()
    m.profile = "runtime"
    kinds = {s.kind for s in m.to_container().sections}
    for name in ("ZBC_SEC_API", "ZBC_SEC_CLASS", "ZBC_SEC_ROOT", "ZBC_SEC_IMPORT"):
        assert spec.sec_kind(name) not in kinds
