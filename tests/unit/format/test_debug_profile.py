"""The debug profile: codegen plus the PSS sources (SRC), flagged in the header
(pssc P1.5b)."""

import pytest

from zuspec.be.bc.format import spec
from zuspec.be.bc.format.emit_dataclass import CLASSES as DC
from zuspec.be.bc.format.reader import read_image
from zuspec.be.bc.format.writer import write_image
from zuspec.be.bc.interp import oracle
from zuspec.be.bc.model import CoroDescriptor, EntryPoint, ZbcModel

SOURCES = {"b.pss": "component pss_top {}\n", "a.pss": "package p {}\n"}


def _model(profile):
    m = ZbcModel(coros=[CoroDescriptor(name="t_a")], profile=profile)
    m.entries = {"t_a": EntryPoint(coro=0)}
    m.sources = dict(SOURCES)
    return m


def _kinds(m):
    return [s.kind for s in m.to_container().sections]


def test_debug_image_carries_the_sources():
    back = ZbcModel.from_bytes(_model("debug").to_bytes())
    assert back.profile == "debug"
    assert back.sources == SOURCES
    assert back.entries == {"t_a": EntryPoint(coro=0)}


def test_header_flags_say_debug():
    data = _model("debug").to_bytes()
    assert read_image(data).profile == "debug"
    f = DC["zbc_header"].from_bytes(data).flags
    assert f & spec.flag_value("zbc_hdr_flags", "ZBC_HDR_PROFILE_DEBUG")
    assert f & spec.flag_value("zbc_hdr_flags", "ZBC_HDR_HAS_PROV")


def test_codegen_drops_the_sources_and_keeps_its_bytes():
    src = spec.sec_kind("ZBC_SEC_SRC")
    assert src in _kinds(_model("debug"))
    assert src not in _kinds(_model("codegen"))
    plain = _model("codegen")
    plain.sources = {}
    assert _model("codegen").to_bytes() == plain.to_bytes()


def test_debug_differs_from_codegen_only_by_src_and_flag():
    dbg = _model("debug")
    dbg.sources = {}
    a, b = dbg.to_bytes(), _model("codegen").to_bytes()
    assert len(a) == len(b)
    assert sum(x != y for x, y in zip(a, b)) <= 9  # flags byte + content hash


def test_runtime_refuses_src():
    img = _model("debug").to_container()
    img.profile = "runtime"
    with pytest.raises(ValueError):
        write_image(img)


def test_oracle_roundtrip_keeps_debug_sources():
    m = _model("debug")
    assert oracle.roundtrip(m).sources == SOURCES


# ----- IDENT: API hash + generation id (pssc P1.5c) ----- #

from zuspec.be.bc.model import api_hash_of

IMPORTS = {"bus_write": {"fn_id": 1, "blocking": True, "arg_types": [[32, False], [32, False]],
                         "ret_type": None, "string_at": []}}


def _ident_model(profile, generation=0):
    m = _model(profile)
    m.imports = dict(IMPORTS)
    m.api_hash = api_hash_of(m.entries, m.imports)
    m.generation = generation
    return m


def test_debug_image_carries_api_hash_and_generation():
    m = _ident_model("debug", generation=7)
    assert spec.sec_kind("ZBC_SEC_IDENT") in _kinds(m)
    back = ZbcModel.from_bytes(m.to_bytes())
    assert (back.api_hash, back.generation) == (m.api_hash, 7)
    oracle.roundtrip(m)


def test_codegen_leaves_ident_out_and_keeps_its_bytes():
    m = _ident_model("codegen", generation=7)
    assert spec.sec_kind("ZBC_SEC_IDENT") not in _kinds(m)
    plain = _model("codegen")
    plain.imports = dict(IMPORTS)
    assert m.to_bytes() == plain.to_bytes()


def test_import_names_survive_a_read_back():
    m = _ident_model("debug")
    back = ZbcModel.from_bytes(m.to_bytes())
    assert back.imports == IMPORTS
    assert back.import_names == {sig["fn_id"]: n for n, sig in IMPORTS.items()}
    assert back.to_bytes() == m.to_bytes()
    assert spec.sec_kind("ZBC_SEC_IMPORT") not in _kinds(_model("debug"))


def test_api_hash_follows_signatures_not_coroutine_indices():
    entries = {"t_a": EntryPoint(coro=0), "c::f": EntryPoint(
        coro=1, kind="function", comp_type="c", params=[("x", {"kind": "int", "width": 32})])}
    h = api_hash_of(entries, IMPORTS)
    assert h != 0
    moved = {"c::f": EntryPoint(coro=5, kind="function", comp_type="c",
                                params=[("x", {"kind": "int", "width": 32})]),
             "t_a": EntryPoint(coro=3)}
    assert api_hash_of(moved, IMPORTS) == h
    renamed = dict(entries, **{"c::f": EntryPoint(
        coro=1, kind="function", comp_type="c", params=[("y", {"kind": "int", "width": 32})])})
    assert api_hash_of(renamed, IMPORTS) != h
    imp2 = {"bus_write": dict(IMPORTS["bus_write"], arg_types=[[64, False], [32, False]])}
    assert api_hash_of(entries, imp2) != h
    assert api_hash_of(entries, {}) != h


META = {"pssc.catalog": '{"tests": []}', "pssc.registers": "{}"}


def test_debug_image_carries_host_metadata():
    m = _model("debug")
    m.meta = dict(META)
    assert spec.sec_kind("ZBC_SEC_META") in _kinds(m)
    assert ZbcModel.from_bytes(m.to_bytes()).meta == META
    assert oracle.roundtrip(m).meta == META


def test_codegen_and_empty_meta_keep_their_bytes():
    plain = _model("debug")
    m = _model("debug")
    m.meta = {}
    assert m.to_bytes() == plain.to_bytes()
    c = _model("codegen")
    c.meta = dict(META)
    assert spec.sec_kind("ZBC_SEC_META") not in _kinds(c)
    assert c.to_bytes() == _model("codegen").to_bytes()
