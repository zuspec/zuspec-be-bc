"""The binary trace form, ``.pstrace`` (pssc P4.4c): lossless both ways
with the NDJSON form, compact, and strict about damaged files."""

import io

import pytest

from zuspec.be.bc.interp import oracle
from zuspec.be.bc.model import CoroDescriptor, Instr, Op, ZbcModel
from zuspec.be.bc.trace import BinarySink, MemorySink, serialize
from zuspec.be.bc.trace import binary as B
from zuspec.be.bc.trace import events as ev


def _call_model():
    # f() { return 7; }   root: f(); return
    f = CoroDescriptor(name="f", src_ref=0, code=[
        Instr(Op.CONST, args=(0,), imm=7), Instr(Op.RET, args=(0,))],
        stmts=[(0, 5)])
    root = CoroDescriptor(name="root", code=[
        Instr(Op.CALL, args=(1, 0xFFFFFFFF)), Instr(Op.RET)])
    return ZbcModel(coros=[root, f], entry_coro=0)


_EDGE = (
    '{"name":"t::a","seed":3,"zbc_trace":2}\n'
    '{"coro":0,"detail":{"pc":0},"kind":"stmt","seq":0,"src_ref":5}\n'
    '{"coro":2,"detail":{"addr":18446744073709551615,"data":-1,"rw":"w",'
    '"width":64},"file":"a.pss","kind":"access","line":12,"seq":1,'
    '"src_ref":7,"t":1500000}\n'
    '{"coro":0,"detail":{"args":[1,[true,false,null],{"k":"\\u00e9"}],'
    '"big":-340282366920938463463374607431768211456,"f":0.5},'
    '"kind":"import","seq":5,"src_ref":0}\n'
    '{"coro":0,"detail":{},"kind":"message","seq":6,"src_ref":0}\n'
)


def test_ndjson_round_trip_is_byte_identical():
    data = B.from_ndjson(_EDGE)
    assert B.is_binary(data)
    assert B.to_ndjson(data) == _EDGE


def test_vm_events_round_trip():
    mem, buf = MemorySink(), io.BytesIO()
    oracle.run_model(_call_model(), events=mem, watch=ev.EV_ALL)
    oracle.run_model(_call_model(), events=BinarySink(buf, name="x"),
                     watch=ev.EV_ALL)
    assert mem.events
    text = serialize(mem.events).replace('{"zbc_trace":2}',
                                         '{"name":"x","zbc_trace":2}', 1)
    assert B.to_ndjson(buf.getvalue()) == text


def test_binary_is_smaller_and_strings_are_interned():
    lines = ['{"zbc_trace":2}'] + [
        '{"coro":1,"detail":{"addr":%d,"data":%d,"rw":"r","width":32},'
        '"kind":"access","seq":%d,"src_ref":9}' % (0x1000 + 4 * i, i, i)
        for i in range(100)]
    text = "\n".join(lines) + "\n"
    data = B.from_ndjson(text)
    assert len(data) * 3 < len(text.encode())
    assert data.count(b"access") == 1


def test_damaged_files_are_refused():
    data = B.from_ndjson(_EDGE)
    with pytest.raises(ValueError, match="truncated"):
        B.read(data[:-1])
    with pytest.raises(ValueError, match="magic"):
        B.read(b"XSTRACE\0" + data[8:])
    with pytest.raises(ValueError, match="newer"):
        B.read(B.MAGIC + b"\x03" + data[9:])


def test_load_reads_either_form(tmp_path):
    nd, bn = tmp_path / "a.ndjson", tmp_path / "a.pstrace"
    nd.write_text(_EDGE)
    bn.write_bytes(B.from_ndjson(_EDGE))
    assert B.load(str(nd)) == B.load(str(bn))
    version, header, recs = B.load(str(bn))
    assert (version, header) == (2, {"name": "t::a", "seed": 3})
    assert [r["seq"] for r in recs] == [0, 1, 5, 6]
