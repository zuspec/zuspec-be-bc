"""The Python engine's heap (F6.9c; docs/spec/heap.md): handles, ops,
value copies, reclaim, cap, determinism, and the HEAP opcode in the VM."""

import pytest

from zuspec.be.bc.determinism import SeedStream
from zuspec.be.bc.interp import run_model, roundtrip, VMError
from zuspec.be.bc.interp.heap import Heap, STR_BIT
from zuspec.be.bc.model import (
    ZbcModel, CoroDescriptor, Instr, Op, heap_imm,
    HEAP_SIZE, HEAP_GET, HEAP_SET, HEAP_INSERT, HEAP_DELETE, HEAP_PUSH_BACK,
    HEAP_PUSH_FRONT, HEAP_POP_FRONT, HEAP_POP_BACK, HEAP_CLEAR, HEAP_IN,
    HEAP_EQ, HEAP_COPY, HEAP_FREE, HEAP_KEYS, HEAP_VALUES, HEAP_TO_LIST,
    HEAP_TO_SET, HEAP_SHUFFLE, HEAP_JOIN, HEAP_STR_SLICE, HEAP_STR_SPLIT,
    HEAP_STR_EQ, HEAP_STR_UPPER,
    HEAPK_STR, HEAPK_LIST, HEAPK_MAP, HEAPK_SET,
    HEAP_CELL_INT, HEAP_CELL_STR, HEAP_CELL_COLL,
)

M64 = (1 << 64) - 1


class _F:
    seed = SeedStream(1)


def op(hp, sub, kind=HEAPK_LIST, a=0, b=0, c=0, ck=HEAP_CELL_INT, kk=HEAP_CELL_INT):
    return hp.op(_F, sub, kind, ck, kk, a, b, c)


def lst(hp, *xs):
    h = 0
    for x in xs:
        h = op(hp, HEAP_PUSH_BACK, a=h, b=x)
    return h


def test_handle_0_is_empty_and_ops_allocate_lazily():
    hp = Heap()
    assert op(hp, HEAP_SIZE, a=0) == 0 and hp.live == 0
    h = op(hp, HEAP_PUSH_BACK, a=0, b=5)
    assert h == 1 and hp.live == 1
    assert op(hp, HEAP_PUSH_BACK, a=h, b=6) == h       # in place: same handle
    assert hp.items(h) == [5, 6]


def test_list_ops():
    hp = Heap()
    h = lst(hp, 1, 2, 3)
    h = op(hp, HEAP_PUSH_FRONT, a=h, b=0)
    h = op(hp, HEAP_INSERT, a=h, b=4, c=9)              # i == size appends
    h = op(hp, HEAP_SET, a=h, b=1, c=7)
    h = op(hp, HEAP_DELETE, a=h, b=2)
    assert hp.items(h) == [0, 7, 3, 9]
    assert op(hp, HEAP_GET, a=h, b=3) == 9
    assert op(hp, HEAP_IN, a=h, b=7) == 1 and op(hp, HEAP_IN, a=h, b=8) == 0
    assert op(hp, HEAP_POP_FRONT, a=h) == (0, h)
    assert op(hp, HEAP_POP_BACK, a=h) == (9, h)
    assert op(hp, HEAP_CLEAR, a=h) == h and op(hp, HEAP_SIZE, a=h) == 0


@pytest.mark.parametrize("sub,b,msg", [
    (HEAP_GET, 3, "index out of range"),
    (HEAP_GET, -1 & M64, "index out of range"),
    (HEAP_SET, 5, "index out of range"),
    (HEAP_INSERT, 4, "index out of range"),
    (HEAP_DELETE, 3, "index out of range"),
])
def test_list_index_errors(sub, b, msg):
    hp = Heap()
    with pytest.raises(VMError, match=msg):
        op(hp, sub, a=lst(hp, 1, 2, 3), b=b, c=0)


def test_pop_of_an_empty_list_and_bad_handle():
    hp = Heap()
    with pytest.raises(VMError, match="pop of an empty list"):
        op(hp, HEAP_POP_BACK, a=0)
    h = lst(hp, 1)
    op(hp, HEAP_POP_BACK, a=h)
    with pytest.raises(VMError, match="pop of an empty list"):
        op(hp, HEAP_POP_FRONT, a=h)
    op(hp, HEAP_FREE, a=h)
    with pytest.raises(VMError, match="bad heap handle"):
        op(hp, HEAP_SIZE, a=h)
    assert lst(hp, 1) == h + 1                           # never reused


def test_map_keeps_insertion_order_and_replaces_in_place():
    hp = Heap()
    m = 0
    for k, v in ((3, 30), (1, 10), (2, 20), (3, 33)):
        m = op(hp, HEAP_SET, HEAPK_MAP, a=m, b=k, c=v)
    assert hp.items(m) == [33, 10, 20]
    assert op(hp, HEAP_GET, HEAPK_MAP, a=m, b=1) == 10
    assert op(hp, HEAP_IN, HEAPK_MAP, a=m, b=2) == 1
    with pytest.raises(VMError, match="missing map key"):
        op(hp, HEAP_GET, HEAPK_MAP, a=m, b=9)
    op(hp, HEAP_DELETE, HEAPK_MAP, a=m, b=9)             # missing key: no error
    assert hp.items(op(hp, HEAP_KEYS, HEAPK_MAP, a=m)) == [3, 1, 2]
    assert hp.items(op(hp, HEAP_VALUES, HEAPK_MAP, a=m)) == [33, 10, 20]


def test_set_dedups_in_insertion_order_and_converts():
    hp = Heap()
    s = 0
    for x in (5, 1, 5, 3):
        s = op(hp, HEAP_INSERT, HEAPK_SET, a=s, b=x)
    assert hp.items(s) == [5, 1, 3]
    assert hp.items(op(hp, HEAP_TO_LIST, HEAPK_SET, a=s)) == [5, 1, 3]
    assert hp.items(op(hp, HEAP_TO_SET, a=lst(hp, 2, 2, 1))) == [2, 1]


def test_eq_lists_in_order_sets_and_maps_as_sets():
    hp = Heap()
    assert op(hp, HEAP_EQ, a=lst(hp, 1, 2), b=lst(hp, 1, 2)) == 1
    assert op(hp, HEAP_EQ, a=lst(hp, 1, 2), b=lst(hp, 2, 1)) == 0
    empty = op(hp, HEAP_POP_BACK, a=lst(hp, 1))[1]
    assert op(hp, HEAP_EQ, a=0, b=empty) == 1           # handle 0 is the empty list
    s1 = op(hp, HEAP_TO_SET, a=lst(hp, 1, 2))
    s2 = op(hp, HEAP_TO_SET, a=lst(hp, 2, 1))
    assert op(hp, HEAP_EQ, a=s1, b=s2) == 1


def test_copy_is_deep_and_free_reclaims_everything():
    hp = Heap()
    inner = lst(hp, 1, 2)
    outer = op(hp, HEAP_PUSH_BACK, a=0, b=inner, ck=HEAP_CELL_COLL)
    live, nbytes = hp.live, hp.bytes
    cp = op(hp, HEAP_COPY, a=outer)
    assert hp.live == 2 * live
    assert op(hp, HEAP_EQ, a=outer, b=cp) == 1
    op(hp, HEAP_PUSH_BACK, a=hp.items(cp)[0], b=3)      # change the copy's inner
    assert hp.items(inner) == [1, 2]
    op(hp, HEAP_FREE, a=cp)
    assert (hp.live, hp.bytes) == (live, nbytes)
    op(hp, HEAP_FREE, a=outer)
    assert (hp.live, hp.bytes) == (0, 0)
    assert op(hp, HEAP_COPY, a=0) == 0


def test_set_and_delete_reclaim_the_cells_they_drop():
    hp = Heap(strings=["a"])
    l = op(hp, HEAP_PUSH_BACK, a=0, b=hp.new_str(b"x"), ck=HEAP_CELL_STR)
    op(hp, HEAP_SET, a=l, b=0, c=hp.new_str(b"y"), ck=HEAP_CELL_STR)
    assert hp.live == 2
    op(hp, HEAP_DELETE, a=l, b=0)
    assert hp.live == 1


def test_set_of_a_nested_collection_to_itself_keeps_it():
    """``ll[0].push_back(1)`` stores the inner list back in its own slot."""
    hp = Heap()
    inner = op(hp, HEAP_PUSH_BACK, a=0, b=1)
    outer = op(hp, HEAP_PUSH_BACK, a=0, b=inner, ck=HEAP_CELL_COLL)
    op(hp, HEAP_SET, a=outer, b=0, c=inner, ck=HEAP_CELL_COLL)
    m = op(hp, HEAP_SET, kind=HEAPK_MAP, a=0, b=5, c=hp.copy(inner, HEAP_CELL_COLL),
           ck=HEAP_CELL_COLL)
    v = hp.items(m)[0]
    op(hp, HEAP_SET, kind=HEAPK_MAP, a=m, b=5, c=v, ck=HEAP_CELL_COLL)
    assert hp.live == 4 and hp.items(hp.items(outer)[0]) == [1] == hp.items(v)
    hp.free(outer, HEAP_CELL_COLL)
    hp.free(m, HEAP_CELL_COLL)
    assert (hp.live, hp.bytes) == (0, 0)


def test_strings_table_and_heap():
    hp = Heap(strings=["", "Hello", "a,b,,c", ","])
    up = op(hp, HEAP_STR_UPPER, HEAPK_STR, a=1)
    assert up & STR_BIT and hp.text(up) == b"HELLO"
    assert op(hp, HEAP_SIZE, HEAPK_STR, a=1) == 5
    assert op(hp, HEAP_STR_EQ, HEAPK_STR, a=1, b=hp.new_str(b"Hello")) == 1
    assert op(hp, HEAP_STR_EQ, HEAPK_STR, a=1, b=up) == 0
    assert hp.text(op(hp, HEAP_STR_SLICE, HEAPK_STR, a=1, b=1, c=3)) == b"ell"
    assert hp.text(op(hp, HEAP_STR_SLICE, HEAPK_STR, a=1, b=2, c=M64)) == b"llo"
    parts = op(hp, HEAP_STR_SPLIT, HEAPK_STR, a=2, b=3)
    assert [hp.text(x) for x in hp.items(parts)] == [b"a", b"b", b"", b"c"]
    assert hp.text(op(hp, HEAP_JOIN, a=parts, b=1)) == b"aHellobHelloHelloc"
    assert op(hp, HEAP_COPY, HEAPK_STR, a=1) == 1        # a literal is its own copy


def test_cap_stops_with_heap_exhausted():
    hp = Heap(cap=200)
    h = 0
    with pytest.raises(VMError, match="heap exhausted"):
        for i in range(100):
            h = op(hp, HEAP_PUSH_BACK, a=h, b=i)


def test_shuffle_is_seeded():
    def run(seed):
        hp = Heap()
        _F.seed = SeedStream(seed)
        return hp.items(op(hp, HEAP_SHUFFLE, a=lst(hp, *range(10))))
    assert run(7) == run(7)
    assert sorted(run(7)) == list(range(10))


def test_heap_opcode_runs_in_the_vm_and_round_trips():
    lp = heap_imm(HEAP_PUSH_BACK, HEAPK_LIST)
    code = [
        Instr(Op.CONST, (0,), imm=0),
        Instr(Op.CONST, (1,), imm=7),
        Instr(Op.HEAP, (2, 0, 1), imm=lp),
        Instr(Op.CONST, (1,), imm=8),
        Instr(Op.HEAP, (2, 2, 1), imm=lp),
        Instr(Op.HEAP, (3, 2), imm=heap_imm(HEAP_POP_FRONT, HEAPK_LIST)),
        Instr(Op.HEAP, (5, 2), imm=heap_imm(HEAP_SIZE, HEAPK_LIST)),
        Instr(Op.ADD, (6, 3, 5)),
        Instr(Op.RET, (6,)),
    ]
    m = ZbcModel(coros=[CoroDescriptor(name="h", code=code)], entry_coro=0)
    assert run_model(roundtrip(m)).retval == 8           # popped 7 + size 1
