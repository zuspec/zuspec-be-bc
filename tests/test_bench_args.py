"""The generic bench's memory windows from an image's memory map (F7.6a)."""
from types import SimpleNamespace

from zuspec.be.bc.bench import plusargs, windows


def test_windows_from_regions():
    mm = {"spaces": [{"path": "s"}], "regions": [
        {"space": 0, "name": "b", "addr": 0x2000, "size": 0x100},
        {"space": 0, "name": "a", "addr": 0x1000, "size": 0x1000},   # touches b
        {"space": 0, "name": "c", "base": 0x8000, "size": 0x10},     # contiguous
        {"space": 0, "name": "d", "size": 0x10},                     # not placed
    ]}
    assert windows(mm) == [(0x1000, 0x1100), (0x8000, 0x10)]
    assert plusargs(SimpleNamespace(memmap=mm)) == ["+pssvm_win=1000:1100,8000:10"]


def test_no_regions_no_plusarg():
    assert plusargs(SimpleNamespace(memmap={})) == []
    assert plusargs(SimpleNamespace(memmap={"spaces": [{}], "regions": []})) == []
