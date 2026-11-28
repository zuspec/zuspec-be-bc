"""
bench.py -- the generic bench's arguments an image implies (F7.6a).

The generic bench (pssc's share/sv/tb/pssvm_tb.sv) is an open sparse memory: any
address reads and writes. An image with a memory map (``ZbcModel.memmap``,
the MEMMAP section) names the regions its tests may touch; the bench takes
them as ``+pssvm.tb.win=<addr>:<size>,..`` (hex) and fails an access outside
every window that is not a register's::

    from zuspec.be.bc.bench import plusargs
    subprocess.run([simv, f"+pssvm.img={img}", *plusargs(model)])
"""
from typing import Dict, List, Tuple


def windows(memmap: Dict) -> List[Tuple[int, int]]:
    """``(lo, size)`` of each region of *memmap* that has an address: a
    transparent region's ``addr``, else a contiguous one's opaque ``base``.
    Merged where they touch or overlap, in address order."""
    out: List[Tuple[int, int]] = []
    for r in (memmap or {}).get("regions") or ():
        lo = r.get("addr") if r.get("addr") is not None else r.get("base")
        if lo is None or not r.get("size"):
            continue
        out.append((int(lo), int(r["size"])))
    merged: List[Tuple[int, int]] = []
    for lo, size in sorted(out):
        if merged and lo <= merged[-1][0] + merged[-1][1]:
            plo, psize = merged[-1]
            merged[-1] = (plo, max(psize, lo + size - plo))
        else:
            merged.append((lo, size))
    return merged


def plusargs(model) -> List[str]:
    """The bench arguments *model*'s memory map implies: ``+pssvm.tb.win=..``,
    or none when it has no placed regions (the bench stays open)."""
    win = windows(getattr(model, "memmap", None) or {})
    if not win:
        return []
    return ["+pssvm.tb.win=" + ",".join(f"{lo:x}:{size:x}" for lo, size in win)]
