"""
layout.py -- the normative native layout of an object (value-abi.md §2.1).

An object (an action, a component's subtree, a plain-data struct) is a list of
scalar *leaves* in slot order, each named by its dotted path from the object
(``cfg.src``, ``ch.2.id``) and typed by a descriptor (``{"kind", "width",
"signed"}``, as in ``ENTRY.obj``). The interpreters address leaves by slot; this
module says where each leaf lives in the **native** representation an AOT
implementation, a host struct or a zero-copy view would use. It is C's layout:

* a leaf's size is its natural C size: ``bool`` 1 byte; an integer or enum of
  ``w`` bits the smallest of 1, 2, 4, 8 bytes holding it (wider: whole 8-byte
  words); a float ``w / 8`` (4 or 8); a string, chandle, collection, executor,
  component reference or opaque leaf 8 bytes. Its alignment is its size (8
  for a multi-word integer);
* the leaves that share a path prefix are a nested struct (``cfg.src`` and
  ``cfg.dst`` are struct ``cfg``), laid out recursively, aligned to its most
  aligned member and padded to a multiple of that, as C pads a struct; an
  array element (``ch.2``) is such a member too, so an array of structs is a C
  array;
* members are placed in slot order, each at the next offset aligned for it,
  and the object is padded to a multiple of its alignment.

Both engines keep addressing slots internally (the operands of ``LD_FIELD``
etc. are slot indices in this format version); the layout is metadata an
implementation and a host agree on byte for byte (the ``CLASS`` and ``LAYOUT``
sections carry it).
"""

from typing import Dict, List, Sequence, Tuple

#: a leaf: (dotted path, type descriptor)
Leaf = Tuple[str, dict]

_WORD = ("string", "chandle", "list", "map", "set", "exec", "compref", "opaque")


def leaf_size(desc: dict) -> int:
    """The native size (= alignment, but see :func:`leaf_align`) of a leaf of
    descriptor *desc*, in bytes."""
    kind = desc.get("kind", "int")
    width = int(desc.get("width") or 0)
    if kind == "bool":
        return 1
    if kind == "float":
        return 4 if width == 32 else 8
    if kind in _WORD:
        return 8
    if width <= 8:
        return 1
    if width <= 16:
        return 2
    if width <= 32:
        return 4
    return 8 * ((width + 63) // 64)


def leaf_align(desc: dict) -> int:
    return min(leaf_size(desc), 8)


def _align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def _tree(leaves: Sequence[Leaf]):
    """Group *leaves* by their first path segment, keeping slot order:
    ``[(segment, slot, desc) | (segment, [children])]``."""
    out: List[list] = []
    index: Dict[str, int] = {}
    for slot, (name, desc) in enumerate(leaves):
        head, dot, rest = name.partition(".")
        if not dot:
            out.append([head, (slot, desc)])
            continue
        i = index.get(head)
        if i is None or not isinstance(out[i][1], list):
            index[head] = len(out)
            out.append([head, []])
            i = index[head]
        out[i][1].append((rest, slot, desc))
    return out


def _place(leaves: List[Tuple[str, int, dict]], base: int,
           rows: Dict[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Lay *leaves* (``(path, slot, desc)``) out from offset *base*; fill
    *rows* (slot -> (offset, size)); return ``(size, align)``."""
    off, align = 0, 1
    for head, item in _tree([(p, d) for p, _, d in leaves]):
        if isinstance(item, tuple):
            k, desc = item
            slot = leaves[k][1]
            a, sz = leaf_align(desc), leaf_size(desc)
            off = _align_up(off, a)
            rows[slot] = (base + off, sz)
            off += sz
            align = max(align, a)
            continue
        # a nested struct: its members are the leaves under ``head.``
        members = [(rest, leaves[k][1], d) for rest, k, d in item]
        sub_rows: Dict[int, Tuple[int, int]] = {}
        sz, a = _place(members, 0, sub_rows)
        off = _align_up(off, a)
        for slot, (o, s) in sub_rows.items():
            rows[slot] = (base + off + o, s)
        off += sz
        align = max(align, a)
    return _align_up(off, align), align


def native_layout(leaves: Sequence[Leaf]) -> Tuple[List[Tuple[str, dict, int, int]], int, int]:
    """The native layout of an object whose slots are *leaves*
    (``[(name, descriptor), ...]`` in slot order): ``(rows, size, align)``,
    ``rows`` one ``(name, descriptor, offset, size)`` per slot, in slot
    order; ``size`` the object's padded size in bytes. An empty object has
    size 0 and alignment 1."""
    leaves = list(leaves)
    if not leaves:
        return [], 0, 1
    rows: Dict[int, Tuple[int, int]] = {}
    size, align = _place([(n, i, d) for i, (n, d) in enumerate(leaves)], 0, rows)
    return ([(n, d, rows[i][0], rows[i][1]) for i, (n, d) in enumerate(leaves)],
            size, align)
