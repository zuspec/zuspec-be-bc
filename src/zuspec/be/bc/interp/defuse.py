"""Static def/use of each PSS statement (debug lowering), for trace queries.

A statement's code runs from its start pc (``CoroDescriptor.stmts``) to the
next statement's start. Its *uses* are the values it loads and its *defs* the
values it stores, each named by a token:

- ``l:<name>`` a PSS local (``CoroDescriptor.locals``; ``l#<slot>`` when the
  slot has no name in scope);
- ``a#<i>`` argument ``i`` of the frame's call;
- ``f#<slot>`` a field of the frame's object (action, struct);
- ``c#<slot>`` a field of the frame's component instance; the VM's ``stmt``
  event makes the slot absolute in the component tree (:func:`abs_comp`), so
  a write in one call matches a read in another.

Branches inside the range are ignored (a may-def/may-use over-approximation).
``trace why`` walks these backwards through a run's ``stmt`` events.
"""
from typing import Dict, List, Tuple

from ..model import Op

_USE = {Op.LD_LOCAL: "l", Op.LD_ARG: "a", Op.LD_FIELD: "f", Op.LD_COMP: "c"}
_DEF = {Op.ST_LOCAL: "l", Op.ST_FIELD: "f", Op.ST_COMP: "c"}


def _token(kind: str, slot: int, pc: int, lvars) -> str:
    if kind == "l":
        for v in lvars:
            if v.slot == slot and v.pc_start <= pc < v.pc_end:
                return f"l:{v.name}"
    return f"{kind}#{slot}"


def defuse(coro) -> Dict[int, Tuple[List[str], List[str]]]:
    """``{stmt pc: (defs, uses)}`` for ``coro``'s statements, each list in
    first-seen order without repeats; statements with neither are left out."""
    starts = sorted({pc for pc, _ in getattr(coro, "stmts", None) or ()})
    code = coro.code
    lvars = getattr(coro, "locals", None) or ()
    out = {}
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(code)
        defs, uses = [], []
        for pc in range(s, end):
            ins = code[pc]
            k = _USE.get(ins.op)
            if k is not None:
                t = _token(k, ins.args[1], pc, lvars)
                if t not in uses:
                    uses.append(t)
                continue
            k = _DEF.get(ins.op)
            if k is not None:
                t = _token(k, ins.args[1], pc, lvars)
                if t not in defs:
                    defs.append(t)
        if defs or uses:
            out[s] = (defs, uses)
    return out


def abs_comp(detail: dict, cbase: int) -> None:
    """Rebase the ``c#<slot>`` tokens of a ``stmt`` event's *detail* (a copy)
    by the frame's component base: the slot in the whole component tree."""
    for k in ("defs", "uses"):
        v = detail.get(k)
        if v and any(t.startswith("c#") for t in v):
            detail[k] = [f"c#{cbase + int(t[2:])}" if t.startswith("c#") else t
                         for t in v]
