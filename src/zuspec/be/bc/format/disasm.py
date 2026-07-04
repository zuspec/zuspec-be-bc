"""
disasm.py -- opcode-level ZBC disassembly (T1-C golden dumps).

The generic container inspector (:mod:`.inspect`) shows the *section directory*;
this shows the *program*: each coroutine's frame locals, FSM blocks, and decoded
instruction stream, plus the const pool and the in-memory SOLVE problem table.
That is what makes a lowering change reviewable in a diff, so the output is
checked in as a golden.

Output is a pure function of the model (no addresses, hashes, or timestamps), so
it is stable across runs and machines. It disassembles the rich in-memory
:class:`~zuspec.be.bc.model.ZbcModel` (which still carries the M1 problem side
channel), not the serialized bytes.
"""

from typing import List

from ..model import (
    Op, INSTR_F_FROM_POOL, INSTR_F_BLOCKING, INSTR_F_HAS_RET,
)

_FLAG_NAMES = [
    (INSTR_F_FROM_POOL, "FROM_POOL"),
    (INSTR_F_BLOCKING, "BLOCKING"),
    (INSTR_F_HAS_RET, "HAS_RET"),
]


def _fmt_flags(flags: int) -> str:
    names = [name for bit, name in _FLAG_NAMES if flags & bit]
    extra = flags & ~sum(bit for bit, _ in _FLAG_NAMES)
    if extra:
        names.append(f"0x{extra:02x}")
    return "|".join(names)


def _fmt_instr(ins) -> str:
    parts: List[str] = []
    if ins.args:
        parts.append("args=(" + ", ".join(str(a) for a in ins.args) + ")")
    if ins.imm:
        parts.append(f"imm={ins.imm}")
    if ins.flags:
        parts.append("flags=" + _fmt_flags(ins.flags))
    if ins.src_ref:
        parts.append(f"src={ins.src_ref}")
    tail = ("  " + " ".join(parts)) if parts else ""
    return f"{ins.op.name:<9}{tail}"


def _op_name(value: int) -> str:
    try:
        return Op(value).name
    except ValueError:
        return f"0x{value:02x}"


def disassemble(model) -> str:
    """Return a deterministic opcode-level dump of an in-memory ``ZbcModel``."""
    out: List[str] = []
    out.append(
        f"zbc disassembly  abi={model.abi_id} profile={model.profile} "
        f"entry={model.entry_coro}"
    )

    if model.consts.entries:
        out.append("consts:")
        for i, (bits, width) in enumerate(model.consts.entries):
            out.append(f"  #{i}  width={width}  bits=0x{bits:x}")

    if model.problems:
        out.append("problems:")
        for i, p in enumerate(model.problems):
            wb = ", ".join(f"{k}:{p.writeback[k]}" for k in sorted(p.writeback))
            seed = p.seed_kind + (f":{p.seed_value}" if p.seed_kind == "fixed" else "")
            out.append(
                f"  #{i}  vars=[{', '.join(p.var_names)}]  "
                f"writeback={{{wb}}}  seed={seed}"
            )

    if getattr(model, "selects", None):
        out.append("selects:")
        for i, sel in enumerate(model.selects):
            guards = getattr(sel, "guards", None) or [-1] * len(sel.branches)
            entries = ", ".join(
                f"coro{b}:w{w}" + (f":g{g}" if g >= 0 else "")
                for b, w, g in zip(sel.branches, sel.weights, guards))
            none = "  allow_none" if getattr(sel, "allow_none", False) else ""
            out.append(f"  #{i}  branches=[{entries}]{none}")

    for ci, coro in enumerate(model.coros):
        locals_ = ", ".join(coro.frame_locals)
        src = f"  src={coro.src_ref}" if coro.src_ref else ""
        out.append(f"coro #{ci} {coro.name!r}  frame_locals=[{locals_}]{src}")
        for blk in coro.blocks:
            suspend = _op_name(blk.suspend_op) if blk.suspend_op else "-"
            out.append(
                f"  block #{blk.idx}  pc={blk.pc_start}..{blk.pc_end}  "
                f"suspend={suspend}"
            )
            for pc in range(blk.pc_start, blk.pc_end):
                out.append(f"    {pc:>3}: {_fmt_instr(coro.code[pc])}")

    return "\n".join(out) + "\n"
