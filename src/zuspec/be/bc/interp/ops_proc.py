"""
ops_proc.py -- procedural (register-SSA) op handlers (P1-10).

The straight-line compute tier. Register values are held as **unsigned 64-bit bit
patterns** (0 .. 2**64-1); the value codec (:mod:`..abi.codec`) is the single
authority on truncation/overflow, so results wrap exactly as the ABI specifies
rather than by ad-hoc Python. ``CONST`` materializes literals (inline immediate or
constant-pool id); loads/stores move between registers, frame locals, and object
fields; arithmetic/logic/compare follow the opcode groups in :class:`..model.Op`.

M1 semantics note: operands are treated as unsigned 64-bit for arithmetic and
comparison (the differential corpus is non-negative field values / import args).
Signed-aware DIV/MOD/SHR/compare is a documented later refinement; ``NEG`` already
produces the two's-complement bit pattern so signed *round-trips* through storage.
"""

from ..model import Op, INSTR_F_FROM_POOL
from ..abi.value import ScalarType
from ..abi.codec import decode_scalar

MASK64 = (1 << 64) - 1
_U64 = ScalarType(64, signed=False)


class VMError(RuntimeError):
    """Raised on an ill-formed or unsupported instruction at run time."""


def _u64(x: int) -> int:
    return int(x) & MASK64


def _get(frame, reg: int) -> int:
    try:
        return frame.regs[reg]
    except KeyError:
        raise VMError(f"read of undefined register r{reg}")


def _set(frame, reg: int, value: int) -> None:
    frame.regs[reg] = _u64(value)


# --------------------------------------------------------------------------- #
# Handlers. Each mutates ``frame`` in place; branch handlers return a target pc
# (absolute instr index) or None to fall through to the next instruction.
# --------------------------------------------------------------------------- #

def _op_const(frame, ins, model):
    rd = ins.args[0]
    if ins.flags & INSTR_F_FROM_POOL:
        bits, width = model.consts.entries[ins.imm]
        _set(frame, rd, bits)
    else:
        _set(frame, rd, ins.imm)


def _op_mov(frame, ins, model):
    _set(frame, ins.args[0], _get(frame, ins.args[1]))


def _op_ld_local(frame, ins, model):
    _set(frame, ins.args[0], frame.locals[ins.args[1]])


def _op_st_local(frame, ins, model):
    frame.locals[ins.args[1]] = _get(frame, ins.args[0])


def _op_ld_field(frame, ins, model):
    if frame.obj is None:
        raise VMError("LD_FIELD with no active object")
    _set(frame, ins.args[0], frame.obj.get_field(ins.args[1]))


def _op_st_field(frame, ins, model):
    if frame.obj is None:
        raise VMError("ST_FIELD with no active object")
    frame.obj.set_field(ins.args[1], _get(frame, ins.args[0]))


def _binop(fn):
    def handler(frame, ins, model):
        rd, ra, rb = ins.args[0], ins.args[1], ins.args[2]
        _set(frame, rd, fn(_get(frame, ra), _get(frame, rb)))
    return handler


def _cmp(fn):
    def handler(frame, ins, model):
        rd, ra, rb = ins.args[0], ins.args[1], ins.args[2]
        _set(frame, rd, 1 if fn(_get(frame, ra), _get(frame, rb)) else 0)
    return handler


def _div(a, b):
    if b == 0:
        raise VMError("division by zero")
    return a // b


def _mod(a, b):
    if b == 0:
        raise VMError("modulo by zero")
    return a % b


def _op_neg(frame, ins, model):
    _set(frame, ins.args[0], (-_get(frame, ins.args[1])))


def _op_not(frame, ins, model):
    # Bitwise/logical NOT: registers are bit patterns, so complement the 64 bits.
    _set(frame, ins.args[0], ~_get(frame, ins.args[1]))


#: op -> (handler, is_branch). Branch handlers return an absolute target pc.
_PROC = {
    Op.NOP: lambda f, i, m: None,
    Op.CONST: _op_const,
    Op.MOV: _op_mov,
    Op.LD_LOCAL: _op_ld_local,
    Op.ST_LOCAL: _op_st_local,
    Op.LD_FIELD: _op_ld_field,
    Op.ST_FIELD: _op_st_field,
    Op.ADD: _binop(lambda a, b: a + b),
    Op.SUB: _binop(lambda a, b: a - b),
    Op.MUL: _binop(lambda a, b: a * b),
    Op.DIV: _binop(_div),
    Op.MOD: _binop(_mod),
    Op.AND: _binop(lambda a, b: a & b),
    Op.OR: _binop(lambda a, b: a | b),
    Op.XOR: _binop(lambda a, b: a ^ b),
    Op.SHL: _binop(lambda a, b: a << (b & 63)),
    Op.SHR: _binop(lambda a, b: a >> (b & 63)),
    Op.NEG: _op_neg,
    Op.NOT: _op_not,
    Op.CMP_EQ: _cmp(lambda a, b: a == b),
    Op.CMP_NE: _cmp(lambda a, b: a != b),
    Op.CMP_LT: _cmp(lambda a, b: a < b),
    Op.CMP_LE: _cmp(lambda a, b: a <= b),
    Op.CMP_GT: _cmp(lambda a, b: a > b),
    Op.CMP_GE: _cmp(lambda a, b: a >= b),
}

#: Procedural opcodes this module handles (used by the VM to route dispatch).
PROC_OPS = frozenset(_PROC) | {Op.BR, Op.BRZ}


def exec_proc(frame, ins, model):
    """Execute one non-branch procedural op. Returns ``None`` (fall through)."""
    _PROC[ins.op](frame, ins, model)
    return None


def exec_branch(frame, ins, model):
    """Execute BR/BRZ. Returns the absolute target pc, or ``None`` to fall through.

    Branch targets are code-absolute instruction indices in M1, emitted by the
    control-flow lowering (``lower.control``) for non-suspending if/loop bodies.
    """
    if ins.op == Op.BR:
        return ins.args[0]
    if ins.op == Op.BRZ:
        if _get(frame, ins.args[0]) == 0:
            return ins.args[1]
        return None
    raise VMError(f"not a branch op: {ins.op.name}")


def is_proc(op) -> bool:
    return op in PROC_OPS
