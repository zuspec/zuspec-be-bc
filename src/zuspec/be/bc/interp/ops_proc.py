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

import ctypes
import ctypes.util
import math
import struct

from ..model import (Op, INSTR_F_FROM_POOL, FARITH_NEG, FCVT_S2F, FCVT_U2F,
                     FCVT_F2S, FCVT_F32, FMATH_FNS, FMATH_BINARY)
from ..abi.value import ScalarType
from ..abi.codec import decode_scalar

MASK64 = (1 << 64) - 1
_U64 = ScalarType(64, signed=False)


class VMError(RuntimeError):
    """Raised on an ill-formed or unsupported instruction at run time.

    The dispatch loop stamps it with the failing instruction's ``src_ref``
    (or its coroutine's, when the instruction has none) and the matching
    :class:`~zuspec.be.bc.model.Prov`, so a caller can report where in the
    PSS source the run stopped (P1.5).
    """

    src_ref: int = 0
    prov = None

    @property
    def location(self) -> str:
        """``file:line:col`` of the failing statement, or "" if unknown."""
        p = self.prov
        if p is None or not p.file or not p.line:
            return ""
        return f"{p.file}:{p.line}:{p.col}" if p.col else f"{p.file}:{p.line}"


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
    _set(frame, ins.args[0], frame.obj.get_field(frame.base + ins.args[1]))


def _op_st_field(frame, ins, model):
    if frame.obj is None:
        raise VMError("ST_FIELD with no active object")
    frame.obj.set_field(frame.base + ins.args[1], _get(frame, ins.args[0]))


def _comp_slot(frame, ins) -> int:
    if frame.cobj is None:
        raise VMError(f"{ins.op.name} with no component tree")
    if frame.comp is None:
        raise VMError("a component attribute is read before the action's solve "
                      "chose its component instance (comp, LRM 13.4.5)")
    return (frame.cbase + ins.args[1]) & 0xFFFFFFFF    # wraps: before the frame


def _op_ld_comp(frame, ins, model):
    _set(frame, ins.args[0], frame.cobj.get_field(_comp_slot(frame, ins)))


def _op_st_comp(frame, ins, model):
    frame.cobj.set_field(_comp_slot(frame, ins), _get(frame, ins.args[0]))


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


# Floating point (F6.6): registers hold binary64 bit patterns.

def bits_to_float(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b & MASK64))[0]


def float_to_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def round_f32(x: float) -> float:
    """*x* rounded to binary32 precision (nearest even; overflow to inf)."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def f2s(x: float) -> int:
    """binary64 to signed 64-bit, truncated toward zero; NaN or out of range
    gives 0x8000000000000000 (what x86 and rt-eng's C cast give)."""
    if math.isnan(x) or not -2.0 ** 63 <= x < 2.0 ** 63:
        return 1 << 63
    return int(x) & MASK64


def _fdiv(a: float, b: float) -> float:
    if b != 0.0:
        return a / b
    if math.isnan(a) or a == 0.0:
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1.0, b)


_FARITH = (lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b, _fdiv,
           lambda a, b: -a)
_FCMP = (lambda a, b: a == b, lambda a, b: a != b, lambda a, b: a < b,
         lambda a, b: a <= b, lambda a, b: a > b, lambda a, b: a >= b)


def _op_farith(frame, ins, model):
    a = bits_to_float(_get(frame, ins.args[1]))
    b = bits_to_float(_get(frame, ins.args[2])) if ins.imm != FARITH_NEG else 0.0
    _set(frame, ins.args[0], float_to_bits(_FARITH[ins.imm](a, b)))


def _libm():
    """The C library's math functions: the LRM defines the float library by
    them (24.5.2), and the C engine calls them, so both engines agree. Python's
    ``math`` raises on a domain error where C returns NaN or an infinity."""
    lib = ctypes.CDLL(ctypes.util.find_library("m") or None)
    fns = []
    for name in FMATH_FNS:
        f = getattr(lib, name)
        f.restype = ctypes.c_double
        f.argtypes = [ctypes.c_double] * (2 if name in FMATH_BINARY else 1)
        fns.append((f, name in FMATH_BINARY))
    return tuple(fns)


_FMATH = _libm()


def _op_fmath(frame, ins, model):
    f, binary = _FMATH[ins.imm]
    x = bits_to_float(_get(frame, ins.args[1]))
    r = f(x, bits_to_float(_get(frame, ins.args[2]))) if binary else f(x)
    _set(frame, ins.args[0], float_to_bits(r))


def _op_fcmp(frame, ins, model):
    a = bits_to_float(_get(frame, ins.args[1]))
    b = bits_to_float(_get(frame, ins.args[2]))
    _set(frame, ins.args[0], 1 if _FCMP[ins.imm](a, b) else 0)


def _op_fcvt(frame, ins, model):
    v = _get(frame, ins.args[1])
    k = ins.imm
    if k == FCVT_S2F:
        r = float_to_bits(float(v - (1 << 64) if v >> 63 else v))
    elif k == FCVT_U2F:
        r = float_to_bits(float(v))
    elif k == FCVT_F2S:
        r = f2s(bits_to_float(v))
    elif k == FCVT_F32:
        r = float_to_bits(round_f32(bits_to_float(v)))
    else:
        raise VMError(f"FCVT: unknown conversion {k}")
    _set(frame, ins.args[0], r)


def _op_arg(frame, ins, model):
    i = ins.args[1]
    if len(frame.staged) <= i:
        frame.staged.extend([0] * (i + 1 - len(frame.staged)))
    frame.staged[i] = _get(frame, ins.args[0])


def _op_ld_arg(frame, ins, model):
    i = ins.args[1]
    _set(frame, ins.args[0], frame.args[i] if i < len(frame.args) else 0)


def _op_ld_res(frame, ins, model):
    i = ins.args[1]
    _set(frame, ins.args[0], frame.results[i] if i < len(frame.results) else 0)


#: op -> (handler, is_branch). Branch handlers return an absolute target pc.
_PROC = {
    Op.NOP: lambda f, i, m: None,
    Op.CONST: _op_const,
    Op.MOV: _op_mov,
    Op.LD_LOCAL: _op_ld_local,
    Op.ST_LOCAL: _op_st_local,
    Op.LD_FIELD: _op_ld_field,
    Op.ST_FIELD: _op_st_field,
    Op.LD_COMP: _op_ld_comp,
    Op.ST_COMP: _op_st_comp,
    Op.ARG: _op_arg,
    Op.LD_ARG: _op_ld_arg,
    Op.LD_RES: _op_ld_res,
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
    Op.FARITH: _op_farith,
    Op.FCMP: _op_fcmp,
    Op.FCVT: _op_fcvt,
    Op.FMATH: _op_fmath,
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
