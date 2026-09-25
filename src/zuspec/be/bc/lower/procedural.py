"""
procedural.py -- Layer-0 Stmt/Expr -> register-SSA procedural ops.

Exec blocks and the native PSS functions they call are lowered here: locals,
assignment and compound assignment, ``if``/``match``, ``repeat``/``while``/
``repeat-while`` with ``break``/``continue``, ``return``, calls, and ``message``.

**Types.** Every expression is lowered against the static type LRM 8.7 gives it
(:mod:`.types`): operands are converted to the propagated type, results are
re-normalized to it, and assignment truncates or extends to the target. Signed
compare, divide, modulus and right shift are built from the unsigned 64-bit ops
the ISA already has, so none of this adds an opcode.

**Calls.** A native PSS function is *inlined* at each call site: its parameters
and locals get fresh frame slots, and ``return`` stores the result and branches
to the end of the inlined body. A recursive call cannot be inlined and is
rejected -- an honest "unsupported", never a wrong answer.

**message().** Lowers to an ``IMPORT`` of the reserved builtin
:data:`~..model.BUILTIN_MESSAGE`. The format string and each argument's type go
to the model's message table; the interpreter formats per LRM 21.1.1.

Anything else raises :class:`LoweringError` (never a silent miscompile).
Untyped legacy input (``ExprRefField`` / an undeclared ``ExprRefLocal`` with no
layout, as the hand-built scenario tests use) is treated as ``bit[64]``, which
is exactly the unsigned 64-bit arithmetic this module had before types.
"""

import dataclasses as dc
from typing import Dict, List, Optional, Tuple

from zuspec.ir.core import expr as E
from zuspec.ir.core import stmt as S

from ..model import (Op, INSTR_F_HAS_RET, INSTR_F_BLOCKING,
                     BUILTIN_MESSAGE, BUILTIN_ERROR)
from .context import CoroCtx
from .errors import LoweringError, PssSemanticError
from .types import BOOL, I32, STRING, U64, T, from_datatype, literal_type, merge

_VOID = 0xFFFFFFFF
_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63

_ARITH = {
    E.BinOp.Add: Op.ADD, E.BinOp.Sub: Op.SUB, E.BinOp.Mult: Op.MUL,
    E.BinOp.BitAnd: Op.AND, E.BinOp.BitOr: Op.OR, E.BinOp.BitXor: Op.XOR,
}
_DIVMOD = {E.BinOp.Div, E.BinOp.FloorDiv, E.BinOp.Mod}
_SHIFT = {E.BinOp.LShift, E.BinOp.RShift}
_CMP_BIN = {
    E.BinOp.Eq: Op.CMP_EQ, E.BinOp.NotEq: Op.CMP_NE,
    E.BinOp.Lt: Op.CMP_LT, E.BinOp.LtE: Op.CMP_LE,
    E.BinOp.Gt: Op.CMP_GT, E.BinOp.GtE: Op.CMP_GE,
}
_CMP_OP = {
    E.CmpOp.Eq: Op.CMP_EQ, E.CmpOp.NotEq: Op.CMP_NE,
    E.CmpOp.Lt: Op.CMP_LT, E.CmpOp.LtE: Op.CMP_LE,
    E.CmpOp.Gt: Op.CMP_GT, E.CmpOp.GtE: Op.CMP_GE,
}
_AUG = {
    E.AugOp.Add: E.BinOp.Add, E.AugOp.Sub: E.BinOp.Sub,
    E.AugOp.Mult: E.BinOp.Mult, E.AugOp.Div: E.BinOp.Div, E.AugOp.Mod: E.BinOp.Mod,
    E.AugOp.LShift: E.BinOp.LShift, E.AugOp.RShift: E.BinOp.RShift,
    E.AugOp.BitAnd: E.BinOp.BitAnd, E.AugOp.BitOr: E.BinOp.BitOr,
    E.AugOp.BitXor: E.BinOp.BitXor, E.AugOp.FloorDiv: E.BinOp.FloorDiv,
}


def _loc(n):
    return getattr(n, "loc", None)


def _cn(n) -> str:
    return type(n).__name__


# --------------------------------------------------------------------------- #
# Per-coroutine procedural state (hung off CoroCtx.proc)
# --------------------------------------------------------------------------- #

@dc.dataclass
class _Var:
    slot: str           # the frame-local slot name
    type: T


@dc.dataclass
class _Loop:
    breaks: List[int] = dc.field(default_factory=list)
    continues: List[int] = dc.field(default_factory=list)


@dc.dataclass
class _Fn:
    """An inlined call in progress."""
    fn: object
    scope_base: int                 # its locals start at this scope depth
    loop_base: int
    ret_type: Optional[T]
    ret_slot: Optional[str]
    returns: List[int] = dc.field(default_factory=list)


@dc.dataclass
class ProcState:
    fields: Dict[str, Tuple[int, T]] = dc.field(default_factory=dict)
    by_slot: Dict[int, T] = dc.field(default_factory=dict)
    component: Optional[str] = None
    scopes: List[Dict[str, _Var]] = dc.field(default_factory=list)
    loops: List[_Loop] = dc.field(default_factory=list)
    fns: List[_Fn] = dc.field(default_factory=list)
    exec_kind: str = "body"
    #: One scope per exec kind, shared by every ScExecBlock of that kind: an
    #: exec body split at a blocking import is still one scope (20.1).
    exec_scopes: Dict[str, Dict[str, _Var]] = dc.field(default_factory=dict)
    uniq: int = 0


def proc_state(ctx: CoroCtx) -> ProcState:
    st = getattr(ctx, "proc", None)
    if st is None:
        st = ProcState()
        ctx.proc = st
    return st


def init_proc_state(ctx: CoroCtx, coro) -> None:
    """Record the coroutine's attribute layout and owning component."""
    st = proc_state(ctx)
    for f in getattr(coro, "fields", []) or []:
        t = U64
        if f.datatype is not None:
            try:
                t = from_datatype(f.datatype)
            except LoweringError:
                t = None          # an attribute bc cannot type: error only if used
        st.fields[f.name] = (f.slot, t)
        if t is not None:
            st.by_slot[f.slot] = t
    at = getattr(coro, "action_type", None)
    if at and "::" in at:
        st.component = at.rsplit("::", 1)[0]


# --------------------------------------------------------------------------- #
# Emission helpers
# --------------------------------------------------------------------------- #

def _const(ctx: CoroCtx, v: int) -> int:
    return ctx.emit_const(int(v) & _MASK64)


def _op(ctx: CoroCtx, op: Op, a: int, b: int) -> int:
    r = ctx.new_reg()
    ctx.emit(op, (r, a, b))
    return r


def _patch(ctx: CoroCtx, at: int, target: int) -> None:
    ins = ctx.code[at]
    args = list(ins.args)
    args[-1] = int(target)
    ins.args = tuple(args)


def normalize(ctx: CoroCtx, r: int, t: T) -> int:
    """Re-establish the canonical form of an N-bit value (truncate, then extend)."""
    if t.is_bool or t.kind == "string" or t.width >= 64:
        return r
    m = _op(ctx, Op.AND, r, _const(ctx, (1 << t.width) - 1))
    if not t.signed:
        return m
    sb = _const(ctx, 1 << (t.width - 1))
    return _op(ctx, Op.SUB, _op(ctx, Op.XOR, m, sb), sb)


def propagate(ctx: CoroCtx, r: int, src: T, dst: T) -> int:
    """Convert an operand to the type propagated to it (8.7.1).

    The *propagated* type's signedness picks the extension: a signed operand
    propagated to an unsigned type is zero-extended from its own width.
    """
    if dst.is_bool:
        return truth(ctx, r, src)
    if src == dst:
        return r
    if src.signed and not dst.signed and src.width < 64:
        r = _op(ctx, Op.AND, r, _const(ctx, (1 << src.width) - 1))
    return normalize(ctx, r, dst.as_int())


def _fold_propagate(v: int, src: T, dst: T) -> int:
    """:func:`propagate` applied to a constant, at lowering time."""
    if dst.is_bool:
        return int(v != 0)
    if src.signed and not dst.signed and src.width < 64:
        v &= (1 << src.width) - 1
    w = min(dst.width, 64)
    v &= (1 << w) - 1
    if dst.signed and v >> (w - 1):
        v -= 1 << w
    return v


def assign_convert(ctx: CoroCtx, r: int, src: T, dst: T) -> int:
    """8.7.2: truncate or extend to the target; extension follows the SOURCE."""
    if dst.is_bool:
        return truth(ctx, r, src)
    if dst.kind == "string" or src.kind == "string":
        if dst.kind != src.kind:
            raise PssSemanticError("string and non-string types are not assignment-compatible")
        return r
    if src.as_int() == dst.as_int():
        return r
    return normalize(ctx, r, dst.as_int())


def truth(ctx: CoroCtx, r: int, t: T) -> int:
    if t.is_bool:
        return r
    return _op(ctx, Op.CMP_NE, r, _const(ctx, 0))


# --------------------------------------------------------------------------- #
# Scopes
# --------------------------------------------------------------------------- #

def _push_scope(ctx: CoroCtx) -> None:
    proc_state(ctx).scopes.append({})


def _pop_scope(ctx: CoroCtx) -> None:
    proc_state(ctx).scopes.pop()


def _declare(ctx: CoroCtx, name: str, t: T) -> _Var:
    st = proc_state(ctx)
    if not st.scopes:
        st.scopes.append({})
    st.uniq += 1
    v = _Var(slot=f"{name}${st.uniq}", type=t)
    ctx.local_slot(v.slot)
    st.scopes[-1][name] = v
    return v


def _lookup(ctx: CoroCtx, name: str) -> Optional[_Var]:
    st = proc_state(ctx)
    base = st.fns[-1].scope_base if st.fns else 0
    for scope in reversed(st.scopes[base:]):
        if name in scope:
            return scope[name]
    return None


# --------------------------------------------------------------------------- #
# Static typing (no code emitted)
# --------------------------------------------------------------------------- #

def type_of(ctx: CoroCtx, e) -> T:
    cn = _cn(e)
    if isinstance(e, E.ExprConstant):
        if isinstance(e.value, str):
            return STRING
        return literal_type(e.value)
    if isinstance(e, E.ExprRefLocal):
        v = _lookup(ctx, e.name)
        return v.type if v is not None else U64
    if isinstance(e, E.ExprRefField):
        return proc_state(ctx).by_slot.get(e.index, U64)
    if isinstance(e, (E.ExprAttribute, E.ExprRefUnresolved)):
        return _resolve_name(ctx, e)[1]
    if isinstance(e, E.ExprBin):
        if e.op in _CMP_BIN or e.op in (E.BinOp.And, E.BinOp.Or):
            return BOOL
        lt = type_of(ctx, e.lhs)
        if e.op in _SHIFT:
            return lt.as_int()
        return merge(lt, type_of(ctx, e.rhs))
    if isinstance(e, E.ExprUnary):
        if e.op == E.UnaryOp.Not:
            return BOOL
        return type_of(ctx, e.operand).as_int()
    if isinstance(e, (E.ExprCompare, E.ExprBool)):
        return BOOL
    if cn == "ExprIfExp":
        a, b = type_of(ctx, e.body), type_of(ctx, e.orelse)
        if a.is_bool and b.is_bool:
            return BOOL
        if a.kind == "enum" and a == b:
            return a
        return merge(a, b)
    if cn == "ExprCast":
        if e.target_type is None:
            raise LoweringError("a (void) cast has no value", loc=_loc(e))
        return from_datatype(e.target_type)
    if isinstance(e, E.ExprCall):
        return _call_type(ctx, e)
    raise LoweringError(f"unsupported expression {cn} in bc procedural code", loc=_loc(e))


# --------------------------------------------------------------------------- #
# Expressions
# --------------------------------------------------------------------------- #

def eval_expr(ctx: CoroCtx, expr) -> int:
    """Lower ``expr`` at its self-determined type; return the register."""
    return ev(ctx, expr)[0]


def ev(ctx: CoroCtx, e, want: Optional[T] = None) -> Tuple[int, T]:
    """Lower ``e``. With *want*, the result is converted to that (propagated) type."""
    r, t = _ev(ctx, e, want)
    if want is not None and t != want:
        r = propagate(ctx, r, t, want)
        t = want
    return r, t


def _ev(ctx: CoroCtx, e, want: Optional[T]) -> Tuple[int, T]:
    if isinstance(e, E.ExprConstant):
        v = e.value
        if isinstance(v, str):
            return _const(ctx, ctx.lowerer.intern_string(v)), STRING
        if not isinstance(v, (int, bool)):
            raise LoweringError(f"constant {v!r} is not supported by bc", loc=_loc(e))
        if want is not None:
            # Fold the conversion: one CONST of the already-converted value.
            return _const(ctx, _fold_propagate(int(v), literal_type(v), want)), want
        return _const(ctx, int(v)), literal_type(v)

    if isinstance(e, E.ExprRefLocal):
        v = _lookup(ctx, e.name)
        if v is None:
            if proc_state(ctx).fns:
                raise LoweringError(f"unresolved local {e.name!r}", loc=_loc(e))
            r = ctx.new_reg()
            ctx.emit(Op.LD_LOCAL, (r, ctx.local_slot(e.name)))   # legacy untyped local
            return r, U64
        r = ctx.new_reg()
        ctx.emit(Op.LD_LOCAL, (r, ctx.local_slot(v.slot)))
        return r, v.type

    if isinstance(e, E.ExprRefField):
        r = ctx.new_reg()
        ctx.emit(Op.LD_FIELD, (r, e.index))
        return r, proc_state(ctx).by_slot.get(e.index, U64)

    if isinstance(e, (E.ExprAttribute, E.ExprRefUnresolved)):
        kind, t, where = _resolve_name(ctx, e)
        r = ctx.new_reg()
        if kind == "local":
            ctx.emit(Op.LD_LOCAL, (r, ctx.local_slot(where)))
        else:
            ctx.emit(Op.LD_FIELD, (r, where))
        return r, t

    if isinstance(e, E.ExprBin):
        return _ev_bin(ctx, e, want)

    if isinstance(e, E.ExprUnary):
        if e.op == E.UnaryOp.Not:
            r, t = ev(ctx, e.operand)
            return _op(ctx, Op.CMP_EQ, truth(ctx, r, t), _const(ctx, 0)), BOOL
        t = want or type_of(ctx, e)
        r, _ = ev(ctx, e.operand, t)
        if e.op == E.UnaryOp.UAdd:
            return r, t
        out = ctx.new_reg()
        ctx.emit(Op.NEG if e.op == E.UnaryOp.USub else Op.NOT, (out, r))
        return normalize(ctx, out, t), t

    if isinstance(e, E.ExprCompare):
        if len(e.ops) != 1 or len(e.comparators) != 1:
            raise LoweringError("chained comparison is not supported", loc=_loc(e))
        op = _CMP_OP.get(e.ops[0])
        if op is None:
            raise LoweringError(f"unsupported comparison {e.ops[0]}", loc=_loc(e))
        return _compare(ctx, op, e.left, e.comparators[0]), BOOL

    if isinstance(e, E.ExprBool):
        return _logical(ctx, e.op == E.BoolOp.And, e.values), BOOL

    cn = _cn(e)
    if cn == "ExprIfExp":
        t = want or type_of(ctx, e)
        c, ct = ev(ctx, e.test)
        out = ctx.new_reg()
        brz = ctx.emit(Op.BRZ, (truth(ctx, c, ct), 0))
        a, _ = ev(ctx, e.body, t)
        ctx.emit(Op.MOV, (out, a))
        br = ctx.emit(Op.BR, (0,))
        _patch(ctx, brz, len(ctx.code))
        b, _ = ev(ctx, e.orelse, t)
        ctx.emit(Op.MOV, (out, b))
        _patch(ctx, br, len(ctx.code))
        return out, t

    if cn == "ExprCast":
        if e.target_type is None:
            raise LoweringError("a (void) cast has no value", loc=_loc(e))
        dst = from_datatype(e.target_type)
        r, src = ev(ctx, e.value)
        return assign_convert(ctx, r, src, dst), dst

    if isinstance(e, E.ExprCall):
        r, t = _call(ctx, e)
        if t is None:
            raise PssSemanticError("a void function call has no value", loc=_loc(e))
        return r, t

    raise LoweringError(f"unsupported expression {cn} in bc procedural code",
                        loc=_loc(e))


def _ev_bin(ctx: CoroCtx, e, want: Optional[T]) -> Tuple[int, T]:
    op = e.op
    if op in (E.BinOp.And, E.BinOp.Or):
        return _logical(ctx, op == E.BinOp.And, [e.lhs, e.rhs]), BOOL
    if op in _CMP_BIN:
        return _compare(ctx, _CMP_BIN[op], e.lhs, e.rhs), BOOL

    t = want.as_int() if want is not None else type_of(ctx, e)
    for side in (e.lhs, e.rhs):
        st = type_of(ctx, side)
        if st.kind == "string":
            raise LoweringError("string operators are not supported by bc", loc=_loc(e))

    if op in _SHIFT:
        a, _ = ev(ctx, e.lhs, t)
        n, _ = ev(ctx, e.rhs)                       # self-determined (Table 22)
        if op == E.BinOp.LShift:
            return normalize(ctx, _op(ctx, Op.SHL, a, n), t), t
        if t.signed:
            # Arithmetic shift from logical ones: ((x ^ S) >> n) - (S >> n).
            s = _const(ctx, _SIGN64)
            hi = _op(ctx, Op.SHR, _op(ctx, Op.XOR, a, s), n)
            return normalize(ctx, _op(ctx, Op.SUB, hi, _op(ctx, Op.SHR, s, n)), t), t
        return normalize(ctx, _op(ctx, Op.SHR, a, n), t), t

    a, _ = ev(ctx, e.lhs, t)
    b, _ = ev(ctx, e.rhs, t)
    if op in _ARITH:
        return normalize(ctx, _op(ctx, _ARITH[op], a, b), t), t
    if op in _DIVMOD:
        is_mod = op == E.BinOp.Mod
        if not t.signed:
            return _op(ctx, Op.MOD if is_mod else Op.DIV, a, b), t
        return normalize(ctx, _signed_divmod(ctx, a, b, is_mod), t), t
    if op == E.BinOp.Exp:
        raise LoweringError("the ** operator is not supported by bc", loc=_loc(e))
    raise LoweringError(f"unsupported operator {op}", loc=_loc(e))


def _sign_mask(ctx: CoroCtx, x: int) -> int:
    """All ones if canonical *x* is negative, else zero: 0 - (x >> 63)."""
    return _op(ctx, Op.SUB, _const(ctx, 0), _op(ctx, Op.SHR, x, _const(ctx, 63)))


def _abs(ctx: CoroCtx, x: int, m: int) -> int:
    return _op(ctx, Op.SUB, _op(ctx, Op.XOR, x, m), m)


def _signed_divmod(ctx: CoroCtx, a: int, b: int, is_mod: bool) -> int:
    """8.5.1: the quotient truncates toward zero; the remainder has a's sign."""
    ma, mb = _sign_mask(ctx, a), _sign_mask(ctx, b)
    ua, ub = _abs(ctx, a, ma), _abs(ctx, b, mb)
    if is_mod:
        return _abs(ctx, _op(ctx, Op.MOD, ua, ub), ma)        # (r ^ ma) - ma
    return _abs(ctx, _op(ctx, Op.DIV, ua, ub), _op(ctx, Op.XOR, ma, mb))


def _compare(ctx: CoroCtx, op: Op, lhs, rhs) -> int:
    lt, rt = type_of(ctx, lhs), type_of(ctx, rhs)
    if lt.is_bool and rt.is_bool:
        a, _ = ev(ctx, lhs)
        b, _ = ev(ctx, rhs)
        return _op(ctx, op, a, b)
    if lt.kind == "string" or rt.kind == "string":
        raise LoweringError("string comparison is not supported by bc")
    p = merge(lt, rt)
    a, _ = ev(ctx, lhs, p)
    b, _ = ev(ctx, rhs, p)
    return compare_regs(ctx, op, a, b, p)


def compare_regs(ctx: CoroCtx, op: Op, a: int, b: int, p: T) -> int:
    if p.signed and op not in (Op.CMP_EQ, Op.CMP_NE):
        s = _const(ctx, _SIGN64)
        a, b = _op(ctx, Op.XOR, a, s), _op(ctx, Op.XOR, b, s)
    return _op(ctx, op, a, b)


def _logical(ctx: CoroCtx, is_and: bool, operands) -> int:
    """Short-circuit && / ||; operands are self-determined."""
    out = ctx.new_reg()
    ends = []
    for i, x in enumerate(operands):
        r, t = ev(ctx, x)
        ctx.emit(Op.MOV, (out, truth(ctx, r, t)))
        if i == len(operands) - 1:
            break
        if is_and:
            ends.append(ctx.emit(Op.BRZ, (out, 0)))
        else:
            ends.append(ctx.emit(Op.BRZ, (_op(ctx, Op.CMP_EQ, out, _const(ctx, 0)), 0)))
    for j in ends:
        _patch(ctx, j, len(ctx.code))
    return out


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #

def _resolve_name(ctx: CoroCtx, e) -> Tuple[str, T, object]:
    """``self.x`` / ``x`` -> ("local", type, slot-name) or ("field", type, slot)."""
    if isinstance(e, E.ExprAttribute):
        if not isinstance(e.value, E.TypeExprRefSelf):
            raise LoweringError(f"reference through {_cn(e.value)} "
                                f"(.{e.attr}) is not supported by bc", loc=_loc(e))
        name = e.attr
    else:
        name = e.name
    v = _lookup(ctx, name)
    if v is not None:
        return "local", v.type, v.slot
    st = proc_state(ctx)
    if st.fns:
        # Inside an inlined function only its parameters and locals are in scope.
        raise LoweringError(f"function body references {name!r}, which is not a "
                            f"parameter or local (component attributes are not "
                            f"supported by bc)", loc=_loc(e))
    if name in st.fields:
        slot, t = st.fields[name]
        if t is None:
            raise LoweringError(f"attribute {name!r} has a type bc does not support",
                                loc=_loc(e))
        return "field", t, slot
    raise LoweringError(f"unresolved reference {name!r}", loc=_loc(e))


def _store(ctx: CoroCtx, target, r: int, src: T) -> None:
    if isinstance(target, E.ExprRefLocal):
        v = _lookup(ctx, target.name)
        if v is None:
            if proc_state(ctx).fns:
                raise LoweringError(f"unresolved local {target.name!r}", loc=_loc(target))
            ctx.emit(Op.ST_LOCAL, (r, ctx.local_slot(target.name)))   # legacy
            return
        ctx.emit(Op.ST_LOCAL, (assign_convert(ctx, r, src, v.type), ctx.local_slot(v.slot)))
        return
    if isinstance(target, E.ExprRefField):
        t = proc_state(ctx).by_slot.get(target.index, U64)
        ctx.emit(Op.ST_FIELD, (assign_convert(ctx, r, src, t), target.index))
        return
    if isinstance(target, (E.ExprAttribute, E.ExprRefUnresolved)):
        kind, t, where = _resolve_name(ctx, target)
        r = assign_convert(ctx, r, src, t)
        if kind == "local":
            ctx.emit(Op.ST_LOCAL, (r, ctx.local_slot(where)))
        else:
            ctx.emit(Op.ST_FIELD, (r, where))
        return
    raise LoweringError(f"unsupported assignment target {_cn(target)}", loc=_loc(target))


def _target_type(ctx: CoroCtx, target) -> T:
    if isinstance(target, E.ExprRefLocal):
        v = _lookup(ctx, target.name)
        return v.type if v is not None else U64       # legacy untyped local
    return type_of(ctx, target)


def _assign(ctx: CoroCtx, target, value) -> None:
    tt = _target_type(ctx, target)
    vt = type_of(ctx, value)
    if tt.is_bool or vt.is_bool or tt.kind == "string" or vt.kind == "string":
        want = None
    else:
        # 8.7.2: a wider target propagates its size; signedness stays the source's.
        want = T(max(tt.width, vt.as_int().width), vt.as_int().signed)
        if want == vt:
            want = None
    r, src = ev(ctx, value, want)
    _store(ctx, target, r, src)


# --------------------------------------------------------------------------- #
# Calls
# --------------------------------------------------------------------------- #

def _call_name(func) -> Tuple[Optional[str], Optional[str]]:
    """(name, scope) of a call target, as the front end resolved it.

    ``self.f`` -> ("f", "self") and ``comp.f`` -> ("f", "comp"): a function of
    the running component (``self.f`` may also be an import). A bare ``f`` or
    ``pkg::f`` (``ExprRefUnresolved``) -> (name, "pkg"): pssc's front end gives
    that form to a call its linker resolved to a package-scope function, by
    qualified name. The scope is the linker's answer; looking a bare name up in
    the component first would let a component's ``twice`` shadow the package's
    inside a package function.
    """
    if isinstance(func, E.ExprAttribute):
        if isinstance(func.value, E.TypeExprRefSelf):
            return func.attr, "self"
        v = func.value
        if isinstance(v, E.ExprAttribute) and isinstance(v.value, E.TypeExprRefSelf) \
                and v.attr == "comp":
            return func.attr, "comp"
        return None, None
    if isinstance(func, E.ExprRefUnresolved):
        return func.name, "pkg"
    return None, None


def _is_message(name: Optional[str]) -> bool:
    return name in ("message", "std_pkg::message")


def _lookup_function(ctx: CoroCtx, name: str, scope: Optional[str]):
    fns = ctx.lowerer.functions
    if scope == "pkg":
        return fns.get(name)
    st = proc_state(ctx)
    return fns.get(f"{st.component}::{name}") if st.component else None


def _call_type(ctx: CoroCtx, e) -> T:
    name, scope = _call_name(e.func)
    if name is None or _is_message(name):
        raise LoweringError("a void call has no value", loc=_loc(e))
    decl = ctx.lowerer.imports.get(name) if scope != "comp" else None
    if decl is not None:
        rt = decl.get("ret_type")
        if rt is None:
            raise PssSemanticError(f"void import {name!r} has no value", loc=_loc(e))
        return T(min(int(rt[0] or 32), 64), bool(rt[1]))
    fn = _lookup_function(ctx, name, scope)
    if fn is None:
        raise LoweringError(f"call to unknown function {name!r}", loc=_loc(e))
    if fn.returns is None:
        raise PssSemanticError(f"void function {name!r} has no value", loc=_loc(e))
    return from_datatype(fn.returns)


def _call(ctx: CoroCtx, e) -> Tuple[Optional[int], Optional[T]]:
    name, scope = _call_name(e.func)
    if name is None:
        raise LoweringError(f"unsupported call target {_cn(e.func)}", loc=_loc(e))
    if _is_message(name):
        _message(ctx, e)
        return None, None
    if scope != "comp" and name in ctx.lowerer.imports:
        return _import_call(ctx, name, e)
    fn = _lookup_function(ctx, name, scope)
    if fn is None:
        raise LoweringError(f"call to {name!r}: not a known function or import",
                            loc=_loc(e))
    return _inline(ctx, fn, e)


def _import_call(ctx: CoroCtx, name: str, e) -> Tuple[int, T]:
    decl = ctx.lowerer.imports[name]
    arg_regs = [ev(ctx, a)[0] for a in e.args]
    ret_reg = ctx.new_reg()
    flags = INSTR_F_HAS_RET | (INSTR_F_BLOCKING if decl["blocking"] else 0)
    ctx.emit(Op.IMPORT, tuple([decl["fn_id"], ret_reg] + arg_regs), flags=flags)
    ctx.lowerer.import_decls.setdefault(
        decl["fn_id"], {"fn": name, "blocking": decl["blocking"]})
    rt = decl.get("ret_type")
    return ret_reg, (T(min(int(rt[0] or 32), 64), bool(rt[1])) if rt else U64)


def _inline(ctx: CoroCtx, fn, call) -> Tuple[Optional[int], Optional[T]]:
    st = proc_state(ctx)
    if any(f.fn is fn for f in st.fns):
        raise LoweringError(f"recursive call to {fn.name!r}: bc inlines native "
                            f"functions and cannot inline recursion", loc=_loc(call))
    args = getattr(fn, "args", None)
    params = list(getattr(args, "args", []) or []) if args is not None else []
    if args is not None and getattr(args, "vararg", None) is not None:
        raise LoweringError(f"varargs function {fn.name!r} is not supported by bc",
                            loc=_loc(call))
    defaults = list(getattr(args, "defaults", []) or []) if args is not None else []
    if len(call.args) > len(params):
        raise PssSemanticError(f"too many arguments to {fn.name!r}", loc=_loc(call))
    first_default = len(params) - len(defaults)
    if len(call.args) < first_default:
        raise PssSemanticError(f"too few arguments to {fn.name!r}", loc=_loc(call))

    # Actual parameters are evaluated in the CALLER's scope (8.7.2: an
    # assignment-like context against the declared parameter type).
    values = []
    for i, p in enumerate(params):
        pt = from_datatype(p.annotation)
        src = call.args[i] if i < len(call.args) else defaults[i - first_default]
        values.append((p.arg, pt, _eval_for(ctx, src, pt)))

    ret_t = from_datatype(fn.returns) if fn.returns is not None else None
    frame = _Fn(fn=fn, scope_base=len(st.scopes), loop_base=len(st.loops),
                ret_type=ret_t, ret_slot=None)
    st.fns.append(frame)
    st.scopes.append({})
    try:
        for name, pt, r in values:
            v = _declare(ctx, name, pt)
            ctx.emit(Op.ST_LOCAL, (r, ctx.local_slot(v.slot)))
        if ret_t is not None:
            st.uniq += 1
            frame.ret_slot = f"{fn.name}$ret${st.uniq}"
            ctx.emit(Op.ST_LOCAL, (_const(ctx, 0), ctx.local_slot(frame.ret_slot)))
        _lower_block(ctx, fn.body)
        end = len(ctx.code)
        for j in frame.returns:
            _patch(ctx, j, end)
    finally:
        st.scopes.pop()
        st.fns.pop()

    if ret_t is None:
        return None, None
    r = ctx.new_reg()
    ctx.emit(Op.LD_LOCAL, (r, ctx.local_slot(frame.ret_slot)))
    return r, ret_t


def _eval_for(ctx: CoroCtx, value, dst: T) -> int:
    """Evaluate *value* in an assignment-like context and convert it to *dst*."""
    vt = type_of(ctx, value)
    want = None
    if not (dst.is_bool or vt.is_bool or dst.kind == "string" or vt.kind == "string"):
        want = T(max(dst.width, vt.as_int().width), vt.as_int().signed)
        if want == vt:
            want = None
    r, src = ev(ctx, value, want)
    return assign_convert(ctx, r, src, dst)


def _message(ctx: CoroCtx, e) -> None:
    """``message(verbosity, format, args...)`` -> IMPORT of the builtin sink."""
    if len(e.args) < 2:
        raise PssSemanticError("message() needs a verbosity and a format string",
                               loc=_loc(e))
    fmt = e.args[1]
    if not (isinstance(fmt, E.ExprConstant) and isinstance(fmt.value, str)):
        raise LoweringError("message() format must be a string literal in bc", loc=_loc(e))
    vreg, _ = ev(ctx, e.args[0])
    regs, descs = [vreg], []
    for a in e.args[2:]:
        r, t = ev(ctx, a)                           # varargs are self-determined
        regs.append(r)
        descs.append(t.descriptor())
    # An instruction carries at most 4 inline args, so the values travel in
    # frame locals the table entry names: verbosity first, then each argument.
    # One pool per coroutine is safe -- every argument is already in a register
    # (any message() inside an argument's call has run) before the pool is written.
    slots = [ctx.local_slot(f"$msg{i}") for i in range(len(regs))]
    try:
        idx = ctx.lowerer.add_message(fmt.value, descs, slots)
    except ValueError as ex:
        raise PssSemanticError(f"message(): {ex}", loc=_loc(e))
    for r, slot in zip(regs, slots):
        ctx.emit(Op.ST_LOCAL, (r, slot))
    ctx.emit(Op.IMPORT, (BUILTIN_MESSAGE, _VOID, _const(ctx, idx)))


def _runtime_error(ctx: CoroCtx, text: str) -> None:
    idx = ctx.lowerer.intern_string(text)
    ctx.emit(Op.IMPORT, (BUILTIN_ERROR, _VOID, _const(ctx, idx)))


# --------------------------------------------------------------------------- #
# Statements
# --------------------------------------------------------------------------- #

def _lower_block(ctx: CoroCtx, stmts) -> None:
    _push_scope(ctx)
    try:
        for s in stmts or []:
            lower_stmt(ctx, s)
    finally:
        _pop_scope(ctx)


def lower_stmt(ctx: CoroCtx, stmt) -> None:
    """Lower one Layer-0 statement into ``ctx.code``."""
    sr = ctx.lowerer.prov.src_ref(stmt)
    saved, ctx.cur_src_ref = ctx.cur_src_ref, (sr or ctx.cur_src_ref)
    try:
        _lower_stmt(ctx, stmt)
    finally:
        ctx.cur_src_ref = saved


def _lower_stmt(ctx: CoroCtx, s) -> None:
    st = proc_state(ctx)

    if isinstance(s, S.StmtAnnAssign):
        t = from_datatype(s.annotation)
        if s.value is not None:
            r = _eval_for(ctx, s.value, t)          # evaluated before the name is visible
        elif t.kind == "enum" and t.items:
            r = _const(ctx, t.items[0][1])
        elif t.kind == "string":
            r = _const(ctx, ctx.lowerer.intern_string(""))
        else:
            r = _const(ctx, 0)
        name = s.target.name if isinstance(s.target, E.ExprRefLocal) else None
        if name is None:
            raise LoweringError(f"unsupported declaration target {_cn(s.target)}",
                                loc=_loc(s))
        v = _declare(ctx, name, t)
        ctx.emit(Op.ST_LOCAL, (r, ctx.local_slot(v.slot)))
        return

    if isinstance(s, S.StmtAssign):
        for tgt in s.targets:
            _assign(ctx, tgt, s.value)
        return

    if isinstance(s, S.StmtAugAssign):
        op = _AUG.get(s.op)
        if op is None:
            raise LoweringError(f"unsupported compound assignment {s.op}", loc=_loc(s))
        _assign(ctx, s.target, E.ExprBin(lhs=s.target, op=op, rhs=s.value))
        return

    if isinstance(s, S.StmtExpr):
        e = s.expr
        if _cn(e) == "ExprCast" and e.target_type is None:
            e = e.value                             # (void)f()
        if isinstance(e, E.ExprCall):
            _call(ctx, e)
        else:
            ev(ctx, e)
        return

    if isinstance(s, S.StmtReturn):
        if st.fns:
            fr = st.fns[-1]
            if s.value is not None:
                if fr.ret_type is None:
                    raise PssSemanticError("return with a value from a void function",
                                           loc=_loc(s))
                r = _eval_for(ctx, s.value, fr.ret_type)
                ctx.emit(Op.ST_LOCAL, (r, ctx.local_slot(fr.ret_slot)))
            fr.returns.append(ctx.emit(Op.BR, (0,)))
            return
        if s.value is not None:
            # Not PSS (an exec returns no value), but the hand-built scenarios
            # use it to hand the coroutine a result.
            ctx.emit(Op.RET, (ev(ctx, s.value)[0],))
            return
        if st.exec_kind != "body":
            raise LoweringError(f"return in exec {st.exec_kind} is not supported by bc "
                                f"(it would end the whole action)", loc=_loc(s))
        ctx.emit(Op.RET)                            # 20.7.5: the exec ends here
        return

    if isinstance(s, S.StmtIf):
        c, ct = ev(ctx, s.test)
        brz = ctx.emit(Op.BRZ, (truth(ctx, c, ct), 0))
        _lower_block(ctx, s.body)
        if s.orelse:
            br = ctx.emit(Op.BR, (0,))
            _patch(ctx, brz, len(ctx.code))
            _lower_block(ctx, s.orelse)
            _patch(ctx, br, len(ctx.code))
        else:
            _patch(ctx, brz, len(ctx.code))
        return

    if isinstance(s, S.StmtWhile):
        top = len(ctx.code)
        c, ct = ev(ctx, s.test)
        brz = ctx.emit(Op.BRZ, (truth(ctx, c, ct), 0))
        loop = _body_loop(ctx, s.body)
        ctx.emit(Op.BR, (top,))
        _close_loop(ctx, loop, cont=top, exit_=len(ctx.code))
        _patch(ctx, brz, len(ctx.code))
        return

    if isinstance(s, S.StmtRepeatWhile):
        top = len(ctx.code)
        loop = _body_loop(ctx, s.body)
        cont = len(ctx.code)
        c, ct = ev(ctx, s.condition)
        ez = _op(ctx, Op.CMP_EQ, truth(ctx, c, ct), _const(ctx, 0))
        ctx.emit(Op.BRZ, (ez, top))                 # loop while the condition holds
        _close_loop(ctx, loop, cont=cont, exit_=len(ctx.code))
        return

    if isinstance(s, (S.StmtFor, S.StmtRepeat)):
        _lower_repeat(ctx, s)
        return

    if isinstance(s, S.StmtBreak):
        if len(st.loops) <= (st.fns[-1].loop_base if st.fns else 0):
            raise PssSemanticError("break outside a loop", loc=_loc(s))
        st.loops[-1].breaks.append(ctx.emit(Op.BR, (0,)))
        return

    if isinstance(s, S.StmtContinue):
        if len(st.loops) <= (st.fns[-1].loop_base if st.fns else 0):
            raise PssSemanticError("continue outside a loop", loc=_loc(s))
        st.loops[-1].continues.append(ctx.emit(Op.BR, (0,)))
        return

    if isinstance(s, S.StmtMatch):
        _lower_match(ctx, s)
        return

    if isinstance(s, S.StmtPass):
        return

    raise LoweringError(f"unsupported statement {_cn(s)} in bc procedural code",
                        loc=_loc(s))


def _body_loop(ctx: CoroCtx, body) -> _Loop:
    st = proc_state(ctx)
    loop = _Loop()
    st.loops.append(loop)
    try:
        _lower_block(ctx, body)
    finally:
        st.loops.pop()
    return loop


def _close_loop(ctx: CoroCtx, loop: _Loop, cont: int, exit_: int) -> None:
    for j in loop.continues:
        _patch(ctx, j, cont)
    for j in loop.breaks:
        _patch(ctx, j, exit_)


def _lower_repeat(ctx: CoroCtx, s) -> None:
    """repeat ([i :] count): the count is evaluated once (20.7.6)."""
    count = s.iter if isinstance(s, S.StmtFor) else s.count
    target = s.target if isinstance(s, S.StmtFor) else s.iterator
    n, nt = ev(ctx, count)
    if nt.is_bool or nt.kind == "string":
        raise PssSemanticError("repeat count must be an integer", loc=_loc(s))
    p = merge(nt, I32)
    n = propagate(ctx, n, nt, p)

    _push_scope(ctx)
    try:
        name = target.name if isinstance(target, E.ExprRefLocal) else None
        idx = _declare(ctx, name or "$rep", I32)
        slot = ctx.local_slot(idx.slot)
        ctx.emit(Op.ST_LOCAL, (_const(ctx, 0), slot))
        top = len(ctx.code)
        i = ctx.new_reg()
        ctx.emit(Op.LD_LOCAL, (i, slot))
        lt = compare_regs(ctx, Op.CMP_LT, propagate(ctx, i, I32, p), n, p)
        brz = ctx.emit(Op.BRZ, (lt, 0))
        loop = _body_loop(ctx, s.body)
        cont = len(ctx.code)
        i2 = ctx.new_reg()
        ctx.emit(Op.LD_LOCAL, (i2, slot))
        ctx.emit(Op.ST_LOCAL, (normalize(ctx, _op(ctx, Op.ADD, i2, _const(ctx, 1)), I32), slot))
        ctx.emit(Op.BR, (top,))
        _close_loop(ctx, loop, cont=cont, exit_=len(ctx.code))
        _patch(ctx, brz, len(ctx.code))
    finally:
        _pop_scope(ctx)


def _pattern_pred(ctx: CoroCtx, pat, subj: int, st_: T) -> Optional[int]:
    """A 0/1 register for 'subject matches pattern'; None for the default arm."""
    cn = _cn(pat)
    if cn == "PatternAs" and pat.pattern is None:
        return None
    if cn == "PatternAs":
        return _pattern_pred(ctx, pat.pattern, subj, st_)
    if cn == "PatternOr":
        acc = None
        for p in pat.patterns:
            r = _pattern_pred(ctx, p, subj, st_)
            acc = r if acc is None else _op(ctx, Op.OR, acc, r)
        return acc
    if cn == "PatternValue":
        v = pat.value
        if _cn(v) == "ExprRange":
            lo = _match_cmp(ctx, Op.CMP_GE, subj, st_, v.lower)
            if v.upper is None:
                return _match_cmp(ctx, Op.CMP_EQ, subj, st_, v.lower)
            hi = _match_cmp(ctx, Op.CMP_LE, subj, st_, v.upper)
            return _op(ctx, Op.AND, lo, hi)
        return _match_cmp(ctx, Op.CMP_EQ, subj, st_, v)
    raise LoweringError(f"unsupported match pattern {cn}", loc=_loc(pat))


def _match_cmp(ctx: CoroCtx, op: Op, subj: int, st_: T, value) -> int:
    vt = type_of(ctx, value)
    p = merge(st_, vt)
    a = propagate(ctx, subj, st_, p)
    b, _ = ev(ctx, value, p)
    return compare_regs(ctx, op, a, b, p)


def _lower_match(ctx: CoroCtx, s) -> None:
    """20.7.10: exactly one arm matches; none -> default; neither is an error."""
    subj, st_ = ev(ctx, s.subject)
    arms, default = [], None
    for case in s.cases:
        pred = _pattern_pred(ctx, case.pattern, subj, st_)
        if pred is None:
            default = case
        else:
            arms.append((pred, case))

    # d) more than one match is an error; g) no match and no default is an error.
    count = _const(ctx, 0)
    for pred, _ in arms:
        count = _op(ctx, Op.ADD, count, pred)
    many = _op(ctx, Op.CMP_GT, count, _const(ctx, 1))
    ok = ctx.emit(Op.BRZ, (many, 0))
    _runtime_error(ctx, "match: more than one branch matches (20.7.10 d)")
    _patch(ctx, ok, len(ctx.code))
    if default is None:
        none = _op(ctx, Op.CMP_EQ, count, _const(ctx, 0))
        ok = ctx.emit(Op.BRZ, (none, 0))
        _runtime_error(ctx, "match: no branch matches and there is no default (20.7.10 g)")
        _patch(ctx, ok, len(ctx.code))

    ends = []
    for pred, case in arms:
        miss = ctx.emit(Op.BRZ, (pred, 0))
        _lower_block(ctx, case.body)
        ends.append(ctx.emit(Op.BR, (0,)))
        _patch(ctx, miss, len(ctx.code))
    if default is not None:
        _lower_block(ctx, default.body)
    for j in ends:
        _patch(ctx, j, len(ctx.code))


def lower_exec_block(ctx: CoroCtx, exec_block) -> None:
    """Lower a ScExecBlock's Layer-0 statements."""
    sr = ctx.lowerer.prov.src_ref(exec_block)
    saved, ctx.cur_src_ref = ctx.cur_src_ref, (sr or ctx.cur_src_ref)
    st = proc_state(ctx)
    kind = getattr(exec_block, "kind", "body")
    saved_kind, st.exec_kind = st.exec_kind, kind
    st.scopes.append(st.exec_scopes.setdefault(kind, {}))
    try:
        for s in exec_block.stmts or []:
            lower_stmt(ctx, s)
    finally:
        st.scopes.pop()
        st.exec_kind = saved_kind
        ctx.cur_src_ref = saved
