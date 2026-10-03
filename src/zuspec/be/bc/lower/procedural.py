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
to the end of the inlined body. A call to a function already being inlined
(recursion) is a ``CALL`` instead, of the function's *called form*: a
coroutine of its own, lowered once per (function, instance), whose parameters
come from ``LD_ARG`` and whose ``return`` is ``RET`` (bc procedural gaps
B-D5). So a model with no recursion keeps its bytecode.

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

from ..model import (Op, INSTR_F_HAS_RET, INSTR_F_BLOCKING, INSTR_F_SPIN,
                     BUILTIN_MESSAGE, BUILTIN_ERROR, BUILTIN_READ, BUILTIN_WRITE,
                     CALL_MAX_ARGS, CoroDescriptor)
from .context import CoroCtx
from .errors import LoweringError, PssSemanticError
from .types import (BOOL, CHANDLE, I32, STRING, U64, T, StructT, from_datatype,
                    literal_type, merge, value_type)

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
    #: a struct: where each leaf lives (see _Place); None for a scalar
    locs: Optional[list] = None
    #: a foreach index over a component array, in one unrolled copy: its value
    const: Optional[int] = None
    #: a foreach iterator over a component array: the element it names
    comp: Optional["CompCtx"] = None


@dc.dataclass
class _Place:
    """Where a struct value lives: one location per leaf of ``type``, each
    ``("local", slot-name)`` or ``("field", object-slot)``. A struct is never
    in a register; it is moved leaf by leaf."""
    type: StructT
    locs: list


@dc.dataclass
class _Loop:
    breaks: List[int] = dc.field(default_factory=list)
    continues: List[int] = dc.field(default_factory=list)


@dc.dataclass(frozen=True)
class CompCtx:
    """A component instance, relative to the frame's (P1.5): its type, its
    first slot from the frame's instance base, and its instance number from
    the frame's instance. All three are static: a component type's subtree
    has one layout (ir-core ``comp_tree``)."""
    type_qname: str
    slot: int = 0
    inst: int = 0


@dc.dataclass
class _Fn:
    """An inlined call in progress."""
    fn: object
    scope_base: int                 # its locals start at this scope depth
    loop_base: int
    ret_type: Optional[T]
    ret_slot: Optional[str]
    returns: List[int] = dc.field(default_factory=list)
    #: a struct return value's storage
    ret_place: Optional[_Place] = None
    #: a component function: the instance it runs in (its ``self``)
    comp: Optional[CompCtx] = None
    #: the body of a called form (B-D5): ``return`` is RET
    called: bool = False


@dc.dataclass
class ProcState:
    fields: Dict[str, Tuple[int, T]] = dc.field(default_factory=dict)
    by_slot: Dict[int, T] = dc.field(default_factory=dict)
    #: a struct attribute (or a struct field of one), by dotted path: its
    #: leaves' object slots. None when a leaf has a type bc cannot hold.
    struct_fields: Dict[str, Optional[_Place]] = dc.field(default_factory=dict)
    component: Optional[str] = None
    #: action code: the instance ``comp`` names (P1.5)
    action_comp: Optional[CompCtx] = None
    #: component code (construction): the instance ``self`` is
    comp_self: Optional[CompCtx] = None
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
    _index_struct_fields(st, getattr(coro, "fields", []) or [])
    _index_handles(st, getattr(coro, "fields", []) or [],
                   getattr(coro, "subtree", None) or [])
    at = getattr(coro, "action_type", None)
    if at and "::" in at:
        st.component = at.rsplit("::", 1)[0]
        comps = getattr(ctx.lowerer, "comps", None)
        if comps is not None and st.component in comps.types:
            st.action_comp = CompCtx(st.component)


def _index_struct_fields(st: ProcState, fields) -> None:
    """Every struct-valued path in a flattened layout (``s``, ``s.csr``), with
    its leaves, so ``self.s`` is a value and ``self.s.csr.eol`` a slot."""
    groups: Dict[str, list] = {}
    for f in sorted(fields, key=lambda f: f.slot):
        path = f.name.split(".")
        for k in range(1, len(path)):
            groups.setdefault(".".join(path[:k]), []).append(
                (tuple(path[k:]), f.slot, st.fields[f.name][1]))
    for name, leaves in groups.items():
        if any(t is None for _, _, t in leaves):
            st.struct_fields[name] = None
            continue
        st.struct_fields[name] = _Place(
            StructT(tuple((p, t) for p, _, t in leaves)),
            [("field", slot) for _, slot, _ in leaves])


def _index_handles(st: ProcState, own, subtree) -> None:
    """A sub-action's attributes, through its handle (``self.b1.x``,
    ``self.bs[1].s.f``): the subtree layout names each of its slots by path,
    relative to this action's base (P1-D1). A handle is a place like a struct
    attribute, holding only the leaves bc can type (a sub-action's own handle
    slots hold nothing)."""
    own_names = {f.name for f in own}
    groups: Dict[str, list] = {}
    for f in sorted(subtree, key=lambda f: f.slot):
        if f.name in own_names:
            continue
        try:
            t = from_datatype(f.datatype) if f.datatype is not None else U64
        except LoweringError:
            continue
        path = f.name.split(".")
        for k in range(1, len(path)):
            groups.setdefault(".".join(path[:k]), []).append(
                (tuple(path[k:]), f.slot, t))
    for name, leaves in groups.items():
        if name in st.struct_fields:
            continue
        st.struct_fields[name] = _Place(
            StructT(tuple((p, t) for p, _, t in leaves)),
            [("field", slot) for _, slot, _ in leaves])


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


def _declare(ctx: CoroCtx, name: str, t: T, locs: Optional[list] = None) -> _Var:
    """Declare *name* in the innermost scope. A struct gets a frame local per
    leaf, unless *locs* says where it already lives (a by-handle parameter)."""
    st = proc_state(ctx)
    if not st.scopes:
        st.scopes.append({})
    st.uniq += 1
    v = _Var(slot=f"{name}${st.uniq}", type=t)
    if isinstance(t, StructT):
        v.locs = locs if locs is not None else _struct_locals(ctx, v.slot, t).locs
    else:
        ctx.local_slot(v.slot)
    st.scopes[-1][name] = v
    return v


def _struct_locals(ctx: CoroCtx, base: str, t: StructT) -> _Place:
    locs = []
    for path, _ in t.leaves:
        slot = f"{base}.{'.'.join(path)}"
        ctx.local_slot(slot)
        locs.append(("local", slot))
    return _Place(t, locs)


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
        return literal_type(e.value, e.width, e.signed)
    if isinstance(e, E.ExprRefLocal):
        v = _lookup(ctx, e.name)
        return v.type if v is not None else U64
    if isinstance(e, E.ExprRefField):
        return proc_state(ctx).by_slot.get(e.index, U64)
    if isinstance(e, (E.ExprAttribute, E.ExprRefUnresolved, E.ExprSubscript, E.ExprCall)):
        d = _dyn_index(ctx, e)
        if d is not None:
            return type_of(ctx, _subst(e, d[0], 0))    # every element has one type
    if isinstance(e, (E.ExprAttribute, E.ExprRefUnresolved, E.ExprSubscript)):
        p = _place(ctx, e)
        if p is not None:
            return p.type
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
            t = literal_type(v, e.width, e.signed)
            return _const(ctx, _fold_propagate(int(v), t, want)), want
        t = literal_type(v, e.width, e.signed)
        return _const(ctx, int(v)), t

    if isinstance(e, E.ExprRefLocal):
        v = _lookup(ctx, e.name)
        if v is None:
            if proc_state(ctx).fns:
                raise LoweringError(f"unresolved local {e.name!r}", loc=_loc(e))
            r = ctx.new_reg()
            ctx.emit(Op.LD_LOCAL, (r, ctx.local_slot(e.name)))   # legacy untyped local
            return r, U64
        if v.comp is not None:
            raise PssSemanticError(f"{e.name!r} is a component instance, not a "
                                   f"value", loc=_loc(e))
        if v.locs is not None:
            _not_scalar(v.type, e)
        r = ctx.new_reg()
        ctx.emit(Op.LD_LOCAL, (r, ctx.local_slot(v.slot)))
        return r, v.type

    if isinstance(e, E.ExprRefField):
        r = ctx.new_reg()
        ctx.emit(Op.LD_FIELD, (r, e.index))
        return r, proc_state(ctx).by_slot.get(e.index, U64)

    if isinstance(e, (E.ExprAttribute, E.ExprRefUnresolved, E.ExprSubscript)):
        d = _dyn_index(ctx, e)
        if d is not None:
            out, t = ctx.new_reg(), type_of(ctx, e)
            _dispatch(ctx, d, lambda k: ctx.emit(
                Op.MOV, (out, ev(ctx, _subst(e, d[0], k))[0])))
            return out, t
        kind, t, where = _resolve_name(ctx, e)
        _not_opaque(t, e)
        return _load_loc(ctx, (kind, where)), t

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
        if isinstance(t, StructT):
            _not_scalar(t, e)
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
    if isinstance(lt, StructT) or isinstance(rt, StructT):
        return _struct_compare(ctx, op, lhs, rhs, lt, rt)
    if lt.is_bool and rt.is_bool:
        a, _ = ev(ctx, lhs)
        b, _ = ev(ctx, rhs)
        return _op(ctx, op, a, b)
    if lt.kind == "string" or rt.kind == "string":
        return _string_compare(ctx, op, lhs, rhs, lt, rt)
    p = merge(lt, rt)
    a, _ = ev(ctx, lhs, p)
    b, _ = ev(ctx, rhs, p)
    return compare_regs(ctx, op, a, b, p)


def _string_compare(ctx: CoroCtx, op: Op, lhs, rhs, lt: T, rt: T) -> int:
    """7.6: strings compare with == and != only. A string is its index in the
    interned table, and bc makes no string at run time, so equal strings have
    equal indices. An operation that creates a string must bring a real
    comparison with it."""
    if lt.kind != rt.kind:
        raise PssSemanticError("a string compares only with a string", loc=_loc(lhs))
    if op not in (Op.CMP_EQ, Op.CMP_NE):
        raise PssSemanticError("strings compare only with == and !=", loc=_loc(lhs))
    a, _ = ev(ctx, lhs)
    b, _ = ev(ctx, rhs)
    return _op(ctx, op, a, b)


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

def _not_opaque(t, e):
    if getattr(t, "kind", None) == "opaque":
        raise LoweringError(f"a field typed by {t.name or 'an unresolved type'!r} "
                            f"(a struct template parameter: pssc does not "
                            f"specialize struct templates yet) is not supported "
                            f"by bc", loc=_loc(e))


def _not_scalar(t, e):
    raise LoweringError(f"struct {t.name or ''} value used where a scalar is "
                        f"needed", loc=_loc(e))


def _self_comp(ctx: CoroCtx) -> Optional[CompCtx]:
    """The component instance ``self`` is in the code being lowered: an
    inlined component function's, or a construction block's; None in action
    code and package functions."""
    st = proc_state(ctx)
    return st.fns[-1].comp if st.fns else st.comp_self


def _comp_layout(ctx: CoroCtx, c: CompCtx):
    return ctx.lowerer.comps.get(c.type_qname)


def _comp_inst(ctx: CoroCtx, e) -> Optional[CompCtx]:
    """The component instance *e* names, or None if it names none:
    ``self`` in component code, ``comp`` in action code (9.1.5), and
    sub-instance paths below either (``comp.a.sub``, ``ch[1]``)."""
    if getattr(ctx.lowerer, "comps", None) is None:
        return None
    st = proc_state(ctx)
    if isinstance(e, E.TypeExprRefSelf):
        return _self_comp(ctx)
    if isinstance(e, (E.ExprRefLocal, E.ExprRefUnresolved)):
        v = _lookup(ctx, e.name)
        return v.comp if v is not None else None
    if isinstance(e, E.ExprAttribute):
        if isinstance(e.value, E.TypeExprRefSelf):
            if _lookup(ctx, e.attr) is not None:
                return None                     # a parameter or local
            if e.attr == "comp" and not st.fns and st.comp_self is None:
                return st.action_comp
        base = _comp_inst(ctx, e.value)
        if base is None:
            return None
        return _comp_sub(ctx, base, e.attr)
    if isinstance(e, E.ExprSubscript):
        base = _comp_inst(ctx, e.value.value) if isinstance(
            e.value, E.ExprAttribute) else None
        if base is None or e.value.attr not in _comp_layout(ctx, base).arrays:
            return None
        k = _const_index(ctx, e.slice)
        if k is None:
            # Reached only where no dispatch is made (a struct-valued place).
            raise LoweringError("an element of a component array with a computed "
                                "index is not supported by bc here", loc=_loc(e))
        n = _comp_layout(ctx, base).arrays[e.value.attr]
        if not 0 <= k < n:
            raise PssSemanticError(f"index {k} is out of bounds of "
                                   f"component array {e.value.attr!r} [{n}]",
                                   loc=_loc(e))
        return _comp_sub(ctx, base, "%s[%d]" % (e.value.attr, k))
    return None


def _const_index(ctx: CoroCtx, e) -> Optional[int]:
    """The value of index *e* if it is known at lowering: a literal, or the
    index of an unrolled ``foreach`` over a component array."""
    if isinstance(e, E.ExprConstant) and isinstance(e.value, int) \
            and not isinstance(e.value, bool):
        return e.value
    if isinstance(e, (E.ExprRefLocal, E.ExprRefUnresolved)):
        v = _lookup(ctx, e.name)
        if v is not None:
            return v.const
    return None


def _comp_array(ctx: CoroCtx, e) -> Optional[Tuple[CompCtx, str, int]]:
    """``(instance, name, count)`` if *e* names a component array."""
    if not isinstance(e, E.ExprAttribute) or getattr(ctx.lowerer, "comps", None) is None:
        return None
    if isinstance(e.value, E.TypeExprRefSelf) and _lookup(ctx, e.attr) is not None:
        return None
    base = _comp_inst(ctx, e.value)
    if base is None:
        return None
    n = _comp_layout(ctx, base).arrays.get(e.attr)
    return None if n is None else (base, e.attr, n)


# A component array's element with an index known only at run time (B-D2):
# the expression is lowered once per element, with the index a constant, and
# the copy whose index matches runs. An element's attributes are static slots
# of its instance and a call to its function is inlined in it, so neither can
# take its instance from a register.

def _dyn_index(ctx: CoroCtx, e):
    """``(subscript, name, count)`` for the subscript nearest the root of
    *e*'s path that indexes a component array with a run-time index."""
    spine, x = [], e
    while True:
        spine.append(x)
        if isinstance(x, E.ExprCall):
            x = x.func
        elif isinstance(x, (E.ExprAttribute, E.ExprSubscript)):
            x = x.value
        else:
            break
    for x in reversed(spine):
        if isinstance(x, E.ExprSubscript) and _const_index(ctx, x.slice) is None:
            arr = _comp_array(ctx, x.value)
            if arr is not None:
                return x, arr[1], arr[2]
    return None


def _subst(e, node, k: int):
    """*e* with subscript *node*'s index replaced by the constant *k*."""
    if e is node:
        return dc.replace(e, slice=E.ExprConstant(value=k))
    if isinstance(e, E.ExprCall):
        return dc.replace(e, func=_subst(e.func, node, k))
    if isinstance(e, (E.ExprAttribute, E.ExprSubscript)):
        return dc.replace(e, value=_subst(e.value, node, k))
    return e


def _dispatch(ctx: CoroCtx, d, emit_k) -> None:
    """Evaluate the index once, then run ``emit_k(k)`` for the element it
    selects; an index out of bounds is a run-time error (20.7.4)."""
    node, name, n = d
    j, jt = ev(ctx, node.slice)
    if jt.is_bool or jt.kind in ("string", "enum"):
        raise PssSemanticError("a component array index must be an integer",
                               loc=_loc(node))
    ends = []
    for k in range(n):
        brz = ctx.emit(Op.BRZ, (_op(ctx, Op.CMP_EQ, j, _const(ctx, k)), 0))
        emit_k(k)
        ends.append(ctx.emit(Op.BR, (0,)))
        _patch(ctx, brz, len(ctx.code))
    _runtime_error(ctx, f"index out of bounds of component array {name!r} [{n}]")
    for at in ends:
        _patch(ctx, at, len(ctx.code))


def _comp_sub(ctx: CoroCtx, base: CompCtx, key: str) -> Optional[CompCtx]:
    sub = _comp_layout(ctx, base).subs.get(key)
    if sub is None:
        return None
    return CompCtx(sub.type_qname, base.slot + sub.slot, base.inst + sub.inst)


def _comp_data(ctx: CoroCtx, e):
    """A data attribute of a component instance *e* names: ``("leaf",
    type, slot)`` or ``("place", _Place)``; None if *e* names none."""
    if not isinstance(e, (E.ExprAttribute, E.ExprRefUnresolved)):
        return None
    if isinstance(e, E.ExprRefUnresolved):
        base, name = _self_comp(ctx), e.name
        if base is None or _lookup(ctx, name) is not None:
            return None
    else:
        if isinstance(e.value, E.TypeExprRefSelf) and _lookup(ctx, e.attr) is not None:
            return None
        base, name = _comp_inst(ctx, e.value), e.attr
    if base is None:
        return None
    lay = _comp_layout(ctx, base)
    if name in lay.subs or name in lay.arrays:
        return None
    # A data attribute: one slot, or a struct's leaves (``s.f``).
    leaves = [(path, base.slot + i, leaf) for i, (path, leaf) in enumerate(lay.slots)
              if path == name or path.startswith(name + ".")]
    if not leaves:
        return None
    try:
        typed = [(p, slot, from_datatype(leaf.datatype)) for p, slot, leaf in leaves]
    except LoweringError:
        raise LoweringError(f"component attribute {name!r} has a type bc does "
                            f"not support", loc=_loc(e))
    if len(typed) == 1 and typed[0][0] == name:
        return ("leaf", typed[0][2], typed[0][1])
    return ("place", _Place(
        StructT(tuple((tuple(p[len(name) + 1:].split(".")), t) for p, _, t in typed)),
        [("comp", slot) for _, slot, _ in typed]))


def _place(ctx: CoroCtx, e) -> Optional[_Place]:
    """Where struct-valued *e* lives, or None if *e* is not a struct place.

    ``self.s`` / ``s`` look in scope first (a local or parameter), then at the
    action's attributes; ``x.f`` is field ``f`` of struct place ``x``.
    """
    cd = _comp_data(ctx, e)
    if cd is not None:
        return cd[1] if cd[0] == "place" else None
    if isinstance(e, E.ExprRefLocal):
        v = _lookup(ctx, e.name)
        return _Place(v.type, v.locs) if v is not None and v.locs is not None else None
    if isinstance(e, E.ExprRefUnresolved) or (
            isinstance(e, E.ExprAttribute) and isinstance(e.value, E.TypeExprRefSelf)):
        name = e.name if isinstance(e, E.ExprRefUnresolved) else e.attr
        v = _lookup(ctx, name)
        if v is not None:
            return _Place(v.type, v.locs) if v.locs is not None else None
        st = proc_state(ctx)
        if st.fns or name not in st.struct_fields:
            return None
        p = st.struct_fields[name]
        if p is None:
            raise LoweringError(f"attribute {name!r} holds a field of a type bc "
                                f"does not support", loc=_loc(e))
        return p
    if isinstance(e, E.ExprAttribute):
        base = _place(ctx, e.value)
        if base is None:
            return None
        t, i = _member(base, e)
        if isinstance(t, StructT):
            return _Place(t, base.locs[i:i + len(t.leaves)])
    if (isinstance(e, E.ExprSubscript) and isinstance(e.value, E.ExprAttribute)
            and isinstance(e.value.value, E.TypeExprRefSelf)
            and isinstance(e.slice, E.ExprConstant) and isinstance(e.slice.value, int)):
        # An element of a handle array: `self.bs[1]`.
        st = proc_state(ctx)
        name = "%s[%d]" % (e.value.attr, e.slice.value)
        if not st.fns and name in st.struct_fields:
            return st.struct_fields[name]
    return None


def _member(base: _Place, e):
    sub = base.type.sub(e.attr)
    if sub is None:
        raise LoweringError(f"struct {base.type.name or ''} has no field "
                            f"{e.attr!r}", loc=_loc(e))
    return sub


def _resolve_name(ctx: CoroCtx, e) -> Tuple[str, T, object]:
    """``self.x`` / ``x`` / ``s.f`` -> ("local", type, slot-name),
    ("field", type, slot) or ("comp", type, slot)."""
    cd = _comp_data(ctx, e)
    if cd is not None:
        if cd[0] == "place":
            _not_scalar(cd[1].type, e)
        return "comp", cd[1], cd[2]
    if isinstance(e, E.ExprSubscript):
        raise LoweringError("an array element is not supported by bc here",
                            loc=_loc(e))
    if isinstance(e, E.ExprAttribute) and not isinstance(e.value, E.TypeExprRefSelf):
        base = _place(ctx, e.value)
        if base is None:
            raise LoweringError(f"reference through {_cn(e.value)} "
                                f"(.{e.attr}) is not supported by bc", loc=_loc(e))
        t, i = _member(base, e)
        if isinstance(t, StructT):
            _not_scalar(t, e)
        kind, where = base.locs[i]
        return kind, t, where
    name = e.attr if isinstance(e, E.ExprAttribute) else e.name
    v = _lookup(ctx, name)
    if v is not None:
        if v.locs is not None:
            _not_scalar(v.type, e)
        return "local", v.type, v.slot
    st = proc_state(ctx)
    if st.fns or st.comp_self is not None:
        # In a function only its parameters and locals are in scope, and its
        # component's attributes (resolved above).
        raise LoweringError(f"unresolved reference {name!r}: not a parameter, a "
                            f"local or an attribute of the component", loc=_loc(e))
    if name in st.fields:
        slot, t = st.fields[name]
        if t is None:
            raise LoweringError(f"attribute {name!r} has a type bc does not support",
                                loc=_loc(e))
        return "field", t, slot
    if name in st.struct_fields:
        raise LoweringError(f"struct attribute {name!r} used where a scalar is "
                            f"needed", loc=_loc(e))
    raise LoweringError(f"unresolved reference {name!r}", loc=_loc(e))


# --------------------------------------------------------------------------- #
# Struct values: moved leaf by leaf
# --------------------------------------------------------------------------- #

_LOAD = {"field": Op.LD_FIELD, "comp": Op.LD_COMP}
_STORE = {"field": Op.ST_FIELD, "comp": Op.ST_COMP}


def _load_loc(ctx: CoroCtx, loc) -> int:
    kind, where = loc
    r = ctx.new_reg()
    if kind == "local":
        ctx.emit(Op.LD_LOCAL, (r, ctx.local_slot(where)))
    else:
        ctx.emit(_LOAD[kind], (r, where))
    return r


def _store_loc(ctx: CoroCtx, loc, r: int) -> None:
    kind, where = loc
    if kind == "local":
        ctx.emit(Op.ST_LOCAL, (r, ctx.local_slot(where)))
    else:
        ctx.emit(_STORE[kind], (r, where))


def _struct_value(ctx: CoroCtx, e, want: Optional[StructT] = None) -> _Place:
    """The place holding struct-valued *e*: a struct variable or attribute
    (or a field of one), or a call's return value."""
    p = _place(ctx, e)
    if p is None and isinstance(e, E.ExprCall):
        r, t = _call(ctx, e)
        if isinstance(t, StructT):
            p = r
    if p is None:
        raise LoweringError(f"{_cn(e)} is not a struct value bc can move "
                            f"(a struct variable, attribute or call)", loc=_loc(e))
    if want is not None and p.type.leaves != want.leaves:
        raise PssSemanticError(f"struct {p.type.name or ''} is not assignment-"
                               f"compatible with struct {want.name or ''}", loc=_loc(e))
    return p


def _copy(ctx: CoroCtx, dst: _Place, src: _Place) -> None:
    """8.5.3: a struct assignment copies every field. Every leaf is loaded
    before any is stored, so overlapping places copy correctly."""
    if dst.locs == src.locs:
        return
    regs = [_load_loc(ctx, loc) for loc in src.locs]
    for loc, r in zip(dst.locs, regs):
        _store_loc(ctx, loc, r)


def _struct_compare(ctx: CoroCtx, op: Op, lhs, rhs, lt, rt) -> int:
    """7.8: aggregate ``==`` / ``!=`` compare field by field."""
    if op not in (Op.CMP_EQ, Op.CMP_NE):
        raise PssSemanticError("structs compare only with == and !=", loc=_loc(lhs))
    if not (isinstance(lt, StructT) and isinstance(rt, StructT)):
        raise PssSemanticError("a struct compares only with a struct", loc=_loc(lhs))
    a = _struct_value(ctx, lhs)
    b = _struct_value(ctx, rhs, a.type)
    acc = _const(ctx, 1)
    for (_, t), la, lb in zip(a.type.leaves, a.locs, b.locs):
        eq = compare_regs(ctx, Op.CMP_EQ, _load_loc(ctx, la), _load_loc(ctx, lb), t)
        acc = _op(ctx, Op.AND, acc, eq)
    if op == Op.CMP_NE:
        acc = _op(ctx, Op.CMP_EQ, acc, _const(ctx, 0))
    return acc


def _default_value(ctx: CoroCtx, t: T, init=None) -> int:
    """A register holding a declaration's initial value: *init* (a constant)
    or the type's default."""
    if init is not None:
        if not isinstance(init, E.ExprConstant):
            raise LoweringError(f"a struct field initializer that is not a "
                                f"constant is not supported by bc", loc=_loc(init))
        return _eval_for(ctx, init, t)
    if t.kind == "enum" and t.items:
        return _const(ctx, t.items[0][1])
    if t.kind == "string":
        return _const(ctx, ctx.lowerer.intern_string(""))
    return _const(ctx, 0)


def _init_struct(ctx: CoroCtx, place: _Place, annotation) -> None:
    """A struct declared without a value takes its fields' initial values."""
    from zuspec.ir.core.xf.pss_lower import layout
    leaves = layout.value_leaves(annotation, ctx.lowerer.types)
    for (_, t), leaf, loc in zip(place.type.leaves, leaves, place.locs):
        _store_loc(ctx, loc, _default_value(
            ctx, t, getattr(leaf.field, "initial_value", None)))


def _store(ctx: CoroCtx, target, r: int, src: T) -> None:
    if isinstance(target, E.ExprRefLocal):
        v = _lookup(ctx, target.name)
        if v is None:
            if proc_state(ctx).fns:
                raise LoweringError(f"unresolved local {target.name!r}", loc=_loc(target))
            ctx.emit(Op.ST_LOCAL, (r, ctx.local_slot(target.name)))   # legacy
            return
        if v.const is not None or v.comp is not None:
            raise PssSemanticError(f"foreach variable {target.name!r} is read-only",
                                   loc=_loc(target))
        ctx.emit(Op.ST_LOCAL, (assign_convert(ctx, r, src, v.type), ctx.local_slot(v.slot)))
        return
    if isinstance(target, E.ExprRefField):
        t = proc_state(ctx).by_slot.get(target.index, U64)
        ctx.emit(Op.ST_FIELD, (assign_convert(ctx, r, src, t), target.index))
        return
    if isinstance(target, (E.ExprAttribute, E.ExprRefUnresolved, E.ExprSubscript)):
        d = _dyn_index(ctx, target)
        if d is not None:
            _dispatch(ctx, d, lambda k: _store(ctx, _subst(target, d[0], k), r, src))
            return
        kind, t, where = _resolve_name(ctx, target)
        _not_opaque(t, target)
        _store_loc(ctx, (kind, where), assign_convert(ctx, r, src, t))
        return
    raise LoweringError(f"unsupported assignment target {_cn(target)}", loc=_loc(target))


def _target_type(ctx: CoroCtx, target) -> T:
    if isinstance(target, E.ExprRefLocal):
        v = _lookup(ctx, target.name)
        return v.type if v is not None else U64       # legacy untyped local
    return type_of(ctx, target)


def _assign(ctx: CoroCtx, target, value) -> None:
    tt = _target_type(ctx, target)
    if isinstance(tt, StructT):
        dst = _place(ctx, target)
        _copy(ctx, dst, _struct_value(ctx, value, tt))
        return
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

    ``self.f`` -> ("f", "self"): a function of the running component, or an
    import. ``x.f`` -> ("f", "path"): a function of the component instance
    ``x`` names (``comp.f``, ``comp.a.f``, ``ch[1].f``; :func:`_callee`
    resolves it). A bare ``f`` or ``pkg::f`` (``ExprRefUnresolved``) ->
    (name, "pkg"): pssc's front end gives that form to a call its linker
    resolved to a package-scope function, by qualified name. The scope is
    the linker's answer; looking a bare name up in the component first would
    let a component's ``twice`` shadow the package's inside a package
    function.
    """
    if isinstance(func, E.ExprAttribute):
        if isinstance(func.value, E.TypeExprRefSelf):
            return func.attr, "self"
        return func.attr, "path"
    if isinstance(func, E.ExprRefUnresolved):
        return func.name, "pkg"
    return None, None


def _is_message(name: Optional[str]) -> bool:
    return name in ("message", "std_pkg::message")


def _callee(ctx: CoroCtx, func) -> Tuple[Optional[str], Optional[str], Optional[CompCtx]]:
    """``(name, scope, instance)`` of a call target: the component instance a
    component function runs in (its ``self``), None for a package function
    or an import."""
    name, scope = _call_name(func)
    if scope == "self":
        if proc_state(ctx).fns or proc_state(ctx).comp_self is not None:
            return name, scope, _self_comp(ctx)
        return name, scope, proc_state(ctx).action_comp
    if scope == "path":
        inst = _comp_inst(ctx, func.value)
        if inst is not None:
            return name, scope, inst
        v = func.value
        if (isinstance(v, E.ExprAttribute) and isinstance(v.value, E.TypeExprRefSelf)
                and v.attr == "comp" and getattr(ctx.lowerer, "comps", None) is None):
            # No component tree (a hand-built module): the action's component.
            return name, "comp", None
        return None, None, None
    return name, scope, None


def _lookup_function(ctx: CoroCtx, name: str, scope: Optional[str],
                     comp: Optional[CompCtx] = None):
    fns = ctx.lowerer.functions
    if scope == "pkg":
        return fns.get(name)
    comps = getattr(ctx.lowerer, "comps", None)
    if comp is not None and comps is not None:
        return comps.function(comp.type_qname, name, fns)
    if comp is None and scope in ("self", "comp"):
        # No instance (no component tree): the action's component.
        st = proc_state(ctx)
        return fns.get(f"{st.component}::{name}") if st.component else None
    return None


def _no_callee(e):
    f = e.func
    what = f".{f.attr}()" if isinstance(f, E.ExprAttribute) else _cn(f)
    raise LoweringError(f"call {what}: its target is not a function of a component "
                        f"instance bc can resolve (a channel or other library "
                        f"component's built-in, or a computed path)", loc=_loc(e))


def _import_first(ctx: CoroCtx, name: str, scope: Optional[str]) -> bool:
    """``self.f()`` in action code is an import when there is one; in a
    component's code, its own function ``f`` comes first."""
    st = proc_state(ctx)
    return (scope == "self" and not st.fns and st.comp_self is None
            and name in ctx.lowerer.imports)


def _no_string_import(name: str, decl: dict, e) -> None:
    """An import's string parameter or result would cross the platform
    boundary as a table index, which means nothing outside the model."""
    at = decl.get("string_at")
    if at:
        what = "returns a string" if "return" in at else "takes a string"
        raise LoweringError(f"import {name!r} {what}: bc passes only scalars "
                            f"to the platform", loc=_loc(e))


def _call_type(ctx: CoroCtx, e) -> T:
    ch = _channel(ctx, e.func)
    if ch is not None:
        t = _CHAN_RET[e.func.attr]
        if t is None:
            raise PssSemanticError("put() returns no value", loc=_loc(e))
        return ch.elem if t == "elem" else t
    name, scope, comp = _callee(ctx, e.func)
    if name is None:
        _no_callee(e)
    if _is_message(name):
        raise LoweringError("a void call has no value", loc=_loc(e))
    fn = None if _import_first(ctx, name, scope) else _lookup_function(ctx, name, scope, comp)
    decl = ctx.lowerer.imports.get(name) if scope == "self" and fn is None else None
    if fn is None and decl is None:
        lt = _library_type(ctx, name, scope, comp)
        if lt is not None:
            return lt
    if decl is not None:
        _no_string_import(name, decl, e)
        rt = decl.get("ret_type")
        if rt is None:
            raise PssSemanticError(f"void import {name!r} has no value", loc=_loc(e))
        return T(min(int(rt[0] or 32), 64), bool(rt[1]))
    if fn is None:
        raise LoweringError(f"call to unknown function {name!r}", loc=_loc(e))
    if fn.returns is None:
        raise PssSemanticError(f"void function {name!r} has no value", loc=_loc(e))
    return value_type(fn.returns, ctx.lowerer.types)


def _call(ctx: CoroCtx, e) -> Tuple[Optional[int], Optional[T]]:
    d = _dyn_index(ctx, e)
    if d is not None:
        return _call_dispatch(ctx, e, d)
    ch = _channel(ctx, e.func)
    if ch is not None:
        return _channel_call(ctx, ch, e)
    name, scope, comp = _callee(ctx, e.func)
    if name is None:
        _no_callee(e)
    if _is_message(name):
        _message(ctx, e)
        return None, None
    fn = None if _import_first(ctx, name, scope) else _lookup_function(ctx, name, scope, comp)
    if fn is None and scope == "self" and name in ctx.lowerer.imports:
        return _import_call(ctx, name, e)
    if fn is None:
        lib = _library_call(ctx, name, scope, comp, e)
        if lib is not None:
            return lib
        raise LoweringError(f"call to {name!r}: not a known function or import",
                            loc=_loc(e))
    return _inline(ctx, fn, e, comp if scope != "pkg" else None)


# --------------------------------------------------------------------------- #
# Channels (21.9.1; bc procedural gaps B-D3, B-D4)
# --------------------------------------------------------------------------- #

#: the result type of each channel_c function ("elem": the element type)
_CHAN_RET = {"try_put": BOOL, "try_get": BOOL, "get": "elem", "put": None}


@dc.dataclass
class _Chan:
    """A channel's slots in the component object (ir-core comp_tree):
    ``count``, ``head`` and the ring ``bufs``, all static."""
    name: str
    count: int
    head: int
    bufs: List[int]
    elem: T


def _channel(ctx: CoroCtx, func) -> Optional[_Chan]:
    """The channel *func* (``c.try_put``) is a function of, or None."""
    if not (isinstance(func, E.ExprAttribute) and func.attr in _CHAN_RET):
        return None
    if getattr(ctx.lowerer, "comps", None) is None:
        return None
    x = func.value
    if isinstance(x, E.ExprRefUnresolved):
        base, name = _self_comp(ctx), x.name
    elif isinstance(x, E.ExprAttribute):
        if isinstance(x.value, E.TypeExprRefSelf) and _lookup(ctx, x.attr) is not None:
            return None
        base, name = _comp_inst(ctx, x.value), x.attr
    else:
        return None
    if base is None:
        return None
    lay = _comp_layout(ctx, base)
    count = lay.slot_of(name + ".$count")
    if count is None:
        return None
    bufs = [base.slot + i for i, (p, _) in enumerate(lay.slots)
            if p.startswith(name + ".$buf")]
    elem = from_datatype(lay.slots[bufs[0] - base.slot][1].datatype)
    return _Chan(name, base.slot + count, base.slot + lay.slot_of(name + ".$head"),
                 bufs, elem)


def _ring(ctx: CoroCtx, ch: _Chan, idx: int, emit_k) -> None:
    """``emit_k(k)`` for the buffer slot *idx* (a register, in range) names."""
    if len(ch.bufs) == 1:
        emit_k(0)
        return
    ends = []
    for k in range(len(ch.bufs) - 1):
        brz = ctx.emit(Op.BRZ, (_op(ctx, Op.CMP_EQ, idx, _const(ctx, k)), 0))
        emit_k(k)
        ends.append(ctx.emit(Op.BR, (0,)))
        _patch(ctx, brz, len(ctx.code))
    emit_k(len(ch.bufs) - 1)
    for at in ends:
        _patch(ctx, at, len(ctx.code))


def _chan_wait(ctx: CoroCtx, ch: _Chan, full: bool) -> None:
    """Block until the channel is not full (``put``) or not empty (``get``):
    spin on a SPIN yield (INSTR_F_SPIN), so a deadlock is reported. Temporary:
    an event wait replaces it."""
    top = len(ctx.code)
    n = _load_loc(ctx, ("comp", ch.count))
    ready = _op(ctx, Op.CMP_NE, n, _const(ctx, len(ch.bufs) if full else 0))
    brz = ctx.emit(Op.BRZ, (ready, 0))
    br = ctx.emit(Op.BR, (0,))
    _patch(ctx, brz, len(ctx.code))
    ctx.emit(Op.YIELD, (), flags=INSTR_F_SPIN)
    ctx.emit(Op.BR, (top,))
    _patch(ctx, br, len(ctx.code))


def _chan_put(ctx: CoroCtx, ch: _Chan, v: int) -> None:
    """Append *v* at ``(head + count) % depth``; the caller checked room."""
    n = _load_loc(ctx, ("comp", ch.count))
    h = _load_loc(ctx, ("comp", ch.head))
    at = _op(ctx, Op.MOD, _op(ctx, Op.ADD, h, n), _const(ctx, len(ch.bufs)))
    _ring(ctx, ch, at, lambda k: _store_loc(ctx, ("comp", ch.bufs[k]), v))
    _store_loc(ctx, ("comp", ch.count), _op(ctx, Op.ADD, n, _const(ctx, 1)))


def _chan_take(ctx: CoroCtx, ch: _Chan) -> int:
    """Remove the element at ``head``; the caller checked one is there."""
    out = ctx.new_reg()
    n = _load_loc(ctx, ("comp", ch.count))
    h = _load_loc(ctx, ("comp", ch.head))
    _ring(ctx, ch, h, lambda k: ctx.emit(
        Op.MOV, (out, _load_loc(ctx, ("comp", ch.bufs[k])))))
    _store_loc(ctx, ("comp", ch.head), _op(
        ctx, Op.MOD, _op(ctx, Op.ADD, h, _const(ctx, 1)), _const(ctx, len(ch.bufs))))
    _store_loc(ctx, ("comp", ch.count), _op(ctx, Op.SUB, n, _const(ctx, 1)))
    return out


def _channel_call(ctx: CoroCtx, ch: _Chan, e) -> Tuple[Optional[int], Optional[T]]:
    fn = e.func.attr
    want = {"put": 1, "try_put": 1, "get": 0, "try_get": 1}[fn]
    if len(e.args) != want:
        raise PssSemanticError(f"channel {fn}() takes {want} argument"
                               f"{'s' if want != 1 else ''}", loc=_loc(e))
    if fn in ("put", "try_put"):
        v = _eval_for(ctx, e.args[0], ch.elem)     # evaluated whether or not it fits
        if fn == "put":
            _chan_wait(ctx, ch, full=True)
            _chan_put(ctx, ch, v)
            return None, None
        ok = _op(ctx, Op.CMP_NE, _load_loc(ctx, ("comp", ch.count)),
                 _const(ctx, len(ch.bufs)))
        brz = ctx.emit(Op.BRZ, (ok, 0))
        _chan_put(ctx, ch, v)
        _patch(ctx, brz, len(ctx.code))
        return ok, BOOL
    if fn == "get":
        _chan_wait(ctx, ch, full=False)
        return _chan_take(ctx, ch), ch.elem
    # try_get(output T t): t is written only when an element is taken.
    target = e.args[0]
    if not isinstance(target, (E.ExprRefLocal, E.ExprRefField, E.ExprAttribute,
                               E.ExprRefUnresolved, E.ExprSubscript)):
        raise PssSemanticError("try_get()'s argument is an output: it must be "
                               "assignable", loc=_loc(target))
    ok = _op(ctx, Op.CMP_NE, _load_loc(ctx, ("comp", ch.count)), _const(ctx, 0))
    brz = ctx.emit(Op.BRZ, (ok, 0))
    _store(ctx, target, _chan_take(ctx, ch), ch.elem)
    _patch(ctx, brz, len(ctx.code))
    return ok, BOOL


# --------------------------------------------------------------------------- #
# Address handles, memory and executors (LRM 21.7, 21.13; B-D6)
# --------------------------------------------------------------------------- #

_MEM = {f"read{n}": n for n in (8, 16, 32, 64)}
_MEM.update({f"write{n}": n for n in (8, 16, 32, 64)})
_TRANSPARENT = "addr_reg_pkg::transparent_addr_space_c"


def _library_type(ctx: CoroCtx, name: str, scope, comp) -> Optional[T]:
    """The result type of a library function bc implements, or None."""
    if scope == "self":
        if name in _MEM:
            return T(_MEM[name], False) if name.startswith("read") else None
        if name == "make_handle_from_handle":
            return CHANDLE
        if name == "addr_value":
            return U64
    if scope == "path" and name in ("add_region", "add_nonallocatable_region"):
        return CHANDLE
    return None


def _library_call(ctx: CoroCtx, name: str, scope, comp, e):
    """A library function bc implements: ``(reg, type)``, ``(None, None)``
    for a void one, or None if *name* is not one."""
    if scope == "self" and name == "set_executor":
        st = proc_state(ctx)
        if st.comp_self is None or st.fns:
            raise LoweringError("set_executor() outside a component's init block "
                                "is not supported by bc (its executors are "
                                "resolved when the model is lowered)", loc=_loc(e))
        return None, None                       # resolved ahead: _executors
    if scope == "self" and name == "make_handle_from_handle":
        if not 2 <= len(e.args) <= 3:
            raise PssSemanticError("make_handle_from_handle(handle, offset[, sub])",
                                   loc=_loc(e))
        h = _eval_for(ctx, e.args[0], CHANDLE)
        off = _eval_for(ctx, e.args[1], U64)
        return _op(ctx, Op.ADD, h, off), CHANDLE
    if scope == "self" and (name in _MEM or name == "addr_value"):
        return _mem_call(ctx, name, e)
    if scope == "path" and name in ("add_region", "add_nonallocatable_region"):
        lay_t = comp.type_qname if comp is not None else None
        if lay_t is None or not ctx.lowerer.comps.is_a(lay_t, _TRANSPARENT):
            raise LoweringError(f"{name}() on a non-transparent address space is "
                                f"not supported by bc: it models a handle as its "
                                f"address", loc=_loc(e))
        if len(e.args) != 1:
            raise PssSemanticError(f"{name}() takes a region", loc=_loc(e))
        region = _struct_value(ctx, e.args[0])
        sub = region.type.sub("addr")
        if sub is None:
            raise PssSemanticError(f"{name}() of a transparent space takes a "
                                   f"transparent_addr_region_s", loc=_loc(e))
        return _load_loc(ctx, region.locs[sub[1]]), CHANDLE
    return None


def _mem_call(ctx: CoroCtx, name: str, e):
    """``addr_value`` / ``read<N>`` / ``write<N>``: the executor in force for
    the calling code's instance overrides it (21.7.2.6), else the platform
    answers (a builtin import; ``addr_value`` of a transparent handle is the
    handle)."""
    want = 2 if name.startswith("write") else 1
    if not want <= len(e.args) <= want + 1:
        raise PssSemanticError(f"{name}() takes {want} argument"
                               f"{'s' if want > 1 else ''} and a descriptor",
                               loc=_loc(e))
    ex = _executor_for(ctx, e)
    if ex is not None:
        fn = ctx.lowerer.comps.function(ex.type_qname, name, ctx.lowerer.functions)
        if fn is not None:
            return _inline(ctx, fn, e, ex)
    h = _eval_for(ctx, e.args[0], CHANDLE)
    if name == "addr_value":
        return h, U64
    n = _MEM[name]
    if name.startswith("read"):
        rd = ctx.new_reg()
        ctx.emit(Op.IMPORT, (BUILTIN_READ[n], rd, h), flags=INSTR_F_HAS_RET)
        return rd, T(n, False)
    v = _eval_for(ctx, e.args[1], T(n, False))
    ctx.emit(Op.IMPORT, (BUILTIN_WRITE[n], _VOID, h, v))
    return None, None


def _executors(ctx: CoroCtx) -> dict:
    """Instance id -> ``(base, id, type)`` of the executor in force there, or
    the string "dynamic": an instance's own ``set_executor(path)`` in its
    ``exec init_down`` / ``init_up`` (a later one wins), else its parent's
    (21.7.2.6). A ``set_executor`` that is not a top-level statement of the
    block is dynamic: bc resolves executors when it lowers (B-D6)."""
    lw = ctx.lowerer
    if lw.executors is not None:
        return lw.executors
    tree, comps = lw.comp_tree, lw.comps
    out = {}
    for inst in tree.instances:
        mine = None
        for kind in ("init_down", "init_up"):
            blk = comps.exec_block(inst.type_qname, kind)
            for i, stmt in enumerate(getattr(blk, "body", None) or []):
                path = _set_executor_path(stmt)
                if path is not None:
                    sub = comps.get(inst.type_qname).subs.get(path)
                    if sub is None:
                        raise LoweringError(f"set_executor({path}): not a component "
                                            f"instance of {inst.type_qname!r}",
                                            loc=_loc(stmt))
                    mine = (inst.base + sub.slot, inst.id + sub.inst, sub.type_qname)
                elif _mentions_set_executor(stmt):
                    mine = "dynamic"
        out[inst.id] = mine if mine is not None else (
            out.get(inst.parent) if inst.parent is not None else None)
    lw.executors = out
    return out


def _set_executor_path(stmt) -> Optional[str]:
    """``x.y[2]`` if *stmt* is ``set_executor(x.y[2]);``."""
    e = getattr(stmt, "expr", None) if isinstance(stmt, S.StmtExpr) else None
    if not (isinstance(e, E.ExprCall) and isinstance(e.func, E.ExprAttribute)
            and isinstance(e.func.value, E.TypeExprRefSelf)
            and e.func.attr == "set_executor" and len(e.args) == 1):
        return None
    parts, x = [], e.args[0]
    while True:
        if isinstance(x, E.ExprAttribute):
            parts.append(x.attr)
            x = x.value
        elif isinstance(x, E.ExprSubscript) and isinstance(x.value, E.ExprAttribute) \
                and isinstance(x.slice, E.ExprConstant):
            parts.append("%s[%d]" % (x.value.attr, x.slice.value))
            x = x.value.value
        else:
            break
    if not isinstance(x, E.TypeExprRefSelf) or not parts:
        return None
    return ".".join(reversed(parts))


def _mentions_set_executor(node) -> bool:
    if isinstance(node, E.ExprAttribute) and node.attr == "set_executor":
        return True
    if dc.is_dataclass(node) and not isinstance(node, type):
        for f in dc.fields(node):
            v = getattr(node, f.name)
            for x in (v if isinstance(v, list) else [v]):
                if (dc.is_dataclass(x) and not isinstance(x, type)
                        and _mentions_set_executor(x)):
                    return True
    return False


def _executor_for(ctx: CoroCtx, e) -> Optional[CompCtx]:
    """The executor in force for the calling code's instance, relative to
    the frame's instance; None for none (the platform)."""
    lw = ctx.lowerer
    if lw.comps is None or getattr(lw, "comp_tree", None) is None:
        return None
    st = proc_state(ctx)
    me = _self_comp(ctx) or st.action_comp
    frame = _frame_instance(ctx, e)
    if me is None or frame is None:
        return None
    by_id = {i.id: i for i in lw.comp_tree.instances}
    ex = _executors(ctx).get(frame.id + me.inst)
    if ex is None:
        return None
    if ex == "dynamic":
        raise LoweringError("the executor in force here is set by a set_executor() "
                            "that is not a top-level statement of an init block; "
                            "bc resolves executors when it lowers", loc=_loc(e))
    base, ex_id, ex_t = ex
    count = lw.comps.get(frame.type_qname).count
    if not frame.id <= ex_id < frame.id + count:
        raise LoweringError(f"the executor in force here ({by_id[ex_id].path!r}) is "
                            f"outside this code's component subtree, which bc "
                            f"cannot address yet", loc=_loc(e))
    return CompCtx(ex_t, base - frame.base, ex_id - frame.id)


def _frame_instance(ctx: CoroCtx, e):
    """The frame's own instance (an ``ScCompInstance``), when it is static:
    instance 0 for construction, else the one instance of the action's
    component type."""
    lw, st = ctx.lowerer, proc_state(ctx)
    insts = lw.comp_tree.instances
    if st.comp_self is not None:
        return insts[0]
    if st.component is None:
        return None
    mine = [i for i in insts if lw.comps.is_a(i.type_qname, st.component)]
    if len(mine) != 1:
        raise LoweringError(f"a memory access in an action of {st.component!r}, "
                            f"which has {len(mine)} instances: bc resolves the "
                            f"executor when it lowers, so it needs one", loc=_loc(e))
    return mine[0]


def _call_dispatch(ctx: CoroCtx, e, d) -> Tuple[Optional[int], Optional[T]]:
    """``ch[j].f()``: a call inlined in each element, one of which runs."""
    out, res = ctx.new_reg(), []

    def one(k):
        r, t = _call(ctx, _subst(e, d[0], k))
        if isinstance(t, StructT):
            raise LoweringError("a struct returned through a component array "
                                "index known only at run time is not supported "
                                "by bc", loc=_loc(e))
        if t is not None:
            ctx.emit(Op.MOV, (out, r))
        res.append(t)
    _dispatch(ctx, d, one)
    return (None, None) if res[0] is None else (out, res[0])


def _import_call(ctx: CoroCtx, name: str, e) -> Tuple[int, T]:
    decl = ctx.lowerer.imports[name]
    _no_string_import(name, decl, e)
    arg_regs = [ev(ctx, a)[0] for a in e.args]
    ret_reg = ctx.new_reg()
    flags = INSTR_F_HAS_RET | (INSTR_F_BLOCKING if decl["blocking"] else 0)
    ctx.emit(Op.IMPORT, tuple([decl["fn_id"], ret_reg] + arg_regs), flags=flags)
    ctx.lowerer.import_decls.setdefault(
        decl["fn_id"], {"fn": name, "blocking": decl["blocking"]})
    rt = decl.get("ret_type")
    return ret_reg, (T(min(int(rt[0] or 32), 64), bool(rt[1])) if rt else U64)


def _inline(ctx: CoroCtx, fn, call,
            comp: Optional[CompCtx] = None) -> Tuple[Optional[int], Optional[T]]:
    st = proc_state(ctx)
    if any(f.fn is fn for f in st.fns):
        return _call_recursive(ctx, fn, call, comp)
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
    # A struct parameter is a handle to the caller's instance (20.3.2): the
    # parameter names the argument's storage, and nothing is copied.
    types = ctx.lowerer.types
    values = []
    for i, p in enumerate(params):
        pt = value_type(p.annotation, types)
        src = call.args[i] if i < len(call.args) else defaults[i - first_default]
        if isinstance(pt, StructT) and not pt.leaves:
            # An empty struct (``mem_access_desc_s desc = {}``) holds nothing.
            values.append((p.arg, pt, _Place(pt, [])))
        elif isinstance(pt, StructT):
            values.append((p.arg, pt, _struct_value(ctx, src, pt)))
        else:
            values.append((p.arg, pt, _eval_for(ctx, src, pt)))

    ret_t = value_type(fn.returns, types) if fn.returns is not None else None
    frame = _Fn(fn=fn, scope_base=len(st.scopes), loop_base=len(st.loops),
                ret_type=ret_t, ret_slot=None, comp=comp)
    st.fns.append(frame)
    st.scopes.append({})
    try:
        for name, pt, r in values:
            if isinstance(pt, StructT):
                _declare(ctx, name, pt, locs=r.locs)
                continue
            v = _declare(ctx, name, pt)
            ctx.emit(Op.ST_LOCAL, (r, ctx.local_slot(v.slot)))
        if isinstance(ret_t, StructT):
            st.uniq += 1
            frame.ret_place = _struct_locals(ctx, f"{fn.name}$ret${st.uniq}", ret_t)
            for loc in frame.ret_place.locs:
                _store_loc(ctx, loc, _const(ctx, 0))
        elif ret_t is not None:
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
    if frame.ret_place is not None:
        return frame.ret_place, ret_t
    r = ctx.new_reg()
    ctx.emit(Op.LD_LOCAL, (r, ctx.local_slot(frame.ret_slot)))
    return r, ret_t


def _params(fn, call):
    """``[(name, type, argument-or-default)]`` of a call to *fn*."""
    args = getattr(fn, "args", None)
    params = list(getattr(args, "args", []) or []) if args is not None else []
    if args is not None and getattr(args, "vararg", None) is not None:
        raise LoweringError(f"varargs function {fn.name!r} is not supported by bc",
                            loc=_loc(call))
    defaults = list(getattr(args, "defaults", []) or []) if args is not None else []
    first_default = len(params) - len(defaults)
    if call is not None:
        if len(call.args) > len(params):
            raise PssSemanticError(f"too many arguments to {fn.name!r}", loc=_loc(call))
        if len(call.args) < first_default:
            raise PssSemanticError(f"too few arguments to {fn.name!r}", loc=_loc(call))
    return [(p.arg, p.annotation,
             (call.args[i] if call is not None and i < len(call.args)
              else defaults[i - first_default] if i >= first_default else None))
            for i, p in enumerate(params)]


def _call_recursive(ctx: CoroCtx, fn, call, comp) -> Tuple[Optional[int], Optional[T]]:
    """A call to *fn* while it is being inlined: a CALL of its called form.
    The arguments are all evaluated before any is staged, since evaluating
    one may itself CALL."""
    types = ctx.lowerer.types
    params = _params(fn, call)
    if len(params) > CALL_MAX_ARGS:
        raise LoweringError(f"recursive function {fn.name!r} has more than "
                            f"{CALL_MAX_ARGS} parameters", loc=_loc(call))
    regs = []
    for name, ann, src in params:
        pt = value_type(ann, types)
        if isinstance(pt, StructT):
            raise LoweringError(f"recursive function {fn.name!r} has a struct "
                                f"parameter {name!r}: not supported by bc", loc=_loc(call))
        regs.append(_eval_for(ctx, src, pt))
    ret_t = value_type(fn.returns, types) if fn.returns is not None else None
    if isinstance(ret_t, StructT):
        raise LoweringError(f"recursive function {fn.name!r} returns a struct: "
                            f"not supported by bc", loc=_loc(call))
    target = _called_form(ctx, fn, comp)
    for i, r in enumerate(regs):
        ctx.emit(Op.ARG, (r, i))
    rd = ctx.new_reg() if ret_t is not None else _VOID
    ctx.emit(Op.CALL, (target, rd))
    return (rd, ret_t) if ret_t is not None else (None, None)


def _called_form(ctx: CoroCtx, fn, comp: Optional[CompCtx]) -> int:
    """The coroutine index of *fn*'s called form in instance *comp* (relative
    to the frame's, which a callee shares), lowering it on first use."""
    lw = ctx.lowerer
    forms = getattr(lw, "called_forms", None)
    if forms is None:
        forms = lw.called_forms = {}
    key = (id(fn), comp)
    if key in forms:
        return forms[key]
    from .driver import new_ctx_like
    sub = new_ctx_like(ctx, f"{fn.name}$call{len(forms)}")
    desc = CoroDescriptor(name=sub.coro_name, code=[], blocks=[], frame_locals=[])
    forms[key] = idx = lw.add_branch_coro(desc)
    st = proc_state(sub)
    types = lw.types
    ret_t = value_type(fn.returns, types) if fn.returns is not None else None
    frame = _Fn(fn=fn, scope_base=0, loop_base=0, ret_type=ret_t, ret_slot=None,
                comp=comp, called=True)
    st.fns.append(frame)
    st.scopes.append({})
    for i, (name, ann, _) in enumerate(_params(fn, None)):
        v = _declare(sub, name, value_type(ann, types))
        r = sub.new_reg()
        sub.emit(Op.LD_ARG, (r, i))
        sub.emit(Op.ST_LOCAL, (r, sub.local_slot(v.slot)))
    _lower_block(sub, fn.body)
    sub.emit(Op.RET, (_const(sub, 0),) if ret_t is not None else ())
    desc.code, desc.frame_locals = sub.code, sub.frame_locals
    return idx


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
        t = value_type(s.annotation, ctx.lowerer.types)
        if isinstance(t, StructT):
            name = s.target.name if isinstance(s.target, E.ExprRefLocal) else None
            if name is None:
                raise LoweringError(f"unsupported declaration target {_cn(s.target)}",
                                    loc=_loc(s))
            # The value is evaluated before the name is visible.
            src = _struct_value(ctx, s.value, t) if s.value is not None else None
            v = _declare(ctx, name, t)
            dst = _Place(t, v.locs)
            if src is not None:
                _copy(ctx, dst, src)
            else:
                _init_struct(ctx, dst, s.annotation)
            return
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
        if st.fns and st.fns[-1].called:
            fr = st.fns[-1]
            if s.value is not None:
                if fr.ret_type is None:
                    raise PssSemanticError("return with a value from a void function",
                                           loc=_loc(s))
                ctx.emit(Op.RET, (_eval_for(ctx, s.value, fr.ret_type),))
            else:
                ctx.emit(Op.RET)
            return
        if st.fns:
            fr = st.fns[-1]
            if s.value is not None:
                if fr.ret_type is None:
                    raise PssSemanticError("return with a value from a void function",
                                           loc=_loc(s))
                if fr.ret_place is not None:
                    _copy(ctx, fr.ret_place, _struct_value(ctx, s.value, fr.ret_type))
                    fr.returns.append(ctx.emit(Op.BR, (0,)))
                    return
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

    if isinstance(s, S.StmtForeach):
        _lower_foreach(ctx, s)
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

    if isinstance(s, S.StmtYield):
        # PSS `yield;` (LRM 20.7.14): let other ready threads run, then go on.
        # YIELD re-queues this frame at the back of the ready queue.
        ctx.emit(Op.YIELD, ())
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


def _lower_foreach(ctx: CoroCtx, s) -> None:
    """20.7.8 over a component array: unrolled, element 0 first (20.7.8 d),
    so in each copy the index is a constant and the element a static
    instance (B-D2). ``continue`` ends one copy, ``break`` all of them."""
    arr = _comp_array(ctx, s.iter)
    if arr is None:
        raise LoweringError("foreach over anything but a component array is not "
                            "supported by bc", loc=_loc(s))
    base, name, n = arr
    st = proc_state(ctx)
    idx_name = s.index_var.name if isinstance(s.index_var, E.ExprRefLocal) else None
    it_name = s.target.name if isinstance(s.target, E.ExprRefLocal) else None
    loop = _Loop()
    st.loops.append(loop)
    try:
        for k in range(n):
            _push_scope(ctx)
            try:
                if idx_name is not None:
                    v = _declare(ctx, idx_name, I32)
                    v.const = k
                    ctx.emit(Op.ST_LOCAL, (_const(ctx, k), ctx.local_slot(v.slot)))
                if it_name is not None and it_name != idx_name:
                    st.scopes[-1][it_name] = _Var(
                        slot="", type=U64,
                        comp=_comp_sub(ctx, base, "%s[%d]" % (name, k)))
                first = len(loop.continues)
                _lower_block(ctx, s.body)
                for at in loop.continues[first:]:
                    _patch(ctx, at, len(ctx.code))
            finally:
                _pop_scope(ctx)
    finally:
        st.loops.pop()
    for at in loop.breaks:
        _patch(ctx, at, len(ctx.code))


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
            if v.upper is None:
                return _match_cmp(ctx, Op.CMP_EQ, subj, st_, v.lower)
            lo = _match_cmp(ctx, Op.CMP_GE, subj, st_, v.lower)
            hi = _match_cmp(ctx, Op.CMP_LE, subj, st_, v.upper)
            return _op(ctx, Op.AND, lo, hi)
        return _match_cmp(ctx, Op.CMP_EQ, subj, st_, v)
    raise LoweringError(f"unsupported match pattern {cn}", loc=_loc(pat))


def _match_cmp(ctx: CoroCtx, op: Op, subj: int, st_: T, value) -> int:
    vt = type_of(ctx, value)
    if st_.kind == "string" or vt.kind == "string":
        # As _string_compare: by interned index, so equality only.
        if st_.kind != vt.kind:
            raise PssSemanticError("a string matches only a string", loc=_loc(value))
        if op != Op.CMP_EQ:
            raise LoweringError("a range pattern on a string is not supported "
                                "by bc", loc=_loc(value))
        b, _ = ev(ctx, value)
        return _op(ctx, op, subj, b)
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


def lower_comp_init(tree, lowerer):
    """The coroutine constructing component tree *tree* (P1.5): each
    ``ScCompInit`` block in order, run in its instance (``self``), each its
    own scope. Its frame runs in instance 0, so an instance's slots are its
    absolute base."""
    from ..model import Block, CoroDescriptor
    ctx = CoroCtx.create(lowerer, [], coro_name="$comp_init")
    st = proc_state(ctx)
    for blk in tree.init:
        inst = tree.instances[blk.instance]
        st.comp_self = CompCtx(inst.type_qname, inst.base, inst.id)
        st.exec_kind = blk.kind
        sr = lowerer.prov.src_ref(blk)
        saved, ctx.cur_src_ref = ctx.cur_src_ref, (sr or ctx.cur_src_ref)
        try:
            _lower_block(ctx, blk.stmts)
        finally:
            ctx.cur_src_ref = saved
    return CoroDescriptor(name="$comp_init", code=ctx.code,
                          blocks=[Block(idx=0, pc_start=0, pc_end=len(ctx.code),
                                        suspend_op=0)],
                          frame_locals=ctx.frame_locals, src_ref=0)


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
