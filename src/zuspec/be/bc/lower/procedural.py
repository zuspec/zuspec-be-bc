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

import contextlib
import re
import dataclasses as dc
from typing import Dict, List, Optional, Tuple

from zuspec.ir.core import expr as E
from zuspec.ir.core import stmt as S
from zuspec.ir.core.expr_phase2 import ExprList
from zuspec.ir.core.data_type import DataTypeRegister, DataTypeRegisterGroup
from zuspec.ir.core.xf.pss_lower.comp_tree import HANDLE

from ..model import (Op, INSTR_F_HAS_RET, INSTR_F_BLOCKING, INSTR_F_SPIN,
                     BUILTIN_MESSAGE, BUILTIN_ERROR, BUILTIN_ERROR_MSG,
                     BUILTIN_FATAL_MSG, BUILTIN_PRINT, BUILTIN_READ, BUILTIN_WRITE,
                     BUILTIN_FILE_OPEN, BUILTIN_FILE_CLOSE, BUILTIN_FILE_EXISTS,
                     BUILTIN_FILE_WRITE, BUILTIN_URANDOM, BUILTIN_URANDOM_RANGE,
                     BUILTIN_FILE_READ, BUILTIN_FILE_READ_LINES,
                     BUILTIN_FILE_WRITE_LINES, CALL_MAX_ARGS, CoroDescriptor, FARITH_ADD, FARITH_SUB,
                     FARITH_MUL, FARITH_DIV, FARITH_NEG, FCMP_EQ, FCMP_NE, FCMP_LT,
                     FCMP_LE, FCMP_GT, FCMP_GE, FCVT_S2F, FCVT_U2F, FCVT_F2S,
                     FCVT_F32, FMATH_FNS, FMATH_BINARY, HEAP_SIZE, HEAP_STR_SLICE,
                     HEAP_STR_FIND, HEAP_STR_FIND_LAST, HEAP_STR_LOWER,
                     HEAP_STR_UPPER, HEAP_FORMAT, HEAPK_STR, heap_imm,
                     HEAP_STR_FIND_ALL, HEAP_STR_SPLIT, HEAP_STR_CHARS,
                     HEAP_JOIN, HEAP_STR_FROM_CHARS,
                     HEAP_GET, HEAP_SET, HEAP_INSERT, HEAP_DELETE,
                     HEAP_PUSH_FRONT, HEAP_PUSH_BACK, HEAP_POP_FRONT,
                     HEAP_POP_BACK, HEAP_CLEAR, HEAP_IN, HEAP_EQ, HEAP_COPY,
                     HEAP_FREE, HEAP_SHUFFLE, HEAPK_LIST, HEAP_CELL_INT,
                     HEAP_CELL_STR, HEAP_CELL_COLL, HEAPK_MAP, HEAPK_SET, HEAP_KEYS,
                     HEAP_VALUES, HEAP_TO_LIST, HEAP_TO_SET)
from .context import CoroCtx
from .errors import LoweringError, PssSemanticError
from .types import (COLLECTIONS, BOOL, CHANDLE, EXEC, FLOAT64, I32, NULL_REF, STRING, U64, T, StructT,
                    from_datatype, literal_type, merge, value_type)

_VOID = 0xFFFFFFFF
_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63

_ARITH = {
    E.BinOp.Add: Op.ADD, E.BinOp.Sub: Op.SUB, E.BinOp.Mult: Op.MUL,
    E.BinOp.BitAnd: Op.AND, E.BinOp.BitOr: Op.OR, E.BinOp.BitXor: Op.XOR,
}
_DIVMOD = {E.BinOp.Div, E.BinOp.FloorDiv, E.BinOp.Mod}
_FARITH = {E.BinOp.Add: FARITH_ADD, E.BinOp.Sub: FARITH_SUB, E.BinOp.Mult: FARITH_MUL,
           E.BinOp.Div: FARITH_DIV, E.BinOp.FloorDiv: FARITH_DIV}
_FCMP = {Op.CMP_EQ: FCMP_EQ, Op.CMP_NE: FCMP_NE, Op.CMP_LT: FCMP_LT,
         Op.CMP_LE: FCMP_LE, Op.CMP_GT: FCMP_GT, Op.CMP_GE: FCMP_GE}
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
    #: debug lowering: its entries in ``CoroCtx.named_locals``
    named: Optional[list] = None
    #: a struct parameter of a called form: its collection leaves are the
    #: caller's handles, handed back at RET, never freed here
    borrowed: bool = False


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
class _InstExpr:
    """A component instance in an expression, standing for a ``ref C`` in
    one copy of a reference's dispatch (F6.8b; see ``_ref_index``)."""
    comp: CompCtx
    loc: object = None


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
    #: a called form's struct parameters: their leaves are staged for the
    #: caller (LD_RES) at each RET, after a struct result's (calls.md)
    out_places: List[_Place] = dc.field(default_factory=list)
    #: an action's function (a pssc extension): ``self`` is the action
    action: bool = False


@dc.dataclass
class ProcState:
    fields: Dict[str, Tuple[int, T]] = dc.field(default_factory=dict)
    by_slot: Dict[int, T] = dc.field(default_factory=dict)
    #: a struct attribute (or a struct field of one), by dotted path: its
    #: leaves' object slots. None when a leaf has a type bc cannot hold.
    struct_fields: Dict[str, Optional[_Place]] = dc.field(default_factory=dict)
    component: Optional[str] = None
    #: action code: the action's qualified type (its functions' scope)
    action_type: Optional[str] = None
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
    st.action_type = at
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
    if t.is_bool or t.kind in ("string", "float") or t.width >= 64:
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
    if src.is_float or dst.is_float:
        return fconvert(ctx, r, src, dst)
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
    if src.is_float or dst.is_float:
        return fconvert(ctx, r, src, dst)
    if src.as_int() == dst.as_int():
        return r
    return normalize(ctx, r, dst.as_int())


def _fcvt(ctx: CoroCtx, r: int, kind: int) -> int:
    out = ctx.new_reg()
    ctx.emit(Op.FCVT, (out, r), imm=kind)
    return out


def fconvert(ctx: CoroCtx, r: int, src: T, dst: T) -> int:
    """8.7.2 with a float side (F6.6): an integer becomes the nearest float, a
    float an integer truncated toward zero, and a float32 is rounded to
    binary32 precision."""
    if dst.is_bool:
        return truth(ctx, r, src)
    if dst.is_float:
        if not src.is_float:
            if src.kind not in ("int", "bool", "enum"):
                raise PssSemanticError(f"a {src.kind} value does not convert to "
                                       f"a floating-point type")
            r = _fcvt(ctx, r, FCVT_S2F if src.signed else FCVT_U2F)
        elif src.width <= dst.width:
            return r
        return _fcvt(ctx, r, FCVT_F32) if dst.width == 32 else r
    if dst.kind not in ("int", "enum"):
        raise PssSemanticError(f"a floating-point value does not convert to "
                               f"a {dst.kind} type")
    return normalize(ctx, _fcvt(ctx, r, FCVT_F2S), dst.as_int())


def fold_convert(v, src: T, dst: T) -> int:
    """:func:`propagate` / :func:`fconvert` of a constant (a Python int or
    float), at lowering time: the register bits."""
    from ..interp.ops_proc import f2s, float_to_bits, round_f32  # (import cycle)
    if dst.is_bool:
        return int(v != 0)
    if dst.is_float:
        x = float(v)
        return float_to_bits(round_f32(x) if dst.width == 32 else x)
    if isinstance(v, float):
        return _fold_propagate(f2s(v), U64, dst) & _MASK64
    return _fold_propagate(int(v), src, dst) & _MASK64


def truth(ctx: CoroCtx, r: int, t: T) -> int:
    if t.is_bool:
        return r
    if t.is_float:
        out = ctx.new_reg()
        ctx.emit(Op.FCMP, (out, r, _const(ctx, 0)), imm=FCMP_NE)
        return out
    return _op(ctx, Op.CMP_NE, r, _const(ctx, 0))


# --------------------------------------------------------------------------- #
# Scopes
# --------------------------------------------------------------------------- #

def _push_scope(ctx: CoroCtx) -> None:
    proc_state(ctx).scopes.append({})


def _pop_scope(ctx: CoroCtx) -> None:
    scope = proc_state(ctx).scopes.pop()
    _list_free_scope(ctx, scope)
    _close_scope(ctx, scope)


def _close_scope(ctx: CoroCtx, scope: dict) -> None:
    """Debug lowering: the locals of *scope* leave scope here."""
    for v in scope.values():
        for ent in v.named or ():
            ent[3] = len(ctx.code)


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
    if ctx.lowerer.debug:
        _name_locals(ctx, name, v)
    st.scopes[-1][name] = v
    return v


def _name_locals(ctx: CoroCtx, name: str, v: _Var) -> None:
    """Debug lowering: record *v*'s frame locals under its PSS name."""
    if isinstance(v.type, StructT):
        leaves = [(f"{name}.{'.'.join(path)}", loc[1], t)
                  for (path, t), loc in zip(v.type.leaves, v.locs) if loc[0] == "local"]
    else:
        leaves = [(name, v.slot, v.type)]
    v.named = [[n, slot, len(ctx.code), None, t] for n, slot, t in leaves]
    ctx.named_locals.extend(v.named)


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


def _this_lookup(ctx: CoroCtx, name: str) -> Optional[_Var]:
    """The variable ``self.name`` names, or None if it names an attribute.

    The front end writes a local as ``ExprRefLocal`` and a function's
    parameter as ``self.name``. So in a function ``self.name`` is a
    parameter (the frame's first scope) or else the attribute, never a
    local of the body; in an action's exec block it is the action's
    attribute, even where a local shadows it (``this.x``)."""
    st = proc_state(ctx)
    if st.fns:
        base = st.fns[-1].scope_base
        if base < len(st.scopes) and name in st.scopes[base]:
            return st.scopes[base][name]
        return None if _comp_code(st) or name in st.fields \
            or name in st.struct_fields else _lookup(ctx, name)
    if (st.comp_self is None
            and (name in st.fields or name in st.struct_fields)):
        return None
    return _lookup(ctx, name)


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
    c = _ref_value(ctx, e)
    if c is not None:
        return T(32, False, "compref", name=c.type_qname)
    if isinstance(e, E.ExprSubscript) and _is_list(ctx, e.value):
        return type_of(ctx, e.value).elem
    if isinstance(e, (E.ExprAttribute, E.ExprRefUnresolved, E.ExprSubscript, E.ExprCall)):
        d = _dyn_index(ctx, e)
        if d is not None:
            if d[3] == "reference" and not d[4]:
                raise LoweringError(f"component reference {d[1]!r} can refer to no "
                                    f"instance here", loc=_loc(e))
            k = d[4][0] if d[3] == "reference" else 0
            return type_of(ctx, _subst(e, d[0], k))    # every element has one type
    if _is_nullfilehandle(ctx, e):
        return CHANDLE
    if isinstance(e, E.ExprSubscript) and _is_string(ctx, e.value):
        return STRING
    if _of_call(e):
        return _call_member_type(ctx, e)
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
    if isinstance(e, (E.ExprCompare, E.ExprBool)) or cn == "ExprIn":
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
    if cn == "ExprNull":
        return CHANDLE
    raise LoweringError(f"unsupported expression {cn} in bc procedural code", loc=_loc(e))


# --------------------------------------------------------------------------- #
# Expressions
# --------------------------------------------------------------------------- #

def eval_expr(ctx: CoroCtx, expr) -> int:
    """Lower ``expr`` at its self-determined type; return the register."""
    return ev(ctx, expr)[0]


def ev(ctx: CoroCtx, e, want: Optional[T] = None) -> Tuple[int, T]:
    """Lower ``e``. With *want*, the result is converted to that (propagated) type."""
    if want is not None and (want.is_float or type_of(ctx, e).is_float):
        # A float is self-determined: an integer operand of a float operator
        # is evaluated at its own type, then converted (8.7.1, Ex 44).
        if isinstance(e, E.ExprConstant) and isinstance(e.value, (int, float)):
            return _const(ctx, fold_convert(e.value, type_of(ctx, e), want)), want
        want_, want = want, None
        r, t = _ev(ctx, e, None)
        return (r, t) if t == want_ else (propagate(ctx, r, t, want_), want_)
    if want is not None and want.kind == "compref":
        return _ref_convert(ctx, e, want), want
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
        if isinstance(v, float):
            return _const(ctx, fold_convert(v, FLOAT64, FLOAT64)), FLOAT64
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
            # A legacy untyped local is one a legacy store wrote by its bare
            # name (or a repeat's index_var). A declared local's slot is never
            # a bare name, so reading one that nothing wrote is a scope bug
            # (a local read out of its scope): it read 0, silently.
            if proc_state(ctx).fns or e.name not in ctx.local_slots:
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

    c = _ref_value(ctx, e)
    if c is not None:
        return (_const(ctx, c.inst + _ref_origin(ctx)),
                T(32, False, "compref", name=c.type_qname))

    if _is_nullfilehandle(ctx, e):
        return _const(ctx, 0), CHANDLE
    if isinstance(e, E.ExprSubscript) and _is_list(ctx, e.value):
        return _list_get(ctx, e)
    if isinstance(e, E.ExprSubscript) and _is_string(ctx, e.value):
        return _substring(ctx, e), STRING
    if isinstance(e, (E.ExprAttribute, E.ExprRefUnresolved, E.ExprSubscript)):
        d = _dyn_index(ctx, e)
        if d is not None:
            out, t = ctx.new_reg(), type_of(ctx, e)
            _dispatch(ctx, d, lambda k: ctx.emit(
                Op.MOV, (out, ev(ctx, _subst(e, d[0], k))[0])))
            return out, t
        if _of_call(e):
            return _call_member(ctx, e)
        kind, t, where = _resolve_name(ctx, e)
        _not_opaque(t, e)
        if kind == "comp" and t.kind == "compref":
            return _ref_rebase(ctx, _load_loc(ctx, (kind, where)), e, -1), t
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
        if t.is_float:
            if e.op != E.UnaryOp.USub:
                raise PssSemanticError("~ does not apply to a floating-point "
                                       "value", loc=_loc(e))
            out = ctx.new_reg()
            ctx.emit(Op.FARITH, (out, r, r), imm=FARITH_NEG)
            return out, t
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
    if cn == "ExprIn":
        if _is_list(ctx, e.container):
            return _list_in(ctx, e), BOOL
        return _in(ctx, e), BOOL

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
        if dst.kind == "compref":
            return _ref_cast(ctx, e, dst), dst
        r, src = ev(ctx, e.value)
        return assign_convert(ctx, r, src, dst), dst

    if isinstance(e, E.ExprCall):
        r, t = _call(ctx, e)
        if t is None:
            raise PssSemanticError("a void function call has no value", loc=_loc(e))
        if isinstance(t, StructT):
            _not_scalar(t, e)
        return r, t

    if cn == "ExprNull":
        if want is not None and want.kind == "compref":
            return _const(ctx, NULL_REF), want
        return _const(ctx, 0), CHANDLE

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

    if t.is_float:
        return _float_bin(ctx, e, t)

    if op == E.BinOp.Exp:
        return normalize(ctx, _int_pow(ctx, e, t), t), t

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
    raise LoweringError(f"unsupported operator {op}", loc=_loc(e))


def _int_pow(ctx: CoroCtx, e, t: T) -> int:
    """Integer ``**`` (8.5.1, Table 12) by square-and-multiply over the
    exponent's bits; the exponent is self-determined. A negative exponent
    gives 1 for a base of 1, 1 or -1 for -1 (by parity), else 0: the base is
    kept only when it is 1 or -1 and the exponent becomes 2 or 3."""
    a, _ = ev(ctx, e.lhs, t)
    n, nt = ev(ctx, e.rhs)
    base, exp, out = ctx.new_reg(), ctx.new_reg(), ctx.new_reg()
    ctx.emit(Op.MOV, (base, a))
    ctx.emit(Op.MOV, (exp, n))
    if nt.signed:
        brz = ctx.emit(Op.BRZ, (_op(ctx, Op.SHR, n, _const(ctx, 63)), 0))
        unit = _op(ctx, Op.OR, _op(ctx, Op.CMP_EQ, a, _const(ctx, 1)),
                   _op(ctx, Op.CMP_EQ, a, _const(ctx, -1)))
        ctx.emit(Op.MOV, (base, _op(ctx, Op.AND, a,
                                    _op(ctx, Op.SUB, _const(ctx, 0), unit))))
        ctx.emit(Op.MOV, (exp, _op(ctx, Op.OR, _op(ctx, Op.AND, n, _const(ctx, 1)),
                                   _const(ctx, 2))))
        _patch(ctx, brz, len(ctx.code))
    ctx.emit(Op.MOV, (out, _const(ctx, 1)))
    top = len(ctx.code)
    done = ctx.emit(Op.BRZ, (exp, 0))
    skip = ctx.emit(Op.BRZ, (_op(ctx, Op.AND, exp, _const(ctx, 1)), 0))
    ctx.emit(Op.MOV, (out, _op(ctx, Op.MUL, out, base)))
    _patch(ctx, skip, len(ctx.code))
    ctx.emit(Op.MOV, (base, _op(ctx, Op.MUL, base, base)))
    ctx.emit(Op.MOV, (exp, _op(ctx, Op.SHR, exp, _const(ctx, 1))))
    ctx.emit(Op.BR, (top,))
    _patch(ctx, done, len(ctx.code))
    return out


def _float_bin(ctx: CoroCtx, e, t: T) -> Tuple[int, T]:
    """A float arithmetic operator (8.5.1, 8.7.1): binary64, then rounded to
    binary32 for a float32 result."""
    if e.op == E.BinOp.Exp:                     # pow(), binary64 (8.5.1)
        a = _eval_for(ctx, e.lhs, FLOAT64)
        out = _fmath(ctx, "pow", a, _eval_for(ctx, e.rhs, FLOAT64))
        return (_fcvt(ctx, out, FCVT_F32) if t.width == 32 else out), t
    k = _FARITH.get(e.op)
    if k is None:
        raise PssSemanticError(f"operator {e.op.name} does not apply to a "
                               f"floating-point value", loc=_loc(e))
    a, _ = ev(ctx, e.lhs, t)
    b, _ = ev(ctx, e.rhs, t)
    out = ctx.new_reg()
    ctx.emit(Op.FARITH, (out, a, b), imm=k)
    return (_fcvt(ctx, out, FCVT_F32) if t.width == 32 else out), t


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
    if _is_list(ctx, lhs) or _is_list(ctx, rhs):
        return _list_compare(ctx, op, lhs, rhs)
    lits = (ExprList, E.ExprStructLiteral)
    if isinstance(lhs, lits) and not isinstance(rhs, lits):
        lhs, rhs = rhs, lhs                 # 7.9.2.1: an array == a literal
    if isinstance(rhs, lits):
        lt = type_of(ctx, lhs)
        return _struct_compare(ctx, op, lhs, rhs, lt, lt)
    lt, rt = type_of(ctx, lhs), type_of(ctx, rhs)
    if isinstance(lt, StructT) or isinstance(rt, StructT):
        return _struct_compare(ctx, op, lhs, rhs, lt, rt)
    if lt.is_bool and rt.is_bool:
        a, _ = ev(ctx, lhs)
        b, _ = ev(ctx, rhs)
        return _op(ctx, op, a, b)
    if lt.kind == "string" or rt.kind == "string":
        return _string_compare(ctx, op, lhs, rhs, lt, rt)
    if "exec" in (lt.kind, rt.kind):
        return _exec_compare(ctx, op, lhs, rhs)
    if "compref" in (lt.kind, rt.kind):
        return _ref_compare(ctx, op, lhs, rhs)
    p = merge(lt, rt)
    a, _ = ev(ctx, lhs, p)
    b, _ = ev(ctx, rhs, p)
    return compare_regs(ctx, op, a, b, p)


def _exec_compare(ctx: CoroCtx, op: Op, lhs, rhs) -> int:
    """``executor() == null`` (24.6.2.5): an executor reference is its EXEC
    row id, and null is 0xFFFFFFFF, not a null chandle's 0."""
    if op not in (Op.CMP_EQ, Op.CMP_NE):
        raise PssSemanticError("executor references compare only with == and !=",
                               loc=_loc(lhs))
    regs = []
    for side in (lhs, rhs):
        if _cn(side) == "ExprNull":
            regs.append(_const(ctx, 0xFFFFFFFF))
        elif type_of(ctx, side).kind != "exec":
            raise PssSemanticError("an executor reference compares only with an "
                                   "executor reference or null", loc=_loc(side))
        else:
            regs.append(ev(ctx, side)[0])
    return _op(ctx, op, regs[0], regs[1])


def _ref_compare(ctx: CoroCtx, op: Op, lhs, rhs) -> int:
    """``r == null``, ``r != s`` (7.10): component references compare by the
    instance they refer to, only with == and !=."""
    if op not in (Op.CMP_EQ, Op.CMP_NE):
        raise PssSemanticError("component references compare only with == and !=",
                               loc=_loc(lhs))
    regs = []
    for side in (lhs, rhs):
        if _cn(side) == "ExprNull":
            regs.append(_const(ctx, NULL_REF))
        elif type_of(ctx, side).kind != "compref":
            raise PssSemanticError("a component reference compares only with a "
                                   "component reference or null", loc=_loc(side))
        else:
            regs.append(ev(ctx, side)[0])
    return _op(ctx, op, regs[0], regs[1])


def _string_compare(ctx: CoroCtx, op: Op, lhs, rhs, lt: T, rt: T) -> int:
    """7.6: strings compare with == and != only. A literal is its index in the
    interned table, so equal literals have equal indices; the compare is
    recorded, and made a content compare (HEAP STR_EQ) if the model makes
    strings at run time (``Lowerer.string_cmps``), so a model without them
    still runs on the C engine."""
    if lt.kind != rt.kind:
        raise PssSemanticError("a string compares only with a string", loc=_loc(lhs))
    if op not in (Op.CMP_EQ, Op.CMP_NE):
        raise PssSemanticError("strings compare only with == and !=", loc=_loc(lhs))
    a, _ = ev(ctx, lhs)
    b, _ = ev(ctx, rhs)
    out = ctx.new_reg()
    pc = ctx.emit(op, (out, a, b))
    ctx.lowerer.string_cmps.append((ctx.code, pc, op == Op.CMP_NE))
    return out


def _heap(ctx: CoroCtx, sub: int, *regs: int) -> int:
    """``HEAP`` sub-op *sub* on a string (heap.md): the result register."""
    ctx.lowerer.uses_heap = True
    out = ctx.new_reg()
    ctx.emit(Op.HEAP, (out,) + regs, imm=heap_imm(sub, HEAPK_STR))
    return out


#: string methods (7.6.3) bc runs: (sub-op, least and most args, result type)
_STR_METHODS = {"size": (HEAP_SIZE, 0, 0, I32), "find": (HEAP_STR_FIND, 1, 2, I32),
                "find_last": (HEAP_STR_FIND_LAST, 1, 2, I32),
                "lower": (HEAP_STR_LOWER, 0, 0, STRING),
                "upper": (HEAP_STR_UPPER, 0, 0, STRING)}
#: ... and those that return a new list the caller owns (7.6.3)
_STR_LIST_METHODS = {"find_all": (HEAP_STR_FIND_ALL, 1, I32),
                     "split": (HEAP_STR_SPLIT, 1, STRING),
                     "chars": (HEAP_STR_CHARS, 0, T(8, False))}


def _is_string(ctx: CoroCtx, x) -> bool:
    try:
        return type_of(ctx, x).kind == "string"
    except (LoweringError, PssSemanticError):
        return False


def _string_method(ctx: CoroCtx, e, type_only: bool = False):
    """``s.size()``, ``s.find(sub)``, ``s.lower()``... (7.6.3): ``(reg, type)``
    (reg None with *type_only*), or None if *e* is not a string method call."""
    f = e.func
    if not (isinstance(f, E.ExprAttribute)
            and (f.attr in _STR_METHODS or f.attr in _STR_LIST_METHODS)
            and _is_string(ctx, f.value)):
        return None
    if f.attr in _STR_LIST_METHODS:
        sub, lo, et = _STR_LIST_METHODS[f.attr]
        hi = lo
        rt = T(64, False, "list", elem=et)
    else:
        sub, lo, hi, rt = _STR_METHODS[f.attr]
    if not lo <= len(e.args) <= hi:
        raise PssSemanticError(f"string {f.attr}() takes {lo}"
                               + (f" to {hi}" if hi != lo else "") + " arguments",
                               loc=_loc(e))
    if type_only:
        return None, rt
    s, _ = ev(ctx, f.value)
    regs = [s]
    if f.attr in ("find", "find_last", "find_all", "split"):
        if not _is_string(ctx, e.args[0]):
            raise PssSemanticError(f"{f.attr}(): the "
                                   + ("separator" if f.attr == "split" else "sub-string")
                                   + " must be a string", loc=_loc(e))
        regs.append(ev(ctx, e.args[0])[0])
    if f.attr in ("find", "find_last"):
        regs.append(ev(ctx, e.args[1], I32)[0] if len(e.args) > 1
                    else _const(ctx, 0 if f.attr == "find" else -1 & _MASK64))
    r = _heap(ctx, sub, *regs)
    return (normalize(ctx, r, I32) if rt == I32 else r), rt


def _substring(ctx: CoroCtx, e) -> int:
    """``s[i]``, ``s[a..b]``, ``s[..b]``, ``s[a..]`` (8.6.3): a new string; a
    position outside the string is a run-time error."""
    s, _ = ev(ctx, e.value)
    if _cn(e.slice) in ("ExprSlice", "ExprRange"):
        lo, hi = e.slice.lower, e.slice.upper
        a = _const(ctx, 0) if lo is None else ev(ctx, lo, I32)[0]
        b = _const(ctx, -1 & _MASK64) if hi is None else ev(ctx, hi, I32)[0]
    else:
        a = b = ev(ctx, e.slice, I32)[0]
    return _heap(ctx, HEAP_STR_SLICE, s, a, b)


# --------------------------------------------------------------------------- #
# Lists, maps and sets (7.9.3-7.9.5; F6.10a-b; spec/heap.md)
# --------------------------------------------------------------------------- #
# A collection local's slot holds a heap handle (0 = the empty collection).
# A collection is a value: a declaration or assignment from a named one
# copies it, the old value is freed on overwrite and at the end of its scope;
# a string stored into or read out of a collection is copied, since the
# collection owns its cells. Maps and sets keep insertion order.

_HEAPK = {"list": HEAPK_LIST, "map": HEAPK_MAP, "set": HEAPK_SET}


def _is_list(ctx: CoroCtx, x) -> bool:
    """*x* is a list, map or set."""
    if isinstance(x, (ExprList, E.ExprRangeList)):
        return False
    try:
        return type_of(ctx, x).kind in COLLECTIONS
    except (LoweringError, PssSemanticError):
        return False


def _cell(t: Optional[T]) -> int:
    if t is not None and t.kind in COLLECTIONS:
        return HEAP_CELL_COLL
    return HEAP_CELL_STR if t is not None and t.kind == "string" else HEAP_CELL_INT


def _lheap(ctx: CoroCtx, sub: int, lt: T, *regs: int, out=None) -> int:
    """``HEAP`` sub-op *sub* on a collection of type *lt*: the result register."""
    ctx.lowerer.uses_heap = True
    out = ctx.new_reg() if out is None else out
    ctx.emit(Op.HEAP, (out,) + regs,
             imm=heap_imm(sub, _HEAPK[lt.kind], _cell(lt.elem), _cell(lt.key)))
    return out


def _owned(ctx: CoroCtx, r: int, t: T) -> int:
    """A string value *r* that a collection (or the reader of one) owns."""
    return _heap(ctx, HEAP_COPY, r) if t.kind == "string" else r


def _elem_value(ctx: CoroCtx, x, t: T) -> int:
    """Element, key or value *x* of type *t* as a value a collection keeps:
    an owned string, or an owned collection (a nested one, F6.10c3)."""
    if t.kind in COLLECTIONS:
        return _list_value(ctx, x, t)
    return _owned(ctx, _eval_for(ctx, x, t), t)


def _list_value(ctx: CoroCtx, e, lt: T) -> int:
    """Collection value *e* as a handle the caller owns: a new one from a
    literal ``{a, b}`` / ``{k: v}``, a copy of a named one."""
    if _cn(e) == "ExprDict":
        if lt.kind != "map":
            raise PssSemanticError(f"a {_tname(lt)} value is needed here", loc=_loc(e))
        h = _const(ctx, 0)
        for k, v in zip(e.keys, e.values):
            kr = _owned(ctx, _eval_for(ctx, k, lt.key), lt.key)
            vr = _elem_value(ctx, v, lt.elem)
            h = _lheap(ctx, HEAP_INSERT, lt, h, kr, vr)
        return h
    if isinstance(e, ExprList) or _cn(e) == "ExprStructLiteral" and not getattr(e, "fields", None):
        if lt.kind == "map" and getattr(e, "elts", None):
            raise PssSemanticError(f"a {_tname(lt)} value is needed here", loc=_loc(e))
        h = _const(ctx, 0)
        sub = HEAP_PUSH_BACK if lt.kind == "list" else HEAP_INSERT
        for x in getattr(e, "elts", None) or ():
            v = _elem_value(ctx, x, lt.elem)
            h = _lheap(ctx, sub, lt, h, v)
        return h
    vt = type_of(ctx, e)
    if (vt.kind, vt.elem, vt.key) != (lt.kind, lt.elem, lt.key):
        raise PssSemanticError(f"a {_tname(lt)} value is needed here", loc=_loc(e))
    r, _ = ev(ctx, e)
    if isinstance(e, (E.ExprRefLocal, E.ExprRefField, E.ExprAttribute, E.ExprSubscript)):
        r = _lheap(ctx, HEAP_COPY, lt, r)
    return r


def _tname(t: T) -> str:
    if t.kind == "map":
        return f"map<{_tname(t.key)}, {_tname(t.elem)}>"
    if t.kind in COLLECTIONS:
        return f"{t.kind}<{_tname(t.elem)}>"
    return t.name or t.kind if t.kind != "int" else ("int" if t.signed else "bit") + f"[{t.width}]"


def _list_slot(ctx: CoroCtx, e):
    """Where collection *e* lives (a ``_Place`` location), which a mutating
    method stores back: a local, a parameter, a field of a struct place, or
    an element of a collection (``ll[0].push_back(1)``, F6.10c3)."""
    if isinstance(e, E.ExprSubscript) and _is_list(ctx, e.value):
        return ("elem", e)
    v = (_lookup(ctx, e.name) if isinstance(e, E.ExprRefLocal)
         # a parameter, as the front end writes it: ``self.name``
         else _this_lookup(ctx, e.attr) if (isinstance(e, E.ExprAttribute)
                                        and isinstance(e.value, E.TypeExprRefSelf))
         else None)
    if v is not None and v.slot and v.locs is None:
        return ("local", v.slot)
    if v is None and isinstance(e, E.ExprAttribute):
        base = _place(ctx, e.value) if not isinstance(e.value, E.TypeExprRefSelf) else None
        if base is not None:
            t, i = _member(base, e)
            if not isinstance(t, StructT):
                return base.locs[i]
    if v is None and isinstance(e, (E.ExprAttribute, E.ExprRefLocal)):
        # an action or component attribute
        kind, t, where = _resolve_name(ctx, e)
        if isinstance(t, T) and t.kind in COLLECTIONS:
            return kind, where
    raise LoweringError(f"only a local {type_of(ctx, e).kind} can be changed in bc yet",
                        loc=_loc(e))


def _list_replace(ctx: CoroCtx, loc, r: int, lt: T) -> None:
    """Store owned handle *r* to collection location *loc*, freeing its old value."""
    _lheap(ctx, HEAP_FREE, lt, _load_loc(ctx, loc))
    _store_loc(ctx, loc, r)


def _owns(v: _Var) -> bool:
    """Struct variable *v* holds its own leaves (not a by-handle parameter)."""
    return not v.borrowed and all(loc[0] == "local" and loc[1].startswith(v.slot + ".")
                                  for loc in v.locs)


def _list_free_scope(ctx: CoroCtx, scope: dict) -> None:
    """The collection locals of *scope* (and collection leaves of its struct
    locals) leave it: free them (a ``break`` or ``return`` skips this; the
    declaration frees a value left over)."""
    for v in scope.values():
        if isinstance(v.type, T) and v.type.kind in COLLECTIONS and v.slot:
            _list_replace(ctx, ("local", v.slot), _const(ctx, 0), v.type)
        elif isinstance(v.type, StructT) and v.locs is not None and _owns(v):
            for (_, t), loc in zip(v.type.leaves, v.locs):
                if t.kind in COLLECTIONS:
                    _list_replace(ctx, loc, _const(ctx, 0), t)


def _list_free_fn(ctx: CoroCtx, fr, inner: bool = False) -> None:
    """A ``return`` leaves function *fr*: free the collection locals of its
    scopes (with *inner*, not its outermost one, which the inlined body
    frees where every return joins)."""
    st = proc_state(ctx)
    for scope in st.scopes[fr.scope_base + (1 if inner else 0):]:
        _list_free_scope(ctx, scope)


def _index_of(ctx: CoroCtx, lt: T, x, owned: bool = False) -> int:
    """List index or map key *x* (a key the map keeps is *owned*)."""
    if lt.kind == "map":
        k = _eval_for(ctx, x, lt.key)
        return _owned(ctx, k, lt.key) if owned else k
    if lt.kind == "set":
        raise PssSemanticError("a set has no index (7.9.5)", loc=_loc(x))
    return ev(ctx, x, I32)[0]


def _list_get(ctx: CoroCtx, e) -> Tuple[int, T]:
    """``l[i]`` / ``m[k]``: the element (a string is copied); out of range or
    a missing key = error."""
    lt = type_of(ctx, e.value)
    h, _ = ev(ctx, e.value)
    i = _index_of(ctx, lt, e.slice)
    return _owned(ctx, _lheap(ctx, HEAP_GET, lt, h, i), lt.elem), lt.elem


def _list_set(ctx: CoroCtx, target, value) -> None:
    """``l[i] = v`` / ``m[k] = v`` (a new key is added)."""
    lt = type_of(ctx, target.value)
    slot = _list_slot(ctx, target.value)
    h, _ = ev(ctx, target.value)
    i = _index_of(ctx, lt, target.slice, owned=True)
    v = _elem_value(ctx, value, lt.elem)
    _store_loc(ctx, slot, _lheap(ctx, HEAP_SET, lt, h, i, v))


#: collection methods (7.9.3.2, 7.9.4.2, 7.9.5.2): name -> (sub-op, arguments,
#: result). Arguments: "i" an index, "k" a key or element looked up, "K" one
#: the collection keeps, "v" a value it keeps. Results: None (changes the
#: collection), "size", "pop" (an element, and changes it), or the kind of a
#: new collection: "list"/"set" of the elements, "keys" (set), "values" (list).
_METHODS = {
    "list": {"size": (HEAP_SIZE, "", "size"), "clear": (HEAP_CLEAR, "", None),
             "delete": (HEAP_DELETE, "i", None), "insert": (HEAP_INSERT, "iv", None),
             "push_front": (HEAP_PUSH_FRONT, "v", None),
             "push_back": (HEAP_PUSH_BACK, "v", None),
             "pop_front": (HEAP_POP_FRONT, "", "pop"),
             "pop_back": (HEAP_POP_BACK, "", "pop"),
             "shuffle": (HEAP_SHUFFLE, "", None), "to_set": (HEAP_TO_SET, "", "set")},
    "map": {"size": (HEAP_SIZE, "", "size"), "clear": (HEAP_CLEAR, "", None),
            "delete": (HEAP_DELETE, "k", None), "insert": (HEAP_INSERT, "Kv", None),
            "keys": (HEAP_KEYS, "", "keys"), "values": (HEAP_VALUES, "", "values")},
    "set": {"size": (HEAP_SIZE, "", "size"), "clear": (HEAP_CLEAR, "", None),
            "delete": (HEAP_DELETE, "k", None), "insert": (HEAP_INSERT, "K", None),
            "to_list": (HEAP_TO_LIST, "", "list")},
}
#: list methods that make a new string (7.9.3.2): (sub-op, element kind, args)
_LIST_TO_STR = {"join": (HEAP_JOIN, "string", 1),
                "str_from_chars": (HEAP_STR_FROM_CHARS, "int", 0)}


def _method_result(lt: T, res) -> Optional[T]:
    if res == "size":
        return I32
    if res == "pop":
        return lt.elem
    if res in ("list", "set"):
        return T(64, False, res, elem=lt.elem)
    if res == "keys":
        return T(64, False, "set", elem=lt.key)
    if res == "values":
        return T(64, False, "list", elem=lt.elem)
    return None


def _list_method(ctx: CoroCtx, e, type_only: bool = False):
    """``l.push_back(v)``, ``m.keys()``, ``s.size()``... : ``(reg, type)``
    (reg None with *type_only*; type None for a method with no result), or
    None if *e* is not a collection method call. A new collection a method
    makes is owned by the caller."""
    f = e.func
    if not (isinstance(f, E.ExprAttribute) and _is_list(ctx, f.value)):
        return None
    lt = type_of(ctx, f.value)
    if lt.kind == "list" and f.attr in _LIST_TO_STR:
        return _list_to_str(ctx, e, lt, type_only)
    if f.attr not in _METHODS[lt.kind]:
        return None
    sub, args, res = _METHODS[lt.kind][f.attr]
    if len(e.args) != len(args):
        raise PssSemanticError(f"{lt.kind} {f.attr}() takes {len(args)} argument"
                               + ("" if len(args) == 1 else "s"), loc=_loc(e))
    rt = _method_result(lt, res)
    if type_only:
        return None, rt
    changes = res is None or res == "pop"
    if changes:
        slot = _list_slot(ctx, f.value)
    h, _ = ev(ctx, f.value)
    regs = [h]
    for a, x in zip(args, e.args):
        if a == "i":
            regs.append(ev(ctx, x, I32)[0])
        elif a in "kK":
            kt = lt.key if lt.kind == "map" else lt.elem
            regs.append(_elem_value(ctx, x, kt) if a == "K"
                        else _eval_for(ctx, x, kt))
        else:
            regs.append(_elem_value(ctx, x, lt.elem))
    if res == "size":
        return normalize(ctx, _lheap(ctx, HEAP_SIZE, lt, h), I32), I32
    if res == "pop":
        out = ctx.new_reg()
        hr = ctx.new_reg()          # POP_*: rd = the value, rd + 1 = the handle
        assert hr == out + 1
        _lheap(ctx, sub, lt, *regs, out=out)
        _store_loc(ctx, slot, hr)
        return out, lt.elem
    if not changes:
        return _lheap(ctx, sub, lt, *regs), rt
    _store_loc(ctx, slot, _lheap(ctx, sub, lt, *regs))
    return None, None


def _list_to_str(ctx: CoroCtx, e, lt: T, type_only: bool):
    """``l.join(sep)`` on a ``list<string>``, ``l.str_from_chars()`` on a
    ``list<bit[8]>`` (7.9.3.2): a new string."""
    f = e.func
    sub, ek, n = _LIST_TO_STR[f.attr]
    if lt.elem.kind != ek or ek == "int" and lt.elem.width > 8:
        raise PssSemanticError(f"{f.attr}() needs a list<"
                               + ("string" if ek == "string" else "bit[8]")
                               + f">, not a {_tname(lt)}", loc=_loc(e))
    if len(e.args) != n:
        raise PssSemanticError(f"list {f.attr}() takes {n} argument"
                               + ("" if n == 1 else "s"), loc=_loc(e))
    if type_only:
        return None, STRING
    h, _ = ev(ctx, f.value)
    regs = [h]
    if n:
        if not _is_string(ctx, e.args[0]):
            raise PssSemanticError("join(): the connector must be a string",
                                   loc=_loc(e))
        regs.append(ev(ctx, e.args[0])[0])
    r = _lheap(ctx, sub, lt, *regs)
    if isinstance(f.value, E.ExprCall):     # a temporary list (s.split(..).join(..))
        _lheap(ctx, HEAP_FREE, lt, h)
    return r, STRING


def _list_in(ctx: CoroCtx, e) -> int:
    """``x in l`` / ``in s`` / ``in m.keys()`` (8.5.9): by value, strings by
    content; a collection a method made here is freed after."""
    lt = type_of(ctx, e.container)
    v = _eval_for(ctx, e.value, lt.key if lt.kind == "map" else lt.elem)
    h, _ = ev(ctx, e.container)
    r = _lheap(ctx, HEAP_IN, lt, h, v)
    if isinstance(e.container, E.ExprCall):
        _lheap(ctx, HEAP_FREE, lt, h)
    return r


def _list_compare(ctx: CoroCtx, op: Op, lhs, rhs) -> int:
    """``c1 == c2`` / ``!=`` (7.9.3.1, 7.9.4.1, 7.9.5.1): lists equal in order,
    sets and maps regardless of order; a literal side is a temporary."""
    lt = type_of(ctx, lhs if _is_list(ctx, lhs) else rhs)
    if op not in (Op.CMP_EQ, Op.CMP_NE):
        raise PssSemanticError(f"{lt.kind}s compare only with == and !=", loc=_loc(lhs))
    regs, temps = [], []
    for x in (lhs, rhs):
        if _is_list(ctx, x):
            regs.append(ev(ctx, x)[0])
        else:
            regs.append(_list_value(ctx, x, lt))
            temps.append(regs[-1])
    r = _lheap(ctx, HEAP_EQ, lt, *regs)
    for t in temps:
        _lheap(ctx, HEAP_FREE, lt, t)
    return _op(ctx, Op.XOR, r, _const(ctx, 1)) if op == Op.CMP_NE else r


def _lower_foreach_list(ctx: CoroCtx, s) -> None:
    """``foreach (e : l)`` / ``foreach (l[i])`` (20.7.8): a loop over the
    indices; the iterator stands for ``l[i]``, the size is read each time.
    A set walks a list copy of it, and a map a list of its keys (the index is
    the key, the iterator ``m[key]``), both in insertion order and freed
    after the loop."""
    st = proc_state(ctx)
    idx_name = s.index_var.name if isinstance(s.index_var, E.ExprRefLocal) else None
    it_name = s.target.name if isinstance(s.target, E.ExprRefLocal) else None
    ct = type_of(ctx, s.iter)
    _push_scope(ctx)
    try:
        seq = s.iter
        if ct.kind != "list":
            lt = T(64, False, "list", elem=ct.key if ct.kind == "map" else ct.elem)
            h, _ = ev(ctx, s.iter)
            if ct.kind == "map":
                ks = _lheap(ctx, HEAP_KEYS, ct, h)
                kst = T(64, False, "set", elem=ct.key)
                r = _lheap(ctx, HEAP_TO_LIST, kst, ks)
                _lheap(ctx, HEAP_FREE, kst, ks)
            else:
                r = _lheap(ctx, HEAP_TO_LIST, ct, h)
            cname = f"$c{st.uniq + 1}"
            _list_replace(ctx, ("local", _declare(ctx, cname, lt).slot), r, lt)
            seq = E.ExprRefLocal(name=cname)
        counter = idx_name if ct.kind == "list" else None
        iname = counter or f"$i{st.uniq + 1}"
        slot = ctx.local_slot(_declare(ctx, iname, I32).slot)
        ctx.emit(Op.ST_LOCAL, (_const(ctx, 0), slot))
        body = s.body
        elem = E.ExprSubscript(value=seq, slice=E.ExprRefLocal(name=iname))
        if ct.kind == "map":
            if idx_name is not None:
                body = _replace_local(body, idx_name, elem)
            if it_name is not None and it_name != idx_name:
                body = _replace_local(body, it_name,
                                      E.ExprSubscript(value=s.iter, slice=elem))
        else:
            if ct.kind == "set" and idx_name is not None:
                raise PssSemanticError("a set has no index (7.9.5)", loc=_loc(s))
            if it_name is not None and it_name != idx_name:
                body = _replace_local(body, it_name, elem)
        top = len(ctx.code)
        i = ctx.new_reg()
        ctx.emit(Op.LD_LOCAL, (i, slot))
        lt = type_of(ctx, seq)
        n = _lheap(ctx, HEAP_SIZE, lt, ev(ctx, seq)[0])
        brz = ctx.emit(Op.BRZ, (_op(ctx, Op.CMP_LT, i, n), 0))
        loop = _body_loop(ctx, body)
        cont = len(ctx.code)
        i2 = ctx.new_reg()
        ctx.emit(Op.LD_LOCAL, (i2, slot))
        ctx.emit(Op.ST_LOCAL, (_op(ctx, Op.ADD, i2, _const(ctx, 1)), slot))
        ctx.emit(Op.BR, (top,))
        _close_loop(ctx, loop, cont=cont, exit_=len(ctx.code))
        _patch(ctx, brz, len(ctx.code))
    finally:
        _pop_scope(ctx)


def _format_string(ctx: CoroCtx, e) -> int:
    """``format_string(fmt, args...)`` (24.1): a heap string, formatted as
    message() formats (its message entry, HEAP FORMAT)."""
    if not e.args or not (isinstance(e.args[0], E.ExprConstant)
                          and isinstance(e.args[0].value, str)):
        raise LoweringError("format_string() needs a string-literal format in bc",
                            loc=_loc(e))
    regs, descs = [_const(ctx, 0)], []
    for a in e.args[1:]:
        r, t = ev(ctx, a)
        regs.append(r)
        descs.append(t.descriptor())
    slots = [ctx.local_slot(f"$msg{i}") for i in range(len(regs))]
    try:
        idx = ctx.lowerer.add_message(e.args[0].value, descs, slots)
    except ValueError as ex:
        raise PssSemanticError(f"format_string(): {ex}", loc=_loc(e))
    for r, slot in zip(regs, slots):
        ctx.emit(Op.ST_LOCAL, (r, slot))
    return _heap(ctx, HEAP_FORMAT, _const(ctx, idx))


def _is_format_string(ctx: CoroCtx, e) -> bool:
    if "format_string" not in str(getattr(e.func, "name", getattr(e.func, "attr", ""))):
        return False
    name, scope, _ = _callee(ctx, e.func)
    return (name is not None and scope != "path"
            and _std_base(name) == "format_string"
            and _lookup_function(ctx, name, scope, None) is None)


def compare_regs(ctx: CoroCtx, op: Op, a: int, b: int, p: T) -> int:
    if p.is_float:
        out = ctx.new_reg()
        ctx.emit(Op.FCMP, (out, a, b), imm=_FCMP[op])
        return out
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


def _in_comp_fn(st: ProcState) -> bool:
    """Lowering an inlined component or package function (not an action's)."""
    return bool(st.fns) and not st.fns[-1].action


def _comp_code(st: ProcState) -> bool:
    """Lowering component code: a component or package function, or a
    construction block. An action's function is action code."""
    return _in_comp_fn(st) or st.comp_self is not None


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
    if isinstance(e, _InstExpr):
        return e.comp
    if isinstance(e, E.TypeExprRefSelf):
        return _self_comp(ctx)
    if isinstance(e, (E.ExprRefLocal, E.ExprRefUnresolved)):
        v = _lookup(ctx, e.name)
        return v.comp if v is not None else None
    if isinstance(e, E.ExprAttribute):
        if isinstance(e.value, E.TypeExprRefSelf):
            if _this_lookup(ctx, e.attr) is not None:
                return None                     # a parameter or local
            if e.attr == "comp" and not _comp_code(st):
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
    if isinstance(e.value, E.TypeExprRefSelf) and _this_lookup(ctx, e.attr) is not None:
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
# take its instance from a register. A data array (F6.1b) is dispatched the
# same way: its elements are separate slots, like a struct's fields.

def _array_len(ctx: CoroCtx, e) -> Optional[int]:
    """The element count if *e* is a data array place, else None."""
    p = _place(ctx, e)
    lv = p.type.leaves if p is not None else ()
    if not lv or not lv[0][0][0].isdigit():
        return None
    return len({path[0] for path, _ in lv})


def _array_method(ctx: CoroCtx, e) -> Optional[Tuple[int, T]]:
    """``a.size()`` / ``a.sum()`` of a data array (7.9.2.2), else None: the
    size is a constant; the sum is an ``int`` (Table 8) adding every element
    with 32-bit wrap, as the op-model targets' array_methods lowers it."""
    f = e.func
    if e.args or not isinstance(f, E.ExprAttribute) or f.attr not in ("size", "sum"):
        return None
    n = _array_len(ctx, f.value)
    if n is None:
        return None
    if f.attr == "size":
        return _const(ctx, n), I32
    acc = _const(ctx, 0)
    for k in range(n):
        r, _ = ev(ctx, E.ExprSubscript(value=f.value, slice=E.ExprConstant(value=k)), I32)
        acc = normalize(ctx, _op(ctx, Op.ADD, acc, r), I32)
    return acc, I32


def _in(ctx: CoroCtx, e) -> int:
    """``x in c`` (8.5.9) as a 0/1 register: ``c`` is a value list (items and
    ``[lo..hi]`` ranges), a data array, or a slice of one, whose bounds may be
    known only at run time; a slice outside the array or with ``lo > hi``
    holds no elements."""
    c = e.container
    terms: List[int] = []
    # The operands are context-determined, as in a comparison (8.5.9): the
    # value is evaluated at the type merged with every item's.
    if isinstance(c, (ExprList, E.ExprRangeList)):
        items = c.ranges if isinstance(c, E.ExprRangeList) else c.elts
        others = [x for it in items for x in (
            (it.lower, it.upper) if _cn(it) == "ExprRange" else (it,)) if x is not None]
    else:
        arr = c.value if isinstance(c, E.ExprSubscript) and _cn(c.slice) in (
            "ExprSlice", "ExprRange") else c
        others = [E.ExprSubscript(value=arr, slice=E.ExprConstant(value=0))] \
            if _array_len(ctx, arr) else []
    st_ = type_of(ctx, e.value)
    if st_.kind in ("int", "enum", "float") and not isinstance(st_, StructT):
        for x in others:
            xt = type_of(ctx, x)
            if xt.kind in ("int", "enum", "float") and not isinstance(xt, StructT):
                st_ = merge(st_, xt)
    subj, st_ = ev(ctx, e.value, st_)
    if isinstance(c, (ExprList, E.ExprRangeList)):
        for it in items:
            if _cn(it) == "ExprRange" and it.upper is not None:
                terms.append(_op(ctx, Op.AND,
                                 _match_cmp(ctx, Op.CMP_GE, subj, st_, it.lower),
                                 _match_cmp(ctx, Op.CMP_LE, subj, st_, it.upper)))
            else:
                v = it.lower if _cn(it) == "ExprRange" else it
                terms.append(_match_cmp(ctx, Op.CMP_EQ, subj, st_, v))
    else:
        arr, lo, hi = c, None, None
        if isinstance(c, E.ExprSubscript) and _cn(c.slice) in ("ExprSlice", "ExprRange"):
            arr, lo, hi = c.value, c.slice.lower, c.slice.upper
        n = _array_len(ctx, arr)
        if n is None:
            raise LoweringError(f"'in' over {_cn(c)} is not supported by bc: the "
                                f"right side is a value list, an array or a slice "
                                f"of one", loc=_loc(c))
        bounds = []
        for b, op in ((lo, Op.CMP_GE), (hi, Op.CMP_LE)):
            if b is None:
                continue
            kb = _const_index(ctx, b)
            if kb is not None:
                bounds.append((op, kb, None, None))
            else:
                r, t = ev(ctx, b)
                bounds.append((op, None, r, t))
        for k in range(n):
            pred, skip = None, False
            for op, kb, r, t in bounds:
                if kb is not None:
                    skip = skip or not (k >= kb if op == Op.CMP_GE else k <= kb)
                    continue
                p = merge(I32, t)
                g = compare_regs(ctx, op, propagate(ctx, _const(ctx, k), I32, p),
                                 propagate(ctx, r, t, p), p)
                pred = g if pred is None else _op(ctx, Op.AND, pred, g)
            if skip:
                continue
            eq = _match_cmp(ctx, Op.CMP_EQ, subj, st_,
                            E.ExprSubscript(value=arr, slice=E.ExprConstant(value=k)))
            terms.append(eq if pred is None else _op(ctx, Op.AND, pred, eq))
    if not terms:
        return _const(ctx, 0)
    acc = terms[0]
    for t in terms[1:]:
        acc = _op(ctx, Op.OR, acc, t)
    return acc


def _path_name(e) -> str:
    """``a`` / ``s.a`` / ``m[1]`` for an error message."""
    if isinstance(e, E.ExprAttribute):
        return e.attr if isinstance(e.value, E.TypeExprRefSelf) else \
            f"{_path_name(e.value)}.{e.attr}"
    if isinstance(e, E.ExprSubscript):
        k = e.slice.value if isinstance(e.slice, E.ExprConstant) else "?"
        return f"{_path_name(e.value)}[{k}]"
    return getattr(e, "name", _cn(e))


def _dyn_index(ctx: CoroCtx, e):
    """``(subscript, name, count, kind)`` for the subscript nearest the root
    of *e*'s path that indexes a component or data array with a run-time
    index."""
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
        if isinstance(x, E.ExprAttribute):
            t = _ref_var(ctx, x.value)
            if t is not None:
                cands = _ref_targets(ctx, t, x)
                return x.value, _path_name(x.value), len(cands), "reference", cands
        if isinstance(x, E.ExprSubscript) and _const_index(ctx, x.slice) is None:
            arr = _comp_array(ctx, x.value)
            if arr is not None:
                return x, arr[1], arr[2], "component array"
            n = _array_len(ctx, x.value)
            if n is not None:
                return x, _path_name(x.value), n, "array"
    return None


def _subst(e, node, k: int):
    """*e* with subscript *node*'s index replaced by the constant *k*."""
    if e is node:
        if isinstance(k, CompCtx):
            return _InstExpr(k, _loc(e))
        return dc.replace(e, slice=E.ExprConstant(value=k))
    if isinstance(e, E.ExprCall):
        return dc.replace(e, func=_subst(e.func, node, k))
    if isinstance(e, (E.ExprAttribute, E.ExprSubscript)):
        return dc.replace(e, value=_subst(e.value, node, k))
    return e


def _dispatch(ctx: CoroCtx, d, emit_k) -> None:
    """Evaluate the index once, then run ``emit_k(k)`` for the element it
    selects; an index out of bounds is a run-time error (20.7.4)."""
    if d[3] == "reference":
        node, name, n, kind, cands = d
        j, _ = ev(ctx, node)
        ends = []
        for c in cands:
            brz = ctx.emit(Op.BRZ, (_op(ctx, Op.CMP_EQ, j,
                                        _const(ctx, c.inst + _ref_origin(ctx))), 0))
            emit_k(c)
            ends.append(ctx.emit(Op.BR, (0,)))
            _patch(ctx, brz, len(ctx.code))
        _runtime_error(ctx, f"component reference {name!r} is null")
        for at in ends:
            _patch(ctx, at, len(ctx.code))
        return
    node, name, n, kind = d
    j, jt = ev(ctx, node.slice)
    if jt.is_bool or jt.kind in ("string", "enum"):
        raise PssSemanticError(f"{'an' if kind[0] == 'a' else 'a'} {kind} index "
                               f"must be an integer",
                               loc=_loc(node))
    ends = []
    for k in range(n):
        brz = ctx.emit(Op.BRZ, (_op(ctx, Op.CMP_EQ, j, _const(ctx, k)), 0))
        emit_k(k)
        ends.append(ctx.emit(Op.BR, (0,)))
        _patch(ctx, brz, len(ctx.code))
    _runtime_error(ctx, f"index out of bounds of {kind} {name!r} [{n}]")
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
        if _lookup(ctx, name) is not None:
            return None
    else:
        if isinstance(e.value, E.TypeExprRefSelf) and _this_lookup(ctx, e.attr) is not None:
            return None
        base, name = _comp_inst(ctx, e.value), e.attr
    st = proc_state(ctx)
    if base is None and not _comp_code(st) and name not in st.fields \
            and name not in st.struct_fields \
            and isinstance(getattr(e, "value", None), (type(None), E.TypeExprRefSelf)):
        # Action code: a bare name that is not the action's own is a static
        # member of its component (9.1.4.1 f; the front end refuses
        # ``comp.`` for one).
        base = st.action_comp
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
        v = _lookup(ctx, name) if isinstance(e, E.ExprRefUnresolved) else _this_lookup(ctx, name)
        if v is not None:
            return _Place(v.type, v.locs) if v.locs is not None else None
        st = proc_state(ctx)
        if _in_comp_fn(st) or name not in st.struct_fields:
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
        if not _in_comp_fn(st) and name in st.struct_fields:
            return st.struct_fields[name]
    if isinstance(e, E.ExprSubscript):
        base = _place(ctx, e.value)
        if base is None:
            return None
        t, i = _element(ctx, base, e)
        if isinstance(t, StructT):
            return _Place(t, base.locs[i:i + len(t.leaves)])
    return None


def _element(ctx: CoroCtx, base: _Place, e):
    """Element ``e.slice`` of data array place *base* (F6.1a: a fixed-size
    array is laid out like a struct whose fields are its indices), as
    :func:`_member` gives a field."""
    lv = base.type.leaves
    if not lv or not lv[0][0][0].isdigit():
        raise LoweringError(f"{_cn(e.value)} is not an array", loc=_loc(e))
    i = _const_index(ctx, e.slice)
    if i is None:
        raise LoweringError("an array index bc cannot compute at lowering "
                            "(a run-time index) is not supported by bc yet",
                            loc=_loc(e))
    sub = base.type.sub(str(i))
    if sub is None:
        n = len({p[0] for p, _ in lv})
        raise LoweringError(f"index {i} is out of bounds of an array of {n} "
                            f"elements", loc=_loc(e))
    return sub


def _of_call(e) -> bool:
    """``f().x`` / ``r.read().done.y``: a member of a call's struct result."""
    if not isinstance(e, E.ExprAttribute):
        return False
    while isinstance(e, E.ExprAttribute):
        e = e.value
    return isinstance(e, E.ExprCall)


def _call_result_member(e, t: T):
    """The member path from the call under *e* down to *e*, checked against
    the call's struct type *t*: (call, [(member type, leaf index) ...])."""
    path = []
    while isinstance(e, E.ExprAttribute):
        path.append(e)
        e = e.value
    if not isinstance(t, StructT):
        raise LoweringError(f"{_cn(e)} returns no struct, so it has no member "
                            f".{path[-1].attr}", loc=_loc(path[-1]))
    i0 = 0
    for m in reversed(path):
        if not isinstance(t, StructT):
            raise LoweringError(f"member .{m.attr} of a value that is not a struct",
                                loc=_loc(m))
        sub = t.sub(m.attr)
        if sub is None:
            raise LoweringError(f"struct {t.name or ''} has no field {m.attr!r}",
                                loc=_loc(m))
        t, i = sub
        i0 += i
    return e, t, i0


def _call_member_type(ctx: CoroCtx, e) -> T:
    call = e
    while isinstance(call, E.ExprAttribute):
        call = call.value
    return _call_result_member(e, _call_type(ctx, call))[1]


def _call_member(ctx: CoroCtx, e) -> Tuple[int, T]:
    """Read member *e* of a call's struct result: the call is made into its
    result temporary, and the leaf read from there."""
    call = e
    while isinstance(call, E.ExprAttribute):
        call = call.value
    _, t, i = _call_result_member(e, _call_type(ctx, call))
    if isinstance(t, StructT):
        _not_scalar(t, e)
    p = _struct_value(ctx, call)
    _not_opaque(t, e)
    return _load_loc(ctx, p.locs[i]), t


def _member(base: _Place, e):
    sub = base.type.sub(e.attr)
    if sub is None:
        raise LoweringError(f"struct {base.type.name or ''} has no field "
                            f"{e.attr!r}", loc=_loc(e))
    return sub


def _is_nullfilehandle(ctx: CoroCtx, e) -> bool:
    """std_pkg ``nullfilehandle`` (24.2): the handle 0 no file has. Unqualified,
    it reaches bc as ``self.nullfilehandle`` when no field has that name."""
    if isinstance(e, E.ExprRefUnresolved):
        return e.name in ("nullfilehandle", "std_pkg::nullfilehandle")
    return (isinstance(e, E.ExprAttribute) and e.attr == "nullfilehandle"
            and isinstance(e.value, E.TypeExprRefSelf)
            and _lookup(ctx, e.attr) is None
            and e.attr not in proc_state(ctx).fields)


def _resolve_name(ctx: CoroCtx, e) -> Tuple[str, T, object]:
    """``self.x`` / ``x`` / ``s.f`` -> ("local", type, slot-name),
    ("field", type, slot) or ("comp", type, slot)."""
    cd = _comp_data(ctx, e)
    if cd is not None:
        if cd[0] == "place":
            _not_scalar(cd[1].type, e)
        return "comp", cd[1], cd[2]
    if isinstance(e, E.ExprSubscript):
        base = _place(ctx, e.value)
        if base is None:
            raise LoweringError("an array element is not supported by bc here",
                                loc=_loc(e))
        t, i = _element(ctx, base, e)
        if isinstance(t, StructT):
            _not_scalar(t, e)
        kind, where = base.locs[i]
        return kind, t, where
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
    v = _this_lookup(ctx, name) if isinstance(e, E.ExprAttribute) else _lookup(ctx, name)
    if v is not None:
        if v.locs is not None:
            _not_scalar(v.type, e)
        return "local", v.type, v.slot
    st = proc_state(ctx)
    if _comp_code(st):
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
    if kind == "elem":
        return _list_get(ctx, where)[0]
    r = ctx.new_reg()
    if kind == "local":
        ctx.emit(Op.LD_LOCAL, (r, ctx.local_slot(where)))
    else:
        ctx.emit(_LOAD[kind], (r, where & 0xFFFFFFFF))   # comp: may be < 0
    return r


def _store_loc(ctx: CoroCtx, loc, r: int) -> None:
    kind, where = loc
    if kind == "elem":
        # a nested collection: store its (perhaps new) handle in the outer one
        lt = type_of(ctx, where.value)
        slot = _list_slot(ctx, where.value)
        h, _ = ev(ctx, where.value)
        i = _index_of(ctx, lt, where.slice, owned=True)
        _store_loc(ctx, slot, _lheap(ctx, HEAP_SET, lt, h, i, r))
        return
    if kind == "local":
        ctx.emit(Op.ST_LOCAL, (r, ctx.local_slot(where)))
    else:
        ctx.emit(_STORE[kind], (r, where & 0xFFFFFFFF))


def _struct_value(ctx: CoroCtx, e, want: Optional[StructT] = None) -> _Place:
    """The place holding struct-valued *e*: a struct variable or attribute
    (or a field of one), or a call's return value."""
    if isinstance(e, ExprList) and want is not None:
        return _array_literal(ctx, e, want)
    if isinstance(e, E.ExprStructLiteral):
        if want is None:
            raise LoweringError("a struct literal needs a struct type from where "
                                "it is used", loc=_loc(e))
        return _struct_literal(ctx, e, want)
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


def _array_literal(ctx: CoroCtx, e, want: StructT) -> _Place:
    """``{7, 8, 9}`` as array *want* (F6.1b): each element is evaluated into
    a temporary, so ``w = {w[1], w[0]}`` reads before it writes."""
    keys = []
    for path, _ in want.leaves:
        if path[0] not in keys:
            keys.append(path[0])
    if not keys or not keys[0].isdigit():
        raise LoweringError("an aggregate literal for a struct is not supported "
                            "by bc", loc=_loc(e))
    if len(e.elts) != len(keys):
        raise PssSemanticError(f"an array of {len(keys)} elements is assigned "
                               f"{len(e.elts)} values", loc=_loc(e))
    ps = proc_state(ctx)
    ps.uniq += 1
    place = _struct_locals(ctx, f"$lit${ps.uniq}", want)
    for k, item in zip(keys, e.elts):
        t, i = want.sub(k)
        if isinstance(t, StructT):
            _copy(ctx, _Place(t, place.locs[i:i + len(t.leaves)]),
                  _struct_value(ctx, item, t))
        else:
            _store_leaf(ctx, place.locs[i], t, item)
    return place


def _struct_literal(ctx: CoroCtx, e, want: StructT, dt=None) -> _Place:
    """``{.x = 1, .q = {.z = 2}}`` as struct *want* (F6.12): the members it
    names take its values, the others their declared initial values. It is
    built in a temporary, so ``p = {.x = p.y}`` reads before it writes."""
    from zuspec.ir.core.xf.pss_lower import layout
    types = ctx.lowerer.types
    if dt is None and want.name:
        dt = (types or {}).get(want.name)
    if dt is None:
        # An array element or a member is typed by its leaves alone: find the
        # one struct with that layout.
        def same(t):
            try:
                return value_type(t, types).leaves == want.leaves
            except LoweringError:
                return False
        found = [t for t in (types or {}).values()
                 if layout.is_struct(t, types) and same(t)]
        dt = found[0] if len(found) == 1 else None
    if dt is None:
        raise LoweringError(f"a struct literal for {want.name or 'this struct'} "
                            f"is not supported by bc", loc=_loc(e))
    members = {f.name: f.datatype
               for f in layout.struct_fields(layout.resolve(dt, types), types)}
    ps = proc_state(ctx)
    ps.uniq += 1
    place = _struct_locals(ctx, f"$lit${ps.uniq}", want)
    _init_struct(ctx, place, dt)
    for fe in e.fields:
        if fe.name not in members:
            raise PssSemanticError(f"struct {want.name} has no member "
                                   f"{fe.name!r}", loc=_loc(e))
        t, i = want.sub(fe.name)
        if isinstance(t, StructT):
            sub = _Place(t, place.locs[i:i + len(t.leaves)])
            src = (_struct_literal(ctx, fe.value, t, members[fe.name])
                   if isinstance(fe.value, E.ExprStructLiteral)
                   else _struct_value(ctx, fe.value, t))
            _copy(ctx, sub, src)
        else:
            _store_leaf(ctx, place.locs[i], t, fe.value)
    return place


def _store_leaf(ctx: CoroCtx, loc, t: T, value) -> None:
    """Store *value* to scalar leaf *loc* of type *t* (a collection leaf
    takes a copy and frees its old value)."""
    if t.kind in COLLECTIONS:
        _list_replace(ctx, loc, _list_value(ctx, value, t), t)
    else:
        _store_loc(ctx, loc, _eval_for(ctx, value, t))


def _copy(ctx: CoroCtx, dst: _Place, src: _Place) -> None:
    """8.5.3: a struct assignment copies every field. Every leaf is loaded
    before any is stored, so overlapping places copy correctly."""
    if dst.locs == src.locs:
        return
    regs = [_load_loc(ctx, loc) for loc in src.locs]
    # a collection leaf is copied (spec/heap.md: value semantics); the old
    # value of the destination leaf is freed
    regs = [_lheap(ctx, HEAP_COPY, t, r) if t.kind in COLLECTIONS else r
            for (_, t), r in zip(src.type.leaves, regs)]
    for (_, t), loc, r in zip(dst.type.leaves, dst.locs, regs):
        if t.kind in COLLECTIONS:
            _list_replace(ctx, loc, r, t)
        else:
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
        ra, rb = _load_loc(ctx, la), _load_loc(ctx, lb)
        eq = (_lheap(ctx, HEAP_EQ, t, ra, rb) if t.kind in COLLECTIONS
              else compare_regs(ctx, Op.CMP_EQ, ra, rb, t))
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
        init = getattr(leaf.field, "initial_value", None)
        if t.kind in COLLECTIONS:
            # empty, or a copy of the initializer; a temporary's old value is freed
            _list_replace(ctx, loc, _const(ctx, 0) if init is None
                          else _list_value(ctx, init, t), t)
        else:
            _store_loc(ctx, loc, _default_value(ctx, t, init))


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
    if isinstance(target, E.ExprSubscript) and _is_list(ctx, target.value):
        _list_set(ctx, target, value)
        return
    tt = _target_type(ctx, target)
    if tt.kind in COLLECTIONS:
        slot = _list_slot(ctx, target)
        _list_replace(ctx, slot, _list_value(ctx, value, tt), tt)
        return
    if isinstance(tt, StructT):
        dst = _place(ctx, target)
        _copy(ctx, dst, _struct_value(ctx, value, tt))
        return
    if tt.kind == "compref":
        r = _ref_convert(ctx, value, tt)
        if _comp_data(ctx, target) is not None:
            r = _ref_rebase(ctx, r, target, +1)
        _store(ctx, target, r, tt)
        return
    vt = type_of(ctx, value)
    if (tt.is_bool or vt.is_bool or tt.kind == "string" or vt.kind == "string"
            or tt.is_float or vt.is_float):
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
    return name in ("message", "std_pkg::message") or name in _FAIL_BUILTINS


#: std_pkg ``error()`` / ``fatal()``: a message that stops the run.
#: ``print()`` is a message with no verbosity (always printed) that goes on.
_FAIL_BUILTINS = {"error": BUILTIN_ERROR_MSG, "std_pkg::error": BUILTIN_ERROR_MSG,
                  "fatal": BUILTIN_FATAL_MSG, "std_pkg::fatal": BUILTIN_FATAL_MSG,
                  "print": BUILTIN_PRINT, "std_pkg::print": BUILTIN_PRINT}


def _callee(ctx: CoroCtx, func) -> Tuple[Optional[str], Optional[str], Optional[CompCtx]]:
    """``(name, scope, instance)`` of a call target: the component instance a
    component function runs in (its ``self``), None for a package function
    or an import."""
    name, scope = _call_name(func)
    if scope == "self":
        if _comp_code(proc_state(ctx)):
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
        fn = comps.function(comp.type_qname, name, fns)
        # A package function called as a method's receiver (``f(x).size()``)
        # reaches here as ``self.f``: the component has none of that name.
        return fn if fn is not None or scope != "self" else fns.get(name)
    if comp is None and scope in ("self", "comp"):
        # No instance (no component tree): the action's component.
        st = proc_state(ctx)
        return fns.get(f"{st.component}::{name}") if st.component else None
    return None


def _action_function(ctx: CoroCtx, name: str, scope: Optional[str]):
    """``self.f()`` in action code, when the action declares (or inherits)
    function ``f``: ``ScenarioModule.functions`` keys it ``<action>::f``."""
    st = proc_state(ctx)
    if scope != "self" or _comp_code(st) or not st.action_type:
        return None
    return ctx.lowerer.functions.get(f"{st.action_type}::{name}")


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
    return (scope == "self" and not _comp_code(st)
            and name in ctx.lowerer.imports)


def _string_import(ctx: CoroCtx, decl: dict) -> None:
    """An import returning a string, or a struct with a string field, gets
    heap strings back from the host (host-abi.md "String results")."""
    if "return" in (decl.get("string_at") or []):
        ctx.lowerer.uses_heap = True
    elif "return" in (decl.get("struct_at") or {}):
        ret_t = value_type(decl["struct_at"]["return"], ctx.lowerer.types)
        if any(lt == STRING for _, lt in getattr(ret_t, "leaves", ())):
            ctx.lowerer.uses_heap = True


def _import_ret_t(rt) -> T:
    """The type of an import's scalar result: ``(width, signed[, "float" |
    "string"])``; a float comes back as its binary64 bits (F6.6f2), a string
    as a heap string handle (F6.3b2)."""
    if len(rt) > 2 and rt[2] == "float":
        return FLOAT64
    if len(rt) > 2 and rt[2] == "string":
        return STRING
    return T(min(int(rt[0] or 32), 64), bool(rt[1]))


def _call_type(ctx: CoroCtx, e) -> T:
    f = e.func
    if not e.args and isinstance(f, E.ExprAttribute) and f.attr in ("size", "sum") \
            and _array_len(ctx, f.value) is not None:
        return I32
    lm = _list_method(ctx, e, type_only=True)
    if lm is not None:
        return lm[1]
    sm = _string_method(ctx, e, type_only=True)
    if sm is not None:
        return sm[1]
    if _is_format_string(ctx, e):
        return STRING
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
    if _reg_kind(ctx, name, comp) is not None:
        return _reg_type(ctx, name, comp, e)
    fn = (_action_function(ctx, name, scope)
          or (None if _import_first(ctx, name, scope)
              else _lookup_function(ctx, name, scope, comp)))
    decl = ctx.lowerer.imports.get(name) if scope == "self" and fn is None else None
    if fn is None and decl is None:
        lt = _library_type(ctx, name, scope, comp)
        if lt is not None:
            return lt
    if decl is not None:
        if "return" in (decl.get("struct_at") or {}):
            return value_type(decl["struct_at"]["return"], ctx.lowerer.types)
        rt = decl.get("ret_type")
        if rt is None:
            raise PssSemanticError(f"void import {name!r} has no value", loc=_loc(e))
        return _import_ret_t(rt)
    if fn is None:
        raise LoweringError(f"call to unknown function {name!r}", loc=_loc(e))
    if fn.returns is None:
        raise PssSemanticError(f"void function {name!r} has no value", loc=_loc(e))
    return value_type(fn.returns, ctx.lowerer.types)


def _call(ctx: CoroCtx, e) -> Tuple[Optional[int], Optional[T]]:
    d = _dyn_index(ctx, e)
    if d is not None:
        return _call_dispatch(ctx, e, d)
    am = _array_method(ctx, e)
    if am is not None:
        return am
    lm = _list_method(ctx, e)
    if lm is not None:
        return lm
    sm = _string_method(ctx, e)
    if sm is not None:
        return sm
    if _is_format_string(ctx, e):
        return _format_string(ctx, e), STRING
    ch = _channel(ctx, e.func)
    if ch is not None:
        return _channel_call(ctx, ch, e)
    name, scope, comp = _callee(ctx, e.func)
    if name is None:
        _no_callee(e)
    if _is_message(name):
        _message(ctx, e, _FAIL_BUILTINS.get(name))
        return None, None
    if _reg_kind(ctx, name, comp) is not None:
        return _reg_call(ctx, name, comp, e)
    afn = _action_function(ctx, name, scope)
    if afn is not None:
        return _inline(ctx, afn, e, action=True)
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
        if isinstance(x.value, E.TypeExprRefSelf) and _this_lookup(ctx, x.attr) is not None:
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
            _chan_mark(ctx, ch, fn, v)
            return None, None
        ok = _op(ctx, Op.CMP_NE, _load_loc(ctx, ("comp", ch.count)),
                 _const(ctx, len(ch.bufs)))
        brz = ctx.emit(Op.BRZ, (ok, 0))
        _chan_put(ctx, ch, v)
        _patch(ctx, brz, len(ctx.code))
        _chan_mark(ctx, ch, fn, v, ok)
        return ok, BOOL
    if fn == "get":
        _chan_wait(ctx, ch, full=False)
        out = _chan_take(ctx, ch)
        _chan_mark(ctx, ch, fn, out)
        return out, ch.elem
    # try_get(output T t): t is written only when an element is taken.
    target = e.args[0]
    if not isinstance(target, (E.ExprRefLocal, E.ExprRefField, E.ExprAttribute,
                               E.ExprRefUnresolved, E.ExprSubscript)):
        raise PssSemanticError("try_get()'s argument is an output: it must be "
                               "assignable", loc=_loc(target))
    ok = _op(ctx, Op.CMP_NE, _load_loc(ctx, ("comp", ch.count)), _const(ctx, 0))
    brz = ctx.emit(Op.BRZ, (ok, 0))
    out = _chan_take(ctx, ch)
    _store(ctx, target, out, ch.elem)
    _patch(ctx, brz, len(ctx.code))
    _chan_mark(ctx, ch, fn, out, ok)
    return ok, BOOL


def _chan_mark(ctx: CoroCtx, ch: _Chan, fn: str, val: int, ok: int = -1) -> None:
    """Debug lowering: the channel operation ends at a NOP of its own (no
    branch lands on it), where the VM raises its ``chan`` event."""
    if ctx.lowerer.debug:
        ctx.chans.append((len(ctx.code), ctx.cur_src_ref, ch.name, fn, val, ok))
        ctx.emit(Op.NOP, src_ref=ctx.cur_src_ref)


# --------------------------------------------------------------------------- #
# Address handles, memory and executors (LRM 21.7, 21.13; B-D6)
# --------------------------------------------------------------------------- #

_MEM = {f"read{n}": n for n in (8, 16, 32, 64)}
_MEM.update({f"write{n}": n for n in (8, 16, 32, 64)})
_TRANSPARENT = "addr_reg_pkg::transparent_addr_space_c"


#: the float bit functions (LRM 24.5.3): result types; to_float's arguments
_FLOAT_BITS = {"float_mantissa": T(52, False), "float_exponent": T(11, False),
               "float_sign": T(1, False), "to_float": FLOAT64}


def _fmath(ctx: CoroCtx, name: str, x: int, y: Optional[int] = None) -> int:
    """FMATH *name* of binary64 registers (a 1-argument one repeats *x*)."""
    out = ctx.new_reg()
    ctx.emit(Op.FMATH, (out, x, x if y is None else y),
             imm=FMATH_FNS.index(name))
    return out


def _float_library(ctx: CoroCtx, name: str, e):
    """A call to the float library (24.5.2, 24.5.3): ``(reg, type)``, or None
    if *name* is not one of its functions."""
    name = name[len("std_pkg::"):] if name.startswith("std_pkg::") else name
    if name in FMATH_FNS:
        n = 2 if name in FMATH_BINARY else 1
        if len(e.args) != n:
            raise PssSemanticError(f"{name}() takes {n} float64 argument"
                                   f"{'s' if n > 1 else ''}", loc=_loc(e))
        return _fmath(ctx, name, *(_eval_for(ctx, a, FLOAT64)
                                   for a in e.args)), FLOAT64
    if name not in _FLOAT_BITS:
        return None
    if name == "to_float":                      # (mantissa, exp, sign)
        if len(e.args) != 3:
            raise PssSemanticError("to_float(mantissa, exp, sign)", loc=_loc(e))
        m, x, sg = (_eval_for(ctx, a, w) for a, w in zip(
            e.args, (T(52, False), T(11, False), T(1, False))))
        r = _op(ctx, Op.OR, m, _op(ctx, Op.SHL, x, _const(ctx, 52)))
        return _op(ctx, Op.OR, r, _op(ctx, Op.SHL, sg, _const(ctx, 63))), FLOAT64
    if len(e.args) != 1:
        raise PssSemanticError(f"{name}() takes one float64 argument",
                               loc=_loc(e))
    b = _eval_for(ctx, e.args[0], FLOAT64)
    if name == "float_mantissa":
        return _op(ctx, Op.AND, b, _const(ctx, (1 << 52) - 1)), _FLOAT_BITS[name]
    if name == "float_exponent":
        b = _op(ctx, Op.SHR, b, _const(ctx, 52))
        return _op(ctx, Op.AND, b, _const(ctx, 0x7FF)), _FLOAT_BITS[name]
    return _op(ctx, Op.SHR, b, _const(ctx, 63)), _FLOAT_BITS[name]


#: std_pkg file (24.2) and urandom (24.4) functions the engine serves: result
#: types (None: void)
_FILE_FNS = {"file_open": CHANDLE, "file_close": None, "file_exists": BOOL,
             "file_write": None, "urandom": T(32, False), "urandom_range": T(32, False),
             "file_read": STRING, "file_read_lines": T(64, False, "list", elem=STRING),
             "file_write_lines": None}


def _std_base(name: str) -> str:
    return name[len("std_pkg::"):] if name.startswith("std_pkg::") else name


def _file_library(ctx: CoroCtx, name: str, e):
    """A call to a std_pkg file function: ``(reg, type)``, ``(None, None)``
    for a void one, or None if *name* is not one of them."""
    base = name[len("std_pkg::"):] if name.startswith("std_pkg::") else name
    if base not in _FILE_FNS:
        return None
    if base == "file_write":
        _message(ctx, e, BUILTIN_FILE_WRITE)
        return None, None
    want = {"file_open": (STRING, None), "file_close": (CHANDLE,),
            "file_exists": (STRING,), "urandom": (),
            "urandom_range": (T(32, False), T(32, False)),
            "file_read": (CHANDLE, I32), "file_read_lines": (STRING,),
            "file_write_lines": (STRING, _FILE_FNS["file_read_lines"], None)}[base]
    args = list(e.args)
    if base == "file_read" and len(args) == 1:      # size = -1: to the end
        args.append(E.ExprConstant(value=-1))
    if base == "file_write_lines" and len(args) == 3:
        lt = type_of(ctx, args[1])
        if not (lt.kind == "list" and lt.elem is not None and lt.elem.kind == "string"):
            raise PssSemanticError("file_write_lines(): lines must be a list<string>",
                                   loc=_loc(e))
    if len(args) != len(want):
        raise PssSemanticError(f"{base}() takes {len(want) or 'no'} argument"
                               f"{'s' if len(want) > 1 else ''}", loc=_loc(e))
    regs = [ev(ctx, a)[0] if t is None or t.kind == "list" else _eval_for(ctx, a, t)
            for a, t in zip(args, want)]
    fn = {"file_open": BUILTIN_FILE_OPEN, "file_close": BUILTIN_FILE_CLOSE,
          "file_exists": BUILTIN_FILE_EXISTS, "urandom": BUILTIN_URANDOM,
          "urandom_range": BUILTIN_URANDOM_RANGE, "file_read": BUILTIN_FILE_READ,
          "file_read_lines": BUILTIN_FILE_READ_LINES,
          "file_write_lines": BUILTIN_FILE_WRITE_LINES}[base]
    rt = _FILE_FNS[base]
    if rt is None:
        emit_import(ctx, base, fn, _VOID, regs, 0, loc=_loc(e))
        return None, None
    rd = ctx.new_reg()
    emit_import(ctx, base, fn, rd, regs, INSTR_F_HAS_RET, loc=_loc(e))
    return rd, rt


def _library_type(ctx: CoroCtx, name: str, scope, comp) -> Optional[T]:
    """The result type of a library function bc implements, or None."""
    base = name[len("std_pkg::"):] if name.startswith("std_pkg::") else name
    if base in _FILE_FNS and scope != "path":
        return _FILE_FNS[base]
    if base in FMATH_FNS:
        return FLOAT64
    if base in _FLOAT_BITS:
        return _FLOAT_BITS[base]
    if name == "executor":
        return EXEC
    if scope == "self":
        if name in _MEM:
            return T(_MEM[name], False) if name.startswith("read") else None
        if name in ("read_struct", "write_struct"):
            return None
        if name in ("make_handle_from_handle", "make_handle_from_claim"):
            return CHANDLE
        if name in ("addr_value", "addr_value_solve"):
            return U64
        if name == "addr_value_abs":
            return BOOL
        if name == "get_tag":
            return STRING
    if scope == "path" and name in ("add_region", "add_nonallocatable_region"):
        return CHANDLE
    return None


def _library_call(ctx: CoroCtx, name: str, scope, comp, e):
    """A library function bc implements: ``(reg, type)``, ``(None, None)``
    for a void one, or None if *name* is not one."""
    fl = _float_library(ctx, name, e)
    if fl is not None:
        return fl
    if scope != "path":
        fl = _file_library(ctx, name, e)
        if fl is not None:
            return fl
    if name == "executor":
        if e.args:
            raise PssSemanticError("executor() takes no arguments", loc=_loc(e))
        r = ctx.new_reg()
        ctx.emit(Op.EXEC_GET, (r,), src_ref=ctx.cur_src_ref)
        return r, EXEC
    if scope == "self" and name == "set_executor":
        st = proc_state(ctx)
        if st.comp_self is None or st.fns:
            raise LoweringError("set_executor() outside a component's init block "
                                "is not supported by bc (its executors are "
                                "resolved when the model is lowered)", loc=_loc(e))
        return None, None                       # resolved ahead: _executors
    if (name == "add_executor" and comp is not None
            and ctx.lowerer.comps.is_a(comp.type_qname, _GROUP_BASE)):
        st = proc_state(ctx)
        if st.comp_self is None or st.fns:
            raise PssSemanticError("add_executor() may only be called in exec "
                                   "init_down and init_up blocks (24.6.1.2.1)",
                                   loc=_loc(e))
        return None, None                       # resolved ahead: exec_groups
    if scope == "self" and name == "make_handle_from_handle":
        if not 2 <= len(e.args) <= 3:
            raise PssSemanticError("make_handle_from_handle(handle, offset[, sub])",
                                   loc=_loc(e))
        h = _eval_for(ctx, e.args[0], CHANDLE)
        off = _eval_for(ctx, e.args[1], U64)
        return _op(ctx, Op.ADD, h, off), CHANDLE
    if scope == "self" and name == "make_handle_from_claim":
        return _handle_from_claim(ctx, e), CHANDLE
    if scope == "self" and (name in _MEM or name == "addr_value"):
        return _mem_call(ctx, name, e)
    if scope == "self" and name in ("read_struct", "write_struct"):
        return _struct_mem(ctx, name, e)
    if scope == "self" and name in ("addr_value_solve", "addr_value_abs"):
        return _addr_value_solve(ctx, name, e)
    if scope == "self" and name == "get_tag":
        return _get_tag(ctx, e), STRING
    if scope == "path" and name in ("add_region", "add_nonallocatable_region"):
        lay_t = comp.type_qname if comp is not None else None
        if lay_t is None:
            raise LoweringError(f"{name}() of an unknown address space is not "
                                f"supported by bc", loc=_loc(e))
        if len(e.args) != 1:
            raise PssSemanticError(f"{name}() takes a region", loc=_loc(e))
        if not ctx.lowerer.comps.is_a(lay_t, _TRANSPARENT):
            return _opaque_region(ctx, comp, e), CHANDLE
        region = _struct_value(ctx, e.args[0])
        sub = region.type.sub("addr")
        if sub is None:
            raise PssSemanticError(f"{name}() of a transparent space takes a "
                                   f"transparent_addr_region_s", loc=_loc(e))
        return _load_loc(ctx, region.locs[sub[1]]), CHANDLE
    return None


def _handle_from_claim(ctx: CoroCtx, e):
    """``make_handle_from_claim(claim[, offset[, sub]])`` (24.10.4.1): the
    claim's placed address plus *offset*. A transparent claim's ``addr`` is
    a field the solve placed (F7.4a); a contiguous claim's is the hidden
    ``$addr`` the run writes when it places the claim (F7.4b, activation.py
    ``_addr_claim``)."""
    if not 1 <= len(e.args) <= 3:
        raise PssSemanticError("make_handle_from_claim(claim[, offset[, sub]])",
                               loc=_loc(e))
    claim = _struct_value(ctx, e.args[0])
    sub = claim.type.sub("addr") or claim.type.sub("$addr")
    if sub is None:
        raise LoweringError("make_handle_from_claim() of a claim not bound to "
                            "an address space is not supported by bc",
                            loc=_loc(e))
    h = _load_loc(ctx, claim.locs[sub[1]])
    if len(e.args) < 2:
        return h
    return _op(ctx, Op.ADD, h, _eval_for(ctx, e.args[1], U64))


def _addr_value_solve(ctx: CoroCtx, name: str, e):
    """``addr_value_solve`` / ``addr_value_abs`` (24.10.6-7): a transparent
    handle is its absolute address; a handle in a contiguous space is opaque
    (``_opaque_region``: at or above ``1 << 48``), so ``addr_value_solve``
    gives the handle itself and ``addr_value_abs`` false."""
    if not 1 <= len(e.args) <= 2:
        raise PssSemanticError(f"{name}(handle[, mem_access_desc])", loc=_loc(e))
    h = _eval_for(ctx, e.args[0], CHANDLE)
    if name == "addr_value_solve":
        return h, U64
    hi = _op(ctx, Op.SHR, h, _const(ctx, 48))
    return _op(ctx, Op.CMP_EQ, hi, _const(ctx, 0)), BOOL


def _get_tag(ctx: CoroCtx, e):
    """``get_tag(handle)`` (24.10.8): the tag of the region the handle lies
    in, "" in none. Regions are fixed at build time; the front end gives
    their table (``lower_scenario`` ``regions``), so this is a chain of
    tests: ``addr <= h < addr + size`` for a transparent region, and the
    opaque base's top bits (``_opaque_region``: space instance and k) for
    the k-th region of a contiguous space."""
    if len(e.args) != 1:
        raise PssSemanticError("get_tag(handle)", loc=_loc(e))
    regions = ctx.lowerer.regions
    if regions is None:
        raise LoweringError("get_tag() needs the build's region table, which "
                            "this front end does not give bc", loc=_loc(e))
    h = _eval_for(ctx, e.args[0], CHANDLE)
    out = ctx.new_reg()
    ctx.emit(Op.MOV, (out, _const(ctx, ctx.lowerer.intern_string(""))))
    tree = ctx.lowerer.comp_tree
    paths = [i.path for i in tree.instances] if tree is not None else []
    for r in regions:
        if not r.get("tag"):
            continue
        if r.get("addr") is not None:
            lo = _const(ctx, r["addr"])
            hit = _op(ctx, Op.AND, _op(ctx, Op.CMP_LE, lo, h),
                      _op(ctx, Op.CMP_LT, h, _const(ctx, r["addr"] + r["size"])))
        elif r.get("path") in paths:
            key = ((paths.index(r["path"]) + 1) << 8) | r["k"]
            hit = _op(ctx, Op.CMP_EQ, _op(ctx, Op.SHR, h, _const(ctx, 40)),
                      _const(ctx, key))
        else:
            continue
        brz = ctx.emit(Op.BRZ, (hit, 0))
        ctx.emit(Op.MOV, (out, _const(ctx, ctx.lowerer.intern_string(r["tag"]))))
        _patch(ctx, brz, len(ctx.code))
    return out


def _opaque_region(ctx: CoroCtx, comp, e):
    """The handle ``add_region`` of a contiguous space returns: an opaque
    base, ``(instance + 1) << 48 | k << 40`` for the space's k-th region,
    so regions never overlap; the claims placed in the region lie in
    ``[base, base + size)`` (address-space-bc-design.md §7). Recorded on the
    lowerer for the host's memory map."""
    if proc_state(ctx).comp_self is None:
        raise LoweringError("add_region() of a contiguous address space outside "
                            "a component's init block is not supported by bc",
                            loc=_loc(e))
    regs = ctx.lowerer.opaque_regions
    k = sum(1 for r in regs if r["inst"] == comp.inst)
    base = ((comp.inst + 1) << 48) | (k << 40)
    loc = _loc(e)
    regs.append({"inst": comp.inst, "line": getattr(loc, "line", 0),
                 "base": base})
    return _const(ctx, base)


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


def _split(nbytes: int) -> List[Tuple[int, int]]:
    """``(byte offset, bits)`` accesses covering *nbytes* bytes: one if it is
    1/2/4/8, else the fewest power-of-two pieces, largest (lowest) first --
    the accesses pssc's op-model targets make (packed_image ``_split``)."""
    out, off = [], 0
    while off < nbytes:
        w = 8
        while w * 2 <= 64 and off + (w * 2) // 8 <= nbytes:
            w *= 2
        out.append((off, w))
        off += w // 8
    return out


def _packed_base(ctx: CoroCtx, dt) -> Optional[str]:
    """``packed_s`` or ``packed_s_BIG_ENDIAN`` (as ast2ir names the base) if
    struct type *dt* derives from it, directly or through its bases; else
    None."""
    types = ctx.lowerer.types or {}
    for _ in range(64):                     # a base is a ref or the struct
        sup = getattr(dt, "super", None)
        qn = getattr(sup, "ref_name", None) or getattr(sup, "name", None)
        if not qn:
            return None
        if qn.split("::")[-1].startswith("packed_s"):
            return qn.split("::")[-1]
        dt = sup if hasattr(sup, "super") else types.get(qn)
    return None


@dc.dataclass(frozen=True)
class _Piece:
    """Part of a packed struct's image moved as one word by ``_split``
    accesses: *bits* bits at byte *off*; *leaves* are (leaf index, bit in the
    word, packed width). A *be* word is big-endian: its most significant
    byte is at *off*."""
    off: int
    bits: int
    leaves: Tuple[Tuple[int, int, int], ...]
    be: bool = False


def _struct_plan(ctx: CoroCtx, name: str, st: StructT, e) -> List[_Piece]:
    """The words ``read_struct``/``write_struct`` move struct *st* in, by
    pssc's rule (packed_image ``access_plan``), so bc and the op-model
    targets make the same accesses: a struct of at most 64 bits is one word;
    in a wider one each member is moved on its own -- a scalar of 8/16/32/64
    bits at a byte boundary of its own alignment, a nested struct as a word
    (at most 64 bits) or member by member. A bool is one bit and an enum as
    wide as its base type (21.13.1). Unlike pssc, a word may hold a nested
    struct (bc packs leaves, not a C union). In a packed_s<BIG_ENDIAN>
    struct fields go from the word's most significant bit, don't-care bits
    at the least significant end, and its most significant byte comes first
    in memory (24.10.1, Figure 57); one image has one endianness."""
    from zuspec.ir.core.xf.pss_lower import layout
    types = ctx.lowerer.types or {}
    what = st.name or "struct"
    index = {p: i for i, (p, _) in enumerate(st.leaves)}

    def fail(msg):
        raise LoweringError(f"{name}() of {what}: {msg}", loc=_loc(e))

    def check(sdt, path):
        base = _packed_base(ctx, sdt)
        if base is None:
            fail(f"'{'.'.join(path) or what}' is not a packed_s struct, so it "
                 f"has no memory image (24.10.9.4)")
        if path and (base == "packed_s_BIG_ENDIAN") != be:
            e_in, e_out = ("BIG", "LITTLE") if not be else ("LITTLE", "BIG")
            fail(f"'{'.'.join(path)}' is packed_s<{e_in}_ENDIAN> inside a "
                 f"packed_s<{e_out}_ENDIAN> struct; bc lays out an image in one "
                 f"byte order, so declare both with the same endianness")

    def members(sdt, path):
        """(path, packed width, nested struct type or None) of each field."""
        out = []
        for f in layout.struct_fields(sdt, types):
            dt, p = f.datatype, path + (f.name,)
            cn = type(dt).__name__
            if cn == "DataTypeBool" or (cn == "DataTypeInt"
                                        and getattr(dt, "name", None) == "bool"):
                out.append((p, 1, None))
            elif cn == "DataTypeEnum":
                w = int(getattr(dt, "width", 0) or 0)
                if w <= 0:
                    fail(f"field {'.'.join(p)} is an enum with no base type, so "
                         f"it has no width in a packed struct; declare the "
                         f"enum as 'enum E : bit[N]'")
                out.append((p, w, None))
            elif cn == "DataTypeInt":
                out.append((p, int(getattr(dt, "bits", None) or 32), None))
            elif layout.is_struct(dt, types) and not layout.is_array(dt, types):
                sub = layout.resolve(dt, types)
                check(sub, p)
                out.append((p, sum(w for _, w, _ in members(sub, p)), sub))
            else:
                fail(f"field {'.'.join(p)} is not a scalar or a packed_s "
                     f"struct, so it has no place in a packed image (21.13.1)")
        return out

    def leaves(sdt, path, pos):
        out = []
        for p, w, sub in members(sdt, path):
            if sub is not None:
                out += leaves(sub, p, pos)
            elif p in index:
                out.append((index[p], pos, w))
            pos += w
        return out

    def word(sdt, path, off, bits):
        for o, w in _split((bits + 7) // 8):
            if (off + o) % (w // 8):
                fail(f"'{'.'.join(path) or what}' would be a {w}-bit access at "
                     f"byte offset {off + o} of the image, which is not "
                     f"{w // 8}-byte aligned; place it on a {w // 8}-byte "
                     f"boundary")
        ls = leaves(sdt, path, 0)
        if be:                                  # from the MSB of whole bytes
            top = (bits + 7) // 8 * 8
            ls = [(i, top - pos - w, w) for i, pos, w in ls]
        return _Piece(off, bits, tuple(ls), be)

    def by_member(sdt, path, off):
        out, pos = [], 0
        for p, w, sub in members(sdt, path):
            if pos % 8:
                fail(f"'{'.'.join(p)}' starts at bit {pos}, not on a byte; in a "
                     f"struct wider than 64 bits every member is moved on its "
                     f"own, so group the narrow fields into a packed_s struct "
                     f"of 8, 16, 32 or 64 bits")
            o = off + pos // 8
            if sub is None:
                if w not in (8, 16, 32, 64):
                    fail(f"'{'.'.join(p)}' is {w} bits; in a struct wider than "
                         f"64 bits every scalar member must be 8, 16, 32 or 64 "
                         f"bits, so group it with its neighbours into a "
                         f"packed_s struct")
                if o % (w // 8):
                    word(sdt, p, o, w)          # refuses, located at the member
                out.append(_Piece(o, w, ((index[p], 0, w),), be))
            elif w <= 64:
                out.append(word(sub, p, o, w))
            else:
                out += by_member(sub, p, o)
            pos += w
        return out

    dt = types.get(st.name) if st.name else None
    if dt is None:
        fail("not a packed_s struct, so it has no memory image (24.10.9.4)")
    be = _packed_base(ctx, dt) == "packed_s_BIG_ENDIAN"
    check(dt, ())
    bits = sum(w for _, w, _ in members(dt, ()))
    if bits == 0:
        fail("the struct has no fields")
    return [word(dt, (), 0, bits)] if bits <= 64 else by_member(dt, (), 0)


def _bswap(ctx: CoroCtx, r: int, nbytes: int) -> int:
    """Register *r* with its low *nbytes* bytes in reverse order."""
    if nbytes == 1:
        return r
    out = _const(ctx, 0)
    for i in range(nbytes):
        b = _op(ctx, Op.SHR, r, _const(ctx, 8 * i)) if i else r
        b = _op(ctx, Op.AND, b, _const(ctx, 0xFF))
        j = 8 * (nbytes - 1 - i)
        out = _op(ctx, Op.OR, out, _op(ctx, Op.SHL, b, _const(ctx, j)) if j else b)
    return out


def _struct_mem(ctx: CoroCtx, name: str, e):
    """``read_struct(h, s)`` / ``write_struct(h, s)`` (LRM 24.10.9.4): packed
    struct *s* at *h*, moved in the words of ``_struct_plan``; a word of
    8/16/32/64 bits is one access, else the fewest power-of-two accesses,
    lowest first. Accesses are little-endian, so a big-endian word is
    byte-reversed before a write and after a read."""
    if len(e.args) != 2:
        raise PssSemanticError(f"{name}() takes a handle and a packed struct",
                               loc=_loc(e))
    h = _eval_for(ctx, e.args[0], CHANDLE)
    place = _struct_value(ctx, e.args[1])
    plan = _struct_plan(ctx, name, place.type, e)

    def addr(off):
        return _op(ctx, Op.ADD, h, _const(ctx, off)) if off else h

    for pc in plan:
        split = _split((pc.bits + 7) // 8)
        if name == "write_struct":
            raw = _const(ctx, 0)
            for i, pos, w in pc.leaves:
                v = _op(ctx, Op.AND, _load_loc(ctx, place.locs[i]),
                        _const(ctx, (1 << w) - 1))
                raw = _op(ctx, Op.OR, raw,
                          _op(ctx, Op.SHL, v, _const(ctx, pos)) if pos else v)
            if pc.be:
                raw = _bswap(ctx, raw, (pc.bits + 7) // 8)
            for off, w in split:
                v = _op(ctx, Op.SHR, raw, _const(ctx, off * 8)) if off else raw
                v = _op(ctx, Op.AND, v, _const(ctx, (1 << w) - 1))
                ctx.emit(Op.IMPORT, (BUILTIN_WRITE[w], _VOID, addr(pc.off + off), v))
            continue
        raw = _const(ctx, 0)
        for off, w in split:
            rd = ctx.new_reg()
            ctx.emit(Op.IMPORT, (BUILTIN_READ[w], rd, addr(pc.off + off)),
                     flags=INSTR_F_HAS_RET)
            raw = _op(ctx, Op.OR, raw, _op(ctx, Op.SHL, rd, _const(ctx, off * 8))
                      if off else rd)
        if pc.be:
            raw = _bswap(ctx, raw, (pc.bits + 7) // 8)
        for i, pos, w in pc.leaves:
            t = place.type.leaves[i][1]
            v = _op(ctx, Op.SHR, raw, _const(ctx, pos)) if pos else raw
            v = _op(ctx, Op.AND, v, _const(ctx, (1 << w) - 1))
            _store_loc(ctx, place.locs[i],
                       normalize(ctx, v, t) if t.kind == "int" and t.signed else v)
    return None, None


# Registers and register groups (LRM 21.14)
# --------------------------------------------------------------------------- #
# A register is accessed at its group's handle (the group's ``$handle`` slot,
# ir-core comp_tree) plus its offset in the group (``_reg_offset``), with
# read<SZ>/write<SZ>; a register struct is packed as ``_struct_plan`` packs
# a word (LSB first, or from the MSB of whole bytes for BIG_ENDIAN).

_REG_FNS = {"read": None, "read_val": None, "get_handle": CHANDLE,
            "write": None, "write_val": None, "write_masked": None,
            "write_val_masked": None, "write_field": None}
_GROUP_FNS = {"set_handle": None, "get_handle": CHANDLE}


def _reg_kind(ctx: CoroCtx, name: str, comp: Optional[CompCtx]) -> Optional[str]:
    """"reg" / "group" if *name* is a register (group) function bc implements
    for instance *comp*, else None."""
    if comp is None or getattr(ctx.lowerer, "comps", None) is None:
        return None
    dt = ctx.lowerer.comps.types.get(comp.type_qname)
    if isinstance(dt, DataTypeRegister) and name in _REG_FNS:
        return "reg"
    if isinstance(dt, DataTypeRegisterGroup) and name in _GROUP_FNS:
        return "group"
    return None


def _reg_types(ctx: CoroCtx, comp: CompCtx, e):
    """(R's value type, the register's ``bit[SZ]``) of register *comp*."""
    dt = ctx.lowerer.comps.types[comp.type_qname]
    sz = int(dt.size_bits or 0)
    if sz not in (8, 16, 32, 64):
        raise LoweringError(f"register {comp.type_qname} of {sz} bits: bc accesses "
                            f"registers of 8, 16, 32 or 64 bits", loc=_loc(e))
    return value_type(dt.register_value_type, ctx.lowerer.types), T(sz, False)


def _reg_type(ctx: CoroCtx, name: str, comp: CompCtx, e):
    if name == "get_handle":
        return CHANDLE
    if name in ("read", "read_val"):
        rt, szt = _reg_types(ctx, comp, e)
        return rt if name == "read" else szt
    raise PssSemanticError(f"void function {name!r} has no value", loc=_loc(e))


def _group_handle_loc(ctx: CoroCtx, group: CompCtx):
    return ("comp", group.slot + _comp_layout(ctx, group).slot_of(HANDLE))


def _reg_offset(ctx: CoroCtx, gdt, name: str, index: Optional[int], e) -> int:
    """Byte offset of instance ``name`` (``name[index]``) in group type *gdt*:
    the lowerer's ``reg_offset`` (the group's own offset functions), else the
    dense ``offset_map``, which has registers only."""
    fn = ctx.lowerer.reg_offset
    if fn is not None:
        try:
            return int(fn(gdt, name, index))
        except Exception as ex:
            raise LoweringError(str(ex), loc=_loc(e)) from ex
    off = (getattr(gdt, "offset_map", None) or {}).get(name)
    if off is None or index is not None:
        raise LoweringError(f"register group {gdt.name}: bc places {name!r} only "
                            f"with the front end's offset functions (reg_offset)",
                            loc=_loc(e))
    return off


def _reg_addr(ctx: CoroCtx, e) -> int:
    """The address of the register ``grp.reg`` in call *e* (``grp.reg.f()``),
    or of the instance a register reference stands for in one copy of its
    dispatch (``r.f()`` with ``ref reg_sized_c<32> r``)."""
    reg_e = e.func.value
    if isinstance(reg_e, _InstExpr):
        group, name, index = _reg_of_instance(ctx, reg_e.comp, e)
    else:
        group = (_comp_inst(ctx, reg_e.value) if isinstance(reg_e, E.ExprAttribute)
                 else None)
        name, index = getattr(reg_e, "attr", None), None
    gdt = ctx.lowerer.comps.types.get(group.type_qname) if group is not None else None
    if not isinstance(gdt, DataTypeRegisterGroup):
        raise LoweringError("bc accesses a register only as a named instance of "
                            "a register group (grp.reg.read())", loc=_loc(e))
    off = _reg_offset(ctx, gdt, name, index, e)
    return _op(ctx, Op.ADD, _load_loc(ctx, _group_handle_loc(ctx, group)),
               _const(ctx, off))


def _reg_of_instance(ctx: CoroCtx, c: CompCtx, e):
    """``(group, name, index)`` of register instance *c* (frame-relative):
    the group instance holding it and its name there, from the component
    tree. A reference is numbered from the root only in a static frame."""
    f = _ref_frame(ctx)
    if f is None:
        _frame_instance(ctx, e, "a register reached through a reference")
    insts = {i.id: i for i in ctx.lowerer.comp_tree.instances}
    inst = insts.get(c.inst + f.id)
    if inst is None or inst.parent is None:
        raise LoweringError("a register reference refers to no register of a "
                            "group", loc=_loc(e))
    g = insts[inst.parent]
    m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", inst.path.rsplit(".", 1)[-1])
    if m is None:
        raise LoweringError(f"register instance {inst.path!r} has no name bc can "
                            f"place", loc=_loc(e))
    return (CompCtx(g.type_qname, g.base - f.base, g.id - f.id), m.group(1),
            int(m.group(2)) if m.group(2) is not None else None)


def _reg_layout(ctx: CoroCtx, name: str, st: StructT, e):
    """(bits, [(leaf index, lsb, width)]) of register struct *st*: a
    packed_s struct as ``_struct_plan`` lays out one word (bool 1 bit, enum
    its base width, BIG_ENDIAN from the MSB of whole bytes, as pssc
    reg_field_resolve ``_place``); any other struct LSB first at its leaf
    widths."""
    dt = (ctx.lowerer.types or {}).get(st.name) if st.name else None
    if dt is not None and _packed_base(ctx, dt) is not None:
        plan = _struct_plan(ctx, name, st, e)
        bits = sum(pc.bits for pc in plan)
        return bits, (list(plan[0].leaves) if len(plan) == 1 else [])
    out, off = [], 0
    for i, (_, t) in enumerate(st.leaves):
        out.append((i, off, t.width))
        off += t.width
    return off, out


def _pack(ctx: CoroCtx, place: _Place, sz: int, e) -> int:
    """The ``bit[SZ]`` value of packed struct *place* (``_reg_layout``)."""
    bits, lay = _reg_layout(ctx, "write", place.type, e)
    if bits > sz:
        raise PssSemanticError(f"register struct of {bits} bits is wider than the "
                               f"register's {sz}", loc=_loc(e))
    r = _const(ctx, 0)
    for i, pos, w in lay:
        v = _op(ctx, Op.AND, _load_loc(ctx, place.locs[i]), _const(ctx, (1 << w) - 1))
        r = _op(ctx, Op.OR, r, _op(ctx, Op.SHL, v, _const(ctx, pos)))
    return r


def _unpack(ctx: CoroCtx, raw: int, st: StructT, e) -> _Place:
    ps = proc_state(ctx)
    ps.uniq += 1
    place = _struct_locals(ctx, f"$reg${ps.uniq}", st)
    for i, pos, w in _reg_layout(ctx, "read", st, e)[1]:
        t = st.leaves[i][1]
        v = _op(ctx, Op.SHR, raw, _const(ctx, pos)) if pos else raw
        v = _op(ctx, Op.AND, v, _const(ctx, (1 << w) - 1))
        _store_loc(ctx, place.locs[i],
                   normalize(ctx, v, t) if t.kind == "int" and t.signed else v)
    return place


def _reg_value(ctx: CoroCtx, arg, rt, szt: T, e) -> int:
    """Argument *arg* of type R as the register's ``bit[SZ]`` value."""
    if isinstance(rt, StructT):
        return _pack(ctx, _struct_value(ctx, arg, rt), szt.width, e)
    return _eval_for(ctx, arg, szt)


def _field_bits(ctx: CoroCtx, rt, arg, e) -> Tuple[int, int]:
    """(offset, width) of the field a ``write_field`` names."""
    name = getattr(arg, "value", None)
    if not (isinstance(arg, E.ExprConstant) and isinstance(name, str)):
        raise LoweringError("write_field(): bc needs the field name as a string "
                            "literal", loc=_loc(e))
    if isinstance(rt, StructT):
        for i, pos, w in _reg_layout(ctx, "write_field", rt, e)[1]:
            if rt.leaves[i][0] == (name,):
                return pos, w
    raise PssSemanticError(f"write_field(): register has no field {name!r}",
                           loc=_loc(e))


def _set_handle(ctx: CoroCtx, group: CompCtx, h: int, e) -> None:
    """Bind *group* and, at their offsets, its sub-groups (arrays too)."""
    _store_loc(ctx, _group_handle_loc(ctx, group), h)
    gdt = ctx.lowerer.comps.types[group.type_qname]
    for key, sub in _comp_layout(ctx, group).subs.items():
        if "." in key or not isinstance(ctx.lowerer.comps.types.get(sub.type_qname),
                                        DataTypeRegisterGroup):
            continue
        name, _, idx = key.partition("[")
        off = _reg_offset(ctx, gdt, name, int(idx[:-1]) if idx else None, e)
        _set_handle(ctx, _comp_sub(ctx, group, key),
                    _op(ctx, Op.ADD, h, _const(ctx, off)), e)


def _reg_call(ctx: CoroCtx, name: str, comp: CompCtx, e):
    """A register or register-group function (21.14.1, 21.14.4)."""
    want = {"read": 0, "read_val": 0, "get_handle": 0, "write": 1, "write_val": 1,
            "set_handle": 1, "write_masked": 2, "write_val_masked": 2,
            "write_field": 2}[name]
    if len(e.args) != want:
        raise PssSemanticError(f"{name}() takes {want} argument(s)", loc=_loc(e))
    if _reg_kind(ctx, name, comp) == "group":
        if name == "get_handle":
            return _load_loc(ctx, _group_handle_loc(ctx, comp)), CHANDLE
        _set_handle(ctx, comp, _eval_for(ctx, e.args[0], CHANDLE), e)
        return None, None
    addr = _reg_addr(ctx, e)
    if name == "get_handle":
        return addr, CHANDLE
    rt, szt = _reg_types(ctx, comp, e)
    n = szt.width
    if name in ("read", "read_val"):
        rd = ctx.new_reg()
        _reg_mark(ctx, comp)
        if ctx.lowerer.debug and e is ctx.discard_call:
            ctx.discards.append(len(ctx.code))
        ctx.emit(Op.IMPORT, (BUILTIN_READ[n], rd, addr), flags=INSTR_F_HAS_RET)
        if name == "read" and isinstance(rt, StructT):
            return _unpack(ctx, rd, rt, e), rt
        return rd, (rt if name == "read" else szt)
    if name == "write":
        v = _reg_value(ctx, e.args[0], rt, szt, e)
    elif name == "write_val":
        v = _eval_for(ctx, e.args[0], szt)
    else:
        if name == "write_field":
            off, w = _field_bits(ctx, rt, e.args[0], e)
            mask = _const(ctx, ((1 << w) - 1) << off)
            val = _op(ctx, Op.SHL, _eval_for(ctx, e.args[1], T(w, False)),
                      _const(ctx, off))
        else:
            ev_ = (_reg_value if name == "write_masked"
                   else lambda c, a, r, s, x: _eval_for(c, a, s))
            mask = ev_(ctx, e.args[0], rt, szt, e)
            val = ev_(ctx, e.args[1], rt, szt, e)
        old = ctx.new_reg()
        _reg_mark(ctx, comp)
        ctx.emit(Op.IMPORT, (BUILTIN_READ[n], old, addr), flags=INSTR_F_HAS_RET)
        keep = _op(ctx, Op.AND, old, _op(ctx, Op.XOR, mask, _const(ctx, (1 << n) - 1)))
        v = _op(ctx, Op.OR, keep, _op(ctx, Op.AND, val, mask))
    _reg_mark(ctx, comp)
    ctx.emit(Op.IMPORT, (BUILTIN_WRITE[n], _VOID, addr, v))
    return None, None


def _reg_mark(ctx: CoroCtx, reg: CompCtx) -> None:
    """Debug lowering: the next instruction, an ``IMPORT``, accesses register
    *reg* (P1.7). Release code is unchanged: the table is not in the image."""
    if ctx.lowerer.debug:
        ctx.regs.append((len(ctx.code), reg.inst))


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


_EXECUTOR_BASE = "executor_pkg::executor_base_c"


def exec_rows(lw) -> list:
    """The image's executor instances (EXEC rows, F5.4a): each component
    instance whose type is an ``executor_base_c``, and any instance a
    ``set_executor`` names, in instance order; a row's index is the executor
    id requests carry."""
    tree, comps = getattr(lw, "comp_tree", None), lw.comps
    if tree is None or comps is None:
        return []
    named = {v[1] for v in (lw.executors or {}).values() if isinstance(v, tuple)}
    return [i for i in tree.instances
            if i.id in named or comps.is_a(i.type_qname, _EXECUTOR_BASE)]


_GROUP_BASE = "executor_pkg::executor_group_c"
_EMPTY_TRAIT = "executor_pkg::empty_executor_trait_s"


def exec_trait(lw, qname: str) -> str:
    """The trait struct type of executor or group type *qname*: the type
    argument of the ``executor_c<T>`` / ``executor_group_c<T>`` it derives
    from (the template's default when it names none)."""
    comps = lw.comps
    for tmpl in (_EXECUTOR_BASE[:-len("base_c")] + "c", _GROUP_BASE):
        for t in reversed(comps.chain(qname)):
            if getattr(t, "template", None) != tmpl:
                continue
            spec = comps.qname_of(t) or ""
            short = spec[len(tmpl) + 1:] if spec.startswith(tmpl + "_") else ""
            if not short:
                return _EMPTY_TRAIT
            hits = [q for q in lw.types if q == short or q.endswith("::" + short)]
            hits = [q for q in hits if "::" in q] or hits
            return hits[0] if len(hits) == 1 else short
        if comps.is_a(qname, tmpl):
            return _EMPTY_TRAIT
    return ""


def exec_groups(lw) -> list:
    """The image's executor groups (24.6.1.2): ``{inst, members}`` per
    ``executor_group_c`` instance, in instance order; *members* are the
    instance ids ``add_executor`` put in it, in call order. A call is
    resolved when the model is lowered: a top-level statement of an
    ``exec init_down`` / ``init_up`` block, or of a ``foreach`` over a
    component array there; any other is an error. Adding an executor twice
    to a group, or one whose trait type is not the group's, is an error."""
    if getattr(lw, "exec_group_map", None) is not None:
        return lw.exec_group_map
    tree, comps = getattr(lw, "comp_tree", None), lw.comps
    if tree is None or comps is None:
        return []
    by_id = {i.id: i for i in tree.instances}
    groups = {i.id: [] for i in tree.instances if comps.is_a(i.type_qname, _GROUP_BASE)}
    for inst in tree.instances:
        lay = comps.get(inst.type_qname)

        def sub(path, stmt):
            if path == "":
                return inst
            s = lay.subs.get(path)
            if s is None:
                raise PssSemanticError(f"add_executor: {path!r} is not a component "
                                       f"instance of {inst.type_qname!r}", loc=_loc(stmt))
            return by_id[inst.id + s.inst]

        def add(gpath, xpath, stmt):
            g, x = sub(gpath, stmt), sub(xpath, stmt)
            if g.id not in groups:
                raise PssSemanticError(f"add_executor: {g.path!r} is not an executor "
                                       f"group", loc=_loc(stmt))
            if not comps.is_a(x.type_qname, _EXECUTOR_BASE):
                raise PssSemanticError(f"add_executor({xpath}): {x.path!r} is not an "
                                       f"executor", loc=_loc(stmt))
            gt, xt = exec_trait(lw, g.type_qname), exec_trait(lw, x.type_qname)
            if gt != xt:
                raise PssSemanticError(
                    f"add_executor({xpath}): executor {x.path!r} has trait type "
                    f"{xt!r}, group {g.path!r} has {gt!r} (24.6.1.2.1)", loc=_loc(stmt))
            if x.id in groups[g.id]:
                raise PssSemanticError(
                    f"add_executor({xpath}): executor {x.path!r} is already in group "
                    f"{g.path!r} (24.6.1.2.1 b)", loc=_loc(stmt))
            groups[g.id].append(x.id)

        for kind in ("init_down", "init_up"):
            blk = comps.exec_block(inst.type_qname, kind)
            for stmt in getattr(blk, "body", None) or []:
                call = _add_executor_call(stmt)
                if call is not None and call[1] is not None:
                    add(call[0], call[1], stmt)
                    continue
                if isinstance(stmt, S.StmtForeach) and isinstance(stmt.target, E.ExprRefLocal):
                    arr = _self_path(stmt.iter)
                    n = lay.arrays.get(arr) if arr is not None and "." not in arr else None
                    # other statements (Ex 293's trait assignment) run as usual
                    calls = [_add_executor_call(b) for b in stmt.body
                             if _mentions(b, "add_executor")]
                    if n is not None and calls and all(
                            c is not None and isinstance(c[2], E.ExprRefLocal)
                            and c[2].name == stmt.target.name for c in calls):
                        for k in range(n):
                            for c in calls:
                                add(c[0], "%s[%d]" % (arr, k), stmt)
                        continue
                if _mentions(stmt, "add_executor"):
                    raise LoweringError(
                        "add_executor() that is not a top-level statement of an "
                        "init block (or of a foreach over a component array there) "
                        "is not supported by bc (groups are filled when the model "
                        "is lowered)", loc=_loc(stmt))
    lw.exec_group_map = [{"inst": by_id[g], "members": m} for g, m in groups.items()]
    return lw.exec_group_map


def exec_claim_members(lw, qname: str, inst: int, traits: list, loc) -> list:
    """The executors action *qname* may claim in context instance *inst*
    (24.6.2.3): the members of the nearest group, going up from the
    instance, whose trait type is the claim's -- the instance itself or one
    of its direct sub-instances at each level. ``[(EXEC row, member)]``."""
    if len(traits) != 1:
        raise PssSemanticError(f"action {qname!r} has {len(traits)} executor claims; "
                               f"an action may have at most one (24.6.2.1)", loc=loc)
    want = traits[0] or _EMPTY_TRAIT
    hits = [q for q in lw.types if q == want or q.endswith("::" + want)]
    hits = [q for q in hits if "::" in q] or hits
    want = hits[0] if len(hits) == 1 else want
    tree = lw.comp_tree
    by_id = {i.id: i for i in tree.instances}
    groups = {g["inst"].id: g["members"] for g in exec_groups(lw)}
    rows = [i.id for i in exec_rows(lw)]
    cur = by_id.get(inst)
    while cur is not None:
        cands = [i for i in tree.instances
                 if i.id in groups and (i.id == cur.id or getattr(i, "parent", None) == cur.id)
                 and exec_trait(lw, i.type_qname) == want]
        if len(cands) > 1:
            raise PssSemanticError(
                f"action {qname!r}: executor claim of trait {want!r} matches groups "
                f"{', '.join(repr(c.path) for c in cands)} in {cur.path or '<root>'!r} "
                f"(24.6.2.3)", loc=loc)
        if cands and not groups[cands[0].id]:
            raise PssSemanticError(f"action {qname!r}: executor group "
                                   f"{cands[0].path!r} has no executors", loc=loc)
        if cands:
            return [(rows.index(m), m) for m in groups[cands[0].id]]
        cur = by_id.get(getattr(cur, "parent", None)) if cur.id != 0 else None
    raise PssSemanticError(f"action {qname!r}: no executor group of trait {want!r} in "
                           f"its context or above it (24.6.2.3)", loc=loc)


def _self_path(x) -> Optional[str]:
    """``x.y[2]`` for a path from ``this`` (``""`` for ``this`` itself)."""
    parts = []
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
    if not isinstance(x, E.TypeExprRefSelf):
        return None
    return ".".join(reversed(parts))


def _add_executor_call(stmt):
    """``(group path, executor path or None, executor expr)`` if *stmt* is
    ``[g.]add_executor(x);``."""
    e = getattr(stmt, "expr", None) if isinstance(stmt, S.StmtExpr) else None
    if not (isinstance(e, E.ExprCall) and isinstance(e.func, E.ExprAttribute)
            and e.func.attr == "add_executor" and len(e.args) == 1):
        return None
    g = _self_path(e.func.value)
    if g is None:
        return None
    x = _self_path(e.args[0])
    return g, (x or None), e.args[0]


def lower_exec_set(ctx: CoroCtx) -> None:
    """At an action body's start: ``EXEC_SET`` to the executor its context
    instance's ``set_executor`` put in force (F5.4a), when that is static
    and the action's component has one instance; else the frame keeps the
    executor it inherited. An executor claim's assigned executor (its
    ``$exec``, solved with the action, F5.4b3) wins over both."""
    lw, st = ctx.lowerer, proc_state(ctx)
    claim = [s for n, (s, _t) in st.fields.items() if n.endswith(".$exec")]
    if claim:
        r = ctx.new_reg()
        ctx.emit(Op.LD_FIELD, (r, claim[0]), src_ref=ctx.cur_src_ref)
        ctx.emit(Op.EXEC_SET, (r,), src_ref=ctx.cur_src_ref)
        return
    if (st.component is None or st.comp_self is not None or lw.comps is None
            or getattr(lw, "comp_tree", None) is None):
        return
    ex_map = _executors(ctx)
    mine = [i for i in lw.comp_tree.instances if lw.comps.is_a(i.type_qname, st.component)]
    ex = ex_map.get(mine[0].id) if len(mine) == 1 else None
    if not isinstance(ex, tuple):
        return
    row = [i.id for i in exec_rows(lw)].index(ex[1])
    ctx.emit(Op.EXEC_SET, (_const(ctx, row),), src_ref=ctx.cur_src_ref)


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
    return _mentions(node, "set_executor")


def _mentions(node, attr: str) -> bool:
    if isinstance(node, E.ExprAttribute) and node.attr == attr:
        return True
    if dc.is_dataclass(node) and not isinstance(node, type):
        for f in dc.fields(node):
            v = getattr(node, f.name)
            for x in (v if isinstance(v, list) else [v]):
                if (dc.is_dataclass(x) and not isinstance(x, type)
                        and _mentions(x, attr)):
                    return True
    return False


def _executor_for(ctx: CoroCtx, e) -> Optional[CompCtx]:
    """The executor in force for the calling code's instance, relative to
    the frame's instance; None for none (the platform)."""
    lw = ctx.lowerer
    if lw.comps is None or getattr(lw, "comp_tree", None) is None:
        return None
    if not any(_executors(ctx).values()):
        return None                             # no executors: the platform
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


def _frame_instance(ctx: CoroCtx, e, what: str = "a memory access"):
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
        raise LoweringError(f"{what} in an action of {st.component!r}, "
                            f"which has {len(mine)} instances: bc resolves the "
                            f"executor when it lowers, so it needs one", loc=_loc(e))
    return mine[0]


# A `ref C` (F6.8b) holds the instance number of the component it refers
# to; null is NULL_REF. Numbers are from the root where the frame's instance
# is static (construction, an action of a component with one instance), so a
# reference may leave the frame's subtree; else from the frame's instance,
# and only its subtree can be reached. A component's attributes are static
# slots and its functions are inlined, so a member access through a
# reference is dispatched like a component array's element: lowered once per
# instance of C it may refer to, the copy whose number matches runs. A
# CompCtx stays frame-relative (slot and inst may be negative, F6.8b3b).

def _ref_frame(ctx: CoroCtx):
    """The frame's instance when it is static, else None."""
    lw, st = ctx.lowerer, proc_state(ctx)
    insts = lw.comp_tree.instances
    if st.comp_self is not None:
        return insts[0]
    if st.component is None:
        return None
    mine = [i for i in insts if lw.comps.is_a(i.type_qname, st.component)]
    return mine[0] if len(mine) == 1 else None


def _ref_origin(ctx: CoroCtx) -> int:
    """What a CompCtx's ``inst`` is offset by to give a reference's value."""
    f = _ref_frame(ctx)
    return f.id if f is not None else 0


def _ref_value(ctx: CoroCtx, e) -> Optional[CompCtx]:
    """The instance *e* names when it is used as a reference's value: a
    component path (``comp.s1``, ``self``), never a reference variable."""
    if not isinstance(e, (E.ExprAttribute, E.ExprSubscript, E.TypeExprRefSelf,
                          _InstExpr)):
        return None
    if isinstance(e, E.ExprAttribute) and isinstance(e.value, E.TypeExprRefSelf) \
            and e.attr != "comp":
        v = _this_lookup(ctx, e.attr)
        if v is not None and v.comp is None:
            return None                         # a reference variable
    if _dyn_index(ctx, e) is not None:
        return None                 # an element by a run-time index: dispatched
    return _comp_inst(ctx, e)


def _ref_var(ctx: CoroCtx, e) -> Optional[T]:
    """The type of *e* if it is a variable holding a component reference."""
    name = None
    if isinstance(e, (E.ExprRefLocal, E.ExprRefUnresolved)):
        name = e.name
    elif isinstance(e, E.ExprAttribute) and isinstance(e.value, E.TypeExprRefSelf):
        name = e.attr
    v = None if name is None else (_this_lookup(ctx, name) if isinstance(e, E.ExprAttribute)
                                   else _lookup(ctx, name))
    t = v.type if v is not None else None
    st = proc_state(ctx)
    if v is None and name is not None and not _comp_code(st) and name in st.fields:
        t = st.fields[name][1]                  # an action's `ref` field
        if isinstance(t, T) and t.kind == "compref":
            _action_ref_static(ctx, e)
    elif v is None and isinstance(e, (E.ExprAttribute, E.ExprRefUnresolved)):
        cd = _comp_data(ctx, e)                 # a component's `ref` field
        t = cd[1] if cd is not None and cd[0] == "leaf" else None
    return t if isinstance(t, T) and t.kind == "compref" else None


def _action_ref_static(ctx: CoroCtx, e) -> None:
    """An action's `ref` field is set in its parent's frame (a traversal's
    initializer) and read in its own. A reference is numbered from the root
    only where the frame's instance is static, so both must be: elsewhere a
    number means a different instance in each frame."""
    if _ref_frame(ctx) is None:
        _frame_instance(ctx, e, "an action's component reference field")


def _ref_qname(ctx: CoroCtx, name: Optional[str]) -> str:
    types = ctx.lowerer.comps.types
    if name in types:
        return name
    hits = [q for q in types if q.endswith("::" + str(name))]
    if len(hits) != 1:
        raise LoweringError(f"component type {name!r} of a reference is not found",
                            loc=None)
    return hits[0]


def _ref_targets(ctx: CoroCtx, t: T, e) -> List[CompCtx]:
    """Every instance a reference of type *t* may refer to: each instance of
    its component type (or one derived from it) in the tree, or in the
    frame's subtree when the frame's instance is not static."""
    lw, st = ctx.lowerer, proc_state(ctx)
    if getattr(lw, "comps", None) is None:
        raise LoweringError("a component reference outside a component tree is "
                            "not supported by bc", loc=_loc(e))
    if st.comp_self is not None:
        frame = lw.comp_tree.instances[0].type_qname
    elif st.component is not None:
        frame = st.component
    else:
        raise LoweringError("a component reference used in a package function "
                            "is not supported by bc", loc=_loc(e))
    q = _ref_qname(ctx, t.name)

    def fits(qname: str) -> bool:
        return lw.comps.is_a(qname, q) and _reg_width_fits(ctx, qname, t)

    f = _ref_frame(ctx)
    if f is not None:
        return [CompCtx(i.type_qname, i.base - f.base, i.id - f.id)
                for i in lw.comp_tree.instances if fits(i.type_qname)]
    out = [CompCtx(frame, 0, 0)] if fits(frame) else []
    out += [CompCtx(s.type_qname, s.slot, s.inst)
            for s in lw.comps.get(frame).subs.values() if fits(s.type_qname)]
    return sorted(out, key=lambda c: c.inst)


def _reg_width_fits(ctx: CoroCtx, qname: str, t: T) -> bool:
    """A ``ref reg_sized_c<SZ>`` (``t.reg_size``) refers only to a register
    SZ bits wide (21.14.1); any other reference, to any instance of its type."""
    if t.reg_size is None:
        return True
    dt = ctx.lowerer.comps.types.get(qname)
    return isinstance(dt, DataTypeRegister) and dt.size_bits == t.reg_size


def _ref_convert(ctx: CoroCtx, e, want: T) -> int:
    """*e* as a reference of type *want*: null, a component path or a
    reference of *want*'s type or one derived from it (7.10)."""
    if _cn(e) == "ExprNull":
        return _const(ctx, NULL_REF)
    r, t = _ev(ctx, e, None)
    if t.kind != "compref":
        raise PssSemanticError(f"a reference to component {want.name!r} takes a "
                               f"component, a reference or null", loc=_loc(e))
    comps = ctx.lowerer.comps
    have, need = _ref_qname(ctx, t.name), _ref_qname(ctx, want.name)
    if not comps.is_a(have, need):
        down = "; a downcast needs a cast" if comps.is_a(need, have) else ""
        raise PssSemanticError(f"a reference to {t.name!r} is not a reference to "
                               f"{want.name!r}{down}", loc=_loc(e))
    if want.reg_size is not None:
        # `ref reg_sized_c<SZ>`: a register of that width (21.14.1)
        dt = comps.types.get(have)
        size = (t.reg_size if t.reg_size is not None
                else dt.size_bits if isinstance(dt, DataTypeRegister) else None)
        if size != want.reg_size:
            raise PssSemanticError(
                f"a reference to a {size or 'any'}-bit register is not a "
                f"reference to reg_sized_c<{want.reg_size}>", loc=_loc(e))
    return r


def _ref_cast(ctx: CoroCtx, e, want: T) -> int:
    """``(ref D)x`` (7.12.1): an upcast is *x*; a downcast is *x* when the
    instance it refers to is a D (or derived from it), else null."""
    if _cn(e.value) == "ExprNull":
        return _const(ctx, NULL_REF)
    r, t = _ev(ctx, e.value, None)
    if t.kind != "compref":
        raise PssSemanticError(f"a cast to a reference to component {want.name!r} "
                               f"takes a component or a reference", loc=_loc(e))
    comps = ctx.lowerer.comps
    have, need = _ref_qname(ctx, t.name), _ref_qname(ctx, want.name)
    if comps.is_a(have, need):
        return r
    if not comps.is_a(need, have):
        raise PssSemanticError(f"a reference to {t.name!r} cannot be cast to a "
                               f"reference to {want.name!r}", loc=_loc(e))
    out = ctx.new_reg()
    ctx.emit(Op.MOV, (out, _const(ctx, NULL_REF)))
    for c in _ref_targets(ctx, want, e):
        brz = ctx.emit(Op.BRZ, (_op(ctx, Op.CMP_EQ, r,
                                    _const(ctx, c.inst + _ref_origin(ctx))), 0))
        ctx.emit(Op.MOV, (out, r))
        _patch(ctx, brz, len(ctx.code))
    return out


def _ref_rebase(ctx: CoroCtx, r: int, e, sign: int) -> int:
    """A component's `ref` field holds its instance number from the root, as
    references do wherever the frame's instance is static; a field may refer
    anywhere in the tree, so it cannot be read or set where it is not.
    *sign*: -1 read, +1 stored (kept for the frame-relative case)."""
    if _ref_frame(ctx) is None:
        _frame_instance(ctx, e, "a component reference field")   # several: refused
    return r


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


def emit_import(ctx: CoroCtx, name: str, fn_id: int, ret_reg: int,
                arg_regs: List[int], flags: int, loc=None,
                src_ref: Optional[int] = None, n_res: int = 0) -> None:
    """Emit an IMPORT of *fn_id*: up to two arguments inline (arg2/arg3),
    more staged with ARG as for CALL and counted in ``imm`` bits 0-7
    (F6.5b). *n_res* > 0: the host answers with that many values (a struct
    result's leaves), read back with LD_RES; counted in ``imm`` bits 8-15."""
    if len(arg_regs) > CALL_MAX_ARGS:
        raise LoweringError(f"import {name!r} takes {len(arg_regs)} arguments: "
                            f"bc passes at most {CALL_MAX_ARGS} to the platform",
                            loc=loc)
    if n_res > CALL_MAX_ARGS:
        raise LoweringError(f"import {name!r} returns {n_res} values (one per "
                            f"struct field): bc takes at most {CALL_MAX_ARGS} "
                            f"back from the platform", loc=loc)
    if len(arg_regs) <= 2:
        ctx.emit(Op.IMPORT, tuple([fn_id, ret_reg] + list(arg_regs)), flags=flags,
                 imm=n_res << 8, src_ref=src_ref)
        return
    for i, r in enumerate(arg_regs):
        ctx.emit(Op.ARG, (r, i), src_ref=src_ref)
    ctx.emit(Op.IMPORT, (fn_id, ret_reg), imm=len(arg_regs) | n_res << 8,
             flags=flags, src_ref=src_ref)


def _import_call(ctx: CoroCtx, name: str, e):
    decl = ctx.lowerer.imports[name]
    _string_import(ctx, decl)
    # A struct argument is passed one leaf per argument, in layout order
    # (host-abi.md "Struct arguments"); the host gets copies of its fields.
    structs = decl.get("struct_at") or {}
    arg_regs = []
    for i, a in enumerate(e.args):
        if _is_list(ctx, a):
            raise LoweringError(f"import {name!r} argument {i + 1} is a "
                                f"{type_of(ctx, a).kind}: collections across "
                                f"imports are not supported by bc yet", loc=_loc(e))
        pt = (value_type(structs[i], ctx.lowerer.types) if i in structs
              else None)
        if isinstance(pt, StructT):
            place = _struct_value(ctx, a, pt) if pt.leaves else _Place(pt, [])
            arg_regs.extend(_load_loc(ctx, loc) for loc in place.locs)
        else:
            arg_regs.append(ev(ctx, a)[0])
    # A struct result comes back as its leaves, which the engine stages as
    # a called function's results (LD_RES; host-abi.md "Struct results").
    ret_t = (value_type(structs["return"], ctx.lowerer.types)
             if "return" in structs else None)
    n_res = len(ret_t.leaves) if isinstance(ret_t, StructT) else 0
    ret_reg = _VOID if n_res else ctx.new_reg()
    flags = INSTR_F_HAS_RET | (INSTR_F_BLOCKING if decl["blocking"] else 0)
    emit_import(ctx, name, decl["fn_id"], ret_reg, arg_regs, flags, loc=_loc(e),
                n_res=n_res)
    ctx.lowerer.import_decls.setdefault(
        decl["fn_id"], {"fn": name, "blocking": decl["blocking"]})
    if isinstance(ret_t, StructT):
        st = proc_state(ctx)
        st.uniq += 1
        res = _struct_locals(ctx, f"{name}$iret${st.uniq}", ret_t)
        for i, loc in enumerate(res.locs):
            r = ctx.new_reg()
            ctx.emit(Op.LD_RES, (r, i))
            _store_loc(ctx, loc, r)
        return res, ret_t
    rt = decl.get("ret_type")
    return ret_reg, (_import_ret_t(rt) if rt else U64)


def _inline(ctx: CoroCtx, fn, call, comp: Optional[CompCtx] = None,
            action: bool = False) -> Tuple[Optional[int], Optional[T]]:
    st = proc_state(ctx)
    if action:
        # An action's function runs on the action's object, which a called
        # form has no parameter for: it is always inlined.
        if any(f.fn is fn for f in st.fns):
            raise LoweringError(f"action function {fn.name!r} calls itself: "
                                f"recursion in an action's function is not "
                                f"supported by bc", loc=_loc(call))
    elif any(f.fn is fn for f in st.fns):
        return _call_recursive(ctx, fn, call, comp)
    elif ctx.lowerer.debug and _has_called_form(ctx, fn):
        # Debug lowering keeps each call a CALL, so the function has its own
        # frame and statements; one taking or returning a struct is inlined.
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
        elif pt.kind in COLLECTIONS:
            # a collection is passed by value: the parameter owns a copy
            values.append((p.arg, pt, _list_value(ctx, src, pt)))
        else:
            values.append((p.arg, pt, _eval_for(ctx, src, pt)))

    ret_t = value_type(fn.returns, types) if fn.returns is not None else None
    frame = _Fn(fn=fn, scope_base=len(st.scopes), loop_base=len(st.loops),
                ret_type=ret_t, ret_slot=None, comp=comp, action=action)
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
        # coverage: an inlined body's statements are the function's
        saved_cov = ctx.cov_scope, ctx.cov_path
        ctx.cov_scope = (f"{comp.type_qname}::{fn.name}" if comp
                         else f"{st.action_type}::{fn.name}" if action
                         else fn.name)
        ctx.cov_path = []
        try:
            _lower_block(ctx, fn.body)
        finally:
            ctx.cov_scope, ctx.cov_path = saved_cov
        end = len(ctx.code)
        for j in frame.returns:
            _patch(ctx, j, end)
        _list_free_scope(ctx, st.scopes[-1])        # collection parameters
    finally:
        _close_scope(ctx, st.scopes.pop())
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


def _has_called_form(ctx: CoroCtx, fn) -> bool:
    """Whether a call to *fn* can be a CALL: scalar parameters (at most
    ``CALL_MAX_ARGS``, no varargs) and a scalar or no result."""
    args = getattr(fn, "args", None)
    if args is not None and getattr(args, "vararg", None) is not None:
        return False
    params = list(getattr(args, "args", []) or []) if args is not None else []
    if len(params) > CALL_MAX_ARGS:
        return False
    types = ctx.lowerer.types
    if any(isinstance(value_type(p.annotation, types), StructT) for p in params):
        return False
    return fn.returns is None or not isinstance(value_type(fn.returns, types), StructT)


def _call_recursive(ctx: CoroCtx, fn, call, comp) -> Tuple[Optional[int], Optional[T]]:
    """A call to *fn* while it is being inlined: a CALL of its called form.
    The arguments are all evaluated before any is staged, since evaluating
    one may itself CALL."""
    types = ctx.lowerer.types
    params = _params(fn, call)
    if len(params) > CALL_MAX_ARGS:
        raise LoweringError(f"recursive function {fn.name!r} has more than "
                            f"{CALL_MAX_ARGS} parameters", loc=_loc(call))
    # A struct argument is one argument per leaf; the struct is the caller's
    # (20.3.2), so the callee hands its leaves back (LD_RES) and they are
    # stored to it again: copy-in, copy-out (calls.md).
    regs, outs = [], []
    for name, ann, src in params:
        pt = value_type(ann, types)
        if isinstance(pt, StructT):
            place = _struct_value(ctx, src, pt) if pt.leaves else _Place(pt, [])
            regs.extend(_load_loc(ctx, loc) for loc in place.locs)
            if _may_write(fn.body, name):
                outs.append(place)
        elif pt.kind in COLLECTIONS:
            regs.append(_list_value(ctx, src, pt))  # the callee owns a copy
        else:
            regs.append(_eval_for(ctx, src, pt))
    ret_t = value_type(fn.returns, types) if fn.returns is not None else None
    n_res = (len(ret_t.leaves) if isinstance(ret_t, StructT) else 0) \
        + sum(len(p.locs) for p in outs)
    if len(regs) > CALL_MAX_ARGS or n_res > CALL_MAX_ARGS:
        raise LoweringError(f"recursive function {fn.name!r} passes more than "
                            f"{CALL_MAX_ARGS} values (one per struct field) "
                            f"in or out", loc=_loc(call))
    target = _called_form(ctx, fn, comp)
    for i, r in enumerate(regs):
        ctx.emit(Op.ARG, (r, i))
    scalar = ret_t is not None and not isinstance(ret_t, StructT)
    rd = ctx.new_reg() if scalar else _VOID
    ctx.emit(Op.CALL, (target, rd))
    res = None
    if isinstance(ret_t, StructT):
        st = proc_state(ctx)
        st.uniq += 1
        res = _struct_locals(ctx, f"{fn.name}$cret${st.uniq}", ret_t)
    i = 0
    for place in ([res] if res is not None else []) + outs:
        for loc in place.locs:
            r = ctx.new_reg()
            ctx.emit(Op.LD_RES, (r, i))
            _store_loc(ctx, loc, r)
            i += 1
    if res is not None:
        return res, ret_t
    return (rd, ret_t) if scalar else (None, None)


def _may_write(body, name: str) -> bool:
    """Whether *body* may write struct parameter *name*: assigns into it, or
    passes it whole (or calls a method of it), which may write it by handle.
    Only such a parameter is handed back at RET, so a read-only one adds no
    uses to the function's last statement (``trace why``)."""
    seen = set()

    def root(e):
        while (isinstance(e, (E.ExprAttribute, E.ExprSubscript))
               and not isinstance(e.value, E.TypeExprRefSelf)):
            e = e.value
        return e

    def mine(e) -> bool:
        # a parameter is ``name`` or, as the front end writes it, ``self.name``
        return ((isinstance(e, E.ExprRefLocal) and e.name == name)
                or (isinstance(e, E.ExprAttribute) and e.attr == name
                    and isinstance(e.value, E.TypeExprRefSelf)))

    def walk(n) -> bool:
        if isinstance(n, (list, tuple)):
            return any(walk(x) for x in n)
        if not dc.is_dataclass(n) or isinstance(n, type) or id(n) in seen:
            return False
        seen.add(id(n))
        if isinstance(n, S.StmtAssign) and any(mine(root(t)) for t in n.targets):
            return True
        if isinstance(n, S.StmtAugAssign) and mine(root(n.target)):
            return True
        if isinstance(n, E.ExprCall) and (any(mine(a) for a in n.args)
                                         or mine(root(n.func))):
            return True
        return any(walk(getattr(n, f.name, None)) for f in dc.fields(n))

    return walk(body)


def _stage_results(ctx: CoroCtx, fr: _Fn, place: Optional[_Place] = None) -> None:
    """Before a called form's RET: stage a struct result's leaves (zeros
    with *place* None), then each struct parameter's, for the caller's
    LD_RES."""
    vals = []
    if isinstance(fr.ret_type, StructT):
        vals = ([_load_loc(ctx, loc) for loc in place.locs] if place is not None
                else [_const(ctx, 0)] * len(fr.ret_type.leaves))
    for p in fr.out_places:
        vals.extend(_load_loc(ctx, loc) for loc in p.locs)
    for i, r in enumerate(vals):
        ctx.emit(Op.ARG, (r, i))


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
    desc = CoroDescriptor(name=sub.coro_name, code=[], blocks=[], frame_locals=[],
                          fn_qname=(f"{comp.type_qname}::{fn.name}" if comp
                                    else fn.name))
    forms[key] = idx = lw.add_branch_coro(desc)
    st = proc_state(sub)
    types = lw.types
    ret_t = value_type(fn.returns, types) if fn.returns is not None else None
    frame = _Fn(fn=fn, scope_base=0, loop_base=0, ret_type=ret_t, ret_slot=None,
                comp=comp, called=True)
    st.fns.append(frame)
    st.scopes.append({})
    # A struct parameter takes one argument per leaf, and hands them back
    # at RET (_stage_results).
    i = 0
    for name, ann, _ in _params(fn, None):
        v = _declare(sub, name, value_type(ann, types))
        v.borrowed = isinstance(v.type, StructT)
        if isinstance(v.type, StructT) and _may_write(fn.body, name):
            frame.out_places.append(_Place(v.type, list(v.locs)))
        for loc in (v.locs if isinstance(v.type, StructT)
                    else [("local", v.slot)]):
            r = sub.new_reg()
            sub.emit(Op.LD_ARG, (r, i))
            _store_loc(sub, loc, r)
            i += 1
    sub.cov_scope = desc.fn_qname
    _lower_block(sub, fn.body)
    _list_free_scope(sub, st.scopes[-1])           # collection parameters
    _stage_results(sub, frame)
    sub.emit(Op.RET, (_const(sub, 0),)
             if ret_t is not None and not isinstance(ret_t, StructT) else ())
    desc.code, desc.frame_locals = sub.code, sub.frame_locals
    desc.stmts = sub.stmts
    desc.chans = sub.chans
    desc.cov = sub.cov
    desc.regs = sub.regs
    desc.discards = sub.discards
    desc.locals = sub.debug_locals()
    return idx


def _ret_value(ctx: CoroCtx, value, rt: T) -> int:
    """A function's scalar result *value*; a collection is a copy the caller
    owns (the function's locals are freed as it returns)."""
    if rt.kind in COLLECTIONS:
        return _list_value(ctx, value, rt)
    return _eval_for(ctx, value, rt)


def _eval_for(ctx: CoroCtx, value, dst: T) -> int:
    """Evaluate *value* in an assignment-like context and convert it to *dst*."""
    if dst.kind == "compref":
        return _ref_convert(ctx, value, dst)
    vt = type_of(ctx, value)
    want = None
    # A float on either side is converted by assign_convert alone: widening
    # a float value to an integer type first would truncate it (2.5 -> 2).
    if not (dst.is_bool or vt.is_bool or dst.kind == "string" or vt.kind == "string"
            or dst.is_float or vt.is_float):
        want = T(max(dst.width, vt.as_int().width), vt.as_int().signed)
        if want == vt:
            want = None
    r, src = ev(ctx, value, want)
    return assign_convert(ctx, r, src, dst)


def _message(ctx: CoroCtx, e, fail: Optional[int] = None) -> None:
    """``message(verbosity, format, args...)`` -> IMPORT of the builtin sink.

    ``fail`` is the builtin of an ``error(format, args...)`` or a
    ``fatal(status, format, args...)``: the same message entry, its first slot
    the status (0 for ``error()``), and the run stops after it.
    """
    what = ("message()" if fail is None else "print()" if fail == BUILTIN_PRINT
            else "file_write()" if fail == BUILTIN_FILE_WRITE else "error()/fatal()")
    first = 0 if fail in (BUILTIN_ERROR_MSG, BUILTIN_PRINT) else 1
    if len(e.args) < first + 1:
        raise PssSemanticError(
            "message() needs a verbosity and a format string" if fail is None
            else "file_write() needs a handle and a format string"
            if fail == BUILTIN_FILE_WRITE else f"{what} needs a format string", loc=_loc(e))
    fmt = e.args[first]
    args = list(e.args[first + 1:])
    fmt_text = fmt.value if (isinstance(fmt, E.ExprConstant)
                             and isinstance(fmt.value, str)) else None
    if fmt_text is None:
        # A computed format (a string variable, say) has as many specifiers as
        # arguments (LRM 24.1.1), so with none it is the text itself: print it
        # through "%s".  With arguments it would need run-time formatting.
        if args:
            raise LoweringError(
                f"{what} with arguments needs a string-literal format in bc",
                loc=_loc(e))
        fmt_text, args = "%s", [fmt]
    vreg = (_const(ctx, 0) if first == 0
            else _eval_for(ctx, e.args[0], CHANDLE) if fail == BUILTIN_FILE_WRITE
            else ev(ctx, e.args[0])[0])
    regs, descs = [vreg], []
    for a in args:
        r, t = ev(ctx, a)                           # varargs are self-determined
        regs.append(r)
        descs.append(t.descriptor())
    # An instruction carries at most 4 inline args, so the values travel in
    # frame locals the table entry names: verbosity first, then each argument.
    # One pool per coroutine is safe -- every argument is already in a register
    # (any message() inside an argument's call has run) before the pool is written.
    slots = [ctx.local_slot(f"$msg{i}") for i in range(len(regs))]
    try:
        idx = ctx.lowerer.add_message(fmt_text, descs, slots)
    except ValueError as ex:
        raise PssSemanticError(f"{what}: {ex}", loc=_loc(e))
    for r, slot in zip(regs, slots):
        ctx.emit(Op.ST_LOCAL, (r, slot))
    ctx.emit(Op.IMPORT, (BUILTIN_MESSAGE if fail is None else fail, _VOID,
                         _const(ctx, idx)))


def _runtime_error(ctx: CoroCtx, text: str) -> None:
    idx = ctx.lowerer.intern_string(text)
    ctx.emit(Op.IMPORT, (BUILTIN_ERROR, _VOID, _const(ctx, idx)))


# --------------------------------------------------------------------------- #
# Statements
# --------------------------------------------------------------------------- #

def _lower_block(ctx: CoroCtx, stmts) -> None:
    _push_scope(ctx)
    try:
        for i, s in enumerate(stmts or []):
            _lower_at(ctx, str(i), s)
    finally:
        _pop_scope(ctx)


def _lower_at(ctx: CoroCtx, seg: str, s) -> None:
    """Lower statement *s*, *seg* its place in its block (coverage paths)."""
    ctx.cov_path.append(seg)
    try:
        lower_stmt(ctx, s)
    finally:
        ctx.cov_path.pop()


def _arm(ctx: CoroCtx, label: str, body, node=None, implicit: bool = False) -> None:
    """Lower one arm of a branch, its statements under *label*. Debug
    lowering starts it with a NOP of its own, which only this arm reaches,
    and records it as a coverage arm (``CoroDescriptor.cov``); *implicit*:
    an ``else`` the code did not write."""
    if ctx.lowerer.debug:
        sr = (ctx.lowerer.prov.src_ref(node) if node is not None else 0) or ctx.cur_src_ref
        at = ctx.emit(Op.NOP, src_ref=sr)
        ctx.cov.append(("arm", at, ctx.cov_scope, "/".join(ctx.cov_path + [label]),
                        sr, implicit))
    ctx.cov_path.append(label)
    try:
        _lower_block(ctx, body)
    finally:
        ctx.cov_path.pop()


def lower_stmt(ctx: CoroCtx, stmt) -> None:
    """Lower one Layer-0 statement into ``ctx.code``."""
    sr = ctx.lowerer.prov.src_ref(stmt)
    saved, ctx.cur_src_ref = ctx.cur_src_ref, (sr or ctx.cur_src_ref)
    try:
        if ctx.lowerer.debug:
            _mark_stmt(ctx)
        _lower_stmt(ctx, stmt)
        if ctx.lowerer.debug and ctx.stmts[-1][0] == len(ctx.code):
            ctx.emit(Op.NOP)
    finally:
        ctx.cur_src_ref = saved


def _mark_stmt(ctx: CoroCtx) -> None:
    """A statement marker (debug lowering): the statement starts at the next
    pc. An enclosing statement that has not emitted code yet gets a NOP, so
    each statement has its own pc."""
    if ctx.stmts and ctx.stmts[-1][0] == len(ctx.code):
        ctx.emit(Op.NOP, src_ref=ctx.stmts[-1][1])
    ctx.stmts.append((len(ctx.code), ctx.cur_src_ref))
    ctx.cov.append(("stmt", len(ctx.code), ctx.cov_scope, "/".join(ctx.cov_path),
                    ctx.cur_src_ref, False))


def _lower_stmt(ctx: CoroCtx, s) -> None:
    st = proc_state(ctx)

    if isinstance(s, S.StmtAnnAssign):
        try:
            t = value_type(s.annotation, ctx.lowerer.types)
        except LoweringError as e:
            if e.loc is not None:
                raise
            raise LoweringError(str(e), loc=_loc(s)) from e
        if not isinstance(t, StructT) and t.kind in COLLECTIONS:
            r = _list_value(ctx, s.value, t) if s.value is not None else _const(ctx, 0)
            if not isinstance(s.target, E.ExprRefLocal):
                raise LoweringError(f"unsupported declaration target {_cn(s.target)}",
                                    loc=_loc(s))
            _list_replace(ctx, ("local", _declare(ctx, s.target.name, t).slot), r, t)
            return
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
        elif t.kind == "compref":
            r = _const(ctx, NULL_REF)
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
            ctx.discard_call = e                    # its value is dropped
            try:
                _call(ctx, e)
            finally:
                ctx.discard_call = None
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
                if isinstance(fr.ret_type, StructT):
                    _stage_results(ctx, fr, _struct_value(ctx, s.value, fr.ret_type))
                    ctx.emit(Op.RET)
                    return
                r = _ret_value(ctx, s.value, fr.ret_type)
                _list_free_fn(ctx, fr)
                _stage_results(ctx, fr)
                ctx.emit(Op.RET, (r,))
            else:
                _list_free_fn(ctx, fr)
                _stage_results(ctx, fr)
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
                r = _ret_value(ctx, s.value, fr.ret_type)
                ctx.emit(Op.ST_LOCAL, (r, ctx.local_slot(fr.ret_slot)))
            _list_free_fn(ctx, fr, inner=True)
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
        _arm(ctx, "then", s.body)
        if s.orelse or ctx.lowerer.debug:          # debug: count the implicit else
            br = ctx.emit(Op.BR, (0,))
            _patch(ctx, brz, len(ctx.code))
            _arm(ctx, "else", s.orelse, implicit=not s.orelse)
            _patch(ctx, br, len(ctx.code))
        else:
            _patch(ctx, brz, len(ctx.code))
        return

    if isinstance(s, S.StmtIfChain):
        # Each clause's test runs only after every earlier one was false;
        # each taken body jumps to the end of the chain.
        ends = []
        for k, c in enumerate(s.clauses):
            t, tt = ev(ctx, c.test)
            brz = ctx.emit(Op.BRZ, (truth(ctx, t, tt), 0))
            _arm(ctx, str(k), c.body, node=c)
            if s.orelse or k + 1 < len(s.clauses) or ctx.lowerer.debug:
                ends.append(ctx.emit(Op.BR, (0,)))
            _patch(ctx, brz, len(ctx.code))
        if s.orelse or ctx.lowerer.debug:          # debug: count the implicit else
            _arm(ctx, "else", s.orelse, implicit=not s.orelse)
        for br in ends:
            _patch(ctx, br, len(ctx.code))
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

    if isinstance(s, S.StmtRandomize):
        _lower_randomize(ctx, s)
        return

    raise LoweringError(f"unsupported statement {_cn(s)} in bc procedural code",
                        loc=_loc(s))


def _target_path(e) -> str:
    """The dotted name a ``randomize`` target goes by in its constraints."""
    if isinstance(e, (E.ExprRefLocal, E.ExprRefUnresolved)):
        return e.name
    if isinstance(e, E.ExprAttribute):
        if isinstance(e.value, E.TypeExprRefSelf):
            return e.attr
        return _target_path(e.value) + "." + e.attr
    raise LoweringError(f"randomize: {_cn(e)} is not a variable or attribute "
                        f"bc can randomize", loc=_loc(e))


def _loop_names(s) -> frozenset:
    """The names a constraint ``foreach`` binds: its element and index."""
    return frozenset(v.name for v in (s.target, s.index_var)
                     if isinstance(v, E.ExprRefLocal))


def _as_self(e, keep=frozenset()):
    """*e* with each local ``x`` read as ``self.x``: the problem's object is
    the statement's targets, by name (as an action's is its fields). A
    ``foreach``'s element and index (*keep*) stay locals."""
    if isinstance(e, E.ExprRefLocal):
        return e if e.name in keep else E.ExprAttribute(value=E.TypeExprRefSelf(),
                                                        attr=e.name)
    if not dc.is_dataclass(e) or isinstance(e, type):
        return e
    if isinstance(e, S.StmtForeach):
        inner = keep | _loop_names(e)
        return dc.replace(e, iter=_as_self(e.iter, keep),
                          body=[_as_self(x, inner) for x in e.body or []])
    repl = {}
    for f in dc.fields(e):
        v = getattr(e, f.name)
        if isinstance(v, (E.Expr, S.Stmt)):
            repl[f.name] = _as_self(v, keep)
        elif isinstance(v, list) and any(isinstance(x, (E.Expr, S.Stmt)) for x in v):
            repl[f.name] = [_as_self(x, keep) for x in v]
    return dc.replace(e, **repl) if repl else e


def _unresolved_ref(c, bound=frozenset()) -> Optional[object]:
    """A reference left in constraint *c* that is not a problem variable
    (nor the index of a ``foreach`` around it, *bound*)."""
    if isinstance(c, E.ExprRefField):
        return None
    if isinstance(c, E.ExprRefLocal) and c.name in bound:
        return None
    if getattr(c, "index_var", None) and type(c).__name__ == "ConstraintForeach":
        bound = bound | {c.index_var}
    if isinstance(c, (E.ExprRefLocal, E.ExprRefUnresolved, E.TypeExprRefSelf,
                      E.TypeExprRefTraversed)):
        return c
    if isinstance(c, E.ExprAttribute):
        return c
    if not dc.is_dataclass(c) or isinstance(c, type):
        return None
    for f in dc.fields(c):
        if f.name == "func" and isinstance(c, E.ExprCall):
            continue
        v = getattr(c, f.name)
        for x in (v if isinstance(v, list) else [v]):
            if dc.is_dataclass(x) and not isinstance(x, type):
                bad = _unresolved_ref(x, bound)
                if bad is not None:
                    return bad
    return None


def _pin_inputs(ctx: CoroCtx, e, slots, inputs, keep=frozenset()):
    """*e* with each value it reads that is not a target -- a local, an action
    or component field, a loop index -- replaced by an input variable of the
    problem (F3.7b). *inputs* collects ``(path, expr, type)`` in var order;
    the input's var_id is ``len(slots)`` at the time it is added. *keep*:
    names read as they are (a constraint ``foreach``'s element and index, a
    random list whose ``l[i]`` the solver reads)."""
    if isinstance(e, E.ExprRefLocal) and e.name in keep:
        return e
    if isinstance(e, S.StmtForeach):
        inner = keep | _loop_names(e)
        return dc.replace(e, iter=_pin_inputs(ctx, e.iter, slots, inputs, keep),
                          body=[_pin_inputs(ctx, x, slots, inputs, inner)
                                for x in e.body or []])
    if isinstance(e, (E.ExprRefLocal, E.ExprRefUnresolved, E.ExprAttribute)):
        try:
            path = _target_path(e)
        except LoweringError:
            path = None
        if path is not None and (path in slots
                                 or any(n.startswith(path + ".") for n in slots)):
            return e                    # a target, or a target struct's member
        if path is not None:
            try:
                tt = _target_type(ctx, e)
            except (LoweringError, PssSemanticError):
                tt = None
            if tt is not None and not isinstance(tt, StructT) and tt.kind not in (
                    "string", "chandle"):
                if path not in slots:
                    slots[path] = len(slots)
                    inputs.append((path, e, tt))
                return E.ExprRefField(base=E.TypeExprRefSelf(), index=slots[path])
    if isinstance(e, E.ExprSubscript) and _is_list(ctx, e.value):
        # ``l[i]``, *i* a constant or a non-target: one pinned element
        try:
            ix = e.slice
            ix = (str(ix.value) if isinstance(ix, E.ExprConstant)
                  else _target_path(ix))
            path = f"{_target_path(e.value)}[{ix}]"
            tt = type_of(ctx, e)
            if _target_path(e.value) in keep or ix in keep:
                path = None             # an element the solver reads
        except (LoweringError, PssSemanticError):
            path = None
        if (path is not None and ix not in slots and tt.kind in ("int", "bool", "enum")):
            if path not in slots:
                slots[path] = len(slots)
                inputs.append((path, e, tt))
            return E.ExprRefField(base=E.TypeExprRefSelf(), index=slots[path])
    if not dc.is_dataclass(e) or isinstance(e, type):
        return e
    if (isinstance(e, E.ExprCall) and isinstance(e.func, E.ExprAttribute)
            and e.func.attr == "size" and not e.args):
        # ``l.size()``: the list is an input (an array of the problem, F6.10d)
        n = _pin_inputs(ctx, e.func.value, slots, inputs, keep)
        return e if n is e.func.value else dc.replace(
            e, func=dc.replace(e.func, value=n))
    repl = {}
    for f in dc.fields(e):
        if f.name == "func" and isinstance(e, E.ExprCall):
            continue
        v = getattr(e, f.name)
        if isinstance(v, (E.Expr, S.Stmt)):
            n = _pin_inputs(ctx, v, slots, inputs, keep)
            if n is not v:
                repl[f.name] = n
        elif isinstance(v, list) and any(isinstance(x, (E.Expr, S.Stmt)) for x in v):
            n = [_pin_inputs(ctx, x, slots, inputs, keep) for x in v]
            if any(a is not b for a, b in zip(n, v)):
                repl[f.name] = n
    return dc.replace(e, **repl) if repl else e


def _struct_dt(types, tt):
    """The IR struct type behind lowered struct type *tt*, or None."""
    return types.get(tt.name) or next(
        (t for t in types.values() if getattr(t, "name", None) == tt.name), None)


def _rand_leaf(dt, path, types) -> bool:
    """Whether leaf *path* of struct *dt* is random: each field on the way is
    declared ``rand`` (a field bc cannot find counts as random)."""
    from zuspec.ir.core.xf.pss_lower.layout import struct_fields
    for a in path:
        f = next((f for f in struct_fields(dt, types)
                  if getattr(f, "name", None) == a), None)
        if f is None:
            return True
        if getattr(f, "rand_kind", None) is None:
            return False
        dt = f.datatype
    return True


def _lower_randomize(ctx: CoroCtx, s) -> None:
    """``randomize x, s with { ... };`` (LRM 21.3.1, plan F3.7): one solve
    problem per statement, whose variables are the targets' scalar leaves
    (var_id = leaf order) and whose constraints are each struct target's
    type constraints plus the ``with`` block. SOLVE keeps the values on the
    frame; SOLVE_GET + an ordinary store moves each into its leaf."""
    from zuspec.ir.core import constraint as C
    from zuspec.ir.core.scenario import ScSolveArray, ScSolveProblem, ScSolveVar
    from zuspec.ir.core.xf.pss_lower import constraints as CC
    from zuspec.ir.core.xf.pss_lower import defaults as D
    from zuspec.ir.core.xf.validate import UnsupportedConstructError
    from types import SimpleNamespace
    from ..model import SolveProblem
    from .constraints import build_solve_blob

    from zuspec.ir.core.expr_phase2 import ExprTuple
    targets = (list(s.target.elts) if isinstance(s.target, ExprTuple)
               else [s.target])
    types = ctx.lowerer.types
    leaves = []         # (name, type, struct leaf loc or None, scalar target)
    roots = []          # (name, struct type)
    fixed = []          # (name, expr, type): a struct's non-rand leaves, pinned
    rand_lists = []     # (name, expr, type): lists whose elements are solved
    for tgt in targets:
        name = _target_path(tgt)
        tt = _target_type(ctx, tgt)
        if isinstance(tt, StructT):
            p = _place(ctx, tgt)
            if p is None:
                raise LoweringError(f"randomize: {name!r} is not a struct bc can "
                                    f"move", loc=_loc(tgt))
            dt = _struct_dt(types, tt)
            for (path, lt), loc in zip(tt.leaves, p.locs):
                n = ".".join((name,) + tuple(path))
                if dt is not None and not _rand_leaf(dt, path, types):
                    e = tgt             # 21.3.1: only rand fields are randomized
                    for a in path:
                        e = E.ExprAttribute(value=e, attr=a)
                    fixed.append((n, e, lt))
                else:
                    leaves.append((n, lt, loc, None))
            roots.append((name, tt))
        elif tt.kind in COLLECTIONS:
            # a random list: its elements are solved, its length kept (F6.10e)
            rand_lists.append((name, tgt, tt))
        else:
            _not_opaque(tt, tgt)
            if tt.kind in ("string", "chandle"):
                raise PssSemanticError(f"randomize: {name!r} is a {tt.kind}, "
                                       f"which cannot be random", loc=_loc(tgt))
            leaves.append((name, tt, None, tgt))
    slots = {n: i for i, (n, _, _, _) in enumerate(leaves)}
    inputs = []         # (path, expr, type): pinned values the constraints read
    for n, e, lt in rand_lists + fixed:
        slots[n] = len(slots)
        inputs.append((n, e, lt))
    resolve = CC.slot_resolver(slots)

    softs, cons = [], []
    try:
        for name, tt in roots:
            dt = _struct_dt(types, tt)
            if dt is None:
                continue
            fake = SimpleNamespace(name=name, datatype=dt, kind=None)
            for seq, (prefix, fn) in enumerate(CC._struct_sites([fake], types, ())):
                for st in getattr(fn, "body", []) or []:
                    if D.is_default(st):
                        raise LoweringError("randomize: a 'default' constraint in "
                                            "a struct is not supported here yet",
                                            loc=_loc(st))
                    for c in CC.stmt_to_constraints(st, resolve, fn, prefix, soft=True):
                        if isinstance(c, C.ConstraintSoft):
                            softs.append(((-len(prefix), seq, len(softs)), c))
                        cons.append(c)
        ranked = [c for _, c in sorted(softs, key=lambda kc: kc[0])]
        keep = frozenset(n for n, _, _ in rand_lists)
        for st in s.constraints:
            if isinstance(st, S.StmtForeach):
                st = _as_self(_pin_inputs(ctx, st, slots, inputs, keep))
                for c in CC.stmt_to_constraints(st, resolve, None, (), soft=True):
                    if isinstance(c, C.ConstraintSoft):
                        ranked.append(c)
                    cons.append(c)
                continue
            e = st.expr if isinstance(st, S.StmtExpr) else None
            if e is None:
                raise LoweringError(f"randomize: {_cn(st)} in a 'with' block is "
                                    f"not supported", loc=_loc(st) or _loc(s))
            e = _pin_inputs(ctx, e, slots, inputs, keep)
            for c in CC.expr_to_constraints(_as_self(e), resolve, (), soft=True):
                if getattr(c, "loc", None) is None:
                    c.loc = _loc(st)        # an unsat names it (F3.8)
                if isinstance(c, C.ConstraintSoft):
                    ranked.append(c)        # a `with` soft outranks the type's
                cons.append(c)
    except UnsupportedConstructError as e:
        raise LoweringError(str(e), loc=getattr(e, "loc", None) or _loc(s)) from None
    for rank, c in enumerate(ranked):
        c.rank = rank
    for c in cons:
        bad = _unresolved_ref(c)
        if bad is not None:
            what = _target_path(bad) if isinstance(
                bad, (E.ExprRefLocal, E.ExprRefUnresolved, E.ExprAttribute)) else _cn(bad)
            raise LoweringError(f"randomize: the 'with' constraints read {what!r}, "
                                f"which bc cannot pin as an input of this randomize",
                                loc=_loc(s))

    vars_ = []
    for i, (n, lt, _, _) in enumerate(leaves):
        vars_.append(ScSolveVar(name=n, var_id=i, slot=i, width=lt.width,
                                signed=lt.signed))
        if lt.kind == "enum" and lt.items:
            cons.append(C.ConstraintExpr(expr=E.ExprIn(
                value=E.ExprRefField(base=E.TypeExprRefSelf(), index=i),
                container=E.ExprRangeList(ranges=[
                    E.ExprRange(lower=E.ExprConstant(value=v), upper=None)
                    for v in sorted({v for _, v in lt.items})]))))
    # A list the constraints read (``x in l``, ``l.size()``) is an input
    # array (F6.10d), not a variable: the scalar inputs are numbered without
    # the lists, and each list's handle is pinned at var id (number of
    # variables + its array id), which SOLVE reads as the array's elements.
    scalars = [(slot, x) for slot, x in enumerate(inputs, len(leaves))
               if x[2].kind not in COLLECTIONS]
    lists = [(slot, x) for slot, x in enumerate(inputs, len(leaves))
             if x[2].kind in COLLECTIONS]
    pins = []
    for i, (slot, (n, e, tt)) in enumerate(scalars, len(leaves)):
        # with arrays, dv-solve's handle takes the pins as state: inputs
        vars_.append(ScSolveVar(name=n, var_id=i, slot=slot, width=tt.width or 1,
                                signed=tt.signed, input=bool(lists)))
        r, _ = ev(ctx, e)
        pins.append((i, r))
    arrays = []
    for aid, (slot, (n, e, tt)) in enumerate(lists):
        el = tt.elem
        if any(n == r for r, _, _ in rand_lists) and (
                tt.kind != "list" or el is None or el.kind not in ("int", "bool", "enum")):
            raise LoweringError(f"randomize: {n!r} is a {_tname(tt)}; bc randomizes "
                                f"a list of integers only", loc=_loc(s))
        if tt.kind != "list" or el is None or el.kind not in ("int", "bool", "enum"):
            raise LoweringError(f"randomize: the 'with' constraints read {n!r}, a "
                                f"{_tname(tt)}; bc solves over a list of integers "
                                f"only", loc=_loc(s))
        arrays.append(ScSolveArray(name=n, array_id=aid, slot=slot,
                                   width=el.width or 1, signed=el.signed,
                                   rand=any(n == r for r, _, _ in rand_lists)))
        r, _ = ev(ctx, e)
        pins.append((len(vars_) + aid, r))
    sp = ScSolveProblem(vars=vars_, constraints=cons, writeback={},
                        input_arrays=arrays)
    blob, _ = build_solve_blob(sp)
    pid = ctx.lowerer.add_problem(SolveProblem(
        var_names=[v.name for v in vars_], problem_bytes=blob, source=sp,
        procedural=True,
        input_arrays={len(vars_) + a.array_id: a.array_id for a in arrays},
        array_specs=[(a.width, a.signed) for a in arrays],
        rand_arrays={a.array_id for a in arrays if a.rand}))
    for i, r in pins:
        ctx.emit(Op.SOLVE_PIN, (i, r))
    ctx.emit(Op.SOLVE, (pid,))
    for i, (n, lt, loc, tgt) in enumerate(leaves):
        r = ctx.new_reg()
        ctx.emit(Op.SOLVE_GET, (r, i))
        if loc is not None:
            _store_loc(ctx, loc, r)
        else:
            _store(ctx, tgt, r, lt)


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
    if _is_list(ctx, s.iter):
        return _lower_foreach_list(ctx, s)
    arr = _comp_array(ctx, s.iter)
    if arr is None:
        n = _array_len(ctx, s.iter)
        if n is None:
            raise LoweringError("foreach over anything but an array is not "
                                "supported by bc", loc=_loc(s))
        return _lower_foreach_data(ctx, s, n)
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


def _lower_foreach_data(ctx: CoroCtx, s, n: int) -> None:
    """20.7.8 over a data array of *n* elements: unrolled like a component
    array; the index is a constant and the value iterator stands for
    ``iter[k]`` (a row of a two-dimensional array is itself an array), so
    reads and writes reach the element."""
    st = proc_state(ctx)
    idx_name = s.index_var.name if isinstance(s.index_var, E.ExprRefLocal) else None
    it_name = s.target.name if isinstance(s.target, E.ExprRefLocal) else None
    loop = _Loop()
    st.loops.append(loop)
    try:
        for k in range(n):
            body = s.body
            if it_name is not None and it_name != idx_name:
                elem = E.ExprSubscript(value=s.iter, slice=E.ExprConstant(value=k))
                body = _replace_local(body, it_name, elem)
            _push_scope(ctx)
            try:
                if idx_name is not None:
                    v = _declare(ctx, idx_name, I32)
                    v.const = k
                    ctx.emit(Op.ST_LOCAL, (_const(ctx, k), ctx.local_slot(v.slot)))
                first = len(loop.continues)
                _lower_block(ctx, body)
                for at in loop.continues[first:]:
                    _patch(ctx, at, len(ctx.code))
            finally:
                _pop_scope(ctx)
    finally:
        st.loops.pop()
    for at in loop.breaks:
        _patch(ctx, at, len(ctx.code))


_IR_MODULES = (E.__name__, S.__name__, ExprList.__module__)


def _replace_local(x, name: str, repl):
    """*x* (statements or expressions) with each reference to local *name*
    replaced by *repl*; IR nodes only, the rest is shared."""
    if isinstance(x, E.ExprRefLocal):
        return repl if x.name == name else x
    if isinstance(x, list):
        return [_replace_local(y, name, repl) for y in x]
    if isinstance(x, tuple):
        return tuple(_replace_local(y, name, repl) for y in x)
    if not dc.is_dataclass(x) or isinstance(x, type) \
            or type(x).__module__ not in _IR_MODULES:
        return x
    changes = {}
    for f in dc.fields(x):
        if not f.init:
            continue
        old = getattr(x, f.name)
        new = _replace_local(old, name, repl)
        if new is not old:
            changes[f.name] = new
    return dc.replace(x, **changes) if changes else x


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
    for k, case in enumerate(s.cases):
        pred = _pattern_pred(ctx, case.pattern, subj, st_)
        if pred is None:
            default = case
        else:
            arms.append((pred, case, str(k)))

    # d) more than one match is an error; g) no match and no default is an error.
    count = _const(ctx, 0)
    for pred, _, _ in arms:
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
    for pred, case, label in arms:
        miss = ctx.emit(Op.BRZ, (pred, 0))
        _arm(ctx, label, case.body, node=case)
        ends.append(ctx.emit(Op.BR, (0,)))
        _patch(ctx, miss, len(ctx.code))
    if default is not None:
        _arm(ctx, "default", default.body, node=default)
    for j in ends:
        _patch(ctx, j, len(ctx.code))


def lower_entry_function(symbol: str, lowerer):
    """The entry of function *symbol* (``<comp type>::<fn>``): its called
    form, the parameters leading locals loaded from ``LD_ARG`` (pssc P1.3).
    If the root type is ``<comp type>`` or derives from it, the function is
    looked up virtually from the root type, so a base's test function runs
    its override; otherwise ``<comp type>`` must be the type of an instance
    of the tree and the function is looked up from it. The coroutine runs
    in the instance its frame is put in (component slots are relative), so
    one entry serves every instance of its type. A struct parameter is one
    parameter per leaf (``desc.addr``), in declaration order. Returns
    ``(coroutine index, [(param name, type descriptor)], return descriptor
    or None, component type)``."""
    tree = getattr(lowerer, "comp_tree", None)
    if tree is None:
        raise LoweringError(f"entry {symbol!r}: the model has no component tree")
    owner, _, name = symbol.rpartition("::")
    root = tree.instances[0]
    if owner and lowerer.comps.is_a(root.type_qname, owner):
        type_q = root.type_qname
    elif owner and any(i.type_qname == owner for i in tree.instances):
        type_q = owner
    else:
        raise LoweringError(f"entry {symbol!r}: {owner!r} is not the type of a "
                            f"component instance, nor a base of the root "
                            f"component {root.type_qname!r}")
    fn = lowerer.comps.function(type_q, name, lowerer.functions)
    if fn is None:
        raise LoweringError(f"entry {symbol!r}: no such function")
    types = lowerer.types
    params = []
    for pname, ann, _ in _params(fn, None):
        pt = value_type(ann, types)
        if not isinstance(pt, StructT) and pt.kind in COLLECTIONS:
            raise LoweringError(f"entry {symbol!r} takes a {pt.kind}: collections "
                                f"across the host ABI are not supported by bc",
                                loc=_loc(fn))
        if isinstance(pt, StructT):
            params += [(".".join((pname,) + path), lt.descriptor())
                       for path, lt in pt.leaves]
        else:
            params.append((pname, pt.descriptor()))
    ret = None
    if fn.returns is not None:
        rt = value_type(fn.returns, types)
        if isinstance(rt, StructT) or rt.kind in COLLECTIONS:
            what = "struct" if isinstance(rt, StructT) else rt.kind
            raise LoweringError(f"entry {symbol!r} returns a {what}: not "
                                f"supported by bc", loc=_loc(fn))
        ret = rt.descriptor()
    ctx = CoroCtx.create(lowerer, [], coro_name=f"$entry:{symbol}")
    comp = CompCtx(type_q, 0, 0)
    proc_state(ctx).comp_self = comp
    idx = _called_form(ctx, fn, comp)
    return idx, params, ret, type_q


def entry_function_symbols(lowerer) -> List[str]:
    """``<comp type>::<fn>`` for every target function with a body of every
    component type in the tree (its own or inherited; exec blocks are not
    functions), in tree order: what ``entry_functions=["*"]`` expands to."""
    tree = getattr(lowerer, "comp_tree", None)
    if tree is None:
        return []
    out: List[str] = []
    seen = set()
    for inst in tree.instances:
        if inst.type_qname in seen:
            continue
        seen.add(inst.type_qname)
        names: List[str] = []
        for dt in lowerer.comps.chain(inst.type_qname):
            for fn in getattr(dt, "functions", []) or []:
                meta = getattr(fn, "metadata", None) or {}
                if meta.get("exec_kind") or not getattr(fn, "body", None):
                    continue
                if fn.name not in names:
                    names.append(fn.name)
        out.extend(f"{inst.type_qname}::{n}" for n in names)
    return out


def comp_ref_slots(tree, lowerer) -> List[int]:
    """The absolute slot of each component `ref` field in *tree* (F6.8b)."""
    out = []
    for inst in tree.instances:
        lay = lowerer.comps.get(inst.type_qname)
        for i, (_, leaf) in enumerate(lay.slots):
            if type(leaf.datatype).__name__ == "DataTypeEntityRef":
                out.append(inst.base + i)
    return sorted(set(out))


def lower_comp_init(tree, lowerer, blocks=None, name="$comp_init"):
    """The coroutine constructing component tree *tree* (P1.5): each
    ``ScCompInit`` block in order, run in its instance (``self``), each its
    own scope. Its frame runs in instance 0, so an instance's slots are its
    absolute base. *blocks* (default ``tree.init``) and *name* make the
    tree's ``$run_start`` / ``$run_end`` coroutines the same way."""
    from ..model import Block, CoroDescriptor
    ctx = CoroCtx.create(lowerer, [], coro_name=name)
    st = proc_state(ctx)
    if blocks is None:
        for slot in comp_ref_slots(tree, lowerer):    # a `ref` field starts null
            ctx.emit(Op.ST_COMP, (_const(ctx, NULL_REF), slot))
    for blk in (tree.init if blocks is None else blocks):
        inst = tree.instances[blk.instance]
        st.comp_self = CompCtx(inst.type_qname, inst.base, inst.id)
        st.exec_kind = blk.kind
        sr = lowerer.prov.src_ref(blk)
        saved, ctx.cur_src_ref = ctx.cur_src_ref, (sr or ctx.cur_src_ref)
        # coverage: each instance's blocks of a kind, in order, under its type
        ctx.cov_scope = inst.type_qname
        key = f"{inst.id}:{blk.kind}"
        n = ctx.cov_execs.get(key, 0)
        ctx.cov_execs[key] = n + 1
        ctx.cov_path = [blk.kind if n == 0 else f"{blk.kind}{n}"]
        try:
            _lower_block(ctx, blk.stmts)
        finally:
            ctx.cur_src_ref = saved
    return CoroDescriptor(name=name, code=ctx.code,
                          blocks=[Block(idx=0, pc_start=0, pc_end=len(ctx.code),
                                        suspend_op=0)],
                          frame_locals=ctx.frame_locals, src_ref=0,
                          stmts=ctx.stmts, chans=ctx.chans, cov=ctx.cov, regs=ctx.regs,
                          discards=ctx.discards,
                          locals=ctx.debug_locals())


@contextlib.contextmanager
def exec_scope(ctx: CoroCtx, kind: str):
    """Open the scope of the coroutine's exec *kind*. It is one scope however
    the body was split: the ScExecBlocks of a kind and the ScImports between
    them (a local declared before a blocking import is read at it) all see
    the same locals."""
    st = proc_state(ctx)
    saved_kind, st.exec_kind = st.exec_kind, kind
    st.scopes.append(st.exec_scopes.setdefault(kind, {}))
    try:
        yield
    finally:
        st.scopes.pop()
        st.exec_kind = saved_kind


def lower_exec_block(ctx: CoroCtx, exec_block) -> None:
    """Lower a ScExecBlock's Layer-0 statements."""
    sr = ctx.lowerer.prov.src_ref(exec_block)
    saved, ctx.cur_src_ref = ctx.cur_src_ref, (sr or ctx.cur_src_ref)
    kind = getattr(exec_block, "kind", "body")
    # coverage: the coroutine's exec blocks of a kind, in order
    n = ctx.cov_execs.get(kind, 0)
    ctx.cov_execs[kind] = n + 1
    ctx.cov_path.append(kind if n == 0 else f"{kind}{n}")
    try:
        with exec_scope(ctx, kind):
            if kind == "body":
                lower_exec_set(ctx)
            for i, s in enumerate(exec_block.stmts or []):
                _lower_at(ctx, str(i), s)
    finally:
        ctx.cov_path.pop()
        ctx.cur_src_ref = saved
