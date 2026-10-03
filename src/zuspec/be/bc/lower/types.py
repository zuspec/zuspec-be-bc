"""
types.py -- the static types procedural lowering computes (LRM 8.7).

Registers are 64 bits wide and untyped. The *lowering* knows every value's PSS
type and keeps each register **canonical** for it: an N-bit signed value is held
sign-extended to 64 bits, an unsigned one zero-extended, a ``bool`` as 0/1. With
that invariant the 64-bit ADD/SUB/MUL/AND/OR/XOR/SHL are correct once their
result is re-normalized to the expression's type, and only compare, divide,
modulus and right shift need signed variants -- which :mod:`.procedural` builds
from the existing opcodes, so the ISA (and the native engine that mirrors it)
does not change.

Widths above 64 bits cannot be held and are rejected, never truncated silently.

A ``struct`` is not a register value: it is a :class:`StructT`, a list of
scalar leaves laid out by ir-core's ``pss_lower.layout`` (the same layout the
action object uses), and procedural code moves it leaf by leaf.
"""

import dataclasses as dc
from typing import Optional, Tuple

from .errors import LoweringError


@dc.dataclass(frozen=True)
class T:
    width: int
    signed: bool
    kind: str = "int"              # int | bool | enum | string | chandle | opaque
    items: Optional[Tuple[Tuple[str, int], ...]] = None  # enum: (name, value)
    name: Optional[str] = None

    @property
    def is_bool(self) -> bool:
        return self.kind == "bool"

    def as_int(self) -> "T":
        """The integer type arithmetic sees (an enum is its value; bool is bit)."""
        if self.kind in ("int", "string"):
            return self
        return T(self.width, self.signed)

    def descriptor(self) -> dict:
        """What the runtime formatter needs to print a value of this type."""
        d = {"kind": self.kind, "width": self.width, "signed": self.signed}
        if self.items is not None:
            d["items"] = [list(i) for i in self.items]
        return d


@dc.dataclass(frozen=True)
class StructT:
    """A struct value's type: its scalar leaves, in layout order.

    Two struct values are compatible when their leaves are: the front end has
    already checked that they are the same struct type (LRM 8.5.3).
    """
    leaves: Tuple[Tuple[Tuple[str, ...], T], ...]
    name: Optional[str] = None

    kind = "struct"
    is_bool = False
    signed = False

    def as_int(self):
        raise LoweringError(f"struct {self.name or ''} used where a scalar is needed")

    def sub(self, attr: str):
        """The type of field *attr*: a scalar ``T`` or a nested ``StructT``,
        with the index of its first leaf; None if there is no such field."""
        idx = [i for i, (p, _) in enumerate(self.leaves) if p[0] == attr]
        if not idx:
            return None
        if len(idx) == 1 and len(self.leaves[idx[0]][0]) == 1:
            return self.leaves[idx[0]][1], idx[0]
        return StructT(tuple((p[1:], t) for p, t in
                             (self.leaves[i] for i in idx))), idx[0]

    def descriptor(self) -> dict:
        raise LoweringError(f"struct {self.name or ''} cannot be formatted by message()")


BOOL = T(1, False, "bool")
I32 = T(32, True)
U64 = T(64, False)
STRING = T(64, False, "string")
CHANDLE = T(64, False, "chandle")


def literal_type(v, width: int = 0, signed=None) -> T:
    """Table 21, from ir-core's ``int_literal_type``: an unsized decimal
    constant is int[N] and an unsized hex or binary one bit[N], N minimal but
    >= 32; a sized one (``8'hFF``) is its size. *width* and *signed* are the
    ``ExprConstant``'s. A value that fits only as unsigned 64 bits
    (``18446744073709551615``) is typed ``bit[64]``: the ``int[65]`` the LRM
    gives it cannot be held, and every legal use of it truncates to 64 bits
    anyway.
    """
    if isinstance(v, bool):
        return BOOL
    from zuspec.ir.core.expr import ExprConstant, int_literal_type
    w, s = int_literal_type(ExprConstant(value=v, width=width, signed=signed))
    if w > 64:
        raise LoweringError(f"constant {v} needs more than 64 bits; bc holds 64")
    return I32 if (w, s) == (32, True) else T(w, s)


def from_datatype(dt) -> T:
    """The :class:`T` of a Layer-0 ``DataType``."""
    cn = type(dt).__name__
    if cn == "DataTypeInt":
        if getattr(dt, "name", None) == "bool":
            return BOOL
        bits = getattr(dt, "bits", None)
        if bits is None or bits <= 0:
            bits = 32
        if bits > 64:
            raise LoweringError(f"bit width {bits} exceeds the 64 bits bc holds")
        return T(bits, bool(getattr(dt, "signed", False)))
    if cn == "DataTypeEnum":
        items = tuple((k, int(v)) for k, v in (getattr(dt, "items", {}) or {}).items())
        return T(32, True, "enum", items, getattr(dt, "name", None))
    if cn == "DataTypeString":
        return STRING
    if cn == "DataTypeRef":
        # A type nothing resolves: a struct template's parameter (``TRAIT
        # trait`` in ``addr_region_s<TRAIT>``), since pssc does not specialize
        # struct templates yet. It holds one opaque slot, which a struct copy
        # moves; reading or writing it is refused where it happens.
        return T(64, False, "opaque", name=getattr(dt, "ref_name", None))
    if cn == "DataTypeChandle":
        # addr_handle_t (21.13.3): bc models only transparent address spaces,
        # where a handle is its address (bc procedural gaps B-D6).
        return CHANDLE
    raise LoweringError(f"type {cn} ({getattr(dt, 'name', '')}) is not supported "
                        f"by bc procedural code")


def value_type(dt, types=None):
    """The type of a value of Layer-0 type *dt*: :class:`StructT` for a
    plain-data struct, else :func:`from_datatype`."""
    from zuspec.ir.core.xf.pss_lower import layout
    if not layout.is_struct(dt, types):
        return from_datatype(dt)
    st = layout.resolve(dt, types)
    leaves = []
    for leaf in layout.value_leaves(st, types):
        try:
            leaves.append((leaf.path, from_datatype(leaf.datatype)))
        except LoweringError as e:
            raise LoweringError(f"struct {getattr(st, 'name', '')!s} field "
                                f"{leaf.name}: {e}")
    return StructT(tuple(leaves), getattr(st, "name", None))


def merge(a: T, b: T) -> T:
    """Table 22 for two integer operands: the larger size; signed iff both are."""
    a, b = a.as_int(), b.as_int()
    return T(max(a.width, b.width), a.signed and b.signed)
