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
"""

import dataclasses as dc
from typing import Optional, Tuple

from .errors import LoweringError


@dc.dataclass(frozen=True)
class T:
    width: int
    signed: bool
    kind: str = "int"                                  # int | bool | enum | string
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


BOOL = T(1, False, "bool")
I32 = T(32, True)
U64 = T(64, False)
STRING = T(64, False, "string")


def literal_type(v) -> T:
    """Table 21: an unsized decimal constant is int[N], N minimal but >= 32.

    The IR does not carry a literal's radix, so a hex literal (which the LRM
    types ``bit[N]``) is typed as decimal here. A value that fits only as
    unsigned 64 bits (``18446744073709551615``) is typed ``bit[64]``: the
    ``int[65]`` the LRM gives it cannot be held, and every legal use of it
    truncates to 64 bits anyway.
    """
    if isinstance(v, bool):
        return BOOL
    n = v.bit_length() + 1 if v >= 0 else (-v - 1).bit_length() + 1
    if n <= 32:
        return I32
    if n <= 64:
        return T(n, True)
    if 0 <= v < (1 << 64):
        return U64
    raise LoweringError(f"constant {v} needs more than 64 bits; bc holds 64")


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
    raise LoweringError(f"type {cn} ({getattr(dt, 'name', '')}) is not supported "
                        f"by bc procedural code")


def merge(a: T, b: T) -> T:
    """Table 22 for two integer operands: the larger size; signed iff both are."""
    a, b = a.as_int(), b.as_int()
    return T(max(a.width, b.width), a.signed and b.signed)
