"""
value.py -- the machine-readable value-type descriptor (design D§11).

Everything that has to agree on how a value is laid out in bytes -- the ZBC
field slots, the solver var-map, the (future) SV pack/unpack -- derives from the
types described here. There is exactly one description; the codec, the format
emitters, and the solver mapping all read *this*.

Canonical layout (M1):

* **Little-endian, two's-complement.** A scalar of ``width_bits`` W occupies
  ``ceil(W/8)`` bytes, least-significant byte first. Signed values are stored in
  two's complement; the top unused bits of the final byte are the sign extension
  (signed) or zero (unsigned).
* **Byte-granular packing.** Aggregates concatenate their element encodings, each
  element occupying its own ``byte_len`` bytes. Sub-byte *bit* packing (SV's tight
  ``bit[N-1:0]`` layout across field boundaries) is intentionally **out of M1**
  (:class:`Packing`) -- it is an SV-pack/unpack concern (roadmap P3+), and byte
  granularity is what the oracle and the C reader both index cheaply.
* **>64-bit values live in the constant pool.** A scalar wider than 64 bits does
  not fit a ZBC register/slot; its literal bytes go to ``SEC_CONST`` and the code
  stream references them by index (see :func:`.codec.const_pool_record`).

Field <-> solver-var mapping (the D§15.4 open item): a solver ``var_id`` is the
index of the field's name in the **alphabetically sorted** list of the enclosing
scope's randomizable field names. This is not invented here -- it mirrors
``zuspec-solver``'s own assignment
(``var_id = index in sorted(system.variables.keys())``). :func:`solver_var_map`
computes exactly that so lowering and write-back cannot drift from the solver.
"""

import dataclasses as dc
from typing import Dict, Iterable, Tuple, Union

#: Value-ABI version. Stamped into ``zbc_header.abi_id``; the engine must match.
ABI_ID = 1

#: Widest scalar that fits inline in a ZBC register/slot; wider => constant pool.
INLINE_MAX_BITS = 64


class Packing:
    """Packing policy for aggregates. M1 fixes ``BYTE``; ``BIT`` is reserved."""

    #: Each scalar occupies ``ceil(width/8)`` bytes; aggregates concatenate.
    BYTE = "byte"
    #: Sub-byte tight bit-packing (SV ``bit[N-1:0]``). Reserved for P3+.
    BIT = "bit"


@dc.dataclass(frozen=True)
class ScalarType:
    """A packed integral scalar of ``width_bits`` bits."""

    width_bits: int
    signed: bool = False

    def __post_init__(self):
        if self.width_bits <= 0:
            raise ValueError(f"width_bits must be positive, got {self.width_bits}")

    @property
    def byte_len(self) -> int:
        return (self.width_bits + 7) // 8

    @property
    def inline(self) -> bool:
        """True if the value fits a ZBC register/slot (<= 64 bits)."""
        return self.width_bits <= INLINE_MAX_BITS

    @property
    def mask(self) -> int:
        return (1 << self.width_bits) - 1


@dc.dataclass(frozen=True)
class ArrayType:
    """A fixed-length array of ``count`` elements of ``element``."""

    element: "ValueType"
    count: int

    def __post_init__(self):
        if self.count < 0:
            raise ValueError(f"count must be non-negative, got {self.count}")

    @property
    def byte_len(self) -> int:
        return self.element.byte_len * self.count

    @property
    def inline(self) -> bool:
        # Aggregates are never register-inline; they live in field storage.
        return False


@dc.dataclass(frozen=True)
class StructType:
    """An ordered aggregate. ``fields`` is ``((name, ValueType), ...)``.

    Slot storage is in **declaration order** (natural for generated code); the
    solver var-map is derived separately from *sorted* names (see
    :func:`solver_var_map`), so the two never have to be the same order.
    """

    fields: Tuple[Tuple[str, "ValueType"], ...]

    def __post_init__(self):
        # Normalize a plain list/tuple of pairs into a tuple of tuples so the
        # dataclass stays hashable/frozen-friendly.
        object.__setattr__(
            self, "fields", tuple((n, t) for (n, t) in self.fields)
        )

    @property
    def byte_len(self) -> int:
        return sum(t.byte_len for _, t in self.fields)

    @property
    def inline(self) -> bool:
        return False

    def field_type(self, name: str) -> "ValueType":
        for n, t in self.fields:
            if n == name:
                return t
        raise KeyError(name)

    def field_names(self) -> Tuple[str, ...]:
        return tuple(n for n, _ in self.fields)


ValueType = Union[ScalarType, ArrayType, StructType]


def solver_var_map(field_names: Iterable[str]) -> Dict[str, int]:
    """Map field name -> solver ``var_id`` using the solver's own rule.

    ``var_id`` is the index of the name in the alphabetically sorted list of
    names -- identical to ``zuspec-solver``'s
    ``{name: idx for idx, name in enumerate(sorted(names))}``. Keeping this in one
    place is what guarantees that a value written back from ``get_value(var_id)``
    lands in the right field slot.
    """
    return {name: idx for idx, name in enumerate(sorted(field_names))}
