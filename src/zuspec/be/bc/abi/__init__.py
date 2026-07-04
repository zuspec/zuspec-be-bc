"""
zuspec.be.bc.abi -- the value/layout ABI (design D§11).

This is the single most important cross-tier contract: solve results, exec-block
mutations, SV pack/unpack, and the serialized ``.zbc`` must all agree on scalar
encoding, aggregate packing, endianness, the field->solver-var mapping, and the
representation of values wider than 64 bits.

:mod:`.value` is the machine-readable descriptor (the single source the ZBC field
slots, the solver var-map, and the future SV pack/unpack are derived from).
:mod:`.codec` is the Python pack/unpack used by the oracle and the tests.
"""

from .value import (
    ABI_ID,
    Packing,
    ScalarType,
    ArrayType,
    StructType,
    ValueType,
    solver_var_map,
)
from .codec import (
    encode_scalar,
    decode_scalar,
    encode,
    decode,
    const_pool_record,
    is_inline,
)

__all__ = [
    "ABI_ID",
    "Packing",
    "ScalarType",
    "ArrayType",
    "StructType",
    "ValueType",
    "solver_var_map",
    "encode_scalar",
    "decode_scalar",
    "encode",
    "decode",
    "const_pool_record",
    "is_inline",
]
