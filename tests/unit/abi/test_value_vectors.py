"""T0-E -- value ABI encode/decode vectors (scalars, wide, signed, aggregates)."""

import pytest

from zuspec.be.bc.abi import (
    ScalarType, ArrayType, StructType,
    encode_scalar, decode_scalar, encode, decode,
    is_inline, const_pool_record, solver_var_map,
)


@pytest.mark.parametrize("value,width,signed,expected", [
    (0x7F, 8, False, b"\x7f"),
    (0xFF, 8, False, b"\xff"),
    (0x1FF, 8, False, b"\xff"),          # wrap mod 2^8
    (-1, 8, True, b"\xff"),
    (-128, 8, True, b"\x80"),
    (0xABC, 12, False, b"\xbc\x0a"),      # 12-bit -> 2 bytes
    (0x1234, 16, False, b"\x34\x12"),
    (0x0102030405060708, 64, False,
     b"\x08\x07\x06\x05\x04\x03\x02\x01"),
])
def test_scalar_encode(value, width, signed, expected):
    t = ScalarType(width, signed=signed)
    assert encode_scalar(value, t) == expected


@pytest.mark.parametrize("width,signed", [(8, False), (8, True), (12, False),
                                          (16, True), (32, False), (64, True)])
def test_scalar_roundtrip(width, signed):
    t = ScalarType(width, signed=signed)
    lo = -(1 << (width - 1)) if signed else 0
    hi = (1 << (width - 1)) - 1 if signed else (1 << width) - 1
    for v in (lo, 0, hi):
        assert decode_scalar(encode_scalar(v, t), t) == v


def test_signed_decode_sign_extends():
    assert decode_scalar(b"\xff", ScalarType(8, signed=True)) == -1
    assert decode_scalar(b"\x80", ScalarType(8, signed=True)) == -128
    assert decode_scalar(b"\xff\x0f", ScalarType(12, signed=True)) == -1


def test_wide_value_not_inline_and_pooled():
    t = ScalarType(128, signed=False)
    assert not is_inline(t)
    big = (1 << 100) | 0xABCD
    rec = const_pool_record(big, t)
    assert len(rec) % 8 == 0
    # header: width, byte_len then payload
    width = int.from_bytes(rec[0:4], "little")
    byte_len = int.from_bytes(rec[4:8], "little")
    assert width == 128 and byte_len == 16
    payload = rec[8:8 + byte_len]
    assert decode_scalar(payload, t) == big


def test_array_and_struct():
    arr = ArrayType(ScalarType(16, signed=False), 3)
    b = encode([1, 2, 0x1234], arr)
    assert b == bytes([1, 0, 2, 0, 0x34, 0x12])
    assert decode(b, arr) == [1, 2, 0x1234]

    st = StructType((("a", ScalarType(8)), ("b", ScalarType(16, signed=True))))
    sb = encode({"a": 5, "b": -2}, st)
    assert decode(sb, st) == {"a": 5, "b": -2}


def test_nested_aggregate():
    inner = StructType((("x", ScalarType(8)), ("y", ScalarType(8))))
    arr = ArrayType(inner, 2)
    val = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    assert decode(encode(val, arr), arr) == val


def test_solver_var_map_is_sorted_by_name():
    assert solver_var_map(["z", "a", "m"]) == {"a": 0, "m": 1, "z": 2}
    # Idempotent regardless of input order.
    assert solver_var_map(["m", "z", "a"]) == solver_var_map(["a", "z", "m"])
