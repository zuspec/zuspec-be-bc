"""message() formatting (LRM 21.1.1), independent of lowering."""

import pytest

from zuspec.be.bc.interp.fmt import format_message, parse_format

I32 = {"kind": "int", "width": 32, "signed": True}
U8 = {"kind": "int", "width": 8, "signed": False}
U64 = {"kind": "int", "width": 64, "signed": False}
BOOL = {"kind": "bool", "width": 1, "signed": False}
ENUM = {"kind": "enum", "width": 32, "signed": True, "items": [["RED", 0], ["GREEN", 5]]}
STR = {"kind": "string", "width": 64, "signed": False}
M64 = (1 << 64) - 1


def fmt(f, *pairs, strings=()):
    return format_message(f, [d for d, _ in pairs], [v & M64 for _, v in pairs], list(strings))


@pytest.mark.parametrize("f,pairs,want", [
    ("%d", [(I32, -1)], "-1"),
    ("%d", [(I32, 5)], "5"),                         # no padding
    ("%u", [(U64, -1)], "18446744073709551615"),
    ("%x", [(U8, 0xAB)], "ab"),
    ("%X", [(U8, 0xAB)], "AB"),
    ("%#x", [(U8, 0xAB)], "0xab"),
    ("%#x", [(U8, 0)], "0"),                         # '#' only for non-zero
    ("%o", [(U8, 8)], "10"),
    ("%b", [(U8, 5)], "101"),
    ("%#B", [(U8, 5)], "0B101"),
    ("%x", [(I32, -1)], "ffffffff"),                 # unsigned at the argument's width
    ("%5d", [(I32, 42)], "   42"),
    ("%-5d|", [(I32, 42)], "42   |"),
    ("%05d", [(I32, -42)], "-0042"),
    ("%+d", [(I32, 3)], "+3"),
    ("% d", [(I32, 3)], " 3"),
    ("%.3d", [(I32, 7)], "007"),
    ("%.0d", [(I32, 0)], ""),
    ("%n %n", [(BOOL, 1), (BOOL, 0)], "true false"),
    ("%n", [(ENUM, 5)], "GREEN"),
    ("%.2n", [(ENUM, 5)], "GR"),
    ("100%% %d", [(I32, 1)], "100% 1"),
])
def test_formats(f, pairs, want):
    assert fmt(f, *pairs) == want


def test_strings():
    assert fmt("%s=%3s", (STR, 0), (STR, 1), strings=["a", "b"]) == "a=  b"


def test_d_on_unsigned_is_converted_to_signed_at_its_width():
    # 21.1.1 c): "if %d is used for a parameter of an unsigned type, the value is
    # converted to signed type before being formatted".
    assert fmt("%d", (U8, 200)) == "-56"


@pytest.mark.parametrize("bad", ["%", "abc %q", "%%%", "%5%"])
def test_invalid_specifier_is_an_error(bad):
    with pytest.raises(ValueError):
        parse_format(bad)


@pytest.mark.parametrize("f,pairs", [
    ("%s", [(I32, 1)]),
    ("%n", [(I32, 1)]),
    ("%d", [(STR, 0)]),
    ("%d %d", [(I32, 1)]),
])
def test_type_or_count_mismatch_is_an_error(f, pairs):
    with pytest.raises(ValueError):
        fmt(f, *pairs, strings=["s"])
