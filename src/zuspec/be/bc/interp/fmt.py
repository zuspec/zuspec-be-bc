"""
fmt.py -- PSS string formatting (LRM 21.1.1), for ``message()``.

``%[flags][width][.precision]format`` with flags ``- + space # 0`` and formats
``d u x X o b B n s`` plus ``%%``. Floating-point formats and ``%p`` are not
supported (bc has no floats or chandles) and are rejected when the format is
registered, not at run time.

Each value arrives as the 64-bit canonical register the lowering produced plus
its static type (:meth:`..lower.types.T.descriptor`), which is what decides how
it prints:

* ``%d`` prints the value as signed. Rule c) says an unsigned argument "is
  converted to signed type before being formatted"; this takes that literally,
  at the argument's own width.
* ``%u %x %X %o %b %B`` print the value as unsigned at the argument's width.
* ``%n`` prints a bool as ``true``/``false`` and an enum as its item name.
* ``%s`` prints a string.
"""

import re
from typing import List, NamedTuple, Optional, Sequence

_SPEC = re.compile(r"%(?P<flags>[-+ #0]*)(?P<width>[0-9]+)?(?:\.(?P<prec>[0-9]*))?"
                   r"(?P<conv>[duxXobBnsp%efgEG])?")

_INT_CONV = set("duxXobB")


class Spec(NamedTuple):
    start: int
    end: int
    flags: str
    width: Optional[int]
    prec: Optional[int]
    conv: str


def parse_format(fmt: str) -> List[Spec]:
    """The format specifiers of *fmt*, excluding ``%%``. Raises ValueError (rule a)."""
    out: List[Spec] = []
    i = 0
    while True:
        j = fmt.find("%", i)
        if j < 0:
            return out
        m = _SPEC.match(fmt, j)
        conv = m.group("conv")
        if conv is None:
            raise ValueError(f"'%' at offset {j} does not start a valid format specifier")
        if conv == "%":
            if m.group("flags") or m.group("width") or m.group("prec") is not None:
                raise ValueError(f"malformed '%%' at offset {j}")
            i = m.end()
            continue
        if conv in "efgEGp":
            raise ValueError(f"%{conv} is not supported by bc")
        prec = m.group("prec")
        out.append(Spec(j, m.end(), m.group("flags"),
                        int(m.group("width")) if m.group("width") else None,
                        (int(prec) if prec else 0) if prec is not None else None,
                        conv))
        i = m.end()


def _unsigned(v: int, width: int) -> int:
    return v & ((1 << width) - 1)


def _signed(v: int, width: int) -> int:
    v = _unsigned(v, width)
    return v - (1 << width) if v & (1 << (width - 1)) else v


def _render_int(spec: Spec, value: int, desc: dict) -> str:
    width = max(1, int(desc.get("width", 64)))
    if spec.conv == "d":
        n = _signed(value, width)
    else:
        n = _unsigned(value, width)
    neg = n < 0
    mag = -n if neg else n
    base = {"d": 10, "u": 10, "x": 16, "X": 16, "o": 8, "b": 2, "B": 2}[spec.conv]
    digits = _digits(mag, base, upper=spec.conv in "XB")
    if spec.prec is not None:
        if spec.prec == 0 and mag == 0:
            digits = ""
        digits = digits.rjust(spec.prec, "0")
    prefix = ""
    if "#" in spec.flags and mag != 0 and spec.conv in "oxXbB":
        prefix = {"o": "0", "x": "0x", "X": "0X", "b": "0b", "B": "0B"}[spec.conv]
    sign = "-" if neg else ("+" if "+" in spec.flags and spec.conv == "d"
                            else (" " if " " in spec.flags and spec.conv == "d" else ""))
    body = prefix + digits
    if spec.width and "0" in spec.flags and "-" not in spec.flags and spec.prec is None:
        body = body.rjust(spec.width - len(sign), "0")
    return _pad(spec, sign + body)


def _digits(n: int, base: int, upper: bool) -> str:
    if n == 0:
        return "0"
    s = ""
    while n:
        s = "0123456789abcdef"[n % base] + s
        n //= base
    return s.upper() if upper else s


def _pad(spec: Spec, s: str) -> str:
    if spec.width is None or len(s) >= spec.width:
        return s
    return s.ljust(spec.width) if "-" in spec.flags else s.rjust(spec.width)


def _render_text(spec: Spec, text: str) -> str:
    if spec.prec is not None:
        text = text[:spec.prec]
    return _pad(spec, text)


def format_message(fmt: str, descs: Sequence[dict], values: Sequence[int],
                   strings: Sequence[str]) -> str:
    specs = parse_format(fmt)
    if len(specs) != len(values):
        raise ValueError(f"{len(specs)} format specifier(s) for {len(values)} argument(s)")
    out = []
    pos = 0
    for spec, desc, v in zip(specs, descs, values):
        out.append(fmt[pos:spec.start].replace("%%", "%"))
        pos = spec.end
        kind = desc.get("kind", "int")
        if spec.conv in _INT_CONV:
            if kind == "string":
                raise ValueError(f"%{spec.conv} cannot format a string")
            out.append(_render_int(spec, v, desc))
        elif spec.conv == "n":
            if kind == "bool":
                out.append(_render_text(spec, "true" if v else "false"))
            elif kind == "enum":
                val = _signed(v, int(desc.get("width", 32)))
                names = [n for n, x in desc.get("items", []) if x == val]
                if not names:
                    raise ValueError(f"%n: {val} is not an item of the enum")
                out.append(_render_text(spec, names[0]))
            else:
                raise ValueError("%n formats only an enum or a bool")
        elif spec.conv == "s":
            if kind != "string":
                raise ValueError("%s formats only a string")
            out.append(_render_text(spec, strings[v]))
    out.append(fmt[pos:].replace("%%", "%"))
    return "".join(out)
