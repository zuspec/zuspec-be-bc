"""Functional coverage: covergroup samples binned against the image's COVG
description (F4.5).

The VM records each ``COV_SAMPLE`` as ``(covg_id, values)``: per coverpoint
its value, then its ``iff`` flag if the description says ``iff``; then one
``iff`` flag per cross that has one (be-bc ``lower/coverage.py``). A
:class:`Binner` per description decodes that list and counts bins.

Bins of a coverpoint (LRM 19.3):

- a value or range bin holds its values (an open end is the coverpoint's
  minimum or maximum); ``with`` keeps the values its expression holds for
  (``item``, or the coverpoint's own name, is the candidate value);
- ``b[]`` is one bin per value, named ``b[<value>]``; ``b[N]`` spreads the
  values over N bins ``b[0]``..``b[N-1]`` (the last takes the remainder);
- ``default`` holds the values no other bin holds;
- ``ignore`` and ``illegal`` bins take their values out of every other bin;
  an illegal value is counted in its bin; :func:`illegal_hits` lists the
  hit illegal bins at their source lines (a run with one fails, F4.8c);
- no ``bins`` bin at all: automatic bins, one per value up to
  ``AUTO_BIN_MAX`` values, else ``AUTO_BIN_MAX`` equal ranges
  (``auto[<v>]`` / ``auto[<lo>:<hi>]``); an enum coverpoint (``items`` in
  its description) has one automatic bin per enumerator, ``auto[<name>]``.

A cross has one automatic bin per combination of its coverpoints' ``bins``
bins, named ``<b1,b2,...>``, and the select bins it declares: a sample
counts in each select bin whose expression holds over the coverpoints'
values; a sample that falls in any select bin (``bins``, ``ignore`` or
``illegal``) does not count in the automatic bins.

The report (``bin_samples``): covergroup name -> ``samples``,
``coverpoints`` (name -> ``bins``: bin -> hits, ``kinds``: bin -> kind for
a bin not of kind ``bins``) and ``crosses`` (name -> ``coverpoints``,
``bins``, ``kinds``). Every bin is listed, hit or not.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: automatic bins of a coverpoint without explicit bins (LRM ``auto_bin_max``)
AUTO_BIN_MAX = 64
#: the most values an array bin may enumerate
MAX_ARRAY_VALUES = 1 << 16


class CovError(ValueError):
    """A covergroup description the binner cannot count."""


def _domain(w: int, signed: bool) -> Tuple[int, int]:
    return ((-(1 << (w - 1)), (1 << (w - 1)) - 1) if signed and w > 0
            else (0, (1 << w) - 1))


def _norm(v: int, w: int, signed: bool) -> int:
    """A register's value as the coverpoint's type sees it."""
    v &= (1 << w) - 1
    if signed and w > 0 and v >> (w - 1):
        v -= 1 << w
    return v


_BIN = {
    "Add": lambda a, b: a + b, "Sub": lambda a, b: a - b,
    "Mult": lambda a, b: a * b,
    "Div": lambda a, b: int(a / b) if b else 0,
    "FloorDiv": lambda a, b: a // b if b else 0,
    "Mod": lambda a, b: (a - b * int(a / b)) if b else 0,
    "BitAnd": lambda a, b: a & b, "BitOr": lambda a, b: a | b,
    "BitXor": lambda a, b: a ^ b, "LShift": lambda a, b: a << b,
    "RShift": lambda a, b: a >> b, "ARShift": lambda a, b: a >> b,
    "Eq": lambda a, b: int(a == b), "NotEq": lambda a, b: int(a != b),
    "Lt": lambda a, b: int(a < b), "LtE": lambda a, b: int(a <= b),
    "Gt": lambda a, b: int(a > b), "GtE": lambda a, b: int(a >= b),
}


def eval_expr(e: Dict[str, Any], env: Dict[str, int]) -> int:
    """A ``with``/select tree (``{"c"}``, ``{"ref"}``, ``{"op", "args"}``)
    with ``env`` giving each name's value."""
    if "c" in e:
        return int(e["c"])
    if "ref" in e:
        if e["ref"] not in env:
            raise CovError(f"covergroup expression names {e['ref']!r}, "
                           f"which is not a coverpoint")
        return env[e["ref"]]
    op, args = e["op"], e["args"]
    if op in ("And", "Or"):
        vals = (eval_expr(a, env) for a in args)
        return int(all(vals) if op == "And" else any(vals))
    if len(args) == 1:
        v = eval_expr(args[0], env)
        if op == "Not":
            return int(not v)
        if op == "Invert":
            return ~v
        if op == "USub":
            return -v
        if op == "UAdd":
            return v
    elif op in _BIN:
        return _BIN[op](eval_expr(args[0], env), eval_expr(args[1], env))
    raise CovError(f"covergroup expression operator {op!r} is not supported")


class _Point:
    """One coverpoint's bins: (name, kind, test) in declaration order."""

    def __init__(self, d: Dict[str, Any]) -> None:
        self.name = d["name"]
        self.iff = bool(d.get("iff"))
        self.signed = bool(d.get("signed"))
        self.width = int(d["width"])
        self.lo, self.hi = _domain(self.width, self.signed)
        self.items = [(n, int(v)) for n, v in d.get("items") or ()]
        self.auto_bin_max = int((d.get("options") or {}).get(
            "auto_bin_max", AUTO_BIN_MAX)) or AUTO_BIN_MAX
        #: (bin name, kind, value -> bool)
        self.bins: List[Tuple[str, str, Any]] = []
        self.default: Optional[str] = None
        for b in d.get("bins", ()):
            kind = b.get("kind", "bins")
            if kind == "default":
                self.default = b["name"]
                continue
            if b.get("array"):
                self._array(b, kind)
            else:
                self.bins.append((b["name"], kind, self._test(b)))
        if not any(k == "bins" for _, k, _ in self.bins):
            self._auto()

    def _ranges(self, b) -> List[Tuple[int, int]]:
        vals = b.get("values") or [[None, None]]
        out = []
        for v in vals:
            lo, hi = (v if isinstance(v, list) else (v, v))
            out.append((self.lo if lo is None else lo, self.hi if hi is None else hi))
        return out

    def _with(self, b):
        w = b.get("with")
        if w is None:
            return lambda v: True
        return lambda v: bool(eval_expr(w, {"item": v, self.name: v}))

    def _test(self, b):
        rs, w = self._ranges(b), self._with(b)
        return lambda v: any(lo <= v <= hi for lo, hi in rs) and w(v)

    def _array(self, b, kind) -> None:
        rs, w = self._ranges(b), self._with(b)
        if sum(max(0, hi - lo + 1) for lo, hi in rs) > MAX_ARRAY_VALUES:
            raise CovError(f"coverpoint {self.name!r} bin {b['name']!r}: more "
                           f"than {MAX_ARRAY_VALUES} values for an array bin")
        vals = list(dict.fromkeys(v for lo, hi in rs for v in range(lo, hi + 1)
                                  if w(v)))
        n = b.get("size")
        if n is None:
            for v in vals:
                self.bins.append((f"{b['name']}[{v}]", kind, (lambda x, v=v: x == v)))
            return
        per = max(1, len(vals) // n) if n else 1
        for i in range(n):
            part = set(vals[i * per:] if i == n - 1 else vals[i * per:(i + 1) * per])
            self.bins.append((f"{b['name']}[{i}]", kind, (lambda x, s=part: x in s)))

    def _auto(self) -> None:
        if self.items:
            for n, v in self.items:
                self.bins.append((f"auto[{n}]", "bins", (lambda x, v=v: x == v)))
            return
        size = self.hi - self.lo + 1
        if size <= self.auto_bin_max:
            for v in range(self.lo, self.hi + 1):
                self.bins.append((f"auto[{v}]", "bins", (lambda x, v=v: x == v)))
            return
        per = size // self.auto_bin_max
        for i in range(self.auto_bin_max):
            lo = self.lo + i * per
            hi = self.hi if i == self.auto_bin_max - 1 else lo + per - 1
            self.bins.append((f"auto[{lo}:{hi}]", "bins",
                              (lambda x, lo=lo, hi=hi: lo <= x <= hi)))

    def names(self) -> List[Tuple[str, str]]:
        out = [(n, k) for n, k, _ in self.bins]
        if self.default is not None:
            out.append((self.default, "default"))
        return out

    def hit(self, v: int) -> List[str]:
        """The bins value ``v`` counts in."""
        for kind in ("ignore", "illegal"):
            hits = [n for n, k, t in self.bins if k == kind and t(v)]
            if hits:
                return hits
        hits = [n for n, k, t in self.bins if k == "bins" and t(v)]
        if not hits and self.default is not None:
            return [self.default]
        return hits


class Binner:
    """Counts one covergroup's samples against its COVG description."""

    def __init__(self, doc: Dict[str, Any]) -> None:
        self.doc = doc
        self.name = doc["name"]
        self.points = [_Point(d) for d in doc.get("coverpoints", ())]
        by = {p.name: p for p in self.points}
        self.crosses = []
        for x in doc.get("crosses", ()):
            cps = [by[n] for n in x["coverpoints"]]
            auto = ["<" + ",".join(c) + ">" for c in itertools.product(
                *[[n for n, k, _ in p.bins if k == "bins"] for p in cps])]
            self.crosses.append((x, cps, auto))
        self.report = self.empty()

    def empty(self) -> Dict[str, Any]:
        """The report with every bin at 0."""
        cps = {}
        for p in self.points:
            names = p.names()
            cps[p.name] = {"bins": {n: 0 for n, _ in names},
                           "kinds": {n: k for n, k in names if k != "bins"}}
        for d in self.doc.get("coverpoints", ()):
            if d.get("options"):
                cps[d["name"]]["options"] = dict(d["options"])
        xs = {}
        for x, cps_, auto in self.crosses:
            sel = [(b["name"], b.get("kind", "bins")) for b in x.get("bins", ())]
            xs[x["name"]] = {"coverpoints": list(x["coverpoints"]),
                             "bins": {n: 0 for n in auto + [n for n, _ in sel]},
                             "kinds": {n: k for n, k in sel if k != "bins"}}
            if x.get("options"):
                xs[x["name"]]["options"] = dict(x["options"])
        r = {"samples": 0, "coverpoints": cps, "crosses": xs}
        if self.doc.get("type"):
            r["type"] = self.doc["type"]
        if self.doc.get("options"):
            r["options"] = dict(self.doc["options"])
        return r

    def sample(self, regs: Sequence[int]) -> None:
        regs = list(regs)
        r = self.report
        r["samples"] += 1
        vals: Dict[str, int] = {}
        hits: Dict[str, Optional[List[str]]] = {}
        for p in self.points:
            v = _norm(regs.pop(0), p.width, p.signed)
            on = regs.pop(0) if p.iff else 1
            vals[p.name] = v
            hits[p.name] = p.hit(v) if on else None
            for n in hits[p.name] or ():
                r["coverpoints"][p.name]["bins"][n] += 1
        for x, cps, _ in self.crosses:
            on = regs.pop(0) if x.get("iff") else 1
            if not on or any(hits[p.name] is None for p in cps):
                continue
            xr = r["crosses"][x["name"]]
            selected = False
            for b in x.get("bins", ()):
                if b.get("select") is None or eval_expr(b["select"], vals):
                    xr["bins"][b["name"]] += 1
                    selected = True
            if selected:
                continue
            per = [[n for n in hits[p.name] if n not in
                    r["coverpoints"][p.name]["kinds"]] for p in cps]
            for c in itertools.product(*per):
                xr["bins"]["<" + ",".join(c) + ">"] += 1


def bin_samples(covgs: Sequence[Dict[str, Any]],
                samples: Iterable[Tuple[int, Sequence[int]]]) -> Dict[str, Any]:
    """Every covergroup of ``covgs`` (the image's COVG docs, by id) with the
    run's ``samples`` counted."""
    binners = [Binner(d) for d in covgs]
    for cid, regs in samples:
        binners[cid].sample(regs)
    return {b.name: b.report for b in binners}


def illegal_hits(covgs: Sequence[Dict[str, Any]],
                 report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The illegal bins of ``report`` (a :func:`bin_samples` report of
    ``covgs``) that were hit: one ``{"covergroup", "coverpoint" or "cross",
    "bin", "hits", "file", "line", "message"}`` per bin declaration (an
    array bin's hits summed), in declaration order."""
    out = []
    for d in covgs:
        r = report.get(d["name"])
        if not r:
            continue
        for key, sect in (("coverpoint", "coverpoints"), ("cross", "crosses")):
            for p in d.get(sect, ()):
                hits = r[sect][p["name"]]["bins"]
                for b in p.get("bins", ()):
                    if b.get("kind") != "illegal":
                        continue
                    n = sum(h for bn, h in hits.items()
                            if bn == b["name"] or bn.startswith(b["name"] + "["))
                    if n:
                        out.append({
                            "covergroup": d["name"], key: p["name"],
                            "bin": b["name"], "hits": n,
                            "file": b.get("file", ""), "line": b.get("line", 0),
                            "message": f"illegal bin {b['name']!r} of {key} "
                                       f"{d['name']}.{p['name']} hit {n} time(s)"})
    return out


def samples_from_trace(recs: Iterable[Dict[str, Any]]
                       ) -> List[Tuple[int, List[int]]]:
    """The covergroup samples of a trace (its ``cov`` records, as either VM
    writes them; a simulator's ``.pstrace`` read with ``trace.binary.read``),
    in the form :func:`bin_samples` takes."""
    return [(r["detail"]["covg"], list(r["detail"]["values"]))
            for r in recs if r.get("kind") == "cov"]


def merge(into: Dict[str, Any], other: Dict[str, Any]) -> None:
    """Add ``other``'s covergroups (a ``bin_samples`` report) to ``into``."""
    for name, cg in other.items():
        mine = into.get(name)
        if mine is None:
            into[name] = json.loads(json.dumps(cg))
            continue
        mine["samples"] += cg["samples"]
        for key in ("coverpoints", "crosses"):
            for pn, p in cg[key].items():
                m = mine[key].setdefault(pn, dict(json.loads(json.dumps(p)),
                                                  bins={}))
                for b, n in p["bins"].items():
                    m["bins"][b] = m["bins"].get(b, 0) + n


def _opt(p: Dict[str, Any], key: str, default: int) -> int:
    return int((p.get("options") or {}).get(key, default))


def grade(cgs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Each covergroup of ``cgs`` (a :func:`bin_samples` report) graded by
    its options (LRM 19.6): a bin is covered when its hits reach
    ``at_least``; a coverpoint's or cross's score is the share of its
    counted bins covered; the covergroup's is the ``weight``-weighted mean
    of those. Per covergroup ``{"pct", "goal", "met", "weight",
    "points": {name: {"covered", "total", "pct", "weight", "goal"}}}``
    (percentages as floats; a point with no counted bins scores 100)."""
    out = {}
    for name, cg in cgs.items():
        pts = {}
        for key in ("coverpoints", "crosses"):
            for pn, p in cg.get(key, {}).items():
                at_least = max(1, _opt(p, "at_least", 1))
                ns = [n for b, n in p["bins"].items()
                      if b not in p.get("kinds", {})]
                hit = sum(1 for n in ns if n >= at_least)
                pts[pn] = {"covered": hit, "total": len(ns),
                           "pct": 100.0 * hit / len(ns) if ns else 100.0,
                           "weight": _opt(p, "weight", 1),
                           "goal": _opt(p, "goal", 100)}
        wsum = sum(p["weight"] for p in pts.values())
        pct = (sum(p["pct"] * p["weight"] for p in pts.values()) / wsum
               if wsum else 100.0)
        goal = _opt(cg, "goal", 100)
        out[name] = {"pct": pct, "goal": goal, "met": pct >= goal,
                     "weight": _opt(cg, "weight", 1), "points": pts}
    return out


def counted_bins(cgs: Dict[str, Any]) -> List[int]:
    """The hits of every bin that counts towards coverage (kind ``bins``,
    not default, ignore or illegal)."""
    return [n for cg in cgs.values()
            for p in list(cg["coverpoints"].values()) + list(cg["crosses"].values())
            for b, n in p["bins"].items() if b not in p.get("kinds", {})]
