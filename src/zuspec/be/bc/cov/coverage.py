"""Code coverage of a debug-lowered model (P2.7b).

A :class:`Coverage` is a be-bc trace sink: give it to a Session
(``events=cov, watch=cov.WATCH``) and it counts the ``stmt`` and ``branch``
events the run raises. Each point has a stable ID, ``<scope>/<path>``:
*scope* is the function (``<comp type>::<fn>``), the action's coroutine
(its exec blocks: ``<action>/body/0``) or the component type (its init
blocks), and *path* the statement's place in its block structure
(``2/then/0``). An edit that only moves lines keeps every ID; the line is
kept as source information only.

The points are read from the model's coverage tables
(``CoroDescriptor.cov``), so a point the run never reached is reported
with 0 hits. A point reached at several pcs (one component type's init
block in each instance) is one point.

:meth:`Coverage.report` takes a ``keep`` predicate on a point's scope
(P2.7c2): which scopes count. The front end gives it (pssc
``coverage.boundary``: the code under test, not the tests, fixtures, mocks,
``pssunit_pkg`` or the standard libraries).

Register coverage (P2.7d) comes from a ``RegisterFake`` platform: the
session tells the fake of its ``Coverage`` and the fake reports each
register access, decoded (``register_access``). Per register instance
path: reads, writes and, per field, value bins (:func:`_bins`). Every
register of the model is reported, read or not; the boundary does not
apply to registers (they are the device's).

Envelope coverage (P5.7): per operation under test (a public function of
a component type the front end's boundary keeps, ``<comp type>::<fn>``), the
calls, a value bin per parameter and one for the result, and the crosses
``@pssunit_pkg::cover_cross`` asks for (the front end gives the operations, pssc ``coverage.envelope_ops``; :func:`_point_bins` gives the bins).
Only the outermost call counts: a call of an operation from within
another (``f`` calling ``g``) is the operation's own business.
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from zuspec.be.bc.trace import events as ev
from zuspec.be.bc.trace.sink import TraceSink

#: the report's format version
VERSION = 1


def point_id(scope: str, path: str) -> str:
    return f"{scope}/{path}" if path else scope


class Coverage(TraceSink):
    """Counts statement and branch-arm hits, and register accesses (a
    ``RegisterFake`` session). ``forward``: a sink every
    event is passed on to (the run's own events sink, if it has one)."""

    #: the events coverage needs (``watch=``)
    WATCH = ev.EV_STMT | ev.EV_BRANCH | ev.EV_CALL

    def __init__(self, model=None, forward: Optional[TraceSink] = None,
                 ops: Optional[Dict[str, Dict]] = None) -> None:
        super().__init__()
        self.forward = forward
        #: id -> point: kind, file, line, hits (+ branch, arm, implicit for an arm)
        self.stmts: Dict[str, Dict[str, Any]] = {}
        self.arms: Dict[str, Dict[str, Any]] = {}
        #: register path -> reads, writes, fields (name -> kind, width, bins)
        self.registers: Dict[str, Dict[str, Any]] = {}
        #: the operations of the envelope (pssc ``coverage.envelope_ops``), by name
        self.ops: Dict[str, Dict] = dict(ops or {})
        #: operation -> calls, params (name -> type, bins), result
        self.envelope: Dict[str, Dict[str, Any]] = {
            name: _envelope_point(spec) for name, spec in self.ops.items()}
        #: covergroup name -> samples, coverpoints, crosses (F4.5;
        #: :mod:`.covergroup`); filled by :meth:`add_covergroups`
        self.covergroups: Dict[str, Dict[str, Any]] = {}
        #: a call event's coroutine name -> its function (``<type>::<fn>``)
        self._fns: Dict[str, str] = {}
        #: (coroutine, depth) of the outermost envelope call in progress
        self._open: Optional[tuple] = None
        #: that call's point bins, by point name (for the crosses)
        self._seen: Dict[str, Optional[str]] = {}
        if model is not None:
            self.add_model(model)

    # -- points -------------------------------------------------------------

    def add_model(self, model) -> None:
        """Every coverage point of ``model``, with 0 hits if new. A point
        already counted keeps its hits and takes ``model``'s file and line:
        a hot reload's new generation (``Session.reload``) adds to the
        same points."""
        prov = getattr(model, "prov", None)
        for coro in model.coros:
            if getattr(coro, "fn_qname", None):
                self._fns[coro.name] = coro.fn_qname
            for kind, _pc, scope, path, sr, implicit in getattr(coro, "cov", None) or ():
                scope = scope or coro.name
                pid = point_id(scope, path)
                table = self.stmts if kind == "stmt" else self.arms
                if pid in table:
                    table[pid].update(_where(prov, sr))
                    continue
                p: Dict[str, Any] = {"scope": scope, "path": path, "hits": 0}
                p.update(_where(prov, sr))
                if kind == "arm":
                    branch, _, arm = path.rpartition("/")
                    p.update(branch=point_id(scope, branch), arm=arm,
                             implicit=bool(implicit))
                table[pid] = p
        for reg in (getattr(model, "registers", None) or {}).values():
            self._register(reg)

    def _register(self, reg: Dict) -> Dict[str, Any]:
        """The point of register ``reg`` (``bc_session._registers``)."""
        r = self.registers.get(reg["path"])
        if r is None:
            r = self.registers[reg["path"]] = {
                "comp": reg["comp"], "type": reg.get("type"),
                "comp_type": reg.get("comp_type"), "reads": 0, "writes": 0,
                "fields": {f["name"]: {"kind": f.get("kind", "RW"),
                                       "width": f["width"],
                                       "bins": dict.fromkeys(_bins(f), 0)}
                           for f in reg["fields"]}}
        return r

    def register_access(self, reg: Dict, op: str, data: int,
                        cleared: int = 0) -> None:
        """A ``RegisterFake`` access to ``reg``: ``data`` the value written
        or read, ``cleared`` the bits an RC/W1C side effect cleared."""
        r = self._register(reg)
        r["reads" if op == "read" else "writes"] += 1
        for f in reg["fields"]:
            bins = r["fields"][f["name"]]["bins"]
            m = (1 << f["width"]) - 1
            v = (data >> f["lsb"]) & m
            hit = [f"bit{i}={(v >> i) & 1}" for i in range(f["width"])]
            if "enum" in f:
                hit += [n for n, e in f["enum"].items() if e == v]
            else:
                hit += ["zero"] * (v == 0) + ["max"] * (v == m)
            if (cleared >> f["lsb"]) & m:
                hit.append("cleared")
            for b in hit:
                if b in bins:
                    bins[b] += 1

    # -- sink ---------------------------------------------------------------

    def emit(self, event) -> None:
        if event.kind == ev.STMT:
            self._hit(self.stmts, event.detail, None)
        elif event.kind == ev.BRANCH:
            d = event.detail
            self._hit(self.arms, d, d.get("branch", "") + "/" + d.get("arm", ""))
        elif event.kind == ev.CALL and self.ops and self._open is None:
            self._call(event.detail)
        elif event.kind == ev.RETURN and self._open is not None:
            d = event.detail
            if self._open == (d.get("fn"), d.get("depth")):
                self._open = None
                op = self._fns.get(d.get("fn"), d.get("fn"))
                res = self.envelope[op]["result"]
                if res is not None and "retval" in d:
                    _count(res, self.ops[op]["result"], d["retval"])
                    if isinstance(d["retval"], int):
                        self._seen["result"] = _point_bin(
                            self.ops[op]["result"], d["retval"])
                        _cross(self.envelope[op], self._seen, True)
        if self.forward is not None:
            self.forward.emit(event)

    def _hit(self, table, d, path) -> None:
        if path is None:
            path = d.get("path")
            if path is None:            # a model lowered without coverage
                return
        scope = d.get("scope", "")
        pid = point_id(scope, path)
        p = table.get(pid)
        if p is None:                   # a model not given to add_model
            p = table[pid] = {"scope": scope, "path": path, "hits": 0}
        p["hits"] += 1

    def _call(self, d) -> None:
        op = self._fns.get(d.get("fn"), d.get("fn"))
        spec = self.ops.get(op)
        if spec is None:
            return
        self._open = (d.get("fn"), d.get("depth"))
        e = self.envelope[op]
        e["calls"] += 1
        args = d.get("args") or ()
        self._seen = {}
        for i, p in enumerate(spec["params"]):
            if i < len(args) and p["name"] in e["params"]:
                _count(e["params"][p["name"]], p, args[i])
                if isinstance(args[i], int):
                    self._seen[p["name"]] = _point_bin(p, args[i])
        _cross(e, self._seen, False)

    def merge(self, report: Dict[str, Any]) -> None:
        """Add another ``report()``'s points and hits (``pssc test`` counts
        each test apart, a ``-j`` worker's in its own process)."""
        for key, table in (("statements", self.stmts), ("arms", self.arms)):
            for pid, p in report.get(key, {}).items():
                mine = table.get(pid)
                if mine is None:
                    table[pid] = dict(p)
                else:
                    mine["hits"] += p.get("hits", 0)
        for path, r in report.get("registers", {}).items():
            mine = self.registers.get(path)
            if mine is None:
                self.registers[path] = json.loads(json.dumps(r))
                continue
            mine["reads"] += r["reads"]
            mine["writes"] += r["writes"]
            for name, f in r["fields"].items():
                bins = mine["fields"].setdefault(
                    name, dict(f, bins={}))["bins"]
                for b, n in f["bins"].items():
                    bins[b] = bins.get(b, 0) + n
        for op, e in report.get("envelope", {}).items():
            mine = self.envelope.get(op)
            if mine is None:
                self.envelope[op] = json.loads(json.dumps(e))
                continue
            mine["calls"] += e["calls"]
            pts = [(mine["params"].setdefault(n, dict(p, bins={})), p)
                   for n, p in e["params"].items()]
            if e.get("result") is not None:
                if mine.get("result") is None:
                    mine["result"] = dict(e["result"], bins={})
                pts.append((mine["result"], e["result"]))
            for k, x in (e.get("crosses") or {}).items():
                pts.append((mine.setdefault("crosses", {}).setdefault(
                    k, dict(x, bins={})), x))
            for m, p in pts:
                for b, n in p["bins"].items():
                    m["bins"][b] = m["bins"].get(b, 0) + n
        self.add_covergroups(report.get("covergroups"))

    def add_covergroups(self, cgs: Optional[Dict[str, Any]]) -> None:
        """Add a run's binned covergroups (``RunResult.covergroups``)."""
        from .covergroup import merge as _merge
        _merge(self.covergroups, cgs or {})

    # -- report -------------------------------------------------------------

    def report(self, keep: Optional[Callable[[str], bool]] = None) -> Dict[str, Any]:
        """The counts as a JSON-ready dict: every point by ID, and a summary
        (points hit / points) per kind. ``keep``: a point's scope -> whether
        it counts (pssc ``coverage.boundary``); default, every point."""
        def table(t):
            return {k: dict(v) for k, v in sorted(t.items())
                    if keep is None or keep(v["scope"])}

        def summary(t):
            return {"total": len(t),
                    "hit": sum(1 for p in t.values() if p["hits"])}
        stmts, arms = table(self.stmts), table(self.arms)
        regs = json.loads(json.dumps(dict(sorted(self.registers.items()))))
        bins = [n for r in regs.values() for f in r["fields"].values()
                for n in f["bins"].values()]
        env = json.loads(json.dumps({k: v for k, v in sorted(self.envelope.items())
                                     if keep is None or keep(k)}))
        ebins = [n for e in env.values()
                 for p in list(e["params"].values()) + [e.get("result")]
                 + list((e.get("crosses") or {}).values())
                 if p is not None for n in p["bins"].values()]
        from .covergroup import counted_bins
        cgs = json.loads(json.dumps(dict(sorted(self.covergroups.items()))))
        cbins = counted_bins(cgs)
        return {
            "version": VERSION,
            "statements": stmts,
            "arms": arms,
            "registers": regs,
            "envelope": env,
            "covergroups": cgs,
            "summary": {"statements": summary(stmts),
                        "arms": summary(arms),
                        "registers": {"total": len(regs), "hit": sum(
                            1 for r in regs.values() if r["reads"] or r["writes"])},
                        "bins": {"total": len(bins),
                                 "hit": sum(1 for n in bins if n)},
                        "envelope": {"total": len(ebins),
                                     "hit": sum(1 for n in ebins if n)},
                        "covergroups": {"total": len(cbins),
                                        "hit": sum(1 for n in cbins if n)}},
        }


def from_traces(paths: Iterable[str], model=None
                ) -> Tuple[Coverage, List[Dict[str, Any]]]:
    """The coverage of recorded runs (F2.4): each trace at ``paths``
    (NDJSON or the binary form; a simulation's ``+pssvm.trace=DIR`` files,
    the C engine's ``trace=``, ``pssc test --trace``) replayed into a
    :class:`Coverage` on ``model``'s points (``None``: only the points the
    traces reach). Returns it and a test per trace for ``ucis.to_ucis``
    (``name``: the header's test, else the file's stem; ``seed``; its own
    ``coverage``; no status: a trace does not record one). With ``model``,
    each ``access`` to one of its registers (``reg_id``) counts as that
    register's read or write; RC/W1C clears are not in a trace; its ``cov``
    records are binned against its covergroups (``model.covgs``)."""
    from zuspec.be.bc.trace import binary
    from .covergroup import bin_samples, samples_from_trace
    covgs = getattr(model, "covgs", None) or []
    total, tests = Coverage(model), []
    for path in paths:
        _, header, records = binary.load(path)
        one = Coverage(model)
        regs = getattr(model, "registers", None) or {}
        for r in records:
            d = r.get("detail") or {}
            if r.get("kind") == "access" and d.get("reg_id") in regs:
                one.register_access(regs[d["reg_id"]],
                                    "read" if d.get("rw") == "r" else "write",
                                    d.get("data", 0))
            if ev.BITS.get(r.get("kind"), 0) & Coverage.WATCH:
                one.event(r["kind"], coro=r.get("coro", 0),
                          src_ref=r.get("src_ref", 0), detail=r.get("detail"))
        if covgs:
            one.add_covergroups(bin_samples(covgs, samples_from_trace(records)))
        rep = one.report()
        total.merge(rep)
        name = header.get("test") or os.path.splitext(os.path.basename(path))[0]
        tests.append({"name": name, "status": None, "seed": header.get("seed"),
                      "coverage": rep})
    return total, tests


def _point_bins(spec: Dict) -> list:
    """A point's bins: each enum item; ``false``/``true``; for an int of up
    to 3 bits each value, else ``zero``, ``max``, ``other`` (and, signed,
    ``min`` and ``neg``). Each value falls in exactly one bin."""
    k = spec["kind"]
    if k == "enum":
        return list(spec["items"])
    if k == "bool":
        return ["false", "true"]
    w, sg = spec["width"], spec["signed"]
    if w <= 3:
        lo, hi = (-(1 << (w - 1)), (1 << (w - 1)) - 1) if sg else (0, (1 << w) - 1)
        return [str(v) for v in range(lo, hi + 1)]
    return ["min", "neg", "zero", "other", "max"] if sg \
        else ["zero", "other", "max"]


def _point_bin(spec: Dict, v: int) -> Optional[str]:
    k = spec["kind"]
    if k == "enum":
        return next((n for n, e in spec["items"].items() if e == v), None)
    if k == "bool":
        return "true" if v else "false"
    w, sg = spec["width"], spec["signed"]
    v &= (1 << w) - 1
    if sg and v >> (w - 1):
        v -= 1 << w
    if w <= 3:
        return str(v)
    if sg:
        lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
        return ("min" if v == lo else "max" if v == hi else "zero" if v == 0
                else "neg" if v < 0 else "other")
    return "zero" if v == 0 else "max" if v == (1 << w) - 1 else "other"


def _envelope_point(spec: Dict) -> Dict[str, Any]:
    res = spec.get("result")
    pts = {p["name"]: p for p in spec["params"] if p.get("bins")}
    if res is not None:
        pts["result"] = res
    return {"calls": 0,
            "params": {p["name"]: {"type": p["type"],
                                   "bins": dict.fromkeys(_point_bins(p), 0)}
                       for p in spec["params"] if p.get("bins")},
            "result": None if res is None else {
                "type": res["type"], "bins": dict.fromkeys(_point_bins(res), 0)},
            "crosses": {",".join(c): {"points": list(c), "bins": dict.fromkeys(
                (",".join(t) for t in itertools.product(
                    *(_point_bins(pts[n]) for n in c))), 0)}
                for c in spec.get("crosses") or ()}}


def _cross(e: Dict, seen: Dict[str, Optional[str]], result: bool) -> None:
    """Count a call's cross bins: those without the result at the call
    (``result`` False), those with it at the return."""
    for x in (e.get("crosses") or {}).values():
        if ("result" in x["points"]) != result:
            continue
        b = [seen.get(n) for n in x["points"]]
        if None not in b:
            k = ",".join(b)
            if k in x["bins"]:
                x["bins"][k] += 1


def _count(point: Dict, spec: Dict, v) -> None:
    if not isinstance(v, int):
        return
    b = _point_bin(spec, v)
    if b in point["bins"]:
        point["bins"][b] += 1


def _bins(field: Dict) -> list:
    """A register field's value bins: each bit seen 0 and 1 (``bit<i>=0``,
    ``bit<i>=1``: toggle coverage), then each enum item, or ``zero`` and
    ``max``; ``cleared`` for an RC or W1C field (its side effect seen)."""
    out = [f"bit{i}={b}" for i in range(field["width"]) for b in (0, 1)]
    out += list(field["enum"]) if "enum" in field else ["zero", "max"]
    if field.get("kind") in ("RC", "W1C"):
        out.append("cleared")
    return out


def _where(prov, sr) -> Dict[str, Any]:
    if not sr or prov is None:
        return {}
    try:
        e = prov.entries[sr]
    except (IndexError, KeyError, TypeError):
        return {}
    return {"file": e.file, "line": e.line}
