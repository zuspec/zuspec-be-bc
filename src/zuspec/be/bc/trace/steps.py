"""Replay steps of a trace, and the first difference of two traces.

Moved from pssc ``testing.replay`` (F1.12): a step is a bus access or a user
import call, in the order the run made them (``steps``); ``same_step`` is the
rule a replay checks each with; ``diff`` finds the first step two traces do
not share. Which imports are the test's own (steps) and which the runtime's
is the caller's to say (``user_import``).
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from .writer import trace_path


def _show(d: Dict[str, Any]) -> str:
    if d.get("kind") == "import":
        s = "import %s()" % d.get("name", "?")
    else:
        s = "%s%s 0x%x" % ("read" if d.get("rw") == "r" else "write",
                           d.get("width", 0), d.get("addr", 0))
        if d.get("rw") == "w" and d.get("data") is not None:
            s += " = 0x%x" % d["data"]
    if d.get("file"):
        s += " (%s:%s)" % (d["file"], d.get("line", "?"))
    return s


def _records(path: str) -> tuple:
    """``(header, events)`` of the trace at ``path``: NDJSON, or the binary
    form (``.pstrace``: the C engine's, a simulation's), told by its magic."""
    from zuspec.be.bc.trace import binary
    with open(path, "rb") as f:
        data = f.read()
    if binary.is_binary(data):
        _, header, events = binary.read(data)
        return header, events
    lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
    if not lines or "zbc_trace" not in json.loads(lines[0]):
        raise ValueError("%s: not a pssc trace (no zbc_trace header)" % path)
    return json.loads(lines[0]), [json.loads(ln) for ln in lines[1:]]


def load_trace(path: str, user_import: Optional[Callable[[str], bool]] = None
               ) -> tuple:
    """``(header, steps)`` of the trace at ``path`` (``_records``): each
    step is an access or a user import, ``detail`` flattened with ``kind``/
    ``file``/``line``; ``irqs``: the interrupts recorded since the step
    before. An import the C engine could not name has the name ``""``.
    ``user_import``: whether an import (by name) is a step (``steps``)."""
    header, events = _records(path)
    return header, steps(events, user_import)


def steps(events, user_import: Optional[Callable[[str], bool]] = None
          ) -> List[Dict[str, Any]]:
    """The replay steps of trace *events* (``load_trace``); each also has
    the event's ``seq``. ``user_import(name)``: whether an import is the
    test's (a step) rather than the runtime's; every import is, by default."""
    out, irqs = [], []
    for e in events:
        kind, d = e.get("kind"), e.get("detail") or {}
        if kind == "notify" and "irq" in d:
            irqs.append(int(d["irq"]))
        elif kind == "access" or (kind == "import" and (
                user_import is None or user_import(d.get("name") or ""))):
            step = dict(d, kind=kind)
            if kind == "import":
                step["name"] = d.get("name") or ""
            if e.get("file"):
                step.update(file=e["file"], line=e.get("line"))
            if "seq" in e:
                step["seq"] = e["seq"]
            if irqs:
                step["irqs"], irqs = irqs, []
            out.append(step)
    return out


def same_step(exp: Optional[Dict[str, Any]], actual: Dict[str, Any],
              reads: bool = False) -> bool:
    """Whether step *actual* is the recorded step *exp*: the same import (by
    its name, without the package), or the same access (rw, width, addr; a
    write's data; with *reads*, a read's data too: ``pssc trace diff``)."""
    if exp is None or exp["kind"] != actual["kind"]:
        return False
    if actual["kind"] == "import":
        return (exp.get("name", "").rpartition("::")[2]
                == actual["name"].rpartition("::")[2])
    return (all(exp.get(k) == actual.get(k) for k in ("rw", "width", "addr"))
            and ((actual["rw"] == "r" and not reads)
                 or exp.get("data") == actual.get("data")))


def diff(a, b, user_import: Optional[Callable[[str], bool]] = None
         ) -> Optional[Dict[str, Any]]:
    """The first step where traces *a* and *b* differ, by replay's rule
    (``same_step``, read data included), or None when they do not: ``{"step"
    (1-based), "a", "b"}``, a side that ended first being None. *a* and *b*:
    trace paths (``load_trace``) or lists of steps (``steps``)."""
    sa, sb = [load_trace(x, user_import)[1] if isinstance(x, str) else x
              for x in (a, b)]
    for n in range(max(len(sa), len(sb))):
        x = sa[n] if n < len(sa) else None
        y = sb[n] if n < len(sb) else None
        if x is None or y is None or not same_step(x, y, reads=True):
            return {"step": n + 1, "a": x, "b": y}
    return None


def trace_for(spec_path: str, test: Optional[str]) -> str:
    """The trace file ``replay:<spec_path>`` gives ``test``: the file
    itself, or ``test``'s file in that directory (``--trace-dir``): its
    ``.ndjson`` or, failing that, its ``.pstrace``."""
    if os.path.isdir(spec_path):
        if test is None:
            raise ValueError("replay:%s is a directory; the test is not known"
                             % spec_path)
        path = trace_path(spec_path, test)
        binary = os.path.splitext(path)[0] + ".pstrace"
        if not os.path.isfile(path) and os.path.isfile(binary):
            path = binary
        if not os.path.isfile(path):
            raise ValueError("replay: no recorded trace for %s (%s)"
                             % (test, path))
        return path
    if not os.path.isfile(spec_path):
        raise ValueError("replay: %s does not exist" % spec_path)
    return spec_path


def recorded_seed(spec_path: str, test: Optional[str] = None) -> Optional[int]:
    """The seed the trace(s) of ``replay:<spec_path>`` were recorded with
    (the header's ``seed``): ``test``'s trace, or with no test, the file
    itself or the first trace in the directory. None if none is known."""
    try:
        if os.path.isdir(spec_path) and test is None:
            names = sorted(n for n in os.listdir(spec_path)
                           if n.endswith((".ndjson", ".pstrace")))
            if not names:
                return None
            path = os.path.join(spec_path, names[0])
        else:
            path = trace_for(spec_path, test)
        seed = _records(path)[0].get("seed")
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return int(seed) if isinstance(seed, int) else None
