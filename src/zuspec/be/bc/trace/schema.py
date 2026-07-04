"""
schema.py -- the trace event record and its canonical serialization (P0-10).

An event is deliberately small and flat: a monotonic ``seq``, a ``kind`` (one of
the orchestration ops), the emitting frame/coroutine ``coro``, a ``src_ref``
provenance back-pointer, and a ``detail`` mapping for kind-specific payload
(solve results, the SELECT choice index, the IMPORT function id, ...).

The serialized form is **JSONL with a one-line versioned header**. JSON is emitted
with sorted keys and compact separators so the byte stream is a pure function of
the event contents -- the property the determinism spec (D§15.1) and differential
testing depend on.
"""

import dataclasses as dc
import json
from typing import Any, Dict, List, Tuple

#: Trace schema version. Bumped on any incompatible change to the record shape.
TRACE_SCHEMA_VERSION = 1


class EventKind:
    """Orchestration event kinds (mirror the ZBC orchestration opcodes)."""

    SPAWN = "SPAWN"
    INVOKE = "INVOKE"
    PAR = "PAR"
    JOIN = "JOIN"
    WAIT = "WAIT"
    SELECT = "SELECT"
    SOLVE = "SOLVE"
    BIND = "BIND"
    IMPORT = "IMPORT"
    YIELD = "YIELD"

    ALL = frozenset(
        {SPAWN, INVOKE, PAR, JOIN, WAIT, SELECT, SOLVE, BIND, IMPORT, YIELD}
    )


@dc.dataclass
class TraceEvent:
    seq: int
    kind: str
    coro: int = 0
    src_ref: int = 0
    detail: Dict[str, Any] = dc.field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in EventKind.ALL:
            raise ValueError(f"unknown event kind {self.kind!r}")

    def to_dict(self) -> Dict[str, Any]:
        # Fixed key set; detail is nested so top-level order is stable.
        return {
            "seq": self.seq,
            "kind": self.kind,
            "coro": self.coro,
            "src_ref": self.src_ref,
            "detail": self.detail,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceEvent":
        return cls(
            seq=d["seq"],
            kind=d["kind"],
            coro=d.get("coro", 0),
            src_ref=d.get("src_ref", 0),
            detail=d.get("detail", {}),
        )

    @classmethod
    def from_json(cls, line: str) -> "TraceEvent":
        return cls.from_dict(json.loads(line))


_HEADER_KEY = "zbc_trace"


def serialize(events: List[TraceEvent]) -> str:
    """Serialize events to canonical JSONL (header line + one line per event)."""
    header = json.dumps(
        {_HEADER_KEY: TRACE_SCHEMA_VERSION}, sort_keys=True, separators=(",", ":")
    )
    lines = [header]
    lines.extend(e.to_json() for e in events)
    return "\n".join(lines) + "\n"


def parse(text: str) -> Tuple[int, List[TraceEvent]]:
    """Parse canonical JSONL back into ``(version, events)``."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty trace")
    header = json.loads(lines[0])
    if _HEADER_KEY not in header:
        raise ValueError("missing trace header")
    version = header[_HEADER_KEY]
    events = [TraceEvent.from_json(ln) for ln in lines[1:]]
    return version, events
