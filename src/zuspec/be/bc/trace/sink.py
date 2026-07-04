"""
sink.py -- the trace sink interface the interpreter calls (P0-11).

``TraceSink`` is the minimal surface the VM depends on: ``emit(event)`` plus a
convenience ``event(...)`` that stamps the monotonic ``seq``. Concrete sinks:

* ``MemorySink``  -- collects events in a list (tests, differential comparison).
* ``FileSink``    -- streams canonical JSONL to a file/stream.
* ``NullSink``    -- drops everything (the runtime-profile hot path).
"""

from typing import Any, Dict, List, Optional, TextIO

from .schema import TraceEvent, TRACE_SCHEMA_VERSION, serialize


class TraceSink:
    """Base sink. Subclasses override :meth:`emit`."""

    def __init__(self) -> None:
        self._seq = 0

    def next_seq(self) -> int:
        s = self._seq
        self._seq += 1
        return s

    def event(self, kind: str, coro: int = 0, src_ref: int = 0,
              detail: Optional[Dict[str, Any]] = None) -> TraceEvent:
        """Build a monotonically-sequenced event and emit it."""
        ev = TraceEvent(
            seq=self.next_seq(),
            kind=kind,
            coro=coro,
            src_ref=src_ref,
            detail=detail or {},
        )
        self.emit(ev)
        return ev

    def emit(self, event: TraceEvent) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class NullSink(TraceSink):
    """Drops every event (still advances seq so callers behave identically)."""

    def emit(self, event: TraceEvent) -> None:
        pass


class MemorySink(TraceSink):
    """Collects events in memory; ``serialize()`` yields canonical JSONL."""

    def __init__(self) -> None:
        super().__init__()
        self.events: List[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    def serialize(self) -> str:
        return serialize(self.events)


class FileSink(TraceSink):
    """Streams canonical JSONL to an open text stream (writes header on open)."""

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self._stream = stream
        import json

        self._stream.write(
            json.dumps({"zbc_trace": TRACE_SCHEMA_VERSION},
                       sort_keys=True, separators=(",", ":")) + "\n"
        )

    def emit(self, event: TraceEvent) -> None:
        self._stream.write(event.to_json() + "\n")
