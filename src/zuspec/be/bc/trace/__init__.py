"""
zuspec.be.bc.trace -- the shared trace / event schema (design D§10).

The interpreter (and, later, the native engine) emit an event-sourced trace of
the orchestration ops -- SPAWN/INVOKE/PAR/JOIN/WAIT/SELECT/SOLVE/BIND/IMPORT --
each tagged with a ``src_ref`` back-pointer. The serialized form is canonical
(stable key order, no whitespace variance) so two runs with the same seed produce
**byte-identical** traces and oracle-vs-engine traces are byte-diffable.
``binary`` is the compact form of the same records (``.pstrace``).
``writer`` (``TraceWriter``), ``steps`` (``diff``: two traces' first difference),
``replay`` and ``sanitizers`` came from pssc ``testing`` (F1.12).
"""

from .schema import (
    TRACE_SCHEMA_VERSION,
    EventKind,
    TraceEvent,
    serialize,
    parse,
)
from .sink import TraceSink, MemorySink, FileSink, NullSink, CallbackSink
from .binary import BinarySink
from .steps import diff

__all__ = [
    "TRACE_SCHEMA_VERSION",
    "EventKind",
    "TraceEvent",
    "serialize",
    "parse",
    "TraceSink",
    "MemorySink",
    "FileSink",
    "NullSink",
    "CallbackSink",
    "BinarySink",
    "diff",
]
