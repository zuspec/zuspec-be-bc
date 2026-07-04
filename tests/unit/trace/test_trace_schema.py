"""T0-F -- trace schema round-trip and canonical (byte-stable) serialization."""

import pytest

from zuspec.be.bc.trace import (
    EventKind, TraceEvent, serialize, parse, MemorySink, NullSink,
    TRACE_SCHEMA_VERSION,
)


def _sample_events():
    return [
        TraceEvent(0, EventKind.SPAWN, coro=0, src_ref=3, detail={"child": 1}),
        TraceEvent(1, EventKind.SOLVE, coro=1, src_ref=5,
                   detail={"vars": {"a": 1, "b": 2}, "seed": 42}),
        TraceEvent(2, EventKind.SELECT, coro=1, src_ref=7, detail={"choice": 2}),
        TraceEvent(3, EventKind.IMPORT, coro=1, src_ref=9, detail={"fn": 11}),
        TraceEvent(4, EventKind.JOIN, coro=0, src_ref=3),
    ]


def test_event_roundtrip():
    for ev in _sample_events():
        assert TraceEvent.from_json(ev.to_json()) == ev


def test_serialize_parse_roundtrip():
    events = _sample_events()
    version, back = parse(serialize(events))
    assert version == TRACE_SCHEMA_VERSION
    assert back == events


def test_serialization_is_byte_stable():
    events = _sample_events()
    assert serialize(events) == serialize(events)
    # Same content, freshly built -> identical bytes (canonical form).
    assert serialize(_sample_events()) == serialize(events)


def test_detail_key_order_does_not_affect_bytes():
    a = TraceEvent(0, EventKind.SOLVE, detail={"a": 1, "b": 2})
    b = TraceEvent(0, EventKind.SOLVE, detail={"b": 2, "a": 1})
    assert a.to_json() == b.to_json()  # sort_keys=True


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        TraceEvent(0, "NOPE")


def test_memory_sink_sequences_and_serializes():
    sink = MemorySink()
    sink.event(EventKind.SPAWN, coro=0, src_ref=1)
    sink.event(EventKind.WAIT, coro=0, src_ref=2)
    sink.event(EventKind.JOIN, coro=0, src_ref=1)
    assert [e.seq for e in sink.events] == [0, 1, 2]
    version, back = parse(sink.serialize())
    assert [e.kind for e in back] == ["SPAWN", "WAIT", "JOIN"]


def test_null_sink_still_advances_seq():
    sink = NullSink()
    e0 = sink.event(EventKind.SPAWN)
    e1 = sink.event(EventKind.JOIN)
    assert (e0.seq, e1.seq) == (0, 1)
