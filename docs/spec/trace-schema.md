# Trace / event schema

Status: **P0 contract (M1)** · Version `1` · Authority:
`zuspec.be.bc.trace.schema` · Companion to design `[D§10]`.

An execution emits an **event-sourced** trace of the orchestration ops. The
serialized form is canonical so two runs with the same seed produce a
**byte-identical** trace and oracle-vs-engine traces are byte-diffable.

## Event record

```
TraceEvent { seq: int, kind: str, coro: int, src_ref: int, detail: {..} }
```

- `seq` — monotonic, assigned by the sink.
- `kind` — one of `SPAWN INVOKE PAR JOIN WAIT SELECT SOLVE BIND IMPORT YIELD`.
- `coro` — emitting frame/coroutine id.
- `src_ref` — provenance back-pointer (`zbc_prov` index; 0 = none).
- `detail` — kind-specific payload (e.g. `SOLVE` → `{"vars": {...}, "seed": N}`,
  `SELECT` → `{"choice": i}`, `IMPORT` → `{"fn": id}`).

## Serialized form

JSONL with a one-line versioned header:

```
{"zbc_trace":1}
{"coro":0,"detail":{"child":1},"kind":"SPAWN","seq":0,"src_ref":3}
...
```

JSON is emitted with **sorted keys** and **compact separators**, so the byte
stream is a pure function of event contents — `detail` key order does not affect
the bytes. This is what makes trace equality a reliable determinism check (T1-D).

## Sinks

`MemorySink` (tests / differential comparison), `FileSink` (streams JSONL),
`NullSink` (runtime hot path — still advances `seq` so callers behave
identically).

## Versioning

`TRACE_SCHEMA_VERSION` bumps on any incompatible change to the record shape; the
header carries it so a reader can reject or adapt.
