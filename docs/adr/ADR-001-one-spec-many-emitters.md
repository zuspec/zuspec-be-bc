# ADR-001 — One spec, many emitters

Status: **Accepted** (P0 / M1) · Context: design `[D§12.3, §3a]`.

## Context

The `.zbc` format and the value ABI must be read identically by three consumers:
the Python oracle (in-memory dataclasses), a Python zero-copy reader (`ctypes`
overlay), and the native C engine (`mmap` + cast). Hand-writing three decoders
guarantees eventual drift — a field reordered in one place, a size changed in
another — and drift in a cross-tier ABI is silent and catastrophic (the whole
differential-testing strategy assumes identical bytes).

## Decision

Every on-disk record layout and enum is declared **once**, declaratively, in
`zuspec.be.bc.format.spec` (the container format) and
`zuspec.be.bc.abi.value` (the value types). All three views are **generated**
from those descriptions:

- `format/emit_c.py` → the checked-in C header `zbc_format.h`,
- `format/emit_ctypes.py` → the `ctypes` overlay,
- `format/emit_dataclass.py` → the in-memory dataclasses (with a `struct`-based
  codec).

No second hand-written decoder exists anywhere.

## Consequences

- **Drift is a test failure, not a field bug.** Emitter-agreement (T0-A) asserts
  all three views agree on size *and* field offset; the generated header's
  `_Static_assert`s make the C compiler prove its layout equals the spec; the
  regen no-diff test (T0-D) guards the checked-in header.
- **The C header is checked in as a generated artifact** (roadmap Q4, accepted):
  simpler for rt-eng/AOT to consume, no build-time generation; drift is caught by
  T0-D. Regenerate with `python -m zuspec.be.bc.format.emit_c`.
- **Generated output must be deterministic** — no timestamps or host-specific
  content — or T0-D would flap.

## Related decision — runtime vs codegen profile split

The same single format carries two profiles (`[D§12.2]`): a **codegen/debug**
profile with the string/provenance sections, and a **runtime** profile that omits
them and clears `HAS_PROV`. This is a directory-contents difference, not a second
format; the writer enforces that a runtime profile carries no provenance. Keeping
it one format (one magic, one version, one set of emitters) is what lets the
oracle test the exact bytes the engine ships.
