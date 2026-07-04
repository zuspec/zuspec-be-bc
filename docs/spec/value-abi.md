# Value / layout ABI

Status: **P0 contract (M1)** · ABI version `abi_id = 1` · Source of truth:
`zuspec.be.bc.abi.value` (descriptor) + `zuspec.be.bc.abi.codec` (codec).
Companion to design `[D§11]` and `.zbc` format `[D§12]`.

This is the single most important cross-tier contract. Solve results, exec-block
mutations, SV pack/unpack, and the serialized `.zbc` must all agree on scalar
encoding, aggregate packing, endianness, the field↔solver-var mapping, and the
representation of values wider than 64 bits. Everything below is derived from one
machine-readable description (`abi/value.py`); there is no second hand-written
decoder.

## 1. Scalars

A scalar has a `width_bits` and a `signed` flag.

- **Endianness:** little-endian. Least-significant byte first.
- **Storage width:** `byte_len = ceil(width_bits / 8)` bytes.
- **Signed values:** two's complement. The unused high bits of the final byte are
  sign extension (signed) or zero (unsigned).
- **Truncation / overflow:** encoding reduces the input modulo `2**width_bits`
  (`value & mask`), so out-of-range inputs wrap exactly as fixed-width hardware
  arithmetic does. The codec is therefore the **single authority** on
  truncation/overflow semantics `[D§15.5]` — op handlers must not re-implement it.

Examples (`abi_id = 1`):

| value | type | bytes (hex) |
|---|---|---|
| 0x7F | u8 | `7f` |
| 0x1FF | u8 | `ff` (wrapped) |
| −1 | s8 | `ff` |
| 0xABC | u12 | `bc 0a` |
| 0x1234 | u16 | `34 12` |

## 2. Aggregates — byte-granular packing

- **Arrays:** `count` element encodings concatenated, each occupying its own
  `byte_len`.
- **Structs:** field encodings concatenated in **declaration order**, each field
  occupying its own `byte_len`.

Packing is **byte-granular** in M1: no sub-byte bit packing across field
boundaries. Tight SV `bit[N-1:0]` packing is deferred to the SV pack/unpack path
(roadmap P3+) and selected by the reserved `Packing.BIT` policy; M1 fixes
`Packing.BYTE`. Byte granularity is what both the Python oracle and the C reader
index cheaply, and it is unambiguous for differential testing.

## 3. Values wider than 64 bits — the constant pool

A scalar with `width_bits > 64` does not fit a ZBC register/slot. Its literal
bytes live in the `SEC_CONST` section and the code stream references them by
index. One pool record (`const_pool_record`) is:

```
u32 width_bits | u32 byte_len | u8[byte_len] payload   (record padded to 8 bytes)
```

`payload` is the *same* canonical LE two's-complement encoding as an inline
scalar, so a wide literal decodes identically whether it arrived inline or pooled.
`is_inline(t)` is the single predicate lowering uses to decide inline-vs-pooled.

## 4. Field ↔ solver-var mapping (the D§15.4 open item)

A solver `var_id` is the index of the field's name in the **alphabetically
sorted** list of the enclosing scope's randomizable field names:

```
var_id(name) = index of name in sorted(rand_field_names)
```

This is **inherited, not invented** — it mirrors `zuspec-solver`'s own assignment
(`{name: idx for idx, name in enumerate(sorted(system.variables.keys()))}`,
`ir_translator` / `c_bench_harness`). `solver_var_map()` computes exactly this in
one place, so a value written back from `get_value(var_id)` after a solve lands in
the correct field slot regardless of the struct's declaration order.

Struct **slot storage** stays in declaration order (natural for generated code);
the var-map derives from *sorted names*. The two orders are decoupled on purpose —
only the name→var_id derivation must be shared, and it is.

## 5. Determinism note

The sorted-by-name var ordering is one of the determinism anchors formalized in
`determinism.md` (`[D§15.1]`). The value ABI depends on it but does not own it;
the determinism spec is the authority on ordering, this doc on byte layout.

## 6. Versioning

`abi_id` is a `u32` stamped into `zbc_header.abi_id`. Any change to scalar
encoding, packing, the const-pool record, or the var-map rule bumps `abi_id`; the
engine rejects a `.zbc` whose `abi_id` differs from its own.
