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

### 1.1 Floating point (`float32`, `float64`)

A float register holds the IEEE 754 **binary64 bit pattern** of its value, as an
unsigned 64-bit integer. A `float32` value is the same binary64 pattern after
rounding to binary32 precision (`FCVT_F32`), so both types share one register
format and one set of operations. Float values are never widened to an integer
on the way in: `float64 d = 2.5;` holds `0x4004000000000000`.

| Opcode | Name | Operands | `imm` sub-op |
|---|---|---|---|
| `0x2C` | `FARITH` | `rd = a op b` (`NEG`: `b` unused) | `0` ADD, `1` SUB, `2` MUL, `3` DIV, `4` NEG |
| `0x2D` | `FCVT` | `rd = convert(rs)` | `0` S2F, `1` U2F, `2` F2S, `3` F32 |
| `0x2E` | `FMATH` | `rd = fn(a)` or `fn(a, b)` | index in `FMATH_FNS` |
| `0x36` | `FCMP` | `rd = (a op b) ? 1 : 0` | `0` EQ, `1` NE, `2` LT, `3` LE, `4` GT, `5` GE |

- `FARITH` computes in binary64; a `float32` result is then rounded with
  `FCVT_F32`. Division by zero gives an infinity or NaN, not an error.
- `FCVT_S2F` / `FCVT_U2F`: signed / unsigned 64-bit integer to binary64,
  round to nearest even. `FCVT_F2S`: binary64 to signed 64-bit, truncated
  toward zero; a NaN or out-of-range value gives `0x8000000000000000`.
  `FCVT_F32`: round to binary32 precision, nearest even.
- `FCMP` follows IEEE: every compare with a NaN is false, except `NE`.
- `FMATH` is the PSS float library (LRM Table 28) with C `libm` semantics,
  binary64 in and out: `log log10 exp sqrt pow round floor ceil sin cos tan
  asin acos atan atan2 hypot sinh cosh tanh asinh acosh atanh` (imm 0..21, in
  that order). `pow`, `atan2` and `hypot` take two operands; the others ignore
  `b`. `round` halves away from zero. Domain errors give NaN, not an error.
  `float_mantissa`/`float_exponent`/`float_sign` and `to_float` are not ops:
  they lower to integer shifts and masks of the register's bits.
  A float `**` is `FMATH pow`. An integer `**` is not an op either: it lowers
  to a square-and-multiply loop (`MUL`, `SHR`, `BRZ`) over the exponent's bits,
  truncated to the result width; a negative exponent gives 1 for a base of 1,
  1 or -1 (by parity) for -1, else 0 (LRM Table 12).
- `message()` renders a float operand with `%f`, `%e`/`%E` and `%g`/`%G`
  (C printf semantics, flags and precision included).
  The image marks such an argument `ZBC_MSG_FLOAT` (`zbc_msg_kind` 5) and a
  float named local `ZBC_LOCAL_FLOAT` (`zbc_local_kind` 6); a debugger shows
  the local's value as the binary64 its bits hold. A NaN prints as `nan`
  whatever its sign bit.

The constants are `FARITH_*`, `FCVT_*`, `FCMP_*` and `FMATH_FNS` in `zuspec.be.bc.model`.

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

### 3.1 Strings and collections (heap.md)

A `string`, `list<T>`, `map<K,V>` or `set<T>` value takes one 64-bit slot
(8 bytes in an aggregate), whatever its contents:

- **string:** bit 63 clear = an index into the image's model-string table
  (every literal); bit 63 set = a heap string handle in the low 63 bits.
- **list / map / set:** a heap handle; **0 is the empty collection**, so a
  zeroed slot is a valid, empty value.

Contents live in the engine instance's heap, never in the packed bytes, and
an assignment copies them (heap.md section 4). The encoding is a superset of
the F6.3a string slot, so it does not bump `abi_id`: older images hold only
table indices.

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
