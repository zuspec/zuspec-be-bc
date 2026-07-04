# `.zbc` container format

Status: **P0 contract (M1)** · Magic `Z B C 0x1A` · Version `1.0` · Authority:
`zuspec.be.bc.format.spec` · Companion to design `[D§12]`.

This document mirrors the machine-readable spec; the **spec module is the
authority** and the design `[D§12.1/§12.2]` is the rationale. The C header
(`zuspec-rt-core/.../share/include/zbc_format.h`), the `ctypes` overlay, and the
in-memory dataclasses are all generated from `spec.py` — see ADR-001.

## Layout

```
+----------------+ offset 0
|   zbc_header   |  48 bytes
+----------------+ header.section_dir_off (8-aligned)
| zbc_section[N] |  directory, 32 bytes each
+----------------+
|  payload 0     |  each payload 8-byte aligned, located by (offset, size)
|  ...           |
+----------------+
```

All multi-byte fields are **little-endian**; every record is naturally aligned so
a C reader can `mmap` + cast-index and a Python reader can overlay `ctypes`
directly. Field order in `spec.py` is chosen so there is no implicit padding
(the emitter-agreement test asserts this, and the header's `_Static_assert`s prove
the C compiler agrees).

## Records

| record | size | role |
|---|---|---|
| `zbc_header` | 48 | magic, version, `flags`, `abi_id`, section count/dir offset, `entry_coro`, `file_size`, `content_hash` |
| `zbc_section` | 32 | `kind`, `flags`, `offset`, `size`, `count`, `elem_size` |
| `zbc_prov` | 32 | provenance record (`src_ref` indexes it directly) |
| `zbc_comment` | 8 | comment record (runs owned by a `zbc_prov`) |
| `zbc_file` | 8 | file-table entry |
| `zbc_lineent` | 8 | `pc → src_ref` line-table entry |

`elem_size` is the **forward-compat lever**: a newer writer may append fields to a
record; an older reader strides by the file's `elem_size` and reads the prefix it
knows. Unknown **section kinds** are skipped by readers.

## Section kinds

`CODE=1 CORO=2 TYPE=3 CONST=4 SOLVE=5 STRB=6 STRO=7 FILE=8 PROV=9 CMNT=10 LINE=11`
(extensible). `CONST` holds >64-bit literals (value ABI §3); `SOLVE` blobs are P4.

## Profiles

Same magic + version; the directory contents differ:

- **codegen/debug** — includes `STRB/STRO/FILE/PROV/CMNT/LINE`; `HAS_PROV` set.
- **runtime** — omits all six provenance sections; `PROFILE_RUNTIME` set,
  `HAS_PROV` clear. The writer **refuses** to emit a runtime profile that still
  carries provenance sections.

## Integrity

The engine validates `magic`, `version_major`, `abi_id`, and `file_size` on load
and rejects a mismatch. `content_hash` (FNV-1a/64 over the bytes *after* the hash
field) is verified when non-zero; a single flipped byte is caught.
