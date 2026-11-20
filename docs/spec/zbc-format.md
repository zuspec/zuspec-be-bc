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
| `zbc_entry` | 32 | entry point: `name`, `coro`, `kind`, `comp_type`, `sig` (see below) |

`elem_size` is the **forward-compat lever**: a newer writer may append fields to a
record; an older reader strides by the file's `elem_size` and reads the prefix it
knows. Unknown **section kinds** are skipped by readers.

## Section kinds

`CODE=1 CORO=2 TYPE=3 CONST=4 SOLVE=5 STRB=6 STRO=7 FILE=8 PROV=9 CMNT=10 LINE=11`
(extensible; also `BLOCK=12 OPLIST=13 SELECT=14 SPROB=15 ENTRY=16 LOCAL=17 SRC=18 IDENT=19
META=20 COMP=21`). `CONST` holds >64-bit literals (value ABI §3); `SOLVE` blobs are P4. `COMP`
(every profile, when the model has components) is described in [components.md](components.md).

### Statement markers (`LINE`)

Debug lowering marks where each PSS statement starts in the `LINE` section,
not with an opcode, so release code is the same instruction for instruction.

- One `zbc_lineent` per PSS statement: `pc_start` is the global `CODE` index
  of the statement's first instruction, `src_ref` its `PROV` index. Entries are
  sorted by `pc_start`; `src_at(pc)` is the entry with the largest
  `pc_start <= pc`.
- A statement that would emit no code (`int x;`) gets a `NOP`, so every
  statement has its own pc and a breakpoint on it can be hit.
- Section flag `ZBC_SECF_LINE_STMT` (0x0002) is set when the entries are statement
  starts (debug lowering). Without it the table only maps pcs to sources.
- Release lowering writes no `LINE` section; the runtime profile never carries
  one. A release image is byte-identical whether or not the debugger exists.
- The engine reads the table only when `stmt` events are enabled for a code
  unit; with events off it costs nothing per instruction.

In debug lowering a call to a PSS function is a `CALL` of the function's
called form (`<fn>$callN`, one per function and component instance), so its
statements and locals belong to its own coroutine. A function that takes or
returns a struct is still inlined into its caller.

### Named locals (`LOCAL`)

Debug lowering names the frame locals that hold PSS variables in the `LOCAL`
section (kind 17), one `zbc_localent` (24 bytes) per variable, or per leaf of
a struct variable:

- `coro`, `slot`: the coroutine and its frame-local slot.
- `name`: `StrId` of the PSS name; a struct leaf is `cfg.src`.
- `[pc_start, pc_end)`: global `CODE` indices where the name is in scope.
  Two entries may share a name (a nested block, an inlined struct function);
  the innermost one in scope at a pc is the one with the largest `pc_start`.
- `width`, `kind` (`zbc_local_kind`: int, bool, enum, string, chandle,
  opaque), `flags` (`ZBC_LOCAL_F_SIGNED`).

Entries are sorted by `(coro, pc_start)`. Temporaries the lowering adds
(return slots, `message()` pools) are not named. Like `LINE`, the section is
provenance-family: release lowering and the runtime profile carry none.

### Register accesses and channel operations (`REG`, `CHAN`)

Debug lowering also lists, for the event hooks, which instructions access a
register and which end a channel operation. Both sections are sorted by
global `CODE` pc and are provenance-family, like `LOCAL`.

- `REG` (kind 23), one `zbc_regent` (12 bytes) per register access: `pc` of
  the access's `IMPORT`; `inst`, the register's component instance relative
  to the frame's instance (`0xFFFFFFFF`: none); `flags`
  (`ZBC_REG_F_DISCARD`: a read whose value the statement drops,
  `r.read_val();`). An `access` event's `reg_id` is the frame's instance plus
  `inst`; the C engine puts it in `pssvm_req_t.reg_id`.
- `CHAN` (kind 24), one `zbc_chanent` (24 bytes) per channel operation: `pc`
  of the `NOP` that ends it, its `src_ref`, `channel` and `op` (StrIds), and
  the registers holding the value and, for `try_*`, ok (`0xFFFFFFFF`: none).
  A `chan` event is raised there.
- `COV` (kind 25), one `zbc_covent` (20 bytes) per coverage item
  (`CoroDescriptor.cov`), sorted by `pc` (a statement and an arm may share
  one): `pc` where the statement starts or the arm is taken, its `src_ref`,
  `flags` (`ZBC_COV_F_ARM`: a branch arm, else a statement;
  `ZBC_COV_F_IMPLICIT`: an `else` the code did not write), `scope` (the
  function, `<comp type>::<fn>`; "" = the coroutine's own code) and `path`
  (its place in PSS structure; an arm's is `<branch path>/<arm>`), both
  StrIds. A `stmt` event carries `path` and `scope`; an arm raises a
  `branch` event.
- `IMPORT` (kind 26; codegen and debug profiles), one `zbc_impent` (8 bytes)
  per user import (`ZbcModel.imports`), sorted by `fn_id`: the `fn_id` the
  code's `IMPORT` op calls and the import's PSS `name` (StrId). The C engine
  puts the name in `pssvm_req_t.name`.
- `ACT` (kind 27), one `zbc_acttab` (64 bytes) then its arrays (see the record
  docs in zbc_format.h): each activation (`ZbcModel.activations`) by
  coroutine: its action nodes, activity blocks, traversal sites, solve cones
  with their variables and constraint metadata, pools, claims, outputs and
  picks, slot names and initial values. Lists are u32 runs; strings are
  section-local; 0xFFFFFFFF is "none". Constraint expressions are not here.

### Entry points (`ENTRY`)

Where a host may enter the image (`ZbcModel.entries`): one `zbc_entry` per entry,
in the order the lowering made them.

- `name` (StrId): an action by its coroutine name; a component function by
  `<comp type>::<fn>` (`dev_c::put`).
- `coro`: the `CORO` index the host starts. `kind`: `ZBC_ENTRY_ACTION` (0) or
  `ZBC_ENTRY_FUNCTION` (1). `comp_type` (StrId, 0 = none): a function's component
  type; it runs in an instance of that type (or of a derived type that does not
  override it).
- `sig` (StrId, 0 = no parameters and no result): canonical JSON
  `{"params": [[name, descriptor], ...], "returns": descriptor | null}`. A struct
  parameter is one parameter per leaf, named by its path (`cfg.src`), in
  declaration order; the host puts them in the root frame's `args` (`LD_ARG`).
  A descriptor is `{"kind", "width", "signed"}` plus `"items"` for an enum.

`ENTRY` needs the string table, so only the codegen and debug profiles carry it; a runtime
image's host keeps the table beside it. A model without entries has no `ENTRY`
section, so older images read unchanged.

### PSS sources (`SRC`)

A debug image may carry the PSS source files (`ZbcModel.sources`), so a
debugger can show lines on a machine that has only the image: the `SRC`
section (kind 18), one `zbc_srcent` (8 bytes) per file, sorted by path.
`path` and `text` are StrIds; `path` is spelled as in `FILE`. Only the debug
profile writes it.

### Host metadata (`META`)

A debug image may carry documents its host reads and bc does not
(`ZbcModel.meta`, name -> text): pssc stores its test catalog and register
map there, so an image alone can list and run tests. The `META` section
(kind 20), one `zbc_metaent` (8 bytes) per document, sorted by name; `name`
and `text` are StrIds. Only the debug profile writes it; a model without
documents has no `META` section.

### Image identity (`IDENT`)

A debug image says which API it implements and which build it is: the
`IDENT` section (kind 19), one `zbc_ident` (16 bytes). `api_hash` is FNV-1a/64
of canonical JSON over every entry (symbol, kind, component type, signature)
and every user import (`fn_id`, blocking, argument and return types, string
positions), each sorted by name (`model.api_hash_of`). Coroutine indices are
left out, so an edit to a body keeps the hash; a signature change does not.
`generation` is the caller's build number, never derived, so a build that
leaves it unset is reproducible. 0 means unset in either field. Only the debug
profile writes `IDENT`, and only when a field is set, so release images keep
their bytes.

## Profiles

Same magic + version; the directory contents differ:

- **codegen** — includes `STRB/STRO/FILE/PROV/CMNT` and, from debug lowering,
  `LINE/LOCAL`; `HAS_PROV` set. Release images use it (no `LINE`/`LOCAL`).
- **debug** — codegen plus `SRC` when the model has sources, `META` when it has documents, and `IDENT`;
  `PROFILE_DEBUG` (0x0008) and `HAS_PROV` set. An engine treats it as codegen.
- **runtime** — omits every provenance-family section
  (`STRB/STRO/FILE/PROV/CMNT/LINE/LOCAL/SRC/META`); `PROFILE_RUNTIME` set,
  `HAS_PROV` clear. The writer **refuses** to emit a runtime profile that still
  carries provenance sections.

## Integrity

The engine validates `magic`, `version_major`, `abi_id`, and `file_size` on load
and rejects a mismatch. `content_hash` (FNV-1a/64 over the bytes *after* the hash
field) is verified when non-zero; a single flipped byte is caught.
