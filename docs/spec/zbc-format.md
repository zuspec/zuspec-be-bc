# `.zbc` container format

Status: **P0 contract (M1)** · Magic `Z B C 0x1A` · Version `2.0` · Authority:
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
| `zbc_entry` | 40 | entry point: `name`, `coro`, `kind`, `comp_type`, `sig`, `qname`, `obj`, `flags`, `fixed` (see below) |
| `zbc_impent` | 24 | user import: `fn_id`, `name`, `api`, `flags`, `sig`, `qname` |
| `zbc_apient` | 16 | import-interface API: `id`, `name`, `kind`, `n_fns` |
| `zbc_class` | 40 | type: `qname`, `kind`, `base`, `coro`, `n_fields`, `size`, `align`, `fields`, `methods` |
| `zbc_rootent` | 32 | root candidate: `type`, `ctor`, `api_hash`, `img_off`, `img_size` |

`elem_size` is the **forward-compat lever**: a newer writer may append fields to a
record; an older reader strides by the file's `elem_size` and reads the prefix it
knows. Unknown **section kinds** are skipped by readers.

## Section kinds

`CODE=1 CORO=2 TYPE=3 CONST=4 SOLVE=5 STRB=6 STRO=7 FILE=8 PROV=9 CMNT=10 LINE=11`
(extensible; also `BLOCK=12 OPLIST=13 SELECT=14 SPROB=15 ENTRY=16 LOCAL=17 SRC=18 IDENT=19
META=20 COMP=21 MSG=22 REG=23 CHAN=24 COV=25 IMPORT=26 ACT=27 LAYOUT=28 REGMAP=29 SEEDKEY=30
COVG=31 EXEC=32 EVENT=33 MEMMAP=34 API=35 CLASS=36 ROOT=37 ROOTIMG=38`). `CONST` holds >64-bit literals (value ABI §3); `SOLVE` blobs are P4. `COMP`
(every profile, when the model has components) is described in [components.md](components.md):
one `zbc_comp` whose `flags` are `ZBC_COMP_F_NODE_INST` (0x1: an activation node runs in an
instance other than its parent's) and `ZBC_COMP_F_RUN_HOOKS` (0x2: coroutines `init_coro + 1`,
`$run_start`, and `init_coro + 2`, `$run_end`, run the tree's `exec run_start` blocks top-down
before a root action and its `exec run_end` blocks bottom-up after it). Neither hook is an
`ENTRY`; the host never starts them itself.

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
- `IMPORT` (kind 26; codegen and debug profiles), one `zbc_impent` (24 bytes)
  per user import (`ZbcModel.imports`), sorted by `fn_id`: the `fn_id` the
  code's `IMPORT` op calls; the import's PSS `name` and qualified name
  `qname` (StrIds; the C engine puts the name in `pssvm_req_t.name`); `api`,
  the id of the API row it belongs to (0xFFFFFFFF: none); `flags`
  (`ZBC_IMP_F_BLOCKING` 0x1: the call may consume host time;
  `ZBC_IMP_F_TARGET` 0x2 / `ZBC_IMP_F_SOLVE` 0x4: declared `import target` /
  `import solve`); and `sig` (StrId, 0 = none), canonical JSON
  `{"arg_types": [[width, signed], ...], "ret_type": [width, signed(, "float"|"string")] | null,
  "string_at": [...]}` plus `"ret_types"` for a struct result (host-abi.md
  "Struct results"): the parts of the signature the API hash covers, so a host
  links against the image alone and a reader rebuilds `ZbcModel.imports`.
- `API` (kind 35; codegen and debug profiles), one `zbc_apient` (16 bytes)
  per API of the import interface (host-abi.md), sorted by `id`: `id`, `name`
  (StrId), `kind` (`ZBC_API_BUILTIN` 0, `ZBC_API_USER` 1) and `n_fns`, the
  IMPORT rows naming it. The built-in rows are always there: `mem` (0),
  `config` (1), `seq` (2). User APIs start at 16
  (`ZBC_API_USER_FIRST`), one per scope declaring imports, named by it
  (`dma_pkg`; "" for the global scope), in name order.
- `ACT` (kind 27), one `zbc_acttab` (64 bytes) then its arrays (see the record
  docs in zbc_format.h): each activation (`ZbcModel.activations`) by
  coroutine: its action nodes, activity blocks, traversal sites, solve cones
  with their variables and constraint metadata, pools, claims, outputs and
  picks, slot names and initial values. Lists are u32 runs; strings are
  section-local; 0xFFFFFFFF is "none". Constraint expressions are not here.

### Component layout (`LAYOUT`)

`LAYOUT` (kind 28; codegen and debug profiles) is one `zbc_layout` (16 bytes),
present when the image has a component tree (`COMP`): `root` (StrId) the root
component type; `slots` (StrId) canonical JSON `[[name, descriptor, offset], ...]`, one per
component-object slot in slot order, named by its path from the root (`a.sub.k`,
`ch[2].id`), descriptors as in `ENTRY.obj`, `offset` the slot's byte offset in the
component object's **normative native layout** (value-abi.md §2.1); `insts` (StrId)
JSON `[[path, type], ...]`, one per instance in pre-order (the root's path is "");
`size`, the component object's native size in bytes. A reader names component
fields from it (`CompTable.names`/`slot_types`/`offsets`/`paths`/`types`); without it
(runtime images) slots read as `$c<slot>` and paths as "".

The layout is normative: an implementation that represents objects natively
(an AOT tier, a zero-copy host view) places each slot there. Instruction
operands still address slots (`LD_FIELD`/`ST_FIELD`/`LD_COMP`/`ST_COMP` `arg1`
is a slot index in this version), and so do the ACT, SOLVE and EVENT tables;
moving them to offset + width is deferred (pssc `docs/design/pssvm-plan.md`
step 4).

### Types (`CLASS`)

`CLASS` (kind 36; codegen and debug profiles), one `zbc_class` (40 bytes) per
type: every component type in the tree and its bases, every action with an
entry and its bases, and every plain-data struct their attributes hold;
components, then actions, then structs, each in name order.

- `qname` (StrId): the type's qualified name, as the linker resolved it.
- `kind`: `ZBC_CLASS_COMPONENT` (0), `ZBC_CLASS_ACTION` (1), `ZBC_CLASS_STRUCT` (2).
- `base` (StrId, 0 = none): its base type's qualified name.
- `coro`: an action's coroutine, which is its `traverse` method (0xFFFFFFFF: none,
  e.g. an abstract base).
- `n_fields`, `size`, `align`: the object's leaves (slots), and its native size
  and alignment in bytes.
- `fields` (StrId): canonical JSON `[[name, descriptor, offset, size], ...]` in slot
  order: slot `k` of an object of the type is row `k`. A component's rows are its
  subtree's (sub-instances included, `sub.a`, `ch[2].id`), as in `LAYOUT`; an
  action's are its object's, as in `ENTRY.obj` (sub-action handles included).
- `methods` (StrId, 0 = none): JSON `[[name, entry symbol], ...]`, a component's
  functions that have entries.

The rows name types over today's coroutines; nothing executes from them. They are
what a typed host API, and a C backend's vtable, are generated from.

### Root candidates (`ROOT`, `ROOTIMG`)

An image may be built for several root components, chosen at run time
(pssvm-in-practice.md §3.0.6 note 2, option A). `ROOT` (kind 37; codegen and debug
profiles) is one `zbc_rootent` (32 bytes) per candidate:

- `type` (StrId): the candidate's component type, its qualified name.
- `ctor` (StrId, 0 = none): its constructor as the front end names it, JSON
  `{"name", "params": [{"name", "type"}, ...]}`.
- `api_hash`: the API hash of the image built for this root (as `IDENT`'s).
- `img_off`, `img_size`: where its image is in `ROOTIMG`; `img_size` 0 for row 0.

Row 0 is the image's own root: its `COMP`, `ACT`, `ENTRY`, ... are this image's
tables. Every other candidate's tables are a complete image of its own (same
format version), nested in `ROOTIMG` (kind 38) at `(img_off, img_size)`, each
8-aligned; a nested image's `ROOT` lists only itself. An engine that runs only the
image's own root (the C engine today) ignores both sections. Code is not shared
between candidates yet; doing so would change only the image's size.

### Register map (`REGMAP`)

`REGMAP` (kind 29; codegen and debug profiles) is one `zbc_regmap` (4 bytes),
present when the host set `ZbcModel.registers`: `doc` (StrId) canonical JSON
`{"<reg-id>": register, ...}`, the reg-id (the register's component instance index,
as `REG` accesses carry it) as a decimal string. A register is opaque to bc; pssc
writes `{path, comp, reg, type, comp_type, offset, width, access, fields}`, each field
`{name, lsb, width, kind[, enum]}`. A reader turns the keys back into ints. pssc
sets it only for debug images, so release images have none.

### Covergroups (`COVG`)

`COVG` (kind 31; codegen and debug profiles) is one `zbc_covg` (4 bytes),
present when the model samples any covergroup: `doc` (StrId), a JSON array
with one document per covergroup instance; an instance's index in it is the
`covg` operand of `COV_SAMPLE`. [coverage.md](coverage.md) gives the document
format and how samples are binned.

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
- `qname` (StrId, 0 = none): an action's qualified type (`pkg::dev_c::write`).
- `obj` (StrId, 0 = no object): an action's object layout, canonical JSON
  `[[name, descriptor], ...]` in slot order, a struct attribute one slot per leaf
  (`s.f`); a rand field's descriptor adds `"rand": true`, a type with no value
  descriptor is `{"kind": "opaque", "width": 0, "signed": false}`. A reader names
  the object's fields from it (`ZbcModel.obj_layouts`).
- `flags`: `ZBC_ENTRY_F_EXPORTED` (0x1) marks an exported action: the front end
  exported it (pssc: a PSS `export` statement or `--export-action`), so a host may run
  it by name and a typed API gives it a method. Every action keeps its entry either way.
- `fixed` (StrId, 0 = none): an exported action's fixed fields, JSON `[name, ...]`
  (its `export` statement's parameters), which a typed method takes as arguments.

`ENTRY` needs the string table, so only the codegen and debug profiles carry it; a runtime
image's host keeps the table beside it. A model without entries has no `ENTRY`
section.

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
`IDENT` section (kind 19), one `zbc_ident` (24 bytes). `api_hash` is FNV-1a/64
of canonical JSON over every entry (symbol, kind, component type, signature),
every user import (`fn_id`, blocking, argument and return types, string
positions), each sorted by name, and, when any action is exported, the exported
entries with their fixed fields (`model.api_hash_of`). It is per root: each
`ROOT` row carries its candidate's. Coroutine indices are
left out, so an edit to a body keeps the hash; a signature change does not.
`generation` is the caller's build number, never derived, so a build that
leaves it unset is reproducible. 0 means unset in either field. Only the debug
profile writes `IDENT`, and only when a field is set, so release images keep
their bytes.

## Profiles

Two independent things are called profiles.

**Extensions (what an engine must implement).** The `core` profile is every
opcode in 0x00–0x7F and the sections an engine needs for them. An extension adds
opcodes in a range of its own and sections, and an image that uses it sets its
header flag; bits `ZBC_HDR_EXT_MASK` (0xFF00) are reserved for extensions:

| Extension | Header flag | Opcodes | Sections |
|---|---|---|---|
| `pss` | `ZBC_HDR_EXT_PSS` (0x0100) | 0x80–0xBF: `SELECT` 0x80, `SCOPE_ENTER` 0x81, `SOLVE_NODE` 0x82, `COV_SAMPLE` 0x83, `EXEC_SET` 0x84, `EXEC_GET` 0x85, `LD_COMP` 0x86, `ST_COMP` 0x87 | `ACT`, `COMP`, `EXEC`, `EVENT`, `MEMMAP`, `COVG` |

0xC0–0xFF is for the next front end. A loader **refuses** an image that sets an
extension bit it does not implement, and one with an opcode in an extension's
range whose bit is clear, at load, naming it (pyvm `ZbcModel.from_container`; the
C engine `ZBC_ERR_BAD_PROFILE`). An `INVOKE` with `INSTR_F_NODE` runs an
activation node, so it needs the `ACT` section and with it `pss`. The writer sets
the bits from what the model uses (`ZbcModel.extensions`). A profile is also the
unit of compliance: an implementation claims `core` + `pss`.

**Section sets (what an image carries).** Same magic + version; the directory
contents differ:

- **codegen** — includes `STRB/STRO/FILE/PROV/CMNT` and, from debug lowering,
  `LINE/LOCAL`; `HAS_PROV` set. Release images use it (no `LINE`/`LOCAL`).
- **debug** — codegen plus `SRC` when the model has sources, `META` when it has documents, and `IDENT`;
  `PROFILE_DEBUG` (0x0008) and `HAS_PROV` set. An engine treats it as codegen.
- **runtime** — omits every provenance-family section
  (`STRB/STRO/FILE/PROV/CMNT/LINE/LOCAL/SRC/META/REG/CHAN/IMPORT/API/CLASS/ROOT/ROOTIMG`);
  `PROFILE_RUNTIME` set,
  `HAS_PROV` clear. The writer **refuses** to emit a runtime profile that still
  carries provenance sections.

## Integrity

The engine validates `magic`, `version_major`, `abi_id`, and `file_size` on load
and rejects a mismatch. There is one format version at a time: 2.0 renumbered the
`pss` opcodes, so an engine refuses a 1.x image (and a 1.x engine a 2.0 one)
rather than misread it; rebuild the image. `content_hash` (FNV-1a/64 over the bytes *after* the hash
field) is verified when non-zero; a single flipped byte is caught.
