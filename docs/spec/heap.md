# Heap: run-time strings and collections

Status: **specified (F6.9a), not implemented.** The Python engine implements it
with F6.9c; the C engine refuses it until M14 (section 9). Companion to
value-abi.md (slot formats), host-abi.md (strings across imports) and
determinism.md (the RNG `shuffle` draws from).

PSS strings built at run time (`format_string`, sub-strings, `lower`, `split`,
...) and the collection types `list<T>`, `map<K,V>` and `set<T>` (LRM 3.0
7.6, 7.9.3-7.9.5) have no fixed size, so they do not fit a slot or a packed
object. Their contents live in a heap; a slot holds a handle to them.

## 1. One heap per engine instance

- Each engine instance (one `Session`, one SV `zsp_rt_bc`) owns one heap. It
  is created empty with the instance and freed whole at its teardown.
- It is not per call: a component field of collection type keeps its value
  across entry calls, like any other component field.
- Object kinds:

| Kind | Holds |
|---|---|
| `str` | bytes (UTF-8 as written; no terminator counted in its size) |
| `list` | an ordered sequence of 64-bit cells |
| `map` | key -> value cells, kept in insertion order |
| `set` | cells, kept in insertion order |

- A cell holds exactly what a slot of the element type holds: an integer, a
  float's binary64 bits, a string value (section 3), or the handle of a nested
  collection. Elements of struct type (`list<S>`) are refused at lowering until
  F6.10c adds them.

## 2. Handles

- A collection slot is a u64 **handle**. **Handle 0 is the empty collection**
  of the slot's type. A zeroed object, a default struct and an activation
  table's initial values are therefore already valid, empty collections; no
  constructor runs.
- Allocation is lazy. An op that changes a collection takes its handle and
  **returns the handle** of the result, which may be new (when the input was
  0); lowering stores it back: `l.push_back(v)` lowers to
  `h' = HEAP PUSH_BACK h, v` then a store of `h'` into `l`.
- Handles are numbered per instance in allocation order, starting at 1. They
  are deterministic (same image, same seed, same calls give the same
  handles) but never printed or passed to a host; a debugger shows contents,
  not handles.
- A freed handle number is not reused within an instance (so a stale handle
  is detected, not aliased): an op on a freed or unknown handle is the runtime
  error `bad heap handle`.

## 3. Strings: table index or heap handle

A string slot holds one of two things:

- bit 63 clear: an index into the image's model-string table (unchanged since
  F6.3a; every literal, and every string of an older image);
- bit 63 set: a heap `str` handle in the low 63 bits.

Every string op accepts both. `==` / `!=` compare by index when both sides are
table indices (equal literals intern to one index), else by content. An op
that makes a new string (sub-string, `lower`, `format_string`, ...) always
returns a heap string, even when its text equals a literal. The empty string
`""` is its table index, as today; a heap string may also be empty.

## 4. Value semantics: copy and reclaim

PSS collections and strings are values (LRM 7.9.3): an assignment copies.

- **Copy** (`HEAP COPY`) on: assignment, a by-value argument, a struct copy
  (each collection field), a store into a collection (`l.push_back(m)` stores
  a copy of `m`), and reading a field into a local that is then changed.
  Copying handle 0, or a table-index string, gives the same value back (no
  allocation). Copies are deep: nested collections are copied too.
- A **return moves**: the callee's value becomes the caller's, no copy.
- **Reclaim** (`HEAP FREE`, deep) on: the old value of a slot that is
  overwritten; a function's or exec block's locals at frame exit (except a
  moved return value); an action object's collection fields when the object
  is released; and everything at instance teardown.
- Lowering may skip a copy it can prove unobservable (the source is dead
  afterwards, e.g. a temporary); the result must be as if it had copied.

## 5. Cap

- The heap has a byte cap, `RunOpts.heap_cap` (default 64 MiB), counting each
  object's payload plus a fixed per-object overhead. An op that would pass it
  fails with the located runtime error `heap exhausted` (the action and line
  of the op, like other runtime errors).
- The engine keeps a `live` object count and a byte total; tests read them
  to check that loops do not leak (`live` back to its value before the loop).

## 6. The `HEAP` opcode

One opcode, `HEAP` (0x1B, `zuspec.be.bc.model.Op.HEAP`). Its `imm` is
`model.heap_imm(sub, kind, cell, key)`: bits 0-7 the sub-op (`HEAP_OPS`
order), bits 8-9 the operand's kind (`HEAPK_STR/LIST/MAP/SET`; an op on
handle 0 allocates that kind), bits 10-11 what its elements or values hold
and bits 12-13 what a map's keys hold (`HEAP_CELL_INT/STR/COLL`, so copy,
free and compare know which cells are handles). Bit 14 (`HEAP_NEG`) is used
only on `STR_EQ`: it inverts the result, so `!=` is one op. Operands are registers
`a`, `b`, `c` (unused ones ignored) and the result goes to `rd`. A `SET`,
`DELETE` or `CLEAR` frees the cells it drops; a `POP_*` moves its cell out. These ops are not engine-served imports: they run in loops and
must not enter the host ABI.

"h" is a collection handle, "s" a string value (section 3), "v" a cell value,
"i" an integer index, "k" a map key. Indices are signed 64-bit.

| Sub-op | Operands | Result | Errors |
|---|---|---|---|
| `SIZE` | h (or s) | element count (string: bytes) | |
| `GET` | h, i (list) / h, k (map) | the cell | `index out of range`; `missing map key` |
| `SET` | h, i, v (list) / h, k, v (map: insert or replace; the old value is freed unless it is v itself, a nested collection changed in place) | h' | `index out of range` (list) |
| `INSERT` | h, i, v (list) / h, k, v (map) / h, v (set) | h' | `index out of range` (list, i > size) |
| `DELETE` | h, i (list) / h, k (map, set) | h' | `index out of range` (list); missing key or element is no error |
| `PUSH_FRONT` / `PUSH_BACK` | h, v | h' | |
| `POP_FRONT` / `POP_BACK` | h | the cell; h' via `rd+1` | `pop of an empty list` |
| `CLEAR` | h | h' (0) | |
| `IN` | h, v (list, set) / h, k (map keys) | 1 or 0 | |
| `EQ` | h1, h2 (same type) | 1 or 0 (sizes, then cells in order; sets and maps compare as sets) | |
| `COPY` | h (or s) | a deep copy (0 and table strings unchanged) | `heap exhausted` |
| `FREE` | h (or s) | none | |
| `KEYS` | h (map) | new set | |
| `VALUES` | h (map) | new list | |
| `TO_LIST` | h (set) | new list | |
| `TO_SET` | h (list) | new set | |
| `SHUFFLE` | h (list) | h' (permuted with the run's RNG) | |
| `JOIN` | h (list of strings), s (separator) | new string | |
| `STR_SLICE` | s, a, b (`a..b` inclusive; -1 = open end) | new string | `index out of range` |
| `STR_GET` | s, i | the byte (`bit[8]`) | `index out of range` |
| `STR_FIND` | s, sub, first | index or -1 | |
| `STR_FIND_LAST` | s, sub, first (-1 = from the end) | index or -1 | |
| `STR_FIND_ALL` | s, sub | new `list<int>` | |
| `STR_LOWER` / `STR_UPPER` | s | new string | |
| `STR_SPLIT` | s, sep | new `list<string>` | |
| `STR_CHARS` | s | new `list<bit[8]>` | |
| `STR_FROM_CHARS` | h (list of `bit[8]`) | new string | |
| `STR_EQ` | s1, s2 | 1 or 0 (index fast path, else content) | |
| `FORMAT` | message-table format index, arg window | new string | as `message()` formatting |

- `POP_*` has two results: the popped cell in `rd` and the new handle in
  `rd+1`.
- `FORMAT` renders like `message()` (same directives, `%s` of a heap string
  prints its text), into a heap string instead of the message sink; it backs
  `format_string`. Register `a` holds the index of a message-table entry
  (the one `message()` would use); the values are read from that entry's
  slots, which the lowering fills before the op, as for `message()`.
- String `==` and `!=` lower to `CMP_EQ` / `CMP_NE` on the table indices
  unless the image makes heap strings: when any `HEAP` op is emitted, the
  lowering patches every recorded string compare (`Lowerer.string_cmps`,
  patched in `zuspec.be.bc.lower.driver`) to `STR_EQ`, with `HEAP_NEG` for
  `!=`. So an image that only uses literal strings has no `HEAP` op and still
  runs on engines without a heap.
- Every error above is a runtime error located at the op (action and line),
  as the other runtime errors are; the run stops.
- Iteration is `SIZE` then `GET` by index; a `foreach` over a map or set walks
  `KEYS` / `TO_LIST` (a copy, so changing the collection inside the loop does
  not change the walk).

## 7. Determinism

- Map and set iteration, `KEYS`, `VALUES` and `TO_LIST` follow insertion
  order (re-inserting an existing key keeps its place). Nothing depends on
  hash order, addresses or handle numbers.
- `SHUFFLE` draws from the run's RNG (determinism.md section 1), so a seed
  gives the same order in every engine.
- Handle numbers depend only on the sequence of ops (section 2).

## 8. Strings across imports

- A string **argument** crosses as its slot value (host-abi.md "String
  arguments"); the host resolves either form to text with `pssvm_string(e,
  v)` / `HostCall.string(v)`.
- A string **result**: the host hands text and the engine makes a heap
  string (host-abi.md "String results").
- Collections across imports (as arguments or results) are refused at
  lowering, located, until a later item.

## 9. Engines

- Python engine (oracle): implements everything here (F6.9c on).
- C engine (`zuspec-rt-eng`): no heap until M14 ("heap last"). `zbc_run`
  refuses an image that contains `HEAP`, or an `IMPORT` of the builtins
  `file_read` / `file_read_lines` / `file_write_lines` (they make or read heap
  values; [host-abi.md](host-abi.md)), before running anything:
  `ZBC_ERR_UNSUPPORTED_OP`, `halted_op` = `HEAP`, and the message names the
  heap ("heap ops (strings built at run time, list/map/set) are not
  supported by the C engine yet (M14)"). The simulator lane on the C engine
  lists heap cases as refused by design; the simulator proof of heap cases
  runs on pyvm.
