# Memory: address handles, primitives and executors

bc models **transparent address spaces** only (LRM 21.13). Its address handles,
memory primitives and executors follow from that (bc procedural gaps B-D6).

## Handles

An `addr_handle_t` (a `chandle`) is a 64-bit value: its address.

- `add_region(r)` / `add_nonallocatable_region(r)` on a
  `transparent_addr_space_c` instance returns `r.addr`. On any other address
  space it is a lowering error.
- `make_handle_from_handle(h, offset)` is `h + offset`.
- `addr_value(h)` is `h`, unless the executor in force overrides it.

## Primitives

`read8/16/32/64(h)` and `write8/16/32/64(h, data)` (21.13.9) go to the
**executor in force** for the instance of the calling code (below). When it
declares the primitive, the access is a call to that function, inlined in the
executor's instance like any component function. When it does not, or when no
executor is in force, the platform answers through a builtin import:

| Builtin | fn_id | Args | Result |
|---|---|---|---|
| `BUILTIN_READ[N]`, N = 8/16/32/64 | `BUILTIN_BASE + 2..5` | `(addr)` | the N-bit value |
| `BUILTIN_WRITE[N]` | `BUILTIN_BASE + 6..9` | `(addr, data)` | none |

A value is little-endian: the byte at `addr` is bits [7:0] (21.13.9.1). The
oracle's platform is `interp.extern.Memory`: sparse, one byte per address, 0
where nothing was written. `run_model(memory=...)` substitutes another object
with the same `read(addr, nbytes)` / `write(addr, nbytes, value)`.

## Executors

`set_executor(x)` (21.7.2.6) is resolved **at lowering**, from the tree:

1. An instance's executor is the instance named by the last top-level
   `set_executor(path)` statement of its `exec init_down` or `exec init_up`.
2. Failing that, it is its parent's.

A `set_executor` that is not a top-level statement of an init block (inside an
`if` or a loop, say) makes the executor dynamic. An access whose executor is
dynamic is a lowering error. So is `set_executor` anywhere other than an init
block. At run time the statement does nothing.

The calling code's instance must be static: construction code runs in instance
0, and action code needs the only instance of its component type. The
executor must also lie in the frame's instance subtree, because `LD_COMP` /
`ST_COMP` address only below it. Either violation is a lowering error naming
the cause.

## Native engine

`zbc_run` refuses an image containing an `IMPORT` of a memory builtin
(`ZBC_ERR_UNSUPPORTED_OP`, `halted_op` = `IMPORT`) before running anything: it
has no platform memory to answer from. An executor's override is ordinary
component code, so it is refused as `LD_COMP`/`ST_COMP`/`$comp_init` already
are.

An `IMPORT` of `BUILTIN_ERROR` (a run-time error the LRM requires) halts the
run with `ZBC_ERR_RUNTIME` (-25), as the oracle raises.
