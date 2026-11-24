# Memory: address handles, primitives and executors

bc's address handles, memory primitives and executors model **transparent
address spaces** (LRM 21.13; bc procedural gaps B-D6). Claims in transparent and
contiguous spaces are allocated as described under "Claims and allocation".

## Handles

An `addr_handle_t` (a `chandle`) is a 64-bit value: its address.

- `add_region(r)` / `add_nonallocatable_region(r)` on a
  `transparent_addr_space_c` instance returns `r.addr`. On any other address
  space it is a lowering error.
- `make_handle_from_handle(h, offset)` is `h + offset`.
- `make_handle_from_claim(c, offset = 0)` (24.10.4.1) of a transparent claim
  is `c.addr + offset`. A contiguous claim (`addr_claim_s`) has a hidden
  non-random `$addr` field: the run writes its first-fit address there
  when it places the claim, after the solve and before the action's exec
  blocks, and the handle is `c.$addr + offset`. In the image, each
  contiguous claim of an ACT node names its `$addr` slot (format spec,
  `zbc_actnode.addr0`).
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

## Claims and allocation

Design: pssc `docs/design/address-space-bc-design.md` (F7.1–F7.3).

- **Memory map.** The host records each address space instance and the regions
  added to it in the `MEMMAP` section (`zbc_memmap`: one JSON string,
  `{"spaces": [...], "regions": [...]}`). Region values must be constant.
- **Transparent claims** are solved: the compiler adds constraints that put each
  claim in a matching region, aligned, and disjoint from every claim whose
  owner may run at the same time (parallel branches, several traversals) and
  from every live entry of the run's ledger.
- **Contiguous claims** never reach the solver. After the solve, the run's
  allocator places them first-fit in a matching region, aligned, skipping live
  ledger entries (`ActivationTable._addr_place`); no fit is a run error naming
  the claim.
- **Ledger.** Each action node lists its claims (`zbc_actnode.addr0`: space,
  addr, size and permanent slots; contiguous claims add alignment and trait
  slots). An entry is live until its owner ends; a `permanent` claim stays
  live to the end of the run. Runs from an image keep the same ledger
  (`set_regions` from `MEMMAP`), so addresses match for a seed.
- **Refused** at build time: a claim in a struct, a shared allocation mode
  (`alloc_mode` named in a constraint or `with`), a non-constant region.

## Native engine

The C engine has no claim ledger: `zbc_run` refuses an image whose activation
has address claims (`act_claims_addrs`, `ZBC_ERR_UNSUPPORTED_OP`) before running
anything, until the ledger is ported (M14).

An `IMPORT` of a memory builtin calls the host's `mem_access(ctx, is_write,
addr, nbytes, data)` (`zbc_host_t`, `zbc_interp.h`), as the oracle calls
`vm.memory.read`/`write`: write data is masked to `nbytes` and a read's result
is masked to it too. The call is also recorded in the host's import log. With
no `mem_access`, `zbc_run` refuses the image (`ZBC_ERR_UNSUPPORTED_OP`,
`halted_op` = `IMPORT`) before running anything. An executor's override is
ordinary component code (`LD_COMP`/`ST_COMP`).

An `IMPORT` of `BUILTIN_ERROR` (a run-time error the LRM requires) halts the
run with `ZBC_ERR_RUNTIME` (-25), as the oracle raises.
