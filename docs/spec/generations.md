# Generations

Status: **pssc P3.6b3 (pyvm: API check, load beside N, quiesce, migrate, commit: `Generations`; pssc `Session.reload`, `pssc dbg reload`); C engine from a Python host: rt-eng `reload_engine` (P4.6a)**
· Version `1` · Authority: this file, then `zuspec.be.bc.interp.generations`
(pyvm) and `zuspec-rt-eng` (native) · Companion to the pssc design
(`pss-test-debug-env-design.md`, "Hot reload"). ABI: [host-abi.md](host-abi.md)
§Reload. Image identity: [zbc-format.md](zbc-format.md) §Image identity (`IDENT`).

A running engine holds one image, **generation N**. A reload loads a rebuilt
image as **generation N+1** beside N and moves new calls over to it, so an
edited test or operation runs without restarting the simulator or the
session. A reload never replaces code in an active frame (no on-stack
replacement).

## The steps

1. **Check** (P3.6a1). N+1 must implement N's API, or the host (the SV
   facade, pssc's session) could no longer call it. Refused otherwise, with
   the list of changed signatures (§API check).
2. **Load** (P3.6a2). N+1 is loaded beside N; nothing binds to it yet.
3. **Quiesce** (P3.6a2). The reload applies at the next reload point:
   - an **operation boundary** (default): no call live in N. Calls already
     running finish on N; new `pssvm_call_start`s wait (flag `PSSVM_R_AT_CALL`:
     apply before the next call start instead of waiting for all calls).
     `cancel` ends running calls instead (`--cancel` in pssc).
   - a **test boundary**: between two tests of a run.
4. **Migrate** (P3.6a3). The component tree is rebuilt under N+1 with the
   root constructor arguments recorded at N's construction, in
   **bus-isolated mode**: a memory access or a user import during
   construction refuses the reload (reason `bus`), since it would program the
   device behind the test's back. Channel contents (`wake`, `inflight`
   tokens) are carried across by path. A **structural change** (a component,
   channel or sub-component added or removed) is reported as a diff; untouched
   components migrate, the rest start fresh. With `strict`, any structural
   change refuses the reload (reason `structure`).

   pyvm: the constructing coroutine takes no arguments, so N+1 is constructed
   as N was. The structural check and the bus-isolated construction run at
   reload; the state is copied at commit, so N+1 starts from N's state when
   its last call ended. An instance migrates when its path, type and field
   names are unchanged: its slots of the component object are copied (a
   channel's tokens are fields of its component). The result carries
   `"structure": {"added", "removed", "changed"}` (instance paths, sorted;
   `changed` = another type or field list) and `"migrated"` (paths).
5. **Commit**. New calls bind to N+1 by stable symbol (the `ENTRY` name, never
   a coroutine index). N is freed once no call runs in it. The host logs a
   `reload` event: old and new generation, symbols changed, migration result.

   pyvm: the event is `reload` (`EV_RELOAD`, a host event like `test.*`), its
   detail the result dict. Hosts other than `Generations` (pssc's
   `bc_session.Session`, one tree called synchronously) build on the same
   pieces: `check_reload(old, new, strict, n)` (step 1 and the `strict` check;
   returns the result dict), `construct_isolated(vm, n)` (step 4's
   bus-isolated construction) and `migrate(old_vm, new_vm)` (the copy).

A refused reload leaves N running and unchanged.

## API check

The API is what `model.api_hash_of` hashes: each entry (symbol, kind,
component type, parameters, return type) and each user import (`fn_id`,
blocking, argument and return types, string positions). A body-only edit
keeps it.

`check_api(N, N+1)`:

- both `IDENT.api_hash` set: equal accepts; unequal refuses.
- either unset (a release image): the signatures are compared directly, and
  any difference refuses.

The refusal lists each differing signature, sorted by symbol:

| `symbol` | `what` | `old` / `new` |
|---|---|---|
| `entry:<sym>` | `added`, `removed`, `changed` | `{kind, comp_type, params, returns}` |
| `import:<name>` | same | `{fn_id, blocking, arg_types, ret_type, string_at}` |
| `import:*` | `changed` | none |

Import signatures are not serialized: an image read back from bytes carries
only the hash. When the hashes differ but the entries match and one side has
no imports, the refusal names `import:*`.

pyvm raises `ReloadRefused` (`code` = `PSSVM_E_API` = -4, `reason`,
`changes`); its `to_json()` is the JSON `pssvm_error` returns:

```json
{"code": -4, "reason": "api",
 "message": "reload refused: the API changed: entry:dma_c::start (changed)",
 "changes": [{"symbol": "entry:dma_c::start", "what": "changed",
              "old": {"kind": "function", "comp_type": "dma_c",
                      "params": [["ch", {"kind": "int", "width": 32}]], "returns": null},
              "new": {"...": "..."}}]}
```

## pssc

pssc's hosts of this spec, without a simulator:

- `bc_session.Session.reload(model, cancel=, strict=)`: one tree, called
  synchronously. Between calls the reload applies at once; inside a call
  (`run_entry` counts as one) it is staged and applies when the outermost
  call returns. Host bindings carry over; the result is appended to
  `Session.history` and raised as the `reload` event.
- `pssc dbg reload [--strict] [--cancel] [--json]`: lowers a debug
  session's sources again and calls `Session.reload`; the platform doubles
  are re-imported at once (`vm.Session.reimport`).
- Coverage: points are keyed by scope and block path, not pc, so a
  `Coverage` sink counts N's and N+1's hits on the same points; at apply
  it takes N+1's points and lines (`Coverage.add_model`).

User documentation: pssc `docs/hot-reload.rst`.

## Refusal reasons

| `reason` | When | Step |
|---|---|---|
| `api` | the API differs | check |
| `bus` | construction under N+1 made a memory access or import | migrate |
| `structure` | a structural change under `strict` | migrate |

## Versioning

Version 1. A new refusal reason or a new field in the change record is
additive; a host ignores what it does not know.
