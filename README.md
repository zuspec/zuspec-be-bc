# zuspec-be-bc

The Zuspec ByteCode (ZBC) backend. It lowers a Zuspec scenario (the
`ScenarioModule` that `zuspec-ir-core` builds from a PSS model) to ZBC, a
compact bytecode, and runs it.

It is how [pssc](https://pypi.org/project/pssc/) executes PSS: activities,
constraints solved with [dv-solve](https://pypi.org/project/dv-solve/), flow
objects and resources, component trees, and procedural code, all on one
action object per activation.

## What is in the package

- **Lowering** (`zuspec.be.bc.lower`): Scenario IR to ZBC. A construct it
  cannot lower is refused with its source location, never dropped.
  Constraints go through zuspec-solver's shared translator
  (`zuspec.solver.translate`).
- **The `.zbc` format and the value ABI** (`format`, `abi`). The on-disk
  layouts are described once; the C header, the `ctypes` overlay and the
  Python dataclasses are generated from that description.
- **The Python interpreter** (`interp`): the reference implementation. Given
  a seed, a run is a pure function of the model, and its trace is what the
  native engine ([zuspec-rt-eng](https://pypi.org/project/zuspec-rt-eng/)) must
  reproduce.

## Documentation

`docs/spec/` in the source repository specifies the format, the value ABI,
activations, components, calls, memory, determinism and the trace schema.
`docs/oracle.md` describes the interpreter.

## License

Apache-2.0
