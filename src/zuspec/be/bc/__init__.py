"""
zuspec.be.bc -- the Zuspec ByteCode (ZBC) backend.

This package owns three things:

* the cross-tier contracts (the value ABI, the ``.zbc`` container format, the
  trace/event schema, and the determinism spec) that both the Python oracle and
  the future native engine agree on;
* the lowering from Scenario IR to ZBC; and
* the Python ZBC interpreter that serves as the reference *oracle*.

The single source of truth for every on-disk record layout lives in
:mod:`zuspec.be.bc.format.spec` and :mod:`zuspec.be.bc.abi.value`; the C header,
the ``ctypes`` overlay, and the in-memory dataclasses are all generated from
those descriptions ("one spec, many emitters").
"""

__version__ = "0.0.1"
