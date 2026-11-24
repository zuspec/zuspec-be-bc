"""Files be-bc ships for other tools: ``sv/`` holds the SystemVerilog
runtime (``pssvm_pkg.sv`` and its pyhdl-if engine) that runs a ``.zbc``
image inside a simulation; ``py/`` holds its Python host
(``pssvm_dpi_shim``, the engine SV calls)."""

from pathlib import Path

#: The SV runtime's top package, in :func:`sv_dir`.
SV_PKG = "pssvm_pkg.sv"


def sv_dir() -> Path:
    """The directory of the SV runtime (``pssvm_pkg.sv`` and friends)."""
    return Path(__file__).resolve().parent / "sv"


def py_dir() -> Path:
    """The directory of the SV runtime's Python host (``pssvm_dpi_shim``),
    for PYTHONPATH."""
    return Path(__file__).resolve().parent / "py"
