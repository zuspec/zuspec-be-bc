"""Files be-bc ships for other tools: ``sv/`` holds the SystemVerilog
runtime that runs a ``.zbc`` image inside a simulation -- ``zsp_rt_bc_pkg``
(pyvm's variant, over pyhdl-if; the C engine's variant ships with
zuspec-rt-eng and includes this directory's ``zsp_rt_bc_common.svh``); its
PSS face, ``pssvm_pkg``, is pssc's. ``py/`` holds pyvm's Python host
(``pssvm_dpi_shim``, the engine SV calls). ``rt_bc_api`` is the one method
list both variants' engine class come from."""

from pathlib import Path

#: The generic runtime's package (pyvm's variant here), in :func:`sv_dir`;
#: compile it with :func:`sv_dir` on +incdir.
SV_RT_BC_PKG = "zsp_rt_bc_pkg.sv"


def sv_dir() -> Path:
    """The directory of the generic SV runtime (``zsp_rt_bc_pkg.sv``,
    ``zsp_rt_bc_common.svh`` and friends)."""
    return Path(__file__).resolve().parent / "sv"


def py_dir() -> Path:
    """The directory of the SV runtime's Python host (``pssvm_dpi_shim``),
    for PYTHONPATH."""
    return Path(__file__).resolve().parent / "py"
