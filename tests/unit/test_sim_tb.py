"""The generic SV bench (share/sv/tb): it lints clean under Verilator.

The bench runs an image end to end in its own self-test (sim marker).
"""
import shutil
import subprocess
from pathlib import Path

import pytest

import zuspec.be.bc.share as share

SV = Path(share.__file__).parent / "sv"
ROOT = Path(__file__).resolve().parents[3]       # the workspace's packages/


def _verilator():
    v = shutil.which("verilator")
    if v:
        return v
    v = ROOT / "verilator" / "bin" / "verilator"
    return str(v) if v.exists() else None


VERILATOR = _verilator()


def test_tb_files_ship():
    assert (SV / "tb" / "pssvm_tb_pkg.sv").is_file()
    assert (SV / "tb" / "pssvm_tb.sv").is_file()


@pytest.mark.skipif(VERILATOR is None, reason="verilator not found")
def test_tb_lint():
    """The bench over pyvm's zsp_rt_bc_pkg variant (the C engine's ships
    with zuspec-rt-eng, which checks it there)."""
    try:
        import hdl_if
    except ImportError:  # pragma: no cover
        pytest.skip("pyhdl-if not found")
    dpi = Path(hdl_if.share()) / "dpi"
    # ZERODLY: platform delays (#(lat), wait_time) are run-time values.
    r = subprocess.run(
        [VERILATOR, "--lint-only", "--timing", "--timescale", "1ns/1ns", "-Wno-ZERODLY",
         "-Wno-fatal", f"+incdir+{SV}", f"+incdir+{dpi}", str(dpi / "pyhdl_if.sv"),
         str(SV / "zsp_rt_bc_pkg.sv"), str(SV / "pssvm_pkg.sv"),
         str(SV / "tb" / "pssvm_tb_pkg.sv"), str(SV / "tb" / "pssvm_tb.sv"),
         "--top-module", "pssvm_tb"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ours = [ln for ln in r.stderr.splitlines()
            if ("%Warning" in ln or "%Error" in ln) and "hdl_if" not in ln]
    assert not ours, r.stderr
