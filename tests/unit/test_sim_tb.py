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
def test_tb_lint_dpi():
    # ZERODLY: platform delays (#(lat), wait_time) are run-time values.
    r = subprocess.run(
        [VERILATOR, "--lint-only", "--timing", "--timescale", "1ns/1ns", "-Wno-ZERODLY",
         "+define+PSSVM_DPI", str(SV / "pssvm_pkg.sv"), str(SV / "tb" / "pssvm_tb_pkg.sv"),
         str(SV / "tb" / "pssvm_tb.sv"), "--top-module", "pssvm_tb"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "%Warning" not in r.stderr and "%Error" not in r.stderr, r.stderr
