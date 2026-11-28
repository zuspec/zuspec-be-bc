"""Layering: no be-bc module imports pssc (full-PSS plan F1.1).

be-bc is the backend; pssc is one front end over it. A ``pssc`` import here
would make the backend unusable without the front end.
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


def _pssc_imports(root):
    """Yield ``(path, line, module)`` for every import of ``pssc`` under *root*."""
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            elif (isinstance(node, ast.Call) and node.args
                  and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)
                  and getattr(node.func, "attr", getattr(node.func, "id", ""))
                  in ("import_module", "__import__")):
                names = [node.args[0].value]
            for name in names:
                if name == "pssc" or name.startswith("pssc."):
                    yield path, node.lineno, name


def test_no_pssc_imports():
    found = [f"{p.relative_to(SRC)}:{line}: {mod}"
             for p, line, mod in _pssc_imports(SRC)]
    assert found == []


def test_scanner_finds_each_import_form(tmp_path):
    (tmp_path / "m.py").write_text(
        "import pssc\n"
        "from pssc.testing import x\n"
        "def f():\n"
        "    import importlib\n"
        "    importlib.import_module('pssc.cli')\n"
        "    __import__('pssc')\n"
        "from . import pssc\n"
        "import psscx\n")
    got = [(line, mod) for _, line, mod in _pssc_imports(tmp_path)]
    assert got == [(1, "pssc"), (2, "pssc.testing"), (5, "pssc.cli"),
                   (6, "pssc")]


def test_sv_runtime_ships_in_be_bc():
    """The generic SV runtime (zsp_rt_bc_pkg, pyvm's variant) is be-bc
    package data, found without pssc."""
    from zuspec.be.bc.share import SV_RT_BC_PKG, sv_dir
    d = sv_dir()
    for f in (SV_RT_BC_PKG, "zsp_rt_bc_common.svh", "zsp_rt_bc_if.svh",
              "zsp_rt_bc_defaults.svh"):
        assert (d / f).is_file(), f


def test_sv_runtime_names_no_pss_face():
    """be-bc's SV is the generic layer: it never names pssc's PSS face
    (pssvm_pkg, pssvm_if, the bench), which builds on it (pssc
    pssvm-in-practice 3.0.6 note 7). Comments and message text may."""
    import re
    from zuspec.be.bc.share import sv_dir
    pss_face = re.compile(r"\bpssvm_(pkg|uvm_pkg|tb\w*|if|vm|runner|"
                          r"listener|event|action|void)\b")
    bad = []
    for f in sorted(sv_dir().rglob("*.sv*")):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            code = re.sub(r'"[^"]*"', '""', line).split("//", 1)[0]
            if pss_face.search(code):
                bad.append(f"{f.name}:{n}: {line.strip()}")
    assert not bad, "\n".join(bad)

def test_sv_runtime_reads_no_plusargs():
    """zsp_rt_bc_pkg reads no plusargs, in either variant: its host passes
    every option to ``load`` (pssc plusargs-plan R6; the PSS face reads
    ``+pssvm.*``). A plusarg here is a hidden global input, and its name
    would be the face's vocabulary in the generic layer."""
    import re
    from zuspec.be.bc.share import sv_dir
    dirs = [sv_dir()]
    try:
        from zuspec.rt import eng
        dirs.append(eng.SHARE_DIR)
    except ImportError:  # pragma: no cover -- rt-eng not installed
        pass
    read = re.compile(r"\$(value|test)\$plusargs")
    bad = []
    for d in dirs:
        for f in sorted(__import__("pathlib").Path(d).rglob("*.sv*")):
            for n, line in enumerate(f.read_text().splitlines(), 1):
                if read.search(line.split("//", 1)[0]):
                    bad.append(f"{f}:{n}: {line.strip()}")
    assert not bad, "\n".join(bad)
