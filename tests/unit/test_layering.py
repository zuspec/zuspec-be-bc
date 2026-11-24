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
    """The SV runtime (F1.13a) is be-bc package data, found without pssc."""
    from zuspec.be.bc.share import SV_PKG, sv_dir
    d = sv_dir()
    for f in (SV_PKG, "pssvm_hif_pkg.sv", "pssvm_hif_api_pkg.sv"):
        assert (d / f).is_file(), f
