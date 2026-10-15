"""Smoke test: the package imports and reports its version."""

import importlib.util
import pathlib


def test_import_package():
    import zuspec.be.bc as bc

    # The version is written once, in __version__.py; the package reports it.
    path = pathlib.Path(bc.__file__).with_name("__version__.py")
    spec = importlib.util.spec_from_file_location("_bc_version", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert bc.__version__ == mod.VERSION
