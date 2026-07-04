"""Smoke test: the package imports and reports its version."""


def test_import_package():
    import zuspec.be.bc as bc

    assert bc.__version__ == "0.0.1"
