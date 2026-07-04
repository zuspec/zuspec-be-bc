"""T0-D -- regenerating the C header must produce no diff vs the checked-in file."""

import os

from zuspec.be.bc.format import emit_c


def test_checked_in_header_matches_generator():
    path = emit_c.default_header_path()
    assert os.path.exists(path), (
        f"checked-in header missing at {path}; run "
        f"`python -m zuspec.be.bc.format.emit_c`"
    )
    with open(path) as fp:
        on_disk = fp.read()
    assert on_disk == emit_c.render_header(), (
        "generated zbc_format.h drifted from the spec; "
        "regenerate with `python -m zuspec.be.bc.format.emit_c`"
    )
