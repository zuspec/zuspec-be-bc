"""
zuspec.be.bc.format -- the ``.zbc`` container format (design D§12).

One declarative spec (:mod:`.spec`) describes every on-disk record and enum. The
C header (:mod:`.emit_c`), the ``ctypes`` overlay (:mod:`.emit_ctypes`), and the
in-memory dataclasses (:mod:`.emit_dataclass`) are all generated from it, so the
Python writer/reader and the C engine interpret identical bytes by construction.

:mod:`.writer` / :mod:`.reader` serialize and parse a whole image (header +
section directory + payloads); :mod:`.inspect` walks the directory generically.
"""

from . import spec  # noqa: F401
