"""Coverage of a run: the collector (:mod:`.coverage`, a trace sink) and its
report as a UCIS database (:mod:`.ucis`; needs covsight, imported only
there)."""

from .coverage import VERSION, Coverage, point_id

__all__ = ["VERSION", "Coverage", "point_id"]
