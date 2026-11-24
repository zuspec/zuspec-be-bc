"""Platforms a run may be given: sparse memory (:class:`Platform`) and the
register fake (:class:`RegisterFake`), with field kinds read from a register
description (:mod:`.reg_semantics`)."""

from .base import Platform, reg_names
from .fake import Access, RegisterFake, on_read, on_write

__all__ = ["Platform", "reg_names", "Access", "RegisterFake", "on_read",
           "on_write"]
