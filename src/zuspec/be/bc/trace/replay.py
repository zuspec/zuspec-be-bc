"""``--platform replay:<trace>``: run a test against a recorded run (from pssc, F1.12).

The trace is what ``pssc test --trace`` wrote (NDJSON, one file per test),
or the same records in the binary form (``.pstrace``: what the C engine and
a simulation write; its events have no source line):
each bus access (``access``: rw, width, addr, data) and each user import
call (``import``: name, ret) in the order the run made them. Replay answers
the run's reads and imports from it, in order, and checks that each access
and import is the one recorded. The first that is not stops the test
(``ReplayDivergence``): the model, the test or the seed no longer does what
the recorded run did. A run that ends before the recording does diverges
too (``ReplayPlatform.left``).

Interrupts the recorded run raised (``notify`` events) are raised again
(``Ctx.raise_irq``) while the replay serves the access or import recorded
after them -- the one whose double raised them -- so the handler runs at
the same point. One recorded after the last access or import is not.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..interp import VMError
from ..platform.base import Platform
from .steps import _show, load_trace, same_step


class ReplayDivergence(VMError):
    """The replayed run made an access or import the recording did not."""


class ReplayPlatform(Platform):
    """Answers reads and user imports from a recorded trace (see the
    module). ``mem`` and ``log`` are kept as ``Platform``'s are.
    ``user_import``: which recorded imports are steps (``steps``)."""

    def __init__(self, path: str, test: Optional[str] = None,
                 user_imports=(), user_import=None):
        super().__init__()
        self.path = path
        self.header, self.steps = load_trace(path, user_import)
        rec = self.header.get("test")
        if test is not None and rec is not None and rec != test:
            raise ValueError("replay: %s is a trace of %s, not %s"
                             % (path, rec, test))
        self.pos = 0
        self._session = None
        # every user import is answered (or diverges) here, recorded or not
        self._names = {s["name"].rpartition("::")[2] for s in self.steps
                       if s["kind"] == "import"} | set(user_imports)

    def attach_session(self, session) -> None:
        """The session the replay raises recorded interrupts in
        (``Session`` calls it)."""
        self._session = session

    def _next(self, actual: Dict[str, Any]) -> Dict[str, Any]:
        exp = self.steps[self.pos] if self.pos < len(self.steps) else None
        if not same_step(exp, actual):
            raise ReplayDivergence(
                "first divergent access (step %d of %d recorded): expected %s, "
                "got %s" % (self.pos + 1, len(self.steps),
                            _show(exp) if exp else "the end of the recording",
                            _show(actual)))
        self.pos += 1
        if exp.get("irqs") and self._session is not None:
            for irq in exp["irqs"]:
                self._session.ctx.raise_irq(irq)
        return exp

    def _read(self, width: int, addr: int) -> int:
        d = self._next({"kind": "access", "rw": "r", "width": width,
                        "addr": addr})
        v = int(d.get("data") or 0)
        Platform._write(self, width, addr, v)
        self.log[-1] = ("read", width, addr, v)
        return v

    def _write(self, width: int, addr: int, data: int) -> None:
        data &= (1 << width) - 1
        self._next({"kind": "access", "rw": "w", "width": width,
                    "addr": addr, "data": data})
        Platform._write(self, width, addr, data)

    def __getattr__(self, name):
        if name in self.__dict__.get("_names", ()):
            def call(*args):
                return int(self._next({"kind": "import", "name": name})
                           .get("ret") or 0)
            return call
        raise AttributeError(name)

    def left(self) -> Optional[str]:
        """Why the run diverged by ending early, or None: the first recorded
        step it did not make."""
        if self.pos >= len(self.steps):
            return None
        return ("first divergent access (step %d of %d recorded): expected "
                "%s, got the end of the test" % (
                    self.pos + 1, len(self.steps), _show(self.steps[self.pos])))
