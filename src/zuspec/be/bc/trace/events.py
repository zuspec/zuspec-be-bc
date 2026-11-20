"""
events.py -- the debug event stream's masks (pssc P1.6).

Debug events are separate from the orchestration trace: they go to the VM's
``events`` sink, never to its trace ``sink``, so a run's orchestration trace is
the same bytes whether events are on or off. Which events a code unit (a
coroutine) raises is a bit mask, set per coroutine with ``VM.watch``; the VM
tests it once each time a frame is entered or resumed, so a run with no events
on pays one test per frame entry, not one per instruction.

Kinds and their bits:

* ``stmt``                    -- ``EV_STMT``: a PSS statement starts (debug
  lowering's LINE table, ``CoroDescriptor.stmts``); ``src_ref`` is the
  statement's, ``detail`` = ``{"pc"}`` (+ ``path`` and, in a function,
  ``scope``: the statement's coverage item, ``CoroDescriptor.cov``; + ``defs``
  and ``uses``: the values it stores and loads, ``interp.defuse``).
* ``branch``                  -- ``EV_BRANCH``: a branch arm (of an ``if``,
  if-chain or ``match``) is taken: ``{"pc", "branch"`` (the branch
  statement's path)``, "arm"`` (``then``, ``else``, a clause or case index,
  ``default``)``}`` (+ ``scope``; ``implicit`` for an unwritten ``else``).
  Needs debug lowering (``CoroDescriptor.cov``).
* ``call`` / ``return``       -- ``EV_CALL``: a called function's frame starts /
  ends (``detail`` = ``{"fn", "depth"}``; ``return`` adds ``retval`` when it is
  an integer). Raised by the callee's mask.
* ``test.start`` / ``test.end`` -- ``EV_TEST``: raised by the host (pssc) around
  a test, through ``VM.host_event``.
* ``access``                  -- ``EV_ACCESS``: a memory builtin
  (``read32``/``write32``...): ``detail`` = ``{"rw": "r"|"w", "width"`` (bits)
  ``, "addr", "data"}``.
* ``import``                  -- ``EV_IMPORT``: a user import returned:
  ``{"name", "fn_id", "args", "blocking", "mocked"}`` (+ ``ret``). ``mocked``
  is the import provider's ``is_mocked(fn_id)``, False when it has none.
* ``message``                 -- ``EV_MESSAGE``: ``message()`` printed a line,
  or ``error()``/``fatal()`` is about to stop the run: ``{"severity"`` (
  ``"info"|"error"|"fatal"``)``, "text"}`` (+ ``level`` / ``status``).
* ``check``                   -- ``EV_CHECK``: a function the host named a
  check (``VM.on_check``) was called; the host's hook builds the detail
  (pass/fail, what) and may give the src_ref of the check's call site.
* ``chan``                    -- ``EV_CHAN``: a channel operation (``put``,
  ``get``, ``try_put``, ``try_get``) ended: ``{"channel"`` (the name as the
  code wrote it)``, "op", "value"`` (the element put or taken; not on a
  failed try)``}`` (+ ``ok`` for a try). Needs debug lowering
  (``CoroDescriptor.chans``).
* ``notify``                  -- ``EV_NOTIFY``: raised by the host (the
  platform) through ``VM.host_event`` when a notification (an interrupt)
  reaches the model. pssc's ``Ctx.raise_irq`` gives ``{"irq", "delivered"}``;
  a host-ABI notification (``HostCall.notify``, C ``pssvm_notify``) gives
  ``{"irq"|"chan": id, "value", "delivered"}`` (``{"kind", ...}`` for an
  unknown kind), raised as the next ``run`` takes it.
* ``sanitizer``               -- ``EV_SANITIZER``: raised by the host's
  sanitizers (hooks on this stream) through ``VM.host_event``:
  ``{"name"`` (``inflight``, ``unmapped``...)``, "verdict"`` (``"fail"``)``,
  "what"}``, src_ref = the location it blames.
* ``reload``                  -- ``EV_RELOAD``: raised by the host when a new
  generation of the image is applied (``interp/generations.py``): the
  reload's result dict (``old``, ``new``, ``changed``, ``structure``,
  ``migrated``, ``cancelled``, ``applied``).

The VM raises access/import/message from the IMPORT op of a watched
coroutine, check from a watched check function's frame entry, and chan at the
NOP that ends a channel operation. test, notify, sanitizer and reload are the host's
(``VM.host_event``); a ``watch`` of every coroutine sets them.
"""

from typing import Iterable

EV_STMT = 1 << 0
EV_CALL = 1 << 1
EV_TEST = 1 << 2
EV_ACCESS = 1 << 3
EV_IMPORT = 1 << 4
EV_MESSAGE = 1 << 5
EV_CHECK = 1 << 6
EV_CHAN = 1 << 7
EV_NOTIFY = 1 << 8
EV_SANITIZER = 1 << 9
EV_BRANCH = 1 << 10
EV_RELOAD = 1 << 11

EV_ALL = (EV_STMT | EV_CALL | EV_TEST | EV_ACCESS | EV_IMPORT | EV_MESSAGE
          | EV_CHECK | EV_CHAN | EV_NOTIFY | EV_SANITIZER | EV_BRANCH | EV_RELOAD)
#: the events the host raises (``VM.host_event``)
EV_HOST = EV_TEST | EV_NOTIFY | EV_SANITIZER | EV_RELOAD
#: the platform side of a run: what the host answered (read data, import
#: returns) and when it notified; a replay (pssc P5.1) feeds these back
EV_PLATFORM = EV_ACCESS | EV_IMPORT | EV_NOTIFY
#: the events raised from an IMPORT op
EV_AT_IMPORT = EV_ACCESS | EV_IMPORT | EV_MESSAGE

STMT = "stmt"
CALL = "call"
RETURN = "return"
TEST_START = "test.start"
TEST_END = "test.end"
ACCESS = "access"
IMPORT = "import"
MESSAGE = "message"
CHECK = "check"
CHAN = "chan"
NOTIFY = "notify"
SANITIZER = "sanitizer"
BRANCH = "branch"
RELOAD = "reload"

#: event name -> its mask bit (``call`` and ``return`` share one; so do the
#: two test events)
BITS = {
    STMT: EV_STMT,
    CALL: EV_CALL, RETURN: EV_CALL,
    "test": EV_TEST, TEST_START: EV_TEST, TEST_END: EV_TEST,
    ACCESS: EV_ACCESS, IMPORT: EV_IMPORT, MESSAGE: EV_MESSAGE, CHECK: EV_CHECK,
    CHAN: EV_CHAN, NOTIFY: EV_NOTIFY, SANITIZER: EV_SANITIZER, BRANCH: EV_BRANCH,
    RELOAD: EV_RELOAD,
    "platform": EV_PLATFORM,
    "all": EV_ALL,
}


def mask_of(names: Iterable[str]) -> int:
    """The mask for event names (``["stmt", "call"]``); unknown names raise."""
    m = 0
    for n in names:
        try:
            m |= BITS[n]
        except KeyError:
            raise ValueError(f"unknown event {n!r} (have: "
                             f"{', '.join(sorted(BITS))})") from None
    return m


def platform_side(events: Iterable) -> list:
    """The platform's part of a run, in order: one tuple per answer the model
    took from the host -- ``("read", addr, width, data)`` for a read access
    (writes are the model's, not the platform's), ``("import", fn_id, ret)``
    for a user import (``ret`` None when it returns nothing) and
    ``("notify", detail)`` for a notification. *events* are ``TraceEvent``s
    or their dicts (NDJSON / ``binary.read``); other kinds are skipped. Two
    runs that took the same answers have equal lists, whichever engine ran
    them."""
    out = []
    for e in events:
        kind = e["kind"] if isinstance(e, dict) else e.kind
        d = (e.get("detail") if isinstance(e, dict) else e.detail) or {}
        if kind == ACCESS and d.get("rw") == "r":
            out.append(("read", d.get("addr"), d.get("width"), d.get("data")))
        elif kind == IMPORT:
            out.append(("import", d.get("fn_id"), d.get("ret")))
        elif kind == NOTIFY:
            out.append(("notify", dict(d)))
    return out
