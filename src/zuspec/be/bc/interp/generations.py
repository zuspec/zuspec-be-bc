"""Generations: loading an image beside a running one (``docs/spec/generations.md``, P3.6a).

A reload loads image N+1 next to N and, at a quiescent boundary, moves new
calls over to it. It is only possible when N+1 implements the same API as N:
:func:`check_api` compares the two images' API (each entry's and each user
import's signature, the parts :func:`~..model.api_hash_of` hashes) and raises
:class:`ReloadRefused` with the list of changed signatures when they differ::

    try:
        check_api(old_model, new_model)
    except ReloadRefused as e:
        for c in e.changes:
            print(c.symbol, c.what)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

#: return code of a refused reload (``pssvm_reload``)
PSSVM_E_API = -4

#: the symbol a refusal names when the API hashes differ but neither image
#: carries the import signatures (a read-back image keeps only the hash)
UNKNOWN_IMPORTS = "import:*"


@dataclass(frozen=True)
class ApiChange:
    """One signature that differs between generation N and N+1."""
    symbol: str                 # "entry:<sym>" or "import:<name>"
    what: str                   # "added" | "removed" | "changed"
    old: Any = None             # the signature in N (None when added)
    new: Any = None             # the signature in N+1 (None when removed)

    def to_json(self) -> dict:
        return {"symbol": self.symbol, "what": self.what, "old": self.old, "new": self.new}


class ReloadRefused(RuntimeError):
    """A reload that cannot apply (``PSSVM_E_API``). :attr:`changes` lists
    what differs; :attr:`reason` is one word for the event and the CLI."""
    code = PSSVM_E_API

    def __init__(self, reason: str, message: str, changes: Optional[List[ApiChange]] = None):
        super().__init__(message)
        self.reason = reason
        self.changes = list(changes or [])

    def to_json(self) -> dict:
        return {"code": self.code, "reason": self.reason, "message": str(self),
                "changes": [c.to_json() for c in self.changes]}


def api_of(model) -> Dict[str, Any]:
    """Symbol -> signature of each entry and each user import of ``model``,
    in the form :func:`~..model.api_hash_of` hashes. Imports are present for
    an image lowered in this process and one read from an image whose IMPORT
    rows carry their signatures (format 2.0)."""
    api: Dict[str, Any] = {}
    for sym, e in (model.entries or {}).items():
        api["entry:" + sym] = {"kind": e.kind, "comp_type": e.comp_type,
                               "params": [[n, d] for n, d in e.params],
                               "returns": e.returns}
    for name, sig in (model.imports or {}).items():
        api["import:" + name] = {k: sig.get(k) for k in
                                 ("fn_id", "blocking", "arg_types", "ret_type", "string_at",
                                  "ret_types")
                                 if k != "ret_types" or k in sig}
    return api


def api_diff(old, new) -> List[ApiChange]:
    """The signatures that differ between ``old`` and ``new``, sorted by
    symbol. Imports are compared only when both models carry them."""
    a, b = api_of(old), api_of(new)
    if not (old.imports and new.imports):
        a = {k: v for k, v in a.items() if not k.startswith("import:")}
        b = {k: v for k, v in b.items() if not k.startswith("import:")}
    changes = []
    for sym in sorted(set(a) | set(b)):
        if sym not in b:
            changes.append(ApiChange(sym, "removed", old=a[sym]))
        elif sym not in a:
            changes.append(ApiChange(sym, "added", new=b[sym]))
        elif a[sym] != b[sym]:
            changes.append(ApiChange(sym, "changed", old=a[sym], new=b[sym]))
    return changes


def check_api(old, new) -> None:
    """Refuse a reload of ``new`` over ``old`` unless both implement the same
    API. With both API hashes set, equal hashes accept and unequal ones
    refuse; the change list is :func:`api_diff`, or :data:`UNKNOWN_IMPORTS`
    when only imports neither model carries can account for the difference.
    With either hash unset the signatures are compared directly."""
    if old.api_hash and new.api_hash:
        if old.api_hash == new.api_hash:
            return
        changes = api_diff(old, new) or [ApiChange(UNKNOWN_IMPORTS, "changed")]
    else:
        changes = api_diff(old, new)
        if not changes:
            return
    names = ", ".join(f"{c.symbol} ({c.what})" for c in changes)
    raise ReloadRefused("api", f"reload refused: the API changed: {names}", changes)


def changed_symbols(old, new) -> List[str]:
    """The entry symbols whose own coroutine's code differs between ``old``
    and ``new`` (a body-only edit; callees are not followed)."""
    out = []
    for sym, e in sorted((new.entries or {}).items()):
        o = (old.entries or {}).get(sym)
        if o is None or old.coros[o.coro].code != new.coros[e.coro].code:
            out.append(sym)
    return out


class _Gen:
    """One loaded image: its model, its VM and its number."""

    def __init__(self, number: int, model, vm):
        self.number = number
        self.model = model
        self.vm = vm
        self.calls: List[Any] = []     # HostCalls started in it, not yet over

    def live(self) -> List[Any]:
        self.calls = [c for c in self.calls if not c.over]
        return self.calls


class Generations:
    """The engine's images (spec steps 2, 3 and 5): generation N, which new
    calls bind to, and at most one staged N+1.

    ``make_vm(model)`` builds a VM for an image (default ``VM(model)``); the
    holder constructs its component tree once; N+1's is constructed
    bus-isolated and takes N's state per unchanged instance at commit
    (spec step 4, :func:`migrate`; ``strict`` refuses a structural change). :meth:`call` starts a :class:`~.host.HostCall` by
    ENTRY symbol; :meth:`reload` checks and stages an image, which applies at
    the next reload point: no call of N live (checked at :meth:`reload`,
    :meth:`call` and :meth:`at_test_boundary`). Each reload's result dict is
    appended to :attr:`history`::

        g = Generations(model_n)
        call = g.call("dma_c::start", 3)
        r = g.reload(model_n1)          # {"applied": False, ...}: call is live
        run_sync(call); call.end()
        g.call("smoke_t")               # applies N+1 first, runs there
    """

    def __init__(self, model, make_vm: Optional[Callable] = None, seed: int = 0,
                 strict: bool = False):
        self.make_vm = make_vm or _default_vm
        self.seed = seed
        #: refuse a reload that changes the component tree (reason ``structure``)
        self.strict = strict
        self.cur = self._load(model, 1)
        self.pending: Optional[_Gen] = None
        self._result: Optional[dict] = None   # the pending reload's result
        self.history: List[dict] = []

    # -- state ------------------------------------------------------------ #

    @property
    def generation(self) -> int:
        """The current generation number (1 for the first image)."""
        return self.cur.number

    @property
    def model(self):
        return self.cur.model

    @property
    def vm(self):
        return self.cur.vm

    def live(self) -> List[Any]:
        """The calls of the current generation not yet over."""
        return self.cur.live()

    # -- calls ------------------------------------------------------------ #

    def call(self, symbol: str, *args, comp: int = 0, **kw):
        """A :class:`~.host.HostCall` of entry ``symbol`` in the current
        generation, after applying a staged reload if no call is live. A
        function runs on the tree as it stands, in component instance
        ``comp`` with ``args``; an action runs as ``VM.run`` does, on a tree
        constructed for it. ``kw`` goes to ``HostCall``. The caller runs and
        ends it."""
        from .host import HostCall, HostStateError
        self._apply_if_quiet()
        g = self.cur
        e = (g.model.entries or {}).get(symbol)
        if e is None:
            raise KeyError(f"no entry {symbol!r} in generation {g.number}")
        if g.live():
            raise HostStateError(f"a call is live in generation {g.number} "
                                 f"(one call per VM at a time)")
        vm = g.vm
        if e.kind == "action":
            if args:
                raise TypeError(f"action {symbol} takes no arguments")
            call = HostCall(vm, vm.root_frame(e.coro, self.seed), seed=self.seed, **kw)
        else:
            if len(args) != len(e.params):
                raise TypeError(f"{symbol} takes {len(e.params)} arguments, "
                                f"not {len(args)}")
            frame = vm.sched.new_frame(g.model.coros[e.coro], seed=self.seed,
                                       obj=vm._new_obj(e.coro))
            frame.cobj = getattr(vm, "cobj", None)
            vm.set_comp(frame, comp)
            frame.args = list(args)
            call = HostCall(vm, frame, seed=self.seed, construct=False, **kw)
        g.calls.append(call)
        return call

    # -- reload ----------------------------------------------------------- #

    def reload(self, model, cancel: bool = False) -> dict:
        """Check ``model`` against N (:func:`check_api`; a refusal raises
        :class:`ReloadRefused` and leaves N and any staged image as they
        were), load it beside N as N+1 (replacing an image staged before),
        and apply it now if no call of N is live. ``cancel`` ends N's live
        calls first. Returns the reload's result dict; ``applied`` says
        whether N+1 is current yet."""
        result = check_reload(self.cur.model, model, self.strict,
                              self.cur.number)
        g = self._load(model, self.cur.number + 1, fenced=True)
        self.pending = g
        self._result = result
        if cancel:
            live = self.cur.live()
            for c in live:
                c.end()
            self._result["cancelled"] = len(live)
        result = self._result
        self._apply_if_quiet()
        return result

    def at_test_boundary(self) -> Optional[dict]:
        """A reload point between two tests: apply a staged reload. Returns
        its result, or None with nothing staged; a call still live is an
        error (a test boundary has none)."""
        from .host import HostStateError
        if self.pending is None:
            return None
        if self.cur.live():
            raise HostStateError("a call is live at a test boundary")
        result = self._result
        self._apply_if_quiet()
        return result

    # -- internals -------------------------------------------------------- #

    def _load(self, model, number: int, fenced: bool = False) -> _Gen:
        """A VM for ``model`` with its component tree constructed. The
        constructing coroutine takes no arguments in pyvm, so there are no
        root arguments to record. ``fenced`` (N+1): construction runs
        bus-isolated, and a memory access or user import refuses the reload
        (reason ``bus``)."""
        vm = self.make_vm(model)
        if getattr(model, "components", None) is not None:
            # a cobj for the tree, then construct it (no host: it must not yield)
            vm.root_frame(model.entry_coro, self.seed)
            if fenced:
                construct_isolated(vm, number)
            else:
                vm.construct_components()
        return _Gen(number, model, vm)

    def _apply_if_quiet(self) -> None:
        if self.pending is None or self.cur.live():
            return
        self._result["migrated"] = migrate(self.cur.vm, self.pending.vm)
        self.cur, self.pending = self.pending, None
        self._result["applied"] = True
        self.history.append(self._result)
        self._result = None


# -- migration (spec step 4) ------------------------------------------------ #

def _instances(model) -> Dict[str, tuple]:
    """Each instance of ``model``'s component tree by path: (type, its field
    names, its slot range in the component object). Empty without a tree."""
    comps = getattr(model, "components", None)
    if comps is None:
        return {}
    n = len(comps.names)
    out = {}
    for path, typ, base in zip(comps.paths, comps.types, comps.bases):
        # an instance's slots run to the next instance's base; one that shares
        # its base with a sub-instance has none
        if any(b == base and p.startswith(path + ".")
               for p, b in zip(comps.paths, comps.bases)):
            end = base
        else:
            end = min((b for b in comps.bases if b > base), default=n)
        out[path] = (typ, tuple(comps.names[base:end]), (base, end))
    return out


def native_inst_lines(model) -> List[str]:
    """``model``'s instances as the C engine's ``pssc.native`` META reads
    them, in the tree's order: ``inst <path> <type> <base> <end> <sig>``
    (tab-separated; ``sig`` a CRC-32 of the field names, hex), so that
    ``pssvm_reload`` can migrate by path with :func:`migrate`'s rules."""
    import zlib
    return [f"inst\t{path}\t{typ}\t{s0}\t{s1}\t"
            f"{zlib.crc32(','.join(names).encode()):08x}"
            for path, (typ, names, (s0, s1)) in _instances(model).items()]


def structure_diff(old, new) -> Dict[str, List[str]]:
    """The component tree's structural change from ``old`` to ``new``:
    instance paths ``added``, ``removed``, and ``changed`` (same path, another
    type or another field list: a channel or field added or removed). Each
    list is sorted; all empty when the trees match."""
    a, b = _instances(old), _instances(new)
    return {"added": sorted(set(b) - set(a)),
            "removed": sorted(set(a) - set(b)),
            "changed": sorted(p for p in set(a) & set(b) if a[p][:2] != b[p][:2])}


def _diff_text(diff) -> str:
    return ", ".join(f"{p} ({what})" for what, ps in diff.items() for p in ps)


def check_reload(old, new, strict: bool = False, number: int = 1) -> dict:
    """The checks of a reload from image ``old`` (generation ``number``) to
    ``new`` that need no VM: :func:`check_api`, then (``strict``) no
    structural change. Raises :class:`ReloadRefused`; else returns the
    reload's result dict, not yet applied."""
    check_api(old, new)
    diff = structure_diff(old, new)
    if strict and any(diff.values()):
        raise ReloadRefused("structure", "reload refused: the component tree "
                            f"changed: {_diff_text(diff)}")
    return {"old": number, "new": number + 1,
            "changed": changed_symbols(old, new),
            "structure": diff, "migrated": [],
            "cancelled": 0, "applied": False}


def construct_isolated(vm, number: int = 2) -> None:
    """Construct ``vm``'s component tree (its ``cobj`` made beforehand)
    bus-isolated: a memory access or user import refuses the reload of
    generation ``number`` (reason ``bus``). The VM's host is restored after."""
    host, vm.host = vm.host, _BusFence()
    try:
        vm.construct_components()
    except _BusTouched as e:
        raise ReloadRefused("bus", f"reload refused: construction under "
                            f"generation {number} {e}") from None
    finally:
        vm.host = host


def migrate(old_vm, new_vm) -> List[str]:
    """Copy N's component state into N+1's freshly constructed tree, by
    instance path: each instance unchanged in type and fields (channel
    ``wake``/``inflight`` tokens are fields too) takes N's values; the rest
    keep their fresh state. Returns the migrated paths, sorted."""
    src, dst = getattr(old_vm, "cobj", None), getattr(new_vm, "cobj", None)
    if src is None or dst is None:
        return []
    a, b = _instances(old_vm.model), _instances(new_vm.model)
    done = []
    for path in sorted(set(a) & set(b)):
        if a[path][:2] != b[path][:2]:
            continue
        (s0, s1), (d0, _) = a[path][2], b[path][2]
        dst.values[d0:d0 + s1 - s0] = src.values[s0:s1]
        done.append(path)
    return done


class _BusTouched(Exception):
    pass


class _BusFence:
    """The host of a bus-isolated construction: any access or user import
    stops it."""

    def access(self, frame, ins, kind, addr, size, data=None):
        raise _BusTouched(f"made a memory {'read' if kind == 'r' else 'write'} "
                          f"at 0x{addr:x}")

    def import_(self, frame, ins, fn_id, args, blocking):
        raise _BusTouched(f"called import {fn_id}")

    def flush(self):
        return False

    def live(self):
        return False


def _default_vm(model):
    from .vm import VM
    return VM(model)
