"""Object handles of the host ABI (``docs/spec/host-abi.md`` "Objects";
pssc pssvm-plan step 6).

A host names what it calls by handle, not by path: ``new`` constructs the
root component (its tree) and gives its handle, ``sub`` gives an instance's
under it, and ``new`` under a component makes an action object, which a
traversal runs on and whose fields stay readable after it. ``call`` is keyed
by handle. Handles are ints from 1, never reused; ``release`` frees one
explicitly (no reference counts), and a released handle is refused by name
wherever it is used. Inside, an object is still a slot range: an instance is
``(tree, base)``, an action object its own ``Obj``.

:class:`Handles` is ``CallTable.handles``; :func:`reachable_uses` is what
``CallTable.link_check`` checks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Set

from ..model import BUILTIN_BASE, BUILTIN_READ, BUILTIN_WRITE, Op
from .ops_proc import VMError

#: handle kinds
ROOT, COMP, ACTION = "root", "comp", "action"


@dataclass
class _Obj:
    kind: str
    type: str
    #: the component instance (ACTION: the instance it runs from)
    comp: int = 0
    path: str = ""
    #: ACTION: its entry's coroutine, its object, and its last call
    coro: int = 0
    obj: object = None
    call: object = None
    #: ACTION: field overrides (:meth:`Handles.pin`), slot -> (value, (width,
    #: signed)), pinned in every solve of its traversals
    pins: dict = None

    def describe(self) -> str:
        if self.kind == ACTION:
            return f"action object {self.type}"
        return f"component {self.path or '<root>'}: {self.type}"


class Handles:
    """The handle table of one engine (``CallTable.handles``)."""

    def __init__(self, table):
        self.t = table
        self.tab: Dict[int, _Obj] = {}
        #: released handles -> what they were, for the refusal's message
        self.dead: Dict[int, str] = {}
        self.next = 1
        #: the root's handle (0: no tree)
        self.root = 0
        #: a traversal ran since the tree was built: the first runs the
        #: tree's ``exec run_start`` blocks (F2.5)
        self.run_started = False

    def _err(self, msg: str):
        from .host import HostStateError
        return HostStateError(msg)

    def get(self, h: int, *kinds: str) -> _Obj:
        """Handle ``h``'s object, of one of ``kinds`` (none: any)."""
        o = self.tab.get(h)
        if o is None:
            if h in self.dead:
                raise self._err(f"handle {h} ({self.dead[h]}) was released")
            raise self._err(f"no handle {h}")
        if kinds and o.kind not in kinds:
            raise self._err(f"handle {h} is a {o.describe()}, not "
                            f"{' or '.join(kinds)}")
        return o

    def _add(self, o: _Obj) -> int:
        h, self.next = self.next, self.next + 1
        self.tab[h] = o
        return h

    # -- new ----------------------------------------------------------- #

    def new(self, type_: str, parent: int = 0, args=(), seed: int = 0) -> int:
        if parent:
            return self._new_action(type_, self.get(parent, ROOT, COMP), args)
        return self._new_root(type_, args, seed)

    def _new_root(self, type_: str, args, seed: int) -> int:
        from .host import DONE, ERROR, HostCall
        t = self.t
        vm = t.vm
        model = vm.model
        comps = getattr(model, "components", None)
        if comps is None:
            raise self._err("the image has no component tree")
        own = comps.types[0] if comps.types else comps.root
        roots = [r["type"] for r in getattr(model, "roots", None) or ()] or [own]
        if type_ != own:
            if type_ in roots:
                raise self._err(
                    f"root candidate {type_} is an image of its own (ROOTIMG): "
                    f"load model.root_model({type_!r}) for it")
            raise self._err(f"no root candidate '{type_}' (the image has "
                            f"{', '.join(roots)})")
        if self.root:
            raise self._err(f"the tree of handle {self.root} is live: release "
                            f"it first (one root per engine)")
        if args:
            raise self._err(f"{type_} takes no constructor arguments on bc yet")
        vm.cobj = comps.new_obj()
        vm._seed = seed
        t.fatal_exc = None          # a new tree: the VM goes on
        t.owner = None
        self.run_started = False
        if comps.init_coro is not None:
            frame = t.home.new_frame(model.coros[comps.init_coro], seed=seed)
            frame.cobj = vm.cobj
            vm.set_comp(frame, 0)
            c = HostCall(vm, frame, seed=seed, construct=False)
            c._constructing = True
            c.escalate = True         # a construction that fails is fatal
            try:
                r = c.run()
                if r.status not in (DONE, ERROR):
                    r = c.abort(VMError(f"construct-yield: construction "
                                        f"waits ({r.status_name} {r.on})"))
                if r.status == ERROR:
                    raise r.exc
            except BaseException:
                vm.cobj = None        # no tree, and no handle
                raise
            finally:
                if not c.over:
                    c.end()
        self.root = self._add(_Obj(ROOT, own))
        return self.root

    def _new_action(self, type_: str, parent: _Obj, args) -> int:
        vm = self.t.vm
        model = vm.model
        entries = getattr(model, "entries", None) or {}
        e = entries.get(type_)
        if e is None:
            e = next((x for x in entries.values()
                      if x.kind == "action" and x.qname == type_), None)
        if e is None or e.kind != "action":
            raise self._err(f"no action entry '{type_}' (an action is called "
                            f"through its export or entry)")
        if args:
            raise self._err(f"{type_}: an exported action's arguments go to "
                            f"call, not new")
        from .extern import Obj
        table = getattr(model, "activations", {}).get(e.coro)
        obj = (Obj(field_names=table.names) if table is not None
               else vm._new_obj(e.coro))
        o = _Obj(ACTION, e.qname or type_, comp=parent.comp, path=parent.path,
                 coro=e.coro, obj=obj, pins={})
        if parent.comp != 0 or parent.kind == COMP:
            self._place(o, e, table)
        return self._add(o)

    def _place(self, o: _Obj, e, table) -> None:
        """Run action object ``o`` in the instance it was made under (the
        ``comp=`` override, pssvm-in-practice 3.0.7): one of its root
        node's candidates, pinned in the cone when the solve chooses among
        several. Refused, naming the candidates, when the instance is not
        one."""
        comps = self.t.vm.model.components
        cands = (list(table.tree.nodes[0].comp) if table is not None
                 else [i for i, t in enumerate(comps.types)
                       if t == e.comp_type])
        if o.comp not in cands:
            names = ", ".join(comps.paths[i] or "<root>" for i in cands)
            raise self._err(
                f"{o.type} does not run in {o.path or '<root>'} "
                f"({comps.types[o.comp]}): its instances are "
                f"{names or 'none'}")
        slot = table.tree.nodes[0].comp_slot if table is not None else None
        if slot is not None:
            o.pins[slot] = (o.comp, (32, False))

    # -- field overrides -------------------------------------------------- #

    def pin(self, obj: int, path: str, text: str) -> None:
        """Fix field ``path`` of action object ``obj`` (``sz``, ``x1.sz``)
        to the value ``text`` names, in every solve of its later
        traversals: an override (pssvm-in-practice 3.0.3, 4.7). ``text`` is
        read by the field's type: an integer literal (``64``, ``0x40``,
        ``-1``, ``16'h40``), an enum item (``MEM2PERIPH``), ``true`` or
        ``false``. An unknown field, an out-of-range value or an unknown
        item is refused here, before anything runs."""
        o = self.get(obj, ACTION)
        if o.call is not None and not o.call.over:
            raise self._err(f"handle {obj}: a call runs on it")
        model = self.t.vm.model
        e = next((x for x in (getattr(model, "entries", None) or {}).values()
                  if x.kind == "action" and x.coro == o.coro), None)
        descs = dict(e.obj) if e is not None else {}
        d = descs.get(path)
        if d is None or not overridable(path, d):
            known = [n for n, x in descs.items() if overridable(n, x)]
            raise self._err(
                f"{o.type} has no field '{path}' to override"
                + (f" (fields: {', '.join(known)})" if known else ""))
        names = o.obj.field_names
        if path not in names:
            raise self._err(f"{o.type}: field '{path}' has no slot")
        v = parse_value(text, d, f"{o.type}: override {path}={text}")
        o.pins[names.index(path)] = (v, (int(d.get("width") or 64),
                                        bool(d.get("signed"))))

    def sub(self, parent: int, path: str) -> int:
        p = self.get(parent, ROOT, COMP)
        comps = self.t.vm.model.components
        full = f"{p.path}.{path}" if p.path and path else (path or p.path)
        try:
            i = comps.paths.index(full)
        except ValueError:
            raise self._err(f"no component instance '{full}'") from None
        return self._add(_Obj(COMP, comps.types[i], comp=i, path=full))

    # -- release --------------------------------------------------------- #

    def release(self, h: int) -> None:
        o = self.get(h)
        if o.kind == ROOT:
            live = [c for c in self.t.calls if not c.over]
            if live:
                raise self._err(f"cannot release root handle {h}: calls are "
                                f"live ({', '.join(c.name for c in live)})")
            for k in list(self.tab):
                self._kill(k)
            self.root = 0
            self.t.vm.cobj = None
            return
        if o.kind == ACTION and o.call is not None and not o.call.over:
            raise self._err(f"cannot release handle {h}: a call runs on it")
        self._kill(h)

    def _kill(self, h: int) -> None:
        self.dead[h] = self.tab.pop(h).describe()

    # -- calls ------------------------------------------------------------ #

    def call(self, obj: int, method: str = "", args=(), seed: int = 0):
        from .host import HostCall
        o = self.get(obj)
        vm = self.t.vm
        model = vm.model
        if o.kind == ACTION:
            if method and method != o.type:
                raise self._err(f"handle {obj} is an {o.describe()}: it is "
                                f"traversed, it has no method '{method}'")
            if o.call is not None and not o.call.over:
                raise self._err(f"handle {obj}: a call already runs on it")
            hooks = "start" if not self.run_started else False
            self.run_started = True
            try:
                frame = vm.root_frame(o.coro, seed, obj=o.obj, args=list(args),
                                      cobj=vm.cobj, hooks=hooks, pins=o.pins,
                                      comp=o.comp)
            except TypeError as e:
                raise self._err(str(e)) from None
            c = HostCall(vm, frame, seed=seed, construct=False)
            o.call = c
            return c
        comps = model.components
        if method == "$run_end":
            if o.kind != ROOT or comps.run_end_coro is None:
                raise self._err("$run_end is called on the root of a tree "
                                "with exec run_end blocks")
            self.run_started = False
            coro, params = comps.run_end_coro, []
            comp = o.comp
        else:
            e = self._function(o, method)
            coro, params = e.coro, e.params
            # a package function runs in the root, where its `pss_top` is
            comp = o.comp if e.comp_type is not None else 0
        if len(args) != len(params):
            raise self._err(f"{o.type}::{method} takes {len(params)} "
                            f"arguments, not {len(args)}")
        frame = self.t.home.new_frame(model.coros[coro], seed=seed,
                                      obj=vm._new_obj(coro))
        frame.cobj = vm.cobj
        vm.set_comp(frame, comp)
        frame.args = list(args)
        return HostCall(vm, frame, seed=seed, construct=False)

    def _function(self, o: _Obj, method: str):
        """The entry of function ``method`` of component ``o`` (by symbol
        or name), or of a package-scope function by its qualified name
        (which any component handle calls; it runs in the root, where its
        ``pss_top`` is)."""
        entries = getattr(self.t.vm.model, "entries", None) or {}
        for sym, e in entries.items():
            if (e.kind == "function" and e.comp_type == o.type
                    and (sym == method or sym.rpartition("::")[2] == method)):
                return e
        e = entries.get(method)
        if e is not None and e.kind == "function" and e.comp_type is None:
            return e
        raise self._err(f"{o.type} has no function entry '{method}'")

    def field(self, obj: int, name: str) -> int:
        o = self.get(obj, ACTION)
        if o.call is None:
            raise self._err(f"handle {obj}: no call has run on it")
        if not o.call.over:
            raise self._err(f"handle {obj}: its call is still running")
        if not o.obj.has_field(name):
            raise self._err(f"{o.type} has no field '{name}'")
        return o.obj.get_field_name(name)


def reachable_uses(model) -> Dict[int, Set[str]]:
    """API id -> the imports (``mem``: ``read``/``write``) that the
    coroutines reachable from an entry, the image's entry, construction or
    the run hooks call; through CALL, SPAWN, INVOKE and SELECT branches."""
    from .host import API_MEM, API_USER
    coros = model.coros
    seen: Set[int] = set()
    todo = [e.coro for e in (getattr(model, "entries", None) or {}).values()]
    todo.append(getattr(model, "entry_coro", 0))
    comps = getattr(model, "components", None)
    if comps is not None and comps.init_coro is not None:
        todo += [c for c in (comps.init_coro, comps.run_start_coro,
                             comps.run_end_coro) if c is not None]
    sigs = {sig.get("fn_id"): (name, sig.get("api", API_USER))
            for name, sig in (getattr(model, "imports", None) or {}).items()}
    reads, writes = set(BUILTIN_READ.values()), set(BUILTIN_WRITE.values())
    selects = getattr(model, "selects", None) or []
    out: Dict[int, Set[str]] = {}
    while todo:
        k = todo.pop()
        if k in seen or not 0 <= k < len(coros):
            continue
        seen.add(k)
        for ins in coros[k].code:
            op = ins.op
            if op in (Op.CALL, Op.SPAWN, Op.INVOKE):
                todo.append(ins.args[0])
            elif op == Op.SELECT and ins.args[0] < len(selects):
                todo += list(selects[ins.args[0]].branches)
            elif op == Op.IMPORT:
                fn = ins.args[0]
                if fn in reads or fn in writes:
                    out.setdefault(API_MEM, set()).add(
                        "read" if fn in reads else "write")
                elif fn < BUILTIN_BASE:
                    name, api = sigs.get(fn, (f"fn {fn}", API_USER))
                    out.setdefault(api, set()).add(name)
    return out


def overridable(path: str, d: dict) -> bool:
    """Can field ``path`` (descriptor ``d``) take an override? A scalar
    the model declares: an integer, ``bool`` or enum leaf; not a handle,
    a sub-action's own slot or an engine-internal one (``$`` in its
    name)."""
    return "$" not in path and d.get("kind") in ("int", "bool", "enum")


def type_text(d: dict) -> str:
    """A field descriptor as PSS writes its type: ``bit[16]``,
    ``int[8]``, ``bool``, ``enum xfer_mode_e`` (or the enum's items when
    the image does not name it)."""
    k = d.get("kind")
    if k == "bool":
        return "bool"
    if k == "enum":
        if d.get("enum"):
            return f"enum {d['enum']}"
        return "enum {" + ", ".join(n for n, _ in d.get("items") or ()) + "}"
    if k == "int":
        return f"{'int' if d.get('signed') else 'bit'}[{d.get('width') or 64}]"
    return k or "opaque"


_SIZED = re.compile(r"^(-?)(\d*)'([sS]?)([hHdDbBoO])([0-9a-fA-F_xXzZ]+)$")
_BASES = {"h": 16, "d": 10, "b": 2, "o": 8}


def parse_value(text: str, d: dict, what: str) -> int:
    """The value ``text`` names for a field of descriptor ``d`` (see
    :meth:`Handles.pin`), as the field holds it (signed fields negative).
    :class:`HostStateError` naming ``what`` when it is not one."""
    from .host import HostStateError
    t = (text or "").strip()
    k = d.get("kind")
    if k == "enum":
        items = d.get("items") or ()
        name = t.rsplit("::", 1)[-1]
        for n, v in items:
            if n == name:
                return int(v)
        try:
            v = _int(t)
        except ValueError:
            v = None
        if v is not None and any(int(x) == v for _, x in items):
            return v
        raise HostStateError(f"{what}: not an item of "
                             f"{type_text(d)} ({', '.join(n for n, _ in items)})")
    if k == "bool":
        b = {"true": 1, "false": 0, "1": 1, "0": 0}.get(t.lower())
        if b is None:
            raise HostStateError(f"{what}: not a bool (true or false)")
        return b
    try:
        v = _int(t)
    except ValueError:
        raise HostStateError(f"{what}: not an integer") from None
    w, signed = int(d.get("width") or 64), bool(d.get("signed"))
    lo, hi = (-(1 << (w - 1)), (1 << (w - 1)) - 1) if signed else (0, (1 << w) - 1)
    if not lo <= v <= hi:
        raise HostStateError(f"{what}: out of range for {type_text(d)} "
                             f"({lo}..{hi})")
    return v


def _int(t: str) -> int:
    """An integer literal: Python's forms (``64``, ``0x40``, ``0b1``,
    ``1_000``, a leading ``-``) or a sized PSS/SV one (``16'h40``)."""
    m = _SIZED.match(t)
    if m:
        neg, _, _, base, digits = m.groups()
        v = int(digits.replace("_", ""), _BASES[base.lower()])
        return -v if neg else v
    return int(t, 0)
