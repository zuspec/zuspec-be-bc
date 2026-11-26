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
        if parent.comp != 0:
            raise self._err(f"an action object under {parent.describe()}: a "
                            f"traversal runs from the root for now")
        if args:
            raise self._err(f"{type_}: an exported action's arguments go to "
                            f"call, not new")
        from .extern import Obj
        table = getattr(model, "activations", {}).get(e.coro)
        obj = (Obj(field_names=table.names) if table is not None
               else vm._new_obj(e.coro))
        return self._add(_Obj(ACTION, e.qname or type_, comp=parent.comp,
                              path=parent.path, coro=e.coro, obj=obj))

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
                                      cobj=vm.cobj, hooks=hooks)
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
        else:
            e = self._function(o, method)
            coro, params = e.coro, e.params
        if len(args) != len(params):
            raise self._err(f"{o.type}::{method} takes {len(params)} "
                            f"arguments, not {len(args)}")
        frame = self.t.home.new_frame(model.coros[coro], seed=seed,
                                      obj=vm._new_obj(coro))
        frame.cobj = vm.cobj
        vm.set_comp(frame, o.comp)
        frame.args = list(args)
        return HostCall(vm, frame, seed=seed, construct=False)

    def _function(self, o: _Obj, method: str):
        entries = getattr(self.t.vm.model, "entries", None) or {}
        for sym, e in entries.items():
            if (e.kind == "function" and e.comp_type == o.type
                    and (sym == method or sym.rpartition("::")[2] == method)):
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
