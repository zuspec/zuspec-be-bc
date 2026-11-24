"""A component tree constructed once, whose functions a host calls one after
another (``pssvm``): the session a simulation's Python host (``share/py``)
and a tool's test runner build on. A tool subclasses :class:`Session` to add
its own platform, ledger or engines."""

from __future__ import annotations

from typing import Callable, Optional, Sequence


class Instance:
    """A component instance of a :class:`Session`: its id and path in the
    tree (the root's path is ``""``) and its type."""

    def __init__(self, id: int, path: str, type_qname: str):
        self.id, self.path, self.type_qname = id, path, type_qname

    def __repr__(self) -> str:
        return f"Instance({self.path or '<root>'}: {self.type_qname})"


def drain(vm, frame) -> None:
    """Run ``frame`` and every thread it starts until nothing can run, as
    ``VM._drain`` does, over the host ABI: one ``HostCall`` on the tree as it
    stands, served by ``run_sync`` from the VM's own memory and imports. A
    frame left blocked returns normally; the caller checks ``frame.done``."""
    from zuspec.be.bc.interp.host import HostCall, run_sync

    call = HostCall(vm, frame, construct=False)
    try:
        run_sync(call)
    finally:
        call.end()


class Session:
    """One component tree, constructed once, whose functions a host calls
    one after another: state a call leaves in the tree is there for the
    next::

        s = Session(model)
        root = s.construct()
        s.call(root, "ch[0].transfer_single", 1, 2)

    Construction runs the model's ``$comp_init`` coroutine, as a run does
    before its entry. A function is called through its entry (a function
    ENTRY record), in the instance it is called on.
    """

    #: what the session's accesses reach, for a host that keeps it (a
    #: register fake whose values follow the simulation); None: none
    platform = None

    def __init__(self, model, seed: int = 0,
                 out: Optional[Callable[[str], None]] = None,
                 events=None, watch: int = 0, memory=None, imports=None,
                 platform=None):
        self.model, self.seed, self.out = model, seed, out
        if platform is not None:
            self.platform = platform
        #: the platform's seams: what memory builtins reach (``read(addr,
        #: n)`` / ``write(addr, n, v)``) and what user imports call
        #: (``call(fn_id, args)``); None: be-bc's defaults
        self.memory, self.imports = memory, imports
        #: the debug events sink and mask (``zuspec.be.bc.trace.events``)
        self.events, self.watch = events, watch
        self.vm = None
        self._cache = None
        #: the image's generation: 1, plus one per applied :meth:`reload`
        self.generation = 1
        #: each applied reload's result dict, oldest first
        self.history = []
        self._depth = 0           # calls running (a reload in one is staged)
        self._pending = None      # (model, vm, cache, result) staged

    @property
    def comp_obj(self):
        """The component object (every instance's fields, by path)."""
        return getattr(self.vm, "cobj", None)

    def construct(self, root_type: Optional[str] = None,
                  args: Sequence[int] = ()) -> Instance:
        """(Re)build the component tree; returns its root. ``root_type``, if
        given, must be the model's root type: the tree is fixed at lowering.
        PSS components take no constructor arguments, so ``args`` must be
        empty."""
        comps = getattr(self.model, "components", None)
        if comps is None:
            raise ValueError("the model has no component tree")
        if root_type is not None and root_type != comps.types[0]:
            raise ValueError(f"the model's root component is {comps.types[0]!r}, "
                             f"not {root_type!r}")
        if args:
            raise ValueError("components take no constructor arguments")
        self._new_vm()
        self.vm.cobj = comps.new_obj()
        self.vm._seed = self.seed
        self.vm.construct_components()
        return self.instance("")

    def _new_vm(self) -> None:
        """A fresh VM on the platform, with no component tree yet."""
        self.close()
        self.vm, self._cache = self._make_vm(self.model)

    def _make_vm(self, model):
        """A VM for ``model`` on the platform, and its solve cache."""
        from zuspec.be.bc.interp import NativeBlobBackend
        from zuspec.be.bc.interp.solve_cache import SolveCache
        from zuspec.be.bc.interp.vm import VM
        from zuspec.be.bc.trace import MemorySink

        cache = SolveCache()
        vm = VM(model, solve_backend=NativeBlobBackend(),
                sink=MemorySink(), out=self.out, solve_cache=cache,
                events=self.events, memory=self.memory,
                import_provider=self.imports)
        if self.watch:
            vm.watch(self.watch)
        self._setup_vm(vm)
        return vm, cache

    def _setup_vm(self, vm) -> None:
        """Called on each new VM: a tool adds its event hooks here."""

    def close(self) -> None:
        if self._cache is not None:
            self._cache.clear()
        self.vm = self._cache = None

    def instance(self, path: str, under: Optional[Instance] = None) -> Instance:
        """The instance at ``path`` (``ch[0]``, ``a.sub``), from ``under``
        (default: the root)."""
        comps = self.model.components
        if under is not None and under.path:
            path = f"{under.path}.{path}" if path else under.path
        try:
            i = comps.paths.index(path)
        except ValueError:
            raise KeyError(f"no component instance {path!r}") from None
        return Instance(i, path, comps.types[i])

    def entry(self, inst: Instance, fn: str):
        """The entry of function ``fn`` for ``inst``'s type."""
        for sym, e in self.model.entries.items():
            if (e.kind == "function" and e.comp_type == inst.type_qname
                    and sym.rpartition("::")[2] == fn):
                return e
        raise KeyError(f"{inst.type_qname}::{fn} has no entry: lower with "
                       f"functions=[{inst.type_qname + '::' + fn!r}] or ['*']")

    def call(self, inst: Instance, target: str, *args: int):
        """Call function ``target`` (``f``, or ``ch[0].f`` below ``inst``) and
        return its value (None for a void function). A struct parameter
        takes one argument per field, in declaration order (the entry's
        ``params`` name them: ``cfg.src``, ``cfg.dst``, ...)."""
        from zuspec.be.bc.interp import VMError

        if self.vm is None:
            raise RuntimeError("construct() the component tree first")
        path, _, fn = target.rpartition(".")
        if path:
            inst = self.instance(path, under=inst)
        e = self.entry(inst, fn)
        if len(args) != len(e.params):
            raise TypeError(f"{target} takes {len(e.params)} arguments, "
                            f"not {len(args)}")
        vm = self.vm
        frame = vm.sched.new_frame(self.model.coros[e.coro], seed=self.seed,
                                   obj=vm._new_obj(e.coro))
        frame.cobj = vm.cobj
        vm.set_comp(frame, inst.id)
        frame.args = list(args)
        self._depth += 1
        try:
            drain(vm, frame)
        finally:
            self._depth -= 1
            if not self._depth and self._pending is not None:
                self._apply_reload()
        if not frame.done:
            raise VMError(f"{target} did not finish: every thread left is "
                          f"blocked")
        return frame.retval

    def reload(self, model, cancel: bool = False, strict: bool = False) -> dict:
        """Hot reload (``interp/generations.py``): make ``model``, a new
        image of the same sources, the session's generation N+1.

        The API check comes first, then (``strict``) no change to the
        component tree; a refusal raises ``ReloadRefused`` (its
        ``to_json()`` is the error) and leaves the session as it was. With
        a tree constructed, N+1's is constructed beside it bus-isolated (a
        memory access or import there refuses, reason ``bus``) and, when
        applied, takes N's state, channel tokens included, per unchanged
        instance path; bindings carry over where the function is still
        called (``unbound`` lists the rest). A reload from inside a call (a
        binding, or a debugger stopped in one) is staged: the call finishes
        on N and N+1 applies when it returns; ``cancel`` is refused there,
        since a synchronous call cannot be ended early. Returns the result
        dict (``applied`` says whether N+1 is current yet); the applied
        result is raised as a ``reload`` event on N+1's VM."""
        from zuspec.be.bc.interp.generations import (check_reload,
                                                     construct_isolated)

        if cancel and self._depth:
            raise RuntimeError("cannot cancel the call running: reload once "
                               "it returns, or without cancel")
        result = check_reload(self.model, model, strict, self.generation)
        result["unbound"] = []
        vm = cache = None
        if self.vm is not None and getattr(self.vm, "cobj", None) is not None:
            vm, cache = self._make_vm(model)
            comps = getattr(model, "components", None)
            if comps is not None:
                vm.cobj = comps.new_obj()
                vm._seed = self.seed
                try:
                    construct_isolated(vm, result["new"])
                except BaseException:
                    cache.clear()
                    raise
        if self._pending is not None and self._pending[2] is not None:
            self._pending[2].clear()
        self._pending = (model, vm, cache, result)
        if not self._depth:
            self._apply_reload()
        return result

    def _apply_reload(self) -> None:
        """Make the staged generation current (:meth:`reload`)."""
        from zuspec.be.bc.interp.generations import migrate

        model, vm, cache, result = self._pending
        self._pending = None
        old = self.vm
        if vm is not None:
            result["migrated"] = migrate(old, vm)
            seen = {}
            for qname, fn in old.bindings.values():
                seen.setdefault(qname, fn)
            for qname, fn in seen.items():
                try:
                    vm.bind(qname, fn)
                except KeyError:
                    result["unbound"].append(qname)
            self.close()
            self.vm, self._cache = vm, cache
        else:
            self.close()
        self.model = model
        self.generation = result["new"]
        add = getattr(self.events, "add_model", None)
        if add is not None:             # a Coverage sink: N+1's points
            add(model)
        result["applied"] = True
        self.history.append(result)
        from zuspec.be.bc.trace import events as ev
        if self.events is not None and self.watch & ev.EV_RELOAD:
            self.events.event(ev.RELOAD, src_ref=0, detail=result)

    def bind(self, qname: str, fn) -> None:
        """Run host callable ``fn`` in place of function ``qname`` until the
        tree is next constructed (``VM.bind``)."""
        if self.vm is None:
            raise ValueError("construct() the tree before binding")
        self.vm.bind(qname, fn)
