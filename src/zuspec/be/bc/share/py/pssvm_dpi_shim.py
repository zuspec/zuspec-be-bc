"""pssvm_dpi_shim -- pyvm behind ``zsp_rt_bc``, the SV engine class of
``zsp_rt_bc_pkg``'s pyvm variant (pyhdl-if).

The SV side (``zsp_rt_bc_pkg.sv``, in zuspec-be-bc ``share/sv``) holds a
``ZspRtBcPy`` and calls it synchronously: the host ABI (zuspec-be-bc ``docs/spec/
host-abi.md``) with int call handles, the pending request read field by
field and call arguments pushed one at a time (``push_arg``) before
``call_start``. Bus accesses and the model's own imports go back to SV
(``zsp_rt_bc_platform_if``); pssunit_pkg's imports (checks, ``base``) and
imports a ``use`` double implements run here, in the run loop, without
leaving Python.

Put this directory (``zuspec.be.bc.share.py_dir()``) on PYTHONPATH for
the generic engine; it imports only zuspec-be-bc. A tool (pssc) ships its
own ``pssvm_dpi_shim`` that subclasses ``Engine`` -- loading its images,
its session, its daemon -- and builds ``ZspRtBcPy`` with ``make_api``.
``Engine`` is the same API with no pyhdl-if, for tests and other hosts.

Values: SV ``longint unsigned`` crosses as a signed 64-bit Python int
(pyhdl-if converts with ``PyLong_AsLong``); ``_u64``/``_s64`` convert.
"""
import ctypes as ct
import json

_M64 = (1 << 64) - 1


def _u64(v) -> int:
    return int(v or 0) & _M64


def _s64(v) -> int:
    v = _u64(v)
    return v - (1 << 64) if v >> 63 else v


#: the hook lists of an image, as ``Engine.hook`` names them
HOOK_KINDS = ("session", "setup", "teardown")


def attribution_line(r: dict) -> str:
    """One line for the simulation log naming the action a failing bench
    report (``VM.bench_report``'s record) is blamed on (F2.10c)."""
    types = {o["stim"]: o["type"].split("::")[-1] for o in r.get("owners") or []}
    types.update({f["stim"]: f["type"] for f in r.get("in_flight") or []})
    stim = r.get("stim")
    head = "PSSVM: %s %s %s" % (r.get("kind"), r.get("severity"), r.get("id"))
    if stim is None:
        return head + ": no PSS action in flight"
    ev = ",".join(r["suspects"][0]["evidence"])
    # M9b1b: an action bc inferred says which action it was inferred for
    served = {f["stim"]: f["inferred_for"] for f in r.get("in_flight") or []
              if f.get("inferred_for")}
    flight = " ".join("pss#%s %s%s" % (f["stim"], f["type"],
                                       " (inferred for pss#%s)" % served[f["stim"]]
                                       if f["stim"] in served else "")
                      for f in r.get("in_flight") or [])
    own = next((o for o in r.get("owners") or [] if o["stim"] == stim), None)
    if own is not None:
        # F7.7a: the region of its claim the reported address is in
        ev += " of %s [0x%x, 0x%x)%s" % (own["space"], own["addr"],
                                         own["addr"] + own["size"],
                                         "" if own["live"] else ", released")
    inf = (", inferred for pss#%s" % served[stim]) if stim in served else ""
    return "%s: action %s pss#%s (%s)%s; in flight: %s" % (
        head, types.get(stim, "?"), stim, ev, inf, flight or "none")


class _Events:
    """A session's events sink as ``trace`` uses it: the events ``watch``
    selects go to ``sink`` (a ``TraceWriter``, or None)."""

    def __init__(self):
        self.sink, self.watch = None, 0

    def event(self, kind: str, coro: int = 0, src_ref: int = 0, detail=None):
        from zuspec.be.bc.trace import events as ev
        if self.sink is not None and ev.BITS.get(kind, 0) & self.watch:
            self.sink.event(kind, coro=coro, src_ref=src_ref, detail=detail)


class _Listener:
    """The engine's listener on each VM it runs (host-abi.md "Listeners"):
    an error is the call's status and ``error(h)``, as before; a fatal
    (the VM cannot go on) is kept, located, for ``fatal()``. Every other
    call has been released by then (its ``run`` gives -1). SV hears it
    there: ``zsp_rt_bc_ctxt`` asks ``fatal()`` when a call ends in ERROR
    and gives it to the SV listeners, or ends the simulation with
    ``$fatal`` when there are none."""

    def __init__(self, engine):
        self.engine = engine

    def on_error(self, ev) -> None:
        pass

    def on_fatal(self, ev) -> None:
        self.engine.fatal_msg = ev.located


class _DoneCall:
    """A host call with nothing to run: DONE, no value."""
    result = None
    over = True

    def run(self):
        from zuspec.be.bc.interp import host as H
        return H.Request(H.DONE)

    def end(self) -> None:
        pass


class Engine:
    """pyvm over the host ABI, by int handles. Every method catches: an
    error is a status (``run`` -> -1, ``call_start`` -> 0) and its JSON is
    ``error(h)`` (h = 0: the engine's own, e.g. a failed ``load``)."""

    def __init__(self):
        self.session = None
        self.root = ""
        self.calls = {}
        self.chan_notes = []   # N_CHAN notifies before any call (notify)
        #: per call: the request SV answers now, and those it is not yet given
        self.last, self.taken, self.queued = {}, {}, {}
        #: (h, rid) -> {i: v}: a struct result's fields SV staged (set_ret)
        self.rets = {}
        self.next_id = 1
        self.args = []
        self.errors = {}
        #: h -> [entry, file, line]: where call h last was (``cur_*``)
        self.where = {}
        self.tests = []
        self.hooks = {"session": [], "setup": [], "teardown": []}
        self.root_type = None
        self.now_fs = 0
        #: the tree's ``exec run_start`` blocks ran (F2.5): a simulation
        #: is one run, they run before its first action and ``$run_end``
        #: runs the ``exec run_end`` blocks once
        self.run_started = False
        #: bench reports with verdict fail so far (``bench_failed``)
        self.n_bench_fail = 0
        #: the fatal the VM stopped on, located ("": none; ``fatal``)
        self.fatal_msg = ""
        self._listener = _Listener(self)
        #: call -> (outer call, req_id): started inside that request
        #: (``call_inside``)
        self.inside = {}

    # -- set-up ------------------------------------------------------------ #

    #: load options pyvm knows (``zsp_rt_bc::load``'s ``opts``)
    OPTIONS = ("root", "platform", "seed")

    def load(self, path: str, opts: str = "") -> int:
        """Load image ``path`` (``_load_image``) and construct its tree
        (``load_model``). ``opts``, comma-separated: ``root=<type>`` (""
        = the image's), ``platform=fake`` (the register fake keeps what
        the bus returned), ``seed=<n>`` (the tree's and the calls'). An
        option pyvm does not know is ignored and reported. Returns 0, or
        -1 (``error(0)``)."""
        try:
            o = {}
            for item in (opts or "").split(","):
                k, _, v = item.strip().partition("=")
                if not k:
                    continue
                if k not in self.OPTIONS:
                    print(f"PSSVM-WARNING: zsp_rt_bc::load: option '{k}' is "
                          f"not known to pyvm: ignored", flush=True)
                    continue
                o[k] = v
            model, tests, hooks, root = self._load_image(path)
            self.load_model(model, o.get("root") or None,
                            o.get("platform", ""), root)
            self.image = path
            self.tests = list(tests)
            self.hooks = {k: list(hooks.get(k) or []) for k in HOOK_KINDS}
            if o.get("seed"):
                return self.set_seed(int(o["seed"], 0))
            return 0
        except Exception as e:
            return self._fail(0, e)

    def _load_image(self, path: str):
        """``(model, tests, hooks, root)`` of image ``path``: its bc model,
        the tests ``test_name``/``test_expect`` list (each with ``qname``
        and ``expect_error``), ``{kind: [comp::fn]}`` for ``HOOK_KINDS``
        and the root component's name. Here: any .zbc, whose function
        ENTRY records are the tests (none expects an error) and which has
        no hooks; a tool's engine overrides it for its own images."""
        import types
        from zuspec.be.bc.run import load
        model = load(path)
        tests = [types.SimpleNamespace(qname=q, expect_error="")
                 for q, e in model.entries.items() if e.kind == "function"]
        comps = getattr(model, "components", None)
        return model, tests, {}, getattr(comps, "root", "") or ""

    def load_model(self, model, root_type=None, platform: str = "",
                   root: str = ""):
        """Set ``self.session`` to a session on ``model`` and construct its
        tree. The engine uses of a session: ``model``, ``vm``, ``seed``,
        ``platform``, ``construct(root_type)``, ``instance(path)``,
        ``entry(inst, name)``, and for ``trace`` ``events`` and ``watch``.
        Here: be-bc's ``Session``; ``platform`` "fake": be-bc's register
        fake keeps the registers SV accesses, else the base platform. A
        tool's engine overrides it."""
        from zuspec.be.bc.platform import base, fake
        from zuspec.be.bc.session import Session
        plat = fake.RegisterFake() if platform == "fake" else base.Platform()
        attach = getattr(plat, "attach", None)
        if attach is not None:
            attach(model)
        self.session = Session(model, platform=plat, events=_Events())
        self.session.construct(root_type)
        self.root_type = root_type
        self.root = root
        self.run_started = False

    def set_seed(self, seed: int) -> int:
        """Make ``seed`` the session's seed (SV ``+pss_seed``) and build the
        tree again with it, as a standalone run with that seed does.
        Returns 0, or -1 (``error(0)``)."""
        try:
            s = self.session
            if s is None:
                raise RuntimeError("no image loaded")
            s.seed = _u64(seed)
            s.construct(self.root_type)
            self.run_started = False
            return 0
        except Exception as e:
            return self._fail(0, e)

    def fresh(self) -> int:
        """Build the component tree again, for the next test (plan P3.8):
        a new VM on the same model (the current generation's), platform and
        doubles. The pssunit check counts start again at 0, and an attached
        debugger follows the new VM. Returns 0, or -1 (``error(0)``)."""
        try:
            s = self.session
            if s is None:
                raise RuntimeError("no image loaded")
            for h in list(self.calls):
                self.call_end(h)
            s.construct(self.root_type)
            self.run_started = False
            dbg = getattr(self, "_dbg", None)
            if dbg is not None:
                dbg[0].rebind(s.vm)
            return 0
        except Exception as e:
            return self._fail(0, e)

    # -- calls ------------------------------------------------------------- #

    def push_arg(self, v: int) -> None:
        self.args.append(_u64(v))

    def call_start(self, entry: str, comp_path: str = "") -> int:
        """Start function ``entry`` (``comp::f`` or plain ``f``) on the
        instance at ``comp_path`` ("" = the root) with the pushed args.
        An action entry (``pss_top::A``), or "" for the image's own entry,
        runs as a root action: on its own object, in the session's tree, as
        the C engine's ``pssvm_call_start`` does; the pushed args are an
        exported action's parameters. A simulation is one PSS run: the
        tree's ``exec run_start`` blocks run before its first action, and
        entry ``$run_end`` runs its ``exec run_end`` blocks (once; a no-op
        call without them)."""
        from zuspec.be.bc.interp.host import HostCall
        args, self.args = self.args, []
        try:
            s = self.session
            if s is None:
                raise RuntimeError("no image loaded")
            if entry == "$run_end":
                return self._started(self._run_end(), entry)
            e = s.model.entries.get(entry)
            if e is None:  # an action by its qualified type
                e = next((x for x in s.model.entries.values()
                          if x.kind == "action" and x.qname == entry), None)
            if entry == "" or (e is not None and e.kind == "action"):
                coro = s.model.entry_coro if entry == "" else e.coro
                vm = s.vm
                # an exported action's args pin the fields it names
                hooks = "start" if not self.run_started else False
                self.run_started = True
                call = HostCall(vm, vm.root_frame(coro, s.seed, args=args,
                                                  cobj=vm.cobj, hooks=hooks),
                                seed=s.seed, construct=False)
                return self._started(call, entry or "<entry>")
            inst = s.instance(comp_path)
            if e is None or e.kind != "function":
                e = s.entry(inst, entry.rpartition("::")[2])
            if len(args) != len(e.params):
                raise TypeError(f"{entry} takes {len(e.params)} arguments, "
                                f"not {len(args)}")
            vm = s.vm
            frame = vm.sched.new_frame(s.model.coros[e.coro], seed=s.seed,
                                       obj=vm._new_obj(e.coro))
            frame.cobj = vm.cobj
            vm.set_comp(frame, inst.id)
            frame.args = list(args)
            call = HostCall(vm, frame, construct=False)
        except Exception as e:
            self._fail(0, e)
            return 0
        return self._started(call, entry)

    def _run_end(self):
        """A call that runs the tree's ``$run_end`` coroutine in the root
        instance, if a run started and the image has one; else one already
        DONE."""
        from zuspec.be.bc.interp.host import HostCall
        s = self.session
        coro = s.model.components.run_end_coro \
            if getattr(s.model, "components", None) is not None else None
        started, self.run_started = self.run_started, False
        if coro is None or not started:
            return _DoneCall()
        vm = s.vm
        frame = vm.sched.new_frame(s.model.coros[coro], seed=s.seed)
        frame.cobj = vm.cobj
        vm.set_comp(frame, 0)
        return HostCall(vm, frame, construct=False)

    def _started(self, call, entry: str) -> int:
        from zuspec.be.bc.interp.host import calls_of
        vm = getattr(self.session, "vm", None)
        if vm is not None:
            t = calls_of(vm)               # a VM per construction: listen once
            if self._listener not in t.listeners:
                t.add_listener(self._listener)
            if t.fatal_exc is None:
                self.fatal_msg = ""
        h, self.next_id = self.next_id, self.next_id + 1
        self.calls[h] = call
        if hasattr(call, "notify"):
            for note in self.chan_notes:
                call.notify(*note)
            self.chan_notes = []
        self.where[h] = [entry, "", 0]
        return h

    def run(self, h: int) -> int:
        """Run call ``h`` until SV is needed: the run status (host ABI 2):
        -1 ERROR, 0 DONE, 1 REQS (``take`` the new requests), 3 WAIT,
        4 STOP. Imports a double mocks are answered here, never shown to SV."""
        from zuspec.be.bc.interp import host as H
        try:
            call = self.calls[h]
            while True:
                r = self._run_inside(h, call)
                if r.status == H.REQS and self._collect(h, call) == 0:
                    continue
                return self._status(h, r)
        except Exception as e:
            return self._fail(h, e)

    def _run_inside(self, h: int, call):
        """``call.run()``; for a call started inside a request SV serves
        (``call_inside``), as the sync loops run one: that request is
        marked served on this thread, so a traversal that would queue
        behind the call making it ends in ERROR (self-wait)."""
        inside = self.inside.pop(h, None)
        if inside is None:
            return call.run()
        from zuspec.be.bc.interp.host import calls_of
        import threading
        outer, rid = inside
        oc, q = self.calls.get(outer), self.taken.get(outer, {}).get(rid)
        if oc is None or q is None or getattr(call, "_thread", 1) is not None:
            return call.run()
        t = calls_of(self.session.vm)
        t.serving.append((oc, q, threading.get_ident()))
        try:
            return call.run()
        finally:
            t.serving.pop()

    def _collect(self, h: int, call) -> int:
        """Take call ``h``'s new requests: those a double mocks are
        answered here, the rest queued for SV's ``take``. The number
        queued."""
        from zuspec.be.bc.interp import host as H
        taken = self.taken.setdefault(h, {})
        q = self.queued.setdefault(h, [])
        n_new = 0
        while (n := call.take()) is not None:
            if n.api != H.API_MEM and self._local(n.fn):
                try:
                    v = self.session.vm.import_provider.call(n.fn, n.args)
                except BaseException as e:
                    call.fail(n.req_id, e)
                    continue
                call.complete(n.req_id, v)
                continue
            taken[n.req_id] = n
            q.append(n.req_id)
            n_new += 1
        self._drive_adopted()
        return n_new

    def _status(self, h: int, r) -> int:
        from zuspec.be.bc.interp import host as H
        self.last[h] = r
        self._at(h, r.src_ref)
        if r.status == H.ERROR:
            self.errors[h] = json.dumps(r.error or {"message": str(r.exc)})
        return r.status

    def _drive_adopted(self) -> None:
        """Run the function calls a double started while serving (an
        interrupt handler: ``ctx.raise_irq``; host-abi.md "Concurrent
        calls") to their ends, answering their mocked imports here. One
        that needs SV (a memory access, an SV import) or waits fails, and
        fails the call that raised it: under SV the testbench calls the
        handler, an exported function, itself."""
        from zuspec.be.bc.interp import host as H
        from zuspec.be.bc.interp.ops_proc import VMError
        t = H.calls_of(self.session.vm)
        why = ("an interrupt handler a double started (ctx.raise_irq) {} "
               "under SV: call the handler from the testbench instead")
        while (adopted := t.take_adopted()):
            for c in adopted:
                while not c.over:
                    r = c.run()
                    if r.status == H.REQS:
                        while (n := c.take()) is not None:
                            if n.api != H.API_MEM and self._local(n.fn):
                                try:
                                    v = self.session.vm.import_provider.call(
                                        n.fn, n.args)
                                except BaseException as e:
                                    c.fail(n.req_id, e)
                                    continue
                                c.complete(n.req_id, v)
                            else:
                                c.fail(n.req_id, VMError(why.format(
                                    "made a request SV serves")))
                    elif r.status == H.WAIT:
                        r = c.abort(VMError(why.format("waits")))
                if c.request.status == H.ERROR:
                    raise c.request.exc

    def take(self, h: int) -> int:
        """The next new request's req_id (issue order); -1 when none."""
        q = self.queued.get(h)
        if not q:
            return -1
        rid = q.pop(0)
        self._at(h, self.taken[h][rid].src_ref)
        return rid

    def _at(self, h: int, src_ref: int) -> None:
        """Call ``h`` stopped at ``src_ref`` (0: no place; keep the last)."""
        if not src_ref:
            return
        try:
            e = self.session.model.prov.entries[src_ref]
        except (AttributeError, IndexError, KeyError, TypeError):
            return
        w = self.where.get(h)
        if w is not None:
            w[1], w[2] = e.file or "", e.line or 0

    def cur_op(self, h: int) -> str:
        """The function call ``h`` runs (as ``call_start`` named it)."""
        return self.where.get(h, ["", "", 0])[0]

    def cur_file(self, h: int) -> str:
        """The source file of the statement call ``h`` last stopped at
        (an access or an import); "" before the first."""
        return self.where.get(h, ["", "", 0])[1]

    def cur_line(self, h: int) -> int:
        """That statement's line; 0 before the first."""
        return self.where.get(h, ["", "", 0])[2]

    def _local(self, fn_id: int) -> bool:
        """An import Python serves, not SV: here, one a double mocks."""
        prov = self.session.vm.import_provider
        return bool(getattr(prov, "is_mocked", None) and prov.is_mocked(fn_id))

    def _req(self, h: int, rid: int):
        """Taken request ``rid`` of call ``h``; -1: the last run status."""
        if rid < 0:
            return self.last[h]
        return self.taken[h][rid]

    def req_int(self, h: int, rid: int, field: str) -> int:
        """An int field of request ``rid`` (-1: the run status): req_id,
        api (0 = mem), executor (-1: none), thread, kind (1 = write), size,
        reg_id (-1: none), fn_id, blocking, has_ret, nret (a struct
        result's field count; 0: a scalar
        or none), on (0 TIME, 1 EVENT, 2 REQS, 3 TURN: a traversal queued
        behind the active one)."""
        r = self._req(h, rid)
        if field == "kind":
            return 1 if r.kind == "w" else 0
        if field == "on":
            return {"TIME": 0, "EVENT": 1, "REQS": 2, "TURN": 3}.get(r.on, 0)
        if field == "reg_id":
            return -1 if r.reg_id is None else int(r.reg_id)
        if field == "executor":
            return _s64(r.executor) if r.executor != 0xFFFFFFFF else -1
        if field in ("fn_id", "fn"):
            return int(r.fn or 0)
        if field == "has_ret":
            return 1
        if field == "nret":                  # as the C engine's DPI
            return r.nret if r.nret > 1 else 0
        return int(getattr(r, field) or 0)

    def req_u64(self, h: int, rid: int, field: str) -> int:
        """addr, data or delay."""
        return _s64(getattr(self._req(h, rid), field))

    def req_str(self, h: int, rid: int, field: str) -> str:
        """name (an import's, as the model names it), src_ref, stim_id (the
        action run that made the request; "" outside one), reason."""
        r = self._req(h, rid)
        if field == "name":
            prov = self.session.vm.import_provider
            return r.name or getattr(prov, "names", {}).get(r.fn, "") or ""
        return str(getattr(r, field, "") or "")

    def req_nargs(self, h: int, rid: int) -> int:
        return len(self._req(h, rid).args)

    def req_arg(self, h: int, rid: int, i: int) -> int:
        return _s64(self._req(h, rid).args[i])

    def set_ret(self, h: int, rid: int, i: int, v: int) -> int:
        """Stage field ``i`` of request ``rid``'s struct result (``nret``
        fields; a one-field struct may answer through ``complete``'s
        ``data``); ``complete`` then answers with the staged values."""
        try:
            self.rets.setdefault((h, rid), {})[i] = _u64(v)
            return 0
        except Exception as e:
            return self._fail(h, e)

    def string_ret(self, h: int, rid: int, i: int, text: str) -> int:
        """Stage the text of string result leaf ``i`` (0 for a scalar) of
        import request ``rid``, before ``complete`` (host-abi.md "String
        results")."""
        try:
            self.calls[h].ret_string(rid, text, i)
            return 0
        except Exception as e:
            return self._fail(h, e)

    def complete(self, h: int, rid: int, data: int) -> int:
        """Answer request ``rid``: a read's value (ignored for a write) or
        an import's result; a struct result (``nret`` > 0) is the values
        ``set_ret`` staged (``data`` is ignored)."""
        try:
            call = self.calls[h]
            r = self.taken[h].pop(rid)
            staged = self.rets.pop((h, rid), {})
            access = r.api == 0                # API_MEM
            write = access and r.kind == "w"
            n = getattr(r, "nret", 0)
            if n:
                if not staged:
                    staged = {0: _u64(data)}     # a one-field struct
                call.complete(rid, [staged.get(i, 0) for i in range(n)])
                return 0
            call.complete(rid, None if write else _u64(data))
            if access:
                self._shadow(r, r.data if write else _u64(data))
            return 0
        except Exception as e:
            return self._fail(h, e)

    def _shadow(self, r, v: int) -> None:
        """Keep the register fake's value of a register SV accessed: the
        simulation is the truth, the fake answers back-door reads
        (``check_reg``, ``peek_reg``)."""
        plat = self.session.platform
        if not hasattr(plat, "reg_at"):
            return
        plat.note_access(r.reg_id, r.src_ref)   # places the device, as a
        cell = plat._reg(r.addr)                # bus access would
        if cell is not None:
            plat.values[cell] = v

    def result(self, h: int) -> int:
        try:
            return _s64(self.calls[h].result)
        except Exception as e:
            self._fail(h, e)
            return 0

    def obj_field(self, h: int, name: str) -> int:
        """Field ``name`` of call ``h``'s action object after DONE, before
        ``call_end`` (F5.6; the C engine's ``pssvm_call_field``); 0 and the
        call's error set if it has none."""
        try:
            call = self.calls[h]
            last = self.last.get(h)
            if getattr(last, "status", None) != 0:
                raise RuntimeError(f"field '{name}' read before the call is done")
            obj = getattr(getattr(call, "root", None), "obj", None)
            if obj is None or not obj.has_field(name):
                raise KeyError(f"no field '{name}' in the call's action object")
            return _s64(obj.get_field_name(name))
        except Exception as e:
            self._fail(h, e)
            return 0

    def call_end(self, h: int) -> None:
        call = self.calls.pop(h, None)
        self.where.pop(h, None)
        self.last.pop(h, None)
        self.taken.pop(h, None)
        self.queued.pop(h, None)
        for k in [k for k in self.rets if k[0] == h]:
            del self.rets[k]
        if call is not None:
            call.end()

    def notify(self, kind: int, id: int, value: int) -> int:
        """A host event for the next run. A channel put (``N_CHAN`` 2) goes
        to one call (the oldest; with none, the next one started), as the C
        engine's notification queue takes it once; -1: no such channel."""
        if kind == 2:
            if not 0 <= id < len(getattr(self.session.model, "events", None) or []):
                return -1
            for call in self.calls.values():
                if hasattr(call, "notify"):
                    call.notify(kind, id, _u64(value))
                    return 0
            self.chan_notes.append((kind, id, _u64(value)))
            return 0
        for call in self.calls.values():
            call.notify(kind, id, _u64(value))
        return 0

    def report(self, h: int, kind: str, t_fs: int, js: str) -> int:
        """The bench reported something (``pssvm_report``; F2.7): a
        ``uvm_error`` the report catcher caught (``kind`` "uvm"), a
        ``pssvm_bench_fail`` ("bench"), an assertion. ``js`` is a JSON
        object ``{severity, id, msg, keys}``; ``h`` the call it came from,
        or 0 (any time, from any thread). The VM records it against the
        action in flight (``keys["stim"]`` when the bench names one); an
        error or fatal fails the test (``bench_failed``). 0, or -1
        (``error(h)``)."""
        try:
            d = json.loads(js) if js else {}
            if not isinstance(d, dict):
                raise ValueError("report: not a JSON object")
            keys = d.get("keys") or {}
            r = self.session.vm.bench_report(
                kind or "bench", str(d.get("severity") or "error"),
                str(d.get("msg") or ""), str(d.get("id") or ""),
                time=_u64(t_fs), keys=keys)
            if r["verdict"] == "fail":
                self.n_bench_fail += 1
                print(attribution_line(r), flush=True)
            return 0
        except Exception as e:
            return self._fail(h, e)

    def bench_failed(self) -> int:
        """Bench reports so far in this engine that fail a test (an error
        or fatal, ``report``); the runner fails a test when it grows."""
        return self.n_bench_fail

    def error(self, h: int) -> str:
        return self.errors.get(h, "")

    def fatal(self) -> str:
        """The fatal error the VM stopped on, located; "" when none. Every
        call then ends in ERROR, and later calls are refused naming it,
        until ``fresh`` builds the tree again."""
        return self.fatal_msg

    def _fail(self, h: int, e: BaseException) -> int:
        self.errors[h] = json.dumps({"message": f"{type(e).__name__}: {e}"})
        return -1

    # -- objects, links and driving (host-abi.md; zsp_rt_bc, step 7) ------ #

    def _table(self):
        """The VM's CallTable, with this engine listening on it."""
        from zuspec.be.bc.interp.host import calls_of
        s = self.session
        if s is None:
            raise RuntimeError("no image loaded")
        t = calls_of(s.vm)
        if self._listener not in t.listeners:
            t.add_listener(self._listener)
        return t

    def new_obj(self, type_name: str, parent: int) -> int:
        """``new``: ``parent`` 0, the tree of root ``type_name`` (built
        again; a failed construction is the engine's fatal); else an
        action object under component handle ``parent``. The handle, or 0
        (``error(0)``)."""
        try:
            return self._table().new(type_name, parent, seed=self.session.seed)
        except Exception as e:
            self._fail(0, e)
            return 0

    def sub(self, parent: int, path: str) -> int:
        try:
            return self._table().sub(parent, path)
        except Exception as e:
            self._fail(0, e)
            return 0

    def call_obj(self, obj: int, method: str = "") -> int:
        """A call keyed by handle, with the pushed args: a call handle, or
        0 (``error(0)``)."""
        args, self.args = self.args, []
        try:
            call = self._table().call(obj, method, args, seed=self.session.seed)
        except Exception as e:
            self._fail(0, e)
            return 0
        return self._started(call, method or call.name)

    def release_obj(self, obj: int) -> int:
        try:
            self._table().release(obj)
            return 0
        except Exception as e:
            return self._fail(0, e)

    def field(self, obj: int, name: str) -> int:
        try:
            return _s64(self._table().field(obj, name))
        except Exception as e:
            self._fail(0, e)
            return 0

    def api_id(self, name: str) -> int:
        try:
            return self._table().api_id(name)
        except Exception as e:
            return self._fail(0, e)

    def missing_links(self, linked: str) -> int:
        """How many APIs lack a handler, given SV's links ``linked``
        (``api:ctx;...``: -1 any API or context, -2 no executor);
        ``error(0)`` names them. SV keeps its handlers itself (a request
        goes to SV either way), so the table's own links are not touched."""
        from zuspec.be.bc.interp import host as H
        try:
            t = self._table()
            links = {}
            for item in (linked or "").split(";"):
                if not item:
                    continue
                a, _, c = item.partition(":")
                api, ctx = int(a), int(c)
                api = H.API_ANY if api == -1 else api
                ctx = (H.CTX_ANY if ctx == -1 else
                       H.NO_EXECUTOR if ctx == -2 else ctx)
                links[(api, ctx)] = True
            keep, t.links = t.links, links
            try:
                missing = t.missing_links()
            finally:
                t.links = keep
            if missing:
                self.errors[0] = json.dumps({"message": "no handler linked for "
                                             + "; ".join(missing)})
            return len(missing)
        except Exception as e:
            return self._fail(0, e)

    def step(self, budget: int) -> int:
        """``CallTable.step``; then each call's new requests wait for
        ``take`` (those a double mocks are answered). The slices run, or
        -1 (``error(0)``)."""
        try:
            n = self._table().step(budget)
            for h, call in list(self.calls.items()):
                r = getattr(call, "_last", None)
                if r is not None and r is not self.last.get(h):
                    self._status(h, r)
                    if r.status == 1:      # REQS
                        self._collect(h, call)
            return n
        except Exception as e:
            return self._fail(0, e)

    def pending(self) -> int:
        try:
            return self._table().pending().flags
        except Exception as e:
            return self._fail(0, e)

    def next_time(self) -> int:
        try:
            t = self._table().pending().next_time
            return -1 if t is None else int(t)
        except Exception as e:
            return self._fail(0, e)

    def progress(self) -> int:
        """``VM.progress``: grows when a call makes progress."""
        vm = getattr(self.session, "vm", None)
        return int(getattr(vm, "progress", 0)) if vm is not None else -1

    def call_inside(self, h: int, outer: int, rid: int) -> int:
        """Call ``h`` was started inside request ``rid`` of call ``outer``
        (``_run_inside``)."""
        if h not in self.calls:
            return self._fail(0, KeyError(f"no call {h}"))
        self.inside[h] = (outer, rid)
        return 0

    # -- tests ------------------------------------------------------------- #

    def test_name(self, i: int) -> str:
        """Test ``i`` of the image (its function tests, not skipped), as
        ``comp::fn``; "" past the last."""
        return self.tests[i].qname if 0 <= i < len(self.tests) else ""

    # -- executors (F5.4a) ------------------------------------------------- #

    def _executors(self) -> list:
        """The image's executors (its EXEC rows: ``path``, ``type``,
        ``trait``), by id; none before a load."""
        s = self.session
        return list(getattr(getattr(s, "model", None), "executors", None)
                    or [])

    def exec_id(self, path: str) -> int:
        """The id requests carry for the executor at instance ``path``
        (``set_executor``), or -1 if the image has none there."""
        for i, e in enumerate(self._executors()):
            if e.get("path") == path:
                return i
        return -1

    def event_id(self, path: str) -> int:
        """The id ``notify(N_CHAN, id, value)`` takes for the host-fed
        channel at instance ``path`` (its EVENT row, F5.5a), or -1."""
        s = self.session
        for i, e in enumerate(getattr(getattr(s, "model", None), "events", None) or []):
            if e.get("path") == path:
                return i
        return -1

    def time_exp(self) -> int:
        """One WAIT TIME tick is ``10**time_exp`` s (``set_time`` stays fs):
        the image's IDENT, -12 (ps) without one (F5.7)."""
        return getattr(getattr(self.session, "model", None), "time_exp", -12)

    def exec_path(self, id: int) -> str:
        """The instance path of executor ``id`` (a request's ``executor``),
        or "" for none (-1) or an unknown id."""
        execs = self._executors()
        return execs[id].get("path", "") if 0 <= id < len(execs) else ""

    def string_at(self, idx: int) -> str:
        """The text of model string ``idx`` (an import's string argument
        arrives as this index; host-abi.md "String arguments"), or "" out
        of range."""
        strings = getattr(getattr(self.session, "model", None), "strings", None) or []
        return strings[idx] if 0 <= idx < len(strings) else ""

    def test_expect(self, i: int) -> str:
        """The error test ``i`` expects (``expect_error``), or ""."""
        if not 0 <= i < len(self.tests):
            return ""
        return self.tests[i].expect_error or ""

    def hook(self, kind: str, i: int) -> str:
        """Session (``kind`` "session", run once per simulation), setup
        ("setup") or teardown function ``i``, as ``comp::fn``, in the order
        to call them; "" past the last. "setup" does not include the
        session functions."""
        fns = self.hooks.get(kind, [])
        return fns[i] if 0 <= i < len(fns) else ""

    # -- results ------------------------------------------------------------ #

    def checks_failed(self) -> int:
        """Checks failed so far in this engine (the session's ledger)."""
        led = getattr(getattr(self.session, "run_state", None), "ledger", None)
        return int(getattr(led, "n_fail", 0) or 0) if led is not None else 0

    def api_check(self, spec: str) -> str:
        """"" when the image has every function of ``spec`` -- ``;``-separated
        ``comp::fn(p,s.f)`` lines, ``->r`` when it returns a value, as a
        generated facade (target op-model-sv-vm) calls them -- with those
        parameter leaves (the value ABI's, by name and order) and result;
        else what differs, one clause per function."""
        try:
            entries = self.session.model.entries
            bad = []
            for line in filter(None, spec.split(";")):
                sym, _, rest = line.partition("(")
                params, _, ret = rest.partition(")")
                want = [p for p in params.split(",") if p]
                e = entries.get(sym)
                if e is None or e.kind != "function":
                    bad.append(f"{sym}: not in the image")
                    continue
                have = [n for n, _ in e.params]
                if have != want:
                    bad.append(f"{sym}: parameters ({','.join(have)}) in the "
                               f"image, ({','.join(want)}) in the facade")
                elif (e.returns is not None) != (ret == "->r"):
                    bad.append(f"{sym}: " + ("returns a value in the image, "
                               "not in the facade" if e.returns is not None
                               else "returns nothing in the image"))
            return "; ".join(bad)
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    # -- server mode -------------------------------------------------------- #

    def set_time(self, t_fs: int) -> None:
        """The sim time now, in fs (``pssvm_now_fs()``): the SV side sets it
        before each step of a call and before each runner hook, and every
        event to the daemon carries it as ``t``."""
        self.now_fs = _u64(t_fs)

    def trace(self, directory: str, name: str) -> str:
        """Trace test ``name`` (the runner's ``+pssvm_trace=DIR``): its
        events go to ``<directory>/<test-id>.ndjson``, as be-bc's ``TraceWriter``
        writes them, each with ``t``, the sim time in fs. ""
        ``name`` ends the trace. Returns the file, or "" (``error(0)``)."""
        old = getattr(self, "_trace", None)
        tee = self.session.events if self.session is not None else None
        if old is not None:
            self._trace = None
            if tee is not None and tee.sink is old[0]:
                tee.sink, tee.watch = None, 0
                self.session.vm.watch(self.session.watch)
            old[0].stream.close()
        if not name:
            return ""
        try:
            import os
            from zuspec.be.bc.trace.writer import TraceWriter, trace_path
            if self.session is None:
                raise RuntimeError("no image loaded")
            path = trace_path(directory, name)
            os.makedirs(directory or ".", exist_ok=True)
            # Line-buffered: a simulator's $finish need not finalize Python,
            # so a trace nobody ends must still have every event.
            w = TraceWriter(open(path, "w", buffering=1), self.session.model,
                            clock=lambda: self.now_fs, test=name,
                            seed=self.session.seed)
            tee.sink, tee.watch = w, w.watch
            self.session.vm.watch(self.session.watch | w.watch)
            self._trace_checks()
            self._trace = (w, path)
            return path
        except Exception as e:
            self._fail(0, e)
            return ""

    def _trace_checks(self) -> None:
        """Make the session's checks show in a trace just started; a tool's
        engine overrides it."""

    def serve(self, name: str = "", top: str = "") -> str:
        """Register with a debug daemon as ``name``; the session id, or ""
        (``error(0)``). This engine has none: a tool's engine overrides it
        (with ``self.server``, which ``poll`` and ``dbg_wait`` use)."""
        self.server = None
        self._fail(0, NotImplementedError(f"{type(self).__name__} has no daemon"))
        return ""

    def poll(self, timeout_ms: int = 0) -> int:
        """Answer one request from the daemon, waiting up to ``timeout_ms``
        (< 0: until one comes): 1 answered, 2 tests to run (``pending_sel``;
        the testbench runs them, then ``serve_done``), 0 none, -1 the
        channel closed or the daemon ended the session."""
        srv = getattr(self, "server", None)
        if srv is None or srv.ended or srv.closed:
            return -1
        t = None if timeout_ms < 0 else timeout_ms / 1000.0
        done = srv.poll(t)
        if srv.ended:
            srv.close()
            return -1
        if getattr(self, "_run", None) is not None:
            return 2
        if done:
            return 1
        return -1 if srv.closed else 0

    def _on_debug(self, args: dict):
        """``debug`` {on}: on puts a debug stub on the VM (the daemon then
        ``attach``es it with ``entry: false``); it stops inside the calls SV
        makes, blocking them, and with them sim time. Off ends it. Answered
        on the channel's reader thread: a stopped run blocks ``poll``."""
        import threading
        from zuspec.be.bc.interp.debug import Stub
        old = getattr(self, "_dbg", None)
        if old is not None:
            stub, done = old
            self._dbg = None
            self.server.attach_stub(None)
            if stub.state not in ("detached", "exited"):
                stub.request({"cmd": "detach"})
            done.set()
            stub.join(5.0)
        if not args.get("on", True):
            return {}
        if self.session is None:
            raise RuntimeError("no image loaded")
        done = threading.Event()
        stub = Stub(self.session.vm, run=done.wait)
        self._dbg = (stub, done)
        self.server.attach_stub(stub)
        return {"image": getattr(self, "image", "")}

    def dbg_wait(self, why: str = "") -> int:
        """Hold the simulation for a debugger (plan P3.7b; the runner's
        ``+pssvm_dbg_on_fail``). Registers with the daemon if not serving
        yet, sends a ``held`` event {why}, then answers requests (``debug``,
        ``reload``, ...) on this, the SV, thread, so sim time stands still,
        until a ``resume`` {rerun} comes. 1: run the failed test again, 0:
        go on, -1: no daemon, or the session ended."""
        srv = getattr(self, "server", None)
        if srv is None or srv.ended or srv.closed:
            if not self.serve(""):
                return -1
            srv = self.server
        self._resume = None
        self._held = why
        srv.event("held", {"why": why})
        try:
            while self._resume is None:
                srv.poll(0.1)
                if srv.ended or srv.closed:
                    srv.close()
                    return -1
        finally:
            self._held = None
        rerun, self._resume = self._resume, None
        srv.event("resumed", {"why": why, "rerun": rerun})
        return 1 if rerun else 0

    def _on_resume(self, args: dict):
        if getattr(self, "_held", None) is None:
            raise RuntimeError("not held: nothing to resume")
        self._resume = bool(args.get("rerun"))
        return {"why": self._held, "rerun": self._resume}

    def pending_sel(self) -> str:
        """The test selection of the pending ``run`` (``+pssunit`` form;
        "" = every test)."""
        run = getattr(self, "_run", None)
        return run[1] if run else ""

    def serve_test(self, name: str, passed: int) -> None:
        """A test of the pending ``run`` finished: a ``test`` event."""
        srv = getattr(self, "server", None)
        if srv is not None:
            srv.event("test", {"name": name,
                               "status": "pass" if passed else "fail"})

    def serve_done(self, n_pass: int, n_fail: int) -> None:
        """The pending ``run`` finished: answer it with the counts."""
        srv, run = getattr(self, "server", None), getattr(self, "_run", None)
        self._run = None
        if srv is not None and run is not None:
            srv.reply(run[0], {"passed": n_pass, "failed": n_fail})


try:
    import hdl_if as hif
except ImportError:          # Engine alone, without a simulator
    hif = None


def make_api(engine_cls):
    """The pyhdl-if ``ZspRtBcPy`` class over ``engine_cls`` (``Engine`` or a
    subclass), for SV's ``zsp_rt_bc`` (pyvm variant); needs pyhdl-if. Its
    methods are generated from ``zuspec.be.bc.share.rt_bc_api``, the one
    method list both variants of ``zsp_rt_bc_pkg`` come from."""
    if hif is None:
        raise ImportError("make_api needs pyhdl-if (hdl_if)")
    from zuspec.be.bc.share import rt_bc_api
    scope = {"hif": hif, "ct": ct, "engine_cls": engine_cls}
    exec(rt_bc_api.transport_source(), scope)
    cls = scope[rt_bc_api.API_CLASS]
    # pyhdl-if names the API by module and class: the SV API class
    # (zsp_rt_bc_pkg.sv) expects ``pssvm_dpi_shim.ZspRtBcPy``
    cls.__module__ = engine_cls.__module__
    cls.__qualname__ = rt_bc_api.API_CLASS
    return hif.api(cls)


#: the SV side imports module ``pssvm_dpi_shim``: with this directory on
#: PYTHONPATH, that is this file and the engine is ``Engine``; a tool's
#: shim of that name subclasses ``Engine`` and calls ``make_api`` itself
if hif is not None and __name__ == "pssvm_dpi_shim":
    ZspRtBcPy = make_api(Engine)
