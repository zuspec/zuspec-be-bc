"""
ops_orch.py -- orchestration op handlers (P1-10) + trace emission (P1-12).

The suspend-capable tier. Each handler drives the scheduler and emits its trace
event, then tells the VM whether the frame keeps running (:data:`CONTINUE`) or
yields control back to the scheduler (:data:`SUSPEND`). The event carries the
instruction's ``src_ref`` so every orchestration step is traceable to its source
`[D§10]`.

M1 mapping (design D§4.1), with pure-Python / synchronous externs:

* **WAIT**   -- re-queue at ``now + delay``; suspend.
* **SPAWN**  -- fork a child frame (ready now); continue.
* **INVOKE** -- blocking: fork a callee, suspend until it returns (its return value
  is delivered to the caller's result register); non-blocking: fire-and-forget.
* **JOIN**   -- suspend until every live spawned child has completed.
* **IMPORT** -- dispatch through the provider table now; write the result register;
  continue (a blocking import completes synchronously in M1).
* **SOLVE**  -- run the solve backend, write results back per the value ABI; continue.
* **YIELD**  -- re-queue at the back of the ready queue; suspend.
* **BIND**   -- flow-object bind; M1 no-op (traced); continue.
* **SELECT** -- draw one branch (weighted, declaration order) and run it as a
  blocking child; suspend until it returns.

PAR desugars to SPAWN/JOIN at lowering, so the VM never sees a PAR op (its handler
stays as an internal-error guard).
"""

import dataclasses as dc

from ..model import (Op, INSTR_F_BLOCKING, INSTR_F_HAS_RET, INSTR_F_NODE, INSTR_F_INITED, INSTR_F_SUBSET,
                     INSTR_F_SPIN, CALL_MAX_DEPTH,
                     BUILTIN_BASE, BUILTIN_MESSAGE, BUILTIN_ERROR,
                     BUILTIN_ERROR_MSG, BUILTIN_FATAL_MSG, BUILTIN_PRINT,
                     BUILTIN_FILE_OPEN, BUILTIN_FILE_CLOSE, BUILTIN_FILE_EXISTS,
                     BUILTIN_FILE_WRITE, BUILTIN_URANDOM, BUILTIN_URANDOM_RANGE,
                     BUILTIN_FILE_READ, BUILTIN_FILE_READ_LINES,
                     BUILTIN_FILE_WRITE_LINES, BUILTIN_READ, BUILTIN_WRITE,
                     HEAPK_LIST, HEAP_CELL_STR)

_MEM_READ = {v: k for k, v in BUILTIN_READ.items()}
_MEM_WRITE = {v: k for k, v in BUILTIN_WRITE.items()}
from . import faults
from .fmt import format_message
from ..trace.schema import EventKind
from ..trace import events as ev
from .host import PARKED
from .heap import HeapStrings, exec_heap
from .ops_proc import VMError, _get, _set, _u64
from .solve_cache import SolveBudgetError, SolveUnsatError

_VOID = 0xFFFFFFFF


@dc.dataclass
class Action:
    """What an orchestration handler tells the VM loop to do next."""

    suspend: bool = False


CONTINUE = Action(suspend=False)
SUSPEND = Action(suspend=True)


def _emit(vm, frame, kind, ins, detail):
    vm.sink.event(kind, coro=frame.id, src_ref=ins.src_ref, detail=detail)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #

def _op_wait(vm, frame, ins):
    delay = ins.imm if not ins.args else _get(frame, ins.args[0])
    vm.sched.wait(frame, delay)
    _emit(vm, frame, EventKind.WAIT, ins,
          {"delay": int(delay), "at": vm.sched.now + int(delay)})
    return SUSPEND


def _op_spawn(vm, frame, ins):
    target = ins.args[0]
    child = vm.spawn_child(frame, target)
    vm.sched.ready(child)
    _emit(vm, frame, EventKind.SPAWN, ins, {"child": child.id, "target": target})
    return CONTINUE


def _op_invoke(vm, frame, ins):
    target = ins.args[0]
    blocking = bool(ins.flags & INSTR_F_BLOCKING)
    node = (ins.imm, ins.args[2]) if ins.flags & INSTR_F_NODE else None
    start = ins.args[3] if ins.flags & INSTR_F_INITED else 0
    child = vm.spawn_child(frame, target, node=node, start_pc=start)
    if blocking:
        ret_reg = ins.args[1] if (ins.flags & INSTR_F_HAS_RET) else None
        child.ret_target = (frame, ret_reg) if ret_reg is not None else (frame, None)
        child.counted = True         # the callee counts toward this INVOKE's wait
        frame.pending += 1
        vm.sched.ready(child)
        _emit(vm, frame, EventKind.INVOKE, ins,
              {"child": child.id, "target": target, "blocking": True,
               "stim": child.stim})
        return SUSPEND
    vm.sched.ready(child)
    _emit(vm, frame, EventKind.INVOKE, ins,
          {"child": child.id, "target": target, "blocking": False,
           "stim": child.stim})
    return CONTINUE


def _op_join(vm, frame, ins):
    # imm = 0 -> join ALL live children; imm = n > 0 -> resume after the first n
    # complete (FIRST(n)); the surplus is cancelled when the parent resumes.
    live = [c for c in frame.children if not c.done and not c.cancelled]
    if ins.flags & INSTR_F_SUBSET:
        return _join_subset(vm, frame, ins, live)
    for c in live:
        c.counted = True
    n = ins.imm if ins.imm else len(live)
    frame.pending = min(int(n), len(live))
    _emit(vm, frame, EventKind.JOIN, ins,
          {"pending": frame.pending, "first": int(ins.imm)})
    return SUSPEND if frame.pending else CONTINUE


def _join_subset(vm, frame, ins, live):
    """join_select(n) / join_branch (INSTR_F_SUBSET): wait for some of the
    block's branches, the last arg0 children spawned (none has run yet); the
    others are not counted, so they run on and are never cancelled."""
    from ..determinism import select_choice

    branches = live[len(live) - ins.args[0]:]
    if ins.imm:
        # join_select: n picks without replacement, in draw order, each one
        # draw from the frame's seed stream (as SELECT, D§15.1)
        pool = list(range(len(branches)))
        picked = []
        for _ in range(min(int(ins.imm), len(pool))):
            picked.append(pool.pop(select_choice(frame.seed, [1] * len(pool))))
    else:
        picked = [i for i in range(len(branches)) if ins.args[1] >> i & 1]
    for i in picked:
        branches[i].counted = True
    frame.pending = len(picked)
    _emit(vm, frame, EventKind.JOIN, ins,
          {"pending": frame.pending, "first": 0, "joined": sorted(picked)})
    return SUSPEND if frame.pending else CONTINUE


def _ev_mask(vm, frame):
    """The debug events this frame's coroutine raises at an IMPORT (0 when
    none are on: one dict test)."""
    return vm._ev.get(id(frame.coro), 0) & ev.EV_AT_IMPORT if vm._ev else 0


def _op_import(vm, frame, ins):
    fn_id = ins.args[0]
    ret_slot = ins.args[1]
    n_res = (ins.imm >> 8) & 0xFF
    if ins.imm & 0xFF:
        # More than two arguments: staged by ARG as for CALL (F6.5b),
        # consumed once the call completes (a parked IMPORT runs again).
        n = ins.imm & 0xFF
        args = list(frame.staged[:n]) + [0] * (n - len(frame.staged))
    else:
        args = [_get(frame, r) for r in ins.args[2:]]
    blocking = bool(ins.flags & INSTR_F_BLOCKING)
    m = _ev_mask(vm, frame)

    if fn_id >= BUILTIN_BASE:
        result = _builtin(vm, frame, fn_id, args, ins, m)
        if result is PARKED:
            frame.pc -= 1       # re-run once the host completes the request
            frame.parked = True
            return SUSPEND
        frame.parked = False
        if result is not None and ins.flags & INSTR_F_HAS_RET and ret_slot != _VOID:
            _set(frame, ret_slot, _u64(result))
        _emit(vm, frame, EventKind.IMPORT, ins, {"fn_id": fn_id, "args": list(args),
                                                 "blocking": False})
        return CONTINUE

    if frame.dropped:
        # M8d5: a dropped action calls nothing: its result reads 0
        result = [0] * n_res if n_res else 0
    elif vm.host is not None:
        result = vm.host.import_(frame, ins, fn_id, args, blocking)
        if result is PARKED:
            frame.pc -= 1
            frame.parked = True
            return SUSPEND
        frame.parked = False
    else:
        result = vm.import_provider.call(fn_id, args)
    if ins.imm & 0xFF:
        frame.staged = []

    detail = {"fn_id": fn_id, "args": list(args), "blocking": blocking}
    if frame.dropped:
        detail["dropped"] = True
    if n_res:
        # A struct result: the host's n values, read back with LD_RES
        # (host-abi.md "Struct results").
        vals = list(result) if isinstance(result, (list, tuple)) else []
        if len(vals) != n_res:
            raise VMError(f"import {vm.import_name(fn_id)!r} returns a struct of "
                          f"{n_res} fields; the host answered {len(vals)} values")
        frame.results = [_u64(_host_str(vm, v)) for v in vals]
        detail["ret"] = list(frame.results)
    if ins.flags & INSTR_F_HAS_RET and ret_slot != _VOID:
        _set(frame, ret_slot, _u64(_host_str(vm, result) if result is not None else 0))
        detail["ret"] = frame.regs[ret_slot]
    _emit(vm, frame, EventKind.IMPORT, ins, detail)
    if m & ev.EV_IMPORT:
        is_mocked = getattr(vm.import_provider, "is_mocked", None)
        d = {"name": vm.import_name(fn_id), "fn_id": fn_id, "args": list(args),
             "blocking": blocking,
             "mocked": bool(is_mocked(fn_id)) if is_mocked else False}
        if "ret" in detail:
            d["ret"] = detail["ret"]
        vm.events.event(ev.IMPORT, coro=frame.id, src_ref=ins.src_ref, detail=d)
    return CONTINUE


def _host_str(vm, v):
    """A host's string result (a ``str``, or ``bytes``) as a new heap string
    handle; any other value as it is (host-abi.md "String results")."""
    if isinstance(v, str):
        v = v.encode()
    if isinstance(v, (bytes, bytearray)):
        return vm.heap.new_str(v)
    return v


def _access(vm, frame, rw, w, addr, data):
    """An ``access`` event's detail; ``reg_id`` when the access is a
    register's (debug lowering, P1.7)."""
    d = {"rw": rw, "width": w, "addr": addr, "data": data}
    r = vm.reg_id(frame)
    if r is not None:
        d["reg_id"] = r
        if rw == "r" and vm.discarded(frame):
            d["discarded"] = True
    return d


def _note_access(vm, frame, ins):
    """Tell a memory that asks (``note_access(reg_id, src_ref)``) which
    register the access about to happen is for (None: no register's), so a
    register fake can give it the register's behaviour (P1.10)."""
    note = getattr(vm.memory, "note_access", None)
    if note is not None:
        note(vm.reg_id(frame), ins.src_ref if ins is not None else 0)


#: std_pkg file_option_e (24.2) -> open() mode
_FILE_MODES = {0: "w", 1: "a", 2: "r"}


def _file(vm, h):
    """The open file of handle *h* (1-based in the run's table), or a run error."""
    files = getattr(vm, "files", [])
    f = files[h - 1] if 0 < h <= len(files) else None
    if f is None:
        raise VMError(f"file handle {h}: not an open file")
    return f


def _ends_open(name) -> bool:
    """True if file *name* exists, is not empty and does not end in a newline."""
    import os
    if not os.path.exists(name) or os.path.getsize(name) == 0:
        return False
    with open(name, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) != b"\n"


def _file_builtin(vm, frame, fn_id, args):
    """``file_open`` / ``file_close`` / ``file_exists``, ``file_read`` /
    ``file_read_lines`` / ``file_write_lines`` (24.2): the engine serves them
    on the machine it runs on, names relative to its cwd; what they read comes
    back as new heap strings and lists."""
    import os
    if fn_id in (BUILTIN_FILE_EXISTS, BUILTIN_FILE_OPEN, BUILTIN_FILE_READ_LINES,
                 BUILTIN_FILE_WRITE_LINES):
        name = vm.heap.text(args[0]).decode()   # table or heap string
    if fn_id == BUILTIN_FILE_READ:         # at most size chars from where it is
        f = _file(vm, args[0])
        if "r" not in f.mode:
            raise VMError(f"file handle {args[0]}: not opened for reading")
        n = args[1] - (1 << 64) if args[1] >> 63 else args[1]
        return vm.heap.new_str(f.read(n if n >= 0 else -1).encode())
    if fn_id == BUILTIN_FILE_READ_LINES:   # each line, without its newline
        try:
            with open(name) as f:
                text = f.read()
        except OSError as e:
            raise VMError(f"file_read_lines({name!r}): {e.strerror}")
        return vm.heap.alloc(HEAPK_LIST, [vm.heap.new_str(x.encode())
                                          for x in text.splitlines()], HEAP_CELL_STR)
    if fn_id == BUILTIN_FILE_WRITE_LINES:  # a newline after each line
        mode = _FILE_MODES.get(args[2])
        if mode not in ("w", "a"):
            raise VMError(f"file_write_lines({name!r}): option must be TRUNCATE or APPEND")
        lines = [vm.heap.text(x).decode() for x in vm.heap.items(args[1])]
        try:
            # APPEND starts on a new line unless the file ends in one (24.2)
            lead = mode == "a" and _ends_open(name)
            with open(name, mode) as f:
                f.write(("\n" if lead else "") + "".join(x + "\n" for x in lines))
        except OSError as e:
            raise VMError(f"file_write_lines({name!r}): {e.strerror}")
        return None
    if fn_id == BUILTIN_FILE_EXISTS:
        return int(os.path.exists(name))
    if fn_id == BUILTIN_FILE_OPEN:
        mode = _FILE_MODES.get(args[1])
        try:
            f = open(name, mode) if mode else None
        except OSError:
            f = None
        if f is None:
            return 0                       # nullfilehandle: cannot open it so
        if not hasattr(vm, "files"):
            vm.files = []
        vm.files.append(f)
        return len(vm.files)
    _file(vm, args[0]).close()             # file_close
    vm.files[args[0] - 1] = None
    return None


def _builtin(vm, frame, fn_id, args, ins=None, m=0):
    """The interpreter's own imports: ``message()`` and LRM-mandated errors.
    ``m``: the frame's debug events (access, message) to raise."""
    if fn_id in (BUILTIN_FILE_OPEN, BUILTIN_FILE_CLOSE, BUILTIN_FILE_EXISTS,
                 BUILTIN_FILE_READ, BUILTIN_FILE_READ_LINES, BUILTIN_FILE_WRITE_LINES):
        return _file_builtin(vm, frame, fn_id, args)
    if fn_id == BUILTIN_URANDOM:           # 24.4: the frame's own seed stream
        from ..determinism import urandom
        return urandom(frame.seed)
    if fn_id == BUILTIN_URANDOM_RANGE:
        from ..determinism import urandom_range
        return urandom_range(frame.seed, args[0], args[1])
    if fn_id in (BUILTIN_MESSAGE, BUILTIN_PRINT, BUILTIN_ERROR_MSG, BUILTIN_FATAL_MSG,
                 BUILTIN_FILE_WRITE):
        entry = vm.model.messages[args[0]]
        vals = [frame.locals[s] for s in entry["slots"]]
        if fn_id == BUILTIN_MESSAGE and vals[0] > vm.verbosity:
            return                    # 21.1.3: above the run's verbosity -> ignored
        try:
            line = format_message(entry["fmt"], entry["args"], vals[1:],
                                  HeapStrings(vm.heap))
        except ValueError as e:
            raise VMError(f"message(): {e}")
        if fn_id == BUILTIN_FILE_WRITE:    # as written: no newline (24.2)
            f = _file(vm, vals[0])
            if "r" in f.mode:
                raise VMError(f"file handle {vals[0]}: opened for reading, not writing")
            f.write(line)
            f.flush()
            return
        if fn_id in (BUILTIN_MESSAGE, BUILTIN_PRINT):    # print(): any verbosity
            vm.out(line)
            if m & ev.EV_MESSAGE:
                vm.events.event(ev.MESSAGE, coro=frame.id, src_ref=ins.src_ref,
                                detail={"severity": "info", "level": vals[0],
                                        "text": line})
            return
        if frame.dropped:
            return                    # M8d5: a dropped action's errors do not count
        if m & ev.EV_MESSAGE:
            d = ({"severity": "error", "text": line} if fn_id == BUILTIN_ERROR_MSG
                 else {"severity": "fatal", "status": vals[0], "text": line})
            vm.events.event(ev.MESSAGE, coro=frame.id, src_ref=ins.src_ref, detail=d)
        # error()/fatal(): the run stops here, located by the dispatch loop
        if fn_id == BUILTIN_ERROR_MSG:
            raise VMError("error: " + line)
        raise VMError(f"fatal (status {vals[0]}): {line}")
    if fn_id == BUILTIN_ERROR:
        raise VMError(vm.model.strings[args[0]])
    if frame.dropped and (fn_id in _MEM_READ or fn_id in _MEM_WRITE):
        return 0 if fn_id in _MEM_READ else None   # M8d5: no memory effect
    if fn_id in _MEM_READ:
        w = _MEM_READ[fn_id]
        if m & ev.EV_ACCESS and vm.host is None:   # a host notes its requests
            _note_access(vm, frame, ins)
        if vm.host is not None:
            v = vm.host.access(frame, ins, "r", args[0], w // 8)
            if v is PARKED:
                return v
        else:
            v = vm.memory.read(args[0], w // 8)
        if m & ev.EV_ACCESS:
            vm.events.event(ev.ACCESS, coro=frame.id, src_ref=ins.src_ref,
                            detail=_access(vm, frame, "r", w, args[0], v))
        return v
    if fn_id in _MEM_WRITE:
        w = _MEM_WRITE[fn_id]
        if m & ev.EV_ACCESS and vm.host is None:   # a host notes its requests
            _note_access(vm, frame, ins)
        if vm.host is not None:
            if vm.host.access(frame, ins, "w", args[0], w // 8,
                              args[1]) is PARKED:
                return PARKED
        else:
            vm.memory.write(args[0], w // 8, args[1])
        if m & ev.EV_ACCESS:
            vm.events.event(ev.ACCESS, coro=frame.id, src_ref=ins.src_ref,
                            detail=_access(vm, frame, "w", w, args[0], args[1]))
        return None
    raise VMError(f"unknown builtin import 0x{fn_id:x}")


def _op_solve(vm, frame, ins):
    pid = ins.args[0]
    problem = vm.model.problems[pid]

    if problem.seed_kind == "fixed":
        seed = problem.seed_value & ((1 << 64) - 1)
    else:
        seed = frame.seed.next_raw()

    # A problem carrying a dv-solve blob is solved by the real solver (honoring its
    # constraints); a blob-less problem uses the configured backend (minimal stub).
    bounded = frame is vm.bound_frame and problem.problem_bytes
    if bounded and vm.entry_bounds() is not None:
        problem = _bounded(vm, pid, problem)
    if problem.problem_bytes:
        backend = vm.blob_backend
    else:
        backend = vm.solve_backend
    pins, frame.pins = frame.pins, None
    if problem.input_slots and frame.obj is not None:
        # Non-rand fields the constraints read hold their current values.
        pins = dict(pins or {})
        for slot, var_id in problem.input_slots.items():
            pins[var_id] = frame.obj.get_field(frame.base + slot)
    try:
        if problem.input_arrays:
            # Lists the constraints read: their current elements (F6.10d).
            arrays = [None] * len(problem.array_specs)
            handles = {}
            for slot, aid in problem.input_arrays.items():
                # procedural: the list's handle was pinned (SOLVE_PIN)
                h = (pins.pop(slot) if problem.procedural
                     else frame.obj.get_field(frame.base + slot))
                handles[aid] = h
                arrays[aid] = [x & ((1 << 64) - 1) for x in vm.heap.items(h)]
            solved = backend.randomize(frame.obj, problem, seed, pins=pins,
                                       arrays=arrays)
            # A rand list's elements (F6.10e), in place: its length is the
            # one it had when solved.
            for aid in problem.rand_arrays:
                if handles.get(aid):
                    vm.heap.obj(handles[aid]).data[:] = [
                        x & ((1 << 64) - 1) for x in backend.solved_arrays[aid]]
        elif pins:
            solved = backend.randomize(frame.obj, problem, seed, pins=pins)
        else:
            solved = backend.randomize(frame.obj, problem, seed)
    except (SolveBudgetError, SolveUnsatError) as e:
        what = "randomize in" if problem.procedural else "solving"
        err = type(e)("%s %r: %s" % (what, frame.coro.name, e))
        err.core = getattr(e, "core", [])
        raise err from None
    if problem.procedural:
        # `randomize x with {...}` (F3.7): the statement's stores move the
        # values; nothing is committed to the action's node.
        frame.solved = {problem.var_names.index(n): v for n, v in solved.items()
                        if n in problem.var_names}
        if _solve_on(vm, frame):
            _solve_event(vm, frame, ins, problem, seed, solved)
        _emit(vm, frame, EventKind.SOLVE, ins, {"problem": pid, "seed": seed})
        return CONTINUE
    if bounded:
        vm.solved.update(solved)

    # Write results back per the value ABI. The slot map is relative to the
    # frame's base (its node's slots, P1-D1); without one, writeback maps
    # field-name -> var_id, var_names is the sorted-by-name / var_id order,
    # backend keys by var name.
    if frame.obj is not None and problem.writeback_slots:
        for slot, var_id in problem.writeback_slots.items():
            name = problem.var_names[var_id]
            if name in solved:
                frame.obj.set_field(frame.base + slot, solved[name])
    elif frame.obj is not None:
        for field, var_id in problem.writeback.items():
            name = problem.var_names[var_id]
            if name in solved and frame.obj.has_field(field):
                frame.obj.set_field_name(field, solved[name])
    if frame.act is not None:
        frame.act.commit(frame.node, frame.site)
    if vm.on_solve is not None:
        vm.on_solve(frame)
    if _solve_on(vm, frame):
        _solve_event(vm, frame, ins, problem, seed, solved)

    _emit(vm, frame, EventKind.SOLVE, ins, {"problem": pid, "seed": seed})
    return CONTINUE


def _op_solve_pin(vm, frame, ins):
    """Stage input *var_id* = reg's value for the frame's next SOLVE (F3.7b)."""
    var_id, rs = ins.args[0], ins.args[1]
    if frame.pins is None:
        frame.pins = {}
    frame.pins[var_id] = _get(frame, rs)
    return CONTINUE


def _op_solve_get(vm, frame, ins):
    rd, var_id = ins.args[0], ins.args[1]
    solved = getattr(frame, "solved", None)
    if solved is None or var_id not in solved:
        raise VMError("SOLVE_GET of var %d with no procedural solve before it"
                      % var_id)
    _set(frame, rd, solved[var_id])
    return CONTINUE


def _op_cov_sample(vm, frame, ins):
    """COV_SAMPLE (F4.4): record that covergroup instance ``args[0]`` was
    sampled with the ``args[1]`` values staged by ARG (consumed); binning
    them against the image's COVG description is the coverage service's job
    (F4.5). A watched coroutine raises a ``cov`` event (F4.6)."""
    cid, n = ins.args[0], ins.args[1]
    vals = list(frame.staged[:n]) + [0] * (n - len(frame.staged))
    frame.staged = []
    vm.cov_samples.append((cid, vals))
    if vm._ev and vm._ev.get(id(frame.coro), 0) & ev.EV_COV:
        vm.events.event(ev.COV, coro=frame.id, src_ref=ins.src_ref,
                        detail={"covg": cid, "values": vals})
    return CONTINUE


def _op_exec_set(vm, frame, ins):
    """EXEC_SET (F5.4a): the frame's executor is now ``args[0]``'s value
    (an EXEC row id); the requests it issues carry it."""
    frame.executor = _get(frame, ins.args[0]) & 0xFFFFFFFF
    return CONTINUE


def _op_exec_get(vm, frame, ins):
    """EXEC_GET (F5.4c): ``args[0]`` = the frame's executor (an EXEC row id,
    0xFFFFFFFF = none: ``executor()`` is null)."""
    _set(frame, ins.args[0], frame.executor & 0xFFFFFFFF)
    return CONTINUE


def _op_heap(vm, frame, ins):
    """HEAP (F6.9c): a heap sub-op on the VM's heap (``heap.exec_heap``)."""
    exec_heap(vm, frame, ins)
    return CONTINUE


def _bounded(vm, pid, problem):
    """*problem* with ``vm.entry_bounds()`` added as constraints on its rand
    variables (a property test's shrink, P5.5; as
    ``ActivationTable.bound_blob`` does for a cone), or as is."""
    cache = vm._bounded
    if pid not in cache:
        cache[pid] = problem
        from zuspec.ir.core import constraint as C, expr as E
        src = problem.source
        if src is None and problem.problem_bytes:
            src = _arg_source(vm, problem)
        cons = []
        bounds = vm.entry_bounds()
        for v in (src.vars if src is not None else ()):
            b = bounds.get(v.name)
            if b is None and v.slot >= 0:
                b = bounds.get(v.slot)      # an exported action's argument
            if b is None:
                continue
            ref = E.ExprRefField(base=E.TypeExprRefSelf(),
                                 index=v.slot if v.slot >= 0 else v.var_id)
            for op, k in ((E.BinOp.GtE, b[0]), (E.BinOp.LtE, b[1])):
                if k is not None:
                    cons.append(C.ConstraintExpr(expr=E.ExprBin(
                        lhs=ref, op=op, rhs=E.ExprConstant(value=k))))
        if cons:
            from ..lower.constraints import build_solve_blob
            aux = build_solve_blob(dc.replace(src, constraints=cons))[0]
            cache[pid] = dc.replace(problem, problem_bytes=(problem.problem_bytes, aux))
    return cache[pid]


def _arg_source(vm, problem):
    """A stand-in source for *problem* read from an image (no ``source``):
    just its vars at pinned argument slots, typed from the entry's
    parameters, so ``_bounded`` can pin them; None when there are none."""
    from zuspec.ir.core import scenario as S
    types = getattr(vm, "arg_types", None) or {}
    vs = [S.ScSolveVar(name=problem.var_names[vid], var_id=vid, slot=slot,
                       width=types[slot][0], signed=types[slot][1])
          for slot, vid in sorted(problem.writeback_slots.items())
          if slot in types]
    return S.ScSolveProblem(vars=vs) if vs else None


def _solve_on(vm, frame) -> bool:
    """Whether the frame's coroutine raises ``solve`` debug events."""
    return bool(vm._ev) and bool(vm._ev.get(id(frame.coro), 0) & ev.EV_SOLVE)


def _solve_event(vm, frame, ins, problem, seed, solved) -> None:
    """The ``solve`` debug event of a SOLVE op (F3.10): *solved* is the
    backend's ``{name: value}``. A problem without its source (loaded from
    an image) lists no dropped softs."""
    from zuspec.ir.core import constraint as C
    from .extern import _domain_value
    from .activation import describe
    from .unsat_core import dropped_softs, soft_detail
    src = problem.source
    softs = ([c for c in src.constraints if isinstance(c, C.ConstraintSoft)]
             if src is not None else [])
    dropped = []
    if softs:
        vals = {i: _domain_value(solved[n], src.vars[i])
                for i, n in enumerate(problem.var_names) if n in solved}
        by_slot = {v.slot: v.name for v in src.vars if v.slot is not None}
        names = [by_slot.get(i, "$%d" % i) for i in range(max(by_slot, default=-1) + 1)]
        dropped = [soft_detail(c, describe(c, names))
                   for c in dropped_softs(src, vals)]
    vals = {n: (_domain_value(v, src.vars[problem.var_names.index(n)])
                if src is not None else v) for n, v in solved.items()}
    vm.events.event(ev.SOLVE, coro=frame.id, src_ref=ins.src_ref, detail={
        "site": frame.coro.name, "node": frame.node, "type": frame.coro.name,
        "seed": seed,
        "values": vals, "softs": len(softs), "dropped": dropped})


def _op_solve_node(vm, frame, ins):
    """Solve the frame's node in its cone (P1-D2), or -- a node in no cone --
    as SOLVE would (P1-D3)."""
    act = frame.act
    if act is not None and frame.node in act.t.cone_of:
        if act.state_wait(frame.node, frame.comp, frame):
            # M9a9d: a traversal of a state pool it references runs; this
            # SOLVE_NODE runs again when one ends (13.3.1)
            frame.pc -= 1
            return SUSPEND
        seed = frame.seed.next_raw()
        p = act.solve(frame.node, frame.site, seed, frame)
        if p is not None:
            return _run_inferred(vm, frame, ins, p)
        slot = act.t.nodes[frame.node].comp_slot
        if slot is not None:
            # The solve chose the node's component instance (P1-D4).
            vm.set_comp(frame, frame.obj.get_field(slot))
        act.state_enter(frame.node, frame.comp)
        if vm.on_solve is not None:
            vm.on_solve(frame)
        if _solve_on(vm, frame):
            vm.events.event(ev.SOLVE, coro=frame.id, src_ref=ins.src_ref,
                            detail=act.explain(frame.node, seed))
        _emit(vm, frame, EventKind.SOLVE, ins,
              {"node": act.t.nodes[frame.node].path, "seed": seed})
        if faults.on("skip-buffer-wait"):
            act.publish_buffers(frame.node, frame.comp)
        if faults.on("race-state"):
            act.publish_states(frame.node, frame.comp)
        for q in act.partners.pop(frame.node, ()):
            if not faults.on("drop-stream-partner"):
                _start_partner(vm, frame, ins, q)
        return CONTINUE
    if ins.args[0] != _VOID and vm.model.problems[ins.args[0]].var_names:
        return _op_solve(vm, frame, ins)
    if act is not None:
        act.commit(frame.node, frame.site)
    return CONTINUE


def _inferred_child(vm, frame, p):
    act = frame.act
    n = act.t.nodes[p]
    # its one site; owned by the node it serves, or (M9a7, run at the entry
    # of an atomic block) by a node this frame traverses later
    site = next(i for i, st in enumerate(act.t.sites) if st[1] == p)
    owner = act.t.sites[site][0]
    target = next(e.coro for e in vm.model.entries.values()
                  if e.kind == "action" and getattr(e, "qname", None) == n.type_qname)
    # M9b1a: the action it is inferred for, by its stimulus id (the owner
    # not entered yet gets the id its next traversal will have)
    for_stim = (frame.stim if owner == frame.node else "%s.%s#%d" % (
        act.stim, act.t.nodes[owner].path, act.visits[owner]))
    child = vm.spawn_child(frame, target, node=(n.base, None), abs_site=site)
    child.inferred_for = for_stim
    if p in act.infer_comp and n.comp_rel is not None:
        # M9b1c: the consumer's instance is its solve's choice; this node
        # feeds (and pins) the consumer in instance infer_comp[p]
        vm.set_comp(child, act.infer_comp[p] + n.comp_rel)
    elif owner != frame.node:
        rels = (act.t.nodes[owner].comp_rel, n.comp_rel)
        vm.set_comp(child, None if frame.comp is None or None in rels
                    else frame.comp + rels[0] + rels[1])
    return target, child


def _start_partner(vm, frame, ins, p):
    """M9a6b: start inferred stream partner *p* together with this frame's
    node, just solved; the node waits for it at its end."""
    target, child = _inferred_child(vm, frame, p)
    frame.partners.append(child)
    vm.sched.ready(child)
    _emit(vm, frame, EventKind.INVOKE, ins,
          {"child": child.id, "target": target, "blocking": False,
           "stim": child.stim, "inferred": True})


def _run_inferred(vm, frame, ins, p):
    """M9a3: run inferred node *p* to completion, then this SOLVE_NODE
    again, which may pick the object *p* output."""
    frame.pc -= 1
    target, child = _inferred_child(vm, frame, p)
    child.ret_target = (frame, None)
    child.counted = True
    frame.pending += 1
    vm.sched.ready(child)
    _emit(vm, frame, EventKind.INVOKE, ins,
          {"child": child.id, "target": target, "blocking": True,
           "stim": child.stim, "inferred": True})
    return SUSPEND


def _op_scope_enter(vm, frame, ins):
    """Entry to an activity block (13.4.8): reset what it traverses."""
    act = frame.act
    if act is not None:
        scope = act.t.nodes[frame.node].scope_base + ins.imm
        act.enter_scope(scope, seed=frame.seed.next_raw)
        if scope in act.t.atomic:
            # M9a7: what the atomic block needs inferred runs before it
            # (11.3.7); then the block is entered again
            p = act.atomic_infer(scope, frame.seed.next_raw(), frame.comp)
            if p is not None:
                return _run_inferred(vm, frame, ins, p)
    return CONTINUE


def _plan_choice(vm, frame, ins, value) -> None:
    """Record a select's branch in the run's plan (M8b6), against the
    occurrence whose activity it is in (None: no action tree)."""
    if vm.plan is None:
        return
    act = frame.act
    occ = act.occ[frame.node] if act is not None and frame.node is not None else None
    vm.plan.choice(occ, "select", ins.args[0], value)


def _op_select(vm, frame, ins):
    from ..determinism import select_choice

    table = vm.model.selects[ins.args[0]]
    guards = table.guards or [-1] * len(table.branches)
    # A branch is eligible when it is unguarded (-1) or its guard register is set.
    eligible = [i for i in range(len(table.branches))
                if guards[i] < 0 or _get(frame, guards[i]) != 0]

    # M9a7c: a branch a traversal of which can neither take an input object
    # nor have one inferred now is skipped (its block's first instruction
    # enters it; the probe draws no seed, so choices stay as they were)
    act = frame.act
    if act is not None and frame.node is not None and eligible:
        base = act.t.nodes[frame.node].scope_base
        blocked = {}
        for i in eligible:
            code = vm.model.coros[table.branches[i]].code
            ent = next((x for x in code if x.op == Op.SCOPE_ENTER), None)
            if ent is not None:
                why = act.select_blocked(base + ent.imm, 0, frame.comp)
                if why is not None:
                    blocked[i] = why
        if blocked and len(blocked) == len(eligible) and not table.allow_none:
            raise VMError("select: no branch can run: " + "; ".join(
                "branch %d: %s" % (i, w) for i, w in sorted(blocked.items())))
        eligible = [i for i in eligible if i not in blocked]

    if not eligible:
        if table.allow_none:
            _plan_choice(vm, frame, ins, None)
            _emit(vm, frame, EventKind.SELECT, ins, {"choice": None, "none": True})
            return CONTINUE
        raise VMError("select has no eligible branch and allow_none is false")

    # Draw among the eligible branches in declaration order against their cumulative
    # weights, advancing the frame's own seed stream (determinism spec D§15.1).
    weights = [table.weights[i] for i in eligible]
    choice = eligible[select_choice(frame.seed, weights)]
    target = table.branches[choice]
    _plan_choice(vm, frame, ins, choice)
    if act is not None and frame.node is not None:
        code = vm.model.coros[target].code
        ent = next((x for x in code if x.op == Op.SCOPE_ENTER), None)
        if ent is not None:
            act.select_chosen(act.t.nodes[frame.node].scope_base + ent.imm, frame.comp)

    child = vm.spawn_child(frame, target)
    child.ret_target = (frame, None)
    child.counted = True
    frame.pending += 1
    vm.sched.ready(child)
    _emit(vm, frame, EventKind.SELECT, ins,
          {"choice": choice, "target": target, "eligible": eligible, "weights": weights})
    return SUSPEND


def _op_bind(vm, frame, ins):
    _emit(vm, frame, EventKind.BIND, ins, {})
    return CONTINUE


def _op_call(vm, frame, ins):
    """A called function: runs next, then its caller resumes (B-D5)."""
    target, rd = ins.args[0], ins.args[1]
    if frame.depth >= CALL_MAX_DEPTH:
        raise VMError(f"function calls nested deeper than {CALL_MAX_DEPTH}")
    if frame.dropped and target in vm.drop_calls:
        # M8d5: a dropped action's check is not recorded; the call reads 0
        frame.staged = []
        if rd != _VOID:
            _set(frame, rd, 0)
        return CONTINUE
    if vm.bindings:
        bound = vm.bindings.get(target)
        if bound is not None:
            return _call_bound(vm, frame, ins, target, rd, bound)
    child = vm.call_child(frame, target)
    child.args, frame.staged = frame.staged, []
    child.ret_target = (frame, None if rd == 0xFFFFFFFF else rd)
    child.counted = True
    frame.pending += 1
    vm.sched.ready_first(child)
    return SUSPEND


def _call_bound(vm, frame, ins, target, rd, bound):
    """A CALL of a host-bound function (``VM.bind``): the host callable
    runs in place of the PSS body, on the raw argument values, and its
    result (None: 0) is the call's. ``call``/``return`` carry ``mocked``."""
    _qname, fn = bound
    args, frame.staged = frame.staged, []
    coro = vm.model.coros[target]
    m = vm._ev.get(id(coro), 0) if vm._ev else 0
    calls = m & ev.EV_CALL
    depth = frame.depth + 1
    if calls:
        vm.events.event(ev.CALL, coro=frame.id, src_ref=ins.src_ref,
                        detail={"fn": coro.name, "depth": depth,
                                "mocked": True, "args": list(args)})
    result = fn(*args)
    d = {"fn": coro.name, "depth": depth, "mocked": True}
    if rd != _VOID:
        _set(frame, rd, _u64(result if result is not None else 0))
        d["retval"] = frame.regs[rd]
    if calls:
        vm.events.event(ev.RETURN, coro=frame.id, src_ref=ins.src_ref, detail=d)
    return CONTINUE


def _op_yield(vm, frame, ins):
    if ins.flags & INSTR_F_SPIN and frame.resume_pc == frame.pc:
        # Resumed just past this YIELD and back here: its condition is
        # still false (see INSTR_F_SPIN).
        vm.stuck = True
    vm.sched.ready(frame)
    _emit(vm, frame, EventKind.YIELD, ins, {})
    return SUSPEND


def _reject(name):
    def handler(vm, frame, ins):
        raise VMError(f"{name} is not executable in M1 (deferred at lowering)")
    return handler


_ORCH = {
    Op.WAIT: _op_wait,
    Op.SPAWN: _op_spawn,
    Op.INVOKE: _op_invoke,
    Op.JOIN: _op_join,
    Op.IMPORT: _op_import,
    Op.SOLVE: _op_solve,
    Op.BIND: _op_bind,
    Op.YIELD: _op_yield,
    Op.CALL: _op_call,
    Op.SELECT: _op_select,
    Op.SOLVE_NODE: _op_solve_node,
    Op.SCOPE_ENTER: _op_scope_enter,
    Op.SOLVE_GET: _op_solve_get,
    Op.SOLVE_PIN: _op_solve_pin,
    Op.COV_SAMPLE: _op_cov_sample,
    Op.EXEC_SET: _op_exec_set,
    Op.EXEC_GET: _op_exec_get,
    Op.HEAP: _op_heap,
    # PAR desugars to SPAWN/JOIN at lowering, so the VM never sees a PAR op.
    Op.PAR: _reject("PAR"),
}

ORCH_OPS = frozenset(_ORCH)


def exec_orch(vm, frame, ins) -> Action:
    """Execute one orchestration op. Returns the control :class:`Action`."""
    handler = _ORCH.get(ins.op)
    if handler is None:
        raise VMError(f"no orchestration handler for {ins.op.name}")
    return handler(vm, frame, ins)
