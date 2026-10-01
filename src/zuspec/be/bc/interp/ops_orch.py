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

from ..model import (Op, INSTR_F_BLOCKING, INSTR_F_HAS_RET, INSTR_F_NODE, INSTR_F_INITED,
                     INSTR_F_SPIN, CALL_MAX_DEPTH,
                     BUILTIN_BASE, BUILTIN_MESSAGE, BUILTIN_ERROR,
                     BUILTIN_READ, BUILTIN_WRITE)

_MEM_READ = {v: k for k, v in BUILTIN_READ.items()}
_MEM_WRITE = {v: k for k, v in BUILTIN_WRITE.items()}
from .fmt import format_message
from ..trace.schema import EventKind
from .ops_proc import VMError, _get, _set, _u64

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
              {"child": child.id, "target": target, "blocking": True})
        return SUSPEND
    vm.sched.ready(child)
    _emit(vm, frame, EventKind.INVOKE, ins,
          {"child": child.id, "target": target, "blocking": False})
    return CONTINUE


def _op_join(vm, frame, ins):
    # imm = 0 -> join ALL live children; imm = n > 0 -> resume after the first n
    # complete (FIRST(n)); the surplus is cancelled when the parent resumes.
    live = [c for c in frame.children if not c.done and not c.cancelled]
    for c in live:
        c.counted = True
    n = ins.imm if ins.imm else len(live)
    frame.pending = min(int(n), len(live))
    _emit(vm, frame, EventKind.JOIN, ins,
          {"pending": frame.pending, "first": int(ins.imm)})
    return SUSPEND if frame.pending else CONTINUE


def _op_import(vm, frame, ins):
    fn_id = ins.args[0]
    ret_slot = ins.args[1]
    arg_regs = ins.args[2:]
    args = [_get(frame, r) for r in arg_regs]
    blocking = bool(ins.flags & INSTR_F_BLOCKING)

    if fn_id >= BUILTIN_BASE:
        result = _builtin(vm, frame, fn_id, args)
        if result is not None and ins.flags & INSTR_F_HAS_RET and ret_slot != _VOID:
            _set(frame, ret_slot, _u64(result))
        _emit(vm, frame, EventKind.IMPORT, ins, {"fn_id": fn_id, "args": list(args),
                                                 "blocking": False})
        return CONTINUE

    result = vm.import_provider.call(fn_id, args)

    detail = {"fn_id": fn_id, "args": list(args), "blocking": blocking}
    if ins.flags & INSTR_F_HAS_RET and ret_slot != _VOID:
        _set(frame, ret_slot, _u64(result if result is not None else 0))
        detail["ret"] = frame.regs[ret_slot]
    _emit(vm, frame, EventKind.IMPORT, ins, detail)
    return CONTINUE


def _builtin(vm, frame, fn_id, args):
    """The interpreter's own imports: ``message()`` and LRM-mandated errors."""
    if fn_id == BUILTIN_MESSAGE:
        entry = vm.model.messages[args[0]]
        vals = [frame.locals[s] for s in entry["slots"]]
        if vals[0] > vm.verbosity:
            return                    # 21.1.3: above the run's verbosity -> ignored
        try:
            line = format_message(entry["fmt"], entry["args"], vals[1:], vm.model.strings)
        except ValueError as e:
            raise VMError(f"message(): {e}")
        vm.out(line)
        return
    if fn_id == BUILTIN_ERROR:
        raise VMError(vm.model.strings[args[0]])
    if fn_id in _MEM_READ:
        return vm.memory.read(args[0], _MEM_READ[fn_id] // 8)
    if fn_id in _MEM_WRITE:
        vm.memory.write(args[0], _MEM_WRITE[fn_id] // 8, args[1])
        return None
    raise VMError(f"unknown builtin import 0x{fn_id:x}")


#: One shared blob-solver for problems that carry a serialized constraint system.
_blob_backend = None


def _op_solve(vm, frame, ins):
    pid = ins.args[0]
    problem = vm.model.problems[pid]

    if problem.seed_kind == "fixed":
        seed = problem.seed_value & ((1 << 64) - 1)
    else:
        seed = frame.seed.next_raw()

    # A problem carrying a dv-solve blob is solved by the real solver (honoring its
    # constraints); a blob-less problem uses the configured backend (minimal stub).
    if problem.problem_bytes:
        global _blob_backend
        if _blob_backend is None:
            from .extern import NativeBlobBackend
            _blob_backend = NativeBlobBackend()
        backend = _blob_backend
    else:
        backend = vm.solve_backend
    solved = backend.randomize(frame.obj, problem, seed)

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

    _emit(vm, frame, EventKind.SOLVE, ins, {"problem": pid, "seed": seed})
    return CONTINUE


def _op_solve_node(vm, frame, ins):
    """Solve the frame's node in its cone (P1-D2), or -- a node in no cone --
    as SOLVE would (P1-D3)."""
    act = frame.act
    if act is not None and frame.node in act.t.cone_of:
        seed = frame.seed.next_raw()
        act.solve(frame.node, frame.site, seed)
        slot = act.t.nodes[frame.node].comp_slot
        if slot is not None:
            # The solve chose the node's component instance (P1-D4).
            vm.set_comp(frame, frame.obj.get_field(slot))
        _emit(vm, frame, EventKind.SOLVE, ins,
              {"node": act.t.nodes[frame.node].path, "seed": seed})
        return CONTINUE
    if ins.args[0] != _VOID and vm.model.problems[ins.args[0]].var_names:
        return _op_solve(vm, frame, ins)
    if act is not None:
        act.commit(frame.node, frame.site)
    return CONTINUE


def _op_scope_enter(vm, frame, ins):
    """Entry to an activity block (13.4.8): reset what it traverses."""
    act = frame.act
    if act is not None:
        act.enter_scope(act.t.nodes[frame.node].scope_base + ins.imm)
    return CONTINUE


def _op_select(vm, frame, ins):
    from ..determinism import select_choice

    table = vm.model.selects[ins.args[0]]
    guards = table.guards or [-1] * len(table.branches)
    # A branch is eligible when it is unguarded (-1) or its guard register is set.
    eligible = [i for i in range(len(table.branches))
                if guards[i] < 0 or _get(frame, guards[i]) != 0]

    if not eligible:
        if table.allow_none:
            _emit(vm, frame, EventKind.SELECT, ins, {"choice": None, "none": True})
            return CONTINUE
        raise VMError("select has no eligible branch and allow_none is false")

    # Draw among the eligible branches in declaration order against their cumulative
    # weights, advancing the frame's own seed stream (determinism spec D§15.1).
    weights = [table.weights[i] for i in eligible]
    choice = eligible[select_choice(frame.seed, weights)]
    target = table.branches[choice]

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
    child = vm.call_child(frame, target)
    child.args, frame.staged = frame.staged, []
    child.ret_target = (frame, None if rd == 0xFFFFFFFF else rd)
    child.counted = True
    frame.pending += 1
    vm.sched.ready_first(child)
    return SUSPEND


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
