"""
context.py -- shared lowering state (module-level ``Lowerer`` + per-coro ``CoroCtx``).
"""

import dataclasses as dc
from typing import Any, Dict, List, Optional, Tuple

from zuspec.ir.core import scenario as SC

from ..model import (
    Instr, Op, ConstPool, SolveProblem, SelectTable, CoroDescriptor, LocalVar,
)
from ..abi.value import ScalarType
from ..abi.codec import is_inline
from .provenance import ProvenanceBuilder


class Lowerer:
    """Module-level lowering state shared by all coroutines in one image."""

    def __init__(self):
        #: COVG section: each sampled covergroup's JSON description (F4.4)
        self.covgs = []
        self.prov = ProvenanceBuilder()
        self.consts = ConstPool()
        self.problems: List[SolveProblem] = []
        self.selects: List[SelectTable] = []
        # fn_id -> ScImportDecl-like (blocking flag, arg/ret types), if provided.
        self.import_decls: Dict[int, object] = {}
        # import name -> {"fn_id", "blocking", "ret_type"} for resolving import
        # *call expressions* in procedural code (e.g. a non-blocking `getval(7)`).
        self.imports: Dict[str, dict] = {}
        # Coroutines accumulate here (branch sub-coroutines may be appended).
        self.coros: List[CoroDescriptor] = []
        self._coro_index: Dict[str, int] = {}
        # Synthesized branch sub-coroutines (PAR/SELECT). They are appended to the
        # model *after* the top-level coroutines, so their global index is
        # ``n_toplevel + position`` -- stable regardless of how many are created.
        self.n_toplevel: int = 0
        self.branch_coros: List[CoroDescriptor] = []
        # Names of sub-coroutines that block (so a synthesized branch lowers its
        # blocking ScInvoke suspends the same way the top-level coros do).
        self.blocking_targets: List[str] = []
        self._synth_seq: int = 0
        # Solve rand variables with dv-solve even when no constraint names them
        # (a PSS model's default); False keeps the blob-less stub path.
        self.solve_unconstrained: bool = False
        # Native PSS functions exec code may call (ScenarioModule.functions).
        self.functions: Dict[str, object] = {}
        # Layer-0 types by name (ScenarioModule.types): lays out a struct a
        # local or parameter is declared with.
        self.types: Dict[str, object] = {}
        #: component type layouts (ir-core ``comp_tree.CompLayouts``) when
        #: the module has a component tree (P1.5)
        self.comps = None
        #: the elaborated component tree (``ScComponentTree``), with comps
        self.comp_tree = None
        #: instance id -> the executor in force there (procedural._executors)
        self.executors = None
        self.exec_group_map = None   # exec_groups(): add_executor groups (F5.4b)
        #: each ``add_region`` of a contiguous space lowered: ``{inst, line,
        #: base}``, the opaque base it returns (F7.3e)
        self.opaque_regions: List[dict] = []
        #: the build's address regions, for get_tag: ``{path, k, addr,
        #: size, tag}`` (``addr`` None: the k-th region of contiguous space
        #: ``path``); None when the front end gives none
        self.regions: Optional[List[dict]] = None
        #: ``(group type, instance name, index or None) -> byte offset``: the
        #: group's own get_offset_of_instance[_array] (21.14.1), evaluated by
        #: the front end; None uses the type's dense ``offset_map``
        self.reg_offset = None
        #: action types with a node in a cone: their solve is SOLVE_NODE
        self.cone_types: set = set()
        #: some activation has a cone: activity blocks emit SCOPE_ENTER
        self.scoped: bool = False
        #: debug lowering (P1.4): every PSS statement starts a LINE entry
        self.debug: bool = False
        #: coroutine name -> the pc past its initial values (INSTR_F_INITED)
        self.init_end: Dict[str, int] = {}
        #: (INVOKE instr, target coroutine name) whose start pc is patched
        #: once every coroutine is lowered
        self.inited_invokes: List[tuple] = []
        self.strings: List[str] = []
        self._string_ids: Dict[str, int] = {}
        self.messages: List[dict] = []
        #: string ``==``/``!=`` instructions (code, pc, negate): plain index
        #: compares, made content compares if the model makes run-time strings
        self.string_cmps: List[tuple] = []
        self.uses_heap = False

    def intern_string(self, text: str) -> int:
        sid = self._string_ids.get(text)
        if sid is None:
            sid = len(self.strings)
            self.strings.append(text)
            self._string_ids[text] = sid
        return sid

    def add_message(self, fmt: str, args: List[dict], slots: List[int]) -> int:
        """Register one message() call site; checks 21.1.1 a) and b) statically.

        ``slots`` are the frame-local slots holding the verbosity and then each
        argument when the call executes.
        """
        from ..interp.fmt import parse_format
        specs = parse_format(fmt)                # raises ValueError on a bad '%'
        if len(specs) != len(args):
            raise ValueError(
                f"format has {len(specs)} specifier(s) but {len(args)} argument(s)")
        self.messages.append({"fmt": fmt, "args": list(args), "slots": list(slots)})
        return len(self.messages) - 1

    def add_coro(self, coro: CoroDescriptor) -> int:
        idx = len(self.coros)
        self.coros.append(coro)
        if coro.name:
            self._coro_index[coro.name] = idx
        return idx

    def add_branch_coro(self, coro: CoroDescriptor) -> int:
        """Append a synthesized branch coroutine; return its global model index."""
        idx = self.n_toplevel + len(self.branch_coros)
        self.branch_coros.append(coro)
        if coro.name:
            self._coro_index[coro.name] = idx
        return idx

    def next_synth_id(self) -> int:
        self._synth_seq += 1
        return self._synth_seq

    def add_problem(self, problem: SolveProblem) -> int:
        pid = len(self.problems)
        self.problems.append(problem)
        return pid

    def add_select(self, table: SelectTable) -> int:
        sid = len(self.selects)
        self.selects.append(table)
        return sid

    def const_id(self, value: int, width_bits: int) -> int:
        return self.consts.add(value, width_bits)


@dc.dataclass
class CoroCtx:
    """Per-coroutine lowering state."""

    lowerer: Lowerer
    code: List[Instr] = dc.field(default_factory=list)
    n_regs: int = 0
    local_slots: Dict[str, int] = dc.field(default_factory=dict)
    cur_src_ref: int = 0
    #: debug lowering: (pc, src_ref) of each statement's first instruction
    stmts: List[Tuple[int, int]] = dc.field(default_factory=list)
    #: debug lowering: each channel operation (``CoroDescriptor.chans``)
    chans: List[tuple] = dc.field(default_factory=list)
    #: debug lowering: each register access (``CoroDescriptor.regs``)
    regs: List[tuple] = dc.field(default_factory=list)
    #: debug lowering: the pc of each register read's ``IMPORT`` whose value
    #: the statement discards (``CoroDescriptor.discards``)
    discards: List[int] = dc.field(default_factory=list)
    #: debug lowering: coverage items (``CoroDescriptor.cov``)
    cov: List[tuple] = dc.field(default_factory=list)
    #: debug lowering: the structural path of the statement being lowered
    cov_path: List[str] = dc.field(default_factory=list)
    #: debug lowering: the function whose statements are being lowered
    #: (``<comp type>::<fn>``); "" = the coroutine's own
    cov_scope: str = ""
    #: debug lowering: exec blocks of each kind lowered so far (their paths)
    cov_execs: Dict[str, int] = dc.field(default_factory=dict)
    #: the call an expression statement is lowering, whose value is discarded
    discard_call: Any = None
    #: debug lowering: [name, slot name, pc_start, pc_end or None, type] of
    #: each PSS local (a struct: each leaf); pc_end is set when its scope ends
    named_locals: List[list] = dc.field(default_factory=list)
    coro_name: str = ""
    #: the source coroutine's attribute layout / action type, inherited by the
    #: PAR/SELECT branch sub-coroutines synthesized from it (they share its object)
    src_fields: list = dc.field(default_factory=list)
    action_type: Optional[str] = None
    #: procedural-lowering state (see procedural.ProcState)
    proc: object = None

    @classmethod
    def create(cls, lowerer: Lowerer, frame_locals: List[str],
               coro_name: str = "") -> "CoroCtx":
        slots = {name: i for i, name in enumerate(frame_locals)}
        return cls(lowerer=lowerer, local_slots=slots, coro_name=coro_name)

    # -- registers / locals ------------------------------------------------ #

    def new_reg(self) -> int:
        r = self.n_regs
        self.n_regs += 1
        return r

    def local_slot(self, name: str) -> int:
        if name not in self.local_slots:
            self.local_slots[name] = len(self.local_slots)
        return self.local_slots[name]

    def debug_locals(self) -> List[LocalVar]:
        """The named locals, a scope still open running to the end of the code."""
        return sorted((LocalVar(slot=self.local_slots[slot], name=name, pc_start=start,
                                pc_end=len(self.code) if end is None else end,
                                width=t.width, kind=t.kind, signed=t.signed)
                       for name, slot, start, end, t in self.named_locals),
                      key=lambda lv: (lv.pc_start, lv.slot))

    @property
    def frame_locals(self) -> List[str]:
        # slot order
        return [n for n, _ in sorted(self.local_slots.items(), key=lambda kv: kv[1])]

    # -- emit -------------------------------------------------------------- #

    def emit(self, op: Op, args=(), imm: int = 0, flags: int = 0,
             src_ref: Optional[int] = None) -> int:
        ins = Instr(
            op=op,
            args=tuple(args),
            imm=imm,
            flags=flags,
            src_ref=self.cur_src_ref if src_ref is None else src_ref,
        )
        self.code.append(ins)
        return len(self.code) - 1

    def scope_enter(self, scope: Optional[int]) -> None:
        """Entry to an activity block (P1.4): resets the handles traversed in
        it (13.4.8) and commits a branch's structure. Only a model with a
        cone needs it, so only such a model carries it."""
        if scope is not None and self.lowerer.scoped:
            self.emit(Op.SCOPE_ENTER, (), imm=int(scope))

    def emit_const(self, value: int, width_bits: int = 64) -> int:
        """Materialize an integer literal into a fresh register."""
        from ..model import INSTR_F_FROM_POOL

        r = self.new_reg()
        t = ScalarType(width_bits, signed=value < 0)
        if is_inline(t):
            self.emit(Op.CONST, (r,), imm=int(value) & ((1 << 64) - 1))
        else:
            cid = self.lowerer.const_id(value, width_bits)
            self.emit(Op.CONST, (r,), imm=cid, flags=INSTR_F_FROM_POOL)
        return r


def branch_inherit(ctx: CoroCtx) -> dict:
    """ScCoroutine kwargs a PAR/SELECT branch inherits from the coroutine it is
    carved out of: its attribute layout and action type (it runs on that
    action's object). ``fields`` only exists in a zuspec-ir-core that has
    ``ScField``; against an older one a branch simply gets no layout."""
    kw = {"action_type": ctx.action_type}
    if "fields" in getattr(SC.ScCoroutine, "__dataclass_fields__", {}):
        kw["fields"] = ctx.src_fields
    return kw
