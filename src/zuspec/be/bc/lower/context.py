"""
context.py -- shared lowering state (module-level ``Lowerer`` + per-coro ``CoroCtx``).
"""

import dataclasses as dc
from typing import Dict, List, Optional

from zuspec.ir.core import scenario as SC

from ..model import (
    Instr, Op, ConstPool, SolveProblem, SelectTable, CoroDescriptor,
)
from ..abi.value import ScalarType
from ..abi.codec import is_inline
from .provenance import ProvenanceBuilder


class Lowerer:
    """Module-level lowering state shared by all coroutines in one image."""

    def __init__(self):
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
        self.strings: List[str] = []
        self._string_ids: Dict[str, int] = {}
        self.messages: List[dict] = []

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
