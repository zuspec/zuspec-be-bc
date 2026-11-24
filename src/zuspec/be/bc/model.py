"""
model.py -- the in-memory ZBC image (P1-1).

This is the rich, Pythonic form that lowering (P1-2..P1-5) builds and the oracle
(P1-8..P1-13) executes. It serializes *through* the container format
(:mod:`.format.writer`) so the bytes the oracle round-trips are exactly the bytes
the C engine will read.

Two tiers of opcodes live in one flat instruction stream (design D§4):

* an **orchestration tier** (SPAWN/INVOKE/PAR/JOIN/WAIT/SELECT/SOLVE/BIND/YIELD/
  IMPORT) -- the suspend-capable ops that end FSM blocks; and
* a **procedural tier** (register-SSA compute) -- the straight-line work inside a
  block.

Structures:

* :class:`Instr`           -- one fixed-width instruction (op + up to 4 args + imm).
* :class:`Block`           -- one FSM block (a pc range ending at a suspend op).
* :class:`CoroDescriptor`  -- a coroutine/func: its code, blocks, and frame locals.
* :class:`SolveProblem`    -- an entry in the M1 in-memory problem table (D§1.2;
  not serialized in M1 -- SOLVE runs Python-side via the solver backend).
* :class:`ZbcModel`        -- the whole image; ``to_container`` / ``from_container``
  bridge to/from the serialized ``.zbc``.

Supporting tables (:class:`StringTable`, :class:`TypeTable`, :class:`ConstPool`)
are the interned string/type/constant pools. In M1 the string table carries coro
and frame-local *names* (codegen profile); a runtime profile would strip it.
"""

import dataclasses as dc
import enum
import json
import struct
from typing import Any, Dict, List, Optional, Tuple

from .abi.value import ABI_ID, ScalarType, ArrayType, StructType, ValueType
from .abi.codec import const_pool_record, decode_scalar
from .format import spec
from .format.writer import Section, ZbcImage, _fnv1a64, write_image
from .format.reader import read_image
from .format.emit_dataclass import CLASSES as REC

MASK64 = (1 << 64) - 1
_MAX_ARGS = 4


# --------------------------------------------------------------------------- #
# Opcodes
# --------------------------------------------------------------------------- #

class Op(enum.IntEnum):
    """ZBC opcodes. Values are stable and grouped by tier."""

    # control / terminal (0x00-0x0F)
    NOP = 0x00
    RET = 0x01

    # procedural: movement / memory (0x10-0x1F)
    CONST = 0x10       # arg0 = rd; imm = value (or const-pool id if flags & FROM_POOL)
    MOV = 0x11         # arg0 = rd; arg1 = rs
    LD_LOCAL = 0x12    # arg0 = rd; arg1 = frame-local slot
    ST_LOCAL = 0x13    # arg0 = rs; arg1 = frame-local slot
    LD_FIELD = 0x14    # arg0 = rd; arg1 = obj field slot
    ST_FIELD = 0x15    # arg0 = rs; arg1 = obj field slot
    # P1.5, component attributes (rt-eng refuses both until P8):
    LD_COMP = 0x16     # arg0 = rd; arg1 = component-object slot, from the frame's instance
    ST_COMP = 0x17     # arg0 = rs; arg1 = component-object slot, from the frame's instance
    # A called (recursive) function's arguments (see CALL):
    ARG = 0x18         # arg0 = rs; arg1 = i: stage argument i of the next CALL
    LD_ARG = 0x19      # arg0 = rd; arg1 = i: argument i of this frame's call
    LD_RES = 0x1A      # arg0 = rd; arg1 = i: result i of this frame's last CALL
                       # (the callee's ARGs staged before its RET; F6.4a)
    # The heap (F6.9c; docs/spec/heap.md): run-time strings and list/map/set.
    HEAP = 0x1B        # arg0 = rd; arg1..3 = a, b, c; imm: HEAP_* sub-op | kinds

    # procedural: arithmetic / logic (0x20-0x2F)
    ADD = 0x20
    SUB = 0x21
    MUL = 0x22
    DIV = 0x23
    MOD = 0x24
    AND = 0x25
    OR = 0x26
    XOR = 0x27
    SHL = 0x28
    SHR = 0x29
    NEG = 0x2A
    NOT = 0x2B
    # Floating point (F6.6): a register holds an IEEE binary64 bit pattern,
    # a float32 value one rounded to binary32 precision (FCVT imm 3).
    FARITH = 0x2C      # arg0 = rd; arg1 = a; arg2 = b; imm: FARITH_* (NEG: b unused)
    FCVT = 0x2D        # arg0 = rd; arg1 = rs; imm: FCVT_*
    FMATH = 0x2E       # arg0 = rd; arg1 = x; arg2 = y; imm: FMATH index (1-arg: y unused)

    # procedural: compare / branch (0x30-0x3F)
    CMP_EQ = 0x30
    CMP_NE = 0x31
    CMP_LT = 0x32
    CMP_LE = 0x33
    CMP_GT = 0x34
    CMP_GE = 0x35
    FCMP = 0x36        # arg0 = rd (0/1); arg1 = a; arg2 = b; imm: FCMP_*
    BR = 0x38          # arg0 = target instr index (code-absolute)
    BRZ = 0x39         # arg0 = cond reg; arg1 = target instr index (code-absolute)

    # orchestration tier (0x40-0x4F)
    SPAWN = 0x40
    INVOKE = 0x41
    PAR = 0x42
    JOIN = 0x43
    WAIT = 0x44
    SELECT = 0x45
    SOLVE = 0x46
    BIND = 0x47
    YIELD = 0x48
    IMPORT = 0x49      # arg0 = fn_id; arg1 = ret reg; arg2/arg3 = args, or imm = n
                       # > 0 args staged by ARG (as for CALL; consumed)
    # P1.4, the activation's solve scope (rt-eng refuses both until P8):
    SCOPE_ENTER = 0x4A  # imm = local activity-scope index (13.4.8 reset, P1-D2 commit)
    SOLVE_NODE = 0x4B   # arg0 = fallback problem id (0xFFFFFFFF: none): solve the
                        # frame's node in its cone, else as SOLVE (P1-D2/D3)
    # A native function that is recursive is called, not inlined (bc
    # procedural gaps B-D5). The callee runs at once, in the caller's object,
    # base and instance, with the staged arguments; the caller resumes when it
    # returns, with RET's value in rd (0xFFFFFFFF: none). It may suspend.
    CALL = 0x4C         # arg0 = callee coroutine; arg1 = rd
    # Procedural ``randomize`` (F3.7): SOLVE on a ``procedural`` problem keeps
    # its values on the frame instead of writing fields back; SOLVE_GET reads
    # one into a register, and ordinary stores put it in the target's leaves.
    SOLVE_PIN = 0x4D    # arg0 = var_id; arg1 = reg: pin an input of the frame's next solve (F3.7b)
    SOLVE_GET = 0x4E    # arg0 = rd; arg1 = var_id of the frame's last procedural solve
    # Functional coverage (F4.4): covergroup instance arg0 (an index into
    # :attr:`ZbcModel.covgs`) was sampled with the arg1 values staged by
    # ``ARG`` (as for CALL; consumed): per coverpoint its value, then its
    # ``iff`` when the COVG document says it has one; then one ``iff`` per
    # cross that has one.
    COV_SAMPLE = 0x4F   # arg0 = covg id; arg1 = n staged values

    # executor tier (0x50-0x5F; F5.4)
    #: arg0 = rs: the frame's executor becomes the value of rs (an EXEC row
    #: id, 0xFFFFFFFF = none); its requests, and frames it starts, carry it
    EXEC_SET = 0x50
    #: arg0 = rd: rd = the frame's executor (an EXEC row id; 0xFFFFFFFF =
    #: none, which ``executor() == null`` tests; F5.4c, 24.6.2.5)
    EXEC_GET = 0x51


#: Orchestration-tier opcodes (the suspend-capable / scheduler-driving ops).
ORCH_OPS = frozenset({
    Op.SPAWN, Op.INVOKE, Op.PAR, Op.JOIN, Op.WAIT,
    Op.SELECT, Op.SOLVE, Op.BIND, Op.YIELD, Op.IMPORT,
    Op.SCOPE_ENTER, Op.SOLVE_NODE, Op.CALL, Op.SOLVE_GET, Op.SOLVE_PIN,
    Op.COV_SAMPLE, Op.EXEC_SET, Op.EXEC_GET,
})

#: Ops that unconditionally end an FSM block by suspending (D§4.1, coro_fsm).
#: INVOKE/IMPORT suspend only when their blocking flag is set (see INSTR_F_BLOCKING).
UNCONDITIONAL_SUSPEND_OPS = frozenset({Op.WAIT, Op.JOIN, Op.PAR, Op.YIELD, Op.CALL})

#: The most arguments a CALL passes, and the deepest CALL nesting a run
#: allows (deeper is a run-time error, the same in both engines).
CALL_MAX_ARGS = 16
CALL_MAX_DEPTH = 1024

#: Instruction flag bits.
INSTR_F_FROM_POOL = 0x01   # CONST: imm is a const-pool id, not an inline value
INSTR_F_BLOCKING = 0x02    # INVOKE/IMPORT: this call suspends
INSTR_F_HAS_RET = 0x04     # IMPORT/INVOKE: arg1 is a result register
#: INVOKE: the callee is a node of the caller's activation (P1-D1): it runs on
#: the caller's object at ``imm`` slots past the caller's base; arg2 is the
#: traversal site's index among the caller type's sites.
INSTR_F_NODE = 0x08
#: INVOKE (with NODE): the caller has initialized the callee's attributes
#: (LRM 11.3.1 b i-ii); the callee starts at pc arg3, past its own initial
#: values.
INSTR_F_INITED = 0x10
#: YIELD: the frame waits for a condition (a blocking channel get/put spins
#: "while not ready: YIELD"). A frame that resumes after this YIELD and
#: reaches it again found the condition still false, so it made no progress;
#: when every ready frame is in that state, the run is deadlocked unless a
#: timed wait can change something. Temporary: an event wait replaces the spin.
INSTR_F_SPIN = 0x20
#: JOIN: wait for a subset of the arg0 children this block just spawned
#: (its branches, in order); the others run on, not cancelled. imm > 0:
#: ``join_select(imm)``, that many picked at random from the frame's seed
#: stream (LRM 13.3.6 c); imm == 0: ``join_branch``, the branches whose bit
#: is set in arg1 (bit i = branch i; at most 32 branches). The C engine
#: refuses it (M14).
INSTR_F_SUBSET = 0x40

#: ``FARITH`` imm: the binary64 operation (a float32 result is then rounded
#: by ``FCVT_F32``).
FARITH_ADD, FARITH_SUB, FARITH_MUL, FARITH_DIV, FARITH_NEG = range(5)
#: ``FCMP`` imm (IEEE: any compare with a NaN is false but ``NE``).
FCMP_EQ, FCMP_NE, FCMP_LT, FCMP_LE, FCMP_GT, FCMP_GE = range(6)
#: ``FCVT`` imm: signed / unsigned 64-bit integer to binary64 (nearest even);
#: binary64 to signed 64-bit, truncated toward zero (NaN or out of range:
#: 0x8000000000000000); binary64 rounded to binary32 precision (nearest even).
FCVT_S2F, FCVT_U2F, FCVT_F2S, FCVT_F32 = range(4)
#: ``FMATH`` imm: an index in FMATH_FNS. The PSS float library (LRM 24.5.2, Table 28),
#: binary64 in and out, with C libm semantics (``round`` halves away from 0).
FMATH_FNS = ("log", "log10", "exp", "sqrt", "pow", "round", "floor", "ceil",
         "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "hypot",
         "sinh", "cosh", "tanh", "asinh", "acosh", "atanh")
#: the ``FMATH`` functions of two arguments
FMATH_BINARY = frozenset(("pow", "atan2", "hypot"))

#: ``HEAP`` sub-ops (imm bits 0-7; docs/spec/heap.md section 6).
HEAP_OPS = ("SIZE", "GET", "SET", "INSERT", "DELETE", "PUSH_FRONT",
            "PUSH_BACK", "POP_FRONT", "POP_BACK", "CLEAR", "IN", "EQ", "COPY",
            "FREE", "KEYS", "VALUES", "TO_LIST", "TO_SET", "SHUFFLE", "JOIN",
            "STR_SLICE", "STR_GET", "STR_FIND", "STR_FIND_LAST",
            "STR_FIND_ALL", "STR_LOWER", "STR_UPPER", "STR_SPLIT", "STR_CHARS",
            "STR_FROM_CHARS", "STR_EQ", "FORMAT")
(HEAP_SIZE, HEAP_GET, HEAP_SET, HEAP_INSERT, HEAP_DELETE, HEAP_PUSH_FRONT,
 HEAP_PUSH_BACK, HEAP_POP_FRONT, HEAP_POP_BACK, HEAP_CLEAR, HEAP_IN, HEAP_EQ,
 HEAP_COPY, HEAP_FREE, HEAP_KEYS, HEAP_VALUES, HEAP_TO_LIST, HEAP_TO_SET,
 HEAP_SHUFFLE, HEAP_JOIN, HEAP_STR_SLICE, HEAP_STR_GET, HEAP_STR_FIND,
 HEAP_STR_FIND_LAST, HEAP_STR_FIND_ALL, HEAP_STR_LOWER, HEAP_STR_UPPER,
 HEAP_STR_SPLIT, HEAP_STR_CHARS, HEAP_STR_FROM_CHARS, HEAP_STR_EQ,
 HEAP_FORMAT) = range(len(HEAP_OPS))
#: ``HEAP`` imm bits 8-9: the operand's kind (an op on handle 0 allocates one)
HEAPK_STR, HEAPK_LIST, HEAPK_MAP, HEAPK_SET = range(4)
#: imm bits 10-11 (elements/values) and 12-13 (map keys): what a cell holds
HEAP_CELL_INT, HEAP_CELL_STR, HEAP_CELL_COLL = range(3)
#: imm bit 14 on ``STR_EQ``: the result is inverted (``!=``)
HEAP_NEG = 1 << 14


def heap_imm(sub: int, kind: int = 0, cell: int = 0, key: int = 0) -> int:
    """The ``HEAP`` imm of sub-op *sub* on a *kind* object whose cells (and
    map keys) hold *cell* (*key*) values."""
    return sub | kind << 8 | cell << 10 | key << 12

#: IMPORT fn_ids at and above this are interpreter builtins, not user imports.
#: A builtin is an ordinary IMPORT (no new opcode), so the ISA is unchanged.
BUILTIN_BASE = 0xFFFFFF00
#: ``message(verbosity, fmt, args...)``: args = (msg_idx,). The entry
#: ``ZbcModel.messages[msg_idx]`` names the frame-local slots holding the
#: verbosity and each value (an instruction has only 4 inline args).
BUILTIN_MESSAGE = BUILTIN_BASE + 0
#: A run-time error the LRM says "shall" be raised: args = (string_idx,).
BUILTIN_ERROR = BUILTIN_BASE + 1
#: A platform memory access with no executor override (LRM 21.13.9, bc
#: procedural gaps B-D6): ``read<N>`` args = (addr,), result the value, the
#: byte at addr in bits [7:0]; ``write<N>`` args = (addr, data).
BUILTIN_READ = {8: BUILTIN_BASE + 2, 16: BUILTIN_BASE + 3,
                32: BUILTIN_BASE + 4, 64: BUILTIN_BASE + 5}
BUILTIN_WRITE = {8: BUILTIN_BASE + 6, 16: BUILTIN_BASE + 7,
                 32: BUILTIN_BASE + 8, 64: BUILTIN_BASE + 9}
#: ``error(fmt, args...)`` / ``fatal(status, fmt, args...)`` (std_pkg): args =
#: (msg_idx,), a ``messages`` entry as for ``message()`` whose first slot holds
#: the status (0 for ``error``). The run stops with the formatted text (P1.5).
BUILTIN_ERROR_MSG = BUILTIN_BASE + 10
BUILTIN_FATAL_MSG = BUILTIN_BASE + 11
#: ``print(fmt, args...)`` (std_pkg, LRM 24.1): args = (msg_idx,), a
#: ``messages`` entry whose first slot holds 0; printed at any verbosity.
BUILTIN_PRINT = BUILTIN_BASE + 12
#: std_pkg file functions (LRM 24.2), served by the engine on the machine it
#: runs on. A handle (chandle) is a 1-based index into the run's file table,
#: 0 the ``nullfilehandle``; a name is a string-table index.
#: ``file_open`` args = (name, option) -> handle, 0 if it cannot be opened;
#: ``file_close`` args = (handle,); ``file_exists`` args = (name,) -> bool;
#: ``file_write`` args = (msg_idx,), a ``messages`` entry whose first slot
#: holds the handle. A bad handle stops the run.
BUILTIN_FILE_OPEN = BUILTIN_BASE + 13
BUILTIN_FILE_CLOSE = BUILTIN_BASE + 14
BUILTIN_FILE_EXISTS = BUILTIN_BASE + 15
BUILTIN_FILE_WRITE = BUILTIN_BASE + 16
#: std_pkg ``urandom()`` / ``urandom_range(min, max)`` (24.4): a draw from the
#: calling frame's seed stream (determinism.py ``urandom``), so a seed gives the
#: same numbers on every engine. urandom_range is inclusive; min > max swaps.
BUILTIN_URANDOM = BUILTIN_BASE + 17
BUILTIN_URANDOM_RANGE = BUILTIN_BASE + 18
#: std_pkg file functions that make or read heap values (24.2, F6.3b3):
#: ``file_read`` args = (handle, size) -> a new heap string (size -1: to the
#: end); ``file_read_lines`` args = (name,) -> a new list<string>;
#: ``file_write_lines`` args = (name, list, option). The C engine has no heap
#: and refuses an image that calls them, as one with HEAP ops (heap.md s.9).
BUILTIN_FILE_READ = BUILTIN_BASE + 19
BUILTIN_FILE_READ_LINES = BUILTIN_BASE + 20
BUILTIN_FILE_WRITE_LINES = BUILTIN_BASE + 21


def is_orchestration(op: Op) -> bool:
    return op in ORCH_OPS


# --------------------------------------------------------------------------- #
# Instruction
# --------------------------------------------------------------------------- #

@dc.dataclass
class Instr:
    op: Op
    args: Tuple[int, ...] = ()
    imm: int = 0
    flags: int = 0
    src_ref: int = 0

    def __post_init__(self):
        self.op = Op(self.op)
        self.args = tuple(int(a) for a in self.args)
        if len(self.args) > _MAX_ARGS:
            raise ValueError(f"{self.op.name}: at most {_MAX_ARGS} inline args")

    def is_suspend(self) -> bool:
        if self.op in UNCONDITIONAL_SUSPEND_OPS:
            return True
        if self.op in (Op.INVOKE, Op.IMPORT):
            return bool(self.flags & INSTR_F_BLOCKING)
        return False

    def to_record(self):
        rec = REC["zbc_instr"]()
        rec.op = int(self.op)
        rec.nargs = len(self.args)
        rec.flags = self.flags
        rec.src_ref = self.src_ref
        rec.imm = self.imm & MASK64
        a = list(self.args) + [0] * (_MAX_ARGS - len(self.args))
        rec.arg0, rec.arg1, rec.arg2, rec.arg3 = a[0], a[1], a[2], a[3]
        return rec

    @classmethod
    def from_record(cls, rec) -> "Instr":
        all_args = (rec.arg0, rec.arg1, rec.arg2, rec.arg3)
        return cls(
            op=Op(rec.op),
            args=tuple(all_args[: rec.nargs]),
            imm=rec.imm,
            flags=rec.flags,
            src_ref=rec.src_ref,
        )

    def __repr__(self):
        parts = [self.op.name]
        if self.args:
            parts.append("(" + ", ".join(str(a) for a in self.args) + ")")
        if self.imm:
            parts.append(f"imm={self.imm}")
        if self.flags:
            parts.append(f"flags=0x{self.flags:02x}")
        return "Instr[" + " ".join(parts) + "]"


# --------------------------------------------------------------------------- #
# FSM block + coroutine descriptor
# --------------------------------------------------------------------------- #

@dc.dataclass
class Block:
    idx: int
    pc_start: int           # instr index, relative to the coro's code
    pc_end: int             # one past last
    suspend_op: int = 0     # Op value ending the block (0 = terminal block)


#: zbc_local_kind by lowering type kind
LOCAL_KINDS = ("int", "bool", "enum", "string", "chandle", "opaque", "float")


@dc.dataclass(frozen=True)
class LocalVar:
    """A PSS local in debug lowering: its frame slot, its PSS name (a struct
    leaf is ``cfg.src``), the pcs where it is in scope (coroutine-local,
    ``[pc_start, pc_end)``) and its type."""
    slot: int
    name: str
    pc_start: int
    pc_end: int
    width: int
    kind: str = "int"
    signed: bool = False


@dc.dataclass
class CoroDescriptor:
    name: str = ""
    code: List[Instr] = dc.field(default_factory=list)
    blocks: List[Block] = dc.field(default_factory=list)
    frame_locals: List[str] = dc.field(default_factory=list)
    src_ref: int = 0
    #: debug lowering: (pc, src_ref) where each PSS statement starts (LINE)
    stmts: List[Tuple[int, int]] = dc.field(default_factory=list)
    #: debug lowering: the PSS locals, sorted by pc_start (LOCAL)
    locals: List[LocalVar] = dc.field(default_factory=list)
    #: debug lowering: a called form's function, ``<comp type>::<fn>`` (just
    #: ``<fn>`` outside a component). In memory only; the debug event hooks
    #: (``VM.on_check``) find functions by it.
    fn_qname: str = dc.field(default="", compare=False)
    #: debug lowering: ``(pc, src_ref, channel, op, value reg, ok reg)`` of
    #: each channel operation, at the NOP that ends it (regs: -1 = none). In
    #: memory only; the VM raises ``chan`` events from it.
    chans: List[tuple] = dc.field(default_factory=list, compare=False)
    #: debug lowering: ``(pc, inst)`` of each register access's ``IMPORT``:
    #: the register's component instance, from the frame's instance. In
    #: memory only; ``access`` events carry it as ``reg_id`` (P1.7).
    regs: List[tuple] = dc.field(default_factory=list, compare=False)
    #: debug lowering: the pc of each register read's ``IMPORT`` whose value
    #: the statement drops (``r.read_val();``). In memory only; its
    #: ``access`` event says ``discarded``.
    discards: List[int] = dc.field(default_factory=list, compare=False)
    #: debug lowering: the coverage items, ``(kind, pc, scope, path, src_ref,
    #: implicit)``. kind ``"stmt"``: a statement starts at pc; ``"arm"``: a
    #: branch arm (of an ``if``, if-chain or ``match``) is taken at pc, which
    #: no other path reaches; its path is ``<branch path>/<arm>`` (``then``,
    #: ``else``, a clause or case index, ``default``) and ``implicit`` is
    #: True for an ``else`` the code did not write. path is the statement's
    #: place in PSS structure (index in its block, ``/``-joined), so it does
    #: not move with lines; scope is the function it is in (``<comp
    #: type>::<fn>``), "" = the coroutine's own code. Serialized as
    #: ``SEC_COV`` (the C engine raises ``stmt``/``branch`` events from it).
    cov: List[tuple] = dc.field(default_factory=list, compare=False)

    def frame_type(self) -> Optional[StructType]:
        """The frame as a struct type (locals default to 64-bit slots in M1)."""
        if not self.frame_locals:
            return None
        return StructType(tuple((n, ScalarType(64)) for n in self.frame_locals))


# --------------------------------------------------------------------------- #
# M1 in-memory problem table (not serialized in M1; D§1.2)
# --------------------------------------------------------------------------- #

def blob_specs(blob: bytes):
    """The ``(width, signed)`` of each var id (None where no var is declared)
    and the ``(elem width, signed, rand)`` of each array id of a dv-solve
    problem blob (dvs_problem.h: header, then the pool its VarSpec and
    ArraySpec lists live in; offsets are from the pool)."""
    hdr = struct.unpack_from("<30I", blob, 0)
    n_vars, vars_head, n_arrays, arrays_head = hdr[2], hdr[5], hdr[16], hdr[17]
    pool, null = 26 * 4, 0xFFFFFFFF
    specs: list = [None] * n_vars
    off = vars_head
    while off != null:
        nxt, vid, width, signed = struct.unpack_from("<3IB", blob, pool + off)
        if vid < n_vars:
            specs[vid] = (width, bool(signed))
        off = nxt
    arrays: list = [None] * n_arrays
    off = arrays_head
    while off != null:
        nxt, aid, width, signed, kind = struct.unpack_from("<3I2B", blob, pool + off)
        arrays[aid] = (width, bool(signed), kind == 0)   # DVS_VAR_RAND
        off = nxt
    return specs, arrays


@dc.dataclass
class SolveProblem:
    """A SOLVE target: which fields to randomize and how results map back.

    In M1 the oracle resolves this by calling the solver backend on a live object;
    ``var_names`` (sorted -> var_id, per the value ABI) is the write-back key. This
    table is an in-memory side channel -- no SEC_SOLVE bytes until P4.
    """

    var_names: List[str] = dc.field(default_factory=list)
    type_ref: int = 0       # optional TYPE id of the randomized struct
    writeback: Dict[str, int] = dc.field(default_factory=dict)  # field name -> var_id (oracle)
    seed_kind: str = "inherit"   # "inherit" | "fixed"
    seed_value: int = 0
    #: Slot-keyed write-back (object field slot -> var_id) -- the serialized, native
    #: form of ``writeback``. The oracle resolves ``writeback`` field *names* against
    #: the live object; the native engine has only slots, so the SEC_SOLVE bytes carry
    #: this map. Populated by lowering from the object layout (M1: set explicitly).
    writeback_slots: Dict[int, int] = dc.field(default_factory=dict)
    #: Relocatable dv-solve ``SolveProblem`` blob (offset-based, self-contained). When
    #: non-empty it is emitted to the SPROB pool and referenced by the ``zbc_solve``
    #: record's (prob_off, prob_len); the native engine compiles + solves it with the
    #: drawn seed and writes ``dvs_solver_get_value(var_id)`` back per ``writeback_slots``.
    #: Empty (the M1 default) selects the minimal ``slot = seed + var_id`` randomizer.
    problem_bytes: bytes = b""
    #: The ``ScSolveProblem`` the blob was built from (in memory only; None
    #: when read from an image): a property test's shrink adds bounds to it.
    source: Optional[object] = dc.field(default=None, compare=False, repr=False)
    #: A procedural ``randomize`` (F3.7): nothing is written back; SOLVE keeps
    #: the solved values on the frame for SOLVE_GET (``ZBC_SOLVE_PROCEDURAL``;
    #: the pairs then list the kept var_ids).
    procedural: bool = False
    #: Non-rand fields a constraint reads (F3.7b2): object field slot ->
    #: var_id, each pinned to the field's value before the solve; serialized
    #: after the writeback pairs (``zbc_solve.n_input``).
    input_slots: Dict[int, int] = dc.field(default_factory=dict)
    #: Lists a constraint reads (F6.10d): object field slot (holding the
    #: list's heap handle) -> array id; each list's elements are passed per
    #: solve, so such a problem is solved by dv-solve's handle. A
    #: ``procedural`` problem's key is instead the SOLVE_PIN var id its list
    #: handle is pinned at (number of variables + array id). In memory
    #: only: an image cannot carry one yet.
    input_arrays: Dict[int, int] = dc.field(default_factory=dict)
    #: ``(elem width, signed)`` of each input array, by array id
    array_specs: List[tuple] = dc.field(default_factory=list)
    #: The array ids of ``rand`` lists (F6.10e): solved, as long as the list
    #: is, and their elements written back to it.
    rand_arrays: set = dc.field(default_factory=set)
    #: ``(width, signed)`` of each var id (None: no var), read from the
    #: blob of an image-loaded problem that reads lists (its ``source`` is
    #: None); None otherwise.
    var_specs: Optional[list] = dc.field(default=None, compare=False, repr=False)


@dc.dataclass
class SelectTable:
    """A weighted-choice SELECT target: parallel branch-coroutine indices + weights.

    Like :class:`SolveProblem` this is an in-memory M1 side channel (no serialized
    bytes yet). The VM first filters branches by their guard (``guards[i]`` is a
    register index whose live value must be non-zero, or ``-1`` for an unguarded
    branch), then draws among the eligible branches in **declaration order** against
    the cumulative weights (``determinism.select_choice``) using the frame's seed
    stream, and runs the chosen branch coroutine as a blocking child. If nothing is
    eligible, ``allow_none`` chooses between "run nothing" and a hard error.
    """

    branches: List[int] = dc.field(default_factory=list)   # branch coroutine indices
    weights: List[int] = dc.field(default_factory=list)     # matching positive weights
    guards: List[int] = dc.field(default_factory=list)      # guard reg per branch (-1 = none)
    allow_none: bool = False


# --------------------------------------------------------------------------- #
# Interned tables
# --------------------------------------------------------------------------- #

class StringTable:
    """Interned, NUL-terminated UTF-8 string table (design D§12.1)."""

    def __init__(self):
        self.blob = bytearray()
        self.offsets: List[int] = []
        self._map: Dict[str, int] = {}
        self.intern("")  # StrId 0 == "" at offset 0

    def intern(self, s: str) -> int:
        if s in self._map:
            return self._map[s]
        sid = len(self.offsets)
        self.offsets.append(len(self.blob))
        self.blob += s.encode("utf-8") + b"\x00"
        self._map[s] = sid
        return sid

    def get(self, sid: int) -> str:
        start = self.offsets[sid]
        end = self.blob.index(b"\x00", start)
        return self.blob[start:end].decode("utf-8")

    def strb_bytes(self) -> bytes:
        return bytes(self.blob)

    def stro_bytes(self) -> bytes:
        vals = list(self.offsets) + [len(self.blob)]
        return b"".join(v.to_bytes(4, "little") for v in vals)

    @classmethod
    def parse(cls, strb: bytes, stro: bytes) -> "StringTable":
        st = cls.__new__(cls)
        st.blob = bytearray(strb)
        n = len(stro) // 4 - 1  # last entry is the end offset
        st.offsets = [int.from_bytes(stro[i * 4:i * 4 + 4], "little") for i in range(n)]
        st._map = {}
        for sid in range(len(st.offsets)):
            st._map[st.get(sid)] = sid
        return st


class TypeTable:
    """Interned value types. Children are interned first (lower ids)."""

    #: type-descriptor kinds in the TYPE section
    _K_SCALAR, _K_ARRAY, _K_STRUCT = 0, 1, 2

    def __init__(self):
        self.types: List[ValueType] = []
        self._map: Dict[ValueType, int] = {}

    def intern(self, t: ValueType) -> int:
        if t in self._map:
            return self._map[t]
        if isinstance(t, ArrayType):
            self.intern(t.element)
        elif isinstance(t, StructType):
            for _, ft in t.fields:
                self.intern(ft)
        tid = len(self.types)
        self.types.append(t)
        self._map[t] = tid
        return tid

    def to_bytes(self, strtab: StringTable) -> bytes:
        out = bytearray()
        for t in self.types:
            if isinstance(t, ScalarType):
                out += bytes([self._K_SCALAR, 1 if t.signed else 0])
                out += t.width_bits.to_bytes(2, "little")
            elif isinstance(t, ArrayType):
                out += bytes([self._K_ARRAY, 0, 0, 0])
                out += self._map[t.element].to_bytes(4, "little")
                out += t.count.to_bytes(4, "little")
            elif isinstance(t, StructType):
                out += bytes([self._K_STRUCT, 0])
                out += len(t.fields).to_bytes(2, "little")
                for name, ft in t.fields:
                    out += strtab.intern(name).to_bytes(4, "little")
                    out += self._map[ft].to_bytes(4, "little")
            else:
                raise TypeError(f"cannot serialize type {type(t).__name__}")
        return bytes(out)

    @classmethod
    def parse(cls, data: bytes, count: int, strtab: StringTable) -> "TypeTable":
        tt = cls()
        pos = 0
        for _ in range(count):
            kind = data[pos]
            if kind == cls._K_SCALAR:
                signed = data[pos + 1] != 0
                width = int.from_bytes(data[pos + 2:pos + 4], "little")
                pos += 4
                t: ValueType = ScalarType(width, signed)
            elif kind == cls._K_ARRAY:
                elem = int.from_bytes(data[pos + 4:pos + 8], "little")
                cnt = int.from_bytes(data[pos + 8:pos + 12], "little")
                pos += 12
                t = ArrayType(tt.types[elem], cnt)
            elif kind == cls._K_STRUCT:
                nf = int.from_bytes(data[pos + 2:pos + 4], "little")
                pos += 4
                fields = []
                for _ in range(nf):
                    nid = int.from_bytes(data[pos:pos + 4], "little")
                    ftid = int.from_bytes(data[pos + 4:pos + 8], "little")
                    pos += 8
                    fields.append((strtab.get(nid), tt.types[ftid]))
                t = StructType(tuple(fields))
            else:
                raise ValueError(f"unknown TYPE kind {kind}")
            tid = len(tt.types)
            tt.types.append(t)
            tt._map[t] = tid
        return tt


@dc.dataclass
class ConstPool:
    """Pool of wide (>64-bit) literals. Stores raw bits + width; signedness is
    applied at the use site (from the operand type), not stored here."""

    entries: List[Tuple[int, int]] = dc.field(default_factory=list)  # (bits, width)

    def add(self, value: int, width_bits: int) -> int:
        t = ScalarType(width_bits, signed=False)
        bits = int(value) & t.mask
        cid = len(self.entries)
        self.entries.append((bits, width_bits))
        return cid

    def to_bytes(self) -> bytes:
        out = bytearray()
        for bits, width in self.entries:
            out += const_pool_record(bits, ScalarType(width, signed=False))
        return bytes(out)

    @classmethod
    def parse(cls, data: bytes) -> "ConstPool":
        pool = cls()
        pos = 0
        while pos < len(data):
            width = int.from_bytes(data[pos:pos + 4], "little")
            byte_len = int.from_bytes(data[pos + 4:pos + 8], "little")
            payload = data[pos + 8:pos + 8 + byte_len]
            bits = decode_scalar(payload, ScalarType(width, signed=False))
            pool.entries.append((bits, width))
            rec_len = 8 + byte_len
            pos += rec_len + ((-rec_len) % 8)
        return pool


# --------------------------------------------------------------------------- #
# Provenance tables (design D§4.3, §12.1)
# --------------------------------------------------------------------------- #

# Comment kinds (mirror spec.zbc_cmt_kind).
CMT_LEADING = 1
CMT_TRAILING = 2
CMT_INLINE = 3
CMT_DOC = 4


@dc.dataclass
class Comment:
    text: str
    kind: int = CMT_LEADING
    line: int = 0


@dc.dataclass
class Prov:
    """One provenance record: source span + original name + attached comments.

    ``src_ref`` on an :class:`Instr` / :class:`CoroDescriptor` is the index of the
    owning :class:`Prov` in :class:`ProvTable`. Index 0 is the reserved all-zero
    "none" entry.
    """

    name: str = ""
    node_kind: int = 0
    node_id: int = 0
    file: str = ""
    line: int = 0
    col: int = 0
    col_end: int = 0
    line_end: int = 0
    flags: int = 0
    comments: List[Comment] = dc.field(default_factory=list)


@dc.dataclass
class ProvTable:
    """The provenance table. ``entries[0]`` is the reserved "none" sentinel."""

    entries: List[Prov] = dc.field(default_factory=lambda: [Prov()])

    def add(self, prov: Prov) -> int:
        idx = len(self.entries)
        self.entries.append(prov)
        return idx


# --------------------------------------------------------------------------- #
# The image
# --------------------------------------------------------------------------- #

@dc.dataclass
class EntryPoint:
    """Where a host may enter the image (pssc P1.3): a coroutine, and for a
    function its parameters (name, type descriptor) and return descriptor.
    A function's arguments go in the root frame's ``args`` (``LD_ARG``); it
    runs in any instance of ``comp_type`` (or a type derived from it, when
    the function is not overridden there) its frame is put in. An action
    names its qualified type and lists its object's slots (name, type
    descriptor; ``"rand": True`` on a rand field) in slot order."""
    coro: int
    kind: str = "action"            # "action" | "function"
    params: List[Tuple[str, dict]] = dc.field(default_factory=list)
    returns: Optional[dict] = None
    comp_type: Optional[str] = None  # a function's component type
    qname: Optional[str] = None      # an action's qualified type
    obj: List[Tuple[str, dict]] = dc.field(default_factory=list)


def api_hash_of(entries: Dict[str, "EntryPoint"], imports: Dict[str, dict]) -> int:
    """FNV-1a/64 of the image's API (pssc P1.5c): each entry's symbol, kind,
    component type and signature, and each import's signature, as canonical
    JSON sorted by name. Coroutine indices are left out, so a body-only edit
    keeps the hash; a nonzero result (0 means unset)."""
    api = {
        "entries": [[sym, e.kind, e.comp_type, [[n, d] for n, d in e.params], e.returns]
                    for sym, e in sorted(entries.items())],
        "imports": [[name, {k: imports[name].get(k) for k in
                            ("fn_id", "blocking", "arg_types", "ret_type", "string_at",
                             "ret_types")
                            if k != "ret_types" or k in imports[name]}]
                    for name in sorted(imports)],
    }
    h = _fnv1a64(json.dumps(api, sort_keys=True, separators=(",", ":")).encode())
    return h or 1


@dc.dataclass
class ZbcModel:
    coros: List[CoroDescriptor] = dc.field(default_factory=list)
    entry_coro: int = 0
    consts: ConstPool = dc.field(default_factory=ConstPool)
    prov: ProvTable = dc.field(default_factory=ProvTable)
    # In-memory side channels (M1): not serialized, so excluded from == identity.
    problems: List[SolveProblem] = dc.field(default_factory=list, compare=False)
    selects: List[SelectTable] = dc.field(default_factory=list, compare=False)
    #: message() table: {"fmt": str, "args": [type descriptor]} per call site.
    messages: List[dict] = dc.field(default_factory=list, compare=False)
    #: string constants (message format args, string locals), by id.
    strings: List[str] = dc.field(default_factory=list, compare=False)
    #: coroutine index -> attribute names in slot order, for coroutines that are
    #: actions: each traversal of one gets its own object (not its parent's).
    obj_layouts: Dict[int, List[str]] = dc.field(default_factory=dict, compare=False)
    #: coroutine index -> its action tree's runtime table
    #: (``interp.activation.ActivationTable``), for an action run as the root
    #: of an activation (P1.4). In memory only until P7 specifies it.
    activations: Dict[int, Any] = dc.field(default_factory=dict, compare=False)
    #: coroutine index -> the table parsed from the ACT section (P4.10a):
    #: its constraints are None; its cones solve from the blobs ACT carries
    #: (P4.10b2). A model read from an image also runs them as
    #: :attr:`activations` (P4.10c1).
    image_activations: Dict[int, Any] = dc.field(default_factory=dict, compare=False)
    #: the component tree (``interp.components.CompTable``): the component
    #: object's layout, each instance's base, and the coroutine constructing
    #: it, run before the entry (P1.5). In memory only until P7 specifies it.
    components: Any = dc.field(default=None, compare=False)
    abi_id: int = ABI_ID
    profile: str = "codegen"
    #: a coroutine constructs the component tree before the entry runs
    #: (P1.5; header flag ``ZBC_HDR_COMP_INIT``). Which one is in
    #: :attr:`components`, in memory only until P7.
    comp_init: bool = False
    #: symbol -> :class:`EntryPoint`: each top-level action coroutine by its
    #: name, each requested function by ``<comp type>::<fn>`` (pssc P1.3).
    #: Serialized as the ENTRY section in the codegen profile (it needs the
    #: string table); excluded from == so a runtime image still compares.
    entries: Dict[str, EntryPoint] = dc.field(default_factory=dict, compare=False)
    #: path -> text of each PSS source file, so a debugger can show lines on
    #: a machine that has only the image. Serialized as the SRC section in the
    #: debug profile only (pssc P1.5b).
    sources: Dict[str, str] = dc.field(default_factory=dict, compare=False)
    #: name -> text of each host metadata document (pssc: the test catalog,
    #: the register map), opaque to bc. Serialized as the META section in
    #: the debug profile only (pssc P3.2).
    meta: Dict[str, str] = dc.field(default_factory=dict, compare=False)
    #: name -> signature of each user import the lowering was given:
    #: ``{"fn_id", "blocking", "arg_types", "ret_type", "string_at"}``. In
    #: memory only; it feeds :attr:`api_hash` (see :func:`api_hash_of`).
    imports: Dict[str, dict] = dc.field(default_factory=dict, compare=False)
    #: fn_id -> name of each user import, as read back from an image's
    #: IMPORT section (the signatures are not in the image, so a read-back
    #: :attr:`imports` stays empty). Serialized as IMPORT, from
    #: :attr:`imports` when that is set, in the codegen and debug profiles.
    import_names: Dict[int, str] = dc.field(default_factory=dict, compare=False)
    #: FNV-1a/64 of every entry and import signature (0 = unset); the
    #: lowering sets it. With :attr:`generation` it is the IDENT section, which
    #: only the debug profile writes, and only when one of them is set (P1.5c).
    api_hash: int = dc.field(default=0, compare=False)
    #: caller-supplied build generation (0 = unset); never derived, so a
    #: build that leaves it unset stays reproducible.
    generation: int = dc.field(default=0, compare=False)
    #: one WAIT TIME tick is ``10**time_exp`` s (IDENT; -12 = ps, also when
    #: the image has no IDENT). A simulator host converts to its own unit.
    time_exp: int = dc.field(default=-12, compare=False)
    #: reg-id (component instance index) -> register: ``{path, comp, reg,
    #: type, comp_type, offset, width, access, fields}``, as the host (pssc)
    #: builds it; opaque to bc. Serialized as the REGMAP section (JSON) in
    #: the codegen and debug profiles, when set (F1.7b).
    registers: Dict[int, dict] = dc.field(default_factory=dict, compare=False)
    #: address spaces and their regions: ``{"spaces": [...], "regions":
    #: [...]}``, as the host (pssc) builds it; opaque to bc. Serialized as
    #: the MEMMAP section (JSON) in the codegen and debug profiles, when it
    #: has spaces (F7.2c).
    memmap: Dict = dc.field(default_factory=dict, compare=False)
    #: the opaque base each ``add_region`` of a contiguous space returns:
    #: ``{inst, line, base}`` (``inst`` the space's component instance), for
    #: the host to put in ``memmap`` (F7.3e); not serialized
    opaque_regions: List[dict] = dc.field(default_factory=list, compare=False)
    #: each coroutine's seed key (keyed seeds, F3.9; ``determinism.coro_seed_keys``):
    #: written to SEEDKEY, so a loaded image, which keeps no layout for a
    #: non-entry action and no names in the runtime profile, keys as built
    seed_keys: Optional[List[int]] = dc.field(default=None, compare=False)
    #: each covergroup instance a ``COV_SAMPLE`` names, by covg id: its
    #: coverpoints, bins and crosses as plain data (F4.4; JSON in the COVG
    #: section, codegen and debug profiles).
    covgs: List[dict] = dc.field(default_factory=list, compare=False)
    #: F5.4a: each executor instance, by executor id (``EXEC_SET``'s value,
    #: ``Request.executor``): ``{"path", "type", "trait"}``; and each executor
    #: group: ``{"path", "trait", "members": [executor id, ...]}`` (EXEC section,
    #: codegen and debug profiles)
    executors: List[dict] = dc.field(default_factory=list, compare=False)
    exec_groups: List[dict] = dc.field(default_factory=list, compare=False)
    #: F5.5a: each host-fed channel, by event id (``notify(N_CHAN, id, v)``):
    #: ``{"path", "kind", "width", "signed", "count", "head", "buf", "depth"}``
    #: (component slots; EVENT section, codegen and debug profiles)
    events: List[dict] = dc.field(default_factory=list, compare=False)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_container(self) -> ZbcImage:
        strtab = StringTable()
        types = TypeTable()

        code_bytes = bytearray()
        coro_recs = bytearray()
        block_bytes = bytearray()
        n_instrs = 0
        n_blocks = 0
        line_bytes = bytearray()
        n_lines = 0
        local_bytes = bytearray()
        n_locals = 0
        reg_bytes = bytearray()
        n_regs = 0
        chan_bytes = bytearray()
        n_chans = 0
        cov_bytes = bytearray()
        n_covs = 0

        for ci, coro in enumerate(self.coros):
            code_start = n_instrs
            for lv in coro.locals:
                rec = REC["zbc_localent"]()
                rec.coro, rec.slot = ci, lv.slot
                rec.name = strtab.intern(lv.name)
                rec.pc_start = code_start + lv.pc_start
                rec.pc_end = code_start + lv.pc_end
                rec.width = lv.width
                rec.kind = LOCAL_KINDS.index(lv.kind)
                rec.flags = (spec.flag_value("zbc_local_flags", "ZBC_LOCAL_F_SIGNED")
                             if lv.signed else 0)
                local_bytes += rec.to_bytes()
            n_locals += len(coro.locals)
            for pc, sr in coro.stmts:
                lrec = REC["zbc_lineent"]()
                lrec.pc_start = code_start + pc
                lrec.src_ref = sr
                line_bytes += lrec.to_bytes()
            n_lines += len(coro.stmts)
            # Debug side tables (P4.4a): the C engine reads them for events.
            regs = dict(coro.regs or ())
            discards = set(coro.discards or ())
            none = 0xFFFFFFFF
            for pc in sorted(set(regs) | discards):
                rrec = REC["zbc_regent"]()
                rrec.pc = code_start + pc
                rrec.inst = regs.get(pc, none)
                rrec.flags = (spec.flag_value("zbc_reg_flags", "ZBC_REG_F_DISCARD")
                              if pc in discards else 0)
                reg_bytes += rrec.to_bytes()
                n_regs += 1
            for pc, sr, name, op, val, ok in sorted(coro.chans or ()):
                crec = REC["zbc_chanent"]()
                crec.pc, crec.src_ref = code_start + pc, sr
                crec.channel, crec.op = strtab.intern(name), strtab.intern(op)
                crec.val = none if val < 0 else val
                crec.ok = none if ok < 0 else ok
                chan_bytes += crec.to_bytes()
                n_chans += 1
            f_arm = spec.flag_value("zbc_cov_flags", "ZBC_COV_F_ARM")
            f_imp = spec.flag_value("zbc_cov_flags", "ZBC_COV_F_IMPLICIT")
            for kind, pc, scope, path, sr, implicit in sorted(
                    coro.cov or (), key=lambda c: c[1]):
                vrec = REC["zbc_covent"]()
                vrec.pc, vrec.src_ref = code_start + pc, sr
                vrec.flags = ((f_arm if kind == "arm" else 0)
                              | (f_imp if implicit else 0))
                vrec.scope = strtab.intern(scope or "")
                vrec.path = strtab.intern(path)
                cov_bytes += vrec.to_bytes()
                n_covs += 1
            for ins in coro.code:
                code_bytes += ins.to_record().to_bytes()
            n_instrs += len(coro.code)

            block_start = n_blocks
            for blk in coro.blocks:
                brec = REC["zbc_block"]()
                brec.idx = blk.idx
                brec.pc_start = blk.pc_start
                brec.pc_end = blk.pc_end
                brec.suspend_op = blk.suspend_op
                block_bytes += brec.to_bytes()
            n_blocks += len(coro.blocks)

            ftype = coro.frame_type()
            frame_type_id = (types.intern(ftype) + 1) if ftype is not None else 0

            crec = REC["zbc_coro"]()
            crec.name = strtab.intern(coro.name)
            crec.code_start = code_start
            crec.code_count = len(coro.code)
            crec.n_blocks = len(coro.blocks)
            crec.block_start = block_start
            crec.frame_type = frame_type_id  # 1-based; 0 = none
            crec.src_ref = coro.src_ref
            coro_recs += crec.to_bytes()

        sections: List[Section] = []
        instr_size = REC["zbc_instr"].size()
        coro_size = REC["zbc_coro"].size()
        block_size = REC["zbc_block"].size()

        sections.append(Section(spec.sec_kind("ZBC_SEC_CODE"), bytes(code_bytes),
                                count=n_instrs, elem_size=instr_size))
        sections.append(Section(spec.sec_kind("ZBC_SEC_CORO"), bytes(coro_recs),
                                count=len(self.coros), elem_size=coro_size))
        sections.append(Section(spec.sec_kind("ZBC_SEC_BLOCK"), bytes(block_bytes),
                                count=n_blocks, elem_size=block_size))

        const_bytes = self.consts.to_bytes()
        if const_bytes:
            sections.append(Section(spec.sec_kind("ZBC_SEC_CONST"), const_bytes,
                                    count=len(self.consts.entries)))

        # SELECT + SOLVE tables. Formerly M1-only in-memory side channels; now
        # serialized so the native engine can execute them with the *same*
        # determinism stream as the oracle. Both keep their variable-length operands
        # in one shared OPLIST u32 pool (each descriptor records its own offset).
        # Present in both profiles (execution data, not provenance).
        oplist_b = bytearray()
        sprob_b = bytearray()
        select_b = self._serialize_selects(oplist_b) if self.selects else b""
        solve_b = self._serialize_solves(oplist_b, sprob_b) if self.problems else b""
        if oplist_b:
            sections.append(Section(spec.sec_kind("ZBC_SEC_OPLIST"), bytes(oplist_b),
                                    count=len(oplist_b) // 4, elem_size=4))
        if self.selects:
            sections.append(Section(spec.sec_kind("ZBC_SEC_SELECT"), select_b,
                                    count=len(self.selects),
                                    elem_size=REC["zbc_select"].size()))
        if self.problems:
            sections.append(Section(spec.sec_kind("ZBC_SEC_SOLVE"), solve_b,
                                    count=len(self.problems),
                                    elem_size=REC["zbc_solve"].size()))
        if sprob_b:
            sections.append(Section(spec.sec_kind("ZBC_SEC_SPROB"), bytes(sprob_b),
                                    count=len(sprob_b), elem_size=1))

        # Provenance-family sections (codegen and debug profiles): FILE/PROV/CMNT
        # then TYPE + string table (STRB/STRO must be last -- all interning done
        # by then). The debug profile also carries the PSS sources (SRC).
        # Keyed seeds (F3.9): execution data, in every profile.
        keys = self.coro_seed_keys()
        if keys:
            sections.append(Section(spec.sec_kind("ZBC_SEC_SEEDKEY"),
                                    struct.pack("<%dQ" % len(keys), *keys),
                                    count=len(keys), elem_size=8))
        if self.profile in ("codegen", "debug"):
            file_bytes, prov_bytes, cmnt_bytes, n_files, n_prov, n_cmnt = \
                self._serialize_prov(strtab)
            sections.append(Section(spec.sec_kind("ZBC_SEC_FILE"), file_bytes,
                                    count=n_files, elem_size=REC["zbc_file"].size()))
            sections.append(Section(spec.sec_kind("ZBC_SEC_PROV"), prov_bytes,
                                    count=n_prov, elem_size=REC["zbc_prov"].size()))
            sections.append(Section(spec.sec_kind("ZBC_SEC_CMNT"), cmnt_bytes,
                                    count=n_cmnt, elem_size=REC["zbc_comment"].size()))
            if n_lines:
                sections.append(Section(
                    spec.sec_kind("ZBC_SEC_LINE"), bytes(line_bytes),
                    flags=spec.flag_value("zbc_secf_flags", "ZBC_SECF_LINE_STMT"),
                    count=n_lines, elem_size=REC["zbc_lineent"].size()))
            if n_locals:
                sections.append(Section(
                    spec.sec_kind("ZBC_SEC_LOCAL"), bytes(local_bytes),
                    count=n_locals, elem_size=REC["zbc_localent"].size()))
            if n_regs:
                sections.append(Section(
                    spec.sec_kind("ZBC_SEC_REG"), bytes(reg_bytes),
                    count=n_regs, elem_size=REC["zbc_regent"].size()))
            if n_chans:
                sections.append(Section(
                    spec.sec_kind("ZBC_SEC_CHAN"), bytes(chan_bytes),
                    count=n_chans, elem_size=REC["zbc_chanent"].size()))
            if n_covs:
                sections.append(Section(
                    spec.sec_kind("ZBC_SEC_COV"), bytes(cov_bytes),
                    count=n_covs, elem_size=REC["zbc_covent"].size()))
            entry_bytes = self._serialize_entries(strtab)
            if entry_bytes:
                sections.append(Section(spec.sec_kind("ZBC_SEC_ENTRY"), entry_bytes,
                                        count=len(self.entries),
                                        elem_size=REC["zbc_entry"].size()))
            lay_b = self._serialize_layout(strtab)
            if lay_b:
                sections.append(Section(spec.sec_kind("ZBC_SEC_LAYOUT"), lay_b,
                                        count=1, elem_size=REC["zbc_layout"].size()))
            if self.registers:
                rrec = REC["zbc_regmap"]()
                rrec.doc = strtab.intern(json.dumps(
                    {str(k): v for k, v in sorted(self.registers.items())},
                    sort_keys=True, separators=(",", ":")))
                sections.append(Section(spec.sec_kind("ZBC_SEC_REGMAP"), rrec.to_bytes(),
                                        count=1, elem_size=REC["zbc_regmap"].size()))
            if self.memmap.get("spaces"):
                mrec = REC["zbc_memmap"]()
                mrec.doc = strtab.intern(json.dumps(self.memmap, sort_keys=True,
                                                    separators=(",", ":")))
                sections.append(Section(spec.sec_kind("ZBC_SEC_MEMMAP"), mrec.to_bytes(),
                                        count=1, elem_size=REC["zbc_memmap"].size()))
            if self.covgs:
                crec = REC["zbc_covg"]()
                crec.doc = strtab.intern(json.dumps(self.covgs, sort_keys=True,
                                                    separators=(",", ":")))
                sections.append(Section(spec.sec_kind("ZBC_SEC_COVG"), crec.to_bytes(),
                                        count=1, elem_size=REC["zbc_covg"].size()))
            if self.executors or self.exec_groups:
                sections.append(Section(spec.sec_kind("ZBC_SEC_EXEC"),
                                        self._serialize_exec(strtab), count=1,
                                        elem_size=REC["zbc_exectab"].size()))
            if self.events:
                sections.append(Section(spec.sec_kind("ZBC_SEC_EVENT"),
                                        self._serialize_events(strtab), count=len(self.events),
                                        elem_size=REC["zbc_evtent"].size()))
            names = ({sig["fn_id"]: n for n, sig in self.imports.items()}
                     if self.imports else self.import_names)
            if names:
                imp_b = bytearray()
                for fn_id in sorted(names):
                    irec = REC["zbc_impent"]()
                    irec.fn_id = fn_id
                    irec.name = strtab.intern(names[fn_id])
                    imp_b += irec.to_bytes()
                sections.append(Section(spec.sec_kind("ZBC_SEC_IMPORT"), bytes(imp_b),
                                        count=len(names),
                                        elem_size=REC["zbc_impent"].size()))
            if self.profile == "debug" and self.sources:
                src_b = bytearray()
                for path in sorted(self.sources):
                    srec = REC["zbc_srcent"]()
                    srec.path = strtab.intern(path)
                    srec.text = strtab.intern(self.sources[path])
                    src_b += srec.to_bytes()
                sections.append(Section(spec.sec_kind("ZBC_SEC_SRC"), bytes(src_b),
                                        count=len(self.sources),
                                        elem_size=REC["zbc_srcent"].size()))
            if self.profile == "debug" and self.meta:
                meta_b = bytearray()
                for name in sorted(self.meta):
                    mrec = REC["zbc_metaent"]()
                    mrec.name = strtab.intern(name)
                    mrec.text = strtab.intern(self.meta[name])
                    meta_b += mrec.to_bytes()
                sections.append(Section(spec.sec_kind("ZBC_SEC_META"), bytes(meta_b),
                                        count=len(self.meta),
                                        elem_size=REC["zbc_metaent"].size()))
            if self.profile == "debug" and (self.api_hash or self.generation
                                            or self.time_exp != -12):
                irec = REC["zbc_ident"]()
                irec.api_hash = self.api_hash
                irec.generation = self.generation
                irec.time_exp = self.time_exp & 0xFFFFFFFF
                sections.append(Section(spec.sec_kind("ZBC_SEC_IDENT"), irec.to_bytes(),
                                        count=1, elem_size=REC["zbc_ident"].size()))
            type_bytes = types.to_bytes(strtab)
            if type_bytes:
                sections.append(Section(spec.sec_kind("ZBC_SEC_TYPE"), type_bytes,
                                        count=len(types.types)))
            sections.append(Section(spec.sec_kind("ZBC_SEC_STRB"), strtab.strb_bytes()))
            sections.append(Section(spec.sec_kind("ZBC_SEC_STRO"), strtab.stro_bytes()))

        comp_b = self._serialize_comp()
        if comp_b:
            sections.append(Section(spec.sec_kind("ZBC_SEC_COMP"), comp_b,
                                    count=1, elem_size=REC["zbc_comp"].size()))

        msg_b = self._serialize_messages()
        if msg_b:
            sections.append(Section(spec.sec_kind("ZBC_SEC_MSG"), msg_b,
                                    count=1, elem_size=REC["zbc_msgtab"].size()))

        act_b = self._serialize_activations()
        if act_b:
            sections.append(Section(spec.sec_kind("ZBC_SEC_ACT"), act_b,
                                    count=1, elem_size=REC["zbc_acttab"].size()))

        return ZbcImage(
            sections=sections,
            abi_id=self.abi_id,
            entry_coro=self.entry_coro,
            profile=self.profile,
            comp_init=self.comp_init,
        )

    def _serialize_comp(self) -> bytes:
        """COMP: the component object's size, its constructing coroutine
        and each instance's base, so an engine without :attr:`components`
        (the native one) can run component code."""
        comps = self.components
        if comps is None:
            return b""
        rec = REC["zbc_comp"]()
        rec.n_slots = len(comps.names)
        rec.init_coro = 0xFFFFFFFF if comps.init_coro is None else comps.init_coro
        rec.n_inst = len(comps.bases)
        rec.flags = getattr(comps, "comp_flags", 0)
        if getattr(comps, "run_hooks", False):
            rec.flags |= spec.flag_value("zbc_comp_flags", "ZBC_COMP_F_RUN_HOOKS")
        if any(n.comp_rel != 0 or n.comp_slot is not None
               for t in self.activations.values() for n in t.nodes):
            rec.flags |= spec.flag_value("zbc_comp_flags", "ZBC_COMP_F_NODE_INST")
        return rec.to_bytes() + struct.pack(f"<{len(comps.bases)}I", *comps.bases)

    def _serialize_layout(self, strtab: "StringTable") -> bytes:
        """LAYOUT: the component object's slot names and type descriptors
        and each instance's path and type, so a debugger reading only the
        image names component fields (F1.7a)."""
        comps = self.components
        if comps is None or not comps.names:
            return b""
        opaque = {"kind": "opaque", "width": 0, "signed": False}
        tys = list(getattr(comps, "slot_types", None) or [])
        tys += [opaque] * (len(comps.names) - len(tys))
        rec = REC["zbc_layout"]()
        rec.root = strtab.intern(getattr(comps, "root", "") or "")
        rec.slots = strtab.intern(json.dumps([[n, d] for n, d in zip(comps.names, tys)],
                                             sort_keys=True, separators=(",", ":")))
        rec.insts = strtab.intern(json.dumps([[p, t] for p, t in zip(comps.paths, comps.types)],
                                             separators=(",", ":")))
        return rec.to_bytes()

    _NONE = 0xFFFFFFFF
    _CON_KINDS = ("type", "activity", "with", "comp", "infer")

    def _serialize_activations(self) -> bytes:
        """ACT: each activation's tree (nodes, scopes, sites, cones, pools),
        slot names and initial values, so an engine without
        :attr:`activations` (the native one) can run activities. A
        constraint's expression is not here: per cone, a dv-solve blob of its
        variables and one per constraint (with the constraint's text) hold it. A
        model read from an image writes back the tables it read."""
        acts = self.activations or getattr(self, "image_activations", None) or {}
        if not acts:
            return b""
        from zuspec.ir.core import scenario as SC
        N = self._NONE
        kinds = list(SC.ScopeKind)
        strings, sidx, lst, blobs = [], {}, [], bytearray()
        arrs = {k: bytearray() for k in ("act", "node", "scope", "site", "cone", "var", "con",
                                         "pool", "claim", "write", "pick", "init")}
        cnt = dict.fromkeys(arrs, 0)

        def st(x):
            if x is None:
                return N
            if x not in sidx:
                sidx[x] = len(strings)
                strings.append(x)
            return sidx[x]

        def opt(x):
            return N if x is None else x

        def ul(xs):
            i = len(lst)
            lst.extend(xs)
            return i, len(xs)

        def add(kind, rec):
            arrs[kind] += rec.to_bytes()
            cnt[kind] += 1
            return cnt[kind] - 1

        def pairs(ps):
            return ul([v for pr in ps for v in pr])

        def bl(b):
            blobs.extend(b"\0" * (-len(blobs) % 8))
            off = len(blobs)
            blobs.extend(b)
            return off, len(b)

        def write(w):
            r = REC["zbc_actwrite"]()
            r.ref = st(w.ref)
            r.src0, r.n_src = ul(w.src)
            r.pool0, n = pairs(w.pools)
            r.n_pools = n // 2
            return add("write", r)

        for coro in sorted(acts):
            t = acts[coro]
            tree = t.tree
            a = REC["zbc_act"]()
            a.coro, a.size, a.root, a.type = coro, tree.size, st(tree.root), st(tree.type_qname)
            a.flags = 1 if tree.lookahead else 0
            a.node0, a.n_nodes = cnt["node"], len(tree.nodes)
            for i, n in enumerate(tree.nodes):
                assert n.id == i, (n.id, i)
                r = REC["zbc_actnode"]()
                r.path, r.type, r.base, r.size = st(n.path), st(n.type_qname), n.base, n.size
                r.parent, r.decl_scope, r.comp_slot = opt(n.parent), opt(n.decl_scope), opt(n.comp_slot)
                r.comp0, r.n_comp = ul(n.comp)
                r.child0, r.n_children = ul(n.children)
                ac = getattr(n, "addr_claims", None) or ()
                if ac:
                    if not lst:
                        lst.append(0)   # index 0 means "none"
                    # n quads, then m contiguous claims (F7.3e): index in
                    # the quads, alignment slot, ``$addr`` slot (F7.4b), k,
                    # then k (trait, slot) pairs
                    cg = getattr(n, "addr_contig", None) or ()
                    r.addr0, _ = ul([len(ac)] + [v for sp, a, sz, pm in ac
                                                 for v in (st(sp), opt(a), sz, opt(pm))]
                                    + [len(cg)] + [v for i, al, tr, ad in cg
                                                   for v in [i, opt(al), opt(ad), len(tr)]
                                                   + [w for f, sl in tr for w in (st(f), sl)]])
                r.claim0, r.n_claims = cnt["claim"], len(n.claims)
                for c in n.claims:
                    cr = REC["zbc_actclaim"]()
                    cr.ref, cr.lock, cr.iid_slot = st(c.ref), int(bool(c.lock)), c.iid_slot
                    cr.pool0, k = pairs(c.pools)
                    cr.n_pools = k // 2
                    add("claim", cr)
                r.swrite0, r.n_swrites = cnt["write"], len(n.state_writes)
                for w in n.state_writes:
                    write(w)
                r.bwrite0, r.n_bwrites = cnt["write"], len(n.buffer_writes)
                for w in n.buffer_writes:
                    write(w)
                inf = getattr(n, "inferred_for", None)
                r.infer0 = N if inf is None else ul([inf[0], st(inf[1]), inf[2]])[0]
                si = getattr(n, "state_infer", None) or ()
                sm = getattr(n, "stream_infer", None) or ()
                if si or sm:
                    if not lst:
                        lst.append(0)   # index 0 means "none"
                    r.sinfer0, _ = ul([v for grp in (si, sm)
                                       for v in [len(grp)] + [
                                           v for ref, inn in grp
                                           for v in [st(ref), len(inn)]
                                           + [w for i, ns in inn
                                              for w in [i, len(ns)] + list(ns)]]])
                r.pick0, r.n_picks = cnt["pick"], len(n.picks)
                for pk in n.picks:
                    pr = REC["zbc_actpick"]()
                    pr.ref, pr.sel_slot = st(pk.ref), pk.sel_slot
                    pr.slot0, pr.n_slots = ul(pk.slots)
                    pr.pool0, k = pairs(pk.pools)
                    pr.n_pools = k // 2
                    inn = getattr(pk, "infer_nodes", None) or ()
                    pr.infer0, _ = ul([v for i, ns in inn for v in [i, len(ns)] + list(ns)])
                    pr.n_infer = len(inn)
                    add("pick", pr)
                add("node", r)
            a.scope0, a.n_scopes = cnt["scope"], len(tree.scopes)
            for i, sc in enumerate(tree.scopes):
                assert sc.id == i, (sc.id, i)
                r = REC["zbc_actscope"]()
                r.kind, r.parent, r.node = kinds.index(sc.kind), opt(sc.parent), sc.node
                add("scope", r)
            a.site0, a.n_sites = cnt["site"], len(tree.sites)
            for i, x in enumerate(tree.sites):
                assert x.id == i, (x.id, i)
                r = REC["zbc_actsite"]()
                r.owner, r.target, r.scope = x.owner, x.target, x.scope
                add("site", r)
            a.cone0, a.n_cones = cnt["cone"], len(tree.cones)
            for c in tree.cones:
                r = REC["zbc_actcone"]()
                r.id = c.id
                r.node0, r.n_nodes = ul(c.nodes)
                r.var0, r.n_vars = cnt["var"], len(c.vars)
                for v in c.vars:
                    vr = REC["zbc_actvar"]()
                    vr.name, vr.node, vr.slot, vr.width = st(v.name), v.node, v.slot, v.width
                    vr.flags = ((1 if v.signed else 0) | (2 if v.rand else 0) | (4 if v.busy_lock else 0)
                                | (8 if v.array else 0) | (v.busy_word << 16))
                    vr.comp_read, vr.loop_local = opt(v.comp_read), st(v.loop_local)
                    vr.loop_node, vr.live_from = opt(v.loop_node), opt(v.live_from)
                    vr.busy_pool, vr.pick = opt(v.busy_pool), opt(v.pick)
                    if v.attr_pool is not None:
                        # (an engine without F8.6e reads a root variable)
                        vr.flags |= 16 | (v.attr_inst << 16)
                        vr.pick = v.attr_pool
                    vr.loop0 = N
                    if v.loop_local is not None:
                        # Where an engine without frame-local names finds the
                        # counter (pyvm _loop_counter): (coro, slot) per coro.
                        hs = [(ci, co.frame_locals.index(v.loop_local))
                              for ci, co in enumerate(getattr(self, "coros", ()))
                              if v.loop_local in co.frame_locals]
                        vr.loop0 = ul([len(hs)] + [x for h in hs for x in h])[0]
                    add("var", vr)
                base, cblobs, texts = t.cone_image(c.id)
                r.base_off, r.base_size = bl(base)
                r.con0, r.n_cons = cnt["con"], len(c.constraints)
                for k, kb, kt in zip(c.constraints, cblobs, texts):
                    kr = REC["zbc_actcon"]()
                    kr.kind = self._CON_KINDS.index(k.kind.value)
                    kr.owner, kr.scope, kr.site = k.owner, opt(k.scope), opt(k.site)
                    kr.node0, kr.n_nodes = ul(k.nodes)
                    kr.blob_off, kr.blob_size = bl(kb)
                    kr.text = st(kt)
                    add("con", kr)
                add("cone", r)
            pools = getattr(tree, "pools", None) or []
            a.pool0, a.n_pools = cnt["pool"], len(pools)
            for p in pools:
                r = REC["zbc_actpool"]()
                r.id, r.inst, r.name, r.kind, r.size = p.id, p.inst, st(p.name), st(p.kind), opt(p.size)
                r.slot0, r.n_slots = ul(p.slots)
                r.leaf0, r.n_leaves = ul([st(x) for x in p.leaves])
                r.init0, r.n_init = ul(p.init_slots)
                add("pool", r)
            a.names0, a.n_names = ul([st(x) for x in t.names])
            a.init0, a.n_init = cnt["init"], len(t.init_values) + len(getattr(t, "uid_slots", ()))
            from .interp.activation import UID
            for slot, val in sorted(list(t.init_values.items())
                                    + [(x, UID) for x in getattr(t, "uid_slots", ())]):
                r = REC["zbc_actinit"]()
                # unknown: 1 computed at traversal, 2 a node's uid
                r.slot, r.unknown = slot, 2 if val is UID else 1 if val is None else 0
                val = None if val is UID else val
                r.value = 0 if val is None else val & 0xFFFFFFFFFFFFFFFF
                add("init", r)
            add("act", a)
        strb, offs = bytearray(), []
        for x in strings:
            offs.append(len(strb))
            strb += x.encode() + b"\0"
        hdr = REC["zbc_acttab"]()
        for k in ("act", "node", "scope", "site", "cone", "var", "con", "pool", "claim",
                  "write", "pick", "init"):
            setattr(hdr, "n_%ss" % k if k != "init" else "n_init", cnt[k])
        body = bytes(arrs["init"])
        for k in ("act", "node", "scope", "site", "cone", "var", "con", "pool", "claim",
                  "write", "pick"):
            body += bytes(arrs[k])
        body += struct.pack(f"<{len(lst)}I", *lst) + struct.pack(f"<{len(offs)}I", *offs)
        # the blob area starts 8-aligned in the payload
        strb += b"\0" * (-(REC["zbc_acttab"].size() + len(body) + len(strb)) % 8)
        hdr.n_u32, hdr.n_strings, hdr.strb_size = len(lst), len(strings), len(strb)
        hdr.blob_size = len(blobs)
        return hdr.to_bytes() + body + bytes(strb) + bytes(blobs)

    @classmethod
    def _parse_activations(cls, sec: Optional[Section]) -> Dict[int, Any]:
        """ACT back to coroutine index -> ``ActivationTable``, each
        constraint's ``constraint`` None: the table solves from the cone
        blobs and constraint text it reads (see :meth:`_serialize_activations`)."""
        if sec is None:
            return {}
        from zuspec.ir.core import scenario as SC
        from .interp.activation import UID, ActivationTable
        N = cls._NONE
        kinds = list(SC.ScopeKind)
        p = sec.payload
        hdr = REC["zbc_acttab"].from_bytes(p)
        pos = REC["zbc_acttab"].size()

        def recs(name, n):
            nonlocal pos
            r, size = REC[name], REC[name].size()
            out = [r.from_bytes(p[pos + i * size:pos + (i + 1) * size]) for i in range(n)]
            pos += n * size
            return out
        inits = recs("zbc_actinit", hdr.n_init)
        acts = recs("zbc_act", hdr.n_acts)
        nodes = recs("zbc_actnode", hdr.n_nodes)
        scopes = recs("zbc_actscope", hdr.n_scopes)
        sites = recs("zbc_actsite", hdr.n_sites)
        cones = recs("zbc_actcone", hdr.n_cones)
        vars_ = recs("zbc_actvar", hdr.n_vars)
        cons = recs("zbc_actcon", hdr.n_cons)
        pools = recs("zbc_actpool", hdr.n_pools)
        claims = recs("zbc_actclaim", hdr.n_claims)
        writes = recs("zbc_actwrite", hdr.n_writes)
        picks = recs("zbc_actpick", hdr.n_picks)
        lst = list(struct.unpack_from(f"<{hdr.n_u32}I", p, pos))
        pos += 4 * hdr.n_u32
        offs = struct.unpack_from(f"<{hdr.n_strings}I", p, pos)
        pos += 4 * hdr.n_strings
        strb = p[pos:pos + hdr.strb_size]
        strings = [strb[o:strb.index(b"\0", o)].decode() for o in offs]
        pos += hdr.strb_size
        blobs = bytes(p[pos:pos + hdr.blob_size])

        def blob(off, size):
            return blobs[off:off + size]

        def opt(x):
            return None if x == N else x

        def ost(x):
            return None if x == N else strings[x]

        def ul(i, n):
            return lst[i:i + n]

        def pairs(i, n):
            xs = ul(i, 2 * n)
            return [(xs[j], xs[j + 1]) for j in range(0, len(xs), 2)]

        def contig(i):
            # The contiguous claims after a node's address quads (F7.3e).
            if not i:
                return []
            i += 1 + 4 * lst[i]
            out = []
            for _ in range(lst[i] if i < len(lst) else 0):
                idx, al, ad, k = lst[i + 1:i + 5]
                tr = ul(i + 5, 2 * k)
                out.append((idx, opt(al), tuple((strings[tr[j]], tr[j + 1])
                                                for j in range(0, 2 * k, 2)), opt(ad)))
                i += 4 + 2 * k
            return out

        def infer_nodes(i, n):
            out = []
            for _ in range(n):
                k = lst[i + 1]
                out.append((lst[i], ul(i + 2, k)))
                i += 2 + k
            return out

        def state_infer(i):
            # state writers, then stream partners (M9a6b): each a count
            # then its groups
            out = ([], [])
            if not i:
                return out
            for grp in out:
                n, i = lst[i], i + 1
                for _ in range(n):
                    ref, m = strings[lst[i]], lst[i + 1]
                    inn = infer_nodes(i + 2, m)
                    grp.append((ref, inn))
                    i += 2 + sum(2 + len(ns) for _, ns in inn)
            return out

        def write(w):
            return SC.ScStateWrite(ref=strings[w.ref], src=ul(w.src0, w.n_src),
                                   pools=pairs(w.pool0, w.n_pools))
        out = {}
        for a in acts:
            tnodes = []
            for i, r in enumerate(nodes[a.node0:a.node0 + a.n_nodes]):
                tnodes.append(SC.ScActionNode(
                    id=i, path=strings[r.path], type_qname=strings[r.type], base=r.base,
                    size=r.size, parent=opt(r.parent), decl_scope=opt(r.decl_scope),
                    children=ul(r.child0, r.n_children), comp=ul(r.comp0, r.n_comp),
                    comp_slot=opt(r.comp_slot),
                    addr_claims=[(strings[q[0]], opt(q[1]), q[2], opt(q[3])) for q in
                                 (ul(r.addr0 + 1 + 4 * j, 4)
                                  for j in range(lst[r.addr0] if r.addr0 else 0))],
                    addr_contig=contig(r.addr0),
                    claims=[SC.ScClaim(ref=strings[c.ref], lock=bool(c.lock), iid_slot=c.iid_slot,
                                       pools=pairs(c.pool0, c.n_pools))
                            for c in claims[r.claim0:r.claim0 + r.n_claims]],
                    state_writes=[write(w) for w in writes[r.swrite0:r.swrite0 + r.n_swrites]],
                    buffer_writes=[write(w) for w in writes[r.bwrite0:r.bwrite0 + r.n_bwrites]],
                    picks=[SC.ScBufferPick(ref=strings[k.ref], slots=ul(k.slot0, k.n_slots),
                                           sel_slot=k.sel_slot, pools=pairs(k.pool0, k.n_pools),
                                           infer_nodes=infer_nodes(k.infer0, k.n_infer))
                           for k in picks[r.pick0:r.pick0 + r.n_picks]],
                    state_infer=state_infer(r.sinfer0)[0],
                    stream_infer=state_infer(r.sinfer0)[1],
                    inferred_for=None if r.infer0 == N else (
                        lst[r.infer0], strings[lst[r.infer0 + 1]], lst[r.infer0 + 2])))
            tcones = []
            for c in cones[a.cone0:a.cone0 + a.n_cones]:
                tcones.append(SC.ScScopeProblem(
                    id=c.id, nodes=ul(c.node0, c.n_nodes),
                    vars=[SC.ScScopeVar(name=strings[v.name], node=v.node, slot=v.slot,
                                        width=v.width, signed=bool(v.flags & 1),
                                        rand=bool(v.flags & 2), busy_lock=bool(v.flags & 4),
                                        array=bool(v.flags & 8),
                                        comp_read=opt(v.comp_read), loop_local=ost(v.loop_local),
                                        loop_node=opt(v.loop_node), live_from=opt(v.live_from),
                                        busy_pool=opt(v.busy_pool),
                                        busy_word=0 if v.flags & 16 else v.flags >> 16,
                                        pick=None if v.flags & 16 else opt(v.pick),
                                        attr_pool=opt(v.pick) if v.flags & 16 else None,
                                        attr_inst=v.flags >> 16 if v.flags & 16 else 0)
                          for v in vars_[c.var0:c.var0 + c.n_vars]],
                    constraints=[SC.ScScopeConstraint(
                        constraint=None, nodes=ul(k.node0, k.n_nodes),
                        kind=SC.ScopeConstraintKind(cls._CON_KINDS[k.kind]), owner=k.owner,
                        scope=opt(k.scope), site=opt(k.site))
                        for k in cons[c.con0:c.con0 + c.n_cons]]))
            tree = SC.ScActionTree(
                root=strings[a.root], type_qname=strings[a.type], size=a.size, nodes=tnodes,
                scopes=[SC.ScActivityScope(id=i, kind=kinds[r.kind], parent=opt(r.parent), node=r.node)
                        for i, r in enumerate(scopes[a.scope0:a.scope0 + a.n_scopes])],
                sites=[SC.ScTraversalSite(id=i, owner=r.owner, target=r.target, scope=r.scope)
                       for i, r in enumerate(sites[a.site0:a.site0 + a.n_sites])],
                cones=tcones,
                pools=[SC.ScPool(id=r.id, inst=r.inst, name=strings[r.name], kind=strings[r.kind],
                                 size=opt(r.size), slots=ul(r.slot0, r.n_slots),
                                 leaves=[strings[x] for x in ul(r.leaf0, r.n_leaves)],
                                 init_slots=ul(r.init0, r.n_init))
                       for r in pools[a.pool0:a.pool0 + a.n_pools]],
                lookahead=bool(a.flags & 1))
            init = {}
            for r in inits[a.init0:a.init0 + a.n_init]:
                v = r.value - (1 << 64) if r.value >> 63 else r.value
                init[r.slot] = UID if r.unknown == 2 else None if r.unknown else v
            t = out[a.coro] = ActivationTable(tree, [strings[x] for x in ul(a.names0, a.n_names)], init)
            for c in cones[a.cone0:a.cone0 + a.n_cones]:
                ks = cons[c.con0:c.con0 + c.n_cons]
                t.load_cone(c.id, blob(c.base_off, c.base_size),
                            [blob(k.blob_off, k.blob_size) for k in ks], [strings[k.text] for k in ks])
        return out

    _MSG_KINDS = ("int", "bool", "enum", "string", "chandle", "float")

    def _serialize_messages(self) -> bytes:
        """MSG: :attr:`messages` and :attr:`strings`, so an engine without
        them (the native one) can format message()/error()/fatal() text."""
        if not self.messages and not self.strings:
            return b""
        strings = list(self.strings)
        msgs, args, items, slots = bytearray(), bytearray(), bytearray(), []
        n_args = n_items = 0
        for m in self.messages:
            rec = REC["zbc_msg"]()
            rec.fmt = len(strings)
            strings.append(m["fmt"])
            rec.n_args, rec.arg0, rec.slot0 = len(m["args"]), n_args, len(slots)
            msgs += rec.to_bytes()
            assert len(m["slots"]) == len(m["args"]) + 1, m
            slots += m["slots"]
            for d in m["args"]:
                arec = REC["zbc_msgarg"]()
                arec.kind = self._MSG_KINDS.index(d.get("kind", "int"))
                arec.is_signed = 1 if d.get("signed") else 0
                arec.width = int(d.get("width", 64))
                arec.item0, arec.n_items = n_items, len(d.get("items") or ())
                args += arec.to_bytes()
                n_args += 1
                for name, val in d.get("items") or ():
                    irec = REC["zbc_msgitem"]()
                    irec.name = len(strings)
                    strings.append(name)
                    irec.value = val & 0xFFFFFFFFFFFFFFFF
                    items += irec.to_bytes()
                    n_items += 1
        strb, offs = bytearray(), []
        for s in strings:
            offs.append(len(strb))
            strb += s.encode() + b"\0"
        hdr = REC["zbc_msgtab"]()
        hdr.n_msgs, hdr.n_args, hdr.n_items = len(self.messages), n_args, n_items
        hdr.n_slots, hdr.n_strings = len(slots), len(strings)
        hdr.n_model_strings, hdr.strb_size = len(self.strings), len(strb)
        return (hdr.to_bytes() + bytes(msgs) + bytes(args) + bytes(items)
                + struct.pack(f"<{len(slots)}I", *slots)
                + struct.pack(f"<{len(offs)}I", *offs) + bytes(strb))

    @classmethod
    def _parse_messages(cls, sec: Optional[Section]):
        """MSG back to ``(messages, strings)``."""
        if sec is None:
            return [], []
        p = sec.payload
        hdr = REC["zbc_msgtab"].from_bytes(p)
        pos = REC["zbc_msgtab"].size()

        def recs(name, n):
            nonlocal pos
            r, size = REC[name], REC[name].size()
            out = [r.from_bytes(p[pos + i * size:pos + (i + 1) * size]) for i in range(n)]
            pos += n * size
            return out
        msgs = recs("zbc_msg", hdr.n_msgs)
        args = recs("zbc_msgarg", hdr.n_args)
        items = recs("zbc_msgitem", hdr.n_items)
        slots = list(struct.unpack_from(f"<{hdr.n_slots}I", p, pos))
        pos += 4 * hdr.n_slots
        offs = struct.unpack_from(f"<{hdr.n_strings}I", p, pos)
        pos += 4 * hdr.n_strings
        strb = p[pos:pos + hdr.strb_size]
        strings = [strb[o:strb.index(b"\0", o)].decode() for o in offs]

        def desc(a):
            d = {"kind": cls._MSG_KINDS[a.kind], "width": a.width, "signed": bool(a.is_signed)}
            if a.kind == cls._MSG_KINDS.index("enum"):
                d["items"] = [[strings[it.name],
                               it.value - (1 << 64) if it.value >> 63 else it.value]
                              for it in items[a.item0:a.item0 + a.n_items]]
            return d
        messages = [{"fmt": strings[m.fmt],
                     "args": [desc(a) for a in args[m.arg0:m.arg0 + m.n_args]],
                     "slots": slots[m.slot0:m.slot0 + m.n_args + 1]} for m in msgs]
        return messages, strings[:hdr.n_model_strings]

    def _serialize_prov(self, strtab: StringTable):
        """Serialize FILE/PROV/CMNT. Returns (file_b, prov_b, cmnt_b, nf, np, nc)."""
        file_ids: Dict[str, int] = {"": 0}
        file_paths: List[str] = [""]  # FileId 0 = none

        def file_id(path: str) -> int:
            if path not in file_ids:
                file_ids[path] = len(file_paths)
                file_paths.append(path)
            return file_ids[path]

        cmnt_bytes = bytearray()
        prov_bytes = bytearray()
        n_cmnt = 0
        for p in self.prov.entries:
            cmt_first = n_cmnt
            for c in p.comments:
                crec = REC["zbc_comment"]()
                crec.text = strtab.intern(c.text)
                crec.kind = c.kind
                crec.line = c.line
                cmnt_bytes += crec.to_bytes()
                n_cmnt += 1
            prec = REC["zbc_prov"]()
            prec.name = strtab.intern(p.name)
            prec.node_id = p.node_id
            prec.line = p.line
            prec.cmt_first = cmt_first if p.comments else 0
            prec.file = file_id(p.file)
            prec.node_kind = p.node_kind
            prec.col = p.col
            prec.col_end = p.col_end
            prec.line_end = p.line_end
            prec.flags = p.flags
            prec.cmt_count = len(p.comments)
            prov_bytes += prec.to_bytes()

        file_bytes = bytearray()
        for path in file_paths:
            frec = REC["zbc_file"]()
            frec.path = strtab.intern(path)
            file_bytes += frec.to_bytes()

        return (bytes(file_bytes), bytes(prov_bytes), bytes(cmnt_bytes),
                len(file_paths), len(self.prov.entries), n_cmnt)

    #: Guard sentinel: an unguarded SELECT branch (matches SelectTable's -1).
    _SEL_NOGUARD = 0xFFFFFFFF

    def _serialize_selects(self, oplist: bytearray) -> bytes:
        """Serialize the SELECT descriptors, appending operands to the shared pool.

        Each descriptor points at three contiguous u32 runs (from its recorded
        offset): branch coro ids, positive weights, then guard registers. An empty
        ``guards`` list (no guards) is canonicalized to an all-``_SEL_NOGUARD`` run,
        so both the empty and explicit forms round-trip to identical bytes.
        """
        allow_none = spec.flag_value("zbc_sel_flags", "ZBC_SEL_ALLOW_NONE")
        select_b = bytearray()
        for tbl in self.selects:
            n = len(tbl.branches)
            guards = tbl.guards if tbl.guards else [-1] * n
            srec = REC["zbc_select"]()
            srec.oplist_off = len(oplist) // 4
            srec.n_branches = n
            srec.flags = allow_none if tbl.allow_none else 0
            select_b += srec.to_bytes()
            for v in tbl.branches:
                oplist += int(v).to_bytes(4, "little")
            for v in tbl.weights:
                oplist += int(v).to_bytes(4, "little")
            for g in guards:
                oplist += (self._SEL_NOGUARD if g < 0 else int(g) & 0xFFFFFFFF) \
                    .to_bytes(4, "little")
        return bytes(select_b)

    def _serialize_solves(self, oplist: bytearray, sprob: bytearray) -> bytes:
        """Serialize the SOLVE descriptors, appending writeback pairs to the pool.

        Each descriptor records ``n_writeback`` interleaved (field_slot, var_id) u32
        pairs (sorted by slot for determinism) plus the seed policy. Only the
        slot-keyed ``writeback_slots`` is serialized -- the name-keyed ``writeback``
        and ``var_names`` are the oracle's solver-backend keys, reattached in memory.

        A non-empty ``problem_bytes`` (relocatable dv-solve blob) is appended to the
        shared ``sprob`` pool at a 4-byte-aligned offset, and its (offset, length)
        recorded in ``prob_off``/``prob_len`` for the native real-solve path.
        """
        seed_fixed = spec.flag_value("zbc_solve_flags", "ZBC_SOLVE_SEED_FIXED")
        procedural = spec.flag_value("zbc_solve_flags", "ZBC_SOLVE_PROCEDURAL")
        solve_b = bytearray()
        for p in self.problems:
            pairs = sorted(p.writeback_slots.items())
            if p.procedural:
                # Kept, not written back: solved slot k = var_id (F3.7c).
                pairs = [(v, v) for v in range(len(p.var_names))]
            srec = REC["zbc_solve"]()
            srec.seed_value = (p.seed_value & MASK64) if p.seed_kind == "fixed" else 0
            srec.oplist_off = len(oplist) // 4
            srec.n_writeback = len(pairs)
            inputs = sorted(p.input_slots.items())
            if p.input_arrays:
                # The lists a problem reads (F6.10d/e): more input pairs,
                # (key, n_vars + array id); widths and rand come from the blob.
                n_vars = len(blob_specs(p.problem_bytes)[0])
                inputs += sorted((k, n_vars + a) for k, a in p.input_arrays.items())
            srec.n_input = len(inputs)
            srec.flags = ((seed_fixed if p.seed_kind == "fixed" else 0)
                          | (procedural if p.procedural else 0)
                          | (spec.flag_value("zbc_solve_flags", "ZBC_SOLVE_ARRAYS")
                             if p.input_arrays else 0))
            if p.problem_bytes:
                # 4-align so the blob's leading SolveProblem header stays aligned when
                # the engine hands its address straight to dvs_solver_compile().
                while len(sprob) % 4:
                    sprob += b"\x00"
                srec.prob_off = len(sprob)
                srec.prob_len = len(p.problem_bytes)
                sprob += p.problem_bytes
            solve_b += srec.to_bytes()
            for slot, var_id in pairs + inputs:
                oplist += int(slot).to_bytes(4, "little")
                oplist += int(var_id).to_bytes(4, "little")
        return bytes(solve_b)

    @classmethod
    def _parse_solves(cls, solve_sec, oplist_sec, sprob_sec) -> List["SolveProblem"]:
        if solve_sec is None:
            return []
        seed_fixed = spec.flag_value("zbc_solve_flags", "ZBC_SOLVE_SEED_FIXED")
        op = oplist_sec.payload if oplist_sec is not None else b""
        sprob = sprob_sec.payload if sprob_sec is not None else b""

        def u32(i: int) -> int:
            return int.from_bytes(op[i * 4:i * 4 + 4], "little")

        ssz = REC["zbc_solve"].size()
        out: List[SolveProblem] = []
        for i in range(solve_sec.count):
            srec = REC["zbc_solve"].from_bytes(solve_sec.payload[i * ssz:])
            fixed = bool(srec.flags & seed_fixed)
            proc = bool(srec.flags & spec.flag_value("zbc_solve_flags",
                                                     "ZBC_SOLVE_PROCEDURAL"))
            base = srec.oplist_off
            wb = {u32(base + 2 * j): u32(base + 2 * j + 1)
                  for j in range(srec.n_writeback)}
            kept = sorted(wb.values()) if proc else []
            if proc:
                wb = {}
            ins = {u32(base + 2 * j): u32(base + 2 * j + 1)
                   for j in range(srec.n_writeback, srec.n_writeback + srec.n_input)}
            blob = (bytes(sprob[srec.prob_off:srec.prob_off + srec.prob_len])
                    if srec.prob_len else b"")
            arrays, specs, aspecs, rand = {}, None, [], set()
            if srec.flags & spec.flag_value("zbc_solve_flags", "ZBC_SOLVE_ARRAYS"):
                specs, alist = blob_specs(blob)
                arrays = {k: v - len(specs) for k, v in ins.items() if v >= len(specs)}
                ins = {k: v for k, v in ins.items() if v < len(specs)}
                aspecs = [(w, sg) for w, sg, _ in alist]
                rand = {a for a, (_, _, r) in enumerate(alist) if r}
            # The oracle's names are not in the image: name each var by its
            # id, so pyvm's NativeBlobBackend writes back a loaded image's
            # solves (in simulation, on the Python host) as the C engine does.
            names = [""] * (max([*wb.values(), *ins.values(), *kept], default=-1) + 1)
            for v in [*wb.values(), *ins.values(), *kept]:
                names[v] = f"${v}"
            out.append(SolveProblem(
                var_names=names,
                writeback={f"${v}": v for v in sorted(set(wb.values()))},
                writeback_slots=wb,
                input_slots=ins,
                seed_kind="fixed" if fixed else "inherit",
                seed_value=srec.seed_value if fixed else 0,
                problem_bytes=blob,
                procedural=proc,
                input_arrays=arrays,
                array_specs=aspecs,
                rand_arrays=rand,
                var_specs=specs,
            ))
        return out

    @classmethod
    def _parse_selects(cls, select_sec, oplist_sec) -> List["SelectTable"]:
        if select_sec is None or oplist_sec is None:
            return []
        allow_none = spec.flag_value("zbc_sel_flags", "ZBC_SEL_ALLOW_NONE")
        op = oplist_sec.payload

        def u32(i: int) -> int:
            return int.from_bytes(op[i * 4:i * 4 + 4], "little")

        ssz = REC["zbc_select"].size()
        out: List[SelectTable] = []
        for i in range(select_sec.count):
            srec = REC["zbc_select"].from_bytes(select_sec.payload[i * ssz:])
            n = srec.n_branches
            base = srec.oplist_off
            branches = [u32(base + j) for j in range(n)]
            weights = [u32(base + n + j) for j in range(n)]
            guards = [(-1 if u32(base + 2 * n + j) == cls._SEL_NOGUARD
                       else u32(base + 2 * n + j)) for j in range(n)]
            out.append(SelectTable(
                branches=branches, weights=weights, guards=guards,
                allow_none=bool(srec.flags & allow_none),
            ))
        return out

    @staticmethod
    def _parse_prov(prov_sec, cmnt_sec, file_sec, strtab: StringTable) -> ProvTable:
        if prov_sec is None:
            return ProvTable()
        files: List[str] = []
        if file_sec is not None:
            fsz = REC["zbc_file"].size()
            for i in range(file_sec.count):
                frec = REC["zbc_file"].from_bytes(file_sec.payload[i * fsz:])
                files.append(strtab.get(frec.path))
        comments: List[Comment] = []
        if cmnt_sec is not None:
            csz = REC["zbc_comment"].size()
            for i in range(cmnt_sec.count):
                crec = REC["zbc_comment"].from_bytes(cmnt_sec.payload[i * csz:])
                comments.append(Comment(text=strtab.get(crec.text),
                                        kind=crec.kind, line=crec.line))
        entries: List[Prov] = []
        psz = REC["zbc_prov"].size()
        for i in range(prov_sec.count):
            prec = REC["zbc_prov"].from_bytes(prov_sec.payload[i * psz:])
            cmts = (comments[prec.cmt_first: prec.cmt_first + prec.cmt_count]
                    if prec.cmt_count else [])
            entries.append(Prov(
                name=strtab.get(prec.name),
                node_kind=prec.node_kind,
                node_id=prec.node_id,
                file=files[prec.file] if prec.file < len(files) else "",
                line=prec.line,
                col=prec.col,
                col_end=prec.col_end,
                line_end=prec.line_end,
                flags=prec.flags,
                comments=list(cmts),
            ))
        return ProvTable(entries=entries)

    def to_bytes(self) -> bytes:
        return write_image(self.to_container())

    _ENTRY_KINDS = {"action": "ZBC_ENTRY_ACTION", "function": "ZBC_ENTRY_FUNCTION"}

    def _serialize_entries(self, strtab: "StringTable") -> bytes:
        """ENTRY section: one ``zbc_entry`` per :attr:`entries` item, in
        insertion order; the signature is canonical JSON in the string table."""
        out = bytearray()
        for sym, e in self.entries.items():
            rec = REC["zbc_entry"]()
            rec.name = strtab.intern(sym)
            rec.coro = e.coro
            rec.kind = spec.flag_value("zbc_entry_kind", self._ENTRY_KINDS[e.kind])
            rec.comp_type = strtab.intern(e.comp_type) if e.comp_type else 0
            if e.params or e.returns is not None:
                rec.sig = strtab.intern(json.dumps(
                    {"params": [[n, d] for n, d in e.params], "returns": e.returns},
                    sort_keys=True, separators=(",", ":")))
            rec.qname = strtab.intern(e.qname) if e.qname else 0
            if e.obj:
                rec.obj = strtab.intern(json.dumps(
                    [[n, d] for n, d in e.obj], sort_keys=True, separators=(",", ":")))
            out += rec.to_bytes()
        return bytes(out)

    @staticmethod
    def _parse_entries(sec: Optional[Section], strtab: "StringTable") -> Dict[str, "EntryPoint"]:
        entries: Dict[str, EntryPoint] = {}
        if sec is None:
            return entries
        size = REC["zbc_entry"].size()
        kinds = {spec.flag_value("zbc_entry_kind", v): k
                 for k, v in ZbcModel._ENTRY_KINDS.items()}
        for i in range(sec.count):
            rec = REC["zbc_entry"].from_bytes(sec.payload[i * size:])
            sig = json.loads(strtab.get(rec.sig)) if rec.sig else {"params": [], "returns": None}
            entries[strtab.get(rec.name)] = EntryPoint(
                coro=rec.coro, kind=kinds[rec.kind],
                params=[(n, d) for n, d in sig["params"]], returns=sig["returns"],
                comp_type=strtab.get(rec.comp_type) if rec.comp_type else None,
                qname=strtab.get(rec.qname) if rec.qname else None,
                obj=[(n, d) for n, d in json.loads(strtab.get(rec.obj))] if rec.obj else [])
        return entries

    @classmethod
    def from_container(cls, container: ZbcImage) -> "ZbcModel":
        by_kind: Dict[int, Section] = {}
        for s in container.sections:
            by_kind.setdefault(s.kind, s)  # first of a kind

        def sec(name: str) -> Optional[Section]:
            return by_kind.get(spec.sec_kind(name))

        # Rebuild interned tables (codegen profile).
        strb = sec("ZBC_SEC_STRB")
        stro = sec("ZBC_SEC_STRO")
        if strb is not None and stro is not None:
            strtab = StringTable.parse(strb.payload, stro.payload)
        else:
            strtab = StringTable()

        type_sec = sec("ZBC_SEC_TYPE")
        if type_sec is not None:
            types = TypeTable.parse(type_sec.payload, type_sec.count, strtab)
        else:
            types = TypeTable()

        const_sec = sec("ZBC_SEC_CONST")
        consts = ConstPool.parse(const_sec.payload) if const_sec is not None else ConstPool()

        prov = cls._parse_prov(sec("ZBC_SEC_PROV"), sec("ZBC_SEC_CMNT"),
                               sec("ZBC_SEC_FILE"), strtab)

        selects = cls._parse_selects(sec("ZBC_SEC_SELECT"), sec("ZBC_SEC_OPLIST"))
        problems = cls._parse_solves(sec("ZBC_SEC_SOLVE"), sec("ZBC_SEC_OPLIST"),
                                     sec("ZBC_SEC_SPROB"))

        # Decode instruction + block arrays.
        instr_size = REC["zbc_instr"].size()
        block_size = REC["zbc_block"].size()
        code_sec = sec("ZBC_SEC_CODE")
        all_instrs: List[Instr] = []
        if code_sec is not None:
            for i in range(code_sec.count):
                rec = REC["zbc_instr"].from_bytes(code_sec.payload[i * instr_size:])
                all_instrs.append(Instr.from_record(rec))

        block_sec = sec("ZBC_SEC_BLOCK")
        all_blocks: List[Block] = []
        if block_sec is not None:
            for i in range(block_sec.count):
                rec = REC["zbc_block"].from_bytes(block_sec.payload[i * block_size:])
                all_blocks.append(Block(rec.idx, rec.pc_start, rec.pc_end, rec.suspend_op))

        # Rebuild coroutines.
        coro_sec = sec("ZBC_SEC_CORO")
        coro_size = REC["zbc_coro"].size()
        coros: List[CoroDescriptor] = []
        lines: List[Tuple[int, int]] = []
        line_sec = sec("ZBC_SEC_LINE")
        if line_sec is not None:
            line_size = REC["zbc_lineent"].size()
            for i in range(line_sec.count):
                rec = REC["zbc_lineent"].from_bytes(line_sec.payload[i * line_size:])
                lines.append((rec.pc_start, rec.src_ref))
        local_recs = []
        local_sec = sec("ZBC_SEC_LOCAL")
        if local_sec is not None:
            local_size = REC["zbc_localent"].size()
            for i in range(local_sec.count):
                local_recs.append(REC["zbc_localent"].from_bytes(
                    local_sec.payload[i * local_size:]))
        reg_recs = cls._parse_recs(sec("ZBC_SEC_REG"), "zbc_regent")
        chan_recs = cls._parse_recs(sec("ZBC_SEC_CHAN"), "zbc_chanent")
        cov_recs = cls._parse_recs(sec("ZBC_SEC_COV"), "zbc_covent")
        f_arm = spec.flag_value("zbc_cov_flags", "ZBC_COV_F_ARM")
        f_imp = spec.flag_value("zbc_cov_flags", "ZBC_COV_F_IMPLICIT")
        f_discard = spec.flag_value("zbc_reg_flags", "ZBC_REG_F_DISCARD")
        if coro_sec is not None:
            for i in range(coro_sec.count):
                rec = REC["zbc_coro"].from_bytes(coro_sec.payload[i * coro_size:])
                code = all_instrs[rec.code_start: rec.code_start + rec.code_count]
                blocks = all_blocks[rec.block_start: rec.block_start + rec.n_blocks]
                frame_locals: List[str] = []
                if rec.frame_type:
                    ft = types.types[rec.frame_type - 1]
                    if isinstance(ft, StructType):
                        frame_locals = [n for n, _ in ft.fields]
                coros.append(CoroDescriptor(
                    name=strtab.get(rec.name),
                    code=code,
                    blocks=blocks,
                    frame_locals=frame_locals,
                    src_ref=rec.src_ref,
                    stmts=[(pc - rec.code_start, sr) for pc, sr in lines
                           if rec.code_start <= pc < rec.code_start + rec.code_count],
                    locals=[LocalVar(
                        slot=lr.slot, name=strtab.get(lr.name),
                        pc_start=lr.pc_start - rec.code_start,
                        pc_end=lr.pc_end - rec.code_start, width=lr.width,
                        kind=LOCAL_KINDS[lr.kind],
                        signed=bool(lr.flags & spec.flag_value(
                            "zbc_local_flags", "ZBC_LOCAL_F_SIGNED")))
                        for lr in local_recs if lr.coro == i],
                    regs=[(r.pc - rec.code_start, r.inst) for r in reg_recs
                          if rec.code_start <= r.pc < rec.code_start + rec.code_count
                          and r.inst != 0xFFFFFFFF],
                    discards=[r.pc - rec.code_start for r in reg_recs
                              if rec.code_start <= r.pc < rec.code_start + rec.code_count
                              and r.flags & f_discard],
                    chans=[(c.pc - rec.code_start, c.src_ref, strtab.get(c.channel),
                            strtab.get(c.op), -1 if c.val == 0xFFFFFFFF else c.val,
                            -1 if c.ok == 0xFFFFFFFF else c.ok) for c in chan_recs
                           if rec.code_start <= c.pc < rec.code_start + rec.code_count],
                    cov=[("arm" if v.flags & f_arm else "stmt", v.pc - rec.code_start,
                          strtab.get(v.scope), strtab.get(v.path), v.src_ref,
                          bool(v.flags & f_imp)) for v in cov_recs
                         if rec.code_start <= v.pc < rec.code_start + rec.code_count],
                ))

        # The parsed tables carry their cones' blobs (ACT), so the model
        # runs its activities as read; they also stand for the image's ACT.
        acts = cls._parse_activations(sec("ZBC_SEC_ACT"))
        memmap = (json.loads(strtab.get(REC["zbc_memmap"].from_bytes(
            sec("ZBC_SEC_MEMMAP").payload).doc))
            if sec("ZBC_SEC_MEMMAP") is not None else {})
        if memmap.get("spaces"):
            for t in acts.values():
                t.set_regions(memmap)   # contiguous claims are placed in these
        entries = cls._parse_entries(sec("ZBC_SEC_ENTRY"), strtab)
        return cls(
            coros=coros,
            entry_coro=container.entry_coro,
            consts=consts,
            prov=prov,
            selects=selects,
            problems=problems,
            abi_id=container.abi_id,
            profile=container.profile,
            comp_init=container.comp_init,
            entries=entries,
            # Action objects get their field names back from the entries.
            obj_layouts={e.coro: [n for n, _ in e.obj] for e in entries.values() if e.obj},
            sources=cls._parse_sources(sec("ZBC_SEC_SRC"), strtab),
            meta=cls._parse_sources(sec("ZBC_SEC_META"), strtab, "zbc_metaent"),
            import_names=cls._parse_imports(sec("ZBC_SEC_IMPORT"), strtab),
            **cls._parse_ident(sec("ZBC_SEC_IDENT")),
            components=cls._parse_comp(sec("ZBC_SEC_COMP"), sec("ZBC_SEC_LAYOUT"), strtab),
            registers=cls._parse_regmap(sec("ZBC_SEC_REGMAP"), strtab),
            memmap=memmap,
            **cls._parse_exec(sec("ZBC_SEC_EXEC"), strtab),
            events=cls._parse_events(sec("ZBC_SEC_EVENT"), strtab),
            covgs=(json.loads(strtab.get(REC["zbc_covg"].from_bytes(sec("ZBC_SEC_COVG").payload).doc))
                   if sec("ZBC_SEC_COVG") is not None else []),
            seed_keys=cls._parse_seed_keys(sec("ZBC_SEC_SEEDKEY"), len(coros)),
            **dict(zip(("messages", "strings"), cls._parse_messages(sec("ZBC_SEC_MSG")))),
            activations=dict(acts),
            image_activations=acts,
        )

    @staticmethod
    def _parse_seed_keys(sec: Optional[Section], n: int) -> Optional[List[int]]:
        """SEEDKEY back to :attr:`seed_keys` (None if absent or not one per coroutine)."""
        if sec is None or sec.count != n or len(sec.payload) < 8 * n:
            return None
        return list(struct.unpack_from("<%dQ" % n, sec.payload))

    def coro_seed_keys(self) -> List[int]:
        """Each coroutine's seed key: as read, or from names and layouts."""
        if self.seed_keys is None:
            from .determinism import coro_seed_keys
            self.seed_keys = coro_seed_keys(self.coros, self.obj_layouts or {})
        return self.seed_keys

    def _serialize_exec(self, strtab: "StringTable") -> bytes:
        """EXEC: zbc_exectab, then the executor rows, group rows and members."""
        members = [m for g in self.exec_groups for m in g["members"]]
        hdr = REC["zbc_exectab"]()
        hdr.n_exec, hdr.n_groups, hdr.n_members = (
            len(self.executors), len(self.exec_groups), len(members))
        out = bytearray(hdr.to_bytes())
        for e in self.executors:
            rec = REC["zbc_execent"]()
            rec.path, rec.type, rec.trait = (strtab.intern(e["path"]), strtab.intern(e["type"]),
                                             strtab.intern(e.get("trait") or ""))
            out += rec.to_bytes()
        first = 0
        for g in self.exec_groups:
            rec = REC["zbc_execgrp"]()
            rec.path, rec.trait = strtab.intern(g["path"]), strtab.intern(g.get("trait") or "")
            rec.first, rec.count = first, len(g["members"])
            first += rec.count
            out += rec.to_bytes()
        out += struct.pack("<%dI" % len(members), *members)
        return bytes(out)

    _EVT_SLOTS = ("width", "count", "head", "buf", "depth")

    def _serialize_events(self, strtab: "StringTable") -> bytes:
        """EVENT: one zbc_evtent per host-fed channel."""
        out = bytearray()
        for e in self.events:
            rec = REC["zbc_evtent"]()
            rec.path, rec.kind = strtab.intern(e["path"]), e["kind"]
            rec.flags = 1 if e.get("signed") else 0
            for k in self._EVT_SLOTS:
                setattr(rec, k, e[k])
            out += rec.to_bytes()
        return bytes(out)

    @classmethod
    def _parse_events(cls, sec: Optional[Section], strtab: StringTable) -> List[dict]:
        """EVENT back to :attr:`events`."""
        if sec is None or strtab is None:
            return []
        ent, rows = REC["zbc_evtent"], []
        for i in range(sec.count):
            r = ent.from_bytes(sec.payload[i * ent.size():])
            rows.append({"path": strtab.get(r.path), "kind": r.kind,
                         "signed": bool(r.flags & 1),
                         **{k: getattr(r, k) for k in cls._EVT_SLOTS}})
        return rows

    @staticmethod
    def _parse_exec(sec: Optional[Section], strtab: StringTable) -> dict:
        """EXEC back to :attr:`executors` and :attr:`exec_groups`."""
        if sec is None or strtab is None:
            return {}
        p = sec.payload
        hdr = REC["zbc_exectab"].from_bytes(p)
        off = REC["zbc_exectab"].size()
        ent, grp = REC["zbc_execent"], REC["zbc_execgrp"]
        rows = []
        for _ in range(hdr.n_exec):
            r = ent.from_bytes(p[off:])
            rows.append({"path": strtab.get(r.path), "type": strtab.get(r.type),
                         "trait": strtab.get(r.trait)})
            off += ent.size()
        grecs = []
        for _ in range(hdr.n_groups):
            grecs.append(grp.from_bytes(p[off:]))
            off += grp.size()
        members = list(struct.unpack_from("<%dI" % hdr.n_members, p, off))
        groups = [{"path": strtab.get(g.path), "trait": strtab.get(g.trait),
                   "members": members[g.first:g.first + g.count]} for g in grecs]
        return {"executors": rows, "exec_groups": groups}

    @staticmethod
    def _parse_regmap(sec: Optional[Section], strtab: StringTable) -> Dict[int, dict]:
        """REGMAP back to :attr:`registers` (JSON keys back to ints)."""
        if sec is None or not sec.count:
            return {}
        rec = REC["zbc_regmap"].from_bytes(sec.payload)
        return {int(k): v for k, v in json.loads(strtab.get(rec.doc)).items()}

    @staticmethod
    def _parse_comp(sec: Optional[Section], lay: Optional[Section] = None,
                    strtab: Optional[StringTable] = None):
        """COMP back to a ``CompTable``, slot and instance names from LAYOUT
        (``$c<slot>`` without it); its flags kept so the image re-serializes
        the same."""
        if sec is None:
            return None
        from .interp.components import CompTable
        rec = REC["zbc_comp"].from_bytes(sec.payload)
        t = CompTable.__new__(CompTable)
        t.names = [f"$c{i}" for i in range(rec.n_slots)]
        t.bases = list(struct.unpack_from(f"<{rec.n_inst}I", sec.payload, len(rec.to_bytes())))
        t.paths, t.types = [""] * rec.n_inst, [""] * rec.n_inst
        t.init_coro = None if rec.init_coro == 0xFFFFFFFF else rec.init_coro
        t.comp_flags = rec.flags
        t.run_hooks = bool(rec.flags & spec.flag_value("zbc_comp_flags",
                                                       "ZBC_COMP_F_RUN_HOOKS"))
        t.root, t.slot_types = "", []
        if lay is not None and lay.count and strtab is not None:
            lrec = REC["zbc_layout"].from_bytes(lay.payload)
            slots = json.loads(strtab.get(lrec.slots)) if lrec.slots else []
            insts = json.loads(strtab.get(lrec.insts)) if lrec.insts else []
            if len(slots) == rec.n_slots:
                t.names = [n for n, _ in slots]
                t.slot_types = [d for _, d in slots]
            if len(insts) == rec.n_inst:
                t.paths = [p for p, _ in insts]
                t.types = [ty for _, ty in insts]
            t.root = strtab.get(lrec.root) if lrec.root else ""
        return t

    @staticmethod
    def _parse_ident(sec: Optional[Section]) -> dict:
        if sec is None or not sec.count:
            return {}
        size = REC["zbc_ident"].size()
        rec = REC["zbc_ident"].from_bytes(sec.payload.ljust(size, b"\0"))
        # a 16-byte IDENT (before time_exp) means the default unit
        t = rec.time_exp if len(sec.payload) >= size else (-12 & 0xFFFFFFFF)
        return {"api_hash": rec.api_hash, "generation": rec.generation,
                "time_exp": t - (1 << 32) if t & 0x80000000 else t}

    @staticmethod
    def _parse_recs(sec: Optional[Section], record: str) -> list:
        """A section's fixed-size records (none if it is absent)."""
        if sec is None:
            return []
        size = REC[record].size()
        return [REC[record].from_bytes(sec.payload[i * size:]) for i in range(sec.count)]

    @staticmethod
    def _parse_sources(src_sec: Optional[Section], strtab: StringTable,
                       record: str = "zbc_srcent") -> Dict[str, str]:
        """SRC (``zbc_srcent``: path, text) or META (``zbc_metaent``: name,
        text): a table of string pairs either way."""
        sources: Dict[str, str] = {}
        if src_sec is None:
            return sources
        size = REC[record].size()
        for i in range(src_sec.count):
            rec = REC[record].from_bytes(src_sec.payload[i * size:])
            key = rec.path if record == "zbc_srcent" else rec.name
            sources[strtab.get(key)] = strtab.get(rec.text)
        return sources

    @staticmethod
    def _parse_imports(sec: Optional[Section], strtab: StringTable) -> Dict[int, str]:
        """IMPORT back to ``{fn_id: name}``."""
        imports: Dict[int, str] = {}
        if sec is None:
            return imports
        size = REC["zbc_impent"].size()
        for i in range(sec.count):
            rec = REC["zbc_impent"].from_bytes(sec.payload[i * size:])
            imports[rec.fn_id] = strtab.get(rec.name)
        return imports

    @classmethod
    def from_bytes(cls, data: bytes) -> "ZbcModel":
        return cls.from_container(read_image(data))
