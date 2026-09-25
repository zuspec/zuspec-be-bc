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
from typing import Dict, List, Optional, Tuple

from .abi.value import ABI_ID, ScalarType, ArrayType, StructType, ValueType
from .abi.codec import const_pool_record, decode_scalar
from .format import spec
from .format.writer import Section, ZbcImage, write_image
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

    # procedural: compare / branch (0x30-0x3F)
    CMP_EQ = 0x30
    CMP_NE = 0x31
    CMP_LT = 0x32
    CMP_LE = 0x33
    CMP_GT = 0x34
    CMP_GE = 0x35
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
    IMPORT = 0x49


#: Orchestration-tier opcodes (the suspend-capable / scheduler-driving ops).
ORCH_OPS = frozenset({
    Op.SPAWN, Op.INVOKE, Op.PAR, Op.JOIN, Op.WAIT,
    Op.SELECT, Op.SOLVE, Op.BIND, Op.YIELD, Op.IMPORT,
})

#: Ops that unconditionally end an FSM block by suspending (D§4.1, coro_fsm).
#: INVOKE/IMPORT suspend only when their blocking flag is set (see INSTR_F_BLOCKING).
UNCONDITIONAL_SUSPEND_OPS = frozenset({Op.WAIT, Op.JOIN, Op.PAR, Op.YIELD})

#: Instruction flag bits.
INSTR_F_FROM_POOL = 0x01   # CONST: imm is a const-pool id, not an inline value
INSTR_F_BLOCKING = 0x02    # INVOKE/IMPORT: this call suspends
INSTR_F_HAS_RET = 0x04     # IMPORT/INVOKE: arg1 is a result register

#: IMPORT fn_ids at and above this are interpreter builtins, not user imports.
#: A builtin is an ordinary IMPORT (no new opcode), so the ISA is unchanged.
BUILTIN_BASE = 0xFFFFFF00
#: ``message(verbosity, fmt, args...)``: args = (msg_idx,). The entry
#: ``ZbcModel.messages[msg_idx]`` names the frame-local slots holding the
#: verbosity and each value (an instruction has only 4 inline args).
BUILTIN_MESSAGE = BUILTIN_BASE + 0
#: A run-time error the LRM says "shall" be raised: args = (string_idx,).
BUILTIN_ERROR = BUILTIN_BASE + 1


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


@dc.dataclass
class CoroDescriptor:
    name: str = ""
    code: List[Instr] = dc.field(default_factory=list)
    blocks: List[Block] = dc.field(default_factory=list)
    frame_locals: List[str] = dc.field(default_factory=list)
    src_ref: int = 0

    def frame_type(self) -> Optional[StructType]:
        """The frame as a struct type (locals default to 64-bit slots in M1)."""
        if not self.frame_locals:
            return None
        return StructType(tuple((n, ScalarType(64)) for n in self.frame_locals))


# --------------------------------------------------------------------------- #
# M1 in-memory problem table (not serialized in M1; D§1.2)
# --------------------------------------------------------------------------- #

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
    #: drawn seed and writes ``solver_get_value(var_id)`` back per ``writeback_slots``.
    #: Empty (the M1 default) selects the minimal ``slot = seed + var_id`` randomizer.
    problem_bytes: bytes = b""


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
    abi_id: int = ABI_ID
    profile: str = "codegen"

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

        for coro in self.coros:
            code_start = n_instrs
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

        # Provenance-family sections (codegen profile only): FILE/PROV/CMNT then
        # TYPE + string table (STRB/STRO must be last -- all interning done by then).
        if self.profile == "codegen":
            file_bytes, prov_bytes, cmnt_bytes, n_files, n_prov, n_cmnt = \
                self._serialize_prov(strtab)
            sections.append(Section(spec.sec_kind("ZBC_SEC_FILE"), file_bytes,
                                    count=n_files, elem_size=REC["zbc_file"].size()))
            sections.append(Section(spec.sec_kind("ZBC_SEC_PROV"), prov_bytes,
                                    count=n_prov, elem_size=REC["zbc_prov"].size()))
            sections.append(Section(spec.sec_kind("ZBC_SEC_CMNT"), cmnt_bytes,
                                    count=n_cmnt, elem_size=REC["zbc_comment"].size()))
            type_bytes = types.to_bytes(strtab)
            if type_bytes:
                sections.append(Section(spec.sec_kind("ZBC_SEC_TYPE"), type_bytes,
                                        count=len(types.types)))
            sections.append(Section(spec.sec_kind("ZBC_SEC_STRB"), strtab.strb_bytes()))
            sections.append(Section(spec.sec_kind("ZBC_SEC_STRO"), strtab.stro_bytes()))

        return ZbcImage(
            sections=sections,
            abi_id=self.abi_id,
            entry_coro=self.entry_coro,
            profile=self.profile,
        )

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
        solve_b = bytearray()
        for p in self.problems:
            pairs = sorted(p.writeback_slots.items())
            srec = REC["zbc_solve"]()
            srec.seed_value = (p.seed_value & MASK64) if p.seed_kind == "fixed" else 0
            srec.oplist_off = len(oplist) // 4
            srec.n_writeback = len(pairs)
            srec.flags = seed_fixed if p.seed_kind == "fixed" else 0
            if p.problem_bytes:
                # 4-align so the blob's leading SolveProblem header stays aligned when
                # the engine hands its address straight to solver_compile().
                while len(sprob) % 4:
                    sprob += b"\x00"
                srec.prob_off = len(sprob)
                srec.prob_len = len(p.problem_bytes)
                sprob += p.problem_bytes
            solve_b += srec.to_bytes()
            for slot, var_id in pairs:
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
            base = srec.oplist_off
            wb = {u32(base + 2 * j): u32(base + 2 * j + 1)
                  for j in range(srec.n_writeback)}
            blob = (bytes(sprob[srec.prob_off:srec.prob_off + srec.prob_len])
                    if srec.prob_len else b"")
            out.append(SolveProblem(
                writeback_slots=wb,
                seed_kind="fixed" if fixed else "inherit",
                seed_value=srec.seed_value if fixed else 0,
                problem_bytes=blob,
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
                ))

        return cls(
            coros=coros,
            entry_coro=container.entry_coro,
            consts=consts,
            prov=prov,
            selects=selects,
            problems=problems,
            abi_id=container.abi_id,
            profile=container.profile,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "ZbcModel":
        return cls.from_container(read_image(data))
