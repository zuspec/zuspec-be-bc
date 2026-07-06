"""
spec.py -- the single declarative description of the ``.zbc`` format (D§12).

This module is the **authority**. The C header, the ``ctypes`` overlay, and the
in-memory dataclasses are generated from the ``RECORDS`` and ``ENUMS`` tables
below; the format doc mirrors it. Nothing else in the codebase hand-writes a
record layout.

Every record is laid out with **natural alignment** and little-endian fields, so
a C reader can ``mmap`` and cast-index directly and a Python reader can overlay
``ctypes`` with no parsing. Field order below is chosen so there is no implicit
padding (verified by the conformance tests).
"""

import dataclasses as dc
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Primitive field types: name -> (C type, struct-format char, byte size)
# --------------------------------------------------------------------------- #

PRIMS: Dict[str, Tuple[str, str, int]] = {
    "u8": ("uint8_t", "B", 1),
    "u16": ("uint16_t", "H", 2),
    "u32": ("uint32_t", "I", 4),
    "u64": ("uint64_t", "Q", 8),
}


@dc.dataclass(frozen=True)
class Field:
    """One record field. ``count > 1`` (or ``is_array``) makes it a fixed array."""

    name: str
    type: str          # key into PRIMS
    count: int = 1     # >1 => array; 1 => scalar
    comment: str = ""
    is_array: bool = False  # force array form even for count==1 (e.g. magic[4])

    @property
    def elem_size(self) -> int:
        return PRIMS[self.type][2]

    @property
    def byte_len(self) -> int:
        return self.elem_size * self.count

    @property
    def align(self) -> int:
        # A scalar aligns to its own size; an array aligns to its element size.
        return self.elem_size

    @property
    def array(self) -> bool:
        return self.is_array or self.count > 1

    @property
    def struct_fmt(self) -> str:
        ch = PRIMS[self.type][1]
        return f"{self.count}{ch}" if self.array else ch


@dc.dataclass(frozen=True)
class Record:
    name: str          # C struct tag, e.g. "zbc_header"
    fields: Tuple[Field, ...]
    doc: str = ""

    def size(self) -> int:
        return sum(f.byte_len for f in self.fields)

    def struct_fmt(self) -> str:
        # Little-endian, standard sizes, no implicit alignment padding. Valid only
        # because field order is chosen to be naturally aligned with no gaps
        # (asserted by tests).
        return "<" + "".join(f.struct_fmt for f in self.fields)


@dc.dataclass(frozen=True)
class EnumMember:
    name: str
    value: int
    comment: str = ""


@dc.dataclass(frozen=True)
class Enum:
    name: str
    members: Tuple[EnumMember, ...]
    doc: str = ""
    is_flags: bool = False


# --------------------------------------------------------------------------- #
# Format constants
# --------------------------------------------------------------------------- #

MAGIC = (ord("Z"), ord("B"), ord("C"), 0x1A)   # 0x1A traps text-mode corruption
VERSION_MAJOR = 1
VERSION_MINOR = 0

# --------------------------------------------------------------------------- #
# Records (D§12.1 / §12.2)
# --------------------------------------------------------------------------- #

RECORDS: List[Record] = [
    Record(
        "zbc_header",
        (
            Field("magic", "u8", 4, "'Z','B','C',0x1A", is_array=True),
            Field("version_major", "u16", comment="incompatible bump -> engine REJECTS"),
            Field("version_minor", "u16", comment="additive bump -> tolerated"),
            Field("flags", "u32", comment="ZBC_HDR_*"),
            Field("header_size", "u32", comment="sizeof(zbc_header); readers skip grown tail"),
            Field("abi_id", "u32", comment="value-ABI version (D§11); must match engine"),
            Field("section_count", "u32"),
            Field("section_dir_off", "u32"),
            Field("entry_coro", "u32", comment="index of root/entry coroutine descriptor"),
            Field("file_size", "u64", comment="integrity: equals actual length"),
            Field("content_hash", "u64", comment="hash of bytes after this field (0=none)"),
        ),
        doc="File header at offset 0. Magic is byte-order-fixed and checked first.",
    ),
    Record(
        "zbc_section",
        (
            Field("kind", "u32", comment="ZBC_SEC_*"),
            Field("flags", "u32", comment="ZBC_SECF_*"),
            Field("offset", "u64", comment="from file start, 8-byte aligned"),
            Field("size", "u64", comment="byte length"),
            Field("count", "u32", comment="element count for record arrays (0=N/A)"),
            Field("elem_size", "u32", comment="per-element stride (0=N/A); forward-compat lever"),
        ),
        doc="Section directory entry; fixed size so it is itself trivially indexable.",
    ),
    Record(
        "zbc_prov",
        (
            Field("name", "u32", comment="StrId (0=none)"),
            Field("node_id", "u32", comment="stable IR node id (0=none)"),
            Field("line", "u32", comment="1-based start line (0=unknown)"),
            Field("cmt_first", "u32", comment="CMNT index of first attached comment"),
            Field("file", "u16", comment="FileId"),
            Field("node_kind", "u16", comment="origin-node-kind enum"),
            Field("col", "u16", comment="start column (0=unknown)"),
            Field("col_end", "u16", comment="end column (0=unknown)"),
            Field("line_end", "u16", comment="end line (0=same as line)"),
            Field("flags", "u16", comment="ZBC_PROV_*"),
            Field("cmt_count", "u16", comment="comments in this record's run"),
            Field("_rsvd", "u16"),
        ),
        doc="Provenance record (32 bytes). prov[0] is the reserved all-zero 'none' entry.",
    ),
    Record(
        "zbc_comment",
        (
            Field("text", "u32", comment="StrId"),
            Field("kind", "u8", comment="ZBC_CMT_*"),
            Field("_pad", "u8"),
            Field("line", "u16", comment="source line of the comment (0=unknown)"),
        ),
        doc="Comment record (8 bytes). A prov entry owns run [cmt_first, cmt_first+cmt_count).",
    ),
    Record(
        "zbc_file",
        (
            Field("path", "u32", comment="StrId"),
            Field("_rsvd", "u32"),
        ),
        doc="File-table entry (8 bytes). FileId = u16 index into files[].",
    ),
    Record(
        "zbc_lineent",
        (
            Field("pc_start", "u32", comment="sorted ascending"),
            Field("src_ref", "u32", comment="prov index; src_at(pc)=largest pc_start<=pc"),
        ),
        doc="pc->src_ref line table entry (8 bytes). Binary-searched by pc.",
    ),
    # ----- Executable tier (CORO/CODE/BLOCK), concretized by P1 lowering ----- #
    Record(
        "zbc_instr",
        (
            Field("op", "u16", comment="ZBC opcode (see model.Op)"),
            Field("nargs", "u8", comment="count of valid args (0..4)"),
            Field("flags", "u8", comment="per-op flags (e.g. CONST from pool)"),
            Field("src_ref", "u32", comment="provenance index (0=none)"),
            Field("imm", "u64", comment="immediate / const bits (two's complement)"),
            Field("arg0", "u32"),
            Field("arg1", "u32"),
            Field("arg2", "u32"),
            Field("arg3", "u32"),
        ),
        doc="One fixed-width instruction (32 bytes). CODE section is zbc_instr[].",
    ),
    Record(
        "zbc_coro",
        (
            Field("name", "u32", comment="StrId (0=none/runtime profile)"),
            Field("code_start", "u32", comment="first instr index into CODE"),
            Field("code_count", "u32", comment="instr count"),
            Field("n_blocks", "u32", comment="FSM block count"),
            Field("block_start", "u32", comment="first entry into BLOCK table"),
            Field("frame_type", "u32", comment="TYPE id of frame struct (0=none)"),
            Field("src_ref", "u32", comment="provenance index (0=none)"),
            Field("_rsvd", "u32"),
        ),
        doc="Coroutine/func descriptor (32 bytes). CORO section is zbc_coro[].",
    ),
    Record(
        "zbc_block",
        (
            Field("idx", "u32", comment="FSM block index"),
            Field("pc_start", "u32", comment="first instr index (relative to coro)"),
            Field("pc_end", "u32", comment="one past last instr index"),
            Field("suspend_op", "u16", comment="orchestration op ending the block (0=terminal)"),
            Field("_rsvd", "u16"),
        ),
        doc="FSM block (16 bytes). BLOCK section is zbc_block[]; a coro owns a run.",
    ),
    Record(
        "zbc_select",
        (
            Field("oplist_off", "u32", comment="start index into OPLIST u32 pool"),
            Field("n_branches", "u32", comment="branch count"),
            Field("flags", "u32", comment="ZBC_SEL_* (bit0 = allow_none)"),
            Field("_rsvd", "u32"),
        ),
        doc=("Weighted-SELECT descriptor (16 bytes). SELECT section is zbc_select[]; "
             "arg0 of a SELECT instr indexes it. The OPLIST pool holds, contiguously "
             "from oplist_off, three u32 runs of length n_branches: branch coro ids, "
             "positive weights, then guard registers (0xFFFFFFFF = unguarded)."),
    ),
    Record(
        "zbc_solve",
        (
            Field("seed_value", "u64", comment="fixed-seed value (when flags & SEED_FIXED)"),
            Field("oplist_off", "u32", comment="start of (field_slot, var_id) pairs in OPLIST"),
            Field("n_writeback", "u32", comment="writeback pair count"),
            Field("flags", "u32", comment="ZBC_SOLVE_* (bit0 = fixed seed)"),
            Field("prob_off", "u32", comment="byte offset of the problem blob in the SPROB pool"),
            Field("prob_len", "u32", comment="problem blob length in bytes (0 = minimal randomizer)"),
            Field("_rsvd", "u32"),
        ),
        doc=("SOLVE descriptor (32 bytes). SOLVE section is zbc_solve[]; arg0 of a "
             "SOLVE instr indexes it. The OPLIST pool holds, from oplist_off, "
             "n_writeback interleaved (field_slot, var_id) u32 pairs -- the value ABI "
             "write-back keyed by object slot. When prob_len > 0, the (prob_off, "
             "prob_len) slice of the SPROB pool is a relocatable dv-solve SolveProblem "
             "blob: the engine compiles + solves it with the drawn seed and writes "
             "solver_get_value(var_id) back to each field_slot. When prob_len == 0 the "
             "minimal M1 randomizer applies: slot = seed + var_id (FixedSolveBackend, "
             "base 0). Seed is seed_value if SEED_FIXED else the frame's next draw."),
    ),
]

RECORDS_BY_NAME: Dict[str, Record] = {r.name: r for r in RECORDS}

# --------------------------------------------------------------------------- #
# Enums (D§12.1 / §12.2)
# --------------------------------------------------------------------------- #

ENUMS: List[Enum] = [
    Enum(
        "zbc_sec_kind",
        (
            EnumMember("ZBC_SEC_CODE", 1, "opcode stream"),
            EnumMember("ZBC_SEC_CORO", 2, "coroutine/func descriptors"),
            EnumMember("ZBC_SEC_TYPE", 3, "type/field metadata"),
            EnumMember("ZBC_SEC_CONST", 4, ">64-bit literal constant pool"),
            EnumMember("ZBC_SEC_SOLVE", 5, "SolveProblem blobs (P4)"),
            EnumMember("ZBC_SEC_STRB", 6, "string blob"),
            EnumMember("ZBC_SEC_STRO", 7, "string offsets"),
            EnumMember("ZBC_SEC_FILE", 8, "file table"),
            EnumMember("ZBC_SEC_PROV", 9, "provenance records"),
            EnumMember("ZBC_SEC_CMNT", 10, "comment records"),
            EnumMember("ZBC_SEC_LINE", 11, "pc->src_ref line table"),
            EnumMember("ZBC_SEC_BLOCK", 12, "FSM block table (zbc_block[])"),
            EnumMember("ZBC_SEC_OPLIST", 13, "u32 operand lists (PAR/SELECT branch ids)"),
            EnumMember("ZBC_SEC_SELECT", 14, "weighted-SELECT descriptors (zbc_select[])"),
            EnumMember("ZBC_SEC_SPROB", 15, "relocatable dv-solve SolveProblem blobs"),
        ),
        doc="Section kinds. Readers skip unknown kinds (forward-compat).",
    ),
    Enum(
        "zbc_hdr_flags",
        (
            EnumMember("ZBC_HDR_HAS_PROV", 0x0001, "provenance sections present"),
            EnumMember("ZBC_HDR_PROFILE_RUNTIME", 0x0002, "runtime profile (provenance stripped)"),
        ),
        doc="Header flag bits.",
        is_flags=True,
    ),
    Enum(
        "zbc_secf_flags",
        (
            EnumMember("ZBC_SECF_COMPRESSED", 0x0001, "payload is compressed"),
        ),
        doc="Per-section flag bits.",
        is_flags=True,
    ),
    Enum(
        "zbc_prov_flags",
        (
            EnumMember("ZBC_PROV_G_INSN", 0x0001, "granularity: instruction"),
            EnumMember("ZBC_PROV_G_BLOCK", 0x0002, "granularity: block"),
            EnumMember("ZBC_PROV_G_CORO", 0x0004, "granularity: coroutine"),
            EnumMember("ZBC_PROV_G_DECL", 0x0008, "granularity: declaration"),
            EnumMember("ZBC_PROV_F_SYNTH", 0x0100, "compiler-synthesized; no real source"),
        ),
        doc="Provenance granularity + attribute bits.",
        is_flags=True,
    ),
    Enum(
        "zbc_sel_flags",
        (
            EnumMember("ZBC_SEL_ALLOW_NONE", 0x0001, "no eligible branch -> run nothing (not an error)"),
        ),
        doc="SELECT descriptor flag bits.",
        is_flags=True,
    ),
    Enum(
        "zbc_solve_flags",
        (
            EnumMember("ZBC_SOLVE_SEED_FIXED", 0x0001, "use seed_value; else draw from the frame stream"),
        ),
        doc="SOLVE descriptor flag bits.",
        is_flags=True,
    ),
    Enum(
        "zbc_cmt_kind",
        (
            EnumMember("ZBC_CMT_LEADING", 1),
            EnumMember("ZBC_CMT_TRAILING", 2),
            EnumMember("ZBC_CMT_INLINE", 3),
            EnumMember("ZBC_CMT_DOC", 4),
        ),
        doc="Comment kinds.",
    ),
]

ENUMS_BY_NAME: Dict[str, Enum] = {e.name: e for e in ENUMS}


def sec_kind(name: str) -> int:
    """Look up a section-kind value by member name (e.g. 'ZBC_SEC_CODE')."""
    for m in ENUMS_BY_NAME["zbc_sec_kind"].members:
        if m.name == name:
            return m.value
    raise KeyError(name)


def sec_kind_name(value: int) -> Optional[str]:
    """Reverse of :func:`sec_kind`; returns ``None`` for an unknown kind."""
    for m in ENUMS_BY_NAME["zbc_sec_kind"].members:
        if m.value == value:
            return m.name
    return None


def flag_value(enum_name: str, member: str) -> int:
    for m in ENUMS_BY_NAME[enum_name].members:
        if m.name == member:
            return m.value
    raise KeyError(member)
