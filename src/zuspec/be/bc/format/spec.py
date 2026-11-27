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
VERSION_MAJOR = 2
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
    Record(
        "zbc_localent",
        (
            Field("coro", "u32", comment="coroutine index"),
            Field("slot", "u32", comment="frame-local slot"),
            Field("name", "u32", comment="StrId: the PSS name (a struct leaf: 'cfg.src')"),
            Field("pc_start", "u32", comment="global CODE index where it comes into scope"),
            Field("pc_end", "u32", comment="global CODE index where it leaves scope (excl.)"),
            Field("width", "u16", comment="bits"),
            Field("kind", "u8", comment="zbc_local_kind"),
            Field("flags", "u8", comment="zbc_local_flags"),
        ),
        doc="Named local (24 bytes; debug lowering). Sorted by (coro, pc_start).",
    ),
    Record(
        "zbc_regent",
        (
            Field("pc", "u32", comment="global CODE index of the access's IMPORT"),
            Field("inst", "u32",
                  comment="register instance, relative to the frame's component "
                          "instance; 0xFFFFFFFF: none (a discard alone)"),
            Field("flags", "u32", comment="zbc_reg_flags"),
        ),
        doc="Register access (12 bytes; debug lowering). Sorted by pc.",
    ),
    Record(
        "zbc_chanent",
        (
            Field("pc", "u32", comment="global CODE index of the NOP that ends the operation"),
            Field("src_ref", "u32", comment="prov index of the operation"),
            Field("channel", "u32", comment="StrId: the channel's name"),
            Field("op", "u32", comment="StrId: the operation (put, get, try_get, ...)"),
            Field("val", "u32", comment="register holding the value; 0xFFFFFFFF: none"),
            Field("ok", "u32", comment="register holding ok (try_*); 0xFFFFFFFF: none"),
        ),
        doc="Channel operation (24 bytes; debug lowering). Sorted by pc.",
    ),
    Record(
        "zbc_covent",
        (
            Field("pc", "u32", comment="global CODE index where the statement starts / the arm is taken"),
            Field("src_ref", "u32", comment="prov index of the statement or arm"),
            Field("flags", "u32", comment="zbc_cov_flags"),
            Field("scope", "u32", comment="StrId: the function it is in (<comp type>::<fn>)"),
            Field("path", "u32", comment="StrId: its place in PSS structure (/-joined)"),
        ),
        doc="Coverage item (20 bytes; debug lowering). Sorted by pc.",
    ),
    Record(
        "zbc_srcent",
        (
            Field("path", "u32", comment="StrId: the source file's path (as in FILE)"),
            Field("text", "u32", comment="StrId: the file's full text"),
        ),
        doc="One PSS source file carried in a debug image (8 bytes; SRC section).",
    ),
    Record(
        "zbc_ident",
        (
            Field("api_hash", "u64",
                  comment="FNV-1a/64 of every entry and import signature (0=unset)"),
            Field("generation", "u64", comment="caller-supplied build generation (0=unset)"),
            Field("time_exp", "u32",
                  comment="int32: one WAIT TIME tick is 10^time_exp s (-12 = ps; no IDENT: -12)"),
            Field("_pad", "u32"),
        ),
        doc="Image identity (24 bytes; IDENT section, one record; debug profile).",
    ),
    Record(
        "zbc_metaent",
        (
            Field("name", "u32", comment="StrId: the document's name (e.g. pssc.catalog)"),
            Field("text", "u32", comment="StrId: the document's text (the host's format)"),
        ),
        doc="One host metadata document carried in a debug image (8 bytes; META section).",
    ),
    Record(
        "zbc_impent",
        (
            Field("fn_id", "u32", comment="IMPORT fn_id the code calls"),
            Field("name", "u32", comment="StrId: the import's PSS name"),
            Field("api", "u32", comment="API id (an API row): the package or component declaring it"),
            Field("flags", "u32", comment="zbc_imp_flags"),
            Field("sig", "u32",
                  comment="StrId: JSON {arg_types, ret_type, string_at[, ret_types]} (0=none)"),
            Field("qname", "u32", comment="StrId: the import's qualified name (0=none)"),
        ),
        doc=("One user import (24 bytes; IMPORT section, sorted by fn_id; codegen/debug "
             "profiles): its name, API, kind and signature, which a host links against."),
    ),
    Record(
        "zbc_apient",
        (
            Field("id", "u32", comment="the API id requests carry (zbc_api_id)"),
            Field("name", "u32", comment="StrId: mem, config, seq, or the declaring scope"),
            Field("kind", "u32", comment="zbc_api_kind"),
            Field("n_fns", "u32", comment="the IMPORT rows naming this API"),
        ),
        doc=("One API of the import interface (16 bytes; API section, sorted by id; "
             "codegen/debug profiles). Ids 0-15 are built in; user APIs are 16 up, one "
             "per package or component declaring imports, in name order."),
    ),
    Record(
        "zbc_class",
        (
            Field("qname", "u32", comment="StrId: the type's qualified name"),
            Field("kind", "u32", comment="zbc_class_kind"),
            Field("base", "u32", comment="StrId: its base type (0=none)"),
            Field("coro", "u32",
                  comment="an action's coroutine (its traverse method); 0xFFFFFFFF = none"),
            Field("n_fields", "u32", comment="leaves (slots) of an object of the type"),
            Field("size", "u32", comment="native size in bytes (value-abi.md 2.1)"),
            Field("align", "u32", comment="native alignment in bytes"),
            Field("fields", "u32",
                  comment="StrId: JSON [[name, descriptor, offset, size], ...] in slot order"),
            Field("methods", "u32",
                  comment="StrId: JSON [[name, entry symbol], ...]: its functions' entries (0=none)"),
            Field("_pad", "u32"),
        ),
        doc=("One type (40 bytes; CLASS section; codegen/debug profiles): every component "
             "type in the tree and its bases, every action with an entry and its bases, and "
             "every plain-data struct their fields use, by qname. ``fields`` is the type's "
             "normative native layout (value-abi.md 2.1); slot k of an object is row k."),
    ),
    Record(
        "zbc_rootent",
        (
            Field("type", "u32", comment="StrId: the root candidate's component type"),
            Field("ctor", "u32",
                  comment="StrId: JSON {name, params: [{name, type}]}: its constructor (0=none)"),
            Field("api_hash", "u64", comment="the API hash of the image built for this root"),
            Field("img_off", "u64",
                  comment="offset of its image in the ROOTIMG section (img_size 0: this image)"),
            Field("img_size", "u64", comment="byte length of its image (0: this image)"),
        ),
        doc=("One root candidate (32 bytes; ROOT section; codegen/debug profiles). Row 0 is "
             "the image's own root, whose tables (COMP, ACT, ENTRY, ...) are this image's. "
             "Every other candidate's tables are a complete image of its own, nested in "
             "ROOTIMG at (img_off, img_size)."),
    ),
    Record(
        "zbc_comp",
        (
            Field("n_slots", "u32", comment="component object slot count"),
            Field("init_coro", "u32",
                  comment="CORO constructing the tree before the entry (0xFFFFFFFF = none)"),
            Field("n_inst", "u32", comment="instance count; n_inst u32 bases follow"),
            Field("flags", "u32", comment="zbc_comp_flags"),
        ),
        doc=("Component tree (16 bytes; COMP section, one record followed by n_inst "
             "u32 instance bases in pre-order, the root 0). Present when the image "
             "has a component tree (P1.5)."),
    ),
    Record(
        "zbc_layout",
        (
            Field("root", "u32", comment="StrId: the root component type"),
            Field("slots", "u32",
                  comment="StrId: JSON [[name, descriptor, offset], ...], one per component slot in slot order"),
            Field("insts", "u32",
                  comment="StrId: JSON [[path, type], ...], one per instance in pre-order (root path \"\")"),
            Field("size", "u32", comment="native size of the component object in bytes"),
        ),
        doc=("Component layout (16 bytes; LAYOUT section, one record; codegen/debug "
             "profiles): the component object's slots, each with its name, type and offset "
             "in the normative native layout (value-abi.md 2.1), and its instances' paths "
             "and types."),
    ),
    Record(
        "zbc_covg",
        (
            Field("doc", "u32",
                  comment="StrId: JSON [covergroup instance], indexed by COV_SAMPLE's covg id"),
        ),
        doc=("Covergroup instances (4 bytes; COVG section, one record): each one's "
             "coverpoints, bins, crosses and iff flags as plain data (F4.4; codegen/debug)."),
    ),
    Record(
        "zbc_exectab",
        (
            Field("n_exec", "u32", comment="zbc_execent records (executor instances)"),
            Field("n_groups", "u32", comment="zbc_execgrp records (executor groups)"),
            Field("n_members", "u32", comment="u32 group member executor ids"),
            Field("default_exec", "u32",
                  comment="the default executor's row + 1; 0: none (pssvm 8e)"),
        ),
        doc=("Executor table (16 bytes; EXEC section). Followed by zbc_execent[n_exec] "
             "(row index = the executor id requests carry), zbc_execgrp[n_groups] and "
             "u32 members[n_members] (F5.4a; codegen/debug profiles)."),
    ),
    Record(
        "zbc_execent",
        (
            Field("path", "u32", comment="StrId: the executor instance's path"),
            Field("type", "u32", comment="StrId: its component type"),
            Field("trait", "u32", comment="StrId: its trait struct type (\"\" = none)"),
        ),
        doc="One executor instance (12 bytes; EXEC section), in instance order.",
    ),
    Record(
        "zbc_execgrp",
        (
            Field("path", "u32", comment="StrId: the group instance's path"),
            Field("trait", "u32", comment="StrId: its trait struct type"),
            Field("first", "u32", comment="its first member in members[]"),
            Field("count", "u32", comment="its member count"),
        ),
        doc="One executor group (16 bytes; EXEC section); members are executor ids.",
    ),
    Record(
        "zbc_evtent",
        (
            Field("path", "u32", comment="StrId: the channel's instance path (``d0.c``)"),
            Field("kind", "u32", comment="zbc_evt_kind: the notify kind that feeds it"),
            Field("width", "u32", comment="element width in bits"),
            Field("flags", "u32", comment="bit 0: the element is signed"),
            Field("count", "u32", comment="component slot of its element count"),
            Field("head", "u32", comment="component slot of its ring head"),
            Field("buf", "u32", comment="component slot of its first ring element"),
            Field("depth", "u32", comment="ring elements (consecutive slots from buf)"),
        ),
        doc=("One host-fed source (32 bytes; EVENT section): every scalar channel_c "
             "instance, in slot order; the row index is the id N_CHAN notifies "
             "(F5.5a; codegen/debug profiles)."),
    ),
    Record(
        "zbc_regmap",
        (
            Field("doc", "u32",
                  comment="StrId: JSON {reg-id: register}, reg-id an instance index (decimal string)"),
        ),
        doc=("Register map (4 bytes; REGMAP section, one record): each register instance's "
             "path, offset, width, access and fields, as the host built it (debug profile)."),
    ),
    Record(
        "zbc_memmap",
        (
            Field("doc", "u32",
                  comment='StrId: JSON {"spaces": [...], "regions": [...]}'),
        ),
        doc=("Memory map (4 bytes; MEMMAP section, one record): each address space "
             "instance and the regions added to it, as the host built it (F7.2c; "
             "codegen/debug profiles)."),
    ),
    Record(
        "zbc_msgtab",
        (
            Field("n_msgs", "u32", comment="zbc_msg records"),
            Field("n_args", "u32", comment="zbc_msgarg records"),
            Field("n_items", "u32", comment="zbc_msgitem records"),
            Field("n_slots", "u32", comment="u32 frame-local slot numbers"),
            Field("n_strings", "u32", comment="u32 string offsets"),
            Field("n_model_strings", "u32",
                  comment="the first n_model_strings strings are the run's string ids"),
            Field("strb_size", "u32", comment="bytes of NUL-terminated string text"),
            Field("_pad", "u32"),
        ),
        doc=("Message table (32 bytes; MSG section). Followed, in order, by "
             "zbc_msg[n_msgs], zbc_msgarg[n_args], zbc_msgitem[n_items], "
             "u32 slots[n_slots], u32 str_off[n_strings] and the string text. "
             "Self-contained (not the STRB string table), so a runtime image "
             "can format message()/error()/fatal() text (P4.2g)."),
    ),
    Record(
        "zbc_msg",
        (
            Field("fmt", "u32", comment="string index of the format"),
            Field("n_args", "u32", comment="formatted values (slots: n_args + 1)"),
            Field("arg0", "u32", comment="index of its first zbc_msgarg"),
            Field("slot0", "u32",
                  comment="index of its first slot; slot0 = verbosity/status, then each value"),
        ),
        doc="One message()/error()/fatal() call site (16 bytes; MSG section).",
    ),
    Record(
        "zbc_msgarg",
        (
            Field("kind", "u8", comment="zbc_msg_kind"),
            Field("is_signed", "u8"),
            Field("_pad", "u16"),
            Field("width", "u32", comment="value width in bits"),
            Field("item0", "u32", comment="enum: index of its first zbc_msgitem"),
            Field("n_items", "u32", comment="enum: item count"),
        ),
        doc="The static type of one formatted value (16 bytes; MSG section).",
    ),
    Record(
        "zbc_msgitem",
        (
            Field("name", "u32", comment="string index of the item name"),
            Field("_pad", "u32"),
            Field("value", "u64", comment="item value (two's complement)"),
        ),
        doc="One enum item, for %n (16 bytes; MSG section).",
    ),
    # ----- Activation table (ACT section, P4.10a) ----- #
    Record(
        "zbc_acttab",
        (
            Field("n_acts", "u32"),
            Field("n_nodes", "u32"),
            Field("n_scopes", "u32"),
            Field("n_sites", "u32"),
            Field("n_cones", "u32"),
            Field("n_vars", "u32"),
            Field("n_cons", "u32"),
            Field("n_pools", "u32"),
            Field("n_claims", "u32"),
            Field("n_writes", "u32"),
            Field("n_picks", "u32"),
            Field("n_init", "u32", comment="zbc_actinit records"),
            Field("n_u32", "u32", comment="u32 list entries"),
            Field("n_strings", "u32", comment="u32 string offsets"),
            Field("strb_size", "u32", comment="bytes of NUL-terminated string text, padded to 8"),
            Field("blob_size", "u32", comment="bytes of dv-solve blobs after the string text"),
        ),
        doc=("Activation table (64 bytes; ACT section). Followed, in order, by "
             "zbc_actinit[n_init], zbc_act[n_acts], zbc_actnode[n_nodes], zbc_actscope[n_scopes], "
             "zbc_actsite[n_sites], zbc_actcone[n_cones], zbc_actvar[n_vars], zbc_actcon[n_cons], "
             "zbc_actpool[n_pools], zbc_actclaim[n_claims], zbc_actwrite[n_writes], "
             "zbc_actpick[n_picks], u32 lists[n_u32], u32 str_off[n_strings], the string text "
             "and the blob area (blob offsets are relative to its start; each blob 8-aligned). "
             "Record indices (*0) are section-global; ids inside an activation (node, scope, "
             "site, cone, pool) are local to it. Absent when no coroutine has an activity."),
    ),
    Record(
        "zbc_actinit",
        (
            Field("slot", "u32"),
            Field("unknown", "u32", comment="1: computed at traversal (value unused); 2: the node's uid, numbered per run"),
            Field("value", "u64", comment="constant initial value (two's complement)"),
        ),
        doc=("A non-rand attribute's initial value (16 bytes; ACT section)."),
    ),
    Record(
        "zbc_act",
        (
            Field("coro", "u32", comment="coroutine index the activation is rooted at"),
            Field("size", "u32", comment="object slots"),
            Field("root", "u32", comment="string"),
            Field("type", "u32", comment="string: root action type"),
            Field("flags", "u32", comment="bit 0: lookahead"),
            Field("node0", "u32"),
            Field("n_nodes", "u32"),
            Field("scope0", "u32"),
            Field("n_scopes", "u32"),
            Field("site0", "u32"),
            Field("n_sites", "u32"),
            Field("cone0", "u32"),
            Field("n_cones", "u32"),
            Field("pool0", "u32"),
            Field("n_pools", "u32"),
            Field("names0", "u32", comment="u32 list: string id per slot"),
            Field("n_names", "u32"),
            Field("init0", "u32"),
            Field("n_init", "u32"),
            Field("_pad", "u32"),
        ),
        doc=("One activation (80 bytes; ACT section)."),
    ),
    Record(
        "zbc_actnode",
        (
            Field("path", "u32", comment="string"),
            Field("type", "u32", comment="string"),
            Field("base", "u32"),
            Field("size", "u32"),
            Field("parent", "u32", comment="0xFFFFFFFF = none"),
            Field("decl_scope", "u32", comment="0xFFFFFFFF = none"),
            Field("comp0", "u32", comment="u32 list: candidate instances"),
            Field("n_comp", "u32"),
            Field("comp_slot", "u32", comment="0xFFFFFFFF = none"),
            Field("claim0", "u32", comment="zbc_actclaim"),
            Field("n_claims", "u32"),
            Field("swrite0", "u32", comment="zbc_actwrite: state writes"),
            Field("n_swrites", "u32"),
            Field("bwrite0", "u32", comment="zbc_actwrite: buffer writes"),
            Field("n_bwrites", "u32"),
            Field("pick0", "u32", comment="zbc_actpick"),
            Field("n_picks", "u32"),
            Field("child0", "u32", comment="u32 list: child node ids"),
            Field("n_children", "u32"),
            Field("addr0", "u32", comment="u32 list: address claims, 0 = none (F7.3d)"),
            Field("infer0", "u32", comment="u32 list: inferred for (consumer, ref string, "
                  "candidate); 0xFFFFFFFF = traversed (M9a3)"),
            Field("sinfer0", "u32", comment="u32 list: inferred state writers, then stream "
                  "partners; 0 = none (M9a5c, M9a6b)"),
        ),
        doc=("One action node of an activation (88 bytes; ACT section). ``addr0`` "
             "indexes the u32 lists at ``n`` then ``n`` quads (space string, addr slot, "
             "size slot, permanent slot or 0xFFFFFFFF) of the node's address claims, "
             "which the run's ledger keeps apart; list index 0 never holds them. A "
             "contiguous claim has addr slot 0xFFFFFFFF; after the quads come ``m`` "
             "then, per contiguous claim, its quad index, alignment slot (or "
             "0xFFFFFFFF), ``$addr`` slot the run writes the placed address to (or "
             "0xFFFFFFFF; F7.4b), ``k`` and ``k`` (trait field string, slot) pairs (F7.3e). "
             "``sinfer0`` indexes ``n`` then, per state input, its ref string, ``m`` and ``m`` "
             "(instance, ``k``, ``k`` hidden node ids) groups (M9a5c); after them, in the same "
             "form, the inferred stream partners per unpaired stream reference (M9a6b)."),
    ),
    Record(
        "zbc_actscope",
        (
            Field("kind", "u32", comment="index in ScopeKind order"),
            Field("parent", "u32", comment="0xFFFFFFFF = none"),
            Field("node", "u32"),
            Field("_pad", "u32"),
        ),
        doc=("One activity block (16 bytes; ACT section)."),
    ),
    Record(
        "zbc_actsite",
        (
            Field("owner", "u32"),
            Field("target", "u32"),
            Field("scope", "u32"),
            Field("_pad", "u32"),
        ),
        doc=("One traversal site (16 bytes; ACT section)."),
    ),
    Record(
        "zbc_actcone",
        (
            Field("id", "u32"),
            Field("node0", "u32", comment="u32 list"),
            Field("n_nodes", "u32"),
            Field("var0", "u32", comment="zbc_actvar"),
            Field("n_vars", "u32"),
            Field("con0", "u32", comment="zbc_actcon"),
            Field("n_cons", "u32"),
            Field("base_off", "u32", comment="blob area: dv-solve blob of the cone's variables"),
            Field("base_size", "u32"),
            Field("_pad", "u32"),
        ),
        doc=("One solve cone (40 bytes; ACT section). A solve compiles the base blob, then adds "
             "the blob of each constraint in force (dv-solve add_constraint)."),
    ),
    Record(
        "zbc_actvar",
        (
            Field("name", "u32", comment="string"),
            Field("node", "u32"),
            Field("slot", "u32"),
            Field("width", "u32"),
            Field("flags", "u32", comment=("1 signed, 2 rand, 4 busy_lock, 8 array (a list the constraints read), "
                                              "16 attr (a resource instance's rand attribute: pool in pick); "
                                              "bits 16-31: busy_word (busy mask word; 63 instances each: 63*w..), "
                                              "or with 16 the attribute's instance")),
            Field("comp_read", "u32", comment="0xFFFFFFFF = none"),
            Field("loop_local", "u32", comment="string; 0xFFFFFFFF = none"),
            Field("loop_node", "u32", comment="0xFFFFFFFF = none"),
            Field("live_from", "u32", comment="0xFFFFFFFF = none"),
            Field("busy_pool", "u32", comment="0xFFFFFFFF = none"),
            Field("pick", "u32", comment="0xFFFFFFFF = none"),
            Field("loop0", "u32", comment=("u32 list: n, then n (coro, frame slot) pairs of the "
                                           "coros holding loop_local; 0xFFFFFFFF = none")),
        ),
        doc=("One cone variable (48 bytes; ACT section)."),
    ),
    Record(
        "zbc_actcon",
        (
            Field("kind", "u32", comment="0 type, 1 activity, 2 with, 3 comp"),
            Field("owner", "u32"),
            Field("scope", "u32", comment="0xFFFFFFFF = none"),
            Field("site", "u32", comment="0xFFFFFFFF = none"),
            Field("node0", "u32", comment="u32 list"),
            Field("n_nodes", "u32"),
            Field("blob_off", "u32", comment="blob area: the cone's variables and this constraint"),
            Field("blob_size", "u32"),
            Field("text", "u32", comment="string: the constraint as written, for unsat messages"),
            Field("_pad", "u32"),
        ),
        doc=("One cone constraint (40 bytes; ACT section)."),
    ),
    Record(
        "zbc_actpool",
        (
            Field("id", "u32"),
            Field("inst", "u32"),
            Field("name", "u32", comment="string"),
            Field("kind", "u32", comment="string: buffer/stream/state/resource"),
            Field("size", "u32", comment="0xFFFFFFFF = none"),
            Field("slot0", "u32", comment="u32 list"),
            Field("n_slots", "u32"),
            Field("leaf0", "u32", comment="u32 list of strings"),
            Field("n_leaves", "u32"),
            Field("init0", "u32", comment="u32 list"),
            Field("n_init", "u32"),
            Field("_pad", "u32"),
        ),
        doc=("One pool instance (48 bytes; ACT section)."),
    ),
    Record(
        "zbc_actclaim",
        (
            Field("ref", "u32", comment="string"),
            Field("lock", "u32", comment="1 lock, 0 share"),
            Field("iid_slot", "u32"),
            Field("pool0", "u32", comment="u32 list of (inst, pool) pairs"),
            Field("n_pools", "u32", comment="pairs"),
            Field("_pad", "u32"),
        ),
        doc=("One resource claim (24 bytes; ACT section)."),
    ),
    Record(
        "zbc_actwrite",
        (
            Field("ref", "u32", comment="string"),
            Field("src0", "u32", comment="u32 list"),
            Field("n_src", "u32"),
            Field("pool0", "u32", comment="u32 list of (inst, pool) pairs"),
            Field("n_pools", "u32", comment="pairs"),
            Field("_pad", "u32"),
        ),
        doc=("One state or buffer output (24 bytes; ACT section)."),
    ),
    Record(
        "zbc_actpick",
        (
            Field("ref", "u32", comment="string"),
            Field("slot0", "u32", comment="u32 list"),
            Field("n_slots", "u32"),
            Field("sel_slot", "u32"),
            Field("pool0", "u32", comment="u32 list of (inst, pool) pairs"),
            Field("n_pools", "u32", comment="pairs"),
            Field("infer0", "u32", comment="u32 list: per instance, (inst, n, n inferred nodes) (M9a3)"),
            Field("n_infer", "u32", comment="instances"),
        ),
        doc=("One unbound buffer input (32 bytes; ACT section)."),
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
            Field("n_input", "u32", comment="input pair count (after the writeback pairs)"),
        ),
        doc=("SOLVE descriptor (32 bytes). SOLVE section is zbc_solve[]; arg0 of a "
             "SOLVE instr indexes it. The OPLIST pool holds, from oplist_off, "
             "n_writeback interleaved (field_slot, var_id) u32 pairs -- the value ABI "
             "write-back keyed by object slot -- then n_input more: non-rand fields a "
             "constraint reads, each var pinned to its field's value before the solve "
             "and not written back (with ZBC_SOLVE_ARRAYS, some are lists: see that flag). When prob_len > 0, the (prob_off, "
             "prob_len) slice of the SPROB pool is a relocatable dv-solve SolveProblem "
             "blob: the engine compiles + solves it with the drawn seed and writes "
             "dvs_solver_get_value(var_id) back to each field_slot. When prob_len == 0 the "
             "minimal M1 randomizer applies: slot = seed + var_id (FixedSolveBackend, "
             "base 0). Seed is seed_value if SEED_FIXED else the frame's next draw."),
    ),
    Record(
        "zbc_entry",
        (
            Field("name", "u32", comment="StrId: action coroutine name, or <comp type>::<fn>"),
            Field("coro", "u32", comment="CORO index the host starts"),
            Field("kind", "u32", comment="ZBC_ENTRY_*"),
            Field("comp_type", "u32", comment="StrId of a function's component type (0=none)"),
            Field("sig", "u32", comment="StrId of the JSON signature {params, returns} (0=none)"),
            Field("qname", "u32",
                  comment="StrId of an action entry's qualified action type (0=none)"),
            Field("obj", "u32",
                  comment="StrId of the JSON action-object layout (0=none)"),
            Field("flags", "u32", comment="zbc_entry_flags"),
            Field("fixed", "u32",
                  comment="StrId of the JSON list of fields an exported action fixes (0=none)"),
            Field("_pad", "u32"),
        ),
        doc=("Entry point (40 bytes). ENTRY section is zbc_entry[]: where a host may "
             "enter the image. A function's arguments go in the root frame's args "
             "(LD_ARG), one per leaf of each parameter, in the order sig.params lists "
             "them; sig is JSON {\"params\": [[name, type descriptor], ...], "
             "\"returns\": descriptor|null}. An action's object layout (obj) is JSON "
             "[[name, type descriptor], ...] in slot order; a rand field's descriptor "
             "has \"rand\": true."),
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
            EnumMember("ZBC_SEC_ENTRY", 16, "entry points (zbc_entry[])"),
            EnumMember("ZBC_SEC_LOCAL", 17, "named locals (zbc_localent[]; debug lowering)"),
            EnumMember("ZBC_SEC_SRC", 18, "PSS source texts (zbc_srcent[]; debug profile)"),
            EnumMember("ZBC_SEC_IDENT", 19, "API hash + generation id (zbc_ident; debug profile)"),
            EnumMember("ZBC_SEC_META", 20, "host metadata documents (zbc_metaent[]; debug profile)"),
            EnumMember("ZBC_SEC_COMP", 21, "component tree (zbc_comp + u32 bases[])"),
            EnumMember("ZBC_SEC_MSG", 22, "message table (zbc_msgtab + arrays) and run strings"),
            EnumMember("ZBC_SEC_REG", 23, "register accesses (zbc_regent[]; debug lowering)"),
            EnumMember("ZBC_SEC_CHAN", 24, "channel operations (zbc_chanent[]; debug lowering)"),
            EnumMember("ZBC_SEC_COV", 25, "coverage items (zbc_covent[]; debug lowering)"),
            EnumMember("ZBC_SEC_IMPORT", 26, "user import names (zbc_impent[])"),
            EnumMember("ZBC_SEC_ACT", 27, "activation table (zbc_acttab + arrays)"),
            EnumMember("ZBC_SEC_LAYOUT", 28, "component slot and instance names (zbc_layout)"),
            EnumMember("ZBC_SEC_REGMAP", 29, "register map (zbc_regmap)"),
            EnumMember("ZBC_SEC_SEEDKEY", 30,
                       "each coroutine's seed key, u64[coro count] (keyed seeds, F3.9)"),
            EnumMember("ZBC_SEC_COVG", 31, "covergroup instances (zbc_covg; F4.4)"),
            EnumMember("ZBC_SEC_EXEC", 32,
                       "executors and executor groups (zbc_exectab + arrays; F5.4a)"),
            EnumMember("ZBC_SEC_EVENT", 33,
                       "host-fed channels (zbc_evtent[]; F5.5a)"),
            EnumMember("ZBC_SEC_MEMMAP", 34,
                       "address spaces and regions (zbc_memmap; F7.2c)"),
            EnumMember("ZBC_SEC_API", 35, "import-interface APIs (zbc_apient[])"),
            EnumMember("ZBC_SEC_CLASS", 36, "types: layouts and methods (zbc_class[])"),
            EnumMember("ZBC_SEC_ROOT", 37, "root candidates (zbc_rootent[])"),
            EnumMember("ZBC_SEC_ROOTIMG", 38,
                       "the other root candidates' images, each 8-aligned (ROOT locates them)"),
        ),
        doc="Section kinds. Readers skip unknown kinds (forward-compat).",
    ),
    Enum(
        "zbc_evt_kind",
        (
            EnumMember("ZBC_EVT_CHAN", 2, "a channel_c; fed by notify N_CHAN"),
        ),
        doc="zbc_evtent.kind: equal to the notify kind (N_IRQ 1 has no rows).",
    ),
    Enum(
        "zbc_msg_kind",
        (
            EnumMember("ZBC_MSG_INT", 0, "integer"),
            EnumMember("ZBC_MSG_BOOL", 1, "bool"),
            EnumMember("ZBC_MSG_ENUM", 2, "enum (items follow)"),
            EnumMember("ZBC_MSG_STRING", 3, "string id"),
            EnumMember("ZBC_MSG_CHANDLE", 4, "chandle"),
            EnumMember("ZBC_MSG_FLOAT", 5, "float (binary64 bits)"),
        ),
        doc="zbc_msgarg.kind: how a formatted value prints.",
    ),
    Enum(
        "zbc_hdr_flags",
        (
            EnumMember("ZBC_HDR_HAS_PROV", 0x0001, "provenance sections present"),
            EnumMember("ZBC_HDR_PROFILE_RUNTIME", 0x0002, "runtime profile (provenance stripped)"),
            EnumMember("ZBC_HDR_COMP_INIT", 0x0004,
                       "a coroutine constructs the component tree before the entry (P1.5)"),
            EnumMember("ZBC_HDR_PROFILE_DEBUG", 0x0008,
                       "debug profile (provenance kept; may carry PSS sources)"),
            EnumMember("ZBC_HDR_EXT_PSS", 0x0100,
                       "the pss extension: opcodes 0x80-0xBF and the ACT, COMP, EXEC, EVENT, "
                       "MEMMAP and COVG sections"),
        ),
        doc=("Header flag bits. Bits 0x0100-0x8000 (ZBC_HDR_EXT_MASK) each declare an "
             "extension profile the image uses; a loader refuses an image declaring one it "
             "does not implement, and an opcode in an extension's range the image does not "
             "declare."),
        is_flags=True,
    ),
    Enum(
        "zbc_profile",
        (
            EnumMember("ZBC_HDR_EXT_MASK", 0xFF00, "header flag bits that declare extensions"),
            EnumMember("ZBC_OP_CORE_FIRST", 0x00, "core opcodes: every implementation"),
            EnumMember("ZBC_OP_CORE_LAST", 0x7F),
            EnumMember("ZBC_OP_PSS_FIRST", 0x80, "the pss extension's opcodes (ZBC_HDR_EXT_PSS)"),
            EnumMember("ZBC_OP_PSS_LAST", 0xBF),
        ),
        doc=("Profiles: the core opcode and section set, and extensions an image declares in "
             "its header flags, each with its own opcode range (0xC0-0xFF: the next front end)."),
    ),
    Enum(
        "zbc_secf_flags",
        (
            EnumMember("ZBC_SECF_COMPRESSED", 0x0001, "payload is compressed"),
            EnumMember("ZBC_SECF_LINE_STMT", 0x0002,
                       "LINE: each entry starts one PSS statement (debug lowering)"),
        ),
        doc="Per-section flag bits.",
        is_flags=True,
    ),
    Enum(
        "zbc_comp_flags",
        (
            EnumMember("ZBC_COMP_F_NODE_INST", 0x0001,
                       "an activation node runs in an instance other than its parent's "
                       "(a static offset or one its solve chooses)"),
            EnumMember("ZBC_COMP_F_RUN_HOOKS", 0x0002,
                       "coroutines init_coro+1 ($run_start: exec run_start, top-down) and "
                       "init_coro+2 ($run_end: exec run_end, bottom-up) run before and after "
                       "a root action"),
        ),
        doc="COMP flag bits.",
        is_flags=True,
    ),
    Enum(
        "zbc_local_kind",
        (
            EnumMember("ZBC_LOCAL_INT", 0, "integer"),
            EnumMember("ZBC_LOCAL_BOOL", 1, "bool"),
            EnumMember("ZBC_LOCAL_ENUM", 2, "enum (its value)"),
            EnumMember("ZBC_LOCAL_STRING", 3, "string"),
            EnumMember("ZBC_LOCAL_CHANDLE", 4, "chandle"),
            EnumMember("ZBC_LOCAL_OPAQUE", 5, "opaque handle"),
            EnumMember("ZBC_LOCAL_FLOAT", 6, "float (binary64 bits)"),
        ),
        doc="What a named local's slot holds.",
    ),
    Enum(
        "zbc_local_flags",
        (
            EnumMember("ZBC_LOCAL_F_SIGNED", 0x01, "signed integer"),
        ),
        doc="Named-local flag bits.",
        is_flags=True,
    ),
    Enum(
        "zbc_reg_flags",
        (
            EnumMember("ZBC_REG_F_DISCARD", 0x01, "a read whose value the statement drops"),
        ),
        doc="Register-access flag bits.",
        is_flags=True,
    ),
    Enum(
        "zbc_cov_flags",
        (
            EnumMember("ZBC_COV_F_ARM", 0x01, "a branch arm (else a statement)"),
            EnumMember("ZBC_COV_F_IMPLICIT", 0x02, "an arm the code did not write (an implicit else)"),
        ),
        doc="Coverage-item flag bits.",
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
            EnumMember("ZBC_SOLVE_PROCEDURAL", 0x0002,
                       "a procedural randomize: each (k, var_id) pair keeps var_id's value "
                       "in the frame's solved slot k for SOLVE_GET; nothing is written back"),
            EnumMember("ZBC_SOLVE_ARRAYS", 0x0004,
                       "the problem reads lists (dv-solve arrays): each input pair whose "
                       "var_id >= the blob's n_vars is (key, n_vars + array_id), key = the "
                       "field slot holding the list's heap handle (procedural: the SOLVE_PIN "
                       "id it is pinned at); the blob's ArraySpec gives each array's element "
                       "width, signedness and kind (RAND lists are solved and written back). "
                       "An engine without heap lists must refuse it"),
        ),
        doc="SOLVE descriptor flag bits.",
        is_flags=True,
    ),
    Enum(
        "zbc_entry_kind",
        (
            EnumMember("ZBC_ENTRY_ACTION", 0, "a top-level action coroutine"),
            EnumMember("ZBC_ENTRY_FUNCTION", 1, "a component function"),
        ),
        doc="Entry-point kinds.",
    ),
    Enum(
        "zbc_entry_flags",
        (
            EnumMember("ZBC_ENTRY_F_EXPORTED", 0x0001,
                       "an exported action: a host may run it by name and a typed API "
                       "gives it a method (PSS export statements, --export-action)"),
        ),
        doc="Entry-point flag bits.",
        is_flags=True,
    ),
    Enum(
        "zbc_imp_flags",
        (
            EnumMember("ZBC_IMP_F_BLOCKING", 0x0001, "the call may consume host time"),
            EnumMember("ZBC_IMP_F_TARGET", 0x0002, "declared import target"),
            EnumMember("ZBC_IMP_F_SOLVE", 0x0004, "declared import solve"),
        ),
        doc="Import flag bits.",
        is_flags=True,
    ),
    Enum(
        "zbc_api_kind",
        (
            EnumMember("ZBC_API_BUILTIN", 0, "an API the engine defines (mem, config, seq)"),
            EnumMember("ZBC_API_USER", 1, "the PSS imports of one package or component"),
        ),
        doc="API kinds.",
    ),
    Enum(
        "zbc_api_id",
        (
            EnumMember("ZBC_API_MEM", 0, "memory and register access (read, write)"),
            EnumMember("ZBC_API_CONFIG", 1, "solve-time configuration (reserved)"),
            EnumMember("ZBC_API_SEQ", 2, "host sequences (reserved)"),
            EnumMember("ZBC_API_USER_FIRST", 16, "the first user API id"),
        ),
        doc="API ids (host-abi.md, the import interface).",
    ),
    Enum(
        "zbc_class_kind",
        (
            EnumMember("ZBC_CLASS_COMPONENT", 0),
            EnumMember("ZBC_CLASS_ACTION", 1),
            EnumMember("ZBC_CLASS_STRUCT", 2),
        ),
        doc="CLASS row kinds.",
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


#: extension profiles: name -> (header flag, first opcode, last opcode)
EXTENSIONS: Dict[str, Tuple[int, int, int]] = {
    "pss": (flag_value("zbc_hdr_flags", "ZBC_HDR_EXT_PSS"),
            flag_value("zbc_profile", "ZBC_OP_PSS_FIRST"),
            flag_value("zbc_profile", "ZBC_OP_PSS_LAST")),
}
EXT_MASK = flag_value("zbc_profile", "ZBC_HDR_EXT_MASK")


def op_extension(op: int) -> Optional[str]:
    """The extension whose opcode range holds *op*, or None for a core opcode.
    An opcode at or above 0x80 that no extension claims is ``"?"``."""
    if op <= flag_value("zbc_profile", "ZBC_OP_CORE_LAST"):
        return None
    for name, (_, lo, hi) in EXTENSIONS.items():
        if lo <= op <= hi:
            return name
    return "?"
