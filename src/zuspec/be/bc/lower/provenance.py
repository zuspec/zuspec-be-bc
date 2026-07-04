"""
provenance.py -- propagate source provenance into ZBC (P1-4, design D§4.3).

``ProvenanceBuilder`` turns an IR node's ``loc`` (source span) and ``comment``
(``Stmt.comment``, already carried by Layer-0) plus an optional original name into
a :class:`~zuspec.be.bc.model.Prov` entry, returning its ``src_ref`` index. The
interpreter and codegen consumers read these back; T1-E asserts the comment text
and names survive lowering.

M1 does not dedup provenance entries (one per annotated construct); interning is a
size optimization, not a correctness requirement.
"""

from ..model import ProvTable, Prov, Comment, CMT_LEADING

#: Origin-node-kind codes (stable ints; mirror design's node_kind enum intent).
NODE_KIND = {
    "ScCoroutine": 1,
    "ScExecBlock": 2,
    "ScWait": 3,
    "ScJoin": 4,
    "ScPar": 5,
    "ScSelect": 6,
    "ScInvoke": 7,
    "ScSpawn": 8,
    "ScImport": 9,
    "ScSolveProblem": 10,
    "StmtAssign": 11,
    "StmtExpr": 12,
    "StmtReturn": 13,
}


class ProvenanceBuilder:
    def __init__(self):
        self.table = ProvTable()

    def node_kind(self, node) -> int:
        return NODE_KIND.get(type(node).__name__, 0)

    def src_ref(self, node, name: str = "", flags: int = 0) -> int:
        """Return a ``src_ref`` for ``node`` (0 = none if there is nothing to record)."""
        loc = getattr(node, "loc", None)
        comment = getattr(node, "comment", None)
        if loc is None and not comment and not name:
            return 0

        file = ""
        line = col = 0
        if loc is not None:
            file = getattr(loc, "file", None) or ""
            line = getattr(loc, "line", None) or 0
            col = getattr(loc, "pos", None) or 0

        comments = [Comment(comment, kind=CMT_LEADING, line=line)] if comment else []
        prov = Prov(
            name=name,
            node_kind=self.node_kind(node),
            file=file,
            line=line,
            col=col,
            flags=flags,
            comments=comments,
        )
        return self.table.add(prov)
