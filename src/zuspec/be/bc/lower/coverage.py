"""
coverage.py -- sample an action's inline covergroups (F4.4).

An action with covergroups samples them once its values are final: after
``post_solve`` (else after its solve), before its body. :func:`with_cov_sample`
marks that point in the coroutine with an ``ScExecBlock`` of kind
``cov_sample`` whose ``stmts`` are the ``CovergroupDef``s; :func:`lower_cov_sample`
lowers the marker to ordinary bytecode that evaluates each coverpoint (and
``iff``) into registers, stages them with ``ARG``, then one ``COV_SAMPLE covg, n`` per covergroup
(staged, as for CALL: an instruction has at most four inline arguments).

Bins, crosses and iff flags are data, not code: :func:`covg_doc` describes a
covergroup as JSON for the image's COVG section, read by the coverage service.
A bin value or ``with``/select expression it cannot describe is a lowering
error, never dropped.
"""
from __future__ import annotations

import copy
import dataclasses as dc
from typing import List, Optional

from zuspec.ir.core import scenario as SC
from zuspec.ir.core import expr as E
from zuspec.ir.core.data_type import DataTypeBundle, DataTypeClass, DataTypeStruct

from ..model import Op
from .errors import LoweringError

#: the marker's ``ScExecBlock.kind``
COV_SAMPLE_KIND = "cov_sample"


def action_covergroups(coro, types) -> list:
    """The covergroups *coro*'s action samples (none if it has no type): its
    own, then those of each ``rand`` struct field (F4.9, LRM 19.1: a
    struct's covergroups are sampled when the struct is randomized)."""
    t = (types or {}).get(coro.action_type) if coro.action_type else None
    cgs = list(getattr(t, "covergroups", None) or [])
    if t is not None:
        cgs += _struct_covergroups(t, coro.action_type, (), types or {})
    return cgs


def _struct_covergroups(t, owner: str, path: tuple, types, depth: int = 0) -> list:
    """The covergroups of the ``rand`` struct fields of *t* (recursively),
    each re-rooted at its field (``self.v`` -> ``self.s.v``) and named by
    its place in *owner* (``pss_top::A::s.cg``); a covergroup declared
    inline in struct ``S`` is reported as type ``S::cg``."""
    out = []
    if depth > 8:
        return out
    for f in getattr(t, "fields", None) or []:
        if not getattr(f, "rand_kind", None):
            continue
        st = f.datatype
        if not _is_struct(st):
            st = types.get(getattr(f, "type_qname", None) or "")
            if not _is_struct(st):
                continue
        fpath = path + (f.name,)
        for cg in getattr(st, "covergroups", None) or []:
            c = _rebase(copy.deepcopy(cg), fpath)
            inst = cg.instance_name or cg.name
            c.type_name = cg.type_name or f"{cg.parent or st.name}::{inst}"
            c.instance_name = ".".join(fpath + (inst,))
            c.parent = owner
            out.append(c)
        out += _struct_covergroups(st, owner, fpath, types, depth + 1)
    return out


def _is_struct(t) -> bool:
    """A plain struct: not an action, component or bundle (an action handle's
    covergroups are sampled by its own traversal)."""
    return (isinstance(t, DataTypeStruct)
            and not isinstance(t, (DataTypeClass, DataTypeBundle)))


def _rebase(node, path: tuple):
    """*node* with each ``self`` replaced by ``self.<path>``."""
    if isinstance(node, E.TypeExprRefSelf):
        e = node
        for name in path:
            e = E.ExprAttribute(value=e, attr=name)
        return e
    if isinstance(node, list):
        return [_rebase(n, path) for n in node]
    if dc.is_dataclass(node) and not isinstance(node, type):
        for f in dc.fields(node):
            if f.name == "loc":
                continue
            v = getattr(node, f.name)
            if isinstance(v, list) or dc.is_dataclass(v):
                setattr(node, f.name, _rebase(v, path))
    return node


def with_cov_sample(coro, types):
    """*coro* with a ``cov_sample`` marker at its sampling point, when its
    action has covergroups: after ``post_solve``, else after the solve, else
    after its initial values and ``pre_solve``."""
    cgs = action_covergroups(coro, types)
    if not cgs:
        return coro
    body = list(coro.body)
    at = None
    for i, s in enumerate(body):
        if isinstance(s, SC.ScExecBlock) and s.kind == "post_solve":
            at = i + 1
    if at is None:
        for i, s in enumerate(body):
            if isinstance(s, SC.ScSolveProblem):
                at = i + 1
                break
    if at is None:
        at = 0
        while (at < len(body) and isinstance(body[at], SC.ScExecBlock)
               and body[at].kind in ("init", "pre_solve")):
            at += 1
    body.insert(at, SC.ScExecBlock(kind=COV_SAMPLE_KIND, stmts=cgs))
    return dc.replace(coro, body=body)


def lower_cov_sample(ctx, s) -> None:
    """Lower a ``cov_sample`` marker: per covergroup, its coverpoint values
    and iff flags into registers, then ``COV_SAMPLE``."""
    from .procedural import ev, truth
    lw = ctx.lowerer
    for cg in s.stmts:
        regs: List[int] = []
        widths = []
        for cp in cg.coverpoints:
            if cp.target is None:
                raise LoweringError(f"coverpoint {cp.name!r} has no expression",
                                    loc=cp.loc)
            r, t = ev(ctx, cp.target)
            regs.append(r)
            widths.append((t.width, bool(t.signed),
                           t.items if t.kind == "enum" else None))
            if cp.iff is not None:
                regs.append(truth(ctx, *ev(ctx, cp.iff)))
        for cr in cg.crosses:
            if cr.iff is not None:
                regs.append(truth(ctx, *ev(ctx, cr.iff)))
        cid = covg_id(lw, ctx.coro_name, cg, widths)
        for i, r in enumerate(regs):
            ctx.emit(Op.ARG, (r, i), src_ref=ctx.cur_src_ref)
        ctx.emit(Op.COV_SAMPLE, (cid, len(regs)), src_ref=ctx.cur_src_ref)


def covg_id(lw, coro_name: str, cg, widths) -> int:
    """*cg*'s index in ``lw.covgs`` (the COVG section), added on first use."""
    name = f"{cg.parent}::{cg.instance_name or cg.name}" if cg.parent else cg.name
    for i, d in enumerate(lw.covgs):
        if d["name"] == name:
            return i
    lw.covgs.append(covg_doc(cg, name, widths))
    return len(lw.covgs) - 1


def covg_doc(cg, name: str, widths) -> dict:
    """*cg* as plain JSON data: coverpoints (width, signedness, iff, bins, and
    ``items`` [[name, value], ...] for an enum) and crosses (coverpoints, iff,
    select bins)."""
    cps = []
    for cp, (w, sg, items) in zip(cg.coverpoints, widths):
        d = {"name": cp.name, "width": w, "signed": sg,
             "iff": cp.iff is not None,
             "bins": [_bin_doc(cp.name, b) for b in cp.bins]}
        if items:
            d["items"] = [list(i) for i in items]
        _options(d, cp, ("at_least", "weight", "goal", "auto_bin_max", "comment"))
        cps.append(d)
    crosses = [_options({"name": cr.name, "coverpoints": list(cr.coverpoints),
                         "iff": cr.iff is not None,
                         "bins": [_at({"name": b.name, "kind": b.kind.name.lower(),
                                       "select": _expr_doc(b.select, b.loc)}, b)
                                  for b in cr.bins]},
                        cr, ("at_least", "weight", "goal", "comment"))
               for cr in cg.crosses]
    doc = {"name": name, "instance": cg.instance_name or cg.name,
           "coverpoints": cps, "crosses": crosses}
    if getattr(cg, "options", None) is not None:
        _options(doc, cg.options, ("at_least", "weight", "goal", "auto_bin_max",
                                   "per_instance", "comment"))
    # An instance of a covergroup type: ``name`` stays the instance's own
    # (``A::cx``), ``type`` is what a report sums the instances under.
    if getattr(cg, "type_name", None):
        doc["type"] = cg.type_name
    return doc


def _options(d: dict, obj, keys) -> dict:
    """*d* with ``options``: those of *keys* whose value on *obj* (a
    dataclass) differs from the default (none: no ``options`` key)."""
    defaults = {f.name: f.default for f in dc.fields(obj)}
    opts = {k: getattr(obj, k) for k in keys
            if k in defaults and getattr(obj, k) != defaults[k]}
    if opts:
        d["options"] = opts
    return d


def _bin_doc(cp_name: str, b) -> dict:
    vals = []
    for v in b.values:
        if isinstance(v, E.ExprRange):
            vals.append([_const(v.lower, cp_name, b), _const(v.upper, cp_name, b)])
        else:
            vals.append(_const(v, cp_name, b))
    d = {"name": b.name, "kind": ("default" if b.is_default else b.kind.name.lower()),
         "values": vals, "array": bool(b.array)}
    if b.array_size is not None:
        d["size"] = _const(b.array_size, cp_name, b)
    if b.with_expr is not None:
        d["with"] = _expr_doc(b.with_expr, b.loc)
    return _at(d, b)


def _at(d: dict, b) -> dict:
    """An illegal bin's doc with its source ``file`` and ``line``: where a
    hit on it is reported (``cov.covergroup.illegal_hits``)."""
    if d["kind"] == "illegal" and getattr(b.loc, "line", None):
        d["file"] = getattr(b.loc, "file", None) or ""
        d["line"] = int(b.loc.line)
    return d


def _const(e, cp_name: str, b) -> Optional[int]:
    """A bin bound as an int (None: an open end)."""
    if e is None:
        return None
    v = _fold(e)
    if v is None:
        raise LoweringError(f"coverpoint {cp_name!r} bin {b.name!r}: value is not "
                            f"a constant bc can describe yet", loc=b.loc)
    return v


def _fold(e) -> Optional[int]:
    if isinstance(e, E.ExprConstant) and isinstance(e.value, (int, bool)):
        return int(e.value)
    if isinstance(e, E.ExprUnary) and e.op.name in ("USub", "Neg"):
        v = _fold(e.operand)
        return None if v is None else -v
    return None


def _expr_doc(e, loc) -> Optional[dict]:
    """A ``with`` filter or cross select as a small JSON tree: ``{"c": v}``,
    ``{"ref": name}`` (a coverpoint, or ``item``), ``{"op", "args"}``."""
    if e is None:
        return None
    v = _fold(e)
    if v is not None:
        return {"c": v}
    if isinstance(e, E.ExprAttribute) and type(e.value).__name__ == "TypeExprRefSelf":
        return {"ref": e.attr}
    for cls in ("ExprRefLocal", "ExprRefUnresolved", "ExprRefParam"):
        if type(e).__name__ == cls and isinstance(getattr(e, "name", None), str):
            return {"ref": e.name}
    if isinstance(e, E.ExprBin):
        return {"op": e.op.name, "args": [_expr_doc(e.lhs, loc), _expr_doc(e.rhs, loc)]}
    if isinstance(e, E.ExprUnary):
        return {"op": e.op.name, "args": [_expr_doc(e.operand, loc)]}
    if isinstance(e, E.ExprBool):
        return {"op": e.op.name, "args": [_expr_doc(x, loc) for x in e.values]}
    if isinstance(e, E.ExprCompare) and len(e.ops) == 1:
        return {"op": e.ops[0].name,
                "args": [_expr_doc(e.left, loc), _expr_doc(e.comparators[0], loc)]}
    raise LoweringError(f"covergroup expression {type(e).__name__} is not "
                        f"supported on bc yet", loc=loc)
