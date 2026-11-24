"""Covergroup samples binned against COVG descriptions (full-PSS plan F4.5).

The descriptions are written by hand in the ``covg_doc`` format (be-bc
``lower/coverage.py``); pssc's tests bin lowered PSS runs."""

import pytest

from zuspec.be.bc.cov.covergroup import (AUTO_BIN_MAX, Binner, CovError,
                                         bin_samples, counted_bins, merge)


def _cp(name, bins, width=4, signed=False, iff=False):
    return {"name": name, "width": width, "signed": signed, "iff": iff,
            "bins": bins}


def _b(name, values, kind="bins", array=False, size=None, with_=None):
    d = {"name": name, "kind": kind, "values": values, "array": array}
    if size is not None:
        d["size"] = size
    if with_ is not None:
        d["with"] = with_
    return d


def _cg(cps, crosses=(), name="pss_top::A::cs"):
    return {"name": name, "instance": "cs", "coverpoints": cps,
            "crosses": list(crosses)}


def _bins(doc, *samples, cp="cp_x"):
    r = bin_samples([doc], [(0, list(s)) for s in samples])[doc["name"]]
    return r["coverpoints"][cp]["bins"], r


def test_corpus_bins_forms():
    """pss-corpus cov/bins_forms: value, range, array and default bins."""
    doc = _cg([_cp("cp_x", [_b("two", [2]), _b("mid", [[4, 7]]),
                            _b("v", [[8, 10]], array=True),
                            _b("other", [], kind="default")])])
    bins, r = _bins(doc, [2], [2], [5], [9], [9], [10], [14])
    assert r["samples"] == 7
    assert bins == {"two": 2, "mid": 1, "v[8]": 0, "v[9]": 2, "v[10]": 1,
                    "other": 1}
    assert r["coverpoints"]["cp_x"]["kinds"] == {"other": "default"}


def test_open_ends_and_with_filter():
    odd = {"op": "Eq", "args": [{"op": "Mod", "args": [{"ref": "item"}, {"c": 2}]},
                                {"c": 1}]}
    doc = _cg([_cp("cp_x", [_b("hi", [[12, None]]), _b("lo", [[None, 1]]),
                            _b("odd", [[0, 5]], array=True, with_=odd)])])
    bins, _ = _bins(doc, [0], [3], [15], [4])
    assert bins == {"hi": 1, "lo": 1, "odd[1]": 0, "odd[3]": 1, "odd[5]": 0}


def test_sized_array_spreads_values_remainder_last():
    doc = _cg([_cp("cp_x", [_b("q", [[0, 6]], array=True, size=3)])])
    bins, _ = _bins(doc, [0], [1], [2], [3], [6], [5])
    # 7 values over 3 bins: {0,1} {2,3} {4,5,6}
    assert bins == {"q[0]": 2, "q[1]": 2, "q[2]": 2}


def test_ignore_and_illegal_take_values_out_of_other_bins():
    doc = _cg([_cp("cp_x", [_b("all", [[0, 15]]), _b("ig", [3], kind="ignore"),
                            _b("bad", [7], kind="illegal")])])
    bins, r = _bins(doc, [3], [7], [8])
    assert bins == {"all": 1, "ig": 1, "bad": 1}
    assert counted_bins({"cg": r}) == [1]


def test_auto_bins_small_and_large():
    bins, _ = _bins(_cg([_cp("cp_x", [], width=2)]), [1], [3], [3])
    assert bins == {"auto[0]": 0, "auto[1]": 1, "auto[2]": 0, "auto[3]": 2}
    bins, _ = _bins(_cg([_cp("cp_x", [], width=8)]), [0], [3], [4], [255])
    assert len(bins) == AUTO_BIN_MAX
    assert bins["auto[0:3]"] == 2 and bins["auto[4:7]"] == 1
    assert bins["auto[252:255]"] == 1


def test_enum_auto_bins_one_per_enumerator():
    """An enum coverpoint (``items``) has one automatic bin per enumerator,
    in declaration order; a value no enumerator has counts nowhere."""
    cp = _cp("cp_x", [], width=32, signed=True)
    cp["items"] = [["IDLE", 0], ["HW", 4], ["LL", 2]]
    bins, _ = _bins(_cg([cp]), [4], [2], [4], [7])
    assert list(bins) == ["auto[IDLE]", "auto[HW]", "auto[LL]"]
    assert bins == {"auto[IDLE]": 0, "auto[HW]": 2, "auto[LL]": 1}


def test_signed_values_are_sign_extended():
    doc = _cg([_cp("cp_x", [_b("neg", [[None, -1]]), _b("pos", [[0, None]])],
                   signed=True)])
    bins, _ = _bins(doc, [0xF], [0x7], [-2])
    assert bins == {"neg": 2, "pos": 1}


def test_iff_off_skips_the_point_and_its_crosses():
    doc = _cg([_cp("cp_x", [_b("z", [0])], iff=True), _cp("cp_y", [_b("z", [0])])],
              [{"name": "xy", "coverpoints": ["cp_x", "cp_y"], "iff": False,
                "bins": []}])
    r = bin_samples([doc], [(0, [0, 0, 0]), (0, [0, 1, 0])])["pss_top::A::cs"]
    assert r["samples"] == 2
    assert r["coverpoints"]["cp_x"]["bins"] == {"z": 1}
    assert r["coverpoints"]["cp_y"]["bins"] == {"z": 2}
    assert r["crosses"]["xy"]["bins"] == {"<z,z>": 1}


def test_corpus_cross_auto_bins():
    """pss-corpus cov/cross: one bin per combination of the points' bins."""
    doc = _cg([_cp("cp_a", [_b("lo", [[0, 1]]), _b("hi", [[2, 3]])], width=2),
               _cp("cp_b", [_b("z", [0]), _b("o", [1])], width=1)],
              [{"name": "ab", "coverpoints": ["cp_a", "cp_b"], "iff": False,
                "bins": []}])
    r = bin_samples([doc], [(0, [0, 0]), (0, [1, 0]), (0, [3, 1])])
    assert r["pss_top::A::cs"]["crosses"]["ab"] == {
        "coverpoints": ["cp_a", "cp_b"], "kinds": {},
        "bins": {"<lo,z>": 2, "<lo,o>": 0, "<hi,z>": 0, "<hi,o>": 1}}


def test_cross_select_bins_take_samples_out_of_auto_bins():
    sel = {"op": "Eq", "args": [{"ref": "cp_b"}, {"c": 1}]}
    doc = _cg([_cp("cp_a", [_b("lo", [[0, 1]]), _b("hi", [[2, 3]])], width=2),
               _cp("cp_b", [_b("z", [0]), _b("o", [1])], width=1)],
              [{"name": "ab", "coverpoints": ["cp_a", "cp_b"], "iff": False,
                "bins": [{"name": "ones", "kind": "ignore", "select": sel}]}])
    r = bin_samples([doc], [(0, [0, 0]), (0, [3, 1])])["pss_top::A::cs"]
    x = r["crosses"]["ab"]
    assert x["bins"]["ones"] == 1 and x["bins"]["<hi,o>"] == 0
    assert x["bins"]["<lo,z>"] == 1 and x["kinds"] == {"ones": "ignore"}


def test_merge_adds_hits_and_samples():
    doc = _cg([_cp("cp_x", [_b("two", [2])])])
    a = bin_samples([doc], [(0, [2])])
    merge(a, bin_samples([doc], [(0, [2]), (0, [3])]))
    cg = a["pss_top::A::cs"]
    assert cg["samples"] == 3 and cg["coverpoints"]["cp_x"]["bins"] == {"two": 2}


def test_unknown_name_and_huge_array_are_errors():
    bad = {"op": "Eq", "args": [{"ref": "nope"}, {"c": 1}]}
    with pytest.raises(CovError, match="nope"):
        _bins(_cg([_cp("cp_x", [_b("b", [1], with_=bad)])]), [1])
    with pytest.raises(CovError, match="array bin"):
        Binner(_cg([_cp("cp_x", [_b("b", [[0, None]], array=True)], width=32)]))


def test_coverage_report_lists_and_merges_covergroups():
    from zuspec.be.bc.cov import Coverage
    doc = _cg([_cp("cp_x", [_b("two", [2]), _b("ig", [3], kind="ignore")])])
    cov = Coverage()
    cov.add_covergroups(bin_samples([doc], [(0, [2])]))
    rep = cov.report()
    assert rep["covergroups"]["pss_top::A::cs"]["samples"] == 1
    assert rep["summary"]["covergroups"] == {"total": 1, "hit": 1}
    other = Coverage()
    other.merge(rep)
    other.merge(rep)
    assert other.report()["covergroups"]["pss_top::A::cs"]["coverpoints"][
        "cp_x"]["bins"] == {"two": 2, "ig": 0}
