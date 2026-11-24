"""F4.8c: an illegal covergroup bin hit is listed at its declaration."""

from zuspec.be.bc.cov.covergroup import bin_samples, illegal_hits

DOC = {"name": "A::cs", "coverpoints": [
    {"name": "cp", "width": 2, "signed": False, "iff": False, "bins": [
        {"name": "ok", "kind": "bins", "values": [[0, 2]], "array": False},
        {"name": "bad", "kind": "illegal", "values": [3], "array": False,
         "file": "t.pss", "line": 9},
        {"name": "arr", "kind": "illegal", "values": [[1, 2]], "array": True,
         "file": "t.pss", "line": 10}]}],
    "crosses": []}


def test_no_hit_no_error():
    assert illegal_hits([DOC], bin_samples([DOC], [(0, [0])])) == []


def test_hit_is_at_its_bin():
    rep = bin_samples([DOC], [(0, [3]), (0, [3]), (0, [1])])
    hits = illegal_hits([DOC], rep)
    assert [(h["bin"], h["hits"], h["line"]) for h in hits] == [
        ("bad", 2, 9), ("arr", 1, 10)]
    assert hits[0]["coverpoint"] == "cp" and hits[0]["file"] == "t.pss"
