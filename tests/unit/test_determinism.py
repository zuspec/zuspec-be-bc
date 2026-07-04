"""Determinism reference-impl tests (backs P0-12 / docs/spec/determinism.md)."""

import pytest

from zuspec.be.bc.determinism import (
    lcg_next, fork_seed, SeedStream, select_choice,
    LCG_MUL, LCG_ADD, MASK64,
)


def test_lcg_matches_legacy_constants():
    # The inherited anchor: exact constants, exact modulus.
    assert LCG_MUL == 6364136223846793005
    assert LCG_ADD == 1442695040888963407
    assert lcg_next(0) == LCG_ADD & MASK64
    assert lcg_next(1) == (LCG_MUL + LCG_ADD) & MASK64


def test_stream_is_reproducible():
    a = SeedStream(12345)
    b = SeedStream(12345)
    assert [a.next_raw() for _ in range(8)] == [b.next_raw() for _ in range(8)]


def test_streams_with_different_seeds_diverge():
    a = SeedStream(1)
    b = SeedStream(2)
    assert [a.next_raw() for _ in range(4)] != [b.next_raw() for _ in range(4)]


def test_fork_is_deterministic_and_decorrelated():
    parent = SeedStream(999)
    parent.next_raw()  # advance to some fork-point state
    s = parent.state
    c0a = SeedStream(fork_seed(s, 0))
    c0b = parent.fork(0)
    # fork() derives from current state; matches fork_seed on that state.
    assert c0b.state == c0a.state
    # Sibling branches produce different streams.
    c1 = SeedStream(fork_seed(s, 1))
    assert c0a.next_raw() != c1.next_raw()


def test_next_below_range_and_reproducible():
    s = SeedStream(7)
    for _ in range(1000):
        v = s.next_below(10)
        assert 0 <= v < 10
    with pytest.raises(ValueError):
        SeedStream(0).next_below(0)


def test_select_choice_declaration_order_and_weights():
    # Deterministic pick for a given stream state.
    s1 = SeedStream(42)
    s2 = SeedStream(42)
    assert select_choice(s1, [1, 1, 1]) == select_choice(s2, [1, 1, 1])
    # A single non-zero weight is always chosen.
    assert select_choice(SeedStream(3), [0, 5, 0]) == 1
    # Zero total weight is an error.
    with pytest.raises(ValueError):
        select_choice(SeedStream(1), [0, 0])


def test_select_choice_covers_all_branches_over_many_seeds():
    seen = set()
    for seed in range(200):
        seen.add(select_choice(SeedStream(seed), [1, 1, 1]))
    assert seen == {0, 1, 2}
