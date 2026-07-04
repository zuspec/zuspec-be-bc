"""
determinism.py -- the reference implementation of the ordering contract (P0-12).

The determinism spec (``docs/spec/determinism.md``) is authoritative prose; this
module is its executable form so lowering, the oracle, and the tests all draw
from *one* definition of the seed stream, the fork rule, and weighted SELECT.

Anchors are **inherited, not invented**:

* the seed LCG is the legacy constant
  ``s' = (s*6364136223846793005 + 1442695040888963407) mod 2**64``;
* solver var ordering (``sorted(names)`` -> ``var_id``) lives in
  :func:`zuspec.be.bc.abi.value.solver_var_map` and is referenced, not duplicated.

What this module *adds* (the D§15.1 extension) is the specified order for things
the legacy runtime left implicit: how seeds fork for child coroutines / PAR
branches, and how a weighted SELECT draws.
"""

MASK64 = (1 << 64) - 1

# Legacy LCG constants (do not change without an abi/determinism version bump).
LCG_MUL = 6364136223846793005
LCG_ADD = 1442695040888963407

# Odd 64-bit mixing constant (golden ratio) used to decorrelate forked streams.
FORK_MIX = 0x9E3779B97F4A7C15


def lcg_next(state: int) -> int:
    """Advance the LCG one step."""
    return (state * LCG_MUL + LCG_ADD) & MASK64


def fork_seed(parent_state: int, index: int) -> int:
    """Derive a child stream seed from the parent's *current* state + a branch index.

    A child coroutine (SPAWN) or the i-th PAR branch takes
    ``fork_seed(parent.state, i)``. Because it depends on the parent's state at the
    fork point, and traversal order is itself specified, the whole fork tree is
    reproducible.
    """
    return lcg_next((parent_state ^ ((index * FORK_MIX) & MASK64)) & MASK64)


class SeedStream:
    """A deterministic random stream anchored on the legacy LCG."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next_raw(self) -> int:
        """Advance and return the raw 64-bit state (the next draw)."""
        self.state = lcg_next(self.state)
        return self.state

    def next_below(self, n: int) -> int:
        """Return a draw in ``[0, n)``.

        M1 uses modulo reduction. The modest modulo bias is accepted on purpose:
        the contract this module owns is *reproducibility*, not statistical
        quality (constraint randomness is the solver's job). Documented in the
        spec so a later, less-biased mapping is a conscious versioned change.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        return self.next_raw() % n

    def fork(self, index: int) -> "SeedStream":
        """Return a child stream for branch/child ``index`` (parent unchanged)."""
        return SeedStream(fork_seed(self.state, index))


def select_choice(stream: SeedStream, weights) -> int:
    """Pick a branch index from ``weights`` using one draw against cumulative sums.

    Branches are considered in **declaration order**; the first branch whose
    cumulative weight exceeds the draw is chosen. Zero-weight branches are
    unreachable. This is the specified SELECT order (D§15.1).
    """
    weights = list(weights)
    total = sum(weights)
    if total <= 0:
        raise ValueError("SELECT requires positive total weight")
    draw = stream.next_below(total)
    acc = 0
    for i, w in enumerate(weights):
        acc += w
        if draw < acc:
            return i
    return len(weights) - 1  # unreachable given total>0, but total-safe
