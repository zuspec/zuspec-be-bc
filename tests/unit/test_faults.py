"""The test-only fault switches (pssc M9a9 kill matrix)."""
import pytest

from zuspec.be.bc.interp import faults


def test_inject_turns_a_fault_on_for_its_block_only():
    name = faults.ALL[0]
    assert not faults.on(name)
    with faults.inject(name):
        assert faults.on(name)
        assert not any(faults.on(n) for n in faults.ALL if n != name)
    assert not faults.on(name)


def test_inject_refuses_an_unknown_fault():
    with pytest.raises(AssertionError):
        with faults.inject("no-such-fault"):
            pass


def test_every_fault_is_documented():
    assert faults.ALL and all(faults.DOC[n] for n in faults.ALL)


def test_pending_faults_are_injectable_but_not_in_the_matrix():
    assert not set(faults.PENDING) & set(faults.ALL)
    for name in faults.PENDING:
        assert faults.PENDING[name]
        with faults.inject(name):
            assert faults.on(name)
        assert not faults.on(name)
