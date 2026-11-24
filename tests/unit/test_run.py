"""The one run API (zuspec.be.bc.run; docs/spec/run-api.md, full-PSS plan F1.2)."""
import pytest

from zuspec.be.bc import run as bcrun
from zuspec.be.bc.interp import oracle
from zuspec.be.bc.interp.extern import Memory, RecordingImportProvider
from zuspec.be.bc.interp.ops_proc import VMError
from zuspec.be.bc.interp.solve_cache import SolveUnsatError
from zuspec.be.bc.model import (BUILTIN_READ, BUILTIN_WRITE, INSTR_F_HAS_RET,
                                CoroDescriptor, EntryPoint, Instr, Op, ZbcModel)

_NONE = 0xFFFFFFFF


def _model():
    """Three entries: ``main`` reads 0x100, calls import 0 on it and returns
    the result; ``init`` writes 0x41 to 0x100; ``fin`` writes 1 to 0x200;
    ``add`` returns its two arguments' sum."""
    main = [
        Instr(Op.CONST, args=(0,), imm=0x100),
        Instr(Op.IMPORT, args=(BUILTIN_READ[32], 1, 0), flags=INSTR_F_HAS_RET),
        Instr(Op.IMPORT, args=(0, 2, 1), flags=INSTR_F_HAS_RET),
        Instr(Op.RET, args=(2,)),
    ]
    init = [
        Instr(Op.CONST, args=(0,), imm=0x100),
        Instr(Op.CONST, args=(1,), imm=0x41),
        Instr(Op.IMPORT, args=(BUILTIN_WRITE[32], _NONE, 0, 1)),
        Instr(Op.RET),
    ]
    fin = [
        Instr(Op.CONST, args=(0,), imm=0x200),
        Instr(Op.CONST, args=(1,), imm=1),
        Instr(Op.IMPORT, args=(BUILTIN_WRITE[32], _NONE, 0, 1)),
        Instr(Op.RET),
    ]
    add = [
        Instr(Op.LD_ARG, args=(0, 0)),
        Instr(Op.LD_ARG, args=(1, 1)),
        Instr(Op.ADD, args=(2, 0, 1)),
        Instr(Op.RET, args=(2,)),
    ]
    m = ZbcModel(coros=[CoroDescriptor(name="main", code=main),
                        CoroDescriptor(name="init", code=init),
                        CoroDescriptor(name="fin", code=fin),
                        CoroDescriptor(name="add", code=add)], entry_coro=0)
    m.imports = {"ext": {"fn_id": 0, "blocking": False}}
    m.entries = {"main": EntryPoint(coro=0),
                 "C::add": EntryPoint(coro=3, kind="function", comp_type="C")}
    return m


def _platform():
    return bcrun.SimplePlatform(
        Memory(), RecordingImportProvider({0: lambda a: a[0] + 1}))


def test_entry_by_symbol_index_and_default():
    m = _model()
    assert bcrun.resolve_entry(m, None) == 0
    assert bcrun.resolve_entry(m, "C::add") == 3
    assert bcrun.resolve_entry(m, "fin") == 2      # a coroutine by name
    assert bcrun.resolve_entry(m, 1) == 1
    with pytest.raises(ValueError, match="no entry 'nope'"):
        bcrun.resolve_entry(m, "nope")
    with pytest.raises(ValueError):
        bcrun.resolve_entry(m, 9)


def test_function_entry_gets_its_args():
    res = bcrun.run(_model(), "C::add", opts=bcrun.RunOpts(args=(5, 7)))
    assert res.ok and res.verdict == bcrun.OK and res.error is None
    assert res.retval == 12


def test_setup_and_teardown_run_on_the_platform():
    p = _platform()
    res = bcrun.run(_model(), "main", p, opts=bcrun.RunOpts(
        setup=["init"], teardown=[2]))
    assert res.ok and res.retval == 0x42
    assert p.memory.read(0x200, 4) == 1
    assert p.imports.calls


def test_bytes_and_path_images(tmp_path):
    data = _model().to_bytes()
    path = tmp_path / "m.zbc"
    path.write_bytes(data)
    for image in (data, str(path), path):
        assert bcrun.run(image, 3, opts=bcrun.RunOpts(args=(1, 2))).retval == 3


def _raising(exc):
    class Imports:
        calls = []

        def call(self, fn_id, args):
            raise exc
    return bcrun.SimplePlatform(Memory(), Imports())


@pytest.mark.parametrize("exc, verdict", [
    (VMError("boom"), bcrun.RUNTIME_ERROR),
    (SolveUnsatError("unsat"), bcrun.SOLVE_FAIL),
])
def test_an_image_failure_is_a_verdict_not_a_raise(exc, verdict):
    p = _raising(exc)
    res = bcrun.run(_model(), "main", p, opts=bcrun.RunOpts(teardown=["fin"]))
    assert res.verdict == verdict and not res.ok
    assert res.error is exc
    assert p.memory.read(0x200, 4) == 1          # teardown ran anyway
    with pytest.raises(type(exc)):
        res.check()


def test_a_host_exception_raises():
    with pytest.raises(KeyError):
        bcrun.run(_model(), "main", _raising(KeyError("host")))


def test_run_model_delegates_and_raises():
    p = _platform()
    res = oracle.run_model(_model(), memory=p.memory, import_provider=p.imports)
    assert isinstance(res, bcrun.RunResult) and res.retval == 1
    with pytest.raises(VMError, match="boom"):
        oracle.run_model(_model(), import_provider=_raising(VMError("boom")).imports)


def test_named_run_brackets_host_events():
    from zuspec.be.bc.trace import MemorySink
    from zuspec.be.bc.trace.events import EV_TEST
    events = MemorySink()
    bcrun.run(_model(), "C::add", opts=bcrun.RunOpts(
        args=(1, 1), name="t1", events=events, watch=EV_TEST))
    assert [(e.kind, e.detail) for e in events.events] == [
        ("test.start", {"name": "t1", "seed": 0}),
        ("test.end", {"name": "t1", "done": True})]


def test_run_on_a_host_built_vm_over_the_host_abi():
    from zuspec.be.bc.interp.vm import VM
    p = _platform()
    vm = VM(_model(), memory=p.memory, import_provider=p.imports)
    roots = []
    for host_abi in (False, True):
        res = bcrun.run_on(vm, "main", 1, bcrun.RunOpts(
            setup=["init"], teardown=["fin"], host_abi=host_abi,
            on_root=roots.append))
        assert res.ok and res.retval == 0x42
    assert len(roots) == 2 and roots[0] is not roots[1]
    assert p.memory.read(0x200, 4) == 1


# -- F1.2b: time, notifications, observation, coverage ----------------------- #

from zuspec.be.bc.interp.host import N_IRQ  # noqa: E402
from zuspec.be.bc.trace import events as ev  # noqa: E402


def _timed_model():
    """``main`` waits 10 time units, then reads 0x200 and returns it;
    ``fin`` (coroutine 2) writes 1 to 0x200."""
    m = _model()
    m.coros[0] = CoroDescriptor(name="main", code=[
        Instr(Op.WAIT, imm=10),
        Instr(Op.CONST, args=(0,), imm=0x200),
        Instr(Op.IMPORT, args=(BUILTIN_READ[32], 1, 0), flags=INSTR_F_HAS_RET),
        Instr(Op.RET, args=(1,)),
    ])
    return m


class _Clock:
    """A platform with a clock that raises irq 3 at the first wait."""

    def __init__(self, **kw):
        self.memory = Memory()
        self.imports = None
        self.waits, self.notes, self.seen = [], [], []
        self.__dict__.update(kw)

    def wait_time(self, host, delay):
        self.waits.append(delay)
        if len(self.waits) == 1:
            host.notify(N_IRQ, 3, 7)


def test_platform_clock_and_notify():
    p = _Clock()
    p.on_notify = lambda vm, kind, id_, value: p.notes.append(
        (kind, id_, value))
    res = bcrun.run(_timed_model(), "main", p)
    assert res.ok and res.now == 10
    assert p.waits == [10] and p.notes == [(N_IRQ, 3, 7)]
    assert res.retval == 0


def test_irq_entry_runs_on_notify():
    p = _Clock(irq_entry="fin")
    res = bcrun.run(_timed_model(), "main", p)
    assert res.ok and res.retval == 1      # fin ran before main read 0x200


def test_host_abi_without_a_clock_runs_on():
    # without a wait_time hook, time runs on at once
    res = bcrun.run(_timed_model(), "main",
                    opts=bcrun.RunOpts(host_abi=True))
    assert res.ok and res.now == 10


def test_platform_observes_accesses_imports_and_notifies():
    p = _Clock(irq_entry="fin")
    p.observe = lambda e: p.seen.append(e)
    res = bcrun.run(_timed_model(), "main", p)
    assert res.ok
    kinds = [e.kind for e in p.seen]
    assert ev.NOTIFY in kinds and ev.ACCESS in kinds
    assert [e.kind for e in res.debug_events] == kinds   # kept as well


def test_coverage_counts_points():
    res = bcrun.run(_model(), "C::add", opts=bcrun.RunOpts(
        args=(1, 2), coverage=True))
    assert res.coverage == {"stmts": {}, "arms": {}}   # not debug-lowered
    assert bcrun.run(_model(), "C::add",
                     opts=bcrun.RunOpts(args=(1, 2))).coverage is None
    cov = bcrun._CovSink()
    cov.event(ev.STMT, detail={"pc": 0, "path": "0", "scope": "C::f"})
    cov.event(ev.STMT, detail={"pc": 0, "path": "0", "scope": "C::f"})
    cov.event(ev.STMT, detail={"pc": 1})                 # no coverage table
    cov.event(ev.BRANCH, detail={"pc": 2, "branch": "1", "arm": "then",
                                 "scope": "C::f"})
    assert cov.stmts == {"C::f/0": 2} and cov.arms == {"C::f/1/then": 1}
