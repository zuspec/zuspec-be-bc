"""A scripted host for host ABI conformance tests (docs/spec/host-abi.md,
"Conformance tests").

The host drives one call through the version-2 calls only: ``run``, ``take``,
``complete(req_id, value)`` and ``set_time``. A test gives the order in which
outstanding requests are completed; the host records what it saw, so a test
checks per-thread and per-executor order under every completion order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from zuspec.be.bc.interp import host as H
from zuspec.be.bc.model import (BUILTIN_READ, BUILTIN_WRITE, INSTR_F_BLOCKING,
                                INSTR_F_HAS_RET, CoroDescriptor, Instr, Op,
                                ZbcModel)

NONE = 0xFFFFFFFF
#: built-in API ids (host-abi.md "The import interface")
API_MEM = 0


def _status(name: str) -> int:
    return getattr(H, name)


# --- models -----------------------------------------------------------------

def mem_write(addr: int, val: int, src_ref: int = 0) -> List[Instr]:
    return [Instr(Op.CONST, args=(0,), imm=addr, src_ref=src_ref),
            Instr(Op.CONST, args=(1,), imm=val, src_ref=src_ref),
            Instr(Op.IMPORT, args=(BUILTIN_WRITE[32], NONE, 0, 1), src_ref=src_ref)]


def mem_read(addr: int, src_ref: int = 0) -> List[Instr]:
    return [Instr(Op.CONST, args=(0,), imm=addr, src_ref=src_ref),
            Instr(Op.IMPORT, args=(BUILTIN_READ[32], 2, 0), flags=INSTR_F_HAS_RET,
                  src_ref=src_ref)]


def user_import(fn_id: int, arg: int, blocking: bool) -> List[Instr]:
    flags = INSTR_F_HAS_RET | (INSTR_F_BLOCKING if blocking else 0)
    return [Instr(Op.CONST, args=(2,), imm=arg),
            Instr(Op.IMPORT, args=(fn_id, 3, 2), flags=flags)]


def wait(t: int) -> List[Instr]:
    return [Instr(Op.WAIT, imm=t)]


def parallel(*branches: List[Instr], first: int = 0,
             imports: Optional[Dict[str, dict]] = None) -> ZbcModel:
    """``parallel { b0; b1; ... }`` (``first`` > 0: join FIRST(first))."""
    root = [Instr(Op.SPAWN, args=(i + 1,)) for i in range(len(branches))]
    root.append(Instr(Op.JOIN, imm=first))
    coros = [CoroDescriptor(name="root", code=root)]
    coros += [CoroDescriptor(name=f"root$par1_{i}", code=list(b))
              for i, b in enumerate(branches)]
    m = ZbcModel(coros=coros, entry_coro=0)
    m.imports = dict(imports or {})
    return m


# --- the host ---------------------------------------------------------------

@dataclass
class Seen:
    """One request as the host saw it, in take order."""
    req_id: int
    thread: int
    executor: int
    api: int
    what: str            # "w@0x100", "r@0x200", "ext(5)"
    blocking: bool


def describe(r) -> str:
    if r.api == API_MEM:
        return f"{r.kind}@{r.addr:#x}"
    return f"{r.name}({','.join(str(a) for a in r.args)})"


@dataclass
class ScriptedHost:
    """Runs ``call`` to DONE. Each time nothing is ready it completes one
    outstanding request, the one ``pick`` chooses from the outstanding list
    (default: the oldest). ``answer(seen)`` gives a request's value."""
    call: object
    pick: Callable[[List[Seen]], Seen] = lambda out: out[0]
    answer: Callable[[Seen], Optional[int]] = lambda s: None
    taken: List[Seen] = field(default_factory=list)
    completed: List[Seen] = field(default_factory=list)
    statuses: List[str] = field(default_factory=list)

    def take_all(self) -> List[Seen]:
        new = []
        while (r := self.call.take()) is not None:
            s = Seen(r.req_id, r.thread, r.executor, r.api, describe(r),
                     r.blocking)
            new.append(s)
        self.taken += new
        return new

    def run(self, limit: int = 1000):
        REQS, DONE, WAIT, ERROR = (_status(n) for n in ("REQS", "DONE",
                                                        "WAIT", "ERROR"))
        outstanding: List[Seen] = []
        for _ in range(limit):
            r = self.call.run()
            self.statuses.append(r.status_name + (f" {r.on}" if r.on else ""))
            if r.status == DONE:
                assert not outstanding, "DONE with requests outstanding"
                return self.call.result
            if r.status == ERROR:
                raise r.exc
            if r.status == REQS:
                outstanding += self.take_all()
                continue
            assert r.status == WAIT, r.status_name
            if r.on == "TIME" and not outstanding:
                continue                     # no clock: time runs on at once
            assert outstanding, f"WAIT {r.on} with nothing outstanding"
            s = self.pick(outstanding)
            outstanding.remove(s)
            self.call.complete(s.req_id, self.answer(s))
            self.completed.append(s)
        raise AssertionError("script did not finish")

    def per_thread(self) -> Dict[int, List[str]]:
        out: Dict[int, List[str]] = {}
        for s in self.taken:
            out.setdefault(s.thread, []).append(s.what)
        return out
