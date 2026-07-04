"""
zuspec.be.bc.lower -- Scenario IR -> ZBC lowering (P1-W6).

The public entry points are :func:`lower_coroutine` (one ``ScCoroutine`` ->
:class:`~zuspec.be.bc.model.CoroDescriptor`) and :func:`lower_scenario`
(a set of coroutines -> a :class:`~zuspec.be.bc.model.ZbcModel`).

Lowering consumes the **same** :class:`~zuspec.ir.core.xf.coro_fsm.CoroutineFSMPass`
output the C backend will (design D§2): the pass splits a coroutine into FSM
blocks at suspend points; each block's straight-line statements lower to
procedural ops and its trailing suspend lowers to an orchestration op.

M1 scope (plan §1.1): only FSM-representable coroutines. A suspend inside a
loop/conditional makes the FSM pass raise ``UnsupportedConstructError``; lowering
re-raises it as :class:`LoweringError` with a clear message rather than
mis-compiling (test T1-F).
"""

from .errors import LoweringError
from .context import Lowerer
from .driver import lower_coroutine, lower_scenario, lower_module

__all__ = ["LoweringError", "Lowerer", "lower_coroutine", "lower_scenario",
           "lower_module"]
