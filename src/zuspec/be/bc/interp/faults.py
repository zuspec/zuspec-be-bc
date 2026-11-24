"""
faults -- deliberate tool faults for the kill matrix (pssc M9a9, plan F9.3).

Test-only. Each fault breaks one rule of the pyvm's scheduling at one guarded
site (``if faults.on("x"):``); pssc's ``tests/compliance/test_fault_kill.py``
proves each is caught by some corpus test (its checker verdict is not PASS).
Normal runs pay one set lookup per site.

Turn one on with :func:`inject` (a context manager), or for a whole process
with the environment variable ``ZSP_BC_FAULT`` (a comma list of names), read
at import. The C engine is not faulted.
"""
import contextlib
import os

#: name -> what it breaks
DOC = {
    "ignore-lock": "a claim excludes no instance: a lock does not keep "
                   "others out (activation `_busy`)",
    "skip-buffer-wait": "a buffer output is offered to consumers when its "
                        "producer is solved, not when it completes "
                        "(activation `commit` / `exit_node`)",
    "drop-stream-partner": "an inferred stream partner is never started "
                           "(ops_orch `_start_partner`)",
    "stale-prev": "a state write lands one write late: a writer's `prev` "
                  "is the state before the previous (activation "
                  "`publish_states`)",
    "select-no-prune": "a select branch that cannot run is still chosen "
                       "(activation `select_blocked`)",
    "atomic-infer": "an action is inferred inside an atomic block "
                    "(activation `_in_atomic`, `atomic_infer`)",
    "race-state": "a state output becomes the current object when its "
                  "writer is solved, not when it completes (activation "
                  "`publish_states`), and readers do not wait for it",
    "skip-state-wait": "a state reader or writer starts while a writer of "
                       "its pool is still running, or a writer while a "
                       "reader runs (activation `state_wait`)",
    "no-lookahead": "a node is solved greedily: constraints on nodes not "
                    "yet committed are ignored, as with the tree's "
                    "`lookahead` off (activation `_in_force`)",
}

ALL = tuple(DOC)

#: faults no corpus test kills yet; not in :data:`ALL`, so the kill matrix
#: does not list them, but they can be injected. (Not a fault: inferring a
#: buffer producer although an explicit one's object fits -- LRM 3.0 Figure 5
#: allows it; only an inferred action feeding nothing is illegal.)
PENDING = {}

KNOWN = ALL + tuple(PENDING)

#: the faults on now
ACTIVE = set()


def on(name: str) -> bool:
    """Fault *name* is on."""
    return name in ACTIVE


@contextlib.contextmanager
def inject(name: str):
    """Turn fault *name* on for the ``with`` block."""
    assert name in KNOWN, "unknown fault %r (known: %s)" % (name, ", ".join(KNOWN))
    was = name in ACTIVE
    ACTIVE.add(name)
    try:
        yield
    finally:
        if not was:
            ACTIVE.discard(name)


for _n in filter(None, (s.strip() for s in os.environ.get("ZSP_BC_FAULT", "").split(","))):
    assert _n in KNOWN, "ZSP_BC_FAULT: unknown fault %r (known: %s)" % (_n, ", ".join(KNOWN))
    ACTIVE.add(_n)
