"""
components.py -- the component tree at run time (P1.5).

The elaborated component tree (ir-core ``ScComponentTree``) is ONE object, the
*component object*: each instance a slot range (P1-D1). A frame runs in one
instance (``Frame.comp``) and its ``LD_COMP``/``ST_COMP`` slot ``k`` is the
component object's slot ``base(comp) + k``. The coroutine that constructs the
tree -- initial values, ``init_down`` top-down, ``init_up`` bottom-up -- runs
to completion before the entry action starts (LRM 20.1.3: before the root
action's ``pre_solve``).
"""

from typing import List, Optional

from .extern import Obj


class CompTable:
    """The component object's layout, each instance's base, and the
    constructing coroutine."""

    def __init__(self, tree, init_coro: Optional[int]):
        self.names: List[str] = [f.name for f in sorted(tree.fields, key=lambda f: f.slot)]
        self.bases: List[int] = [i.base for i in tree.instances]
        self.paths: List[str] = [i.path for i in tree.instances]
        self.types: List[str] = [i.type_qname for i in tree.instances]
        #: the root component type; each slot's type descriptor (the
        #: lowering fills these; LAYOUT section)
        self.root: str = getattr(tree, "root", "")
        self.slot_types: List[dict] = []
        #: coroutine index constructing the tree, or None if nothing does
        self.init_coro = init_coro
        #: the tree has run_start/run_end blocks: coroutines init_coro + 1
        #: ($run_start) and + 2 ($run_end) run around a root action
        self.run_hooks: bool = False

    @property
    def run_start_coro(self) -> Optional[int]:
        return self.init_coro + 1 if self.run_hooks else None

    @property
    def run_end_coro(self) -> Optional[int]:
        return self.init_coro + 2 if self.run_hooks else None

    def new_obj(self) -> Obj:
        return Obj(field_names=self.names)

    def base(self, comp: Optional[int]) -> int:
        return self.bases[comp] if comp is not None else 0
