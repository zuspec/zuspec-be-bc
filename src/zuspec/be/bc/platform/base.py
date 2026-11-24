"""The platform to start from: sparse little-endian memory with the
``read<w>``/``write<w>`` methods a run's memory seam calls, and the ways a
register of a debug image may be named (:func:`reg_names`)."""

from __future__ import annotations

import re
from typing import Dict, List


class Platform:
    """A platform to start from: sparse little-endian memory behind the
    ``read``/``write`` methods, ``message`` lines kept in ``messages``.
    Add the model's user imports as methods; override an access to give an
    address behaviour.

    ``log`` has every access: ``("read"|"write", width, addr, data)``, the
    shape of ``pssc_rt.MemoryBus.log``."""

    def __init__(self):
        self.mem: Dict[int, int] = {}
        self.log: List[tuple] = []
        self.messages: List[str] = []

    def _read(self, width: int, addr: int) -> int:
        v = 0
        for i in range(width // 8):
            v |= self.mem.get((addr + i) & ((1 << 64) - 1), 0) << (8 * i)
        self.log.append(("read", width, addr, v))
        return v

    def peek(self, addr: int, n: int) -> int:
        """``n`` bytes at ``addr`` through the back door: no log entry, no
        side effect (the debug stub's ``backdoor_read``)."""
        v = 0
        for i in range(n):
            v |= self.mem.get((addr + i) & ((1 << 64) - 1), 0) << (8 * i)
        return v

    def _write(self, width: int, addr: int, data: int) -> None:
        data &= (1 << width) - 1
        for i in range(width // 8):
            self.mem[(addr + i) & ((1 << 64) - 1)] = (data >> (8 * i)) & 0xFF
        self.log.append(("write", width, addr, data))

    def read8(self, addr: int) -> int: return self._read(8, addr)
    def read16(self, addr: int) -> int: return self._read(16, addr)
    def read32(self, addr: int) -> int: return self._read(32, addr)
    def read64(self, addr: int) -> int: return self._read(64, addr)
    def write8(self, addr: int, data: int) -> None: self._write(8, addr, data)
    def write16(self, addr: int, data: int) -> None: self._write(16, addr, data)
    def write32(self, addr: int, data: int) -> None: self._write(32, addr, data)
    def write64(self, addr: int, data: int) -> None: self._write(64, addr, data)

    def message(self, text: str) -> None:
        self.messages.append(text)


def reg_names(registers: Dict[int, Dict], components) -> Dict[str, set]:
    """Each way to name a register -> the reg-ids it names: its instance path
    (``regs.ctl``), the same without its indices, or ``<group type>.<register>``
    (short or qualified type name)."""
    ctype = (dict(zip(components.paths, components.types))
             if components is not None else {})
    names: Dict[str, set] = {}
    for rid, r in registers.items():
        path = r["path"]
        group, _, leaf = path.rpartition(".")
        leaf = re.sub(r"\[\d+\]$", "", leaf)
        keys = {path, re.sub(r"\[\d+\]", "", path)}
        t = ctype.get(group)
        if t:
            keys |= {f"{t}.{leaf}", f"{t.rpartition('::')[2]}.{leaf}"}
        for k in keys:
            names.setdefault(k, set()).add(rid)
    return names
