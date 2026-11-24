"""
heap.py -- the Python engine's heap (F6.9c; docs/spec/heap.md).

One :class:`Heap` per VM (an engine instance) holds the run-time strings and
the list/map/set objects that ``HEAP`` ops make and change. A collection
slot holds a handle (0: the empty collection); a string slot holds a model
string-table index, or a heap string handle with bit 63 set. Handles count
up from 1 and are never reused, so a stale one is an error, not an alias.

An op that changes a collection changes it in place and returns its handle
(a new one when it was 0); lowering copies (``COPY``) wherever PSS value
semantics need it. Cells that a ``SET``, ``DELETE`` or ``CLEAR`` drops are
freed (deep); a ``POP_*`` moves its cell out.
"""

from ..model import (HEAP_OPS, HEAP_SIZE, HEAP_GET, HEAP_SET, HEAP_INSERT,
                     HEAP_DELETE, HEAP_PUSH_FRONT, HEAP_PUSH_BACK,
                     HEAP_POP_FRONT, HEAP_POP_BACK, HEAP_CLEAR, HEAP_IN,
                     HEAP_EQ, HEAP_COPY, HEAP_FREE, HEAP_KEYS, HEAP_VALUES,
                     HEAP_TO_LIST, HEAP_TO_SET, HEAP_SHUFFLE, HEAP_JOIN,
                     HEAP_STR_SLICE, HEAP_STR_GET, HEAP_STR_FIND,
                     HEAP_STR_FIND_LAST, HEAP_STR_FIND_ALL, HEAP_STR_LOWER,
                     HEAP_STR_UPPER, HEAP_STR_SPLIT, HEAP_STR_CHARS,
                     HEAP_STR_FROM_CHARS, HEAP_STR_EQ, HEAP_FORMAT, HEAP_NEG,
                     HEAPK_STR, HEAPK_LIST, HEAPK_MAP, HEAPK_SET,
                     HEAP_CELL_INT, HEAP_CELL_STR, HEAP_CELL_COLL)
from .ops_proc import VMError, _get, _set
from .fmt import format_message

MASK64 = (1 << 64) - 1
STR_BIT = 1 << 63
#: the default byte cap (``RunOpts.heap_cap``) and each object's overhead
HEAP_CAP = 64 << 20
OBJ_OVERHEAD = 32
_KIND_NAMES = ("string", "list", "map", "set")


class HeapObj:
    """One heap object: its kind, what its cells (and map keys) hold, and
    its data (bytes; a list of cells; dict norm(key) -> (key, value);
    dict norm(cell) -> cell)."""

    __slots__ = ("kind", "cell", "key", "data", "size")

    def __init__(self, kind, cell, key, data):
        self.kind, self.cell, self.key, self.data = kind, cell, key, data
        self.size = 0


class Heap:
    """The heap of one engine instance (heap.md sections 1-5)."""

    def __init__(self, strings=(), cap: int = HEAP_CAP):
        #: the image's model-string table (string slots with bit 63 clear)
        self.strings = strings
        self.cap = cap
        self.objs = {}
        self.next = 1
        #: live objects and their bytes (payload + overhead), for leak checks
        self.live = 0
        self.bytes = 0

    # -- objects ------------------------------------------------------------ #

    def _account(self, o: HeapObj) -> None:
        if o.kind == HEAPK_STR:
            n = len(o.data)
        else:
            n = len(o.data) * (16 if o.kind == HEAPK_MAP else 8)
        n += OBJ_OVERHEAD
        self.bytes += n - o.size
        o.size = n
        if self.bytes > self.cap:
            raise VMError(f"heap exhausted ({self.bytes} bytes > cap {self.cap})")

    def alloc(self, kind, data, cell=HEAP_CELL_INT, key=HEAP_CELL_INT) -> int:
        """A new object; its handle (a string's with bit 63 set)."""
        h = self.next
        self.next += 1
        o = HeapObj(kind, cell, key, data)
        self.objs[h] = o
        self.live += 1
        self._account(o)
        return h | STR_BIT if kind == HEAPK_STR else h

    def obj(self, h: int) -> HeapObj:
        o = self.objs.get(h & ~STR_BIT & MASK64)
        if o is None:
            raise VMError(f"bad heap handle {h:#x}")
        return o

    def _coll(self, h, kind, cell, key) -> int:
        """*h*, or a new empty *kind* collection when *h* is 0."""
        if h:
            return h
        return self.alloc(kind, {} if kind in (HEAPK_MAP, HEAPK_SET) else [],
                          cell, key)

    def items(self, h):
        """The cells of collection *h* in order (a map's: its values)."""
        if not h:
            return []
        o = self.obj(h)
        if o.kind == HEAPK_LIST:
            return list(o.data)
        if o.kind == HEAPK_MAP:
            return [v for _, v in o.data.values()]
        return list(o.data.values())

    # -- strings ------------------------------------------------------------ #

    def text(self, s: int) -> bytes:
        """The bytes of string value *s* (table index or heap string)."""
        if s & STR_BIT:
            o = self.obj(s)
            if o.kind != HEAPK_STR:
                raise VMError(f"bad heap handle {s:#x} (not a string)")
            return o.data
        if s >= len(self.strings):
            raise VMError(f"bad string index {s}")
        return self.strings[s].encode()

    def new_str(self, b: bytes) -> int:
        return self.alloc(HEAPK_STR, bytes(b))

    # -- value semantics ---------------------------------------------------- #

    def norm(self, v, ck):
        """A hashable value of cell *v* (holding *ck*) that equal values share."""
        if ck == HEAP_CELL_INT:
            return v
        if ck == HEAP_CELL_STR:
            return ("s", self.text(v))
        o = self.obj(v) if v else None
        if o is None or not o.data:
            return ("c",)                     # handle 0 is any empty collection
        if o.kind == HEAPK_LIST:
            return ("l",) + tuple(self.norm(c, o.cell) for c in o.data)
        if o.kind == HEAPK_SET:
            return ("S", frozenset(o.data))
        return ("m", frozenset((k, self.norm(v2, o.cell))
                               for k, (_, v2) in o.data.items()))

    def copy(self, v, ck):
        """A deep copy of cell *v* (0 and table strings are their own copy)."""
        if ck == HEAP_CELL_INT or not v:
            return v
        if ck == HEAP_CELL_STR:
            return self.new_str(self.text(v)) if v & STR_BIT else v
        o = self.obj(v)
        if o.kind == HEAPK_LIST:
            data = [self.copy(c, o.cell) for c in o.data]
        elif o.kind == HEAPK_SET:
            data = {n: self.copy(c, o.cell) for n, c in o.data.items()}
        elif o.kind == HEAPK_MAP:
            data = {n: (self.copy(k, o.key), self.copy(x, o.cell))
                    for n, (k, x) in o.data.items()}
        else:
            data = o.data
        return self.alloc(o.kind, data, o.cell, o.key)

    def free(self, v, ck) -> None:
        """Reclaim cell *v* and everything it holds (deep)."""
        if ck == HEAP_CELL_INT or not v:
            return
        if ck == HEAP_CELL_STR and not v & STR_BIT:
            return
        o = self.obj(v)
        if o.kind == HEAPK_LIST:
            for c in o.data:
                self.free(c, o.cell)
        elif o.kind == HEAPK_SET:
            for c in o.data.values():
                self.free(c, o.cell)
        elif o.kind == HEAPK_MAP:
            for k, x in o.data.values():
                self.free(k, o.key)
                self.free(x, o.cell)
        del self.objs[v & ~STR_BIT & MASK64]
        self.live -= 1
        self.bytes -= o.size

    def _index(self, o, i, hi):
        i = i - (1 << 64) if i >> 63 else i
        if not 0 <= i < hi:
            raise VMError(f"index out of range: {i} (size {len(o.data)})")
        return i

    # -- ops ---------------------------------------------------------------- #

    def op(self, frame, sub, kind, ck, kk, a, b, c):
        """Run sub-op *sub*; its result (a tuple for ``POP_*``, None for none)."""
        if sub == HEAP_COPY:
            return self.copy(a, HEAP_CELL_STR if kind == HEAPK_STR
                             else HEAP_CELL_COLL)
        if sub == HEAP_FREE:
            self.free(a, HEAP_CELL_STR if kind == HEAPK_STR else HEAP_CELL_COLL)
            return None
        if sub == HEAP_SIZE:
            if kind == HEAPK_STR:
                return len(self.text(a))
            return len(self.obj(a).data) if a else 0
        if sub == HEAP_EQ:
            return int(self.norm(a, HEAP_CELL_COLL) == self.norm(b, HEAP_CELL_COLL))
        if sub == HEAP_STR_EQ:
            return int(a == b or self.text(a) == self.text(b))
        if sub >= HEAP_JOIN:
            return self._str_op(frame, sub, a, b, c)

        if sub in (HEAP_GET, HEAP_IN, HEAP_KEYS, HEAP_VALUES, HEAP_TO_LIST,
                   HEAP_TO_SET) and not a:
            if sub == HEAP_GET:
                raise VMError("missing map key" if kind == HEAPK_MAP
                              else "index out of range: (empty list)")
            if sub == HEAP_IN:
                return 0
            kinds = {HEAP_KEYS: HEAPK_SET, HEAP_VALUES: HEAPK_LIST,
                     HEAP_TO_LIST: HEAPK_LIST, HEAP_TO_SET: HEAPK_SET}
            return self._coll(0, kinds[sub], ck, kk)
        if sub in (HEAP_POP_FRONT, HEAP_POP_BACK) and not a:
            raise VMError("pop of an empty list")
        if sub == HEAP_CLEAR:
            if a:
                o = self.obj(a)
                for x in self.items(a):
                    self.free(x, o.cell)
                if o.kind == HEAPK_MAP:
                    for k, _ in o.data.values():
                        self.free(k, o.key)
                o.data.clear()
                self._account(o)
            return a
        if sub in (HEAP_DELETE, HEAP_SHUFFLE) and not a:
            if sub == HEAP_DELETE and kind == HEAPK_LIST:
                raise VMError(f"index out of range: {b} (size 0)")
            return 0

        h = self._coll(a, kind, ck, kk)
        o = self.obj(h)
        d = o.data
        if o.kind == HEAPK_LIST:
            r = self._list_op(frame, o, sub, b, c)
        elif o.kind == HEAPK_MAP:
            r = self._map_op(o, sub, b, c)
        elif o.kind == HEAPK_SET:
            r = self._set_op(o, sub, b)
        else:
            raise VMError(f"HEAP {HEAP_OPS[sub]} on a {_KIND_NAMES[o.kind]}")
        if sub in (HEAP_GET, HEAP_IN):
            return r
        if sub in (HEAP_KEYS, HEAP_VALUES, HEAP_TO_LIST, HEAP_TO_SET):
            return r
        self._account(o)
        if sub in (HEAP_POP_FRONT, HEAP_POP_BACK):
            return (r, h)
        del d
        return h

    def _list_op(self, frame, o, sub, b, c):
        d = o.data
        if sub == HEAP_GET:
            return d[self._index(o, b, len(d))]
        if sub == HEAP_SET:
            i = self._index(o, b, len(d))
            if d[i] != c:               # a nested list stored back (F6.10c3)
                self.free(d[i], o.cell)
            d[i] = c
        elif sub == HEAP_INSERT:
            d.insert(self._index(o, b, len(d) + 1), c)
        elif sub == HEAP_DELETE:
            self.free(d.pop(self._index(o, b, len(d))), o.cell)
        elif sub == HEAP_PUSH_FRONT:
            d.insert(0, b)
        elif sub == HEAP_PUSH_BACK:
            d.append(b)
        elif sub in (HEAP_POP_FRONT, HEAP_POP_BACK):
            if not d:
                raise VMError("pop of an empty list")
            return d.pop(0 if sub == HEAP_POP_FRONT else -1)
        elif sub == HEAP_IN:
            n = self.norm(b, o.cell)
            return int(any(self.norm(x, o.cell) == n for x in d))
        elif sub == HEAP_TO_SET:
            out = {}
            for x in d:
                n = self.norm(x, o.cell)
                if n not in out:
                    out[n] = self.copy(x, o.cell)
            return self.alloc(HEAPK_SET, out, o.cell)
        elif sub == HEAP_SHUFFLE:
            for i in range(len(d) - 1, 0, -1):
                j = frame.seed.next_below(i + 1)
                d[i], d[j] = d[j], d[i]
        else:
            raise VMError(f"HEAP {HEAP_OPS[sub]} on a list")
        return None

    def _map_op(self, o, sub, b, c):
        d = o.data
        if sub == HEAP_GET:
            e = d.get(self.norm(b, o.key))
            if e is None:
                raise VMError("missing map key")
            return e[1]
        if sub in (HEAP_SET, HEAP_INSERT):
            n = self.norm(b, o.key)
            old = d.get(n)
            if old is not None:
                self.free(old[0], o.key)
                if old[1] != c:
                    self.free(old[1], o.cell)
            d[n] = (b, c)
        elif sub == HEAP_DELETE:
            old = d.pop(self.norm(b, o.key), None)
            if old is not None:
                self.free(old[0], o.key)
                self.free(old[1], o.cell)
        elif sub == HEAP_IN:
            return int(self.norm(b, o.key) in d)
        elif sub == HEAP_KEYS:
            return self.alloc(HEAPK_SET, {n: self.copy(k, o.key)
                                          for n, (k, _) in d.items()}, o.key)
        elif sub == HEAP_VALUES:
            return self.alloc(HEAPK_LIST, [self.copy(v, o.cell)
                                           for _, v in d.values()], o.cell)
        else:
            raise VMError(f"HEAP {HEAP_OPS[sub]} on a map")
        return None

    def _set_op(self, o, sub, b):
        d = o.data
        if sub == HEAP_INSERT:
            n = self.norm(b, o.cell)
            if n in d:
                self.free(b, o.cell)
            else:
                d[n] = b
        elif sub == HEAP_DELETE:
            old = d.pop(self.norm(b, o.cell), None)
            if old is not None:
                self.free(old, o.cell)
        elif sub == HEAP_IN:
            return int(self.norm(b, o.cell) in d)
        elif sub == HEAP_TO_LIST:
            return self.alloc(HEAPK_LIST, [self.copy(x, o.cell)
                                           for x in d.values()], o.cell)
        else:
            raise VMError(f"HEAP {HEAP_OPS[sub]} on a set")
        return None

    def _str_op(self, frame, sub, a, b, c):
        if sub == HEAP_JOIN:
            sep = self.text(b)
            return self.new_str(sep.join(self.text(x) for x in self.items(a)))
        if sub == HEAP_STR_FROM_CHARS:
            return self.new_str(bytes(x & 0xFF for x in self.items(a)))
        s = self.text(a)
        n = len(s)
        sx = lambda v: v - (1 << 64) if v >> 63 else v
        if sub == HEAP_STR_SLICE:
            lo, hi = sx(b), sx(c)
            hi = n - 1 if hi == -1 else hi
            if not (0 <= lo <= n and -1 <= hi < n and lo <= hi + 1):
                raise VMError(f"index out of range: [{lo}..{hi}] (size {n})")
            return self.new_str(s[lo:hi + 1])
        if sub == HEAP_STR_GET:
            i = sx(b)
            if not 0 <= i < n:
                raise VMError(f"index out of range: {i} (size {n})")
            return s[i]
        if sub == HEAP_STR_FIND:
            return s.find(self.text(b), max(sx(c), 0)) & MASK64
        if sub == HEAP_STR_FIND_LAST:
            last = sx(c)
            end = n if last < 0 else last + len(self.text(b))
            return s.rfind(self.text(b), 0, end) & MASK64
        if sub == HEAP_STR_FIND_ALL:
            t, out, i = self.text(b), [], s.find(self.text(b))
            while t and i >= 0:
                out.append(i)
                i = s.find(t, i + 1)
            return self.alloc(HEAPK_LIST, out)
        if sub == HEAP_STR_LOWER:
            return self.new_str(s.lower())
        if sub == HEAP_STR_UPPER:
            return self.new_str(s.upper())
        if sub == HEAP_STR_SPLIT:
            parts = s.split(self.text(b)) if self.text(b) else [s]
            return self.alloc(HEAPK_LIST, [self.new_str(p) for p in parts],
                              HEAP_CELL_STR)
        if sub == HEAP_STR_CHARS:
            return self.alloc(HEAPK_LIST, list(s))
        raise VMError(f"HEAP {HEAP_OPS[sub] if sub < len(HEAP_OPS) else sub}"
                      " is not supported yet")


class HeapStrings:
    """The text of string values, table indices or heap strings, by
    ``[value]``: what message formatting reads a ``%s`` argument through."""

    def __init__(self, heap: Heap):
        self.heap = heap

    def __getitem__(self, v: int) -> str:
        return self.heap.text(v).decode(errors="replace")


def exec_heap(vm, frame, ins) -> None:
    """``HEAP``: arg0 = rd, arg1..3 = a, b, c; imm = sub-op | kinds."""
    imm = ins.imm
    sub, kind = imm & 0xFF, (imm >> 8) & 3
    args = ins.args
    a = _get(frame, args[1]) if len(args) > 1 else 0
    b = _get(frame, args[2]) if len(args) > 2 else 0
    c = _get(frame, args[3]) if len(args) > 3 else 0
    if sub == HEAP_FORMAT:
        # a = the message entry (as message()'s); its slots hold the values
        entry = vm.model.messages[a]
        vals = [frame.locals[s] for s in entry["slots"]]
        try:
            text = format_message(entry["fmt"], entry["args"], vals[1:],
                                  HeapStrings(vm.heap))
        except ValueError as e:
            raise VMError(f"format_string(): {e}")
        _set(frame, args[0], vm.heap.new_str(text.encode()))
        return
    r = vm.heap.op(frame, sub, kind, (imm >> 10) & 3, (imm >> 12) & 3, a, b, c)
    if sub == HEAP_STR_EQ and imm & HEAP_NEG:
        r ^= 1
    if isinstance(r, tuple):
        _set(frame, args[0], r[0])
        _set(frame, args[0] + 1, r[1])
    elif r is not None:
        _set(frame, args[0], r)
