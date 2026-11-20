"""The debug stub over a socket (pssc P4.5c2): ``docs/spec/debug-stub.md``
"Messages", one JSON object per line.

:func:`serve_stub` puts any stub (pyvm :class:`~.debug.Stub`, the C
engine's ``zuspec.rt.eng.debug.NativeStub``) on a connected socket: each
request line is answered by ``stub.request``, and the stub's events follow
as they come. :class:`SocketStub` is the other end, with the stub interface
a debugger drives (``request(msg) -> reply``, ``next_event(timeout)``,
``join(timeout)``, ``state``).

Requests are served one at a time, in order. An event is sent only between
replies, so one a request causes (the stop that ends a ``step``) follows
its reply. Each line the server writes also carries ``stub_state``, the
stub's ``state`` when it was written; the client takes it off and keeps it
as its ``state``. When the client goes away, a stub still attached is
detached (its run goes on unobserved).
"""

import json
import queue
import socket
import threading
from typing import Any, Callable, Dict, Optional

#: the wire key the server adds to each line (the client removes it)
STATE_KEY = "stub_state"


def _dumps(msg: dict) -> bytes:
    return (json.dumps(msg, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _lines(sock: socket.socket):
    """The JSON objects read from ``sock`` until it closes; a line that is
    not JSON yields None."""
    r = sock.makefile("rb")
    try:
        for line in r:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                yield None
    except (OSError, ValueError):
        return
    finally:
        r.close()


def _shut(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


class StubServer:
    """Serves ``stub`` on the connected socket ``sock`` (see the module);
    :func:`serve_stub` makes and starts one."""

    def __init__(self, stub, sock: socket.socket):
        self.stub, self.sock = stub, sock
        self.closed = False
        self._wlock = threading.Lock()
        #: held while a request runs and its reply is written
        self._busy = threading.Lock()
        self._threads = []

    def start(self) -> "StubServer":
        for fn, name in ((self._serve, "stub-sock-rq"), (self._pump, "stub-sock-ev")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def _send(self, msg: dict) -> None:
        with self._wlock:
            if self.closed:
                return
            msg = dict(msg)
            msg[STATE_KEY] = getattr(self.stub, "state", None)
            try:
                self.sock.sendall(_dumps(msg))
            except OSError:
                self.closed = True

    def _serve(self) -> None:
        for msg in _lines(self.sock):
            if self.closed:
                break
            with self._busy:
                if msg is None:
                    rep = {"seq": None, "ok": False,
                           "error": {"code": "bad_request", "message": "a line is not JSON"}}
                else:
                    rep = self.stub.request(msg)
                self._send(rep)
        if getattr(self.stub, "state", None) not in ("detached", "exited", None):
            self.stub.request({"cmd": "detach"})
        self.close()

    def _pump(self) -> None:
        while not self.closed:
            ev = self.stub.next_event(0.2)
            if ev is not None:
                with self._busy:
                    self._send(ev)

    def close(self) -> None:
        with self._wlock:
            if self.closed and self.sock.fileno() < 0:
                return
            self.closed = True
        _shut(self.sock)

    def join(self, timeout: Optional[float] = None) -> None:
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout)


def serve_stub(stub, sock: socket.socket) -> StubServer:
    """Serve ``stub`` on ``sock`` from two daemon threads; returns the
    started :class:`StubServer`."""
    return StubServer(stub, sock).start()


class SocketStub:
    """A stub on the far end of ``sock`` (served by :func:`serve_stub`).

    ``request`` waits up to ``timeout`` seconds for the reply; a closed
    socket or a timeout is an error reply (codes ``closed``, ``timeout``).
    When the socket closes before the run has exited, ``next_event`` gives
    one ``exited`` event with ``detached`` and ``closed`` true. ``vm`` is
    None (the engine is in another thread or process)."""

    vm = None

    def __init__(self, sock: socket.socket, timeout: Optional[float] = 30.0,
                 on_event: Optional[Callable[[dict], None]] = None):
        self.sock, self.timeout, self.on_event = sock, timeout, on_event
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.state = "detached"
        self.closed = False
        self._seq = 0
        self._replies: Dict[int, dict] = {}
        self._cv = threading.Condition()
        self._wlock = threading.Lock()
        self._exited = False
        self._reader = threading.Thread(target=self._read, name="stub-sock-client", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        for msg in _lines(self.sock):
            if not isinstance(msg, dict):
                continue
            with self._cv:
                st = msg.pop(STATE_KEY, None)
                if st is not None:
                    self.state = st
                if "event" in msg:
                    if msg["event"] == "exited":
                        self._exited = True
                    self.events.put(msg)
                    if self.on_event is not None:
                        self.on_event(msg)
                elif isinstance(msg.get("seq"), int):
                    self._replies[msg["seq"]] = msg
                self._cv.notify_all()
        with self._cv:
            self.closed = True
            if not self._exited:
                self._exited = True
                if self.state not in ("detached", "exited"):
                    self.state = "detached"
                self.events.put({"event": "exited", "body": {"detached": True, "closed": True}})
            self._cv.notify_all()

    def request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Send one request; returns its reply (with the caller's ``seq``)."""
        seq = msg.get("seq") if isinstance(msg, dict) else None
        with self._cv:
            self._seq += 1
            wire = self._seq
            closed = self.closed
        if closed:
            return {"seq": seq, "ok": False,
                    "error": {"code": "closed", "message": "the stub's socket is closed"}}
        try:
            with self._wlock:
                self.sock.sendall(_dumps(dict(msg, seq=wire)))
        except OSError as e:
            return {"seq": seq, "ok": False, "error": {"code": "closed", "message": str(e)}}
        with self._cv:
            ok = self._cv.wait_for(lambda: wire in self._replies or self.closed, self.timeout)
            rep = self._replies.pop(wire, None)
        if rep is None:
            code = "closed" if ok else "timeout"
            return {"seq": seq, "ok": False,
                    "error": {"code": code, "message": f"no reply to {msg.get('cmd')!r} ({code})"}}
        rep["seq"] = seq
        return rep

    def next_event(self, timeout: Optional[float] = 5.0) -> Optional[dict]:
        """The next event, or None if none comes within ``timeout``."""
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def join(self, timeout: Optional[float] = 5.0) -> None:
        """Wait until the run has exited, the stub is detached or the socket
        closes."""
        with self._cv:
            self._cv.wait_for(lambda: self.closed or self.state in ("exited", "detached"), timeout)

    def close(self) -> None:
        _shut(self.sock)
        self._reader.join(1.0)
