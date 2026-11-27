"""The method list of ``zsp_rt_bc``, the SV engine class (pssc pssvm-plan
step 7; design ``pssvm-in-practice.md`` §2.1 rule 1).

``zsp_rt_bc_pkg`` comes in two source variants, chosen by which file a
build compiles: pyvm over pyhdl-if (``share/sv/zsp_rt_bc_pkg.sv``, here) and
the C engine over DPI-C (zuspec-rt-eng ``share/sv/zsp_rt_bc_pkg.sv``). Both
include ``zsp_rt_bc_common.svh`` and define ``class zsp_rt_bc extends
zsp_rt_bc_base``. Every method exists in both *by construction*, because
everything that declares one is generated from :data:`METHODS`:

* ``zsp_rt_bc_if.svh`` -- the interface class ``zsp_rt_bc_if``;
* ``zsp_rt_bc_defaults.svh`` -- the body of each method in
  ``zsp_rt_bc_base`` that a variant does not override: a refusal through
  ``error(0)`` (printed once per method), or for an older query, the
  documented answer of an engine that cannot tell;
* ``zsp_rt_bc_pkg.sv`` -- the pyvm variant whole: the pyhdl-if API class
  (``hdl_if api-gen-sv`` over :func:`transport_source`, which the shim's
  ``make_api`` also executes) and ``zsp_rt_bc``, a forwarder per method.

The C variant is hand-written and overrides what the C engine serves;
``tests/unit/test_rt_bc_api.py`` checks its signatures against this list.
Regenerate with ``python -m zuspec.be.bc.share.rt_bc_api`` (``--check``
reports stale files and changes nothing).
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: where the pyvm variant's pyhdl-if API looks for the shim
SHIM_MODULE = "pssvm_dpi_shim"
#: the pyhdl-if class the shim builds (``make_api``) and SV wraps
API_CLASS = "ZspRtBcPy"


@dataclass(frozen=True)
class M:
    """One method of ``zsp_rt_bc_if``.

    ``params`` are ``(sv type, name)``; a name ending ``[$]`` is a queue.
    ``impl`` "engine": each variant serves it or ``zsp_rt_bc_base``'s
    default stands; "common": ``zsp_rt_bc_base`` implements it in SV for
    both. ``refuse``: the value a refusal returns (the default body);
    ``value``: instead, the documented answer of an engine that cannot
    tell (a body of SV statements when it starts with ``{``). ``py``:
    the pyvm variant's body -- None forwards to the pyhdl-if method of
    the same name; "" leaves the default."""
    name: str
    ret: str
    params: Tuple[Tuple[str, str], ...]
    doc: str
    impl: str = "engine"
    refuse: Optional[str] = None
    value: Optional[str] = None
    py: Optional[str] = None


def _p(*pairs: str) -> Tuple[Tuple[str, str], ...]:
    out = []
    for p in pairs:
        t, _, n = p.rpartition(" ")
        out.append((t, n))
    return tuple(out)


_SET_TIME = "set_time(zsp_rt_bc_now_fs());\nsync();\n"

#: ``req_all``'s tuple, which decodes a request in one crossing: these
#: fields in order (``req_int``/``req_u64``/``req_str`` names, and how the
#: pyvm variant's ``request`` stores each), then the request's arguments
REQ_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("api", "int", "req.api      = {};"),
    ("executor", "int", "req.executor = {};"),
    ("thread", "int", "req.thread   = {};"),
    ("kind", "int", "req.write    = {} != 0;"),
    ("addr", "u64", "req.addr     = {};"),
    ("size", "int", "req.size     = {};"),
    ("data", "u64", "req.data     = {};"),
    ("reg_id", "int", "req.reg_id   = {};"),
    ("src_ref", "str", "req.src_ref  = {};"),
    ("stim_id", "str", "req.stim_id  = {};"),
    ("fn_id", "int", "req.fn_id    = {};"),
    ("name", "str", "req.name     = {};"),
    ("blocking", "int", "req.blocking = {} != 0;"),
    ("on", "int", "req.on       = zsp_rt_bc_wait_on_e'({});"),
    ("delay", "u64", "req.delay    = {};"),
    ("reason", "str", "req.reason   = {};"),
    ("nret", "int", "req.nret     = {};"),
)

_PY_AS = {"int": "int'(pyhdl_if::PyLong_AsLong({}))",
          "u64": "pyhdl_if::PyLong_AsUnsignedLongLongMask({})",
          "str": "pyhdl_if::PyUnicode_AsUTF8({})"}


def _decode_sv() -> List[str]:
    """The pyvm variant's ``decode``: a request tuple's fields (from item
    ``off``: ``req_all``'s tuple at 0, a ``run_all`` item at 1, after its
    req_id) into ``req``; the caller holds the GIL."""
    item = "pyhdl_if::PyTuple_GetItem(t, off + {})"
    out = ["// Request tuple `t` (REQ_FIELDS from item `off`, then the",
           "// arguments) into `req`; the GIL held.",
           "protected function void decode(pyhdl_if::PyObject t, longint unsigned off,",
           "                               int rid, zsp_rt_bc_req req);",
           "  longint unsigned n;",
           "  req.req_id   = rid;"]
    for i, (_, kind, store) in enumerate(REQ_FIELDS):
        out.append("  " + store.format(_PY_AS[kind].format(item.format(i))))
    out += ["  req.args.delete();",
            "  n = pyhdl_if::PyTuple_Size(t);",
            f"  for (longint unsigned i = off + {len(REQ_FIELDS)}; i < n; i++)",
            "    req.args.push_back(" + _PY_AS["u64"].format(
                "pyhdl_if::PyTuple_GetItem(t, i)") + ");",
            "  req.rets.delete();",
            "  req.ret_s.delete();",
            "endfunction"]
    return out


#: the pyvm variant's own state and helpers (in class ``zsp_rt_bc``): what
#: its methods bundle into one crossing (pssc pssvm-plan 8d)
_PY_STATE = """\
// Per call: its last run's tuple (run_all: status, progress, result,
// requests) and the next request `take` hands out of it; the results
// of calls DONE. The time set and not yet sent; the progress count the
// last run brought, good until the next crossing; time_exp, once.
protected pyhdl_if::PyObject runs[int];
protected int                run_i[int];
protected longint unsigned   result_c[int];
protected longint unsigned   t_set;
protected bit                t_dirty;
protected longint            progress_c;
protected bit                progress_ok;
protected int                time_exp_c;
protected bit                time_exp_ok;

// The time for a crossing that carries it: -1 when sent already. Any
// crossing may make progress.
protected function longint t_arg();
  progress_ok = 0;
  if (!t_dirty) return -1;
  t_dirty = 0;
  return longint'(t_set);
endfunction

// Before a crossing that does not carry the time: send it, if set.
protected function void sync();
  longint t = t_arg();
  if (t >= 0) py.set_time($unsigned(t));
endfunction

// Forget call `h`'s last run.
protected function void drop(int h);
  pyhdl_if::PyGILState_STATE gs;
  if (!runs.exists(h)) return;
  gs = pyhdl_if::PyGILState_Ensure();
  pyhdl_if::Py_DecRef(runs[h]);
  pyhdl_if::PyGILState_Release(gs);
  runs.delete(h);
  run_i.delete(h);
endfunction

// Forget every call's last run (something other than `run` may have
// queued requests: `take` asks the engine again).
protected function void drop_all();
  int hs[$];
  foreach (runs[h]) hs.push_back(h);
  foreach (hs[i]) drop(hs[i]);
endfunction

// Request `rid` of call `h`'s last run, or null; the GIL held.
protected function pyhdl_if::PyObject cached(int h, int rid);
  pyhdl_if::PyObject reqs;
  if (!runs.exists(h)) return null;
  reqs = pyhdl_if::PyTuple_GetItem(runs[h], 3);
  for (longint unsigned i = 0; i < pyhdl_if::PyTuple_Size(reqs); i++) begin
    pyhdl_if::PyObject it = pyhdl_if::PyTuple_GetItem(reqs, i);
    if (int'(pyhdl_if::PyLong_AsLong(pyhdl_if::PyTuple_GetItem(it, 0))) == rid)
      return it;
  end
  return null;
endfunction

// `args` as a Python tuple (the crossing takes it over).
protected function pyhdl_if::PyObject args_tuple(longint unsigned args[$]);
  pyhdl_if::PyObject t;
  pyhdl_if::PyGILState_STATE gs = pyhdl_if::PyGILState_Ensure();
  t = pyhdl_if::PyTuple_New(longint'(args.size()));
  foreach (args[i])
    void'(pyhdl_if::PyTuple_SetItem(t, longint'(i),
                                    pyhdl_if::PyLong_FromLongLong(longint'(args[i]))));
  pyhdl_if::PyGILState_Release(gs);
  return t;
endfunction
"""

_RUN_PY = """\
pyhdl_if::PyObject t;
pyhdl_if::PyGILState_STATE gs;
zsp_rt_bc_status_e st;
drop(h);
t = py.run_all(h, t_arg());
gs = pyhdl_if::PyGILState_Ensure();
st = zsp_rt_bc_status_e'(int'(pyhdl_if::PyLong_AsLong(pyhdl_if::PyTuple_GetItem(t, 0))));
progress_c  = pyhdl_if::PyLong_AsLong(pyhdl_if::PyTuple_GetItem(t, 1));
progress_ok = 1;
if (st == ZSP_RT_BC_DONE)
  result_c[h] = pyhdl_if::PyLong_AsUnsignedLongLongMask(pyhdl_if::PyTuple_GetItem(t, 2));
pyhdl_if::PyGILState_Release(gs);
runs[h]  = t;
run_i[h] = 0;
return st;"""

_TAKE_PY = """\
pyhdl_if::PyObject reqs;
pyhdl_if::PyGILState_STATE gs;
int rid = -1;
if (!runs.exists(h)) begin
  sync();
  return py.take(h);
end
gs = pyhdl_if::PyGILState_Ensure();
reqs = pyhdl_if::PyTuple_GetItem(runs[h], 3);
if ($unsigned(longint'(run_i[h])) < pyhdl_if::PyTuple_Size(reqs)) begin
  rid = int'(pyhdl_if::PyLong_AsLong(pyhdl_if::PyTuple_GetItem(
      pyhdl_if::PyTuple_GetItem(reqs, $unsigned(longint'(run_i[h]))), 0)));
  run_i[h]++;
end
pyhdl_if::PyGILState_Release(gs);
return rid;"""

_REQUEST_PY = """\
pyhdl_if::PyObject t;
pyhdl_if::PyGILState_STATE gs = pyhdl_if::PyGILState_Ensure();
t = cached(h, rid);
if (t != null) begin
  decode(t, 1, rid, req);
  pyhdl_if::PyGILState_Release(gs);
  return;
end
pyhdl_if::PyGILState_Release(gs);
t = py.req_all(h, rid);
gs = pyhdl_if::PyGILState_Ensure();
decode(t, 0, rid, req);
pyhdl_if::Py_DecRef(t);
pyhdl_if::PyGILState_Release(gs);"""


METHODS: List[M] = [
    # -- calls ------------------------------------------------------------ #
    M("call_start", "int", _p("string entry", "string comp_path",
                              "longint unsigned args[$]"),
      "Start function or action `entry` (\"\" = the image's entry) on the "
      "instance at `comp_path` (\"\" = the root) with `args`. A call handle "
      "(> 0), or 0 (`error(0)` says why).",
      refuse="0",
      py="return py.call_start_at(entry, comp_path, args_tuple(args), t_arg());"),
    M("run", "zsp_rt_bc_status_e", _p("int h"),
      "Run call `h` until the host is needed; its status.",
      refuse="ZSP_RT_BC_ERROR", py=_RUN_PY),
    M("take", "int", _p("int h"),
      "The next request REQS reported (its req_id); -1 when none is left.",
      refuse="-1", py=_TAKE_PY),
    M("request", "void", _p("int h", "int rid", "zsp_rt_bc_req req"),
      "Request `rid`'s fields into `req`; rid -1: the run status's (WAIT, "
      "STOP).",
      refuse="",
      py=_REQUEST_PY),

    M("complete", "int", _p("int h", "int rid", "longint unsigned data"),
      "Answer request `rid`: a read's data or an import's result. 0, or -1.",
      refuse="-1", py="return py.complete_at(h, rid, data, t_arg());"),
    M("set_ret", "int", _p("int h", "int rid", "int i", "longint unsigned v"),
      "Stage field `i` of request `rid`'s struct result, before `complete`.",
      refuse="-1"),
    M("string_ret", "int", _p("int h", "int rid", "int i", "string text"),
      "Stage the text of string result leaf `i` of request `rid`, before "
      "`complete`.",
      refuse="-1"),
    M("result", "longint unsigned", _p("int h"),
      "Call `h`'s result after DONE.",
      refuse="0",
      py="if (result_c.exists(h)) return result_c[h];\nsync();\n"
         "return py.result(h);"),
    M("obj_field", "longint unsigned", _p("int h", "string name"),
      "After DONE of call `h` on an action, before call_end: its object's "
      "field `name` (0, and error(h) set, if it has none).",
      refuse="0"),
    M("call_end", "void", _p("int h"),
      "End call `h` (any state) and free it.",
      refuse="",
      py="drop(h);\nresult_c.delete(h);\nsync();\npy.call_end(h);"),
    M("notify", "int", _p("int kind", "int id", "longint unsigned value"),
      "A host event (an interrupt, a channel put) for calls waiting on "
      "EVENT. 0, or -1.",
      refuse="-1"),
    M("error", "string", _p("int h"),
      "Call `h`'s failure, located (h 0: the engine's own, or what the SV "
      "side refused, once).",
      value="{return take_refusal();",
      py="if (h == 0 && refusal != \"\") return take_refusal();\n"
         "return py.error(h);"),
    M("fatal", "string", _p(),
      "The fatal error the engine stopped on, located; \"\" when none. "
      "Every call then ends in ERROR, and later ones are refused until a "
      "new tree is built.",
      value="\"\""),
    M("progress", "longint", _p(),
      "A count that grows whenever a call makes progress (a step that "
      "neither parked nor stuck, a channel put): a call waiting on EVENT "
      "or TURN is run again when it moves. -1: the engine keeps none.",
      value="-1",
      py="if (progress_ok) return progress_c;\nsync();\nreturn py.progress();"),
    M("call_inside", "int", _p("int h", "int outer", "int rid"),
      "Call `h`, not started yet, was started from inside request `rid` "
      "of call `outer` (the process serving it): a traversal that would "
      "queue behind `outer` then ends in ERROR (self-wait) instead of "
      "waiting for ever. 0, or -1.",
      refuse="-1"),
    # -- objects (host-abi.md "Objects") ----------------------------------- #
    M("new_obj", "int", _p("string type_name", "int parent"),
      "`parent` 0: construct the tree of root component `type_name` and "
      "give its handle; else an action object of action `type_name` under "
      "component handle `parent`. A handle (> 0), or 0 (`error(0)`).",
      refuse="0"),
    M("sub", "int", _p("int parent", "string path"),
      "A handle of the instance at `path` under component handle "
      "`parent`; 0 (`error(0)`) if none.",
      refuse="0"),
    M("call_obj", "int", _p("int obj", "string method",
                            "longint unsigned args[$]"),
      "A call keyed by handle: the traversal of action object `obj` "
      "(`method` \"\"), function `method` of component `obj`, or "
      "`$run_end` on the root. A call handle (> 0), or 0 (`error(0)`).",
      refuse="0",
      py="return py.call_obj_at(obj, method, args_tuple(args), t_arg());"),
    M("release_obj", "int", _p("int obj"),
      "Free handle `obj` (the root's frees the tree). 0, or -1 "
      "(`error(0)`).",
      refuse="-1"),
    M("field", "longint unsigned", _p("int obj", "string name"),
      "Field `name` of action object `obj` once no call on it runs; 0 "
      "and `error(0)` if it has none.",
      refuse="0"),
    M("pin", "int", _p("int obj", "string path", "string value"),
      "Fix field `path` (`sz`, `x1.sz`) of action object `obj` to the "
      "value text `value` names (`64`, `0x40`, `16'h40`, an enum item, "
      "`true`) in its later traversals: an override. An unknown field, an "
      "out-of-range value or an unknown item is refused here. 0, or -1 "
      "(`error(0)`).",
      refuse="-1"),
    # -- the catalog (pssvm-in-practice 3.0.7) ----------------------------- #
    M("root_candidate", "string", _p("int i"),
      "Root candidate `i` of the image (its qualified type); \"\" past "
      "the last. `new_obj(type, 0)` of another candidate switches to it "
      "while nothing is live.",
      refuse="\"\""),
    M("action_name", "string", _p("int i"),
      "Exported action `i` runnable under the current root (its "
      "qualified name, in name order); \"\" past the last.",
      refuse="\"\""),
    M("action_field", "string", _p("string action", "int i"),
      "Field `i` of exported action `action` that `pin` can fix, as "
      "\"<path> <type>\" (`x1.sz bit[16]`, `mode enum xfer_mode_e`); "
      "\"\" past the last.",
      refuse="\"\""),
    M("instance_path", "string", _p("int i"),
      "The path of component instance `i` of the current root's tree, in "
      "pre-order (0 is the root: \"\"). `instance_type` ends the list.",
      refuse="\"\""),
    M("api_hash", "string", _p(),
      "The current root's API hash (`%016x`), what a package generated from "
      "the image (`pssc sv-api`) was built against; \"\" when unknown.",
      refuse="\"\""),
    M("instance_type", "string", _p("int i"),
      "The component type of instance `i`; \"\" past the last.",
      refuse="\"\""),
    # -- linking (host-abi.md "Linking") ----------------------------------- #
    M("api_id", "int", _p("string name"),
      "The id of API `name` (\"mem\", an import API's name); -1 "
      "(`error(0)`) if the image has none.",
      refuse="-1"),
    M("missing_links", "int", _p("string linked"),
      "How many APIs a reachable coroutine uses with no handler for every "
      "executor, given the links `linked` (`api:ctx` pairs, ';'-separated; "
      "-1 any API or context, -2 no executor); `error(0)` names them. -1: "
      "refused.",
      refuse="-1"),
    M("link", "int", _p("string api", "string ctx",
                        "zsp_rt_bc_platform_if handler"),
      "`handler` serves API `api` (\"mem\", an import API's name, or \"*\": "
      "every API with no handler of its own) for requests of executor "
      "`ctx` (an instance path, \"\" for none, \"*\" for all); null unlinks. "
      "The run loop routes each request to its handler, else to its "
      "platform. 0, or -1 (`error(0)`).",
      impl="common"),
    M("link_check", "int", _p(),
      "0 when every API a reachable coroutine uses has a handler linked "
      "for every executor; else -1, and `error(0)` names what is missing.",
      impl="common"),
    M("handler", "zsp_rt_bc_platform_if", _p("int api", "int executor"),
      "The handler linked for `api` and `executor` (-1: none); null if "
      "none.",
      impl="common"),
    M("add_listener", "void", _p("zsp_rt_bc_listener l"),
      "`l.on_error` hears each call that ends in ERROR, `l.on_fatal` the "
      "fatal. With none, an error is the call's status, and a fatal ends "
      "the simulation (`$fatal`).",
      impl="common"),
    # -- driving every call (host-abi.md "step and pending") -------------- #
    M("step", "int", _p("int budget"),
      "Run the calls that can run, one slice each, round after round, "
      "until none can or `budget` slices ran (0: no limit); their new "
      "requests are then `take`n per call. The slices run, or -1.",
      refuse="-1", py="drop_all();\nsync();\nreturn py.step(budget);"),
    M("pending", "int", _p(),
      "What is left across every call: ZSP_RT_BC_PEND_READY | _REQS | "
      "_TIME flags; -1: refused.",
      refuse="-1"),
    M("next_time", "longint", _p(),
      "The earliest timer a call waits for, in engine ticks; -1: none.",
      refuse="-1"),
    M("set_time", "void", _p("longint unsigned t_fs"),
      "The sim time now, in fs (`zsp_rt_bc_now_fs()`); events the engine "
      "sends carry the last time set. Set before each `run` step of a "
      "call.",
      value="{", py="t_set   = t_fs;\nt_dirty = 1;"),
    # -- the image ------------------------------------------------------- #
    M("time_exp", "int", _p(),
      "One WAIT TIME tick is 10**time_exp s (the image's IDENT; -12 = ps).",
      value="-12",
      py="if (!time_exp_ok) begin\n  sync();\n  time_exp_c  = py.time_exp();\n"
         "  time_exp_ok = 1;\nend\nreturn time_exp_c;"),
    M("exec_id", "int", _p("string path"),
      "The id requests carry for the executor at instance `path` "
      "(`zsp_rt_bc_req.executor`), or -1.",
      value="-1"),
    M("exec_path", "string", _p("int id"),
      "The instance path of executor `id`, or \"\" for none (-1) or "
      "unknown.",
      value="\"\""),
    M("event_id", "int", _p("string path"),
      "The id `notify(2, id, value)` takes for the host-fed channel at "
      "instance `path` (the image's EVENT rows), or -1.",
      value="-1"),
    M("string_at", "string", _p("int idx"),
      "The text of model string `idx` (an import's string argument "
      "arrives as this index); \"\" out of range.",
      value="\"\""),
    M("api_check", "string", _p("string spec"),
      "\"\" when the image has every function of `spec` (a generated "
      "facade's `comp::fn(p,s.f)[->r]` lines, ';'-separated) with those "
      "parameter leaves and result; else what differs. An engine that "
      "cannot tell: \"\".",
      value="\"\""),
    M("fresh", "int", _p(),
      "Build the component tree again (a fresh one for the next test); 0, "
      "or -1 (`error(0)`).",
      refuse="-1",
      py="drop_all();\nresult_c.delete();\nsync();\nreturn py.fresh();"),
    M("message_next", "string", _p("output int sev"),
      "message()/error()/fatal() text the runs produced, oldest first; "
      "\"\" when none is left; `sev` 0 message, 1 error, 2 fatal, 3 "
      "runtime. An engine that prints its own messages has none.",
      value="{sev = 0;\nreturn \"\";", py=""),
    # -- tests and hooks (pssvm_runner) ------------------------------------ #
    M("test_name", "string", _p("int i"),
      "Test `i` as `comp::fn`; \"\" past the last.",
      value="\"\""),
    M("test_expect", "string", _p("int i"),
      "The error test `i` expects (`@pssunit_pkg::expect_error`), or \"\".",
      value="\"\""),
    M("hook", "string", _p("string kind", "int i"),
      "Session (\"session\"), setup or teardown function `i`; \"\" past "
      "the last.",
      value="\"\""),
    M("checks_failed", "int", _p(),
      "pssunit checks failed so far in this engine.",
      value="0"),
    M("report", "int", _p("string kind", "longint unsigned t_fs",
                          "string json"),
      "The bench reported something (`pssvm_report`): `kind` \"uvm\", "
      "\"bench\", ...; `json` {severity, id, msg, keys}; `t_fs` its sim "
      "time. 0, or -1 (`error(0)`). An engine that cannot take reports: "
      "0, and they are not counted.",
      value="0", py="sync();\nreturn py.report(0, kind, t_fs, json);"),
    M("bench_failed", "int", _p(),
      "Bench reports so far that fail a test (error or fatal).",
      value="0"),
    M("trace", "string", _p("string dir", "string name"),
      "Trace test `name` into `dir` (one NDJSON file per test, each event "
      "with its sim time); \"\" `name` ends the trace. The file written, "
      "or \"\".",
      value="\"\"", py=_SET_TIME + "return py.trace(dir, name);"),
    # -- the workspace daemon and the debugger ------------------------------ #
    M("serve", "string", _p("string name", "string top"),
      "Server mode: register with the workspace daemon as `name` (top "
      "module `top`); the session id, or \"\" (`error(0)`).",
      value="\"\""),
    M("poll", "int", _p("int timeout_ms"),
      "Answer one request from the daemon, waiting up to `timeout_ms`: 1 "
      "answered, 2 a `run` is pending, 0 none, -1 the session ended.",
      value="-1", py=_SET_TIME + "return py.poll(timeout_ms);"),
    M("pending_sel", "string", _p(),
      "The pending `run`'s selection (`+pssunit` form; \"\" = every test).",
      value="\"\""),
    M("serve_test", "void", _p("string name", "int passed"),
      "A test of the pending `run` finished.",
      value="{", py=_SET_TIME + "py.serve_test(name, passed);"),
    M("serve_done", "void", _p("int n_pass", "int n_fail"),
      "The pending `run` finished: answer it with the counts.",
      value="{", py=_SET_TIME + "py.serve_done(n_pass, n_fail);"),
    M("dbg_wait", "int", _p("string why"),
      "Hold the simulation for a debugger (`why`: what failed) until it "
      "resumes: 1 run the failed test again, 0 go on, -1 cannot hold.",
      value="-1", py=_SET_TIME + "return py.dbg_wait(why);"),
    M("cur_op", "string", _p("int h"),
      "The function call `h` runs (\"\" when the engine does not know).",
      value="\"\""),
    M("cur_file", "string", _p("int h"),
      "The file of the statement call `h` last stopped at.",
      value="\"\""),
    M("cur_line", "int", _p("int h"),
      "That statement's line (0: none).",
      value="0"),
]

#: pyhdl-if methods the pyvm variant uses besides the forwarded ones:
#: (name, ret, params, body over ``self.e``, the shim's Engine)
TRANSPORT_EXTRA = [
    ("load", "int", _p("string path", "string opts"),
     "self.e.load(path, opts)"),
    ("call_start_at", "int", _p("string entry", "string comp_path",
                                "pyhdl_if::PyObject args", "longint t_fs"),
     "self.e.call_start_at(entry, comp_path, args, t_fs)"),
    ("call_obj_at", "int", _p("int obj", "string method",
                              "pyhdl_if::PyObject args", "longint t_fs"),
     "self.e.call_obj_at(obj, method, args, t_fs)"),
    ("run_all", "pyhdl_if::PyObject", _p("int h", "longint t_fs"),
     "self.e.run_all(h, t_fs)"),
    ("take", "int", _p("int h"), "self.e.take(h)"),
    ("complete_at", "int", _p("int h", "int rid", "longint unsigned data",
                              "longint t_fs"),
     "self.e.complete_at(h, rid, data, t_fs)"),
    ("result", "longint unsigned", _p("int h"), "self.e.result(h)"),
    ("call_end", "void", _p("int h"), "self.e.call_end(h)"),
    ("progress", "longint", _p(), "self.e.progress()"),
    ("step", "int", _p("int budget"), "self.e.step(budget)"),
    ("set_time", "void", _p("longint unsigned t_fs"), "self.e.set_time(t_fs)"),
    ("time_exp", "int", _p(), "self.e.time_exp()"),
    ("fresh", "int", _p(), "self.e.fresh()"),
    ("req_all", "pyhdl_if::PyObject", _p("int h", "int rid"),
     "self.e.req_all(h, rid)"),
    ("error", "string", _p("int h"), "self.e.error(h)"),
    ("report", "int", _p("int h", "string kind", "longint unsigned t_fs",
                         "string js"),
     "self.e.report(h, kind, t_fs, js)"),
    ("trace", "string", _p("string dir", "string name"),
     "self.e.trace(dir, name)"),
    ("poll", "int", _p("int timeout_ms"), "self.e.poll(timeout_ms)"),
    ("serve_test", "void", _p("string name", "int passed"),
     "self.e.serve_test(name, passed)"),
    ("serve_done", "void", _p("int n_pass", "int n_fail"),
     "self.e.serve_done(n_pass, n_fail)"),
    ("dbg_wait", "int", _p("string why"), "self.e.dbg_wait(why)"),
]

_CT = {"int": "ct.c_int32", "longint unsigned": "ct.c_uint64",
       "longint": "ct.c_int64", "string": "str",
       "pyhdl_if::PyObject": "ct.py_object"}


def by_name() -> Dict[str, M]:
    return {m.name: m for m in METHODS}


def _is_enum(t: str) -> bool:
    return t.endswith("_e")


def _forwarded(m: M) -> bool:
    return m.impl == "engine" and m.py is None


def transport() -> List[Tuple[str, str, tuple, str]]:
    """The pyhdl-if API's methods: one per forwarded method, then
    :data:`TRANSPORT_EXTRA`."""
    out = []
    for m in METHODS:
        if _forwarded(m):
            ret = "int" if _is_enum(m.ret) else m.ret
            args = ", ".join(n for _, n in m.params)
            out.append((m.name, ret, m.params, f"self.e.{m.name}({args})"))
    return out + list(TRANSPORT_EXTRA)


def transport_source(engine_cls_name: str = "engine_cls") -> str:
    """Python source of the pyhdl-if class over the shim's Engine, for
    ``make_api`` to execute (``hif``, ``ct`` and ``engine_cls`` in scope)."""
    lines = [f"class {API_CLASS}(object):",
             f"    \"\"\"The shim's Engine for SV (pyhdl-if); generated from "
             f"zuspec.be.bc.share.rt_bc_api.\"\"\"",
             "",
             "    def __init__(self):",
             f"        self.e = {engine_cls_name}()",
             ""]
    for name, ret, params, body in transport():
        ps = "".join(f", {n}: {_CT[t]}" for t, n in params)
        r = "" if ret == "void" else f" -> {_CT[ret]}"
        lines += ["    @hif.exp",
                  f"    def {name}(self{ps}){r}:",
                  f"        {'' if ret == 'void' else 'return '}{body}",
                  ""]
    return "\n".join(lines)


# -- SV ------------------------------------------------------------------- #

_BANNER = ("// Generated by zuspec.be.bc.share.rt_bc_api -- do not edit.\n"
           "// Regenerate: python -m zuspec.be.bc.share.rt_bc_api\n")


def _param(t: str, n: str) -> str:
    if n.endswith("[$]"):
        return f"{t} {n}"
    if t.startswith("output "):
        return t + " " + n
    return f"{t} {n}"


def signature(m: M) -> str:
    """``function <ret> <name>(<params>)``, as every declaration spells it."""
    ps = ", ".join(_param(t, n) for t, n in m.params)
    return f"function {m.ret} {m.name}({ps})"


def _wrap(doc: str, indent: str) -> List[str]:
    words, lines, cur = doc.split(), [], ""
    for w in words:
        if cur and len(indent) + 3 + len(cur) + 1 + len(w) > 78:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return [f"{indent}// {ln}" for ln in lines]


def sv_if() -> str:
    """``zsp_rt_bc_if.svh``: the interface class."""
    out = [_BANNER,
           "// zsp_rt_bc_if -- the engine interface both variants of",
           "// zsp_rt_bc_pkg implement (`class zsp_rt_bc`). Included by",
           "// zsp_rt_bc_common.svh.",
           "",
           "interface class zsp_rt_bc_if;"]
    for m in METHODS:
        out += _wrap(m.doc, "  ")
        out.append(f"  pure virtual {signature(m)};")
    out.append("endclass")
    return "\n".join(out) + "\n"


def _body(stmts: str, indent: str) -> List[str]:
    return [f"{indent}{s}" if s else "" for s in stmts.split("\n")]


def sv_defaults() -> str:
    """``zsp_rt_bc_defaults.svh``: ``zsp_rt_bc_base``'s body for each
    engine method a variant may leave to it."""
    out = [_BANNER,
           "// Included in the body of class zsp_rt_bc_base",
           "// (zsp_rt_bc_common.svh): what an engine method does when the",
           "// variant compiled does not serve it.",
           ""]
    for m in METHODS:
        if m.impl != "engine":
            continue
        out.append(f"    virtual {signature(m)};")
        if m.value is not None:
            if m.value.startswith("{"):
                out += _body(m.value[1:], "      ") if m.value != "{" else []
            else:
                out.append(f"      return {m.value};")
        else:
            out.append(f'      refuse("{m.name}");')
            if m.ret != "void":
                out.append(f"      return {m.refuse};")
        out.append("    endfunction")
    return "\n".join(out) + "\n"


def _hif_api(shim_dir: Path) -> str:
    """``hdl_if api-gen-sv`` of the shim's ``ZspRtBcPy`` (bare classes)."""
    import tempfile
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[4])
    env["PYTHONPATH"] = os.pathsep.join([str(shim_dir), src,
                                         env.get("PYTHONPATH", "")])
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "api.sv"
        r = subprocess.run([sys.executable, "-m", "hdl_if", "api-gen-sv",
                            "-m", SHIM_MODULE, "-o", str(out)],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError("hdl_if api-gen-sv failed:\n" + r.stderr)
        return out.read_text()


def sv_py_variant(hif_api: str) -> str:
    """``zsp_rt_bc_pkg.sv``, the pyvm variant, around ``hif_api`` (the
    pyhdl-if API classes)."""
    out = [_BANNER.rstrip("\n"),
           "//",
           "// zsp_rt_bc_pkg, pyvm variant: the bytecode runtime over pyvm",
           "// (zuspec-be-bc's Python interpreter) in the simulator's Python,",
           "// through pyhdl-if. The C engine's variant (zuspec-rt-eng",
           "// share/sv) has the same package and class names: the file a",
           "// build compiles chooses the interpreter.",
           "//",
           "// Compile after pyhdl_if.sv (pyhdl-if's share/dpi, also on",
           "// +incdir), with this directory on +incdir. At run time",
           "// PYTHONPATH names the shim's directory (`pssvm_dpi_shim`).",
           "//",
           "//   zsp_rt_bc eng = zsp_rt_bc::load(\"x.zbc\", \"platform=fake\");",
           "//",
           "// Load options (`opts`, comma-separated): root=<type>,",
           "// platform=fake (the default here: pssunit's back-door checks see",
           "// the values the simulation's bus returned; `platform=` for none),",
           "// seed=<n>; +pss_seed=<n> wins. Any other option is ignored and",
           "// reported.",
           "",
           '`include "pyhdl_if_macros.svh"',
           "",
           "package zsp_rt_bc_pkg;",
           "  import pyhdl_if::*;",
           "",
           '`include "zsp_rt_bc_common.svh"',
           ""]
    out += ["  " + ln if ln else "" for ln in hif_api.rstrip("\n").split("\n")]
    out += ["",
            "  class zsp_rt_bc extends zsp_rt_bc_base;",
            f"    protected {API_CLASS}_exp_impl py;",
            "",
            "    // Load image `img` and construct its tree; null (and why, on",
            "    // stdout) on failure.",
            "    static function zsp_rt_bc load(string img, string opts = \"\");",
            "      zsp_rt_bc e = new();",
            "      opts = zsp_rt_bc_plusarg_opts(opts);",
            "      // under SV, pyvm keeps the register fake unless told (platform=)",
            "      if (!zsp_rt_bc_str_has({\",\", opts}, \",platform=\"))",
            "        opts = {opts, opts == \"\" ? \"\" : \",\", \"platform=fake\"};",
            "      e.py = new();",
            "      if (e.py.load(img, opts) != 0) begin",
            "        $display(\"PSSVM-ERROR: load %s: %s\", img, e.py.error(0));",
            "        return null;",
            "      end",
            "      zsp_rt_bc_loaded(e);",
            "      return e;",
            "    endfunction",
            "",
            "    virtual function string impl();",
            "      return \"pyvm\";",
            "    endfunction",
            ""]
    out += _body(_PY_STATE.rstrip("\n"), "    ")
    out += [""] + _body("\n".join(_decode_sv()), "    ")
    for m in METHODS:
        if m.impl != "engine" or m.py == "":
            continue
        out += ["", f"    virtual {signature(m)};"]
        if m.py is not None:
            out += _body(m.py, "      ")
        else:
            args = ", ".join(n.rstrip("[$]") for _, n in m.params)
            call = f"py.{m.name}({args})"
            out.append("      sync();")
            if m.ret == "void":
                out.append(f"      {call};")
            elif _is_enum(m.ret):
                out.append(f"      return {m.ret}'({call});")
            else:
                out.append(f"      return {call};")
        out.append("    endfunction")
    out += ["  endclass", "", "endpackage"]
    return "\n".join(out) + "\n"


def sv_dir() -> Path:
    return Path(__file__).resolve().parent / "sv"


def py_dir() -> Path:
    return Path(__file__).resolve().parent / "py"


def generated() -> Dict[Path, str]:
    """Every generated file and its text."""
    d = sv_dir()
    return {d / "zsp_rt_bc_if.svh": sv_if(),
            d / "zsp_rt_bc_defaults.svh": sv_defaults(),
            d / "zsp_rt_bc_pkg.sv": sv_py_variant(_hif_api(py_dir()))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m zuspec.be.bc.share.rt_bc_api",
                                 description="regenerate zsp_rt_bc's SV files")
    ap.add_argument("--check", action="store_true",
                    help="report stale files; write nothing")
    a = ap.parse_args(argv)
    stale = []
    for path, text in generated().items():
        if path.exists() and path.read_text() == text:
            continue
        stale.append(path)
        if not a.check:
            path.write_text(text)
    for p in stale:
        print(("stale: " if a.check else "wrote: ") + str(p))
    return 1 if (a.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
