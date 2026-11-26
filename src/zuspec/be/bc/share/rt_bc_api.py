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


_SET_TIME = "py.set_time(zsp_rt_bc_now_fs());\n"

METHODS: List[M] = [
    # -- calls ------------------------------------------------------------ #
    M("call_start", "int", _p("string entry", "string comp_path",
                              "longint unsigned args[$]"),
      "Start function or action `entry` (\"\" = the image's entry) on the "
      "instance at `comp_path` (\"\" = the root) with `args`. A call handle "
      "(> 0), or 0 (`error(0)` says why).",
      refuse="0",
      py="foreach (args[i]) py.push_arg(args[i]);\n"
         "return py.call_start(entry, comp_path);"),
    M("run", "zsp_rt_bc_status_e", _p("int h"),
      "Run call `h` until the host is needed; its status.",
      refuse="ZSP_RT_BC_ERROR"),
    M("take", "int", _p("int h"),
      "The next request REQS reported (its req_id); -1 when none is left.",
      refuse="-1"),
    M("request", "void", _p("int h", "int rid", "zsp_rt_bc_req req"),
      "Request `rid`'s fields into `req`; rid -1: the run status's (WAIT, "
      "STOP).",
      refuse="",
      py="int n;\n"
         "req.req_id   = rid;\n"
         "req.api      = py.req_int(h, rid, \"api\");\n"
         "req.executor = py.req_int(h, rid, \"executor\");\n"
         "req.thread   = py.req_int(h, rid, \"thread\");\n"
         "req.write    = py.req_int(h, rid, \"kind\") != 0;\n"
         "req.addr     = py.req_u64(h, rid, \"addr\");\n"
         "req.size     = py.req_int(h, rid, \"size\");\n"
         "req.data     = py.req_u64(h, rid, \"data\");\n"
         "req.reg_id   = py.req_int(h, rid, \"reg_id\");\n"
         "req.src_ref  = py.req_str(h, rid, \"src_ref\");\n"
         "req.stim_id  = py.req_str(h, rid, \"stim_id\");\n"
         "req.fn_id    = py.req_int(h, rid, \"fn_id\");\n"
         "req.name     = py.req_str(h, rid, \"name\");\n"
         "req.blocking = py.req_int(h, rid, \"blocking\") != 0;\n"
         "req.on       = zsp_rt_bc_wait_on_e'(py.req_int(h, rid, \"on\"));\n"
         "req.delay    = py.req_u64(h, rid, \"delay\");\n"
         "req.reason   = py.req_str(h, rid, \"reason\");\n"
         "req.args.delete();\n"
         "n = py.req_nargs(h, rid);\n"
         "for (int i = 0; i < n; i++) req.args.push_back(py.req_arg(h, rid, i));\n"
         "req.nret     = py.req_int(h, rid, \"nret\");\n"
         "req.rets.delete();\n"
         "req.ret_s.delete();"),
    M("complete", "int", _p("int h", "int rid", "longint unsigned data"),
      "Answer request `rid`: a read's data or an import's result. 0, or -1.",
      refuse="-1"),
    M("set_ret", "int", _p("int h", "int rid", "int i", "longint unsigned v"),
      "Stage field `i` of request `rid`'s struct result, before `complete`.",
      refuse="-1"),
    M("string_ret", "int", _p("int h", "int rid", "int i", "string text"),
      "Stage the text of string result leaf `i` of request `rid`, before "
      "`complete`.",
      refuse="-1"),
    M("result", "longint unsigned", _p("int h"),
      "Call `h`'s result after DONE.",
      refuse="0"),
    M("obj_field", "longint unsigned", _p("int h", "string name"),
      "After DONE of call `h` on an action, before call_end: its object's "
      "field `name` (0, and error(h) set, if it has none).",
      refuse="0"),
    M("call_end", "void", _p("int h"),
      "End call `h` (any state) and free it.",
      refuse=""),
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
      value="-1"),
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
      py="foreach (args[i]) py.push_arg(args[i]);\n"
         "return py.call_obj(obj, method);"),
    M("release_obj", "int", _p("int obj"),
      "Free handle `obj` (the root's frees the tree). 0, or -1 "
      "(`error(0)`).",
      refuse="-1"),
    M("field", "longint unsigned", _p("int obj", "string name"),
      "Field `name` of action object `obj` once no call on it runs; 0 "
      "and `error(0)` if it has none.",
      refuse="0"),
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
      refuse="-1"),
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
      value="{"),
    # -- the image ------------------------------------------------------- #
    M("time_exp", "int", _p(),
      "One WAIT TIME tick is 10**time_exp s (the image's IDENT; -12 = ps).",
      value="-12"),
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
      refuse="-1"),
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
      value="0", py="return py.report(0, kind, t_fs, json);"),
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
    ("push_arg", "void", _p("longint unsigned v"), "self.e.push_arg(v)"),
    ("call_start", "int", _p("string entry", "string comp_path"),
     "self.e.call_start(entry, comp_path)"),
    ("call_obj", "int", _p("int obj", "string method"),
     "self.e.call_obj(obj, method)"),
    ("req_int", "int", _p("int h", "int rid", "string field"),
     "self.e.req_int(h, rid, field)"),
    ("req_u64", "longint unsigned", _p("int h", "int rid", "string field"),
     "self.e.req_u64(h, rid, field)"),
    ("req_str", "string", _p("int h", "int rid", "string field"),
     "self.e.req_str(h, rid, field)"),
    ("req_nargs", "int", _p("int h", "int rid"), "self.e.req_nargs(h, rid)"),
    ("req_arg", "longint unsigned", _p("int h", "int rid", "int i"),
     "self.e.req_arg(h, rid, i)"),
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
       "longint": "ct.c_int64", "string": "str"}


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
            "    endfunction"]
    for m in METHODS:
        if m.impl != "engine" or m.py == "":
            continue
        out += ["", f"    virtual {signature(m)};"]
        if m.py is not None:
            out += _body(m.py, "      ")
        else:
            args = ", ".join(n.rstrip("[$]") for _, n in m.params)
            call = f"py.{m.name}({args})"
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
