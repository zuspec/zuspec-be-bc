// pssvm_pkg -- the PSS face of running a pssc image (`pssc build --image
// x.zbc`) inside a simulation. Hand-written once, shipped with zuspec-be-bc
// (`pssc sv-pssvm-path`); it moves to pssc with the rest of the PSS layer
// (pssc pssvm-plan step 8). Compile it after zsp_rt_bc_pkg -- either
// variant: pyvm's (this directory) or the C engine's (zuspec-rt-eng) --
// which holds the generic run model (the engine, its requests, the
// platform and the run loop zsp_rt_bc_ctxt). Holds:
//   * pssvm_report, pssvm_bench_fail -- the bench reports a failure to the
//                                       engines loaded (bench failure
//                                       attribution)
//   * pssvm_dbg_wait   -- hold sim time for a debugger
//   * pssvm_runner     -- runs the image's pssunit tests
//   * module pssvm_dbg -- where the model is, as signals (`+pssvm_dbgsig`)
//
// Design: docs/design/pss-test-debug-env-design.md (§10),
// docs/design/pssvm-in-practice.md (§2.1, §3.0.3).

package pssvm_pkg;
  import zsp_rt_bc_pkg::*;

  // Hold sim time for a debugger (`zsp_rt_bc::dbg_wait`) on the first
  // engine loaded; what a testbench calls where it wants to stop.
  function int pssvm_dbg_wait(string why = "");
    if (zsp_rt_bc_engines.size() == 0) return -1;
    return zsp_rt_bc_engines[0].dbg_wait(why);
  endfunction

  // `s` as a JSON string literal.
  function automatic string pssvm_json_str(string s);
    string r = "\"";
    for (int i = 0; i < s.len(); i++) begin
      byte unsigned c = s[i];
      if (c == "\"" || c == "\\") r = {r, "\\", string'(c)};
      else if (c == 8'h0a) r = {r, "\\n"};
      else if (c == 8'h09) r = {r, "\\t"};
      else if (c < 8'h20) r = {r, $sformatf("\\u%04x", c)};
      else r = {r, string'(c)};
    end
    return {r, "\""};
  endfunction

  // Report something the bench saw to each engine loaded (bench failure
  // attribution): `kind` names the source ("uvm", "bench", "sva"),
  // `severity` "info", "warning", "error" or "fatal"; `keys` the context
  // keys that name the transaction (`stim` = the action run's id, when
  // the bench knows it). An error or fatal fails the test running. 0, or
  // -1 (no engine loaded, or an engine's `error(0)`).
  function automatic int pssvm_report(string kind, string severity,
                                      string id, string msg,
                                      string keys[string]);
    string js, sep;
    int rc = 0;
    if (zsp_rt_bc_engines.size() == 0) return -1;
    js = {"{\"severity\":", pssvm_json_str(severity), ",\"id\":",
          pssvm_json_str(id), ",\"msg\":", pssvm_json_str(msg),
          ",\"keys\":{"};
    foreach (keys[k]) begin
      js = {js, sep, pssvm_json_str(k), ":", pssvm_json_str(keys[k])};
      sep = ",";
    end
    js = {js, "}}"};
    foreach (zsp_rt_bc_engines[i])
      if (zsp_rt_bc_engines[i].report(kind, zsp_rt_bc_now_fs(), js) != 0) rc = -1;
    return rc;
  endfunction

  // A scoreboard's structured failure: `kind` names the check (it is the
  // report's id), `keys` its context keys, e.g. '{"stim": "pss#3"} or the
  // address it compared.
  function automatic int pssvm_bench_fail(string kind, string keys[string],
                                          string msg);
    return pssvm_report("bench", "error", kind, msg, keys);
  endfunction

  // Runs the image's tests: those `+pssunit=SEL` selects (comma-separated
  // globs, `*` and `?`, on `comp::fn` or `fn`), or every test. Each test
  // runs between the setup and teardown hooks; it passes when its calls
  // complete and no pssunit check fails (or, for a test that expects an
  // error, when one of them does). Prints one `PSSVM-TEST: PASS|FAIL name
  // @start..end` line per test (the sim time it started and ended at) and a `PSSVM-SUMMARY:` line.
  //
  // Each test starts on a fresh model and design: before every test but
  // the first, the runner builds the component tree again (engine
  // `fresh`) and resets the design (platform `reset`). A platform that
  // cannot (`can_reset` 0) runs only the first test; the others print
  // `PSSVM-TEST: SKIP name (...)` and count as neither passed nor failed.
  // An image with `@pssunit_pkg::session` functions runs them once, before
  // the first test, and then keeps the model and design from one test to
  // the next (no fresh tree, no reset).
  //
  // With `+pssvm_serve[=name]` the runner is in server mode instead: it
  // registers with the workspace daemon (`PSSC_SIM_SOCKET`, else the
  // cwd's) and answers its requests until the daemon ends the session;
  // each `run {sel}` request runs the tests `sel` selects, as above.
  //
  // With `+pssvm_dbg_on_fail` a failing test holds the simulation instead
  // of going on (`PSSVM-DBG: held name @t`; `pssvm_dbg_wait`): a debugger
  // can attach (`pssc sim attach`), the code be reloaded (`pssc sim
  // reload`), and `pssc sim resume [--rerun]` goes on, or runs the test
  // again first. Only the last run of a test counts.
  //
  // With `+pssvm_trace=DIR` each run of a test writes a trace of its
  // events, setup and teardown included, to `DIR/<test-id>.ndjson`
  // (`PSSVM-TRACE: file`), each event with its sim time: `pssc trace show
  // FILE --at 1.234us`.
  class pssvm_runner;
    zsp_rt_bc_ctxt ctxt;
    string     sel[$];
    int        n_pass;
    int        n_fail;
    bit        serving;
    string     serve_name;
    string     top = "";
    bit        dbg_on_fail;
    string     trace_dir = "";
    int        n_skip;
    int        n_runs;      // tests run so far in this simulation
    bit        session_ok = 1;

    function new(zsp_rt_bc_ctxt c);
      string arg;
      ctxt = c;
      if ($value$plusargs("pssunit=%s", arg)) split(arg);
      if ($value$plusargs("pssvm_serve=%s", serve_name)) serving = 1;
      else if ($test$plusargs("pssvm_serve")) serving = 1;
      dbg_on_fail = $test$plusargs("pssvm_dbg_on_fail");
      void'($value$plusargs("pssvm_trace=%s", trace_dir));
    endfunction

    // Server mode: register, then answer requests (simulation time moves
    // by 1 between polls) until the session ends. 0, or 1 when the
    // registration failed.
    task serve(output int failed);
      string sid = ctxt.engine.serve(serve_name, top);
      failed = 0;
      if (sid == "") begin
        $display("PSSVM-SERVE: failed %s", ctxt.engine.error(0));
        failed = 1;
        return;
      end
      $display("PSSVM-SERVE: %s", sid);
      forever begin
        int r = ctxt.engine.poll(100);
        if (r < 0) break;
        if (r == 2) serve_run();
        #1;
      end
      $display("PSSVM-SERVE: end");
    endtask

    // Run the pending `run` request's tests and answer it.
    protected task serve_run();
      sel.delete();
      split(ctxt.engine.pending_sel());
      n_pass = 0;
      n_fail = 0;
      n_skip = 0;
      for (int i = 0; ; i++) begin
        string name = ctxt.engine.test_name(i);
        int was_fail = n_fail;
        if (name == "") break;
        if (!selected(name)) continue;
        run_test(i, name);
        ctxt.engine.serve_test(name, (n_fail == was_fail) ? 1 : 0);
      end
      summary();
      ctxt.engine.serve_done(n_pass, n_fail);
    endtask

    // Does `name` (`comp::fn`) match the selection?
    function bit selected(string name);
      string leaf = name;
      if (sel.size() == 0) return 1;
      for (int i = name.len() - 1; i > 0; i--)
        if (name[i] == ":" && name[i-1] == ":") begin
          leaf = name.substr(i + 1, name.len() - 1);
          break;
        end
      foreach (sel[i])
        if (glob(sel[i], name) || glob(sel[i], leaf)) return 1;
      return 0;
    endfunction

    // Run the selected tests; returns the number that failed.
    task run(output int failed);
      if (serving) begin
        serve(failed);
        return;
      end
      for (int i = 0; ; i++) begin
        string name = ctxt.engine.test_name(i);
        if (name == "") break;
        if (selected(name)) run_test(i, name);
      end
      summary();
      failed = n_fail;
    endtask

    // The `PSSVM-SUMMARY:` line ("..., N skipped" only when some were).
    protected function void summary();
      if (n_skip > 0)
        $display("PSSVM-SUMMARY: %0d passed, %0d failed, %0d skipped", n_pass,
                 n_fail, n_skip);
      else
        $display("PSSVM-SUMMARY: %0d passed, %0d failed", n_pass, n_fail);
    endfunction

    // Does the image have `@pssunit_pkg::session` functions?
    function bit has_session();
      return ctxt.engine.hook("session", 0) != "";
    endfunction

    task run_test(int i, string name);
      bit pass;
      if (n_runs > 0 && !has_session() && !ctxt.plat.can_reset()) begin
        $display("PSSVM-TEST: SKIP %s (platform cannot reset: one test per sim)",
                 name);
        n_skip++;
        return;
      end
      forever begin
        run_once(i, name, pass);
        if (pass || !dbg_on_fail) break;
        $display("PSSVM-DBG: held %s @%0t", name, $time);
        if (ctxt.engine.dbg_wait(name) != 1) break;
      end
      if (pass) n_pass++; else n_fail++;
    endtask

    protected task run_once(int i, string name, output bit pass);
      longint unsigned none[$];
      longint unsigned ret;
      string expect_ = ctxt.engine.test_expect(i);
      int checks, bench;
      time t0 = $time;
      bit ok;
      bit all_ok = 1;
      if (has_session()) begin
        if (n_runs == 0) begin
          hooks("session", session_ok);
          if (!session_ok)
            $display("PSSVM-ERROR: a session function failed: every test fails");
        end
      end else if (n_runs > 0 && ctxt.plat.can_reset()) begin
        if (ctxt.engine.fresh() != 0) begin
          $display("PSSVM-ERROR: fresh tree: %s", ctxt.engine.error(0));
          all_ok = 0;
        end
        ctxt.plat.reset();
      end
      n_runs++;
      all_ok &= session_ok;
      checks = ctxt.engine.checks_failed();
      bench = ctxt.engine.bench_failed();
      if (trace_dir != "") begin
        string f = ctxt.engine.trace(trace_dir, name);
        if (f == "") $display("PSSVM-TRACE: failed %s", ctxt.engine.error(0));
        else $display("PSSVM-TRACE: %s", f);
      end
      hooks("setup", all_ok);
      if (all_ok) begin
        ctxt.call(name, "", none, ret, ok);
        all_ok = ok;
      end
      hooks("teardown", all_ok);
      if (trace_dir != "") void'(ctxt.engine.trace(trace_dir, ""));
      if (ctxt.engine.checks_failed() != checks) all_ok = 0;
      if (ctxt.engine.bench_failed() != bench) all_ok = 0;
      pass = (expect_ == "") ? all_ok : !all_ok;
      $display("PSSVM-TEST: %s %s @%0t..%0t", pass ? "PASS" : "FAIL", name,
               t0, $time);
    endtask

    protected task hooks(string kind, inout bit all_ok);
      longint unsigned none[$];
      longint unsigned ret;
      bit ok;
      for (int j = 0; ; j++) begin
        string fn = ctxt.engine.hook(kind, j);
        if (fn == "") break;
        ctxt.call(fn, "", none, ret, ok);
        if (!ok) all_ok = 0;
      end
    endtask

    protected function void split(string arg);
      int b = 0;
      for (int i = 0; i <= arg.len(); i++)
        if (i == arg.len() || arg[i] == ",") begin
          if (i > b) sel.push_back(arg.substr(b, i - 1));
          b = i + 1;
        end
    endfunction

    // `*` and `?` glob match of the whole of `s`.
    static function bit glob(string p, string s);
      int pi = 0, si = 0, star = -1, mark = 0;
      while (si < s.len()) begin
        if (pi < p.len() && (p[pi] == "?" || p[pi] == s[si])) begin
          pi++; si++;
        end else if (pi < p.len() && p[pi] == "*") begin
          star = pi++; mark = si;
        end else if (star >= 0) begin
          pi = star + 1; si = ++mark;
        end else return 0;
      end
      while (pi < p.len() && p[pi] == "*") pi++;
      return pi == p.len();
    endfunction
  endclass

endpackage

// Where the PSS model is, as signals (`+pssvm_dbgsig`): instantiate it
// anywhere (`pssvm_dbg u_pssvm_dbg();`) and add its signals to the
// waveform. `cur_op` is the function a call runs, `cur_file`/`cur_line`
// the statement it last stopped at; `seq` counts the updates.
module pssvm_dbg;
  string cur_op;
  string cur_file;
  int    cur_line;
  int    seq;

  always @(zsp_rt_bc_pkg::zsp_rt_bc_dbg_seq) begin
    cur_op   = zsp_rt_bc_pkg::zsp_rt_bc_dbg_op;
    cur_file = zsp_rt_bc_pkg::zsp_rt_bc_dbg_file;
    cur_line = zsp_rt_bc_pkg::zsp_rt_bc_dbg_line;
    seq      = zsp_rt_bc_pkg::zsp_rt_bc_dbg_seq;
  end
endmodule
