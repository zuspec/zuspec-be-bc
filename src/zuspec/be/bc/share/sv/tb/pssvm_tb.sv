// pssvm_tb -- the generic bench: runs an image's tests on pssvm_tb_plat.
//
//   simv +img=x.zbc [+pssunit=<globs>] [+pssvm_mem=..] [+pssvm_reg=..] [+pssvm_irq=..]
//   simv +img=x.zbc +pssvm_entry[=<name>] [+pssvm_args=<v>,..] [+pss_seed=<n>]
//        [+pssvm_trace=DIR]
//        [+pssvm_opts=seed=<n>,verbosity=<n>,trace=<file>]
//
// With +pssvm_entry the bench runs one entry instead of the image's tests:
// <name>, or the image's own entry (an action root, `lower_module`) when
// none is given, and prints `PSSVM-ENTRY: PASS|FAIL <name>` (FAIL, too, when
// the bench reported a failure: `pssvm_bench_fail`). +pssvm_args are
// its arguments (decimal): a function's, or an exported action's parameters
// (`export A(bit[8] mode);`), which fix those fields. +pssvm_opts is
// the C engine's load options (`seed=N`: the seed of the calls it starts).
// +pss_seed=<n> seeds either engine (pssc test --seed) and is recorded in
// the trace header: pyvm's +pssvm_trace=DIR file, the C engine's `trace=`.
//
// The engine is the C engine over DPI-C with +define+PSSVM_DPI, else pyvm
// under pyhdl-if (pssvm_hif_pkg; +pssvm_platform=<name>, default fake).
// Prints the runner's PSSVM-TEST/PSSVM-SUMMARY lines, then
// `PSSVM-TB: done failed=<n> accesses=<n>`; a load failure counts as one.

`timescale 1ns/1ns
module pssvm_tb;
  import pssvm_pkg::*;
`ifndef PSSVM_DPI
  import pssvm_hif_pkg::*;
`endif
  import pssvm_tb_pkg::*;

  // The engine for `img`, or null (and why, on stdout).
  function automatic pssvm_engine load(string img);
`ifdef PSSVM_DPI
    pssvm_engine_dpi e;
    string opts = "";
    void'($value$plusargs("pssvm_opts=%s", opts));
    e = new(img, "", opts);
    if (e.e == null) begin
      $display("PSSVM-ERROR: load %s: the C engine refused it", img);
      return null;
    end
    return e;
`else
    string platform = "fake";
    void'($value$plusargs("pssvm_platform=%s", platform));
    return pssvm_engine_hif::create(img, "", platform);
`endif
  endfunction

  // Run one entry (+pssvm_entry[=<name>]) to completion.
  task automatic run_entry(pssvm_ctxt ctxt, output int failed);
    string           entry = "";
    longint unsigned args[$], noargs[$];
    longint unsigned ret;
    bit              ok;
    string           dir = "", f = "", a = "";
    int              st = 0;
    int              bench = ctxt.engine.bench_failed();
    void'($value$plusargs("pssvm_entry=%s", entry));
    if ($value$plusargs("pssvm_args=%s", a))
      for (int i = 0; i <= a.len(); i++)
        if (i == a.len() || a[i] == ",") begin
          args.push_back(longint'(a.substr(st, i - 1).atoi()));
          st = i + 1;
        end
    if ($value$plusargs("pssvm_trace=%s", dir)) begin
      f = ctxt.engine.trace(dir, entry == "" ? "entry" : entry);
      if (f != "") $display("PSSVM-TRACE: %s", f);
    end
    ctxt.call(entry, "", args, ret, ok);
    // the end of the PSS run: the tree's exec run_end blocks (F2.5)
    if (ok) ctxt.call("$run_end", "", noargs, ret, ok);
    if (f != "") void'(ctxt.engine.trace(dir, ""));
    // a bench failure (pssvm_bench_fail, the UVM report catcher) fails it
    if (ctxt.engine.bench_failed() != bench) ok = 0;
    failed = ok ? 0 : 1;
    $display("PSSVM-ENTRY: %s %s", ok ? "PASS" : "FAIL", entry == "" ? "<entry>" : entry);
  endtask

  initial begin : run
    pssvm_engine  eng;
    pssvm_tb_out  out;
    pssvm_tb_plat plat;
    pssvm_ctxt    ctxt;
    pssvm_runner  runner;
    pssvm_tb_irq  irq;
    string        img;
    int           failed;
    out  = new;
    plat = new(out);
    if (!$value$plusargs("img=%s", img)) $display("PSSVM-ERROR: no +img=<image.zbc>");
    else eng = load(img);
    if (eng == null) failed = 1;
    else begin
      plat.engine = eng;
      pssvm_set_default_engine(eng);      // pssvm_report / pssvm_bench_fail
      ctxt = new(eng, plat);
      irq  = new(ctxt, out);
      irq.start_plusargs();
      if ($test$plusargs("pssvm_entry")) run_entry(ctxt, failed);
      else begin
        runner = new(ctxt);
        runner.run(failed);
      end
    end
    $display("PSSVM-TB: done failed=%0d accesses=%0d", failed, plat.n_access);
    out.close();
    $finish;
  end
endmodule
