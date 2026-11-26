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
// (`export A(bit[8] mode);`), which fix those fields. +pssvm_opts are
// the engine's load options (`seed=N`: the seed of the calls it starts).
// +pss_seed=<n> seeds either engine (pssc test --seed) and is recorded in
// the trace header: pyvm's +pssvm_trace=DIR file, the C engine's `trace=`.
//
// The engine is whichever zsp_rt_bc_pkg variant the build compiled: the C
// engine over DPI-C (zuspec-rt-eng's) or pyvm under pyhdl-if (be-bc's).
// +pssvm_opts are its load options (`zsp_rt_bc::load`); +pssvm_platform=<name>
// adds pyvm's platform= (pyvm's default under SV: fake).
// Prints the runner's PSSVM-TEST/PSSVM-SUMMARY lines, then
// `PSSVM-TB: done failed=<n> accesses=<n>`; a load failure counts as one.

`timescale 1ns/1ns
module pssvm_tb;
  import zsp_rt_bc_pkg::*;
  import pssvm_pkg::*;
  import pssvm_tb_pkg::*;

  // The engine for `img`, or null (and why, on stdout).
  function automatic zsp_rt_bc load(string img);
    string opts = "", platform;
    void'($value$plusargs("pssvm_opts=%s", opts));
    if ($value$plusargs("pssvm_platform=%s", platform))
      opts = {opts, opts == "" ? "" : ",", "platform=", platform};
    return zsp_rt_bc::load(img, opts);
  endfunction

  // Run one entry (+pssvm_entry[=<name>]) to completion.
  task automatic run_entry(zsp_rt_bc_ctxt ctxt, output int failed);
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
    zsp_rt_bc  eng;
    pssvm_tb_out  out;
    pssvm_tb_plat plat;
    zsp_rt_bc_ctxt    ctxt;
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
