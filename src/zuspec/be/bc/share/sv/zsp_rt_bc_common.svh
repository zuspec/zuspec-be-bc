// zsp_rt_bc_common.svh -- the part of zsp_rt_bc_pkg both variants share.
// Hand-written once, shipped with zuspec-be-bc (`share/sv`), and included
// inside `package zsp_rt_bc_pkg` by each variant: pyvm's (this directory's
// zsp_rt_bc_pkg.sv, over pyhdl-if) and the C engine's (zuspec-rt-eng's
// share/sv/zsp_rt_bc_pkg.sv, over DPI-C). Which variant a build compiles
// chooses the interpreter; user code names neither (design
// pssvm-in-practice.md §2.1). Holds:
//   * zsp_rt_bc_status_e, zsp_rt_bc_req -- the host ABI's run status and a
//                                          taken request (zuspec-be-bc spec
//                                          host-abi)
//   * zsp_rt_bc_platform_if -- what serves requests: bus accesses, imports,
//                              reset, time (the testbench's platform, or a
//                              handler `link`ed for one API)
//   * zsp_rt_bc_if   -- the engine interface (generated: zsp_rt_bc_if.svh)
//   * zsp_rt_bc_base -- what every engine shares: a refusal for each method
//                       the variant does not serve (generated:
//                       zsp_rt_bc_defaults.svh), links and listeners
//   * zsp_rt_bc_ctxt -- engine + platform; `call` is the run loop: one SV
//                       task per call, one SV process per request the call
//                       has outstanding
//
// Values crossing the boundary are scalars (64-bit, little-endian as in the
// host ABI); access data is the read or written value, import arguments are
// one 64-bit value each, strings are SV strings.
//
// An engine error is a located failure: `call` returns ok = 0 and the
// message (from the engine's `error`) is in `last_error`; listeners hear it
// (`add_listener`). A fatal error -- the engine cannot go on -- ends the
// simulation (`$fatal`) unless a listener is registered.

  localparam int ZSP_RT_BC_ABI_VERSION = 2;

  // `run` result.
  typedef enum int {
    ZSP_RT_BC_ERROR  = -1,
    ZSP_RT_BC_DONE   = 0,
    ZSP_RT_BC_REQS   = 1,   // new requests: `take` them
    ZSP_RT_BC_WAIT   = 3,
    ZSP_RT_BC_STOP   = 4
  } zsp_rt_bc_status_e;

  // What a WAIT waits on.
  typedef enum int {
    ZSP_RT_BC_WAIT_TIME  = 0,
    ZSP_RT_BC_WAIT_EVENT = 1,
    ZSP_RT_BC_WAIT_REQS  = 2,   // a completion
    ZSP_RT_BC_WAIT_TURN  = 3    // a traversal queued behind the active one
                                // (pyvm; the C engine refuses it at start)
  } zsp_rt_bc_wait_on_e;

  // `pending()` flags.
  localparam int ZSP_RT_BC_PEND_READY = 1;
  localparam int ZSP_RT_BC_PEND_REQS  = 2;
  localparam int ZSP_RT_BC_PEND_TIME  = 4;

  // What a listener hears.
  typedef enum int {
    ZSP_RT_BC_SEV_ERROR = 1,    // a call failed; the engine goes on
    ZSP_RT_BC_SEV_FATAL = 2     // the engine cannot go on
  } zsp_rt_bc_sev_e;

  // A request a call took (`take`), or (req_id -1) what its run status
  // says: `on`, `delay` after WAIT, `reason` after STOP.
  class zsp_rt_bc_req;
    int                req_id;
    int                api;       // 0: memory access; else an import
    int                executor;  // index into the image's executors; -1: none
    int                thread;    // engine thread
    // memory access
    bit                write;
    longint unsigned   addr;
    int                size;      // bytes: 1, 2, 4 or 8
    longint unsigned   data;      // write data
    int                reg_id;
    string             src_ref;
    // both: the action run that made the request (`top_a#0.l2#0`)
    string             stim_id;
    // import
    int                fn_id;
    string             name;
    longint unsigned   args[$];
    bit                blocking;
    // A struct result's field count (0: a scalar or none). The platform's
    // call_import answers it by filling `rets`, one value per field.
    int                nret;
    longint unsigned   rets[$];
    // A string result's text, by result leaf (0 for a scalar; a struct's
    // string field at its leaf index): call_import sets `ret_s[i]` for each
    // string it returns (`ret`/`rets` are then ignored for that leaf).
    string             ret_s[int];
    // WAIT
    zsp_rt_bc_wait_on_e on;
    longint unsigned   delay;     // engine time units (WAIT TIME)
    // STOP
    string             reason;
  endclass

  // What serves requests: the testbench's platform (every request no
  // linked handler takes), or a handler `link`ed for one API. Bus accesses
  // and imports are tasks: they may consume simulation time. Override the
  // widths the bus has; an access of a width left alone is reported and
  // reads 0.
  virtual class zsp_rt_bc_platform_if;
    // The action run (stimulus id, `top_a#0.l2#0`) behind the access or
    // import being started; "" outside an action. Copy it before consuming
    // time (parallel calls overwrite it), e.g. onto the sequence item, so a
    // scoreboard can name it: pssvm_bench_fail(.., keys{"stim": id}, ..) or
    // a `pss#<id>` token in its uvm_error message.
    string stim_id;

    virtual task read8 (longint unsigned addr, output bit [7:0]  data);
      unsupported("read8", addr);  data = '0;
    endtask
    virtual task read16(longint unsigned addr, output bit [15:0] data);
      unsupported("read16", addr); data = '0;
    endtask
    virtual task read32(longint unsigned addr, output bit [31:0] data);
      unsupported("read32", addr); data = '0;
    endtask
    virtual task read64(longint unsigned addr, output bit [63:0] data);
      unsupported("read64", addr); data = '0;
    endtask
    virtual task write8 (longint unsigned addr, bit [7:0]  data);
      unsupported("write8", addr);
    endtask
    virtual task write16(longint unsigned addr, bit [15:0] data);
      unsupported("write16", addr);
    endtask
    virtual task write32(longint unsigned addr, bit [31:0] data);
      unsupported("write32", addr);
    endtask
    virtual task write64(longint unsigned addr, bit [63:0] data);
      unsupported("write64", addr);
    endtask

    // A memory access request; by default the read/write task of its width.
    // Override to route by executor (`req.executor`; its instance path is
    // `engine.exec_path(req.executor)`, -1: none), e.g. to one agent per
    // core -- or `link` a handler per executor. Requests of one executor
    // arrive one at a time, in issue order.
    virtual task access(zsp_rt_bc_req req, output longint unsigned rdata);
      bit [7:0]  d8;
      bit [15:0] d16;
      bit [31:0] d32;
      bit [63:0] d64;
      rdata = 0;
      if (req.write) begin
        case (req.size)
          1: write8 (req.addr, req.data[7:0]);
          2: write16(req.addr, req.data[15:0]);
          4: write32(req.addr, req.data[31:0]);
          default: write64(req.addr, req.data);
        endcase
      end else begin
        case (req.size)
          1: begin read8 (req.addr, d8);  rdata = 64'(d8);  end
          2: begin read16(req.addr, d16); rdata = 64'(d16); end
          4: begin read32(req.addr, d32); rdata = 64'(d32); end
          default: begin read64(req.addr, d64); rdata = d64; end
        endcase
      end
    endtask

    // A target import the engine yields (`req.name`, `req.args`). A
    // non-blocking one (`req.blocking == 0`) must not consume time. One
    // returning a struct (`req.nret` fields) pushes them on `req.rets`; a
    // string result (or string field) goes in `req.ret_s[leaf]`.
    virtual task call_import(zsp_rt_bc_req req, output longint unsigned ret);
      $display("PSSVM-ERROR: no platform import '%s'", req.name);
      ret = 0;
    endtask

    // A message()/error()/fatal() the engine formatted (`sev` as
    // zsp_rt_bc_if::message_next).
    virtual function void message(int sev, string text);
      $display("PSSVM-MSG: %s", text);
    endfunction

    // One engine tick is 10**time_exp s; zsp_rt_bc_ctxt sets it from the
    // image (engine.time_exp()) at each call.
    int time_exp = -12;

    // Wait `delay` engine ticks, whatever the timescale: `1s` is one
    // second in this package's unit, and the delay rounds to its precision.
    virtual task wait_time(longint unsigned delay);
      #(real'(delay) * (10.0 ** time_exp) * 1s);
    endtask

    // The debug stub holds simulation time here until it resumes (P3.7).
    virtual task dbg_wait(zsp_rt_bc_req req);
    endtask

    // Put the design back in its reset state, between two tests that run
    // in one simulation (pssvm_runner). A platform whose design cannot be
    // reset says so with `can_reset`; the runner then runs one test.
    virtual task reset();
    endtask
    virtual function bit can_reset();
      return 1;
    endfunction

    protected function void unsupported(string what, longint unsigned addr);
      $display("PSSVM-ERROR: platform has no %s (addr 'h%0h)", what, addr);
    endfunction
  endclass

  // What a listener hears: a call that ended in ERROR (`call_id` is the
  // call handle, `entry` what it ran), or the fatal.
  class zsp_rt_bc_event;
    zsp_rt_bc_sev_e sev;
    string          msg;
    int             call_id;
    string          entry;
  endclass

  // Functions: a listener may not consume time.
  interface class zsp_rt_bc_listener;
    pure virtual function void on_error(zsp_rt_bc_event e);
    pure virtual function void on_fatal(zsp_rt_bc_event e);
  endclass

`include "zsp_rt_bc_if.svh"

  // The sim time now in fs, whatever the timescale (`1s` is one second in
  // this package's time unit).
  function longint unsigned zsp_rt_bc_now_fs();
    return longint'($realtime * (1.0e15 / real'(1s)));
  endfunction

  // 1 when `s` contains `sub`.
  function automatic bit zsp_rt_bc_str_has(string s, string sub);
    for (int i = 0; i + sub.len() <= s.len(); i++)
      if (s.substr(i, i + sub.len() - 1) == sub) return 1;
    return 0;
  endfunction

  // Load option `dbg_signal=1`: where the last call step was, for module
  // `pssvm_dbg` (pssvm_pkg) to show in a waveform. `zsp_rt_bc_ctxt::call`
  // sets them after each `run` step; `zsp_rt_bc_dbg_seq` counts the updates.
  bit    zsp_rt_bc_dbg_on;
  string zsp_rt_bc_dbg_op;
  string zsp_rt_bc_dbg_file;
  int    zsp_rt_bc_dbg_line;
  int    zsp_rt_bc_dbg_seq;

  // Remove option `key` from `opts` and return its value ("" when absent;
  // the last one wins when it is given twice).
  function automatic string zsp_rt_bc_take_opt(inout string opts, input string key);
    string kept = "", val = "";
    int b = 0;
    for (int i = 0; i <= opts.len(); i++)
      if (i == opts.len() || opts[i] == ",") begin
        string item = (i > b) ? opts.substr(b, i - 1) : "";
        b = i + 1;
        if (item == "") continue;
        if (item.len() > key.len() && item.substr(0, key.len()) == {key, "="})
          val = item.substr(key.len() + 1, item.len() - 1);
        else
          kept = (kept == "") ? item : {kept, ",", item};
      end
    opts = kept;
    return val;
  endfunction

  // The load options this package serves itself, for either variant,
  // taken out of `opts` before the engine sees them: `dbg_signal=1` (the
  // waveform signal of `zsp_rt_bc_dbgsig_on`) and `dbg_start=1` (hold
  // before the first call). This package reads no plusargs: its host
  // passes every option in `opts`.
  function automatic void zsp_rt_bc_sv_opts(inout string opts, output bit dbg_start);
    string v;
    v = zsp_rt_bc_take_opt(opts, "dbg_signal");
    if (v != "" && v != "0") zsp_rt_bc_dbg_on = 1;
    v = zsp_rt_bc_take_opt(opts, "dbg_start");
    dbg_start = (v != "" && v != "0");
  endfunction

  // Report each option of `opts` whose key `known` lacks: the variant
  // `impl` ignores it.
  function automatic void zsp_rt_bc_check_opts(string impl, string opts,
                                               string known[$]);
    int b = 0;
    for (int i = 0; i <= opts.len(); i++)
      if (i == opts.len() || opts[i] == ",") begin
        string item = (i > b) ? opts.substr(b, i - 1) : "";
        string key = item;
        b = i + 1;
        foreach (item[j])
          if (item[j] == "=") begin
            key = (j > 0) ? item.substr(0, j - 1) : "";
            break;
          end
        if (key == "") continue;
        begin
          bit ok = 0;
          foreach (known[k]) if (known[k] == key) ok = 1;
          if (!ok)
            $display("PSSVM-WARNING: zsp_rt_bc::load: option '%s' is not known to %s: ignored",
                     key, impl);
        end
      end
  endfunction

  // What every engine shares, whichever variant: the method list's
  // defaults (a refusal through error(0) for each engine method the variant
  // does not serve), links, listeners and the fatal.
  virtual class zsp_rt_bc_base implements zsp_rt_bc_if;
    // What the SV side refused, for error(0) once.
    protected string               refusal;
    protected bit                  refused[string];
    // Linked handlers, by "api:ctx" (-1 any, -2 no executor).
    protected zsp_rt_bc_platform_if links[string];
    protected zsp_rt_bc_listener   listeners[$];
    // The fatal the listeners heard (each is heard once).
    protected string               fatal_seen;

    // The interpreter, for messages: "pyvm", "the C engine".
    virtual function string impl();
      return "this engine";
    endfunction

    // Method `name` is not served by this variant: error(0) says so, and
    // the first refusal of each method is printed.
    protected function void refuse(string name);
      refusal = $sformatf("zsp_rt_bc::%s is not served by %s", name, impl());
      if (!refused.exists(name)) begin
        refused[name] = 1;
        $display("PSSVM-ERROR: %s", refusal);
      end
    endfunction

    // The refusal error(0) gives, once; "" when none.
    function string take_refusal();
      string m = refusal;
      refusal = "";
      return m;
    endfunction

`include "zsp_rt_bc_defaults.svh"

    // -- links ------------------------------------------------------------ //

    virtual function int link(string api, string ctx,
                              zsp_rt_bc_platform_if handler);
      int a, c;
      string key;
      if (api == "*") a = -1;
      else if (api == "mem") a = 0;
      else begin
        a = api_id(api);
        if (a < 0) return -1;
      end
      if (ctx == "*") c = -1;
      else if (ctx == "") c = -2;
      else begin
        c = exec_id(ctx);
        if (c < 0) begin
          refusal = $sformatf("link %s: no executor at '%s'", api, ctx);
          return -1;
        end
      end
      key = $sformatf("%0d:%0d", a, c);
      if (handler == null) links.delete(key);
      else links[key] = handler;
      return 0;
    endfunction

    // Executor paths seen, by id (one exec_path per executor, not per request).
    protected string exec_paths[int];

    // An executor as hosts link it: -2 for none, and for the default
    // executor inferred when the model names none (path "", pssvm 8e), which
    // is the platform's own context (`link(api, "")`); else the executor's id.
    virtual function int exec_key(int executor);
      if (executor < 0) return -2;
      if (!exec_paths.exists(executor)) exec_paths[executor] = exec_path(executor);
      return (exec_paths[executor] == "") ? -2 : executor;
    endfunction

    virtual function zsp_rt_bc_platform_if handler(int api, int executor);
      int e = exec_key(executor);
      string keys[4];
      keys[0] = $sformatf("%0d:%0d", api, e);
      keys[1] = $sformatf("%0d:-1", api);
      keys[2] = $sformatf("-1:%0d", e);
      keys[3] = "-1:-1";
      foreach (keys[i])
        if (links.exists(keys[i])) return links[keys[i]];
      return null;
    endfunction

    virtual function int link_check();
      string s;
      foreach (links[k]) s = {s, k, ";"};
      return (missing_links(s) == 0) ? 0 : -1;
    endfunction

    // -- listeners -------------------------------------------------------- //

    virtual function void add_listener(zsp_rt_bc_listener l);
      listeners.push_back(l);
    endfunction

    // Call `h` (on `entry`) ended in ERROR with `msg`: the listeners hear
    // it. When the engine stopped on a fatal, they hear the fatal instead,
    // once (the calls it released have no event of their own); with no
    // listener, the fatal ends the simulation.
    function void call_failed(int h, string entry, string msg);
      zsp_rt_bc_event e;
      string f = fatal();
      if (f == "") fatal_seen = "";
      else if (f == fatal_seen) return;
      e = new;
      e.call_id = h;
      e.entry   = entry;
      if (f != "") begin
        fatal_seen = f;
        e.sev = ZSP_RT_BC_SEV_FATAL;
        e.msg = f;
        if (listeners.size() == 0) $fatal(1, "PSSVM FATAL: %s", f);
        foreach (listeners[i]) listeners[i].on_fatal(e);
        return;
      end
      e.sev = ZSP_RT_BC_SEV_ERROR;
      e.msg = msg;
      foreach (listeners[i]) listeners[i].on_error(e);
    endfunction
  endclass

  function bit zsp_rt_bc_dbgsig_on();
    return zsp_rt_bc_dbg_on;
  endfunction

  // The engines loaded in this simulation, in load order: bench reports
  // (pssvm_pkg's `pssvm_report`) go to each.
  zsp_rt_bc_base zsp_rt_bc_engines[$];

  // `load` of either variant: `e` loaded. With `dbg_start` (load option
  // `dbg_start=1`) it holds sim time (`dbg_wait("start")`) before its first
  // call, so a debugger can attach and set breakpoints in a testbench that
  // does not serve.
  function void zsp_rt_bc_loaded(zsp_rt_bc_base e, bit dbg_start = 0);
    zsp_rt_bc_engines.push_back(e);
    if (!dbg_start) return;
    $display("PSSVM-DBG: held start @%0t", $time);
    void'(e.dbg_wait("start"));
  endfunction

  // An engine and its platform. `call` runs one entry to completion; fork
  // calls for parallel operations on the one model. Each request a call
  // has outstanding is served by an SV process of its own: by the handler
  // `link`ed for its API and executor, else the platform (memory accesses
  // by `access`, imports by `call_import`). An executor multitasks: imports
  // are not ordered, so one thread blocked in an import (an interrupt wait)
  // never holds up another. Memory accesses of one executor, the inferred
  // default included, enter its handler in issue order; whether one finishes
  // before the next starts is the handler's choice (a simple bus serializes,
  // a pipelined one need not). Accesses with no executor (-1) are unordered.
  //
  // A call waiting on EVENT (a channel, a notify) or TURN (a traversal
  // queued behind the active one) runs again when another call ends, when
  // the testbench notifies, and when another call makes progress mid-way
  // (the engine's `progress()`: a function call that puts on a channel and
  // then blocks on an import wakes the traversal waiting on it).
  class zsp_rt_bc_ctxt;
    zsp_rt_bc_base        engine;
    zsp_rt_bc_platform_if plat;
    string                last_error;
    int                   n_errors;
    event                 notified;
    // Per live call: requests completed and still outstanding.
    protected int         n_done[int];
    protected int         n_out[int];
    event                 completed;
    // Per executor: the next memory-access ticket to hand out, and the one
    // whose turn it is to enter the handler.
    protected int         exec_ticket[int];
    protected int         exec_turn[int];
    protected event       exec_next;
    // The request each serving process is inside (self-wait: a call that
    // process starts names it to the engine).
    protected int         serving_h[process];
    protected int         serving_rid[process];
    // engine.progress() when this loop last looked, and the calls asleep
    // on `notified` (it is asked only while one is: a crossing per yield)
    protected longint     progress_seen = -1;
    protected int         n_waiting;

    function new(zsp_rt_bc_base eng, zsp_rt_bc_platform_if platform);
      engine = eng;
      plat   = platform;
    endfunction

    // A host event (interrupt, channel put) for calls waiting on EVENT.
    function void notify(int kind, int id, longint unsigned value);
      void'(engine.notify(kind, id, value));
      ->notified;
    endfunction

    // Called when call `h` on `entry` is DONE, before it ends: a bench
    // that extends zsp_rt_bc_ctxt reads the action's solved fields here
    // with `engine.obj_field(h, name)`.
    virtual function void done(int h, string entry);
    endfunction

    // Run `entry` on the instance at `comp_path` to its end. `ok` = 0 on
    // an engine error.
    task call(string entry, string comp_path, longint unsigned args[$],
              output longint unsigned ret, output bit ok);
      int h;
      engine.set_time(zsp_rt_bc_now_fs());
      plat.time_exp = engine.time_exp();
      h = engine.call_start(entry, comp_path, args);
      run_call(h, entry, ret, ok);
    endtask

    // `call` keyed by handle (engine.new_obj/sub): the traversal of action
    // object `obj` (`method` ""), or function `method` of component `obj`.
    task call_obj(int obj, string method, longint unsigned args[$],
                  output longint unsigned ret, output bit ok);
      int h;
      engine.set_time(zsp_rt_bc_now_fs());
      plat.time_exp = engine.time_exp();
      h = engine.call_obj(obj, method, args);
      run_call(h, method != "" ? method : $sformatf("handle %0d", obj), ret, ok);
    endtask

    // The run loop (design §10.3) of call `h`.
    protected task run_call(int h, string entry, output longint unsigned ret,
                            output bit ok);
      zsp_rt_bc_req req = new;
      process self_p = process::self();
      int seen;
      ret = 0;
      ok  = 0;
      if (h == 0) begin
        fail(0, entry);
        return;
      end
      // Started inside a request this process serves: the engine can tell
      // a traversal that would wait on itself (self-wait).
      if (serving_h.exists(self_p))
        void'(engine.call_inside(h, serving_h[self_p], serving_rid[self_p]));
      n_done[h] = 0;
      n_out[h]  = 0;
      forever begin
        zsp_rt_bc_status_e st;
        engine.set_time(zsp_rt_bc_now_fs());
        st = engine.run(h);
        drain_messages();
        moved();
        if (zsp_rt_bc_dbgsig_on()) begin
          zsp_rt_bc_dbg_op   = engine.cur_op(h);
          zsp_rt_bc_dbg_file = engine.cur_file(h);
          zsp_rt_bc_dbg_line = engine.cur_line(h);
          zsp_rt_bc_dbg_seq++;
        end
        case (st)
          ZSP_RT_BC_DONE: begin
            ret = engine.result(h);
            ok  = 1;
            done(h, entry);
            end_call(h);
            // A call can wake another one waiting on EVENT (a channel put,
            // e.g. an irq hook calling `notify_irq`): let it run again.
            ->notified;
            return;
          end
          ZSP_RT_BC_REQS: begin
            for (int rid = engine.take(h); rid >= 0; rid = engine.take(h)) begin
              automatic zsp_rt_bc_req r = new;
              automatic int ticket = -1;
              engine.request(h, rid, r);
              if (r.api == 0 && r.executor >= 0) begin
                if (!exec_ticket.exists(r.executor)) begin
                  exec_ticket[r.executor] = 0;
                  exec_turn[r.executor]   = 0;
                end
                ticket = exec_ticket[r.executor]++;
              end
              n_out[h]++;
              fork
                serve(h, r, ticket);
              join_none
            end
          end
          ZSP_RT_BC_WAIT: begin
            engine.request(h, -1, req);
            seen = n_done[h];
            case (req.on)
              ZSP_RT_BC_WAIT_TIME:
                if (n_out[h] == 0) plat.wait_time(req.delay);
                // The first of the delay and a completion; the loser is
                // stopped. The outer fork keeps `disable fork` off the
                // serve processes forked above.
                else fork begin
                  fork
                    plat.wait_time(req.delay);
                    while (n_done[h] == seen) @(completed);
                  join_any
                  disable fork;
                end join
              ZSP_RT_BC_WAIT_REQS: while (n_done[h] == seen) @(completed);
              // EVENT, and TURN: a notify, another call's end or progress
              default: begin
                n_waiting++;
                @(notified);
                n_waiting--;
              end
            endcase
          end
          ZSP_RT_BC_STOP: begin
            engine.request(h, -1, req);
            plat.dbg_wait(req);
          end
          default: begin
            fail(h, entry);
            end_call(h);
            ->notified;     // as at DONE: a queued traversal may start
            return;
          end
        endcase
      end
    endtask

    // A call made progress since the loop last looked: wake the calls
    // waiting on EVENT or TURN, so they look again. With none waiting the
    // engine is not asked: a call that starts waiting has run after any
    // progress it could have missed.
    protected function void moved();
      longint p;
      if (n_waiting == 0) return;
      p = engine.progress();
      if (p < 0 || p == progress_seen) return;
      progress_seen = p;
      ->notified;
    endfunction

    // Serve request `r` of call `h` and complete it (unless the call has
    // ended: an error drops its outstanding requests).
    protected task serve(int h, zsp_rt_bc_req r, int ticket);
      longint unsigned v = 0;
      process self_p = process::self();
      zsp_rt_bc_platform_if hd;
      // In issue order into the handler; the next access may enter as soon
      // as this one has (it runs once this process blocks in the handler).
      if (ticket >= 0) begin
        while (exec_turn[r.executor] != ticket) @(exec_next);
        exec_turn[r.executor]++;
        ->exec_next;
      end
      hd = engine.handler(r.api, r.executor);
      serving_h[self_p]   = h;
      serving_rid[self_p] = r.req_id;
      if (hd == null) begin
        plat.stim_id = r.stim_id;
        if (r.api == 0) access(r, v);
        else plat.call_import(r, v);
      end else begin
        hd.stim_id = r.stim_id;
        if (r.api == 0) hd.access(r, v);
        else hd.call_import(r, v);
      end
      serving_h.delete(self_p);
      serving_rid.delete(self_p);
      if (!n_done.exists(h)) return;
      engine.set_time(zsp_rt_bc_now_fs());
      foreach (r.rets[i]) void'(engine.set_ret(h, r.req_id, i, r.rets[i]));
      foreach (r.ret_s[i]) void'(engine.string_ret(h, r.req_id, i, r.ret_s[i]));
      void'(engine.complete(h, r.req_id, v));
      n_out[h]--;
      n_done[h]++;
      ->completed;
    endtask

    protected function void end_call(int h);
      engine.call_end(h);
      n_done.delete(h);
      n_out.delete(h);
    endfunction

    // A memory access no handler is linked for: the platform's.
    protected task access(zsp_rt_bc_req req, output longint unsigned rdata);
      plat.access(req, rdata);
    endtask

    protected function void drain_messages();
      int sev;
      for (string m = engine.message_next(sev); m != ""; m = engine.message_next(sev))
        plat.message(sev, m);
    endfunction

    protected function void fail(int h, string entry);
      last_error = engine.error(h);
      n_errors++;
      $display("PSSVM-ERROR: %s: %s", entry, last_error);
      engine.call_failed(h, entry, last_error);
    endfunction
  endclass
