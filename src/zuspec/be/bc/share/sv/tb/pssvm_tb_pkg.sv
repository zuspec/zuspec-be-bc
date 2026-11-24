// pssvm_tb_pkg -- the parts of the generic bench (pssvm_tb.sv) over pssvm_pkg.
//
//   * pssvm_tb_mem   -- sparse byte memory, little endian, unwritten bytes 0
//   * pssvm_tb_regs  -- a register file (32-bit registers with reset values)
//   * pssvm_tb_out   -- the output channel: `PSSVM-OUT:` lines (and a file)
//   * pssvm_tb_irq   -- an interrupt source (notify kind 1)
//   * pssvm_tb_obs   -- an observation-feed stub (notify kind 2)
//   * pssvm_tb_plat  -- the platform: registers, then memory; imports
//
// Plusargs: +pssvm_mem=<file> (readmemh-style bytes), +pssvm_reg=<name>@<addr>[=<reset>],..,
// +pssvm_irq=<id>@<time>,.., +pssvm_lat=<n>, +pssvm_out=<file>,
// +pssvm_str=<import>:<arg>,.. (string arguments, logged as text),
// +pssvm_win=<addr>:<size>,.. (the image's memory regions; an access outside
// every window, and not a register's, is a bench failure),
// +pssvm_fuzz=<seed>:<lo>:<hi> (timing fuzz: each access takes, and each
// import waits before it returns, a delay drawn from [lo, hi] by a generator
// seeded with <seed>; replaces +pssvm_lat). Numbers in hex.

package pssvm_tb_pkg;
  import pssvm_pkg::*;

  localparam int PSSVM_TB_N_IRQ  = 1;
  localparam int PSSVM_TB_N_CHAN = 2;

  // A hex number of up to 64 bits (string::atohex stops at 32).
  function automatic longint unsigned pssvm_tb_hex(string s);
    longint unsigned v = 0;
    void'($sscanf(s, "%h", v));
    return v;
  endfunction

  // Split "a,b,c" into its parts.
  function automatic void pssvm_tb_split(string s, ref string parts[$]);
    int st = 0;
    for (int i = 0; i <= s.len(); i++) begin
      if (i == s.len() || s[i] == ",") begin
        if (i > st) parts.push_back(s.substr(st, i - 1));
        st = i + 1;
      end
    end
  endfunction

  class pssvm_tb_mem;
    bit [7:0] mem[longint unsigned];

    function bit [7:0] get(longint unsigned addr);
      return mem.exists(addr) ? mem[addr] : 8'h0;
    endfunction

    function void set(longint unsigned addr, bit [7:0] data);
      mem[addr] = data;
    endfunction

    function longint unsigned read(longint unsigned addr, int size);
      longint unsigned v = 0;
      for (int i = size - 1; i >= 0; i--) v = (v << 8) | 64'(get(addr + longint'(i)));
      return v;
    endfunction

    function void write(longint unsigned addr, int size, longint unsigned data);
      for (int i = 0; i < size; i++) set(addr + longint'(i), data[8*i +: 8]);
    endfunction

    function void clear();
      mem.delete();
    endfunction

    // readmemh-style text: `@<hex addr>` moves the address; each other
    // word is one byte in hex; `//` starts a comment to the end of line.
    // Returns the number of bytes loaded, or -1 when the file cannot open.
    function int load(string path);
      int fd = $fopen(path, "r");
      string line, tok;
      longint unsigned addr = 0;
      int n = 0;
      if (fd == 0) return -1;
      while ($fgets(line, fd) > 0) begin
        int i = 0;
        while (i < line.len()) begin
          int st;
          if (line[i] == " " || line[i] == "\t" || line[i] == "\n" || line[i] == "\r") begin
            i++;
            continue;
          end
          if (line[i] == "/" && i + 1 < line.len() && line[i+1] == "/") break;
          st = i;
          while (i < line.len() && !(line[i] == " " || line[i] == "\t" ||
                                     line[i] == "\n" || line[i] == "\r")) i++;
          tok = line.substr(st, i - 1);
          if (tok[0] == "@") addr = pssvm_tb_hex(tok.substr(1, tok.len() - 1));
          else begin
            set(addr, 8'(tok.atohex()));
            addr++;
            n++;
          end
        end
      end
      $fclose(fd);
      return n;
    endfunction
  endclass

  class pssvm_tb_regs;
    string              names[$];
    longint unsigned    addrs[$];
    bit [31:0]          resets[$];
    bit [31:0]          wmasks[$];
    bit [31:0]          vals[$];

    function void add(string name, longint unsigned addr, bit [31:0] rst = 0,
                      bit [31:0] wmask = 32'hffff_ffff);
      names.push_back(name);
      addrs.push_back(addr & ~64'h3);
      resets.push_back(rst);
      wmasks.push_back(wmask);
      vals.push_back(rst);
    endfunction

    // `+pssvm_reg=<name>@<hex addr>[=<hex reset>],..`
    function void add_plusargs();
      string arg, parts[$];
      if (!$value$plusargs("pssvm_reg=%s", arg)) return;
      pssvm_tb_split(arg, parts);
      foreach (parts[k]) begin
        string p = parts[k], name, a, r;
        int at = -1, eq = -1;
        for (int i = 0; i < p.len(); i++) begin
          if (p[i] == "@" && at < 0) at = i;
          else if (p[i] == "=" && eq < 0) eq = i;
        end
        if (at <= 0) begin
          $display("PSSVM-ERROR: +pssvm_reg: bad '%s' (want name@addr[=reset])", p);
          continue;
        end
        name = p.substr(0, at - 1);
        a = (eq > at) ? p.substr(at + 1, eq - 1) : p.substr(at + 1, p.len() - 1);
        r = (eq > at) ? p.substr(eq + 1, p.len() - 1) : "0";
        add(name, pssvm_tb_hex(a), 32'(pssvm_tb_hex(r)));
      end
    endfunction

    function int find(longint unsigned addr);
      foreach (addrs[i]) if (addrs[i] == (addr & ~64'h3)) return i;
      return -1;
    endfunction

    // An access is the register's if it falls inside one 32-bit register.
    function bit claims(longint unsigned addr, int size);
      return find(addr) >= 0 && size <= 4 && ((addr & 64'h3) + longint'(size)) <= 4;
    endfunction

    function longint unsigned read(longint unsigned addr, int size);
      int i = find(addr);
      bit [31:0] v = vals[i] >> (8 * (addr & 64'h3));
      return size >= 4 ? 64'(v) : 64'(v) & ((64'h1 << (8 * size)) - 1);
    endfunction

    function void write(longint unsigned addr, int size, longint unsigned data);
      int i = find(addr);
      int sh = 8 * int'(addr & 64'h3);
      bit [31:0] lanes = (size >= 4) ? 32'hffff_ffff : ((32'h1 << (8 * size)) - 1);
      bit [31:0] m = (lanes << sh) & wmasks[i];
      vals[i] = (vals[i] & ~m) | ((32'(data) << sh) & m);
    endfunction

    function void reset();
      foreach (vals[i]) vals[i] = resets[i];
    endfunction
  endclass

  class pssvm_tb_out;
    int fd;

    function new();
      string path;
      if ($value$plusargs("pssvm_out=%s", path)) begin
        fd = $fopen(path, "w");
        if (fd == 0) $display("PSSVM-ERROR: +pssvm_out: cannot open %s", path);
      end
    endfunction

    function void put(string kind, string msg);
      $display("PSSVM-OUT: %s %s", kind, msg);
      if (fd != 0) begin
        $fdisplay(fd, "%0t %s %s", $time, kind, msg);
        $fflush(fd);
      end
    endfunction

    function void close();
      if (fd != 0) $fclose(fd);
      fd = 0;
    endfunction
  endclass

  class pssvm_tb_irq;
    pssvm_ctxt   ctxt;
    pssvm_tb_out out;
    int          n_raised;

    function new(pssvm_ctxt c, pssvm_tb_out o);
      ctxt = c;
      out  = o;
    endfunction

    function void raise(int id);
      out.put("irq", $sformatf("%0d @%0t", id, $time));
      n_raised++;
      ctxt.notify(PSSVM_TB_N_IRQ, id, 0);
    endfunction

    task raise_at(int id, longint unsigned at);
      fork
        begin
          #(at);
          raise(id);
        end
      join_none
    endtask

    // `+pssvm_irq=<hex id>@<hex time>,..`, counted from now.
    task start_plusargs();
      string arg, parts[$];
      if (!$value$plusargs("pssvm_irq=%s", arg)) return;
      pssvm_tb_split(arg, parts);
      foreach (parts[k]) begin
        string p = parts[k];
        int at = -1;
        for (int i = 0; i < p.len(); i++) if (p[i] == "@") at = i;
        if (at <= 0) begin
          $display("PSSVM-ERROR: +pssvm_irq: bad '%s' (want id@time)", p);
          continue;
        end
        raise_at(int'(pssvm_tb_hex(p.substr(0, at - 1))),
                 pssvm_tb_hex(p.substr(at + 1, p.len() - 1)));
      end
    endtask
  endclass

  // A stub: values a monitor would observe, put on host-fed channels.
  class pssvm_tb_obs;
    pssvm_ctxt   ctxt;
    pssvm_tb_out out;

    function new(pssvm_ctxt c, pssvm_tb_out o);
      ctxt = c;
      out  = o;
    endfunction

    function void put(int chan, longint unsigned value);
      out.put("obs", $sformatf("%0d 'h%0h @%0t", chan, value, $time));
      ctxt.notify(PSSVM_TB_N_CHAN, chan, value);
    endfunction
  endclass

  class pssvm_tb_plat extends pssvm_platform_if;
    pssvm_tb_mem  mem;
    pssvm_tb_regs regs;
    pssvm_tb_out  out;
    string        mem_path;
    longint unsigned lat = 10;
    int           n_access;
    pssvm_engine  engine;       // set by the bench; looks up string arguments
    bit           str_arg[string];  // "<import>:<arg>" from +pssvm_str
    // +pssvm_win: memory windows [lo, lo + size); none = any address
    longint unsigned win_lo[$], win_size[$];
    int           n_outside;
    // +pssvm_fuzz: delays from [fuzz_lo, fuzz_hi], xorshift64 state
    bit           fuzz;
    longint unsigned fuzz_lo, fuzz_hi, fuzz_state;

    function new(pssvm_tb_out o);
      string s;
      out  = o;
      mem  = new;
      regs = new;
      regs.add_plusargs();
      if ($value$plusargs("pssvm_lat=%s", s)) lat = pssvm_tb_hex(s);
      if ($value$plusargs("pssvm_str=%s", s)) begin
        string parts[$];
        pssvm_tb_split(s, parts);
        foreach (parts[i]) str_arg[parts[i]] = 1;
      end
      if ($value$plusargs("pssvm_win=%s", s)) add_windows(s);
      if ($value$plusargs("pssvm_fuzz=%s", s)) set_fuzz(s);
      if ($value$plusargs("pssvm_mem=%s", mem_path)) load_mem();
    endfunction

    // `<hex seed>:<hex lo>:<hex hi>`
    function void set_fuzz(string arg);
      int c[$];
      for (int i = 0; i < arg.len(); i++) if (arg[i] == ":") c.push_back(i);
      if (c.size() != 2 || c[0] == 0 || c[1] == c[0] + 1 || c[1] == arg.len() - 1) begin
        $display("PSSVM-ERROR: +pssvm_fuzz: bad '%s' (want seed:lo:hi)", arg);
        return;
      end
      fuzz_state = pssvm_tb_hex(arg.substr(0, c[0] - 1));
      fuzz_lo    = pssvm_tb_hex(arg.substr(c[0] + 1, c[1] - 1));
      fuzz_hi    = pssvm_tb_hex(arg.substr(c[1] + 1, arg.len() - 1));
      if (fuzz_hi < fuzz_lo) fuzz_hi = fuzz_lo;
      // splitmix64 of the seed: seed 0 and nearby seeds start apart
      fuzz_state += 64'h9e3779b97f4a7c15;
      fuzz_state = (fuzz_state ^ (fuzz_state >> 30)) * 64'hbf58476d1ce4e5b9;
      fuzz_state = (fuzz_state ^ (fuzz_state >> 27)) * 64'h94d049bb133111eb;
      fuzz_state ^= fuzz_state >> 31;
      if (fuzz_state == 0) fuzz_state = 1;
      fuzz = 1;
      $display("PSSVM-TB: fuzz seed='h%s delay 'h%0h..'h%0h", arg.substr(0, c[0] - 1),
               fuzz_lo, fuzz_hi);
    endfunction

    // The next fuzz delay (xorshift64), the same on every simulator.
    function longint unsigned fuzz_next();
      longint unsigned span = fuzz_hi - fuzz_lo + 1;
      fuzz_state ^= fuzz_state << 13;
      fuzz_state ^= fuzz_state >> 7;
      fuzz_state ^= fuzz_state << 17;
      return (span == 0) ? fuzz_lo + fuzz_state : fuzz_lo + fuzz_state % span;
    endfunction

    // One bus access's latency: fixed, or a fuzz delay.
    function longint unsigned access_lat();
      return fuzz ? fuzz_next() : lat;
    endfunction

    // `<hex addr>:<hex size>,..`
    function void add_windows(string arg);
      string parts[$];
      pssvm_tb_split(arg, parts);
      foreach (parts[k]) begin
        string p = parts[k];
        int c = -1;
        for (int i = 0; i < p.len(); i++) if (p[i] == ":" && c < 0) c = i;
        if (c <= 0 || c == p.len() - 1) begin
          $display("PSSVM-ERROR: +pssvm_win: bad '%s' (want addr:size)", p);
          continue;
        end
        win_lo.push_back(pssvm_tb_hex(p.substr(0, c - 1)));
        win_size.push_back(pssvm_tb_hex(p.substr(c + 1, p.len() - 1)));
      end
    endfunction

    // An access is memory's if no windows are set or one holds all of it.
    function bit in_window(longint unsigned addr, int size);
      if (win_lo.size() == 0) return 1;
      foreach (win_lo[i])
        if (addr >= win_lo[i] && addr - win_lo[i] + longint'(size) <= win_size[i])
          return 1;
      return 0;
    endfunction

    // An access outside the image's regions fails the test running (it is
    // not done: a read returns 0).
    protected function bit outside(longint unsigned addr, int size, string dir);
      string keys[string];
      string msg;
      if (regs.claims(addr, size) || in_window(addr, size)) return 0;
      n_outside++;
      msg = $sformatf("memory %s of %0d bytes at 'h%0h outside the image's regions",
                      dir, size, addr);
      $display("PSSVM-ERROR: %s", msg);
      keys["addr"] = $sformatf("'h%0h", addr);
      void'(pssvm_bench_fail("mem.outside", keys, msg));
      return 1;
    endfunction

    function void load_mem();
      if (mem_path != "" && mem.load(mem_path) < 0)
        $display("PSSVM-ERROR: +pssvm_mem: cannot open %s", mem_path);
    endfunction

    protected task rd(longint unsigned addr, int size, output longint unsigned data);
      #(access_lat());
      n_access++;
      if (outside(addr, size, "read")) data = 0;
      else data = regs.claims(addr, size) ? regs.read(addr, size) : mem.read(addr, size);
    endtask

    protected task wr(longint unsigned addr, int size, longint unsigned data);
      #(access_lat());
      n_access++;
      if (outside(addr, size, "write")) return;
      if (regs.claims(addr, size)) regs.write(addr, size, data);
      else mem.write(addr, size, data);
    endtask

    virtual task read8(longint unsigned addr, output bit [7:0] data);
      longint unsigned v;
      rd(addr, 1, v);
      data = v[7:0];
    endtask
    virtual task read16(longint unsigned addr, output bit [15:0] data);
      longint unsigned v;
      rd(addr, 2, v);
      data = v[15:0];
    endtask
    virtual task read32(longint unsigned addr, output bit [31:0] data);
      longint unsigned v;
      rd(addr, 4, v);
      data = v[31:0];
    endtask
    virtual task read64(longint unsigned addr, output bit [63:0] data);
      longint unsigned v;
      rd(addr, 8, v);
      data = v;
    endtask
    virtual task write8 (longint unsigned addr, bit [7:0]  data); wr(addr, 1, 64'(data)); endtask
    virtual task write16(longint unsigned addr, bit [15:0] data); wr(addr, 2, 64'(data)); endtask
    virtual task write32(longint unsigned addr, bit [31:0] data); wr(addr, 4, 64'(data)); endtask
    virtual task write64(longint unsigned addr, bit [63:0] data); wr(addr, 8, data);      endtask

    // pssunit_pkg's back door: mem_poke(addr, data, size) and mem_peek(addr,
    // size) set and read `size` bytes in zero time, with no bus access
    // (not counted, no latency); outside the image's regions they fail the
    // test like an access would. 1: `req` was one of them.
    protected function bit backdoor(pssvm_req req, output longint unsigned ret);
      string n = req.name;
      int    size;
      ret = 0;
      for (int i = n.len() - 1; i > 0; i--)
        if (n[i] == ":" && n[i - 1] == ":") begin n = n.substr(i + 1, n.len() - 1); break; end
      if (n == "mem_poke" && req.args.size() == 3) begin
        size = int'(req.args[2]);
        if (!outside(req.args[0], size, "back-door write"))
          if (regs.claims(req.args[0], size)) regs.write(req.args[0], size, req.args[1]);
          else mem.write(req.args[0], size, req.args[1]);
        return 1;
      end
      if (n == "mem_peek" && req.args.size() == 2) begin
        size = int'(req.args[1]);
        if (!outside(req.args[0], size, "back-door read"))
          ret = regs.claims(req.args[0], size) ? regs.read(req.args[0], size)
                                               : mem.read(req.args[0], size);
        return 1;
      end
      return 0;
    endfunction

    // Message imports go to the output channel; the back door (above) is
    // served; any other import is logged
    // by name and returns 0 (a struct result: field i is i + 1, a stub
    // value that shows the field order). A string argument (+pssvm_str)
    // arrives as a model-string index; it is logged as its text. Under
    // +pssvm_fuzz the import waits a fuzz delay before it returns (the
    // back door stays zero-time).
    virtual task call_import(pssvm_req req, output longint unsigned ret);
      string a = "";
      if (backdoor(req, ret)) return;
      if (fuzz) #(fuzz_next());
      foreach (req.args[i])
        if (engine != null && str_arg.exists($sformatf("%s:%0d", req.name, i)))
          a = {a, " \"", engine.string_at(int'(req.args[i])), "\""};
        else a = {a, $sformatf(" 'h%0h", req.args[i])};
      ret = 0;
      for (int i = 0; i < req.nret; i++) req.rets.push_back(64'(i + 1));
      if (req.name == "print" || req.name == "message" || req.name == "pss_print")
        out.put("msg", {req.name, a});
      else
        out.put("import", {req.name, a});
    endtask

    // Each line of a message()/error()/fatal() text is one `msg`, `error`
    // or `fatal` line on the output channel.
    virtual function void message(int sev, string text);
      string kind = (sev == 0) ? "msg" : (sev == 1) ? "error" : "fatal";
      int    b = 0;
      for (int i = 0; i <= text.len(); i++)
        if (i == text.len() || text[i] == "\n") begin
          out.put(kind, (i > b) ? text.substr(b, i - 1) : "");
          b = i + 1;
        end
    endfunction

    virtual task reset();
      regs.reset();
      mem.clear();
      load_mem();
    endtask
  endclass

endpackage
