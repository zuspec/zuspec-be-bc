// pssvm_uvm_pkg -- forwards UVM reports to the PSS engine (bench failure
// attribution). Compile after uvm_pkg and pssvm_pkg; a bench without UVM
// leaves this file out.
//
//   pssvm_report_catcher::install();   // e.g. in the test's build_phase
//
// Every uvm_error and uvm_fatal (uvm_warning too with `warnings` set) goes
// to the default engine through `pssvm_report`, which records it against
// the action in flight and fails the test; the report itself passes
// through unchanged.
package pssvm_uvm_pkg;
  import uvm_pkg::*;
  import pssvm_pkg::*;

  class pssvm_report_catcher extends uvm_report_catcher;
    `uvm_object_utils(pssvm_report_catcher)

    // Forward uvm_warning reports too (recorded, not failing).
    bit warnings;
    // Forward only reports whose id matches this glob ("" = every id).
    string id_glob;

    static pssvm_report_catcher inst;

    function new(string name = "pssvm_report_catcher");
      super.new(name);
    endfunction

    // Add the catcher to every component (once).
    static function pssvm_report_catcher install();
      if (inst == null) begin
        inst = new();
        uvm_report_cb::add(null, inst);
      end
      return inst;
    endfunction

    virtual function action_e catch();
      string sev, keys[string];
      case (get_severity())
        UVM_FATAL:   sev = "fatal";
        UVM_ERROR:   sev = "error";
        UVM_WARNING: sev = warnings ? "warning" : "";
        default:     sev = "";
      endcase
      if (sev == "" || (id_glob != "" && !uvm_is_match(id_glob, get_id())))
        return THROW;
      keys["component"] = get_client().get_full_name();
      keys["file"] = $sformatf("%s:%0d", get_fname(), get_line());
      void'(pssvm_report("uvm", sev, get_id(), get_message(), keys));
      return THROW;
    endfunction
  endclass
endpackage
