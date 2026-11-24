# Functional coverage: `COV_SAMPLE`, `COVG` and the binning service

How a bc image samples PSS covergroups and how a host turns the samples into
bin counts. Both engines (the Python VM in `interp/`, the C engine in
zuspec-rt-eng) sample the same way; binning is host-side and shared
(`cov/covergroup.py`).

## Sample point

An action's inline covergroups are sampled once per traversal, when its
values are final: after its `post_solve` exec block, else right after its
solve, before its body or activity (`lower/coverage.py` `with_cov_sample`).
An inline covergroup is one type with one instance, so the "sample function"
is a sequence in the action's coroutine: it evaluates each coverpoint (and
each `iff`) into registers, stages them with `ARG`, then executes one
`COV_SAMPLE` per covergroup. An instance of a covergroup type arrives from
the front end already expanded (`CovergroupDef.type_name` set, each parameter
replaced by the field it is bound to), so it is lowered exactly like an
inline covergroup. A struct's covergroups are sampled with the action's,
once for each `rand` struct field (nested too; not action handles): each is
copied with `self` re-rooted at the field (`self.v` -> `self.s.v`), named
`<action>::<field path>.<cg>` and typed by its covergroup type, or
`<struct>::<cg>` for an inline one (`_struct_covergroups`). PSS has no
explicit `sample()`.

## `COV_SAMPLE`

`COV_SAMPLE covg, n` (0x4F) takes `n` staged arguments, as `CALL` does (an
instruction has at most four inline arguments). `covg` is the index of the
covergroup's document in the `COVG` array. The values are, in `COVG`-document order:

1. per coverpoint, its value, then its `iff` flag if the document says `iff`;
2. then one `iff` flag per cross that has one.

The op never suspends. The Python VM appends `(covg, values)` to
`vm.cov_samples` (`RunResult.cov_samples`); both engines raise a `cov` trace
event `{covg, values}` ([trace-schema.md](trace-schema.md)), so a trace
recorded in a simulator carries the run's coverage, sample for sample the
same as a standalone run with the same seed.

## `COVG` document

One JSON document per covergroup instance, in the `COVG` section's array
([zbc-format.md](zbc-format.md) "Covergroups"):

```json
{"name": "<action>::<instance>", "instance": "<instance>",
 "coverpoints": [{"name": "cp", "width": 8, "signed": false, "iff": false,
                  "items": [["READ", 0], ["WRITE", 1]],
                  "bins": [{"name": "lo", "kind": "bins",
                            "values": [1, [2, 15], [16, null]],
                            "array": true, "size": 4,
                            "with": {"op": "Lt", "args": [{"ref": "item"}, {"c": 9}]}}]}],
 "crosses": [{"name": "x", "coverpoints": ["cp_a", "cp_b"], "iff": false,
              "bins": [{"name": "s", "kind": "bins", "select": {"op": "...", "args": []}}]}]}
```

- `type` (an instance of a covergroup type only): the type's qualified name.
  `name` stays the instance's own; the binned report carries `type` through,
  so a reader can sum the instances of one type.
- `options` (covergroup, coverpoint or cross; only when one differs from
  its default): the LRM 19.6 options set, folded to constants --
  `at_least`, `weight`, `goal`, `comment`, plus `auto_bin_max` (covergroup,
  coverpoint) and `per_instance` (covergroup). A coverpoint or cross
  already holds the `at_least` (and `auto_bin_max`) it inherits from its
  covergroup; an instance's `with` options are merged over its type's.
- `kind` is `bins`, `ignore`, `illegal` or `default`.
- `values`: a value, or `[lo, hi]` with `null` for an open end.
- `array` without `size` is `b[]`, with a size `b[N]`; `size` and `with`
  are present only when the bin has them.
- `file`, `line` (an `illegal` bin, coverpoint or cross, when its source is
  known): where a hit on it is reported.
- `items` is present for an enum coverpoint: its enumerators in declaration
  order.
- `with` and `select` are small expression trees: `{"op": name, "args": [...]}`
  (the IR operator's name), `{"ref": name}` (a coverpoint's value; `item` in
  `with`), `{"c": value}`.

A bin value or expression the document cannot describe is a lowering
error, never dropped.

## Binning

`cov/covergroup.py` `bin_samples(covgs, samples)` counts samples against the
documents (LRM 19.3; the module docstring has the rules):

- value and range bins hold their values; `with` keeps the values it holds for;
- `b[]` is one bin per value, `b[N]` spreads the values over N bins;
- `default` holds the values no other bin holds; `ignore` and `illegal` bins
  take their values out of every other bin;
- no `bins` bin: automatic bins, one per enumerator for an enum, else one per
  value up to `auto_bin_max` (the coverpoint's option, else `AUTO_BIN_MAX`,
  64), else that many equal ranges;
- a cross has one automatic bin per combination of its coverpoints' `bins`
  bins and its select bins; a sample in a select bin is not counted in the
  automatic bins.

An illegal hit is counted, and `illegal_hits(covgs, report)` lists each hit
illegal bin declaration (`covergroup`, `coverpoint` or `cross`, `bin`,
`hits`, `file`, `line`, `message`). `run()` turns a run that would be `ok`
into `runtime_error` with an `IllegalBinError` located at the first such
bin; a simulation host binning the trace does the same check. Only bins of kind
`bins` count toward coverage (`counted_bins`).

The binned report carries each `options` through unchanged; options never
change a hit count. `grade(report)` scores it: a bin is covered when its
hits reach `at_least`; a coverpoint's or cross's score is the percentage of
its counted bins covered; a covergroup's is the `weight`-weighted mean of
its coverpoints' and crosses' scores, and `met` says whether it reaches
`goal`.

`samples_from_trace(records)` reads the `cov` records of a trace, so a
simulator's trace is binned the same way. `merge` sums two reports;
`cov/coverage.py` `Coverage.add_covergroups` and `from_traces` put them next
to code and register coverage, and `cov/ucis.py` writes them to a `.cdb`
(a covergroup under its enclosing component type; default, ignore and
illegal bins as UCIS DEFAULTBIN, IGNOREBIN, ILLEGALBIN; each bin's
`at_least` from its options; `weight` and `goal` as `pssunit.weight` and
`pssunit.goal` attributes on the covergroup, coverpoint or cross).
