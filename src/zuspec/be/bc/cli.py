"""``zuspec-bc``: run a ``.zbc`` image from the command line.

``zuspec-bc run IMAGE [--entry E] [--seed N] [--arg V]... [--verbosity V]
[--platform SPEC]``
runs entry ``E`` of ``IMAGE`` through :func:`zuspec.be.bc.run.run`, solving
with dv-solve. What the image prints goes to stdout, then one verdict line,
``PASS <entry> seed=N`` or ``FAIL <entry> seed=N: <verdict>: <error>``.

``--platform py:<module>[:<attr>]`` imports ``<module>`` (the current
directory first) and takes ``<attr>`` (default ``platform``): a
:class:`~zuspec.be.bc.run.Platform` (``memory``, ``imports``), or a callable
that returns one. Without it the image runs on the VM's own memory and has no
user imports.

Exit codes: 0 the run finished; 1 it did not; 2 no such image or entry;
3 the platform did not load.
"""

import argparse
import importlib
import os
import random
import sys
from typing import Optional, Sequence

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_PLATFORM = 3


class PlatformLoadError(Exception):
    pass


def load_platform(spec: str):
    """The platform ``py:<module>[:<attr>]`` names; raises PlatformLoadError."""
    kind, _, rest = spec.partition(":")
    mod_name, _, attr = rest.partition(":")
    if kind != "py" or not mod_name:
        raise PlatformLoadError(
            f"platform {spec!r}: expected py:<module>[:<attr>]")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        module = importlib.import_module(mod_name)
    except Exception as e:
        raise PlatformLoadError(
            f"platform {spec!r}: module {mod_name!r} failed to import: "
            f"{type(e).__name__}: {e}") from e
    obj = getattr(module, attr or "platform", None)
    if obj is None:
        raise PlatformLoadError(
            f"platform {spec!r}: {mod_name} has no {attr or 'platform'!r}")
    if callable(obj) and not hasattr(obj, "memory"):
        obj = obj()
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="zuspec-bc", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run an entry of a .zbc image")
    r.add_argument("image", help="the .zbc file")
    r.add_argument("--entry", default=None,
                   help="an action name, <comp type>::<fn> or a coroutine "
                   "index (default: the image's own entry)")
    r.add_argument("--seed", type=int, default=None,
                   help="solver seed; default random, and always reported")
    r.add_argument("--arg", dest="args", type=lambda v: int(v, 0),
                   action="append", default=[],
                   help="a function entry's argument, in order (repeat)")
    r.add_argument("--verbosity", type=int, default=None,
                   help="the message() verbosity shown: 0 NONE, 1 LOW, "
                   "2 MEDIUM, 3 HIGH, 4 FULL (default MEDIUM)")
    r.add_argument("--platform", default=None,
                   help="py:<module>[:<attr>]: the memory and imports the "
                   "image runs against")
    return p


def cmd_run(a) -> int:
    from .interp import NativeBlobBackend
    from .run import RunOpts, load, resolve_entry, run

    try:
        model = load(a.image)
    except OSError as e:
        print(f"zuspec-bc: error: {e}", file=sys.stderr)
        return EXIT_USAGE
    entry = int(a.entry) if a.entry is not None and a.entry.isdigit() \
        else a.entry
    try:
        resolve_entry(model, entry)
    except (KeyError, IndexError, ValueError) as e:
        print(f"zuspec-bc: error: {e}", file=sys.stderr)
        return EXIT_USAGE
    platform = None
    if a.platform:
        try:
            platform = load_platform(a.platform)
        except PlatformLoadError as e:
            print(f"zuspec-bc: error: {e}", file=sys.stderr)
            return EXIT_PLATFORM
    seed = a.seed if a.seed is not None else random.randrange(1 << 31)
    name = a.entry if a.entry is not None else "<entry>"
    res = run(model, entry, platform, seed, RunOpts(
        args=a.args, solve_backend=NativeBlobBackend(), out=print,
        **({} if a.verbosity is None else {"verbosity": a.verbosity})))
    if not res.ok:
        print(f"FAIL {name} seed={seed}: {res.verdict}: {res.error}")
        return EXIT_FAILED
    print(f"PASS {name} seed={seed}")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return cmd_run(a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
