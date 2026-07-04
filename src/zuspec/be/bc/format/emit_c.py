"""
emit_c.py -- generate ``zbc_format.h`` from the spec (P0-5).

The header is the C view of the ``.zbc`` format. It is **checked in** to
``zuspec-rt-core`` as a generated artifact and guarded by a regen no-diff test
(T0-D): running this emitter must reproduce the committed bytes exactly. The
output is therefore deterministic -- no timestamps, no host-specific content.

Run ``python -m zuspec.be.bc.format.emit_c`` to (re)generate it in place.
"""

import os
from typing import List, Optional

from .spec import (
    RECORDS,
    ENUMS,
    PRIMS,
    MAGIC,
    VERSION_MAJOR,
    VERSION_MINOR,
)

BANNER = (
    "/*\n"
    " * zbc_format.h -- the .zbc container format, C view.\n"
    " *\n"
    " * GENERATED from zuspec.be.bc.format.spec -- DO NOT EDIT BY HAND.\n"
    " * Regenerate with:  python -m zuspec.be.bc.format.emit_c\n"
    " * (A regen no-diff test guards this file against drift.)\n"
    " */\n"
)

GUARD = "ZUSPEC_ZBC_FORMAT_H"


def _c_field_decl(field) -> str:
    ctype = PRIMS[field.type][0]
    if field.array:
        return f"    {ctype:<10} {field.name}[{field.count}];"
    return f"    {ctype:<10} {field.name};"


def render_header() -> str:
    out: List[str] = []
    out.append(BANNER)
    out.append(f"#ifndef {GUARD}")
    out.append(f"#define {GUARD}")
    out.append("")
    out.append("#include <stdint.h>")
    out.append("")

    # Magic + version.
    magic = ", ".join(f"0x{b:02x}" for b in MAGIC)
    out.append("/* File identity */")
    out.append(f"#define ZBC_MAGIC0 0x{MAGIC[0]:02x}")
    out.append(f"#define ZBC_MAGIC1 0x{MAGIC[1]:02x}")
    out.append(f"#define ZBC_MAGIC2 0x{MAGIC[2]:02x}")
    out.append(f"#define ZBC_MAGIC3 0x{MAGIC[3]:02x}")
    out.append(f"#define ZBC_VERSION_MAJOR {VERSION_MAJOR}")
    out.append(f"#define ZBC_VERSION_MINOR {VERSION_MINOR}")
    out.append("")

    # Enums as #defines (stable, C-portable, no enum-width ambiguity).
    for e in ENUMS:
        out.append(f"/* {e.name}: {e.doc} */")
        width = max((len(m.name) for m in e.members), default=0)
        for m in e.members:
            tail = f"  /* {m.comment} */" if m.comment else ""
            out.append(f"#define {m.name:<{width}} 0x{m.value:04x}{tail}")
        out.append("")

    # Structs.
    for r in RECORDS:
        out.append(f"/* {r.name}: {r.doc} */")
        out.append("typedef struct {")
        for f in r.fields:
            tail = f"  /* {f.comment} */" if f.comment else ""
            out.append(_c_field_decl(f) + tail)
        out.append(f"}} {r.name};")
        out.append("")

    # Compile-time size checks so the C side catches layout drift immediately.
    out.append("/* Layout guards: sizes must match the spec (natural alignment). */")
    for r in RECORDS:
        out.append(
            f"_Static_assert(sizeof({r.name}) == {r.size()}, "
            f'"{r.name} size mismatch");'
        )
    out.append("")

    out.append(f"#endif /* {GUARD} */")
    out.append("")
    return "\n".join(out)


def default_header_path() -> str:
    """The checked-in location under rt-core's share/include."""
    try:
        import zuspec.rt.core as rt

        return os.path.join(rt.include_dir(), "zbc_format.h")
    except Exception:
        # Fall back to a path relative to this file's repo layout.
        here = os.path.dirname(__file__)
        return os.path.abspath(
            os.path.join(
                here,
                "..", "..", "..", "..", "..", "..",
                "zuspec-rt-core", "src", "zuspec", "rt", "core",
                "share", "include", "zbc_format.h",
            )
        )


def write_header(path: Optional[str] = None) -> str:
    """Write the generated header; return the path written."""
    path = path or default_header_path()
    text = render_header()
    with open(path, "w") as fp:
        fp.write(text)
    return path


if __name__ == "__main__":
    p = write_header()
    print(f"wrote {p}")
