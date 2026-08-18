#!/usr/bin/env python3
"""THE BAKE FLOOR CHECK — fails if any full-solid-node model is
see-through.

Why this exists (roadmap "HOLES IN THE WALL", 2026-08-16): the
hand-authored 16^3 masks are carved from texture brightness, and the
carve had no floor. A plank's groove line is dark all the way across
its tile, so the whole outer layer along that row went to air; two
faces meeting at a cell edge between them carved all sixteen cells of
the edge row. A cabin wall is ONE node thick, so a straight line of
air through the model IS a hole through the wall — and once the walker
learned to descend into 16^3 cells, those holes became visible
pinholes onto the lit furnace room behind.

The rule enforced at bake time (util/claude_models.py,
`enforce_opaque`) is stronger than "no all-air line", and since
2026-08-17 it is the rule: in a model that represents a FULL SOLID NODE,
EVERY air cell lies within the model's carve depth of EXACTLY ONE face.
That makes the air six disjoint boxes, each open only to its own face,
so a 3D-DDA ray that enters one of them stops inside the node whatever
its direction — which the all-air-line test does not give you, because a
ray is not axis aligned. No-through-lines is now a corollary.

The carve depth is PER MODEL (SOLID_NODE_MODELS maps name -> depth) and
is also the width of the solid rim left on every face, which is why it
is kept as small as each model's own carve allows.

This script is the independent referee for both: it reads the SHIPPED
JSON, not the generator's memory, so a hand-edited mask, a stale file,
or a regression in the generator all read the same — RED.

Run:  python3 util/claude_model_holes.py [model_dir]
Exit: 0 all clear, 1 at least one hole, 2 could not check.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from claude_models import (SOLID_NODE_MODELS, through_lines,  # noqa: E402
                           shell_count, carve_reach, N)

DEFAULT_DIR = os.path.join(HERE, "claude_models")


def scan(model_dir):
    """[(name, path, solid_node, lines, shell)] for every model JSON
    found, sorted by name. `lines` is the through_lines() list; `shell`
    is None for a see-through-by-design model, else
    (depth, reach, stray) where `stray` is the number of air cells that
    are NOT in exactly one face shell — the invariant, counted on the
    shipped mask. 0 is the only passing value."""
    rows = []
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".json") or fn == "manifest.json":
            continue
        path = os.path.join(model_dir, fn)
        with open(path) as f:
            data = json.load(f)
        name = data.get("name", fn[:-5])
        v = np.array(data["voxels"], dtype=np.uint16)
        if v.shape != (N, N, N):
            raise SystemExit("%s: mask is %s, expected (%d,%d,%d)"
                             % (path, v.shape, N, N, N))
        shell = None
        if name in SOLID_NODE_MODELS:
            depth = SOLID_NODE_MODELS[name]
            stray = int(((v == 0) & (shell_count(depth=depth) != 1)).sum())
            shell = (depth, carve_reach(v), stray)
        rows.append((name, path, name in SOLID_NODE_MODELS,
                     through_lines(v), shell))
    return rows


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    if not os.path.isdir(model_dir):
        print("no model dir: %s" % model_dir)
        return 2
    rows = scan(model_dir)
    if not rows:
        print("no model JSON in %s — nothing was checked, which is not "
              "the same as passing" % model_dir)
        return 2

    failures = []
    strays = []
    for name, path, solid_node, lines, shell in rows:
        if solid_node:
            depth, reach, stray = shell
            verdict = "OK" if not lines and not stray else "HOLES"
            if lines:
                failures.append((name, path, lines))
            if stray:
                strays.append((name, path, depth, reach, stray))
            detail = ("%d through-line(s), rim/depth %d, carve reaches %d, "
                      "%d air cell(s) outside exactly one shell"
                      % (len(lines), depth, reach, stray))
        else:
            verdict = "skipped (see-through by design)"
            detail = "%d through-line(s)" % len(lines)
        print("%-20s %-4s %-72s %s"
              % (name, "SOLID" if solid_node else "-", detail, verdict))

    seen = set(n for n, _, _, _, _ in rows)
    missing = sorted(set(SOLID_NODE_MODELS) - seen)
    if missing:
        print("\nNOT PRESENT, so NOT CHECKED: %s" % ", ".join(missing))

    if failures:
        print("\nBAKE FLOOR RED — %d model(s) are see-through:"
              % len(failures))
        for name, path, lines in failures:
            print("  %s (%s)" % (name, path))
            for axis, a, b in lines[:12]:
                if axis == "x":
                    where = "z=%d y=%d" % (a, b)
                elif axis == "y":
                    where = "z=%d x=%d" % (a, b)
                else:
                    where = "y=%d x=%d" % (a, b)
                print("      straight through along %s at %s" % (axis, where))
            if len(lines) > 12:
                print("      ... and %d more" % (len(lines) - 12))
        print("\nRe-bake with util/claude_models.py; do NOT hand-edit a mask.")
        return 1

    if strays:
        print("\nBAKE FLOOR RED — the SHELL INVARIANT is broken on %d "
              "model(s). An air cell in two shells or in none breaks the "
              "disjoint-box proof, and a ray can corner-clip through the "
              "node even with no all-air line anywhere:" % len(strays))
        for name, path, depth, reach, stray in strays:
            print("  %s: %d air cell(s) not in exactly one shell "
                  "(declared depth %d, carve reaches %d) — %s"
                  % (name, stray, depth, reach, path))
        print("\nRe-bake with util/claude_models.py; do NOT hand-edit a mask.")
        return 1

    print("\nBAKE FLOOR GREEN — %d solid-node model(s) opaque on every axis "
          "AND air-tight to the disjoint-box rule, %d see-through-by-design "
          "model(s) skipped."
          % (sum(1 for r in rows if r[2]), sum(1 for r in rows if not r[2])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
