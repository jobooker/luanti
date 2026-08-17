#!/usr/bin/env python3
"""EXPERIMENT B of the descend cost instrument: STEPS PER RAY.

THE QUESTION, in John's words (2026-08-16): "we aren't sending any more
rays, right? ... with a proper tree it should descend very quickly, so
it shouldn't scale that badly." Both halves are right, and together they
say what to count. Ray COUNT did not change when the walker learned to
step inside a class-250 cell. What changed is STEPS PER RAY: inside a
mixed cell the walk has no hierarchy at all -- it advances 1/16 m at a
time, one texture fetch per step, until it hits something or leaves.
A ray head-on into a wall spends one or two of those. A ray grazing a
stair, or running along a groove, spends dozens.

So this script counts, per pixel, and reports the distribution:

  fine steps    1/16 m steps inside descendCell()
  coarse steps  1 m steps in march()'s outer DDA
  descended     did the primary ray step inside a class-250 cell at all

for the PRIMARY (camera) ray alone, and for the WHOLE PATH -- every
bounce and every shadow/next-event ray the pixel spent. `cornell` is the
control: it has no class-250 cell, so every fine count there must be 0,
and a nonzero one means the instrument is lying.

HOW THE NUMBERS COME BACK. The counters live in a measurement-only
shader variant (util/claude_shader_variant.py, `counters`), which
exposes them as claude_view 12-16. Each count is written as a 16-bit
little-endian byte pair in the red and green channels -- k/255 quantises
back to k through an 8-bit framebuffer, so the reader recovers the exact
integer the shader counted -- and the variant's claude_present hands
those views back with ONE nearest tap instead of the joint-bilateral
upsample, because the average of two byte codes is a third byte code
that nobody counted.

THE COUNTED SHADER IS SLOWER THAN THE SHADER IT COUNTS FOR. An add in
both inner loops is not free. No timing number is taken here; experiment
A (util/claude_cost_tax.py) owns those.

VIEW 16 IS THE HUMAN ONE: a gray brightness ladder, one rung per
doubling of the whole-path fine-step count, black for pixels that never
stepped inside a cell. Gray and not colour on purpose. Look at it.

Usage:  util/claude_cost_steps.py [--rooms cozy-ci cornell]
The SERVER must already be up; this script restarts the CLIENT once and
restores the committed shaders on the way out.
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import claude_lab as lab            # noqa: E402
import claude_cost_seat as cs       # noqa: E402

VIEWS = {12: "primary_fine", 13: "primary_coarse",
         14: "path_fine", 15: "path_coarse", 16: "ladder"}

BASE_DIALS = {
    "claude_nee": 0, "claude_bounces": 24, "claude_grid_debug": 3,
    "claude_rng": 1, "claude_grid_follow": 1, "claude_models": 1,
    "claude_descend": 1, "claude_show_hud": 0, "claude_show_chat": 0,
    "claude_input_lock": 1, "claude_stats": 1,
}

# claude_present stamps a 12 px green square in the bottom-left corner to
# prove the traced present path executed. It is not a step count.
MARKER = 16


def decode(png):
    a = np.asarray(Image.open(png).convert("RGB"))
    a = a[:-MARKER, MARKER:]          # drop the marker corner
    lo = a[:, :, 0].astype(np.int32)
    hi = a[:, :, 1].astype(np.int32)
    return lo + 256 * hi, a[:, :, 2]


def describe(counts):
    f = counts.astype(np.float64).ravel()
    return {
        "mean": round(float(f.mean()), 3),
        "median": float(np.median(f)),
        "p90": float(np.percentile(f, 90)),
        "p99": float(np.percentile(f, 99)),
        "max": int(f.max()),
        "frac_zero": round(float((f == 0).mean()), 4),
        "pixels": int(f.size),
    }


def histogram(counts, edges):
    f = counts.ravel()
    h, _ = np.histogram(f, bins=edges)
    return {"edges": [int(e) for e in edges],
            "frac": [round(float(x) / f.size, 5) for x in h]}


EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 1 << 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="+", default=["cozy-ci", "cornell"])
    # WHY THE OFF ARM MATTERS. With descent off a ray stops at the 1 m
    # wall of a class-250 cell, so it does not merely skip the fine
    # steps -- it bounces from a different place, in a different
    # direction, and takes a different number of COARSE steps
    # afterwards. Counting only the on arm would let "descent added N
    # fine steps" be read as the whole change, when the outer walk moved
    # too. One arm per dial value, same shader, same camera.
    ap.add_argument("--descend", nargs="+", type=int, default=[1, 0])
    ap.add_argument("--trace-scale", default="1.0",
                    help="1.0 keeps one traced sample per screen pixel, so "
                         "the PNG is the counter array and not a resample "
                         "of it")
    a = ap.parse_args()

    cs.require_release()
    if os.path.exists(os.path.join(HERE, "claude_shader_variants",
                                   ".installed.json")):
        cs.variant("restore")
    cs.variant("build")
    out = {"trace_scale": a.trace_scale, "rooms": {}, "shots": {}}
    try:
        cs.variant("install", "counters")
        cs.restart_client({"claude_trace_scale": a.trace_scale})
        lab.doorway(claude_view=0, **BASE_DIALS)
        print("time_speed -> %s" % cs.freeze_time())
        for room in a.rooms:
            st = cs.park(room, dials=BASE_DIALS, snapshot_tag="steps_" + room)
            for descend in a.descend:
                key = "%s/descend%d" % (room, descend)
                lab.doorway(claude_descend=descend)
                time.sleep(1.0)
                print("\n=== %s  descend=%d  grid %s  solid %s ==="
                      % (room, descend, st.get("grid_hash"),
                         st.get("grid_solid")), flush=True)
                r = {"grid_hash": st.get("grid_hash"),
                     "grid_solid": st.get("grid_solid"), "descend": descend}
                for view, name in sorted(VIEWS.items()):
                    lab.doorway(claude_view=view)
                    time.sleep(1.0)
                    png = lab.shot("steps_%s_d%d_v%d" % (room, descend, view),
                                   settle=1.0)
                    out["shots"]["%s/%s" % (key, name)] = png
                    if view == 16:
                        img = np.asarray(Image.open(png).convert("L"))
                        r["ladder_png"] = png
                        r["ladder_rung_fracs"] = [
                            round(float(((img >= lo)
                                         & (img < lo + 32)).mean()), 4)
                            for lo in range(0, 256, 32)]
                        continue
                    counts, blue = decode(png)
                    r[name] = describe(counts)
                    r[name + "_hist"] = histogram(counts, EDGES)
                    if view == 12:
                        r["frac_primary_descended"] = round(
                            float((blue > 127).mean()), 4)
                    print("  %-14s median %8.1f  p90 %8.1f  max %8d  "
                          "zero %.3f"
                          % (name, r[name]["median"], r[name]["p90"],
                             r[name]["max"], r[name]["frac_zero"]), flush=True)
                if r.get("path_coarse", {}).get("mean"):
                    r["fine_over_coarse_mean"] = round(
                        r["path_fine"]["mean"] / r["path_coarse"]["mean"], 3)
                print("  primary rays that descended: %.1f %%"
                      % (100 * r.get("frac_primary_descended", 0)))
                out["rooms"][key] = r
            lab.doorway(claude_descend=1)
    finally:
        try:
            cs.variant("restore")
        except SystemExit:
            pass
        print(cs.variant("status"))
    cs.write_json("steps.json", out)


if __name__ == "__main__":
    main()
