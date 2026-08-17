#!/usr/bin/env python3
"""EXPERIMENT C of the descend cost instrument: THE RASTER SLICE.

THE QUESTION (John, 2026-08-16): "how much of the vanilla renderer is
taking up shader space?" None -- the raster renderer and the tracer are
separate GPU programs with separate register budgets, so neither one
crowds the other. But the raster renderer still takes TIME: with tracing
on, Luanti draws the entire world the ordinary way every frame and the
tracer then paints over it, borrowing only the depth buffer to tell sky
from geometry. Nothing on file said how big that slice is. If it is a
large fixed cost, making the tracer twice as fast buys much less than it
sounds.

WHAT THIS PRINTS, per room, per arm: every GPU pass the engine already
times (raster draw, vanilla final merge, the tracer, the upsample+tone
map, the buffer swap), the whole frame, and THE REMAINDER -- frame time
minus the timed passes, which is the wield item, the post-effects, the
HUD and the buffer present, none of which the pass profiler covers.

ARMS
  traced      the normal photo state (claude_grid_debug = 3), descent on
  traced-off  same, claude_descend = 0
  raster      claude_grid_debug = 0 -- the tracer's own pass still runs
              but returns the history texel on its first line, so this
              arm is the raster cost PLUS the price of dragging two
              dormant full-screen passes across the frame. Both are
              reported; do not read it as "vanilla Luanti".
  gridview    claude_grid_debug = 1, which is the ONLY arm where
              second_stage's own seven DDA loops over the trace grid
              execute. At the photo setting of 3 they are gated off and
              cost at most the whole second_stage pass, which this
              measures.

Usage:  util/claude_cost_raster.py [--reps 3]
The SERVER must already be up; the CLIENT is restarted once.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import claude_lab as lab            # noqa: E402
import claude_cost_seat as cs       # noqa: E402

ROOMS = ["cozy-ci", "cornell"]
ARMS = [
    ("traced", {"claude_grid_debug": 3, "claude_descend": 1}),
    ("traced-off", {"claude_grid_debug": 3, "claude_descend": 0}),
    ("raster", {"claude_grid_debug": 0, "claude_descend": 1}),
    ("gridview", {"claude_grid_debug": 1, "claude_descend": 1}),
]
BASE_DIALS = {
    "claude_view": 0, "claude_nee": 0, "claude_bounces": 24,
    "claude_rng": 1, "claude_grid_follow": 1, "claude_models": 1,
    "claude_show_hud": 0, "claude_show_chat": 0,
    "claude_input_lock": 1, "claude_stats": 1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--trace-scale", default="1.0")
    ap.add_argument("--settle", type=float, default=6.0)
    a = ap.parse_args()

    build = cs.require_release()
    if os.path.exists(os.path.join(HERE, "claude_shader_variants",
                                   ".installed.json")):
        cs.variant("restore")
    out = {"build_type": build, "trace_scale": a.trace_scale,
           "readings": []}
    cs.restart_client({"claude_trace_scale": a.trace_scale})
    lab.doorway(**dict(BASE_DIALS, claude_grid_debug=3, claude_descend=1))
    print("time_speed -> %s" % cs.freeze_time())
    for rep in range(a.reps):
        for room in ROOMS:
            cs.park(room, dials=dict(BASE_DIALS, claude_grid_debug=3,
                                     claude_descend=1),
                    snapshot_tag="raster_%s_%d" % (room, rep))
            for label, dials in ARMS:
                lab.doorway(**dials)
                p = cs.read_passes(settle_s=a.settle)
                timed = sum(p[n] for n in cs.PASS_NAMES)
                p["timed_passes"] = timed
                p["remainder"] = p["frame_ms_avg"] - timed
                p.update(arm=label, room=room, rep=rep, dials=dials)
                out["readings"].append(p)
                print("%-9s %-11s rep%d  raster %5.2f  2nd %5.2f  trace %6.2f"
                      "  present %5.2f  timed %6.2f  frame %6.2f  rest %5.2f"
                      % (room, label, rep, p["raster3d"], p["second_stage"],
                         p["trace"], p["present"], timed, p["frame_ms_avg"],
                         p["remainder"]), flush=True)
    # restore the photo state so the seat is left the way CI expects it
    lab.doorway(**dict(BASE_DIALS, claude_grid_debug=3, claude_descend=1))

    summary = {}
    for room in ROOMS:
        for label, _ in ARMS:
            sel = [r for r in out["readings"]
                   if r["room"] == room and r["arm"] == label]
            if not sel:
                continue
            summary["%s/%s" % (room, label)] = {
                k: cs.mmm([r[k] for r in sel])
                for k in ("raster3d", "second_stage", "trace", "present",
                          "timed_passes", "remainder", "frame_ms_avg",
                          "draw_ms", "busy_ms")}
    out["summary"] = summary
    print("\nmedian ms")
    print("%-22s %8s %8s %8s %8s %8s %8s"
          % ("room/arm", "raster", "2nd", "trace", "present", "rest", "frame"))
    for k, v in summary.items():
        print("%-22s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f"
              % (k, v["raster3d"]["median"], v["second_stage"]["median"],
                 v["trace"]["median"], v["present"]["median"],
                 v["remainder"]["median"], v["frame_ms_avg"]["median"]))
    cs.write_json("raster_slice.json", out)


if __name__ == "__main__":
    main()
