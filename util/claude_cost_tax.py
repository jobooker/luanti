#!/usr/bin/env python3
"""EXPERIMENT A of the descend cost instrument: THE SHADER-SIZE TAX.

THE QUESTION. When the walker learned to step inside a class-250 cell
("descend", 2026-08-16), the trace pass got slower in the cosy cabin --
which was expected, it does more work there -- and ALSO in Cornell,
which contains no class-250 cell at all, with the descend dial set to
OFF. A switch that is off cannot cost time by running. Two explanations
survive that sentence:

  (a) the code's mere PRESENCE costs. A bigger, branchier shader needs
      more registers per thread; past a threshold the GPU keeps fewer
      threads in flight (lower "occupancy"), and then EVERY ray in the
      program is slower, including the ones that never take the branch.
  (b) it does not, and the earlier 11.19 -> 15.15 ms reading in Cornell
      was something else (a different binary, a different conf, drift).

This script separates them. Three shader arms, ONE binary, one server,
one world, one session:

  absent   descent COMPILED OUT -- descendCell() and both call sites
           deleted from the shader text (util/claude_shader_variant.py)
  off      the committed shader, claude_descend = 0
  on       the committed shader, claude_descend = 1

and, from 2026-08-17, further pinned arms -- each one a shader this
repo actually shipped, installed verbatim from its commit, so that
"did that change pay" is answered on ONE binary, ONE server and ONE
parked camera instead of across two days. The seat drifts about 5-7 %
between sessions and that drift lands entirely on a cross-session
comparison, which is the whole reason these arms exist:

  old-off       the TWO-WALK shader (march() + a separate
  old-on        descendCell()), pinned at 64b005975.
  twoscale-off  the ONE-LOOP, TWO-RUNG walk (1 m and 1/16 m, no
  twoscale-on   sub-brick summary), pinned at f731209f9. This is the
                arm the sub-brick hierarchy has to beat.

crossed with two rooms: `cornell` (no class-250 cell exists, so `on` and
`off` must agree) and `cozy-ci` (95.6 % of its solid cells are class
250). The number reported for each is the claude_trace GPU pass in
milliseconds, min/median/max over the reps.

HOW TO READ THE ANSWER, decided before the run:
  * absent FAST and off SLOW in Cornell  ->  hypothesis (a) survives:
    the code costs by existing, and no dial can switch that off. The fix
    is structural.
  * absent == off in Cornell             ->  hypothesis (a) is dead and
    the original reading was measuring something else. Say so plainly.

WHAT MOVES BETWEEN ARMS, and nothing else: the shader text (one client
restart per variant) and the claude_descend uniform (a live dial, no
restart, camera provably unmoved between the off and on readings of the
same rep). The grid hash is recorded for every reading; more than one
value in a room means the world moved under the measurement.

Usage:  util/claude_cost_tax.py [--reps 3] [--trace-scale 1.0]
The SERVER must already be up; this script restarts the CLIENT twice.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import claude_lab as lab            # noqa: E402
import claude_cost_seat as cs       # noqa: E402

ROOMS = ["cornell", "cozy-ci"]
# The arm labels each variant contributes to the table. `sampleronly` is
# experiment E's control: descent code absent but claudeSubvoxTex still
# read from a branch that can never be taken, which separates "the code
# costs" from "the extra 3D sampler costs".
VARIANT_ARM = {"live": [("off", 0), ("on", 1)],
               "nodescend": [("absent", 0)],
               "twoscale": [("twoscale-off", 0), ("twoscale-on", 1)],
               "twowalk": [("old-off", 0), ("old-on", 1)],
               "cheapbit": [("cheapbit-off", 0)],
               "sampleronly": [("absent+sampler", 0)]}

# The photo state every reading is taken in. Identical to CI's
# CANONICAL_DIALS for everything that touches shader cost; pushed
# explicitly because an unset claude_* dial is a silent default.
BASE_DIALS = {
    "claude_view": 0, "claude_nee": 0, "claude_bounces": 24,
    "claude_grid_debug": 3, "claude_rng": 1, "claude_grid_follow": 1,
    "claude_models": 1, "claude_show_hud": 0, "claude_show_chat": 0,
    "claude_input_lock": 1, "claude_stats": 1,
}


def arms_for(variant):
    """(label, descend-dial) pairs. An arm whose shader does not read
    claude_descend still pushes it -- the uniform is simply ignored
    there, and pushing it keeps every arm's dial state identical."""
    return VARIANT_ARM[variant]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--variants", nargs="+", default=["live", "nodescend"],
                    choices=sorted(VARIANT_ARM))
    ap.add_argument("--rooms", nargs="+", default=ROOMS)
    ap.add_argument("--out", default="shader_tax.json")
    ap.add_argument("--trace-scale", default="1.0",
                    help="claude_trace_scale; read at pipeline "
                         "construction, so it needs the client restart "
                         "this script does anyway")
    ap.add_argument("--settle", type=float, default=6.0)
    a = ap.parse_args()

    build = cs.require_release()
    # A variant left installed by a crashed run would make every arm
    # below a lie about which shader it timed. Clear it first.
    if os.path.exists(os.path.join(HERE, "claude_shader_variants",
                                   ".installed.json")):
        cs.variant("restore")
    cs.variant("build")

    out = {"build_type": build, "trace_scale": a.trace_scale,
           "reps": a.reps, "readings": [], "started": time.time()}
    try:
        for variant in a.variants:
            if variant != "live":
                cs.variant("install", variant)
            print("\n=== shader variant: %s ===" % variant, flush=True)
            cs.restart_client({"claude_trace_scale": a.trace_scale})
            lab.doorway(**BASE_DIALS)
            print("time_speed -> %s" % cs.freeze_time())
            for rep in range(a.reps):
                for room in a.rooms:
                    cs.park(room, dials=BASE_DIALS,
                            snapshot_tag="tax_%s_%s_%d" % (variant, room, rep))
                    for label, descend in arms_for(variant):
                        lab.doorway(claude_descend=descend)
                        p = cs.read_passes(settle_s=a.settle)
                        p.update(variant=variant, arm=label, room=room,
                                 rep=rep, descend=descend)
                        out["readings"].append(p)
                        print("%-9s %-8s %-9s rep%d  trace %7.3f "
                              "(spread %.3f)  frame %6.2f  grid %s"
                              % (variant, room, label, rep, p["trace"],
                                 p["trace_spread"], p["frame_ms_avg"],
                                 p["grid_hash"]), flush=True)
            if variant != "live":
                cs.variant("restore")
    finally:
        # The live shaders must be the committed ones when this exits,
        # whatever happened -- CI compares against them.
        try:
            cs.variant("restore")
        except SystemExit:
            pass
        print(cs.variant("status"))

    # --- the 2x3 table ------------------------------------------------
    table = {}
    arm_names = [lab_ for v in a.variants for lab_, _ in VARIANT_ARM[v]]
    for room in a.rooms:
        for arm in arm_names:
            v = [r["trace"] for r in out["readings"]
                 if r["room"] == room and r["arm"] == arm]
            if v:
                table["%s/%s" % (room, arm)] = cs.mmm(v)
    out["table_trace_ms"] = table
    out["grid_hashes"] = {room: sorted({r["grid_hash"] for r in out["readings"]
                                        if r["room"] == room})
                          for room in a.rooms}
    out["one_grid_per_room"] = all(len(v) == 1
                                   for v in out["grid_hashes"].values())
    out["cap_artifacts"] = [r["cap_artifact"] for r in out["readings"]
                            if r["cap_artifact"]]
    print("\nTRACE PASS ms (min / median / max over %d reps)" % a.reps)
    print("%-10s" % "room" + "".join("%-24s" % n for n in arm_names))
    for room in a.rooms:
        cells = []
        for arm in arm_names:
            t = table.get("%s/%s" % (room, arm))
            cells.append("%.2f / %.2f / %.2f" % (t["min"], t["median"],
                                                 t["max"]) if t else "-")
        print("%-10s" % room + "".join("%-24s" % c for c in cells))
    cs.write_json(a.out, out)


if __name__ == "__main__":
    main()
