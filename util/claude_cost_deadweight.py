#!/usr/bin/env python3
"""PROBE 2 of the cost investigation: DOES A BIGGER SHADER COST MORE?

THE WORK, PLAINLY. On 2026-08-16 the ray walk learned to step INSIDE a
1 m cell so a stair renders as a stair. The trace pass got much slower
-- and it also got slower in Cornell, a room containing no such cell,
with the feature's own dial switched OFF. Code that never runs was
costing about 1.4-1.5x. That surcharge is the "presence tax", and after
three experiments at least 60 % of it still has no owner.

Every attempt so far REMOVED something and asked whether the tax fell:
  * lean-descend deleted the second walk's whole live state ->  0 %
  * cheapbit removed subvoxSolid()'s bit extraction         -> <=25 % in
    Cornell, invisible in the cabin
  * sampleronly isolated the extra 3D sampler               ->  14 %
None of those could ever come back "the theory is wrong"; they could
only come back "still unexplained", and they did.

THIS ONE CAN COME BACK NEGATIVE. Instead of taking weight out, it puts
weight IN: K extra vec4 locals and K dependent reads of the trace grid,
inside the walk's loop, behind `if (claudeDescend > 5.0)` -- a branch no
pixel can ever take, because game.cpp clamps that setting to [0,1]. The
compiler must still emit it (claudeDescend is a uniform, so its value is
not knowable at compile time) and must still allocate for it (the result
reaches gl_FragColor, as an addition of exactly 0.0, so it cannot be
dead-stripped). See util/claude_shader_variant.py for the three shapes.

THE PREDICTION, WRITTEN DOWN BEFORE THE RUN.
  * If trace-pass ms GROWS with K, then a bigger shader is a slower one
    on this GPU even when the extra code never executes. The occupancy
    story is CONFIRMED, the tax finally has a mechanism, and the slope
    (ms per vec4) is a dial every future change can be costed against.
  * If the curve is FLAT, occupancy is dead too. Nothing anyone has
    proposed explains the tax, and the next move is a profiler, which
    macOS does not have for OpenGL -- i.e. the Windows rig.
  * The registers-only and reads-only arms split a positive result:
    which of the two is doing it.

THE SETTLE-SPREAD GUARD, which the subbrick session owed this file.
An earlier cost table taken at --settle 6 did not read noisy, it read
WRONG IN A DIRECTION (spec/measured.md, subbrick experiment, section A):
pass_ms is an exponential moving average, and a reading taken before it
has caught up is biased toward the arm measured before it. The spread
across the three samples of one reading is already computed by
claude_cost_seat.read_passes, and this script REFUSES a reading whose
spread exceeds --max-spread (default 0.5 ms), retrying it up to
--retries times with a longer settle and recording every attempt. A
reading that never settles is kept and flagged rather than dropped, so
the table can never be quietly built out of the readings that happened
to behave.

Usage:  util/claude_cost_deadweight.py [--reps 3] [--settle 12]
The SERVER must already be up; this script restarts the CLIENT once per
shader text and restores the committed shaders on the way out.
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

# (variant, arm label, K, what the K buys). `live` is K = 0 by
# definition -- it IS the committed shader -- so no K = 0 variant is
# generated and no client restart is spent proving a banner is free.
ARMS = [
    ("nodescend", "absent",   None, "the pre-descend shader, for scale"),
    ("live",      "K=0",      0,    "the committed shader"),
    ("dw4",       "K=4",      4,    "4 vec4 + 4 dependent grid reads"),
    ("dw8",       "K=8",      8,    "8 vec4 + 8 dependent grid reads"),
    ("dw16",      "K=16",     16,   "16 vec4 + 16 dependent grid reads"),
    ("dw32",      "K=32",     32,   "32 vec4 + 32 dependent grid reads"),
    ("dwreg16",   "K=16 reg", 16,   "16 vec4, NO grid reads"),
    ("dwfetch16", "K=16 fetch", 16, "16 dependent grid reads, 1 live vec4"),
    ("dwreg32",   "K=32 reg", 32,   "32 vec4, NO grid reads"),
    ("dwfetch32", "K=32 fetch", 32, "32 dependent grid reads, 1 live vec4"),
]

# The photo state every reading is taken in, pushed explicitly because
# an unset claude_* dial is a silent default (environment-laws, the
# hidden-default class).
BASE_DIALS = {
    "claude_view": 0, "claude_nee": 0, "claude_bounces": 24,
    "claude_grid_debug": 3, "claude_rng": 1, "claude_grid_follow": 1,
    "claude_models": 1, "claude_show_hud": 0, "claude_show_chat": 0,
    "claude_input_lock": 1, "claude_stats": 1,
}


def settled_reading(settle, max_spread, retries, label):
    """One reading, with the EMA-spread guard. Returns the reading and
    the list of attempts that were rejected before it."""
    rejected = []
    s = settle
    for attempt in range(retries + 1):
        p = cs.read_passes(settle_s=s)
        p["settle_s"] = s
        p["attempt"] = attempt
        if p["trace_spread"] <= max_spread or attempt == retries:
            p["spread_ok"] = p["trace_spread"] <= max_spread
            p["rejected_attempts"] = rejected
            if not p["spread_ok"]:
                print("    !! %s KEPT WITH A BAD SPREAD %.3f ms after %d "
                      "attempts -- flagged, not dropped"
                      % (label, p["trace_spread"], attempt + 1), flush=True)
            return p
        print("    .. %s spread %.3f ms > %.2f, re-reading at settle %.0f"
              % (label, p["trace_spread"], max_spread, s * 1.5), flush=True)
        rejected.append({"settle_s": s, "trace": round(p["trace"], 3),
                         "trace_spread": round(p["trace_spread"], 3)})
        s *= 1.5


def frame_check(a):
    """THE PROOF THAT THE DEAD WEIGHT REALLY IS DEAD.

    The argument that the inert block never executes is static -- the
    guard is `claudeDescend > 5.0` and game.cpp clamps that setting to
    [0,1]. Static arguments about shaders are exactly what this repo
    keeps getting caught by, so here is the frame. If the block ever
    ran, g_dw would be the sum of K dependent texture reads accumulated
    at every step of every ray -- hundreds of units of radiance added to
    a pixel whose normal value is single digits -- and the picture would
    be flat white. So the check is the mean luminance of the frame, and
    it does not need a tolerance argument to be convincing: the failure
    mode is not subtle.

    The two frames come from different client sessions and therefore
    different accumulator depths, so an RMS between them is NOT expected
    to be zero and is reported for information only, against the arm's
    own 500-frame CI floor of 2.48-2.93."""
    import numpy as np
    from PIL import Image
    cs.require_release()
    if os.path.exists(os.path.join(HERE, "claude_shader_variants",
                                   ".installed.json")):
        cs.variant("restore")
    cs.variant("build")
    out = {"room": "cozy-ci", "shots": {}}
    try:
        for name in ("live", "dw32"):
            if name != "live":
                cs.variant("install", name)
            cs.restart_client({"claude_trace_scale": a.trace_scale})
            lab.doorway(**BASE_DIALS)
            cs.freeze_time()
            cs.park("cozy-ci", dials=BASE_DIALS,
                    snapshot_tag="dwframe_%s" % name)
            lab.doorway(claude_descend=1)
            time.sleep(a.settle)
            png = lab.shot(token="dwframe_%s" % name)
            arr = np.asarray(Image.open(png).convert("RGB"),
                             dtype=np.float32)
            luma = float((0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1]
                          + 0.0722 * arr[:, :, 2]).mean())
            out["shots"][name] = {"png": png, "mean_luma": round(luma, 4),
                                  "frac_saturated": round(
                                      float((arr > 250).mean()), 5)}
            print("  %-5s mean luma %8.4f  saturated %.5f  %s"
                  % (name, luma, out["shots"][name]["frac_saturated"], png),
                  flush=True)
            if name != "live":
                cs.variant("restore")
    finally:
        try:
            cs.variant("restore")
        except SystemExit:
            pass
        print(cs.variant("status"))
    a_, b_ = out["shots"]["live"], out["shots"]["dw32"]
    out["luma_delta"] = round(b_["mean_luma"] - a_["mean_luma"], 4)
    out["rms_between_sessions"] = lab.rms_diff(a_["png"], b_["png"])
    print("\nmean luma  live %.4f  dw32 %.4f  delta %+.4f"
          % (a_["mean_luma"], b_["mean_luma"], out["luma_delta"]))
    print("rms between the two sessions: %s  (cozy-ci 500-frame CI floor "
          "is 2.48-2.93; these are different accumulator depths, so this "
          "is information, not a gate)" % out["rms_between_sessions"])
    cs.write_json("deadweight_frame.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rooms", nargs="+", default=ROOMS)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="variant names; default is every arm in ARMS")
    # WHY THIS EXISTS, and it is not a convenience. The default sweep
    # visits the arms in increasing K, once each, so K and WALL-CLOCK
    # TIME move together -- and this seat drifts. The same shader text
    # in the same room read 14.06 ms at the start of a session and
    # 17.3 ms forty minutes later (2026-08-17), which is larger than
    # the whole effect being looked for. A monotone sweep cannot tell a
    # bigger shader from a warmer machine. `--order` takes an explicit
    # sequence WITH REPEATS, so the two endpoints can be alternated
    # (live dw32 live dw32 ...) and each dw32 reading compared against
    # the live readings either side of it. That is a paired design and
    # it cancels any drift that is smooth over one A/B pair.
    ap.add_argument("--order", nargs="+", default=None,
                    help="explicit variant sequence, repeats allowed, e.g. "
                         "--order live dw32 live dw32")
    ap.add_argument("--out", default="deadweight.json")
    ap.add_argument("--trace-scale", default="1.0")
    # 12 is the settle the subbrick session's CLEAN run used; its 6 was
    # the one that read wrong in a direction.
    ap.add_argument("--settle", type=float, default=12.0)
    ap.add_argument("--max-spread", type=float, default=0.5,
                    help="reject a reading whose 3-sample trace-ms spread "
                         "exceeds this and re-read at a longer settle")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--frame-check", action="store_true",
                    help="photograph cozy-ci on the committed shader and "
                         "on dw32 and compare, instead of timing anything")
    a = ap.parse_args()

    if a.frame_check:
        return frame_check(a)

    byname = {r[0]: r for r in ARMS}
    if a.order:
        arms = [byname[v] for v in a.order]
    else:
        arms = [r for r in ARMS if a.arms is None or r[0] in a.arms]
    # With repeats, a visit needs its own identity or three visits to
    # `live` would be averaged into one cell and the whole point lost.
    seen = {}
    visits = []
    for v, l, k, w in arms:
        seen[v] = seen.get(v, 0) + 1
        visits.append((v, "%s#%d" % (l, seen[v]) if a.order else l, k, w))
    arms = visits
    build = cs.require_release()
    if os.path.exists(os.path.join(HERE, "claude_shader_variants",
                                   ".installed.json")):
        cs.variant("restore")
    cs.variant("build")

    out = {"build_type": build, "trace_scale": a.trace_scale,
           "reps": a.reps, "settle": a.settle, "max_spread": a.max_spread,
           "arms": [{"variant": v, "label": l, "K": k, "what": w}
                    for v, l, k, w in arms],
           "readings": [], "started": time.time()}
    try:
        for variant, label, K, what in arms:
            if variant != "live":
                cs.variant("install", variant)
            print("\n=== %s  (%s) ===" % (label, what), flush=True)
            cs.restart_client({"claude_trace_scale": a.trace_scale})
            lab.doorway(**BASE_DIALS)
            print("time_speed -> %s" % cs.freeze_time())
            for rep in range(a.reps):
                for room in a.rooms:
                    cs.park(room, dials=BASE_DIALS,
                            snapshot_tag="dw_%s_%s_%d" % (variant, room, rep))
                    # BOTH dial states. The tax is the descend=0 column;
                    # the descend=1 column says whether dead weight also
                    # slows the walk down where it actually runs, which
                    # is the same occupancy claim on the other half of
                    # the bill.
                    for descend in (0, 1):
                        lab.doorway(claude_descend=descend)
                        tag = "%s %s d%d rep%d" % (label, room, descend, rep)
                        p = settled_reading(a.settle, a.max_spread,
                                            a.retries, tag)
                        p.update(variant=variant, arm=label, K=K, room=room,
                                 rep=rep, descend=descend)
                        out["readings"].append(p)
                        print("  %-11s %-8s d%d rep%d  trace %7.3f "
                              "(spread %.3f)  frame %6.2f  grid %s"
                              % (label, room, descend, rep, p["trace"],
                                 p["trace_spread"], p["frame_ms_avg"],
                                 p["grid_hash"]), flush=True)
            if variant != "live":
                cs.variant("restore")
    finally:
        try:
            cs.variant("restore")
        except SystemExit:
            pass
        print(cs.variant("status"))

    # --- the curve ----------------------------------------------------
    table = {}
    for _v, label, _k, _w in arms:
        for room in a.rooms:
            for descend in (0, 1):
                v = [r["trace"] for r in out["readings"]
                     if r["room"] == room and r["arm"] == label
                     and r["descend"] == descend]
                if v:
                    table["%s/d%d/%s" % (room, descend, label)] = cs.mmm(v)
    out["table_trace_ms"] = table
    out["grid_hashes"] = {room: sorted({r["grid_hash"] for r in out["readings"]
                                        if r["room"] == room})
                          for room in a.rooms}
    out["one_grid_per_room"] = all(len(v) == 1
                                   for v in out["grid_hashes"].values())
    out["bad_spread_readings"] = [
        {"arm": r["arm"], "room": r["room"], "descend": r["descend"],
         "rep": r["rep"], "trace_spread": round(r["trace_spread"], 3)}
        for r in out["readings"] if not r.get("spread_ok", True)]
    out["max_spread_seen"] = round(max(r["trace_spread"]
                                       for r in out["readings"]), 3)
    out["cap_artifacts"] = [r["cap_artifact"] for r in out["readings"]
                            if r["cap_artifact"]]

    print("\nTRACE PASS ms (median of %d reps)" % a.reps)
    hdr = "%-12s" % "arm"
    for room in a.rooms:
        for descend in (0, 1):
            hdr += "%-14s" % ("%s d%d" % (room[:8], descend))
    print(hdr)
    for _v, label, _k, _w in arms:
        line = "%-12s" % label
        for room in a.rooms:
            for descend in (0, 1):
                t = table.get("%s/d%d/%s" % (room, descend, label))
                line += "%-14s" % ("%.2f" % t["median"] if t else "-")
        print(line)
    print("\nworst EMA spread over all readings: %.3f ms  (guard %.2f)"
          % (out["max_spread_seen"], a.max_spread))
    if out["bad_spread_readings"]:
        print("READINGS KEPT WITH A BAD SPREAD: %d -- see the JSON"
              % len(out["bad_spread_readings"]))
    cs.write_json(a.out, out)


if __name__ == "__main__":
    main()
