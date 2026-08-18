#!/usr/bin/env python3
"""claude_sealed_probe — the sealed-plank sweep, and the proof it is not blind.

WHAT THIS IS FOR, in plain terms. On 2026-08-17 the cabin's walls were
found leaking light: the baked 16^3 masks of the plank models carve
grooves into every face, the grooves of one block line up with those of
the next, and a ray that is not axis aligned walks straight through a
wall one node thick. b945042ee fixed it. NOTHING IN CI WOULD HAVE CAUGHT
IT AND NOTHING WOULD CATCH IT COMING BACK -- cave-glass is the only
sealed arm and it is built from a PLAIN wall node, with no relief in it
at all.

John asked for this room in these words: "a fully enclosed house, with a
wall of emissive blocks surrounding it entirely would be good. the room
should be completely dark, any holes would bleed through... check out the
old broken code from before the fix and verify this test fails on that
and then passes on the fix."

THE ROOM (util/claude_bridge_gallery.lua, OPS.sealedbox; deployed at
(0,8,250)): an emissive shell, a one-node air gap, an oak-plank shell
EXACTLY one node thick, and a 5x4x5 interior with no light source of any
kind. Any non-black interior pixel is a leak.

A SWEEP, NOT A VIEW. John's own observation of the original defect was
that it "seems to occur only when looking east". A leak is a particular
ray through a particular groove pair, so a single fixed vantage can miss
it entirely -- a referee that looks one way is a referee that can be
walked around. Six aims off one standing position: +Z, -Z, +X, -X, up,
down.

DEEP, NOT FAST. Measured 2026-08-17 on the same broken build: the
origin-cell leak read 0 leaked pixels at a 500-frame settle and 144 at
2000. A shallow run can certify a rare defect ABSENT.

AND THE ZERO IS NOT BELIEVED UNTIL THE CAPTURE IS SHOWN TO CARRY SIGNAL.
A black frame is a uniform frame, and this project has now paid six times
for an instrument that reported success while measuring nothing (the
usampler golden diff, the -grid check gated on area_total, the
reset_accumulation sampler, the stats file that could not tell traced
from raster, the leak probe that counted the harness's own marker, and
`screencapture` returning an all-black image with no screen-recording
grant). So every run of this probe also shoots the emissive shell from
OUTSIDE, and REFUSES to report a pass if that frame is not bright --
exactly as util/claude_sliver_ab.py refuses when both its frames are
black.

Usage:
  python3 util/claude_sealed_probe.py                  # seat + full sweep
  python3 util/claude_sealed_probe.py --skip-seat      # a seat is already up
  python3 util/claude_sealed_probe.py --tag broken     # label the run dir
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_ci as ci          # noqa: E402
import claude_lab as lab        # noqa: E402
import claude_sealed_check as sealed   # noqa: E402

AIMS = ("north", "south", "east", "west", "up", "down")
OUTSIDE = "sealed-plank-outside"
# The live-signal floor for the outside shot. The emissive shell fills
# most of that frame at point-blank range and reads far above this; the
# number only has to be high enough that a blinded (all-black or
# near-black) capture cannot clear it.
OUTSIDE_MIN_MEAN_BYTE = 8.0


def shoot(name, vantage_name, vantages, dials, settle, rundir):
    v = vantages[vantage_name]
    png, info = ci.capture({"name": name}, v, ci.park_for(vantage_name, vantages),
                           dials, rundir, settle, vantage_name=vantage_name)
    return png, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=int, default=ci.SEALED_SETTLE)
    ap.add_argument("--skip-seat", action="store_true")
    ap.add_argument("--skip-build", action="store_true", default=True)
    ap.add_argument("--allow-debug", action="store_true")
    ap.add_argument("--skip-deploy", action="store_true")
    ap.add_argument("--aims", default=",".join(AIMS))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rundir = os.path.join(ci.CI_DIR, "sealed-%s%s"
                          % (time.strftime("%Y%m%d-%H%M%S"),
                             ("-" + args.tag) if args.tag else ""))
    os.makedirs(rundir, exist_ok=True)
    run = {}
    if not args.skip_seat:
        err = ci.bring_up_seat(rundir, run, args)
        if err:
            print("SEAT: " + err)
            return 2

    vantages = lab.load_vantages()
    out = {"rundir": rundir, "git": ci.git_state(), "settle": args.settle,
           "aims": [], "outside": None}
    dials = dict(ci.CANONICAL_DIALS)      # photo mode, 24 bounces, real sky

    # ---- 1. THE LIVE-SIGNAL CONTROL, FIRST, before any zero is read ----
    png, info = shoot("outside", OUTSIDE, vantages, dials, ci.SETTLE_FRAMES,
                      rundir)
    im = sealed.unmarked(png)
    mean_byte = float(im.mean())
    out["outside"] = {"png": png, "mean_byte": mean_byte,
                      "max_byte": int(im.max()),
                      "still_frames": info.get("still_frames")}
    print("OUTSIDE (emissive shell from open ground): mean byte %.2f, "
          "max %d" % (mean_byte, im.max()))
    live = mean_byte >= OUTSIDE_MIN_MEAN_BYTE
    if not live:
        print("REFUSING TO REPORT: the outside shot is not bright "
              "(mean byte %.2f < %.1f). A black frame is a uniform frame; "
              "this capture path is blind, or the shell is not emitting, "
              "and a zero from the interior would mean nothing."
              % (mean_byte, OUTSIDE_MIN_MEAN_BYTE))
        json.dump(out, open(os.path.join(rundir, "sealed.json"), "w"), indent=2)
        return 3

    # ---- 2. THE SWEEP -------------------------------------------------
    total = 0
    for aim in [a for a in args.aims.split(",") if a]:
        vn = "sealed-plank-" + aim
        png, info = shoot(aim, vn, vantages, dials, args.settle, rundir)
        c = sealed.count(png)
        # PER-FRAME PROOF THAT THIS BLACK FRAME IS A RENDERED ONE. The
        # 12x12 marker claude_ci paints into every capture is the
        # harness's own evidence that the TRACED present path drew this
        # image; without it a zero here is indistinguishable from a
        # window that never rendered. The count above masks the marker,
        # so this reads it back before it is thrown away.
        mk_ok, mk = ci.trace_marker(png)
        c["traced"] = {"ok": bool(mk_ok), "detail": mk}
        if not mk_ok:
            print("  !! %s: NO TRACED MARKER (%s) — this frame is not "
                  "evidence of anything" % (aim, mk))
        c.update({"aim": aim, "png": png, "yaw": vantages[vn]["yaw"],
                  "pitch": vantages[vn]["pitch"],
                  "still_frames": info.get("still_frames"),
                  "settle": (info.get("settle") or {}).get("still_frames"),
                  "room_hash": info.get("room_hash")})
        out["aims"].append(c)
        total += c["nonzero_px"]
        print("SWEEP %-6s yaw %-3s pitch %-3s  leaked %-8d max byte %-3d  "
              "brightest at %s"
              % (aim, c["yaw"], c["pitch"], c["nonzero_px"], c["max_byte"],
                 c["brightest_at"]))

    out["total_leaked_px"] = total
    untraced = [c["aim"] for c in out["aims"] if not c["traced"]["ok"]]
    out["untraced_aims"] = untraced
    json.dump(out, open(os.path.join(rundir, "sealed.json"), "w"), indent=2)
    print("\nwrote %s" % os.path.join(rundir, "sealed.json"))
    print("TOTAL over %d aims at a %d-frame settle: %d non-black pixels"
          % (len(out["aims"]), args.settle, total))
    if untraced:
        print("REFUSING TO REPORT: no traced marker on %s — those frames "
              "were not drawn by the traced present path, so their zeros "
              "are not measurements." % ", ".join(untraced))
        return 3
    print("SEALED-PLANK GATE (interior exactly black): %s"
          % ("PASS" if total == 0 else "FAIL"))
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
