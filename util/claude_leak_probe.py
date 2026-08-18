#!/usr/bin/env python3
"""claude_leak_probe — the sealed-room gate, and the instrument behind it.

WHAT THIS ASKS. A room with no opening receives no sky. `cave-glass` is
that room: the 7x5x7 twin of `cave-skylight` with its one ceiling opening
PLUGGED with the wall node, built as the differential control that
separates "light arrived through the hole" from "light leaked through the
wall". So under a uniform test sky of any brightness its frame must be
EXACTLY BLACK -- zero non-zero pixels, not "within tolerance". That is an
ANALYTIC gate, not a pinned golden, and it is ~40x sharper than the
tightest tolerance in this harness: on 2026-08-17 the room leaked 444
pixels out of 2,073,600 while its three region means still read 0.000000
and its RMS sat at 0.026 against furnace-050's own noise floor of 0.054.

Two modes, and the second is why the first can be believed:

  black   claude_view 0, claude_sky_uniform 50 (a 50x sky, so a leak is
          50x easier to see), one capture per bounce depth. Reports the
          count of non-zero pixels and the brightest one. THE VERDICT.

  view18  claude_view 18, the isolating instrument: take the primary
          hit, draw one cosine-hemisphere bounce exactly as the path loop
          does, and report whether the cell that ray STARTS in is solid
          -- which is the mechanism, since march() never tests the cell a
          ray starts in. Three channels, all on one scale (see the
          shader's own block): R = the event's per-pixel rate, G = the
          share of those events whose starting cell also differs
          TANGENTIALLY from the cell that was hit (a concave corner),
          B = the share differing by exactly +n on the hit face's own
          axis, which is the walk's own answer for "which cell is the
          ray in now".
          READ IT ONLY IN PLAIN-CUBE ROOMS: a class-250 cell is "solid"
          to this test and is not a defect there, so a room full of
          authored models reads ~46 % of frame for that reason alone.

A RATE IS NOT A VERDICT. An event measured over N frames can be made
rarer by a bad fix and still leak given enough frames, so `black` is the
gate and `view18` is the explanation. Run both.

Usage:
  python3 util/claude_leak_probe.py            # both modes, both rooms
  python3 util/claude_leak_probe.py --skip-seat  # a seat is already up
  python3 util/claude_leak_probe.py --mode black --bounces 1,4,24
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_ci as ci  # noqa: E402
import claude_lab as lab  # noqa: E402
import claude_regions as regions  # noqa: E402  (the ONE display inverse)

# The hot sky. 50x a real one: the leak is linear in sky radiance
# (measured 2026-08-17, 4.2e-08 -> 1.9e-06 mean linear from L_sky 1 ->
# 50), so a brighter test sky buys sensitivity and nothing else. It is a
# TEST dial -- claude_sky_uniform replaces the whole dome with one
# constant -- so it also removes the sun, the moon and the moon phase
# from the measurement, which is three fewer things to reproduce.
HOT_SKY = 50

# The rate view needs frames, not seconds: the event is ~1e-6 per pixel
# per frame, so the mean only becomes readable once thousands of pixel
# samples have fired. 2000 is CI's old settle and about 40 s here.
VIEW18_FRAMES = 2000


def unmarked(png):
    """The frame with the harness's own TRACED MARKER blacked out.

    claude_ci's proof that the traced present path drew a frame is a
    12x12 pure-GREEN square in the bottom-left corner (claude_ci
    .trace_marker), i.e. 144 deliberately non-black pixels in every
    capture this harness takes. A gate that says "exactly black" and
    counts them is not measuring the renderer, and a rate channel that
    sums them reports a G/R ratio above 1. It cost a reading on the
    first run of this probe: 144 leaked pixels at claude_bounces = 1 was
    the marker and nothing else."""
    im = np.asarray(Image.open(png).convert("RGB")).copy()
    h = im.shape[0]
    im[h - ci.TRACE_MARKER_PX:h, 0:ci.TRACE_MARKER_PX] = 0
    return im


def nonzero_stats(png):
    """The gate's own arithmetic: how many pixels are not black, and how
    bright the brightest one is. Display bytes, not linear -- "exactly
    black" is a statement about the image, and inverting a tone curve
    first would only add arithmetic to a test that needs none."""
    im = unmarked(png)
    nz = im.max(axis=2) > 0
    return {"nonzero_px": int(nz.sum()), "total_px": int(nz.size),
            "max_byte": int(im.max()),
            "mean_linear": float(regions.aces_inverse(
                    (im.astype(np.float64) / 255.0) ** 2.2).mean())}


def view18_stats(png):
    """Rate, corner share and depth, from the three accumulated channels.

    Every channel is inverted through the SAME display transform the
    region means and the furnace referee use, per channel, because
    claude_present applies ACES and gamma per channel. The frame mean of
    the inverted R channel divided by 1000 is the per-pixel rate; G/R and
    B/R are plain fractions of the events because all three channels were
    written on one scale."""
    lin = regions.aces_inverse((unmarked(png).astype(np.float64) / 255.0)
                               ** 2.2)
    r = lin[:, :, 0]
    sr = float(r.sum())
    return {"rate": sr / r.size / 1000.0,
            "px_ever_positive": int((r > 0).sum()),
            "sideways_share": (float(lin[:, :, 1].sum()) / sr)
                              if sr > 0 else None,
            "across_share": (float(lin[:, :, 2].sum()) / sr)
                            if sr > 0 else None}


def one(name, vantage_name, dials, settle, rundir, vantages):
    shot = {"name": name}
    v = vantages[vantage_name]
    park = ci.park_for(vantage_name, vantages)
    png, info = ci.capture(shot, v, park, dials, rundir, settle,
                           vantage_name=vantage_name)
    return png, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both",
                    choices=["black", "view18", "both"])
    ap.add_argument("--rooms", default="cave-glass,cornell")
    ap.add_argument("--bounces", default="1,4,24")
    ap.add_argument("--skip-seat", action="store_true",
                    help="a seat is already up and frozen")
    ap.add_argument("--skip-build", action="store_true", default=True)
    ap.add_argument("--allow-debug", action="store_true")
    ap.add_argument("--skip-deploy", action="store_true")
    ap.add_argument("--settle", type=int, default=ci.SETTLE_FRAMES,
                    help="accumulated frames before the black-mode shutter")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rundir = os.path.join(ci.CI_DIR, "leak-probe-%s%s"
                          % (time.strftime("%Y%m%d-%H%M%S"),
                             ("-" + args.tag) if args.tag else ""))
    os.makedirs(rundir, exist_ok=True)
    run = {}
    if not args.skip_seat:
        err = ci.bring_up_seat(rundir, run, args)
        if err:
            print("SEAT: " + err)
            return 2
    # The referee rooms must be SHUT to be referees -- an open door leaks
    # the sky into a sealed room by design, which is the one way this
    # gate can be made to fail honestly and mean nothing.
    ci.set_doors(True)

    vantages = lab.load_vantages()
    rooms = [r for r in args.rooms.split(",") if r]
    out = {"rundir": rundir, "git": ci.git_state(), "black": [], "view18": []}

    if args.mode in ("black", "both"):
        for b in [int(x) for x in args.bounces.split(",")]:
            d = dict(ci.CANONICAL_DIALS)
            d.update({"claude_sky_uniform": HOT_SKY, "claude_bounces": b,
                      "claude_view": 0})
            png, info = one("black-b%d" % b, "cave-glass", d,
                            args.settle, rundir, vantages)
            st = nonzero_stats(png)
            st.update({"bounces": b, "png": png,
                       "still_frames": info.get("still_frames")})
            out["black"].append(st)
            print("BLACK cave-glass bounces=%-2d  nonzero=%-7d max=%-3d "
                  "mean_linear=%.3e" % (b, st["nonzero_px"], st["max_byte"],
                                        st["mean_linear"]))

    if args.mode in ("view18", "both"):
        for room in rooms:
            d = dict(ci.CANONICAL_DIALS)
            d.update({"claude_sky_uniform": HOT_SKY, "claude_view": 18})
            png, info = one("view18-%s" % room, room, d, VIEW18_FRAMES,
                            rundir, vantages)
            st = view18_stats(png)
            st.update({"room": room, "png": png,
                       "still_frames": info.get("still_frames")})
            out["view18"].append(st)
            print("VIEW18 %-12s rate=%.3e px=%-7d sideways=%s across=%s"
                  % (room, st["rate"], st["px_ever_positive"],
                     ("%.3f" % st["sideways_share"])
                     if st["sideways_share"] is not None else "-",
                     ("%.3f" % st["across_share"])
                     if st["across_share"] is not None else "-"))

    json.dump(out, open(os.path.join(rundir, "leak.json"), "w"), indent=2)
    print("\nwrote %s" % os.path.join(rundir, "leak.json"))
    verdict = all(s["nonzero_px"] == 0 for s in out["black"]) \
        if out["black"] else None
    if verdict is not None:
        print("GATE 1 (cave-glass exactly black under a %dx sky): %s"
              % (HOT_SKY, "PASS" if verdict else "FAIL"))
    return 0 if (verdict in (None, True)) else 1


if __name__ == "__main__":
    sys.exit(main())
