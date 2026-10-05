#!/usr/bin/env python3
"""claude_glass_bars — does glass light up the grooves of the wood around it?

WHAT THIS IS FOR, in plain terms. On 2026-10-04 John said the glass was
"all janky". Measured the same day (luanti-docs spec/measured.md, "glass
beside carved wood"): seen through a window, the wood faces around the
opening carried bright bars along every plank groove, in the colours of
the view outside. A ray inside the glass met a groove's 1/16 m air pocket
at a grazing angle, totally internally reflected, and carried the outside
view into the groove. John's call: glass sits flush against carved wood
(claude_glass_flush = 1, the shader's marchMed() fineHere gate).

THE ROOM is a small oak hut at x 100..112, z 200..206, well outside every
CI bubble. Its south wall (z = 200) holds a 3x2 pane window (x 102..104)
and a 3x2 glass-block window (x 108..110). The camera stands inside,
close to each window, looking out with the sun up.

THE MEASUREMENT is per frame: a "bar pixel" is a lit pixel that is not
part of the window view (see bars()). Three frames at one aim:
claude_glass_flush 0 (the old walk), 1, and the window replaced by air.
The open hole is the floor a correct frame scores; the verdict is flush
against the old walk, as a ratio.

AND THE COUNT IS NOT BELIEVED UNLESS THE FRAMES CARRY SIGNAL. The hole
frame must show the outdoors through the opening; a black frame is a
uniform frame, and a uniform frame has no bars in it.

Usage:
  python3 util/claude_glass_bars.py              # headless seat + run
  python3 util/claude_glass_bars.py --skip-seat  # a seat is already up
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_ci as ci          # noqa: E402
import claude_lab as lab        # noqa: E402

import numpy as np              # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

HUT = dict(x0=100, x1=112, z0=200, z1=206, y0=9, y1=13)
WINDOWS = {
    "pane": {"node": "mcl_panes:pane_natural", "x": (102, 104),
             "aim": {"pos": [103, 9.5, 202.2], "yaw": 180, "pitch": 12,
                     "time": 0.5}},
    "glass-block": {"node": "mcl_core:glass", "x": (108, 110),
                    "aim": {"pos": [109, 9.5, 202.2], "yaw": 180,
                            "pitch": 12, "time": 0.5}},
}
WIN_Y = (10, 11)
PLANKS = "mcl_trees:wood_oak"
SETTLE = 1500
BAR_DARK = 60       # a pixel at least this bright (0-255) is "lit"
BAR_NEIGHBOURS = 10 # lit pixels in its 5x5 for a stray to count as a bar
# TUNED: pass if flush leaves at most this fraction of the old walk's bar
# pixels | learn by: John's eyes on the saved frames, and the edge-shift
# floor measured here (both frames differ at the opening's edge anyway).
MAX_RATIO = 0.10
# live signal: through the opening, the hole frame must show daylight
MIN_OPENING_MEAN = 80.0


def P(x, y, z):
    return {"x": x, "y": y, "z": z}


def build_hut():
    h = HUT
    lab.rpc("fill", p1=P(h["x0"] - 1, h["y0"], h["z0"] - 1),
            p2=P(h["x1"] + 1, h["y1"] + 2, h["z1"] + 1), name="air")
    walls = (((h["x0"], h["y0"], h["z0"]), (h["x1"], h["y1"], h["z0"])),
             ((h["x0"], h["y0"], h["z1"]), (h["x1"], h["y1"], h["z1"])),
             ((h["x0"], h["y0"], h["z0"]), (h["x0"], h["y1"], h["z1"])),
             ((h["x1"], h["y0"], h["z0"]), (h["x1"], h["y1"], h["z1"])),
             ((h["x0"], h["y1"] + 1, h["z0"]), (h["x1"], h["y1"] + 1, h["z1"])))
    for a, b in walls:
        lab.rpc("fill", p1=P(*a), p2=P(*b), name=PLANKS)


def set_window(w, name):
    x0, x1 = w["x"]
    lab.rpc("fill", p1=P(x0, WIN_Y[0], HUT["z0"]),
            p2=P(x1, WIN_Y[1], HUT["z0"]), name=name)


def shoot(tag, aim, flush):
    dials = dict(ci.CANONICAL_DIALS)
    dials["claude_glass_flush"] = flush
    ci.push_dials(dials, "%s_%d" % (tag, time.time_ns()))
    lab.goto(aim)
    ci.await_frames(SETTLE)
    png = lab.shot(tag, settle=0.5, record=False)
    st = lab.read_stats() or {}
    return png, st


def gray(png):
    return np.asarray(Image.open(png).convert("L"), dtype=float)


def bars(png):
    """Bright pixels that are NOT part of the window view, i.e. not
    connected to the bright region at the centre of the frame (the aims
    look straight out of the window), and that sit among other bright
    pixels (>= BAR_NEIGHBOURS in their 5x5), so a lone noise speck is not
    a bar. Measured per frame, so refraction — which moves and magnifies
    the window view, a metre of glass more than a pane — cannot be scored
    as bars the way an A/B against the open hole scored it (first version
    of this probe: 122,585 "bars" on a frame with none visible)."""
    # 5x5 box blur first: path-traced grain puts dark specks inside the
    # sky and bright fireflies in the dark, which break the window view
    # into pieces and can seed the fill on a speck
    g = np.asarray(Image.open(png).convert("L").filter(ImageFilter.BoxBlur(2)),
                   dtype=float)
    bright = g >= BAR_DARK
    H, W = bright.shape
    bright[H - 14:, :14] = False              # the traced marker
    m = Image.fromarray((bright * 255).astype(np.uint8)).copy()  # writable
    # seed on the brightest pixel of the central region (the sky through
    # the window) — the exact centre can land on a tree, which is dark
    cy, cx = H // 4, W // 4
    sub = g[cy:H - cy, cx:W - cx]
    sy, sx = np.unravel_index(int(np.argmax(sub)), sub.shape)
    ImageDraw.floodfill(m, (int(sx) + cx, int(sy) + cy), 128)
    main = np.asarray(m) == 128
    stray = bright & ~main
    b = bright.astype(np.int32)
    c = np.pad(b, 2).cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    nb = (c[5:, 5:] - c[:-5, 5:] - c[5:, :-5] + c[:-5, :-5])
    return int((stray & (nb >= BAR_NEIGHBOURS)).sum())


def opening_mean(hole_png):
    h = gray(hole_png)
    H, W = h.shape
    return float(h[H // 3: 2 * H // 3, W // 3: 2 * W // 3].mean())


def start_seat():
    ci.stop_seat()
    ci.pin_conf()
    for tag, cmd, wait in (("server", ci.SERVER_CMD, ci.SEAT_BOOT_WAIT),
                           ("client", ci.HEADLESS_WRAP + ci.CLIENT_CMD, 0)):
        subprocess.Popen(cmd, cwd=ci.REPO,
                         stdout=open("/tmp/claude_glass_bars.%s.log" % tag, "wb"),
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
        time.sleep(wait)
    if not ci.wait_for_client():
        sys.exit("client never became ready")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-seat", action="store_true")
    args = ap.parse_args()
    if not args.skip_seat:
        start_seat()
    lab.rpc("time", set=0.5)
    lab.goto(WINDOWS["pane"]["aim"])
    time.sleep(3)          # let the hut's blocks load before building
    build_hut()
    out, ok = {}, True
    for name, w in WINDOWS.items():
        set_window(w, w["node"])
        time.sleep(2)
        frames = {}
        for flush in (0, 1):
            png, st = shoot("%s-flush%d" % (name, flush), w["aim"], flush)
            seen = st.get("claude_glass_flush")
            if seen is None or int(round(float(seen))) != flush:
                print("REFUSED: %s flush=%d but the client reports "
                      "claude_glass_flush=%r" % (name, flush, seen))
                return 2
            frames[flush] = png
        set_window(w, "air")
        time.sleep(2)
        hole, _ = shoot("%s-hole" % name, w["aim"], 1)
        set_window(w, w["node"])
        om = opening_mean(hole)
        if om < MIN_OPENING_MEAN:
            print("REFUSED: %s hole frame is not showing daylight through "
                  "the opening (mean %.1f < %.1f) — no signal, no verdict"
                  % (name, om, MIN_OPENING_MEAN))
            return 2
        b0, b1, bh = bars(frames[0]), bars(frames[1]), bars(hole)
        ratio = b1 / b0 if b0 else float("nan")
        good = b0 > 0 and ratio <= MAX_RATIO
        ok = ok and good
        out[name] = {"bars_flush0": b0, "bars_flush1": b1, "ratio": ratio,
                     "bars_hole": bh,
                     "opening_mean": om, "frames": {"flush0": frames[0],
                     "flush1": frames[1], "hole": hole}}
        print("%-12s bar pixels: old walk %7d  flush %7d  open hole %6d  "
              "ratio %.3f (max %.2f)  %s" % (name, b0, b1, bh, ratio,
                                             MAX_RATIO,
                                             "PASS" if good else "FAIL"))
    json.dump(out, open("/tmp/claude_glass_bars.json", "w"), indent=1)
    print("GREEN" if ok else "RED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
