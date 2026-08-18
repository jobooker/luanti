#!/usr/bin/env python3
"""The sub-voxel sliver probe: a THRESHOLD-FREE detector.

At the `leak-sliver` vantage the visible surface is one flat wall, so with
claude_descend = 0 the ALBEDO view (2) is a perfectly uniform field and the
NORMAL LADDER (1) is a single brightness step. Any pixel that deviates with
descend = 1 is the defect. No tolerance, no golden, no eyeballing.

MEASURED 2026-08-17 (M4 Air, one-tracer @ 475d84a92+):
    view 2 albedo  descend 0 -> mean 70.000  max  70   (uniform)
                   descend 1 -> mean 70.139  max 186
    view 1 normal  descend 0 -> mean 108.520 max 115   (all -X faces)
                   descend 1 -> mean 112.713 max 255   (+Y faces appear)
A +Y (up) normal inside a -X (west-facing) wall is the "impossible
geometry": the ray is passing through and hitting a horizontal surface
somewhere beyond.

Why this rather than screenshots of the photo view:
  * the CROSSHAIR is drawn over the defect and claude_show_hud = 0 does
    NOT remove it, so any centre pixel count measures the crosshair;
  * the on-screen debug text changes every frame and swamps a naive
    full-frame diff (measured: 4,972 differing px, vs 12 for the body);
  * the photo view averages the leak into the wall and is UNRELIABLE at
    this depth -- one run read 75 differing px, a second read 0.

Usage:  python3 util/claude_sliver_ab.py [--target N] [--view 2]
Requires a live look seat (util/claude_look.sh) and window id lookup.
"""
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_lab as L

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAL = os.path.join(REPO, "worlds/gallery/claude_dial.conf")
STATS = os.path.join(REPO, "claude_stats.json")
VANTS = os.path.join(REPO, "util/claude_vantages.json")
CROP = (300, 880, 200, 1700)   # y0,y1,x0,x1 -- excludes HUD text and hotbar


def window_id():
    out = subprocess.run([sys.executable, "-c", """
import Quartz
wl=Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll
    |Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID)
for w in wl:
    if 'Luanti' in (w.get('kCGWindowName') or ''):
        print(w.get('kCGWindowNumber')); break
"""], capture_output=True, text=True)
    return out.stdout.strip()


def arm(view, descend, target, wid, path):
    open(DIAL, "w").write(
        "fov = 72\nclaude_show_hud = 0\nclaude_show_chat = 0\n"
        "claude_view = %d\nclaude_bounces = 24\nclaude_descend = %d\n"
        % (view, descend))
    time.sleep(2.0)
    L.goto(json.load(open(VANTS))["leak-sliver"])   # teleport = accum reset
    t0 = time.time()
    while time.time() - t0 < 60:
        try:
            d = json.load(open(STATS))
        except Exception:
            time.sleep(0.4); continue
        # Match DEPTH, not wall-clock: the image is a 1/N running average,
        # so a rare event is diluted by depth even as its count grows.
        if (d.get("still_frames", 0) >= target
                and int(d.get("claude_descend", -1)) == descend
                and int(float(d.get("claude_view", -1))) == view):
            break
        time.sleep(0.4)
    subprocess.run(["screencapture", "-x", "-o", "-l", wid, path])
    return json.load(open(STATS)).get("still_frames", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=700)
    ap.add_argument("--view", type=int, default=2, choices=(1, 2))
    a = ap.parse_args()
    import numpy as np
    from PIL import Image
    wid = window_id()
    if not wid:
        sys.exit("no Luanti window found -- start util/claude_look.sh")
    y0, y1, x0, x1 = CROP
    res = {}
    for tag, dsc in (("off", 0), ("on", 1)):
        p = "/tmp/sliver_%s.png" % tag
        n = arm(a.view, dsc, a.target, wid, p)
        img = np.asarray(Image.open(p).convert("L"), dtype=np.int16)[y0:y1, x0:x1]
        res[tag] = img
        print("descend=%d samples=%-5d mean=%7.3f max=%3d"
              % (dsc, n, img.mean(), img.max()))
    ref = res["off"]
    if ref.max() != ref.min():
        print("WARNING: the descend=0 reference is NOT uniform (%d..%d). "
              "The vantage no longer shows a single flat wall, so this "
              "probe is BLIND -- re-aim it before believing any number."
              % (ref.min(), ref.max()))
    bad = np.abs(res["on"].astype(int) - int(ref.min())) > 2
    n = int(bad.sum())
    print("\nSLIVER PIXELS: %d" % n)
    if n:
        ys, xs = np.nonzero(bad)
        print("bbox x%d-%d y%d-%d" % (xs.min()+x0, xs.max()+x0,
                                      ys.min()+y0, ys.max()+y0))
    print("PASS" if n == 0 else "FAIL -- sub-voxel reads disagree with solid")
    return 0 if n == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
