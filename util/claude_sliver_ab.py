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
A +Y (up) normal inside a -X (west-facing) wall LOOKS like impossible
geometry and is not: every building block in this world carries baked
16^3 relief, and the floor of a groove is an up-facing face inside a
west-facing wall. See the note on view 1 below.

FIXED 2026-08-17 (one-tracer b945042ee), in the BAKE, not the renderer:
    view 2 albedo  descend 1 -> mean 70.000  max  70   -- 3096 px -> 0
The masks themselves carried tunnels that crossed a node seam; every air
cell of a full-solid model now sits in exactly one face's carve shell,
which makes the node opaque to any straight ray. Goldens re-pinned to
run 20260818-034432_b945042ee, CI GREEN 116/116.

Why this rather than screenshots of the photo view:
  * the CROSSHAIR is drawn over the defect and claude_show_hud = 0 does
    NOT remove it, so any centre pixel count measures the crosshair;
  * the on-screen debug text changes every frame and swamps a naive
    full-frame diff (measured: 4,972 differing px, vs 12 for the body);
  * the photo view averages the leak into the wall and is UNRELIABLE at
    this depth -- one run read 75 differing px, a second read 0.

VIEW 2 IS THE ARM; VIEW 1 IS NOT, and the numbers above say why if you
read them twice: the normal ladder's descend = 0 reference is 108.520
MEAN against a 115 MAX, i.e. it was never uniform, and the probe's own
guard says so. It cannot be uniform -- claude_descend = 1 legitimately
turns a groove's floor into a +Y face inside a -X wall, because every
building block in this world carries baked relief. ALBEDO is the honest
channel: it is read from the CELL, so it can only change when the hit
lands in a DIFFERENT cell, which is exactly the leak and nothing else.

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
# The client's own screenshot is the 1920x1080 render target rather than
# the window's Retina backing store, and its hotbar starts at row 861 --
# so the window CROP's bottom edge clips into it and the reference stops
# being uniform. Same wall, 25 rows shorter. Kept as a SECOND constant
# rather than by shrinking the first, so every number this probe reported
# through the window path is still reproducible exactly as it was taken.
CROP_CLIENT = (300, 855, 200, 1700)


def window_id():
    # /usr/bin/python3 on purpose: Quartz (pyobjc) ships with macOS's
    # own python and is absent from a homebrew one, so `python3
    # claude_sliver_ab.py` printed "no Luanti window found" on a seat
    # that was up (2026-08-17). numpy and PIL are present in both.
    out = subprocess.run(["/usr/bin/python3", "-c", """
import Quartz
wl=Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll
    |Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID)
for w in wl:
    if 'Luanti' in (w.get('kCGWindowName') or ''):
        print(w.get('kCGWindowNumber')); break
"""], capture_output=True, text=True)
    return out.stdout.strip()


# HOW THE FRAME GETS HERE, and why there are two ways.
#
# `screencapture -l <window>` needs macOS SCREEN RECORDING permission,
# which a non-interactive session does not have -- it returns a fully
# black image and rc 1, which is indistinguishable from a black frame if
# nobody checks (2026-08-18: it did exactly that). The client can take
# its own screenshot through the same channel claude_ci uses, needs no
# permission at all, and is the same pixels.
#
# The window path stays the default so every number this probe has ever
# reported is still reproducible the way it was taken. THE BLINDNESS
# GUARD COVERS THE DIFFERENCE: the two paths frame slightly differently
# (window capture is the Retina backing store, the client's is the render
# target), so if CROP lands somewhere that is not a flat wall, the
# descend = 0 reference stops being uniform and this probe refuses to
# report a number rather than reporting a wrong one.
def take_shot(wid, path):
    if wid:
        r = subprocess.run(["screencapture", "-x", "-o", "-l", wid, path],
                           capture_output=True)
        if r.returncode == 0:
            return "window"
    # a UNIQUE token per shot: the client ignores a repeat of the one it
    # already served, so a fixed token gives you the first frame twice.
    png = L.shot(token="sliver%d" % int(time.time() * 1000), settle=0.0,
                 record=False)
    import shutil
    shutil.copy2(png, path)
    return "client"


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
    how = take_shot(wid, path)
    return json.load(open(STATS)).get("still_frames", 0), how


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=700)
    ap.add_argument("--view", type=int, default=2, choices=(1, 2))
    a = ap.parse_args()
    import numpy as np
    from PIL import Image
    wid = window_id()   # empty is fine: take_shot() falls back to the client
    res = {}
    for tag, dsc in (("off", 0), ("on", 1)):
        p = "/tmp/sliver_%s.png" % tag
        n, how = arm(a.view, dsc, a.target, wid, p)
        y0, y1, x0, x1 = CROP if how == "window" else CROP_CLIENT
        img = np.asarray(Image.open(p).convert("L"), dtype=np.int16)[y0:y1, x0:x1]
        res[tag] = img
        print("descend=%d samples=%-5d mean=%7.3f max=%3d  (%s shot)"
              % (dsc, n, img.mean(), img.max(), how))
    ref = res["off"]
    # A BLACK FRAME IS UNIFORM, and the guard below would bless it.
    # `screencapture` without screen-recording rights returns all zeros;
    # so does a frame taken before the client has drawn. Either way the
    # reference passes "uniform", every |on - ref| is 0, and the probe
    # reports PASS on nothing at all -- the exact failure this file's
    # own docstring describes, one level further in. Refuse instead.
    # 2 is well below any real surface here (the flat wall reads 70) and
    # well above sensor-free zero, so this cannot misfire on live pixels.
    if ref.max() <= 2 and res["on"].max() <= 2:
        print("REFUSED: both frames are black (ref max %d, on max %d). "
              "Nothing was measured -- this is a blind capture, not a "
              "clean wall. Check screen-recording permission, or that "
              "the client had drawn a frame before the shot."
              % (ref.max(), res["on"].max()))
        return 2
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
