#!/usr/bin/env python3
"""claude_far_energy — the LOD energy referee (roadmap 7a; physics-contract
§7: "geometry may be laddered with distance; the light law may not").

The same camera, the same samples (rng 2), twice: once as the world is
(the 1 m grid, with its 1/16 m rung), once with claude_far_only = 1 (the
1 m grid skipped, so the same terrain is seen through the 2 m far level).
Reported in LINEAR light (the display transform inverted at a fixed
manual exposure): the energy ratio of the whole frame and of regions.

AND IT MUST NOT PASS BLIND (measured.md:2196: a fold test once read
1.000047 while the shader ignored the finer geometry entirely). So the
two frames must also DIFFER: RMS between them above a floor, or the
verdict is REFUSED, not passed.

Usage: python3 util/claude_far_energy.py [--skip-seat] [--frames 1200]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_ci as ci                       # noqa: E402
import claude_lab as lab                     # noqa: E402
import claude_denoise_price as price         # noqa: E402
from claude_furnace_check import aces_inverse  # noqa: E402
import numpy as np                           # noqa: E402
from PIL import Image                        # noqa: E402

VIEWS = {
    "plains": {"pos": [5, 8.5, -25], "yaw": 0, "pitch": -12, "time": 0.5},
    "forest": {"pos": [146.5, 8.5, 123.5], "yaw": 270, "pitch": -5, "time": 0.5},
    "overlook": {"pos": [60, 44.5, 60], "yaw": 45, "pitch": -35, "time": 0.45},
}
EXPOSURE = 0.06       # daylight, below the shoulder; same for both arms
MIN_DIFF_RMS = 1.0    # 0..255: the two representations must differ
# TUNED: pass band on the whole-frame energy ratio | learn by: the spread
# of repeated runs, and what John's eye calls a visible seam
TOL = 0.05


def lin(png):
    a = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64) / 255.0
    h = a.shape[0]
    a = a[int(h * 0.12):int(h * 0.86)]
    clip = (a >= 0.995).any(axis=2)
    return aces_inverse(a ** 2.2) / EXPOSURE, clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-seat", action="store_true")
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--views", nargs="*", default=list(VIEWS))
    args = ap.parse_args()
    if not args.skip_seat:
        price.start_seat()
    fr = ci.do_freeze()
    if not fr.get("deepening"):
        print("REFUSED: time did not freeze")
        return 2
    vs = lab.load_vantages()
    rundir = os.path.join(ci.REPO, "screenshots", "far-energy",
                          time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(rundir, exist_ok=True)
    ok = True
    for name in args.views:
        v = VIEWS[name]
        pngs = {}
        for fo in (0, 1):
            d = dict(ci.CANONICAL_DIALS)
            d.update(claude_cascades=1, claude_far_levels=3, claude_far_only=fo,
                     claude_units=1, claude_nee=1, claude_auto_exposure=0,
                     claude_exposure=EXPOSURE)
            png, cap = ci.capture({"name": "%s-fo%d" % (name, fo)}, v,
                                  vs["furnace-050"], d, rundir, args.frames,
                                  vantage_name=name)
            pngs[fo] = png
            aim = (cap.get("aim_at_shutter") or {})
            if not aim.get("ok"):
                print("REFUSED: %s fo%d was not shot from the view (%s)"
                      % (name, fo, aim.get("detail")))
                return 2
            st = lab.read_stats() or {}
            if fo == 1 and st.get("far_levels", 0) < 1:
                print("REFUSED: %s far levels not built (%r)" % (name, st.get("far_levels")))
                return 2
        A, ca = lin(pngs[0])
        B, cb = lin(pngs[1])
        m = ~(ca | cb)
        drms = lab.rms_diff(pngs[0], pngs[1])["rms"]
        whole = B[m].mean() / A[m].mean()
        H, W = m.shape
        regs = []
        for (r0, r1, c0, c1) in ((0, H // 2, 0, W // 2), (0, H // 2, W // 2, W),
                                 (H // 2, H, 0, W // 2), (H // 2, H, W // 2, W)):
            mm = m[r0:r1, c0:c1]
            regs.append(B[r0:r1, c0:c1][mm].mean() / max(A[r0:r1, c0:c1][mm].mean(), 1e-12))
        if drms < MIN_DIFF_RMS:
            verdict = "REFUSED (the two frames do not differ: blind)"
            ok = False
        else:
            good = abs(whole - 1.0) <= TOL
            ok = ok and good
            verdict = "PASS" if good else "FAIL"
        print("%-9s 2m/1m energy %.4f  regions %s  frame diff RMS %.2f  clipped %.3f  %s"
              % (name, whole, " ".join("%.3f" % r for r in regs), drms,
                 1 - m.mean(), verdict), flush=True)
    print(rundir)
    print("GREEN" if ok else "RED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
