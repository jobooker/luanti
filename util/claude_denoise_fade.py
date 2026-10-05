#!/usr/bin/env python3
"""claude_denoise_fade — does the denoiser switch itself off as the image
converges? Paired frames: claude_denoise 0 and 1 at the SAME frame count
(rng 2: identical samples), so their difference is exactly what the
filter changes at that depth. Reported raw (RMS) and after a 9x9 mean of
the signed difference, which averages removed noise away and keeps any
systematic shift (bias). A filter that converges to the truth shows the
9x9 figure falling as N grows.

Usage: python3 util/claude_denoise_fade.py [--skip-seat] [--vantage V] [--ns 250 1000 4000 16000]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_ci as ci                       # noqa: E402
import claude_lab as lab                     # noqa: E402
import claude_denoise_price as price         # noqa: E402
import numpy as np                           # noqa: E402
from PIL import Image                        # noqa: E402


def box(x, k=9):
    c = np.pad(x, ((k // 2 + 1, k // 2), (k // 2 + 1, k // 2)),
               mode="edge").cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def signed(a, b):
    ia = np.asarray(Image.open(a).convert("RGB"), dtype=float)
    ib = np.asarray(Image.open(b).convert("RGB"), dtype=float)
    h = ia.shape[0]
    return (ia - ib)[int(h * 0.12):int(h * 0.86)].mean(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-seat", action="store_true")
    ap.add_argument("--vantage", default="cozy-night-ci")
    ap.add_argument("--ns", nargs="*", type=int, default=[250, 1000, 4000, 16000])
    ap.add_argument("--den", type=float, default=1.0)
    ap.add_argument("--nee", type=float, default=0.0)
    args = ap.parse_args()
    if not args.skip_seat:
        price.start_seat()
    vs = lab.load_vantages()
    rundir = os.path.join(ci.REPO, "screenshots", "denoise-fade",
                          time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(rundir, exist_ok=True)
    ci.set_doors(True)
    # LAB RULE #1 (claude_ci.do_freeze): a moving sun resets the
    # accumulator, and a pair that did not average the same frames is not
    # a pair (2026-10-05: cozy-day-ci ref 3709 frames vs filtered 417)
    fr = ci.do_freeze()
    if not fr.get("deepening"):
        print("REFUSED: time did not freeze: %r" % (fr,))
        return 2
    v = args.vantage
    for n in args.ns:
        pngs = {}
        for d in (0, args.den):
            dials = dict(ci.CANONICAL_DIALS)
            dials["claude_denoise"] = d
            dials["claude_nee"] = args.nee
            png, cap = ci.capture({"name": "%s-n%d-d%g" % (v, n, d)}, vs[v],
                                  ci.park_for(v, vs), dials, rundir, n,
                                  vantage_name=v)
            got = (cap.get("settle") or {}).get("still_frames")
            pngs[d] = (png, got)
        if pngs[0][1] != pngs[args.den][1]:
            print("N=%-6d UNPAIRED (%s vs %s frames): no row" % (n, pngs[0][1], pngs[args.den][1]))
            continue
        d = signed(pngs[args.den][0], pngs[0][0])
        db = box(d)
        print("N=%-6d frames %s/%s  filter change: rms %.3f  9x9-mean rms %.3f "
              "max %.2f" % (n, pngs[0][1], pngs[args.den][1],
                            (d ** 2).mean() ** 0.5, (db ** 2).mean() ** 0.5,
                            np.abs(db).max()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
