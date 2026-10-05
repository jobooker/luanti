#!/usr/bin/env python3
"""claude_denoise_play — what the PLAYER sees at a few frames: the play
defaults (NEE on, the direct/bounced split ramp) with the filter off and
on, and on with the split off. Pictures for eyes; no verdict.
Usage: python3 util/claude_denoise_play.py [--skip-seat] [--vantage V] [--n 16]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_ci as ci                       # noqa: E402
import claude_lab as lab                     # noqa: E402
import claude_denoise_price as price         # noqa: E402
from PIL import Image                        # noqa: E402

ARMS = [("raw-split", {"claude_denoise": 0, "claude_split": 48}),
        ("den-split", {"claude_denoise": 1, "claude_split": 48}),
        ("den-nosplit", {"claude_denoise": 1, "claude_split": 0})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-seat", action="store_true")
    ap.add_argument("--vantage", default="cozy-night-ci")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    if not args.skip_seat:
        price.start_seat()
    vs = lab.load_vantages()
    rundir = os.path.join(ci.REPO, "screenshots", "denoise-play",
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
    pngs = []
    for tag, extra in ARMS:
        dials = dict(ci.CANONICAL_DIALS, claude_nee=1, **extra)
        png, _ = ci.capture({"name": "%s-%s" % (v, tag)}, vs[v],
                            ci.park_for(v, vs), dials, rundir, args.n,
                            vantage_name=v)
        pngs.append(png)
    ims = [Image.open(p).convert("RGB").resize((960, 540)) for p in pngs]
    o = Image.new("RGB", (960 * len(ims), 540))
    for i, im in enumerate(ims):
        o.paste(im, (960 * i, 0))
    out = os.path.join(rundir, "%s-n%d.jpg" % (v, args.n))
    o.save(out, quality=88)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
