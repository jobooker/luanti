#!/usr/bin/env python3
"""claude_ab — one dial, two values, same vantage: noise and agreement.

For each N in --ns, captures the vantage with the dial at A and at B and
reports RMS (display bytes, HUD cropped) against a reference: A at --ref
frames. Then B at --ref frames against that reference: the RMS and the
9x9-mean (systematic) part. Two ESTIMATORS of the same picture must agree
at depth up to noise; two different PICTURES will not.

  python3 util/claude_ab.py --vantage cozy-night-ci --dial claude_torch_nee \
      --a 0 --b 1 --ns 16 64 256 --ref 2000 --set claude_nee=1
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_ci as ci                       # noqa: E402
import claude_lab as lab                     # noqa: E402
import claude_denoise_price as price         # noqa: E402
import claude_denoise_fade as fade           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantage", required=True,
                    help="a claude_vantages.json name, or a label with --pos")
    ap.add_argument("--pos", nargs=3, type=float)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--time", type=float, default=0.5)
    ap.add_argument("--dial", required=True)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--b", type=float, required=True)
    ap.add_argument("--ns", nargs="*", type=int, default=[16, 64, 256])
    ap.add_argument("--ref", type=int, default=2000)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--skip-seat", action="store_true")
    args = ap.parse_args()
    if not args.skip_seat:
        price.start_seat()
    vs = lab.load_vantages()
    v = args.vantage
    if args.pos:
        vs[v] = {"pos": list(args.pos), "yaw": args.yaw, "pitch": args.pitch,
                 "time": args.time}
    rundir = os.path.join(ci.REPO, "screenshots", "ab", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(rundir, exist_ok=True)
    ci.set_doors(True)
    fr = ci.do_freeze()
    if not fr.get("deepening"):
        print("REFUSED: time did not freeze")
        return 2
    base = dict(ci.CANONICAL_DIALS)
    for kv in args.set:
        k, _, val = kv.partition("=")
        base[k] = float(val)

    def shot(tag, val, n):
        d = dict(base)
        d[args.dial] = val
        png, cap = ci.capture({"name": "%s-%s" % (v, tag)}, vs[v],
                              vs["furnace-050"], d, rundir, n, vantage_name=v)
        got = (cap.get("settle") or {}).get("still_frames")
        if got != n:
            print("WARNING: %s wanted %d frames, got %s" % (tag, n, got))
        return png

    ref = shot("ref-a", args.a, args.ref)
    for n in args.ns:
        pa = shot("a%d" % n, args.a, n)
        pb = shot("b%d" % n, args.b, n)
        ra = lab.rms_diff(pa, ref)["rms"]
        rb = lab.rms_diff(pb, ref)["rms"]
        print("N=%-5d RMS vs ref: %s=%g %.2f   %s=%g %.2f   (a/b)^2 = %.2f"
              % (n, args.dial, args.a, ra, args.dial, args.b, rb, (ra / rb) ** 2 if rb else 0),
              flush=True)
    pb = shot("ref-b", args.b, args.ref)
    d = fade.signed(pb, ref)
    db = fade.box(d)
    print("at %d frames, B vs A: rms %.3f  9x9-mean rms %.3f max %.2f  mean %.3f"
          % (args.ref, (d ** 2).mean() ** 0.5, (db ** 2).mean() ** 0.5,
             abs(db).max(), d.mean()))
    print(rundir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
