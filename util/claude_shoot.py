#!/usr/bin/env python3
"""claude_shoot — one frame-exact capture anywhere, for looking at.

Freezes time, applies CANONICAL_DIALS plus --dial overrides, stands the
player at --pos/--yaw/--pitch at --time, and fires the client's shutter
at exactly --frames frames. Prints the PNG path. No referee, no verdict.

  python3 util/claude_shoot.py --pos 146.5 9.5 123.5 --yaw 270 --pitch 4 \
      --time 0.235 --frames 1500 --dial claude_nee=1 --name forest
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_ci as ci                       # noqa: E402
import claude_lab as lab                     # noqa: E402
import claude_denoise_price as price         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", nargs=3, type=float, required=True)
    ap.add_argument("--yaw", type=float, required=True)
    ap.add_argument("--pitch", type=float, required=True)
    ap.add_argument("--time", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--dial", action="append", default=[],
                    help="name=value, repeatable")
    ap.add_argument("--name", default="shot")
    ap.add_argument("--skip-seat", action="store_true")
    # PIN THE POSE ON THE CLIENT (claude_path, one key): a server teleport
    # lands within the aim tolerance, not on the same pixel -- two
    # 4096-frame references of the plains differed by 2 px of yaw and a
    # sliver of height (judge validation, 2026-10-06), which a
    # reference-based judge reads as a big difference. Pinned, the pose is
    # the same float every frame of every shot.
    ap.add_argument("--pin", action="store_true")
    args = ap.parse_args()
    if not args.skip_seat:
        price.start_seat()
    fr = ci.do_freeze()
    if not fr.get("deepening"):
        print("REFUSED: time did not freeze: %r" % (fr,))
        return 2
    dials = dict(ci.CANONICAL_DIALS)
    for kv in args.dial:
        k, _, v = kv.partition("=")
        dials[k.strip()] = float(v)
    vant = {"pos": list(args.pos), "yaw": args.yaw, "pitch": args.pitch,
            "time": args.time}
    vs = lab.load_vantages()
    if args.pin:
        # the capture resets by teleporting away and back, which a pinned
        # camera never does: reset explicitly instead, after the arm's
        # dials are on (capture calls this right after pushing them), and
        # prove it with the client's reset counter
        def pinned_reset(vantage, park):
            before = (lab.read_stats() or {}).get("accum_resets")
            with open(lab.PATCH, "w") as f:
                f.write("claude_reset_accum = %d\n" % time.time_ns())
            t0 = time.time()
            while time.time() - t0 < 5:
                time.sleep(0.2)
                now = (lab.read_stats() or {}).get("accum_resets")
                if before is not None and now is not None and now > before:
                    return None
            return "pinned reset not seen (accum_resets %s)" % before
        ci.reset_accumulation = pinned_reset
        pf = os.path.join("/tmp", "claude_pin_%d.txt" % os.getpid())
        with open(pf, "w") as f:
            f.write("0 %r %r %r %r %r\n" % (args.pos[0], args.pos[1], args.pos[2],
                                           args.yaw, args.pitch))
        with open(lab.PATCH, "w") as f:
            f.write("claude_path = %s\n" % pf)
        time.sleep(1.5)
    rundir = os.path.join(ci.REPO, "screenshots", "shoot",
                          time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(rundir, exist_ok=True)
    png, cap = ci.capture({"name": args.name}, vant, vs["furnace-050"],
                          dials, rundir, args.frames, vantage_name=args.name)
    if args.pin:
        with open(lab.PATCH, "w") as f:
            f.write("claude_path = 0\n")
    if cap.get("reset_error"):
        # an unproven reset means the frame may hold the previous shot
        print("REFUSED: %s" % cap["reset_error"])
        return 2
    print("frames", (cap.get("settle") or {}).get("still_frames"))
    print(png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
