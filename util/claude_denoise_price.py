#!/usr/bin/env python3
"""claude_denoise_price — what does the denoiser buy, and what does it cost?

THE CONTRACT'S QUESTION (roadmap, "the denoiser pays rent like every
estimator: it earns its ms only if it replaces more sampling-ms than it
costs, measured, or it goes"). Per vantage:

  ref         claude_denoise 0, REF_FRAMES frames: the near-converged truth
  raw@N       claude_denoise 0, N frames
  den@N       claude_denoise 1, N frames  (same N, same samples: rng 2
              keys the noise on the frame index since reset, so raw@N and
              den@N average IDENTICAL samples and differ only by the filter)
  den@REF     claude_denoise 1, REF_FRAMES frames: what the filter does to
              a converged image, which by design should be ~nothing

and reports RMS against ref (display bytes, HUD cropped), the sample
multiplier (RMS_raw / RMS_den)^2 (RMS ~ 1/sqrt(N), so this is how many
times more frames raw needs to match the filter), and frame time both
ways. The reference is itself noisy at REF_FRAMES, which floors every
RMS; the multipliers are therefore UNDER-estimates at large N.

Usage:
  python3 util/claude_denoise_price.py              # headless seat + run
  python3 util/claude_denoise_price.py --skip-seat
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

VANTAGES = ["cozy-day-ci", "cozy-night-ci", "exterior-ci"]
NS = [4, 16, 64]
REF_FRAMES = 4000


def start_seat():
    import claude_gpu_lock   # one game on the GPU at a time, across worktrees
    claude_gpu_lock.hold("start_seat")
    ci.stop_seat()
    ci.pin_conf()
    # the bridge mod is assembled from util/ fragments; only claude_ci did
    # it, so a seat started here ran the last assembled copy (2026-10-10:
    # after the demo-mode merge, playtests failed "unknown op: freeze")
    ok, info = ci.assemble_bridge()
    if not ok:
        sys.exit("bridge assembly failed: %s" % info.get("stderr"))
    for tag, cmd, wait in (("server", ci.SERVER_CMD, ci.SEAT_BOOT_WAIT),
                           ("client", ci.HEADLESS_WRAP + ci.CLIENT_CMD, 0)):
        subprocess.Popen(cmd, cwd=ci.REPO,
                         stdout=open("/tmp/claude_denoise_price.%s.log" % tag, "wb"),
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=ci.seat_env())
        time.sleep(wait)
    if not ci.wait_for_client():
        sys.exit("client never became ready")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-seat", action="store_true")
    ap.add_argument("--vantages", nargs="*", default=VANTAGES)
    ap.add_argument("--ref", type=int, default=REF_FRAMES,
                    help="reference depth (a scene that clamps often needs less)")
    ap.add_argument("--nee", type=float, default=0.0,
                    help="claude_nee for every arm (0 = photo mode, as CI)")
    ap.add_argument("--den", type=float, default=1.0,
                    help="claude_denoise value for the filtered arms")
    args = ap.parse_args()
    if not args.skip_seat:
        start_seat()
    vs = lab.load_vantages()
    rundir = os.path.join(ci.REPO, "screenshots", "denoise-price",
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
    out = {}

    def shot(vname, tag, denoise, n):
        dials = dict(ci.CANONICAL_DIALS)
        dials["claude_denoise"] = denoise
        dials["claude_nee"] = args.nee
        name = "%s-%s" % (vname, tag)
        png, cap = ci.capture({"name": name}, vs[vname],
                              ci.park_for(vname, vs), dials, rundir, n,
                              vantage_name=vname)
        st = (cap.get("stats_at_shutter") or {})
        seen = (lab.read_stats() or {}).get("claude_denoise")
        frames = (cap.get("settle") or {}).get("still_frames")
        return {"png": png, "frames": frames, "denoise_seen": seen,
                "frame_ms": st.get("frame_ms_avg"),
                "pass_ms": st.get("pass_ms")}

    for vname in args.vantages:
        r = {"ref": shot(vname, "ref", 0, args.ref)}
        for n in NS:
            r["raw%d" % n] = shot(vname, "raw%d" % n, 0, n)
            r["den%d" % n] = shot(vname, "den%d" % n, args.den, n)
        r["denref"] = shot(vname, "denref", args.den, args.ref)
        for k, v in r.items():
            want = args.den if k.startswith("den") else 0
            if v["denoise_seen"] is None or abs(float(v["denoise_seen"]) - want) > 0.01:
                print("REFUSED: %s %s wanted claude_denoise=%g, client says %r"
                      % (vname, k, want, v["denoise_seen"]))
                return 2
        for k, v in r.items():
            pair = {"ref": "denref", "denref": "ref"}.get(k) or (
                    ("den" + k[3:]) if k.startswith("raw") else ("raw" + k[3:]))
            if v["frames"] != r[pair]["frames"]:
                print("REFUSED: %s %s averaged %s frames, its pair %s %s"
                      % (vname, k, v["frames"], pair, r[pair]["frames"]))
                return 2
        ref = r["ref"]["png"]
        rows = []
        for n in NS:
            a = lab.rms_diff(r["raw%d" % n]["png"], ref)["rms"]
            b = lab.rms_diff(r["den%d" % n]["png"], ref)["rms"]
            mult = (a / b) ** 2 if b > 0 else float("inf")
            rows.append({"n": n, "rms_raw": a, "rms_den": b, "mult": mult,
                         "frames_raw": r["raw%d" % n]["frames"],
                         "frames_den": r["den%d" % n]["frames"]})
            print("%-14s N=%-4d RMS raw %6.2f  denoised %6.2f  -> x%.1f samples"
                  % (vname, n, a, b, mult))
        conv = lab.rms_diff(r["denref"]["png"], ref)["rms"]
        print("%-14s at %d frames the filter moves the image by RMS %.3f "
              "(frame ms raw %s, denoised %s)"
              % (vname, args.ref, conv, r["ref"]["frame_ms"],
                 r["denref"]["frame_ms"]))
        out[vname] = {"rows": rows, "converged_rms": conv, "shots": r}
    json.dump(out, open(os.path.join(rundir, "price.json"), "w"), indent=1)
    print("written", os.path.join(rundir, "price.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
