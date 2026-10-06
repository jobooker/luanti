#!/usr/bin/env python3
"""claude_judge_validate — the judge earns trust on KNOWN ANSWERS before
it drives anything (research: luanti-docs research/perceptual-judge.md).

  capture  shoot the ladders on the seat (renderer only; no scoring, so
           the GPU is not shared while frames are timed)
  score    score them with claude_judge (run in ~/.venvs/judge)

Ladders, and the answer each must give:
  frames   the same parked view at 4 .. 1024 frames vs a 4096-frame
           reference: quality rises at every step (Spearman = 1)
  floor    a second, independent 4096-frame reference vs the first: near
           10 JOD, near 0 FLIP. Also the blind-instrument check: a judge
           fed two identical broken frames would score 10 too, so the
           frames ladder must span a real range.
  bias     the reference scaled in linear light by 0.90/0.95/0.98/1.02/
           1.05/1.10: quality falls with |bias|, brightness reads it signed
  blur     the reference down/up-sampled x2/x4/x8: quality falls
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "screenshots", "judge-validate")
VANTAGES = {
    "plains": dict(pos=[5, 8.5, -25], yaw=0, pitch=-12, time=0.5),
    "forest": dict(pos=[146.5, 8.5, 123.5], yaw=270, pitch=-5, time=0.5),
}
FRAMES = [4, 16, 64, 256, 1024]
REF = 4096


def shoot(v, frames, name, first):
    cmd = ["python3", os.path.join(HERE, "claude_shoot.py"),
           "--pos"] + [str(x) for x in v["pos"]] + [
           "--yaw", str(v["yaw"]), "--pitch", str(v["pitch"]),
           "--time", str(v["time"]), "--frames", str(frames), "--name", name,
           "--dial", "claude_cascades=1", "--pin"]
    if not first:
        cmd.append("--skip-seat")
    # a shot is refused when the accumulator resets while the camera is
    # parked (the world reader still filling the far levels after a
    # teleport); try again, the far levels settle
    for attempt in range(4):
        out = subprocess.run(cmd, capture_output=True, text=True,
                             cwd=REPO).stdout.strip().splitlines()[-1]
        if out.endswith(".png"):
            return out
        if "--skip-seat" not in cmd:
            cmd.append("--skip-seat")
        time.sleep(10)
    return out


def capture():
    run = os.path.join(OUT, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run)
    shots = {}
    first = True
    for vn, v in VANTAGES.items():
        for f in FRAMES + [REF, REF]:
            key = "%s-%d" % (vn, f)
            if key in shots:
                key += "b"
            shots[key] = shoot(v, f, key, first)
            first = False
            print(key, shots[key], flush=True)
    json.dump(shots, open(os.path.join(run, "shots.json"), "w"), indent=1)
    print(run)


def spearman(a, b):
    import numpy as np
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def score(run):
    import numpy as np
    from PIL import Image
    import claude_judge as J
    shots = json.load(open(os.path.join(run, "shots.json")))
    res = {}
    for vn in VANTAGES:
        ref = J.load_png(shots["%s-%d" % (vn, REF)])
        rows = []
        for f in FRAMES:
            s = J.still(J.load_png(shots["%s-%d" % (vn, f)]), ref)
            rows.append((f, s))
            print("%-7s frames %5d  JOD %.2f  FLIP %.4f  bright %.4f"
                  % (vn, f, s["jod"], s["flip"], s["brightness"]), flush=True)
        fl = J.still(J.load_png(shots["%s-%db" % (vn, REF)]), ref)
        print("%-7s FLOOR (ref b vs ref)  JOD %.2f  FLIP %.4f  bright %.4f"
              % (vn, fl["jod"], fl["flip"], fl["brightness"]), flush=True)
        jods = [s["jod"] for _, s in rows]
        flips = [-s["flip"] for _, s in rows]
        res[vn] = {"frames": rows, "floor": fl,
                   "spearman_jod": spearman(FRAMES, jods),
                   "spearman_flip": spearman(FRAMES, flips)}
        # synthetic ladders on the reference
        L = J.lin(ref)
        bias = []
        for k in (0.90, 0.95, 0.98, 1.02, 1.05, 1.10):
            x = np.clip(L * k, 0, 1)
            srgb = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
            t = (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)
            s = J.still(t, ref)
            bias.append((k, s))
            print("%-7s bias x%.2f  JOD %.2f  FLIP %.4f  bright %.4f"
                  % (vn, k, s["jod"], s["flip"], s["brightness"]), flush=True)
        blur = []
        h, w = ref.shape[:2]
        for k in (2, 4, 8):
            t = np.asarray(Image.fromarray(ref).resize((w // k, h // k), Image.BILINEAR)
                           .resize((w, h), Image.BILINEAR))
            s = J.still(t, ref)
            blur.append((k, s))
            print("%-7s blur x%d  JOD %.2f  FLIP %.4f" % (vn, k, s["jod"], s["flip"]), flush=True)
        res[vn]["bias"] = bias
        res[vn]["blur"] = blur
        lo = [s["jod"] for k, s in bias if k < 1]
        hi = [s["jod"] for k, s in bias if k > 1]
        res[vn]["bias_monotone"] = lo == sorted(lo) and hi == sorted(hi, reverse=True)
        bj = [s["jod"] for _, s in blur]
        res[vn]["blur_monotone"] = bj == sorted(bj, reverse=True)
        print("%-7s VERDICT frames spearman JOD %.2f FLIP %.2f | bias monotone %s | blur monotone %s"
              % (vn, res[vn]["spearman_jod"], res[vn]["spearman_flip"],
                 res[vn]["bias_monotone"], res[vn]["blur_monotone"]), flush=True)
    json.dump(res, open(os.path.join(run, "score.json"), "w"), indent=1, default=str)


def recapture(run):
    p = os.path.join(run, "shots.json")
    shots = json.load(open(p))
    for key, v in shots.items():
        if v.endswith(".png"):
            continue
        vn, f = key.split("-")[0], int(key.split("-")[1].rstrip("b"))
        shots[key] = shoot(VANTAGES[vn], f, key, False)
        print(key, shots[key], flush=True)
    json.dump(shots, open(p, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture()
    elif sys.argv[1] == "recapture":
        recapture(sys.argv[2])
    else:
        sys.path.insert(0, HERE)
        score(sys.argv[2])
