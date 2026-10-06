#!/usr/bin/env python3
"""claude_motion_validate — the motion judge on known answers.

  capture  (renderer) the reference video of a path (claude_path_hold:
           each pose held to HOLD frames) and the real-time test arms
  score    (~/.venvs/judge) ColorVideoVDP on the video, per arm, with the
           frame pacing beside it, plus the synthetic ladders:
             flicker  the test with every other frame brightened by
                      +2 / +5 / +10 % in linear light -> must score worse
                      with amplitude
             lag      the test shown 1 / 2 / 4 frames late against the
                      reference (a laggy picture trails the pose) -> worse
                      with delay
             self     the reference against itself -> 10 (the ceiling is
                      real, the instrument is not blind)
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "screenshots", "motion-validate")
PATH = os.path.join(HERE, "paths", "plains-walk.txt")
HOLD = 256
ARMS = {"raw": [], "denoise": ["--dial", "claude_denoise=1"]}


def motion(args, name):
    cmd = ["python3", os.path.join(HERE, "claude_motion.py"), "--path", PATH,
           "--scale", "1", "--name", name, "--dial", "claude_cascades=1",
           "--skip-seat"] + args
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                         timeout=7200).stdout
    print(out.strip().splitlines()[-2:], flush=True)
    return out.strip().splitlines()[-1]


def capture():
    run = os.path.join(OUT, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run)
    subprocess.run(["python3", os.path.join(HERE, "claude_shoot.py"), "--pos",
                    "5", "8.5", "-25", "--yaw", "0", "--pitch", "-12",
                    "--frames", "30", "--dial", "claude_cascades=1",
                    "--name", "warm"], cwd=REPO, capture_output=True)
    dumps = {"ref": motion(["--dial", "claude_path_hold=%d" % HOLD], "ref")}
    for arm, a in ARMS.items():
        dumps[arm] = motion(a, arm)
    json.dump(dumps, open(os.path.join(run, "dumps.json"), "w"), indent=1)
    print(run)


def score(run):
    import numpy as np
    import claude_judge as J
    dumps = json.load(open(os.path.join(run, "dumps.json")))
    R, _ = J.load_dump(dumps["ref"])
    res = {}

    def judge(name, T, rows=None):
        q = J.jod_video(T, R[:len(T)], 60.0)
        p = J.pacing(rows) if rows else {}
        res[name] = dict(jod=q, **p)
        print("%-14s JOD %.2f  %s" % (name, q, " ".join(
            "%s %.1f" % (k, v) for k, v in p.items())), flush=True)

    judge("self", R)
    for arm in ARMS:
        T, rows = J.load_dump(dumps[arm])
        judge(arm, T, rows)
        res[arm]["brightness"] = float(np.mean([J.brightness_ratio(t, r)
                                                for t, r in zip(T[::10], R[::10])]))
        print("%-14s brightness vs reference %.4f" % (arm, res[arm]["brightness"]))
    # KNOWN-ANSWER ARTIFACTS ON A CLEAN BASE (the reference itself): on a
    # noisy base they are masked, and a dark-biased base even "improves"
    # when brightened (first try, 2026-10-06: flicker raised raw's JOD)
    n = len(R)
    for amp in (0.02, 0.05, 0.10):
        F = []
        for i, f in enumerate(R):
            if i % 2:
                x = np.clip(J.lin(f) * (1 + amp), 0, 1)
                f = (np.clip(np.where(x <= 0.0031308, x * 12.92,
                                      1.055 * x ** (1 / 2.4) - 0.055), 0, 1)
                     * 255 + 0.5).astype(np.uint8)
            F.append(f)
        judge("ref flicker+%d%%" % int(amp * 100), F)
    for k in (1, 2, 4):
        judge("ref lag %d" % k, [R[max(0, i - k)] for i in range(n)])
    rng = np.random.default_rng(0)
    for sd in (2, 5, 10):
        judge("ref noise sd%d" % sd, [np.clip(f.astype(np.int16) + rng.normal(0, sd, f.shape),
                                              0, 255).astype(np.uint8) for f in R])
    json.dump(res, open(os.path.join(run, "score.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture()
    else:
        sys.path.insert(0, HERE)
        score(sys.argv[2])
