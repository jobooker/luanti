#!/usr/bin/env python3
"""claude_ml_guide_capture — data for the ML guide test (DECISIONS 0w,
"bounce direction": can a network fill the guide's tables better than
counting?). Per place: play settings, the guide on and every path teaching
(claude_guide_keep 1), the voxel grid exported once, then the guide's
read table dumped (sparse) once per ~second for N windows. One window = 8
frames of counting: the noisy input. The average of the others: the target.

  python3 util/claude_ml_guide_capture.py OUT_DIR [--windows 64]
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab  # noqa: E402

PLACES = {
    "doorway-room": (241.7, 8.5, 231.7, 315, -12, 0.5),
    "cozy": (5, 8.56, 2.5, 0, 15, 0.5),
    "cozy-night": (7, 9, 2, 55, -5, 0.0),
    "spawn-cabin": (0, 9.5, -12, 0, -5, 0.5),
    "forest": (146.5, 8.5, 123.5, 270, -5, 0.5),
    "plains": (5, 8.5, -25, 0, -12, 0.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    first = True
    for name, (x, y, z, yaw, pitch, t) in PLACES.items():
        if a.only and name not in a.only:
            continue
        d = os.path.abspath(os.path.join(a.out, name))
        os.makedirs(d, exist_ok=True)
        cmd = ["python3", os.path.join(HERE, "claude_shoot.py"), "--pin", "--pos", str(x), str(y), str(z),
               "--yaw", str(yaw), "--pitch", str(pitch), "--time", str(t), "--frames", "64", "--name", "mlguide",
               "--dial", "claude_nee=1", "--dial", "claude_guide=1", "--dial", "claude_guide_impl=2",
               "--dial", "claude_guide_keep=1", "--dial", "claude_rng_seed=7"]
        if not first:
            cmd.append("--skip-seat")
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip().splitlines()
        if first:
            lab.rpc("abm", on=False)
            time.sleep(60)
            out = subprocess.run(cmd + ["--skip-seat"], cwd=REPO, capture_output=True, text=True).stdout.strip().splitlines()
        first = False
        print(name, "shot:", out[-1] if out else "?", flush=True)
        open(lab.PATCH, "w").write("claude_export_grid = %s\n" % os.path.join(d, "grid.bin"))
        time.sleep(2.5)
        for w in range(a.windows):
            open(lab.PATCH, "w").write("claude_guide_dump = %s\n" % os.path.join(d, "w%03d.bin" % w))
            time.sleep(1.2)
        n = len([f for f in os.listdir(d) if f.startswith("w")])
        print(name, "windows dumped:", n, flush=True)
    lab.rpc("abm", on=True)
    open(lab.PATCH, "w").write("claude_guide_keep = 0.25\n")


if __name__ == "__main__":
    main()
