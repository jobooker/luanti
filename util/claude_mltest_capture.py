#!/usr/bin/env python3
"""claude_mltest_capture — data for "can the blocks around a face predict
its bounced light?" (luanti-docs spec/ml-light-test-plan.md, step 1).

Per vantage, pose pinned on the client, ABMs off:
  bounce_a.png, bounce_b.png  claude_view 23 at FRAMES frames, twice
                              (independent: each shot resets explicitly),
                              for the split-half noise floor
  faces.png                   claude_view 22 (which cell and face)
  grid.bin                    claude_export_grid at that moment
Renderer only: nothing is scored here, so the GPU is not shared.

  python3 util/claude_mltest_capture.py [--frames 2048] [--only NAME ...]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab  # noqa: E402

OUT = os.path.join(REPO, "screenshots", "mltest")

# name: (x, y, z, yaw, pitch, time). Variety over volume: open sky, canopy,
# lamp-lit interiors, a lone torch, lava, skylights, the gallery rooms.
VANTAGES = {
    "plains": (5, 8.5, -25, 0, -12, 0.5),
    "plains-down": (5, 8.5, -25, 90, -30, 0.5),
    "overlook": (60, 44.5, 60, 45, -35, 0.5),
    "overlook-west": (60, 44.5, 60, 225, -25, 0.5),
    "forest": (146.5, 8.5, 123.5, 270, -5, 0.5),
    "forest-up": (146.5, 8.5, 123.5, 90, 25, 0.5),
    "treeline": (100, 12, 123.5, 270, -3, 0.5),
    "treeline-back": (100, 12, 123.5, 90, -10, 0.5),
    "torchroom": (241.7, 9, 231.7, 315, -12, 0.5),
    "torchroom-floor": (246.3, 11, 236.3, 135, -40, 0.5),
    "lavaroom": (253.7, 9, 231.7, 315, -25, 0.5),
    "lavaroom-wall": (258.3, 11, 236.3, 135, -20, 0.5),
}


def shoot(pose, frames, view, name, first):
    x, y, z, yaw, pitch, tm = pose
    cmd = ["python3", os.path.join(HERE, "claude_shoot.py"), "--pin",
           "--pos", str(x), str(y), str(z), "--yaw", str(yaw), "--pitch", str(pitch),
           "--time", str(tm), "--frames", str(frames), "--name", name,
           "--dial", "claude_view=%d" % view, "--dial", "claude_cascades=1",
           "--dial", "claude_nee=1", "--dial", "claude_denoise=0"]
    if not first:
        cmd.append("--skip-seat")
    for attempt in range(5):
        out = subprocess.run(cmd, cwd=REPO, capture_output=True,
                             text=True).stdout.strip().splitlines()
        png = out[-1] if out else ""
        if png.endswith(".png"):
            return png
        if "--skip-seat" not in cmd:
            cmd.append("--skip-seat")
        lab.rpc("abm", on=False)
        time.sleep(20)
    raise RuntimeError("shot refused 5 times: %s" % png[:200])


def export_grid(path):
    before = os.path.getmtime(path) if os.path.exists(path) else 0
    with open(lab.PATCH, "w") as f:
        f.write("claude_export_grid = %s\n" % path)
    t0 = time.time()
    while time.time() - t0 < 20:
        time.sleep(0.5)
        if os.path.exists(path) and os.path.getmtime(path) > before \
                and os.path.getsize(path) > 128 ** 3 * 4:
            return
    raise RuntimeError("grid export did not appear: %s" % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=2048)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    run = os.path.join(OUT, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run)
    first = True
    meta = {"frames": args.frames, "vantages": {}}
    for name, pose in VANTAGES.items():
        if args.only and name not in args.only:
            continue
        d = os.path.join(run, name)
        os.makedirs(d)
        for half in ("a", "b"):
            png = shoot(pose, args.frames, 23, "%s-bounce-%s" % (name, half), first)
            if first:
                lab.rpc("abm", on=False)
            first = False
            shutil.copy(png, os.path.join(d, "bounce_%s.png" % half))
            shutil.copy(png.replace(".png", ".capture.json"),
                        os.path.join(d, "bounce_%s.capture.json" % half))
        png = shoot(pose, 8, 22, name + "-faces", False)
        shutil.copy(png, os.path.join(d, "faces.png"))
        export_grid(os.path.join(d, "grid.bin"))
        meta["vantages"][name] = {"pose": pose}
        print(name, "ok", flush=True)
    lab.rpc("abm", on=True)
    with open(lab.PATCH, "w") as f:
        f.write("claude_view = 0\n")
    json.dump(meta, open(os.path.join(run, "meta.json"), "w"), indent=1)
    print(run)


if __name__ == "__main__":
    main()
