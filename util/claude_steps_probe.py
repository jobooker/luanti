#!/usr/bin/env python3
"""claude_steps_probe — walk steps and rays per pixel per frame
(claude_view 21), per view, for each value of a dial.

  python3 util/claude_steps_probe.py                 # as configured
  python3 util/claude_steps_probe.py claude_pyramid 0 1
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.expanduser("~/code/luanti")
sys.path.insert(0, os.path.join(REPO, "util"))
import claude_lab as lab  # noqa: E402

VIEWS = {
    "plains (sunlit, open)": ["5", "8.5", "-25", "0", "-12", "0.5"],
    "forest (leaves, grass)": ["146.5", "8.5", "123.5", "270", "-5", "0.5"],
    "torch room (one torch)": ["241.7", "8.5", "231.7", "315", "-12", "0.5"],
    "overlook (horizon)": ["60", "44.5", "60", "45", "-6", "0.45"],
}


def shoot(cmd):
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
    return None


def main():
    dial = sys.argv[1] if len(sys.argv) > 1 else None
    values = sys.argv[2:] if len(sys.argv) > 2 else [None]
    first = True
    for name, p in VIEWS.items():
        for v in values:
            cmd = ["python3", "util/claude_shoot.py", "--pin", "--pos", p[0], p[1], p[2],
                   "--yaw", p[3], "--pitch", p[4], "--time", p[5], "--frames", "8",
                   "--name", "steps", "--dial", "claude_view=21",
                   "--dial", "claude_cascades=1"]
            if dial:
                cmd += ["--dial", "%s=%s" % (dial, v)]
            if not first:
                cmd.append("--skip-seat")
            first = False
            png = shoot(cmd)
            if not png:
                print("%-26s %s=%s REFUSED" % (name, dial, v))
                continue
            a = np.asarray(Image.open(png).convert("RGB")).astype(float) / 255.0
            steps = a[..., 0] * 4096
            rays = a[..., 1] * 64
            spr = a[..., 2] * 256
            st = json.load(open(png.replace(".png", ".capture.json")))["stats"]
            print("%-26s %s=%-3s steps/pixel %5.0f (median %4.0f, 99th %5.0f) | rays/pixel %4.1f"
                  " | steps/ray %4.0f | trace %.2f ms"
                  % (name, dial or "", v or "", steps.mean(), np.median(steps),
                     np.percentile(steps, 99), rays.mean(), spr.mean(), st["pass_ms"][2]),
                  flush=True)
    lab.rpc("abm", on=True)


if __name__ == "__main__":
    main()
