#!/usr/bin/env python3
"""Steps per pixel per frame, per view, from claude_view 21."""
import json, subprocess, sys, os
import numpy as np
from PIL import Image
REPO = os.path.expanduser("~/code/luanti")
V = {"cabin (lamp-lit interior)": ("cozy", None),
     "plains (sunlit, open)": (None, ["5", "8.5", "-25", "0", "-12", "0.5"]),
     "forest (leaves, grass)": (None, ["146.5", "8.5", "123.5", "270", "-5", "0.5"]),
     "torch room (one torch)": (None, ["241.7", "9", "231.7", "315", "-12", "0.5"])}
first = True
for name, (vant, pose) in V.items():
    if pose is None:
        continue
    cmd = ["python3", "util/claude_shoot.py", "--pin", "--pos", pose[0], pose[1], pose[2], "--yaw", pose[3],
           "--pitch", pose[4], "--time", pose[5], "--frames", "8", "--name", "steps",
           "--dial", "claude_view=21", "--dial", "claude_cascades=1"]
    if not first:
        cmd.append("--skip-seat")
    first = False
    for attempt in range(5):
        png = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip().splitlines()[-1]
        if png.endswith(".png"):
            break
        if "--skip-seat" not in cmd:
            cmd.append("--skip-seat")
        sys.path.insert(0, os.path.join(REPO, "util")); import claude_lab as lab; lab.rpc("abm", on=False)
        import time; time.sleep(20)
    a = np.asarray(Image.open(png).convert("RGB")).astype(float) / 255.0
    steps = a[..., 0] * 4096; rays = a[..., 1] * 64; spr = a[..., 2] * 256
    st = json.load(open(png.replace(".png", ".capture.json")))["stats"]
    print("%-28s steps/pixel mean %5.0f (median %4.0f, 99th pct %5.0f) | rays/pixel %4.1f | steps/ray %4.0f | trace %.2f ms"
          % (name, steps.mean(), np.median(steps), np.percentile(steps, 99), rays.mean(), spr.mean(),
             st["pass_ms"][2]), flush=True)
