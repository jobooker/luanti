#!/usr/bin/env python3
"""claude_waste_probe — roadmap 3d-0: how much of the path tracer's work
comes back empty (claude_view 27 / 28), per view, in play settings.

Per pixel per frame:
  bounces          bounce rays launched
  wasted bounces   bounces after the last one that added any light
                   (what better bounce directions could at most recover)
  light samples    light-sampler calls (area, sky, flame); per type, only
                   the calls that traced a shadow ray are counted
  empty samples    light samples that returned zero (blocked, facing away,
                   or sun down) — what better light choice could recover

  python3 util/claude_waste_probe.py
  python3 util/claude_waste_probe.py claude_bounces 4 24
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
from claude_steps_probe import shoot  # noqa: E402

VIEWS = {
    "plains (sunlit, open)": ["5", "8.5", "-25", "0", "-12", "0.5"],
    "forest (leaves, grass)": ["146.5", "8.5", "123.5", "270", "-5", "0.5"],
    "torch room (one torch)": ["241.7", "8.5", "231.7", "315", "-12", "0.5"],
    "overlook (horizon)": ["60", "44.5", "60", "45", "-6", "0.45"],
    "spawn cabin": ["0", "9.5", "-12", "0", "-5", "0.5"],
}
PLAY = {"claude_nee": 1, "claude_bounces": 24, "claude_far_levels": 3, "claude_denoise": 1,
        "claude_descend": 1, "claude_model_far": 1, "claude_torch_nee": 1, "claude_pyramid": 1}


def main():
    dial = sys.argv[1] if len(sys.argv) > 1 else None
    values = sys.argv[2:] if len(sys.argv) > 2 else [None]
    first = True
    res = {}
    for name, p in VIEWS.items():
        for v in values:
            d = dict(PLAY)
            if dial:
                d[dial] = v
            arrs = {}
            for view in (27, 28, 29):
                cmd = ["python3", "util/claude_shoot.py", "--pin", "--pos", p[0], p[1], p[2],
                       "--yaw", p[3], "--pitch", p[4], "--time", p[5], "--frames", "8",
                       "--name", "waste", "--dial", "claude_view=%d" % view]
                for k, val in d.items():
                    cmd += ["--dial", "%s=%s" % (k, val)]
                if not first:
                    cmd.append("--skip-seat")
                first = False
                png = shoot(cmd)
                if not png:
                    break
                arrs[view] = np.asarray(Image.open(png).convert("RGB")).astype(float) / 255.0
            if len(arrs) < 3:
                print("%-24s %s=%s REFUSED" % (name, dial or "", v or ""), flush=True)
                continue
            a, b, c = arrs[27], arrs[28], arrs[29]
            wasted, bounces = a[..., 0] * 16, a[..., 1] * 16
            nzero = (c * 16).sum(-1)
            ncalls = (b * 16).sum(-1)
            tcalls = [(b[..., i] * 16).mean() for i in range(3)]
            tzero = [(c[..., i] * 16).mean() for i in range(3)]
            r = {"bounces": bounces.mean(), "wasted": wasted.mean(),
                 "wasted_share": wasted.sum() / max(bounces.sum(), 1e-9),
                 "nee": ncalls.mean(), "nee_zero": nzero.mean(),
                 "nee_zero_share": nzero.sum() / max(ncalls.sum(), 1e-9),
                 "clip_bounces": float((a[..., 1] >= 0.999).mean()),
                 "by_type": {t: [tcalls[i], tzero[i]] for i, t in enumerate(("area", "sky", "flame"))}}
            res["%s|%s" % (name, v)] = r
            print("%-24s %s=%-3s bounces/px %5.2f  wasted %5.2f (%4.0f%%) | shadow rays/px %5.2f"
                  "  blocked %5.2f (%4.0f%%) | clipped %.1f%%"
                  % (name, dial or "", v or "", r["bounces"], r["wasted"], 100 * r["wasted_share"],
                     r["nee"], r["nee_zero"], 100 * r["nee_zero_share"], 100 * r["clip_bounces"]),
                  flush=True)
            print("%-24s     " % "" + "   ".join("%s traced %5.2f/px blocked %3.0f%%" % (t, tcalls[i], 100 * tzero[i] / max(tcalls[i], 1e-9))
                                     for i, t in enumerate(("area", "sky", "flame"))), flush=True)
    lab.rpc("abm", on=True)
    json.dump(res, open("/tmp/waste_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
