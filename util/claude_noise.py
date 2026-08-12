#!/usr/bin/env python
"""Temporal-noise profiler for the claude tracer — the sigma-budget.

The GPU profiler answers "where do the milliseconds go"; this answers
"where does the VARIANCE go". With the sim frozen and the camera still,
consecutive frames differ only by sampling noise the temporal
accumulation hasn't absorbed. Capture K consecutive screenshots via the
claude_settings_patch.conf hot-reload channel, stack them, and report
per-pixel temporal std-dev of luminance:

  mean sigma    — average shimmer across the frame (0-255 units)
  p95 sigma     — the noisy tail (edges, disocclusions, fresh caches)
  %>4           — fraction of pixels visibly boiling (sigma > 4/255)
  heatmap PNG   — BRIGHTNESS = sigma (grayscale; colorblind-safe)

Usage:
  claude_noise.py label [K]     capture K (default 8) shots, report
The HUD strip (bottom 22%) and chat area (top 8%) are cropped out.
Run with a python that has numpy+PIL (e.g. ~/.venvs/viz/bin/python).
"""
import sys, os, time, glob

import numpy as np
from PIL import Image

MTDIR = os.path.expanduser("~/Library/Application Support/minetest")
SHOTS = os.path.join(MTDIR, "screenshots")
PATCH = os.path.join(MTDIR, "claude_settings_patch.conf")
OUT = os.path.expanduser("~/Downloads/luanti-shots")


def newest():
    files = glob.glob(os.path.join(SHOTS, "*.png"))
    return max(files, key=os.path.getmtime) if files else None


def capture(tag, k):
    """Trigger k screenshots, one at a time, return file paths."""
    paths = []
    for i in range(k):
        before = newest()
        with open(PATCH, "w") as f:
            f.write("claude_screenshot = noise_%s_%d\n" % (tag, i))
        deadline = time.time() + 8.0
        while time.time() < deadline:
            time.sleep(0.25)
            after = newest()
            if after != before:
                paths.append(after)
                break
        else:
            raise RuntimeError("screenshot %d never appeared" % i)
        time.sleep(0.4)  # let the patch watcher rearm
    return paths


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    os.makedirs(OUT, exist_ok=True)
    paths = capture(tag, k)
    stack = []
    for p in paths:
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        stack.append(im)
    a = np.stack(stack)  # (k, H, W, 3)
    h = a.shape[1]
    a = a[:, int(h * 0.28):int(h * 0.78), :, :]  # crop chat + HUD
    lum = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    # per-frame gain normalization: the capture's own chat toasts nudge
    # auto-exposure, wobbling GLOBAL brightness between shots (observer
    # effect, caught 2026-08-12). Dividing each frame by its mean keeps
    # local shimmer — the thing we're measuring — and drops global gain.
    gains = lum.mean(axis=(1, 2), keepdims=True)
    lum = lum * (gains.mean() / gains)
    sigma = lum.std(axis=0)
    mean_s = float(sigma.mean())
    p95 = float(np.percentile(sigma, 95))
    boiling = float((sigma > 4.0).mean() * 100.0)
    heat = np.clip(sigma * 16.0, 0, 255).astype(np.uint8)  # x16 gain
    heatpath = os.path.join(OUT, "noise_%s_heatmap.png" % tag)
    Image.fromarray(heat).save(heatpath)
    print("noise[%s] over %d frames: mean sigma %.2f | p95 %.2f | "
          "%.1f%% pixels boiling (>4)" % (tag, k, mean_s, p95, boiling))
    print("heatmap: %s (brightness = sigma, gain x16)" % heatpath)


if __name__ == "__main__":
    main()
