#!/usr/bin/env python3
"""claude_clip_plant — calibration clips for the video judge: a clean
reference walk with ONE known defect drawn on it (only the damage is
synthetic; the footage is real). Left panel = the damaged copy, right =
the untouched reference, so the judge is asked about the left.

  python3 util/claude_clip_plant.py REF_DUMP_DIR OUT_DIR
"""
import json
import os
import subprocess
import sys

import numpy as np

ref, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
meta = [json.loads(l) for l in open(os.path.join(ref, "meta.jsonl"))]
H, W = meta[0]["h"], meta[0]["w"]
F = [np.fromfile(os.path.join(ref, "%05d.rgba" % m["i"]), dtype=np.uint8).reshape(H, W, 4)[::-1, :, :3]
     for m in meta]
n = len(F)
fps = 60.0
rng = np.random.default_rng(5)
t0, t1 = int(n * 0.35), int(n * 0.65)          # the defect window, in frames


def plant(kind, i, f):
    g = f.astype(np.float32)
    if not (t0 <= i < t1) and kind != "clean":
        return f
    if kind == "flicker":                       # a region pulses +-25% every 3 frames
        s = 1.25 if (i // 3) % 2 else 0.75
        g[H // 4:H // 2, W // 3:2 * W // 3] *= s
    elif kind == "ghost":                       # 45% of the frame from 12 frames ago
        g = 0.55 * g + 0.45 * F[max(i - 12, 0)].astype(np.float32)
    elif kind == "dark_patch":                  # lower-left quarter at 40% brightness
        g[H // 2:, :W // 2] *= 0.4
    elif kind == "speckles":                    # 400 isolated bright dots per frame
        ys, xs = rng.integers(0, H, 400), rng.integers(0, W, 400)
        g[ys, xs] = 255.0
    return np.clip(g, 0, 255).astype(np.uint8)


for kind in ("clean", "flicker", "ghost", "dark_patch", "speckles"):
    for swap in (0, 1):
        path = os.path.join(out, "plant3-%s%s.mp4" % (kind, "-swapped" if swap else ""))
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", "%dx%d" % (3 * W, H), "-r", str(fps), "-i", "-", "-c:v", "libx264",
                              "-crf", "16", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        for i, f in enumerate(F):
            d = plant(kind, i, f)
            pair = [d, f][::(-1 if swap else 1)]
            # THIRD PANEL: |real-time - reference| x 4, grey; black = identical
            diff = np.clip(np.abs(d.astype(np.int16) - f.astype(np.int16)).max(-1) * 4, 0, 255).astype(np.uint8)
            pair.append(np.repeat(diff[..., None], 3, -1))
            p.stdin.write(np.ascontiguousarray(np.concatenate(pair, 1)).tobytes())
        p.stdin.close()
        p.wait()
json.dump({"window_s": [t0 / fps, t1 / fps], "frames": n}, open(os.path.join(out, "plant.json"), "w"))
print("planted window %.2f-%.2f s of %.2f s" % (t0 / fps, t1 / fps, n / fps))
