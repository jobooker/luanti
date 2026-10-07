#!/usr/bin/env python3
"""claude_flicker — does a region's error jump up and down from frame to
frame? (the temporal artifact the video judge cannot see at its sampling
rate). Per 16x16 block, the mean of (real-time - reference) luminance over
the block; per-pixel noise averages out inside a block, so what is left
moving fast is flicker. Score: per block, the mean over time of
|d(t) - (d(t-1) + d(t+1))/2| (a temporal second difference), as a share of
the block's reference brightness; reported as the 99th percentile over
blocks and the share of blocks above 5%.

  python3 util/claude_flicker.py RT_DUMP REF_DUMP [--plant flicker]
"""
import json
import os
import sys

import numpy as np


def load(d):
    meta = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
    return [np.fromfile(os.path.join(d, "%05d.rgba" % m["i"]), dtype=np.uint8)
            .reshape(m["h"], m["w"], 4)[::-1, :, :3] for m in meta]


def lum(f):
    x = f.astype(np.float32) / 255.0
    x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return x @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def blocks(y, b=16):
    h, w = (y.shape[0] // b) * b, (y.shape[1] // b) * b
    return y[:h, :w].reshape(h // b, b, w // b, b).mean((1, 3))


def score(T, R, plant=None):
    n = min(len(T), len(R))
    D, B = [], []
    for i in range(n):
        t = T[i]
        if plant == "flicker" and int(n * 0.35) <= i < int(n * 0.65):
            t = t.astype(np.float32)
            t[t.shape[0] // 4:t.shape[0] // 2, t.shape[1] // 3:2 * t.shape[1] // 3] *= 1.25 if (i // 3) % 2 else 0.75
            t = np.clip(t, 0, 255).astype(np.uint8)
        lt, lr = lum(t), lum(R[i])
        D.append(blocks(lt - lr))
        B.append(blocks(lr))
    D, B = np.array(D), np.array(B)
    if (B.mean((1, 2)) < 1e-3).any():
        return {"refused": "black reference frames"}
    sd = np.abs(D[1:-1] - 0.5 * (D[:-2] + D[2:])).mean(0) / np.maximum(B.mean(0), 1e-3)
    return {"p99": float(np.percentile(sd, 99)), "share_over_5pct": float((sd > 0.05).mean())}


if __name__ == "__main__":
    plant = sys.argv[sys.argv.index("--plant") + 1] if "--plant" in sys.argv else None
    print(json.dumps(score(load(sys.argv[1]), load(sys.argv[2]), plant)))
