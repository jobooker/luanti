#!/usr/bin/env python3
"""claude_clip — dumped frames -> an MP4 a person (or Gemini) can watch.

  python3 util/claude_clip.py OUT.mp4 DUMP_DIR [DUMP_DIR ...] [--fps 60]
        [--label LEFT --label RIGHT] [--max 600]

Several dump dirs are put side by side, frame i next to frame i (the
real-time arm and the reference arm of one claude_motion path share frame
indices). Frames are 8-bit display images, stored bottom-up by the GL
readback, so they are flipped here.
"""
import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw


def frames(d):
    meta = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
    return meta


def load(d, m):
    a = np.fromfile(os.path.join(d, "%05d.rgba" % m["i"]), dtype=np.uint8)
    return a.reshape(m["h"], m["w"], 4)[::-1, :, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--max", type=int, default=100000)
    ap.add_argument("--diff", action="store_true",
                    help="a third panel: |first - second| x 4, grey (black = identical)")
    a = ap.parse_args()
    metas = [frames(d) for d in a.dirs]
    n = min(min(len(m) for m in metas), a.max)
    h, w = metas[0][0]["h"], metas[0][0]["w"]
    panels = len(a.dirs) + (1 if a.diff else 0)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", "%dx%d" % (w * panels, h), "-r", str(a.fps), "-i", "-",
                          "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", a.out],
                         stdin=subprocess.PIPE)
    for i in range(n):
        ims = [load(d, m[i]) for d, m in zip(a.dirs, metas)]
        if a.diff:
            dd = np.clip(np.abs(ims[0].astype(np.int16) - ims[1].astype(np.int16)).max(-1) * 4, 0, 255).astype(np.uint8)
            ims.append(np.repeat(dd[..., None], 3, -1))
        row = np.concatenate(ims, 1)
        if a.label:
            im = Image.fromarray(row)
            dr = ImageDraw.Draw(im)
            for k, t in enumerate(a.label):
                dr.text((k * w + 8, 6), t, fill=(255, 255, 255))
            row = np.asarray(im)
        p.stdin.write(np.ascontiguousarray(row).tobytes())
    p.stdin.close()
    p.wait()
    print(a.out, n, "frames")


if __name__ == "__main__":
    main()
