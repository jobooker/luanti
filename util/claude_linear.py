#!/usr/bin/env python3
"""claude_linear -- read the linear accumulated radiance (claude_accum_dump).

The key `claude_accum_dump = <prefix>` (settings patch) makes the client
write the NEXT frame's accumulated radiance -- the tracer's own float
buffer, before exposure, the display curve, the upsample and the 8-bit
quantisation -- to <prefix>.f32 (float32 RGBA) and <prefix>.json
({"w", "h", "format"}). See src/client/render/secondstage.cpp
ClaudeAccumReadback.

ORIENTATION, MEASURED rather than assumed (2026-10-08). GL stores rows
from the bottom, but the file's row 0 is already the TOP of the picture,
as in the PNG (Irrlicht's texture lock() hands back top-first rows).
Luminance correlation against the matching PNG, sky-furnace pad: 0.991
as written, 0.011 flipped. load() therefore does NOT flip; re-check with
`--check PREFIX IMAGE.png` whenever the readback path changes.

SIZE. The buffer is at trace resolution (960x540 for a 1920x1080
frame), so every referee box written in PNG pixels or fractions is
mapped through box_px() / box_frac() rather than re-drawn.

  python3 util/claude_linear.py --check PREFIX IMAGE.png   # orientation
"""
import json
import sys

import numpy as np


def load(prefix):
    """(HxWx3 float64 linear radiance, top row first; meta dict)."""
    m = json.load(open(prefix + ".json"))
    a = np.fromfile(prefix + ".f32", np.float32)
    if a.size != m["w"] * m["h"] * 4:
        raise ValueError("%s.f32 holds %d floats, %dx%dx4 expected"
                         % (prefix, a.size, m["w"], m["h"]))
    a = a.reshape(m["h"], m["w"], 4)[:, :, :3].astype(np.float64)
    return a, m


def box_px(img, x0, y0, x1, y1, fb=(1920, 1080)):
    """The slice of `img` covering the PIXEL box (x0,y0)-(x1,y1) drawn on a
    `fb`-sized frame (the PNG's), whatever the dump's own size."""
    h, w = img.shape[:2]
    sx, sy = w / float(fb[0]), h / float(fb[1])
    return img[int(round(y0 * sy)):int(round(y1 * sy)),
               int(round(x0 * sx)):int(round(x1 * sx))]


def box_frac(img, x0, y0, x1, y1):
    """The slice of `img` covering the FRACTION box (as claude_cornell_check
    draws its regions), using the same int() truncation it uses."""
    h, w = img.shape[:2]
    return img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def mean_se(patch):
    """Per-channel mean and its standard error (pixels taken as
    independent: the accumulation buffer is before the denoiser and the
    upsample, the two steps that correlate neighbours)."""
    p = patch.reshape(-1, 3)
    return p.mean(0), p.std(0) / np.sqrt(max(len(p), 1))


def check(prefix, png):
    """Which orientation of the dump matches the PNG? Correlation of
    luminance, PNG block-averaged to the dump's size."""
    from PIL import Image
    a, m = load(prefix)
    im = np.asarray(Image.open(png).convert("L"), dtype=np.float64)
    k = im.shape[0] // a.shape[0]
    im = im[:a.shape[0] * k, :a.shape[1] * k].reshape(a.shape[0], k, a.shape[1], k).mean((1, 3))
    lum = a @ np.array([0.2126, 0.7152, 0.0722])
    out = {}
    for tag, l in (("as loaded (row 0 = top)", lum),
                   ("flipped (row 0 = bottom)", lum[::-1])):
        out[tag] = float(np.corrcoef(np.log1p(l.ravel()), im.ravel())[0, 1])
    return out


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--check":
        for k, v in check(sys.argv[2], sys.argv[3]).items():
            print("%-36s corr %.4f" % (k, v))
    else:
        print(__doc__)
