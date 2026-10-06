#!/usr/bin/env python3
"""claude_judge — the perceptual judge: how far a render is from what the
player SHOULD see, in units a person feels, with the frame time beside it.

THE REFERENCE is the rule, not a taste: "physics-based realism, if the
real world were axis-aligned blocks" means the right picture is the
converged one -- the same view, the same light law, a parked camera and
thousands of frames. Every score here is a distance to that.

Two scores, both reference-based (research: luanti-docs
research/perceptual-judge.md):
  JOD  ColorVideoVDP (Mantiuk et al. 2024). 10 = no visible difference;
       a 1-JOD gap = ~75 % of people prefer the better one. Stills AND
       video (it models the eye's transient channel, so flicker and noise
       crawl count). Needs a display model: util/judge/display_models.json
       (John's 27-inch QD-OLED; distance, peak and ambient are ASSUMED).
  FLIP NVIDIA LDR-FLIP mean error, 0 = identical; its error MAP says where.
       Pixels per degree from the same display model.
plus a signed brightness ratio (linear-light mean, test / reference): bias
is a direction, which a perceptual magnitude hides.

Performance rides along: every motion report carries the client's own
frame intervals (p50 / p99 / max, frames over 2x the median).

The judge must earn trust before it drives anything: claude_judge_validate.py
runs the known-answer ladders.

  python3 util/claude_judge.py still TEST.png REF.png [--map out.png]
  python3 util/claude_judge.py motion TEST_DUMP REF_DIR   (REF_DIR: dump or PNGs)
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DISPLAY = os.environ.get("CLAUDE_JUDGE_DISPLAY", "john_272up_1080p")
DISPLAY_CFG = os.path.join(HERE, "judge")
_cvvdp = {}


def display():
    d = json.load(open(os.path.join(DISPLAY_CFG, "display_models.json")))[DISPLAY]
    return d


def ppd(width_px=None):
    """pixels per degree at the screen centre for the display model; a
    frame narrower than the display (a half-res dump) is shown scaled up,
    so its pixels are proportionally bigger."""
    d = display()
    w, h = d["resolution"]
    diag_m = d["diagonal_size_inches"] * 0.0254
    width_m = diag_m * w / math.hypot(w, h)
    px_m = width_m / w
    p = 1.0 / math.degrees(2 * math.atan(px_m / 2 / d["viewing_distance_meters"]))
    if width_px:
        p *= width_px / w
    return p


def metric():
    import pycvvdp
    if "m" not in _cvvdp:
        _cvvdp["m"] = pycvvdp.cvvdp(display_name=DISPLAY,
                                    config_paths=[DISPLAY_CFG], quiet=True)
    return _cvvdp["m"]


def load_png(p):
    return np.asarray(Image.open(p).convert("RGB"))


def lin(x):
    x = x.astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def brightness_ratio(test, ref):
    return float(lin(test).mean() / max(lin(ref).mean(), 1e-9))


def fit(img, shape):
    """resize to the display resolution so the display model's geometry
    holds (a half-res dump is what the screen would show scaled up)"""
    if img.shape[:2] == tuple(shape):
        return img
    return np.asarray(Image.fromarray(img).resize((shape[1], shape[0]),
                                                  Image.BILINEAR))


def jod_still(test, ref):
    import torch
    q, _ = metric().predict(test, ref, dim_order="HWC")
    return float(q)


def flip_still(test, ref, map_out=None):
    import flip_evaluator as flip
    t = test.astype(np.float32) / 255.0
    r = ref.astype(np.float32) / 255.0
    emap, mean, _ = flip.evaluate(r, t, "LDR",
                                  parameters={"ppd": ppd(test.shape[1])})
    if map_out:
        e = (np.clip(emap, 0, 1) * 255).astype(np.uint8)
        if e.ndim == 3:
            e = e[..., 0]
        Image.fromarray(e).save(map_out)
    return float(mean)


def still(test, ref, map_out=None):
    """test, ref: HxWx3 uint8 sRGB. Returns the scorecard."""
    shape = display()["resolution"][::-1]
    t, r = fit(test, shape), fit(ref, shape)
    return {"jod": jod_still(t, r), "flip": flip_still(test, fit(ref, test.shape[:2]), map_out),
            "brightness": brightness_ratio(test, fit(ref, test.shape[:2]))}


def load_dump(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
    rows.sort(key=lambda r: r["i"])
    frames = []
    for r in rows:
        a = np.fromfile(os.path.join(d, "%05d.rgba" % r["i"]), np.uint8)
        frames.append(a.reshape(r["h"], r["w"], 4)[::-1, :, :3])
    return frames, rows


def load_frames(d):
    if os.path.exists(os.path.join(d, "meta.jsonl")):
        return load_dump(d)
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    return [load_png(f) for f in files], None


def pacing(rows):
    dts = sorted(r["dt_us"] / 1000.0 for r in rows[1:] if r["dt_us"] > 0)
    if not dts:
        return {}
    med = dts[len(dts) // 2]
    q = lambda p: dts[min(len(dts) - 1, int(p * (len(dts) - 1) + 0.5))]
    return {"frame_ms_p50": med, "frame_ms_p99": q(0.99), "frame_ms_max": dts[-1],
            "fps_mean": 1000.0 / (sum(dts) / len(dts)),
            "spikes_over_2x": sum(1 for v in dts if v > 2 * med)}


def jod_video(test_frames, ref_frames, fps, clip=60):
    """ColorVideoVDP over the video in clips of `clip` frames (a 1080p
    video of hundreds of frames does not fit the GPU at once); the clips'
    JODs are averaged, weighted by length."""
    import torch
    shape = display()["resolution"][::-1]
    n = min(len(test_frames), len(ref_frames))
    tot, w = 0.0, 0
    for a in range(0, n, clip):
        b = min(n, a + clip)
        if b - a < 8:
            break
        T = np.stack([fit(f, shape) for f in test_frames[a:b]])
        R = np.stack([fit(f, shape) for f in ref_frames[a:b]])
        q, _ = metric().predict(T, R, dim_order="FHWC", frames_per_second=fps)
        tot += float(q) * (b - a)
        w += b - a
        del T, R
        torch.cuda.empty_cache()
    return tot / max(w, 1)


def motion(test_dir, ref_dir, fps=60.0, idx=None):
    """test: a dump; ref: frames at the same path frames (a dump of the
    same length, or PNGs named by path frame). idx: which test frames to
    judge (default: all that have a reference)."""
    T, rows = load_dump(test_dir)
    R, rrows = load_frames(ref_dir)
    if idx is None:
        idx = list(range(min(len(T), len(R))))
    out = {"jod_video": jod_video([T[i] for i in idx], [R[i] for i in idx], fps),
           "frames": len(idx)}
    out.update(pacing(rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("still")
    a.add_argument("test")
    a.add_argument("ref")
    a.add_argument("--map")
    b = sp.add_parser("motion")
    b.add_argument("test")
    b.add_argument("ref")
    b.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()
    if args.cmd == "still":
        r = still(load_png(args.test), load_png(args.ref), args.map)
    else:
        r = motion(args.test, args.ref, args.fps)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    sys.exit(main())
