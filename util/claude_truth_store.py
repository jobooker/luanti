#!/usr/bin/env python3
"""claude_truth_store -- render a playtest's truth once, keep it, and CHECK it
instead of re-rendering it (2026-10-09).

John: "preserve the truth and reuse it. We do need a way to know when the
truth changes tho. Not every change invalidates the truth..anything that
helps our guessing or ML shouldn't..but changing the rendering rules overall
would? ... checks just a handful of frames, and then only regenerates truth
if the truth frames are changed".

The truth video of a camera path (each pose held `hold` frames) was the bulk
of a motion playtest's GPU time: 15-20 minutes per scenario, re-rendered for
every run. Now:

  * KEY = the INPUTS only: the scenario, its camera path (frame keys), hold,
    scale and any dials set on every run. The engine version is deliberately
    NOT in the key: whether the engine changed the light is measured, below.
    Display caches and guesses (history, denoisers, the ledger, ML) are off
    in the truth runs by construction, so they never reach it.
  * CHECK = a few poses from the moving part of the stored path, each led by
    its predecessors so the carried history resembles the stored video's,
    rendered at the same hold with a seed the stored truth never uses
    (independent noise; the renderer is deterministic per seed, so the same
    seed would pass an unchanged engine trivially). Compared with the stored
    frames: whole-pixel error, 8x8-block error (noise averages out, a change
    in the light does not) and overall brightness.
  * FLOOR = the same check, measured right after the truth is stored. A later
    check fails when its error exceeds CHECK_K x the floor.
  * history.jsonl in the store records every store, pass and fail with its
    numbers: the data the TUNED values below are to be learned from.
"""
import hashlib
import json
import os
import shutil
import time

import numpy as np

ROOT = os.path.expanduser("~/data/luanti-truth")
CHECK_FRAMES = 4    # TUNED: poses per check | learn by: smallest count that still fails every known rule change in history.jsonl
CHECK_LEAD = 3      # predecessors before each check pose | learn by: floor_blk against lead (does more lead lower the floor?)
CHECK_SEED = 101    # the stored truth renders at seed 0
CHECK_K = 1.5       # TUNED: fail above K x floor | learn by: history.jsonl passes on no-change vs fails on known changes
EPS = 1e-4          # linear units: below this an error is 8-bit rounding, not a change
BLOCK = 8
FPS_STEP = 1.1      # TUNED: paths are laid out at fps rounded to 10 % steps, so runs share a truth | learn by: playtest scores at fps x1.1 vs x1.0 (does a 5 % pace error move them?)


def quantize_fps(fps):
    return round(FPS_STEP ** round(np.log(fps) / np.log(FPS_STEP)), 3)


def key_for(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def lookup(key):
    m = os.path.join(ROOT, key, "manifest.json")
    if not os.path.exists(m):
        return None
    man = json.load(open(m))
    for k in ("reference", "faces"):
        if not os.path.isdir(man[k]):
            return None
    return man


def log(event):
    os.makedirs(ROOT, exist_ok=True)
    event = dict(event, t=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(os.path.join(ROOT, "history.jsonl"), "a") as f:
        f.write(json.dumps(event) + "\n")


def _rows(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
    rows.sort(key=lambda r: r["i"])
    return rows


def _frame(d, r):
    a = np.fromfile(os.path.join(d, "%05d.rgba" % r["i"]), np.uint8)
    x = a.reshape(r["h"], r["w"], 4)[::-1, :, :3].astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _pose(r):
    return (r["pos"][0], r["pos"][1], r["pos"][2], r["yaw"], r["pitch"])


def path_pose(keys, fr):
    """game.cpp claudePathPose, in the same float32 arithmetic: the exact
    pose of path frame `fr` (the dump's logged pose has 6 digits only)"""
    f32 = np.float32
    if fr <= keys[0][0]:
        return tuple(float(f32(v)) for v in keys[0][1:])
    for i in range(1, len(keys)):
        if fr <= keys[i][0]:
            a, b = [f32(v) for v in keys[i - 1]], [f32(v) for v in keys[i]]
            t = f32(f32(fr) - a[0]) / max(f32(1e-6), f32(b[0] - a[0]))
            dyaw = f32(np.fmod(f32(b[4] - a[4] + f32(540.0)), f32(360.0)) - f32(180.0))
            return tuple(float(v) for v in (a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t,
                                            a[3] + (b[3] - a[3]) * t, a[4] + dyaw * t,
                                            a[5] + (b[5] - a[5]) * t))
    return tuple(float(f32(v)) for v in keys[-1][1:])


def _near(p, q):
    return all(abs(a - b) < 1e-2 for a, b in zip(p, q))


def check_plan(ref_dir, keys):
    """the stored frames to check: evenly over the frames where the pose
    changed (a held pose keeps converging, so those frames are older than
    `hold`; the moving frames are each exactly `hold` frames old), each led
    by its CHECK_LEAD predecessors; poses exact from the path keys"""
    rows = _rows(ref_dir)
    moving = [i for i in range(1, len(rows)) if _pose(rows[i]) != _pose(rows[i - 1])] or [1]
    n = min(CHECK_FRAMES, len(moving))
    idxs = sorted({moving[int((j + 0.5) * len(moving) / n)] for j in range(n)})
    poses = []
    for i in idxs:
        for k in range(CHECK_LEAD + 1):
            r = rows[max(0, i - CHECK_LEAD + k)]
            p = path_pose(keys, r["path_frame"])
            if not _near(p, _pose(r)):
                raise RuntimeError("path_pose disagrees with the dump at frame %d: %r vs %r"
                                   % (r["i"], p, _pose(r)))
            poses.append(p)
    return idxs, poses


def write_check_path(poses, path):
    with open(path, "w") as f:
        for j, p in enumerate(poses):
            f.write("%d %r %r %r %r %r\n" % ((j,) + tuple(p)))


def _blocks(x):
    h, w = x.shape[0] // BLOCK * BLOCK, x.shape[1] // BLOCK * BLOCK
    return x[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK, 3).mean((1, 3))


def compare(check_dir, ref_dir, idxs):
    """per checked pose: the check's frame (last of its lead-in) against the
    stored frame; refuses frames with no signal (a black pair "agrees")"""
    crow, rrow = _rows(check_dir), _rows(ref_dir)
    out = []
    for c, i in enumerate(idxs):
        j = c * (CHECK_LEAD + 1) + CHECK_LEAD
        if j >= len(crow):
            return None
        a, b = _frame(check_dir, crow[j]), _frame(ref_dir, rrow[i])
        if not _near(_pose(crow[j]), _pose(rrow[i])):
            return None
        out.append({"frame": i, "px": float(np.sqrt(((a - b) ** 2).mean())),
                    "blk": float(np.sqrt(((_blocks(a) - _blocks(b)) ** 2).mean())),
                    "mean_ratio": float(a.mean() / max(b.mean(), 1e-9)),
                    "ref_mean": float(b.mean()), "still": [crow[j]["still_frames"], rrow[i]["still_frames"]]})
    return out


def verdict(stats, floor):
    """True = the truth stands. Every pose must be within K x its floor."""
    bad = []
    for s, f in zip(stats, floor):
        for m in ("px", "blk"):
            if s[m] > CHECK_K * f[m] + EPS:
                bad.append("frame %d %s %.5f > %.1f x floor %.5f" % (s["frame"], m, s[m], CHECK_K, f[m]))
    return not bad, bad


def store(key, spec, exposure, ref_dir, faces_dir, idxs, floor, extra=None):
    """move the dumps into the store (a rename: same disk) and write the
    manifest; an older entry under the key is removed first"""
    d = os.path.join(ROOT, key)
    if os.path.isdir(d):
        shutil.rmtree(os.path.join(ROOT, key))
    os.makedirs(d)
    ref = os.path.join(d, "reference")
    faces = os.path.join(d, "faces")
    shutil.move(ref_dir, ref)
    shutil.move(faces_dir, faces)
    man = dict(spec=spec, key=key, exposure=exposure, reference=ref, faces=faces,
               check_frames=idxs, floor=floor, stored=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               check=dict(frames=CHECK_FRAMES, lead=CHECK_LEAD, seed=CHECK_SEED, k=CHECK_K, block=BLOCK),
               **(extra or {}))
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    return man
