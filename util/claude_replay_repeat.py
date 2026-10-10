#!/usr/bin/env python3
"""claude_replay_repeat -- is a replay a function of its inputs? Pixel RMS between runs.

Takes playtest run folders (each made with identical arms, e.g. --arm a:
--arm b:) and compares the dumps of every pair of arm runs, frame by frame,
in LINEAR light (RMS of the sRGB-decoded difference, 0..1 scale):
  within a session : arm a vs arm b of the same run (same seat, same world)
  across sessions  : the same arm in two runs (fresh seat, fresh world copy)
Per pair: median and worst frame, frames compared, and whether the camera
poses match frame for frame. Also each arm's scene identity
(claude_lab.SCENE_ID) and the run's world (demo mode or not).

  python3 util/claude_replay_repeat.py RUN_DIR [RUN_DIR ...] [--json OUT]
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import claude_judge as J        # noqa: E402


def frame_rms(A, B):
    n = min(len(A), len(B))
    return [float(np.sqrt(((J.lin(A[i]) - J.lin(B[i])) ** 2).mean())) for i in range(n)]


def poses_match(ra, rb):
    keys = ("pos", "yaw", "pitch")
    n = min(len(ra), len(rb))
    return sum(1 for i in range(n) if all(ra[i].get(k) == rb[i].get(k) for k in keys)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    arms, ids, worlds = {}, {}, {}
    for r in a.runs:
        m = json.load(open(os.path.join(r, "meta.json")))
        tag = os.path.basename(r.rstrip("/"))
        worlds[tag] = m.get("world")
        for sc, v in m["scenarios"].items():
            for arm, d in v["arms"].items():
                arms[(tag, sc, arm)] = d
                ids[(tag, sc, arm)] = (v.get("scene_ids") or {}).get(arm)
    out = {"worlds": worlds, "pairs": [], "scene_ids": {"/".join(k): v for k, v in ids.items()}}
    cache = {}

    def load(k):
        if k not in cache:
            cache[k] = J.load_dump(arms[k])
        return cache[k]

    for sc in sorted({k[1] for k in arms}):
        keys = sorted(k for k in arms if k[1] == sc)
        for ka, kb in itertools.combinations(keys, 2):
            kind = "within" if ka[0] == kb[0] else ("across, same arm" if ka[2] == kb[2] else "across, other arm")
            (A, ra), (B, rb) = load(ka), load(kb)
            d = frame_rms(A, B)
            pm, n = poses_match(ra, rb)
            row = {"scenario": sc, "a": "/".join(ka), "b": "/".join(kb), "kind": kind,
                   "frames": [len(A), len(B)], "median": float(np.median(d)), "worst": float(max(d)),
                   "worst_frame": int(np.argmax(d)), "poses_equal": "%d/%d" % (pm, n),
                   "scene_id_equal": ids[ka] == ids[kb]}
            out["pairs"].append(row)
            print("%-12s %-18s %-40s median %.5f worst %.5f (frame %d) frames %s poses %s scene_id %s"
                  % (sc, kind, row["a"] + " ~ " + row["b"], row["median"], row["worst"], row["worst_frame"],
                     row["frames"], row["poses_equal"], "same" if row["scene_id_equal"] else "DIFFERENT"),
                  flush=True)
        # memory: drop this scenario's frames
        for k in keys:
            cache.pop(k, None)
    for k, v in sorted(out["scene_ids"].items()):
        print("scene_id %-40s %s" % (k, json.dumps(v)))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
