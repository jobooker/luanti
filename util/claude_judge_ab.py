#!/usr/bin/env python3
"""claude_judge_ab — one change, judged: quality AND speed, every arm,
every view, against the truth.

  capture: each arm shot at each view, pose pinned, N frames (renderer
           only; the GPU is not shared with scoring while it is timed).
           Frame time comes from the client's own stats in the capture.
  score:   (~/.venvs/judge) each arm vs the reference arm: JOD, FLIP,
           brightness; plus a side-by-side strip and FLIP maps.

  python3 util/claude_judge_ab.py capture SPEC.json
  ~/.venvs/judge/bin/python util/claude_judge_ab.py score RUN_DIR

SPEC.json:
  {"name": "far-leaves", "frames": 2048,
   "views": {"forest": {"pos": [x,y,z], "yaw": 270, "pitch": -5, "time": 0.5}},
   "reference": "near",
   "arms": {"near": {"claude_cascades": 1},
            "solid": {"claude_cascades": 1, "claude_far_only": 1},
            "medium": {"claude_cascades": 1, "claude_far_only": 1,
                       "claude_far_leaf_medium": 1}}}
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "screenshots", "judge-ab")


def shoot(v, frames, dials, name, first):
    cmd = ["python3", os.path.join(HERE, "claude_shoot.py"), "--pin",
           "--pos"] + [str(x) for x in v["pos"]] + [
           "--yaw", str(v["yaw"]), "--pitch", str(v["pitch"]),
           "--time", str(v.get("time", 0.5)), "--frames", str(frames),
           "--name", name]
    for k, val in dials.items():
        cmd += ["--dial", "%s=%s" % (k, val)]
    if not first:
        cmd.append("--skip-seat")
    for attempt in range(4):
        out = subprocess.run(cmd, capture_output=True, text=True,
                             cwd=REPO).stdout.strip().splitlines()
        out = out[-1] if out else ""
        if out.endswith(".png"):
            return out
        if "--skip-seat" not in cmd:
            cmd.append("--skip-seat")
        time.sleep(10)
    return out


def wait_reader(limit=900):
    """after a seat start the world-file reader fills the far levels for
    a minute or two, and every level it changes resets the accumulator, so
    a parked shot cannot freeze (2026-10-06: an arm refused four times).
    Wait until the count of blocks it folded stops moving."""
    sys.path.insert(0, HERE)
    import claude_lab as lab
    prev, same, t0 = None, 0, time.time()
    while time.time() - t0 < limit and same < 3:
        time.sleep(10)
        n = (lab.read_stats() or {}).get("far_db_blocks")
        same = same + 1 if n == prev else 0
        prev = n
    print("reader settled at %s blocks after %.0f s" % (prev, time.time() - t0), flush=True)


def capture(spec_path):
    spec = json.load(open(spec_path))
    run = os.path.join(OUT, "%s_%s" % (spec["name"], time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(run)
    shutil.copy(spec_path, os.path.join(run, "spec.json"))
    shots = {}
    first = not spec.get("skip_seat", False)
    # every arm's dials are spelled out in full: dials persist on the seat
    # an arm may set its own "_frames" (equal-frames noise tests against a
    # long reference); it is not a dial
    keys = sorted({k for a in spec["arms"].values() for k in a if k != "_frames"})
    for vn, v in spec["views"].items():
        for an, arm in spec["arms"].items():
            d = {k: arm.get(k, 0) for k in keys}
            nf = arm.get("_frames", spec.get("frames", 2048))
            png = shoot(v, nf, d, "%s-%s" % (vn, an), first)
            if first:
                # A STILL WORLD: the server's active block modifiers (grass
                # spreading, leaves decaying) re-send blocks near trees,
                # and every change resets the accumulator -- 2 resets per
                # 40 s at the treeline, 0 with them off (2026-10-06). CI
                # does the same; restored at the end.
                sys.path.insert(0, HERE)
                import claude_lab as lab
                lab.rpc("abm", on=False)
                wait_reader()
                # the first arm was shot while the reader was filling
                png = shoot(v, nf, d, "%s-%s" % (vn, an), False)
            shots["%s/%s" % (vn, an)] = png
            first = False
            print(vn, an, png, flush=True)
    json.dump(shots, open(os.path.join(run, "shots.json"), "w"), indent=1)
    try:
        import claude_lab as lab
        lab.rpc("abm", on=True)
    except Exception:
        pass
    print(run)


def score(run):
    import numpy as np
    from PIL import Image
    sys.path.insert(0, HERE)
    import claude_judge as J
    spec = json.load(open(os.path.join(run, "spec.json")))
    shots = json.load(open(os.path.join(run, "shots.json")))
    res = {}
    for vn in spec["views"]:
        ref = J.load_png(shots["%s/%s" % (vn, spec["reference"])])
        strip = [Image.fromarray(ref).resize((640, 360))]
        for an in spec["arms"]:
            png = shots["%s/%s" % (vn, an)]
            if not png.endswith(".png"):
                print("%-10s %-10s REFUSED: %s" % (vn, an, png[:100]))
                continue
            img = J.load_png(png)
            st = json.load(open(png.replace(".png", ".capture.json")))["stats"]
            # SAME SKY OR NO COMPARISON (2026-10-07, the weather): every arm of
            # a view must have been lit by the same sky and sun as the reference
            rst = json.load(open(shots["%s/%s" % (vn, spec["reference"])].replace(".png", ".capture.json")))["stats"]
            for key in ("sky_lux", "light_lum"):
                va, vr = st.get(key), rst.get(key)
                if isinstance(va, (int, float)) and isinstance(vr, (int, float)) and vr \
                        and abs(va / vr - 1) > 0.01:
                    print("    WARNING: %s differs from the reference arm's (%.4g vs %.4g): weather or "
                          "time changed between shots; this comparison is not valid" % (key, va, vr))
            m = os.path.join(run, "flip_%s_%s.png" % (vn, an))
            s = J.still(img, ref, m) if an != spec["reference"] else \
                {"jod": 10.0, "flip": 0.0, "brightness": 1.0}
            s.update(frame_ms=st.get("frame_ms_avg"), trace_ms=(st.get("pass_ms") or [0, 0, 0])[2])
            res["%s/%s" % (vn, an)] = s
            print("%-10s %-10s JOD %5.2f  FLIP %.4f  bright %.4f  | frame %.1f ms  trace %.2f ms"
                  % (vn, an, s["jod"], s["flip"], s["brightness"], s["frame_ms"] or 0,
                     s["trace_ms"] or 0), flush=True)
            if s.get("warning"):
                print("    WARNING:", s["warning"], flush=True)
            crop = spec["views"][vn].get("crop")
            if crop and an != spec["reference"]:
                # a band of the frame (rows y0..y1 of 1080), when the
                # question is about one part of the picture
                y0, y1 = crop
                c = J.still(img[y0:y1], ref[y0:y1])
                s["crop"] = c
                print("%-10s %-10s   crop rows %d-%d: JOD %5.2f  FLIP %.4f  bright %.4f"
                      % (vn, an, y0, y1, c["jod"], c["flip"], c["brightness"]), flush=True)
            if an != spec["reference"]:
                strip.append(Image.fromarray(img).resize((640, 360)))
        W = Image.new("RGB", (640 * len(strip), 360))
        for i, im in enumerate(strip):
            W.paste(im, (640 * i, 0))
        W.save(os.path.join(run, "strip_%s.jpg" % vn), quality=88)
    json.dump(res, open(os.path.join(run, "score.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2])
    else:
        score(sys.argv[2])
