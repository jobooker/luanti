#!/usr/bin/env python3
"""A/B the SUB-VOXEL DESCENT dial (claude_descend) at a parked camera.

WHAT IS BEING MEASURED. Before 2026-08-16 the tracer's walk stopped at
the 1 m wall of every cell, including the cells that carry a finer 16^3
shape (class 250: stairs, slabs, beds, chests, the campfire). Those
shapes were baked and uploaded and never read. `claude_descend = 1`
makes march() step into such a cell and continue the same walk at 1/16 m
until it hits a sub-voxel or leaves the cell; `= 0` is the old
behaviour, byte-for-byte. That is the ONE variable this script moves.

WHY A DIAL AND NOT TWO BUILDS. The two arms then share a binary, a seat,
a world, a snapshot and a settle. The dial is shader-side only, so both
arms also share the GRID: `grid_hash` must come out identical, which is
asserted below. (The nodebox A/B before this one could not assert that —
its dial changed what the CPU baked.)

WHAT IT REPORTS, per arm and per rep:
  * mean luminance of the frame, and the on/off ratio.
    The energy gate. Note what it is NOT: this step is a geometry
    CORRECTION, not an LOD fold, so the ratio SHOULD move. A stair is
    three quarters of a cube and the cosy cabin's roof is 88 of them, so
    descending removes about a quarter of the roof's occluding volume
    and more light legitimately reaches the room. Read the direction and
    the size against that, not against 1.000.
  * 4x4 region means, so a change that is local to the roof line can be
    told apart from a change spread over the whole frame.
  * frame_ms_avg / busy_ms / trace pass_ms and still_frames at the
    shutter. The cost gate.
  * the fps-cap artifact check (claude_lab.cap_artifact): a capped
    client reads as a slow renderer and has twice been mistaken for one.

SETTLE. Accumulated frames, not wall clock — the same rule claude_ci
uses, for the same reason (60 s bought 3,928 frames one run and 7,491
another). Each rep teleports away and back and forces a snapshot first,
so the accumulator restarts from a hard reset rather than deepening a
previous arm's image into this one.

ARMS ALTERNATE (1,0,1,0,...) so a slow drift in the seat cannot be
mistaken for the dial.

Usage:  python3 util/claude_descend_ab.py [--vantage cozy-ci]
                                          [--reps 2] [--frames 2000]
The seat (server + client) must already be up. Prints JSON on stdout and
writes it to screenshots/descend_ab_<vantage>.json.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab  # noqa: E402

# THE PARK IS ANOTHER VANTAGE, never an offset from this one. A vantage
# is a position the player's body already RESTS at; an arbitrary offset
# lands them in a wall, in the air or in a lava flow, and the first
# attempt at this script killed the player outright (2026-08-16 —
# "You died" rendered over the capture and the frame mean went 42 -> 132).
# A physics slide after a bad landing also resets accumulation for up to
# a minute, which is the cozy-ci landmine claude_ci.park_for exists for.
PARK_VANTAGE = "furnace-050"
PARK_ALT = "cornell"


def stats():
    return json.load(open(lab.STATS))


def settle_to(frames, hard_max_s=240.0):
    """Wait until the accumulator is `frames` deep. Returns the stats
    sample the shutter is about to fire on."""
    t0 = time.time()
    last = None
    while time.time() - t0 < hard_max_s:
        try:
            last = stats()
        except Exception:
            time.sleep(0.5)
            continue
        if last.get("still_frames", 0) >= frames:
            return last, True
        time.sleep(0.5)
    return last, False


def region_means(png, rows=4, cols=4):
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64)
    lum = a @ np.array([0.2126, 0.7152, 0.0722])
    h, w = lum.shape
    out = []
    for r in range(rows):
        out.append([round(float(lum[r * h // rows:(r + 1) * h // rows,
                                    c * w // cols:(c + 1) * w // cols].mean()), 4)
                    for c in range(cols)])
    return round(float(lum.mean()), 4), out


def arm(vantage, descend, tag, frames, dial="claude_descend"):
    vs = lab.load_vantages()
    v = vs[vantage]
    # park at another REST position, then come back: game.cpp resets the
    # accumulator on a camera move, so the previous arm's converged image
    # cannot bleed into this one through the history buffer.
    away = vs[PARK_VANTAGE if vantage != PARK_VANTAGE else PARK_ALT]
    # dials BEFORE the reset: the dial channel is a ~1 Hz poll and the
    # accumulator is a true 1/N average with no floor, so an arm that
    # pushes its dials after the reset keeps M frames of the PREVIOUS
    # arm at exactly M/N weight, forever (claude_ci.capture's docstring).
    # THE SEAT MUST BE ALIVE AND ITS SCREEN CLEAR. A dead player's
    # "You died" dialog is client-side and survives a server-side heal,
    # a client restart and any number of teleports; it renders OVER the
    # capture while every stat reads healthy. Four arms of this script
    # were lost to it on 2026-08-16. revive() is idempotent, so it costs
    # one RPC per arm and closes the only failure the numbers cannot see.
    lab.rpc("revive", player="claude")
    lab.doorway(**{dial: descend})
    lab.goto(away)
    time.sleep(1.0)
    lab.goto(v)
    lab.doorway(claude_grid_snapshot="%s_d%d" % (tag, descend))
    time.sleep(2.0)
    st, ok = settle_to(frames)
    png = lab.shot("%s_d%d" % (tag, descend), settle=0.5)
    mean, regions = region_means(png)
    return {
        "dial": dial,
        "descend": descend,
        "png": png,
        "hp": (lab.rpc("players") or [{}])[0].get("hp"),
        "settled": ok,
        "still_frames": st.get("still_frames"),
        "grid_hash": st.get("grid_hash"),
        "grid_solid": st.get("grid_solid"),
        "accum_resets": st.get("accum_resets"),
        "fps": st.get("fps"),
        "frame_ms_avg": st.get("frame_ms_avg"),
        "busy_ms": st.get("busy_ms"),
        "draw_ms": st.get("draw_ms"),
        "pass_ms": st.get("pass_ms"),
        "cap_artifact": lab.cap_artifact(st, lab.fps_caps()),
        "mean_luma": mean,
        "regions": regions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantage", default="cozy-ci")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--frames", type=int, default=2000)
    # WHICH DIAL. Default claude_descend, which is what this script was
    # written for and what spec/measured.md "Walker descend" records.
    # claude_models is the second customer (2026-08-16): it empties the
    # authored-model table so the room renders as stock geometry only.
    #
    # The two differ in ONE important way and it is asserted, not
    # assumed. claude_descend is shader-side, so both its arms must see
    # the SAME baked grid and a second grid_hash means the world moved
    # under the measurement. claude_models changes what the CPU BAKES,
    # so its arms MUST see two hashes -- one hash there would mean the
    # dial did nothing. --grid says which answer is the pass.
    ap.add_argument("--dial", default="claude_descend")
    ap.add_argument("--grid", choices=("same", "moved"), default=None,
                    help="expected grid_hash behaviour across arms; "
                         "defaults to `same` for a shader-side dial and "
                         "`moved` for claude_models")
    a = ap.parse_args()
    if a.grid is None:
        a.grid = "moved" if a.dial == "claude_models" else "same"

    lab.doorway(claude_view=0, claude_nee=0, claude_bounces=24,
                claude_grid_debug=3, claude_rng=1, claude_grid_follow=1,
                time_speed=0)
    time.sleep(1.0)

    runs = []
    for r in range(a.reps):
        for d in (1, 0):
            runs.append(arm(a.vantage, d, "ab%d" % r, a.frames, a.dial))
            print("rep %d %s=%d  mean %.4f  still %s  frame_ms %.2f  grid %s"
                  % (r, a.dial, d, runs[-1]["mean_luma"],
                     runs[-1]["still_frames"],
                     runs[-1]["frame_ms_avg"] or -1,
                     runs[-1]["grid_hash"]), flush=True)

    on = [x["mean_luma"] for x in runs if x["descend"] == 1]
    off = [x["mean_luma"] for x in runs if x["descend"] == 0]
    hashes = sorted({x["grid_hash"] for x in runs})
    out = {
        "dial": a.dial,
        "vantage": a.vantage,
        "grid_expectation": a.grid,
        "runs": runs,
        "mean_on": on,
        "mean_off": off,
        "ratio_on_over_off": (sum(on) / len(on)) / (sum(off) / len(off)),
        # THE ASSERTION that makes this a one-variable experiment: the
        # dial is shader-side, so both arms must be looking at the same
        # baked grid. More than one hash here means the world changed
        # under the measurement and the numbers are not comparable.
        "grid_hashes": hashes,
        "one_grid": len(hashes) == 1,
        "grid_as_expected": (len(hashes) == 1) == (a.grid == "same"),
        "same_arm_spread_on": max(on) - min(on),
        "same_arm_spread_off": max(off) - min(off),
    }
    dst = os.path.join(lab.SHOTS, "%s_ab_%s.json"
                       % (a.dial.replace("claude_", ""), a.vantage))
    open(dst, "w").write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
    print("written: %s" % dst)


if __name__ == "__main__":
    main()
