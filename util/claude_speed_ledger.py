#!/usr/bin/env python3
"""claude_speed_ledger — what each renderer feature costs, in ms.

One seat, one session (GPU timings drift ~30 % between sessions, so only
rows from the SAME run compare). For each scene: the play defaults, then
each feature switched OFF alone; reports the trace pass (GPU timer
query, claude_stats pass_ms[2]), the whole post chain, and the frame.
A feature's cost = its OFF row's saving against the baseline.

Usage: python3 util/claude_speed_ledger.py [--skip-seat] [--frames 240]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_ci as ci                       # noqa: E402
import claude_lab as lab                     # noqa: E402
import claude_denoise_price as price         # noqa: E402

SCENES = {
    "cabin-night": {"pos": [7, 8.5, 2], "yaw": 55, "pitch": -5, "time": 0.0},
    "outdoor-noon": {"pos": [5, 8.5, -25], "yaw": 0, "pitch": -5, "time": 0.5},
    "forest-noon": {"pos": [146.5, 9.5, 123.5], "yaw": 270, "pitch": 4, "time": 0.5},
    "torch-room": {"pos": [241.7, 9, 231.7], "yaw": 315, "pitch": -12, "time": 0.0},
}
PLAY = {"claude_units": 1, "claude_nee": 1, "claude_denoise": 1,
        "claude_auto_exposure": 1, "claude_white_balance": 1,
        "claude_night_vision": 1, "claude_torch_nee": 1,
        "claude_leaf_transmit": 1, "claude_model_far": 1,
        "claude_descend": 1, "claude_split": 48, "claude_reproject": 1}
OFF = [("NEE (all light aiming)", {"claude_nee": 0}),
       ("flame aiming", {"claude_torch_nee": 0}),
       ("leaf translucency", {"claude_leaf_transmit": 0}),
       ("model shapes past ring", {"claude_model_far": 0}),
       ("all 1/16 detail", {"claude_descend": 0}),
       ("denoiser", {"claude_denoise": 0})]


def measure(scene, dials, frames, name):
    v = SCENES[scene]
    vs = lab.load_vantages()
    d = dict(ci.CANONICAL_DIALS)
    d.update(dials)
    rundir = "/tmp/speed_ledger"
    os.makedirs(rundir, exist_ok=True)
    ci.capture({"name": name}, v, vs["furnace-050"], d, rundir, frames,
               vantage_name=name)
    rows = []
    for _ in range(3):
        lab.doorway(claude_stats=1)
        time.sleep(1.3)
        st = lab.read_stats() or {}
        pm = st.get("pass_ms") or []
        if len(pm) > 10:
            rows.append((pm[2], sum(pm[3:10]), st.get("frame_ms_avg") or 0.0))
    if not rows:
        return None
    rows.sort()
    return rows[len(rows) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-seat", action="store_true")
    ap.add_argument("--frames", type=int, default=240)
    args = ap.parse_args()
    if not args.skip_seat:
        price.start_seat()
    ci.do_freeze()
    out = []
    for scene in SCENES:
        base = measure(scene, PLAY, args.frames, scene + "-base")
        print("%-13s %-26s trace %6.2f  post %5.2f  frame %6.2f"
              % (scene, "PLAY DEFAULTS", base[0], base[1], base[2]), flush=True)
        for label, d in OFF:
            r = measure(scene, dict(PLAY, **d), args.frames, scene + "-x")
            print("%-13s %-26s trace %6.2f  post %5.2f  frame %6.2f   "
                  "cost %+6.2f ms trace, %+6.2f ms frame"
                  % (scene, "without " + label, r[0], r[1], r[2],
                     base[0] - r[0], base[2] - r[2]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
