#!/usr/bin/env python3
"""claude_perfprobe — where a frame's time goes in PLAY settings, one knob at
a time (DECISIONS 0u: 60 fps at 1080p or higher).

Each arm states every knob (play defaults, one changed), so nothing leaks
from the arm before. Parked, pinned, ~5 s per arm; reads the client's own
frame time and per-pass times (pass_ms[2] is the trace pass).

  python3 util/claude_perfprobe.py
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab  # noqa: E402
from claude_playtest import scrub_conf_like_look  # noqa: E402

VIEWS = {
    "plains": (5, 8.5, -25, 0, -12),
    "forest": (146.5, 8.5, 123.5, 270, -5),
    "horizon": (60, 44.5, 60, 45, -6),
    "spawn-cabin": (0, 9.5, -12, 0, -5),
    "torchroom": (241.7, 8.5, 231.7, 315, -12),
}
if os.environ.get("PERF_VIEWS"):
    VIEWS = {k: v for k, v in VIEWS.items() if k in os.environ["PERF_VIEWS"].split(",")}
PLAY = {"claude_nee": 1, "claude_bounces": 24, "claude_far_levels": 3, "claude_denoise": 1,
        "claude_descend": 1, "claude_model_far": 1, "claude_torch_nee": 1, "claude_pyramid": 1,
        # every dial an arm can change is spelled out, or it leaks from the arm before
        "claude_area_nee": 1, "claude_guide": 0, "claude_guide_deposit": 1,
        "claude_guide_impl": 2, "claude_guide_keep": 0.25, "claude_area_pick": 0, "claude_area_skip": 0,
        # ladder stage 2 instruments (a dial missing here leaks between arms:
        # claude_tree_dirs did, 2026-10-07)
        "claude_view": 0, "claude_tree_variant": 0, "claude_tree_dirs": 0, "claude_bricks": 0, "claude_bricks_far": 0}
ARMS = [("play", {}), ("play-again", {}), ("nee 0", {"claude_nee": 0}),
        ("bounces 2", {"claude_bounces": 2}), ("bounces 4", {"claude_bounces": 4}),
        ("far 0", {"claude_far_levels": 0}), ("denoise 0", {"claude_denoise": 0}),
        ("descend 0", {"claude_descend": 0}), ("model_far 0", {"claude_model_far": 0}),
        ("torch_nee 0", {"claude_torch_nee": 0})]
if os.environ.get("PERF_ARMS") == "air":
    # Koschmieder: extinction = 3.912 / visibility
    ARMS = [("haze 0.004 (old default)", {"claude_air_scatter": 0.004}),
            ("clear day 50 km", {"claude_air_scatter": round(3.912 / 50000, 7)}),
            ("no air", {"claude_air_scatter": 0})]
if os.environ.get("PERF_ARMS") == "time":
    ARMS = [("noon", {"time": 0.5}), ("morning 0.30", {"time": 0.30}), ("low sun 0.27", {"time": 0.27}),
            ("evening 0.72", {"time": 0.72}), ("night 0.0", {"time": 0.0}), ("noon-again", {"time": 0.5})]
if os.environ.get("PERF_ARMS") == "lights":
    # roadmap 3d-0: what each light sampler costs
    ARMS = [("play", {}), ("area 0", {"claude_area_nee": 0}), ("flame 0", {"claude_torch_nee": 0}),
            ("area 0 + flame 0", {"claude_area_nee": 0, "claude_torch_nee": 0}), ("play-again", {})]
if os.environ.get("PERF_ARMS") == "guide":
    # roadmap 3d-i: what guided bounces cost
    ARMS = [("play", {}), ("guide 1", {"claude_guide": 1}), ("play-again", {}), ("guide 1 again", {"claude_guide": 1})]
if os.environ.get("PERF_ARMS") == "guide-cost":
    # where the guide's milliseconds go: writing, picking, pricing
    ARMS = [("off", {"claude_guide": 0}), ("on", {"claude_guide": 1}),
            ("books only (table ignored)", {"claude_guide": 2}),
            ("on, writing off", {"claude_guide": 1, "claude_guide_deposit": 0}),
            ("off-again", {"claude_guide": 0}), ("on-again", {"claude_guide": 1}),
            ("v1 on", {"claude_guide": 1, "claude_guide_impl": 1}),
            ("v2 keep 1.0", {"claude_guide": 1, "claude_guide_keep": 1.0})]
if os.environ.get("PERF_ARMS") == "pick":
    # light lists: even / by bound / bound x visibility / + skipping
    ARMS = [("even", {}), ("bound", {"claude_area_pick": 1}), ("bound x visibility", {"claude_area_pick": 2}),
            ("+ skip", {"claude_area_pick": 2, "claude_area_skip": 1}), ("even-again", {})]
if os.environ.get("PERF_ARMS") == "dnyoung":
    ARMS = [("young 0", {"claude_denoise_young": 0}), ("young 64", {"claude_denoise_young": 64}), ("young 0 again", {"claude_denoise_young": 0}), ("young 64 again", {"claude_denoise_young": 64})]
if os.environ.get("PERF_ARMS") == "treespeed":
    # ladder stage 2: what each walk costs, camera rays only (views 36-38)
    ARMS = [("no walk", {"claude_view": 38}), ("today's walk", {"claude_view": 37}),
            ("tree walk", {"claude_view": 36}), ("today's walk again", {"claude_view": 37}),
            ("tree walk again", {"claude_view": 36}), ("no walk again", {"claude_view": 38})]
if os.environ.get("PERF_ARMS") == "treeablate":
    # what makes the tree walk's steps expensive (views 36-38, one walk only)
    ARMS = [("no walk", {"claude_view": 38}), ("today's walk", {"claude_view": 37}),
            ("tree, path arrays", {"claude_view": 36, "claude_tree_variant": 0}),
            ("tree, from root", {"claude_view": 36, "claude_tree_variant": 1}),
            ("tree, path arrays again", {"claude_view": 36, "claude_tree_variant": 0}),
            ("tree, from root again", {"claude_view": 36, "claude_tree_variant": 1})]
if os.environ.get("PERF_ARMS") == "treefast":
    # the fast tree walk (variant 2) against today's walk and the first port
    ARMS = [("no walk", {"claude_view": 38}), ("today's walk", {"claude_view": 37}),
            ("tree, first port", {"claude_view": 36, "claude_tree_variant": 0}),
            ("tree, fast", {"claude_view": 36, "claude_tree_variant": 2}),
            ("today's walk again", {"claude_view": 37}),
            ("tree, fast again", {"claude_view": 36, "claude_tree_variant": 2})]
if os.environ.get("PERF_ARMS") == "treelong":
    # camera rays and long random rays (claude_tree_dirs 1), each walk
    ARMS = [("no walk", {"claude_view": 38}),
            ("cam: today", {"claude_view": 37}), ("cam: tree", {"claude_view": 36, "claude_tree_variant": 2}),
            ("long: today", {"claude_view": 37, "claude_tree_dirs": 1}),
            ("long: tree", {"claude_view": 36, "claude_tree_variant": 2, "claude_tree_dirs": 1}),
            ("long: today again", {"claude_view": 37, "claude_tree_dirs": 1}),
            ("long: tree again", {"claude_view": 36, "claude_tree_variant": 2, "claude_tree_dirs": 1}),
            ("cam: tree again", {"claude_view": 36, "claude_tree_variant": 2})]
if os.environ.get("PERF_ARMS") == "bricks":
    # ladder B2: the walk reading the piece pool instead of the ring + atlas
    ARMS = [("ring + atlas", {"claude_bricks": 0}), ("pool", {"claude_bricks": 1}),
            ("ring + atlas again", {"claude_bricks": 0}), ("pool again", {"claude_bricks": 1})]
if os.environ.get("PERF_ARMS") == "bricksfar":
    # ladder B3: node-box shapes past the ring
    ARMS = [("pool", {"claude_bricks": 1}), ("pool + far shapes", {"claude_bricks": 1, "claude_bricks_far": 1}),
            ("pool again", {"claude_bricks": 1}), ("pool + far shapes again", {"claude_bricks": 1, "claude_bricks_far": 1})]
if os.environ.get("PERF_ARMS") == "split":
    ARMS = [("play", {}), ("split 0", {"claude_split": 0}), ("reproject 0", {"claude_reproject": 0}),
            ("split 0 + reproject 0", {"claude_split": 0, "claude_reproject": 0}), ("play-again", {})]


def main():
    scrub_conf_like_look()
    first = True
    res = {}
    for vn, (x, y, z, yaw, pitch) in VIEWS.items():
        pin = "/tmp/perf_pin.txt"
        open(pin, "w").write("0 %s %s %s %s %s\n" % (x, y, z, yaw, pitch))
        for an, over in ARMS:
            d = dict(PLAY, **{k: v for k, v in over.items() if k != "time"})
            cmd = ["python3", "util/claude_motion.py", "--play", "--nodump", "--path", pin,
                   "--frames", "240", "--name", "perf"]
            for k, v in d.items():
                cmd += ["--dial", "%s=%s" % (k, v)]
            if not first:
                cmd.append("--skip-seat")
            subprocess.run(cmd, cwd=REPO, capture_output=True)
            if first:
                lab.rpc("abm", on=False)
                time.sleep(60)
            first = False
            g = {"pos": [x, y, z], "yaw": yaw, "pitch": pitch}
            if "time" in over:
                g["time"] = over["time"]
            lab.goto(g)
            open(lab.PATCH, "w").write("claude_path = %s\n" % pin)
            # wait for the blocks to stop arriving (an empty or half-loaded
            # grid is cheap and wrong: the 2026-10-06 pinned-shot lesson)
            prev, same, t0 = None, 0, time.time()
            while same < 6 and time.time() - t0 < 60:
                time.sleep(0.5)
                n = (lab.read_stats() or {}).get("grid_solid")
                same = same + 1 if (n == prev and n) else 0
                prev = n
            time.sleep(4)
            st = lab.read_stats() or {}
            pm = st.get("pass_ms") or [0] * 12
            res["%s/%s" % (vn, an)] = {"frame_ms": st.get("frame_ms_avg"), "trace_ms": pm[2],
                                       "other_ms": sum(pm) - pm[2], "fps": st.get("fps"),
                                       "grid_solid": st.get("grid_solid")}
            r = res["%s/%s" % (vn, an)]
            print("%-12s %-12s frame %6.1f ms (%5.1f fps)  trace %6.2f ms  other passes %5.2f ms  grid %s"
                  % (vn, an, r["frame_ms"] or 0, r["fps"] or 0, r["trace_ms"], r["other_ms"], r["grid_solid"]),
                  flush=True)
    lab.rpc("abm", on=True)
    open(lab.PATCH, "w").write("claude_path = 0\n")
    json.dump(res, open("/tmp/perfprobe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
