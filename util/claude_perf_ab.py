#!/usr/bin/env python3
"""Restart-based A/B of the perf dials, at a fixed vantage.

Several dials (trace_scale, face_direct) do not take effect on a live
settings-patch flip — known gap #5, "runtime dial flip still broken
(per-frame invalidation + pipeline mis-scale)". Measuring them live
reports "no effect" when the truth is "did not apply", so each config
here gets its own client launch from a freshly written conf.

Camera is parked at the same vantage every run (scene drift killed
cross-restart A/Bs before), and claude_refine is forced OFF so we are
measuring the MOTION-mode cost — the number that decides whether
60 fps while moving is reachable.

Usage: util/claude_perf_ab.py            (server must already be up)
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from claude_lab import rpc, STATS  # noqa: E402

CONF = os.path.join(REPO, "minetest.conf")
BIN = os.path.join(REPO, "bin", "luanti")

BASE = """name = claude
address = 127.0.0.1
remote_port = 30000
screen_w = 1920
screen_h = 1080
fullscreen = false
invert_mouse = true
fast_move = true
free_move = true
viewing_range = 120
enable_dynamic_shadows = false
enable_bloom = false
enable_auto_exposure = false
enable_volumetric_lighting = false
tone_mapping = false
undersampling = 1
fsaa = 0
claude_grid_debug = 3
claude_stats = 1
claude_subvox = 1
claude_cascades = 0
claude_light_ladder = 0
claude_denoise = 0
ssao_strength = 0
claude_refine = 0
default_privs = interact, shout, fly, fast, noclip, teleport, settime, privs, give, debug, server
"""

CONFIGS = [
    ("baseline (pure)", dict(claude_tiers=0, claude_trace_scale=1.0,
                             claude_face_direct=0)),
    ("tiers=1", dict(claude_tiers=1, claude_trace_scale=1.0,
                     claude_face_direct=0)),
    ("tiers=1 + trace_scale 0.5", dict(claude_tiers=1, claude_trace_scale=0.5,
                                       claude_face_direct=0)),
    ("tiers=1 + scale 0.5 + face_direct", dict(claude_tiers=1,
                                               claude_trace_scale=0.5,
                                               claude_face_direct=1)),
]


def kill_client():
    subprocess.run(["pkill", "-f", "bin/luanti --go"],
                   capture_output=True)
    time.sleep(2)


def run(label, dials, vantage="cozy-night"):
    kill_client()
    with open(CONF, "w") as f:
        f.write(BASE)
        for k, v in dials.items():
            f.write("%s = %s\n" % (k, v))
    log = "/tmp/claude_ab_client.log"
    p = subprocess.Popen([BIN, "--go", "--address", "127.0.0.1",
                          "--port", "30000", "--name", "claude"],
                         cwd=REPO, stdout=open(log, "w"),
                         stderr=subprocess.STDOUT)
    # wait for the client to be in-world and writing stats
    deadline = time.time() + 90
    while time.time() < deadline:
        time.sleep(2)
        try:
            if os.path.getmtime(STATS) > time.time() - 4:
                break
        except OSError:
            continue
    time.sleep(3)
    try:
        rpc("tp", **VANT[vantage])
    except Exception as e:
        print("  (vantage failed: %s)" % e)
    time.sleep(6)  # let the view settle at the parked camera
    best = None
    for _ in range(4):
        time.sleep(2.5)
        s = json.load(open(STATS))
        if best is None or s["pass_ms"][5] < best["pass_ms"][5]:
            best = s
    print("%-38s trace %7.1f ms   fps %6.2f" %
          (label, best["pass_ms"][5], best["fps"]))
    p.terminate()
    return best["pass_ms"][5], best["fps"]


v = json.load(open(os.path.join(HERE, "claude_vantages.json")))
VANT = {k: dict(pos={"x": d["pos"][0], "y": d["pos"][1], "z": d["pos"][2]},
                yaw=d["yaw"], pitch=d["pitch"])
        for k, d in v.items()}

if __name__ == "__main__":
    print("restart-based A/B, parked at cozy-night, refine OFF (motion cost)")
    print()
    results = []
    for label, dials in CONFIGS:
        results.append((label,) + run(label, dials))
    print()
    b_ms, b_fps = results[0][1], results[0][2]
    print("speedup vs baseline:")
    for label, ms, fps in results:
        print("  %-38s %5.2fx   (%.1f fps)" % (label, b_ms / ms, fps))
    kill_client()
