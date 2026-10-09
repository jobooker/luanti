#!/usr/bin/env python3
"""live pricing, pinned: waits for the scene to stop changing, prints its
identity (grid hash, area lights in the list, far blocks loaded, sun) on
every arm, interleaves arms, every dial spelled every arm."""
import sys, time, json
sys.path.insert(0, "util")
import claude_lab as lab
import claude_gpu_lock   # the GPU lock for this tool's whole life
claude_gpu_lock.hold('util/claude_live_price.py')
VIEWS = {"forest": (146.5, 8.5, 123.5, 270, -5), "torchroom": (241.7, 8.5, 231.7, 315, -12),
         "plains": (5, 8.5, -25, 0, -12)}
BASE = {"claude_nee": 1, "claude_bounces": 24, "claude_area_pick": 0, "claude_area_skip": 0,
        "claude_area_nee": 1, "claude_descend": 1, "claude_leaf_transmit": 1, "claude_far_levels": 3,
        "claude_torch_nee": 1, "claude_model_far": 1, "claude_bricks": 1, "claude_bricks_far": 1,
        "claude_walk_exact": 1, "claude_relief": 0,
        "claude_denoise": 1, "claude_denoise_learned": 0}
ARMS = json.loads(sys.argv[1])
# a dial set but not reset between arms leaks into every later arm (it
# happened three times on 2026-10-08): refuse any arm key BASE lacks
missing = sorted({k for _, over in ARMS for k in over} - set(BASE))
if missing:
    sys.exit("REFUSED: arms set %s, which BASE does not reset" % missing)
ID = ("grid_hash", "area_emitters", "far_db_blocks", "sun_lux")
def ident():
    st = lab.read_stats() or {}
    return tuple(st.get(k) for k in ID)
for vn in sys.argv[2].split(","):
    x, y, z, yaw, pitch = VIEWS[vn]
    open("/tmp/perf_pin.txt", "w").write("0 %s %s %s %s %s\n" % (x, y, z, yaw, pitch))
    lab.goto({"pos": [x, y, z], "yaw": yaw, "pitch": pitch})
    open(lab.PATCH, "w").write("".join("%s = %s\n" % kv for kv in BASE.items()))
    time.sleep(1.5)
    # the scene loaded the same way every time, whatever came before (2026-10-08;
    # see claude_lab.load_scene): grid box re-centred on the pose, every block sent
    ls = lab.load_scene([x, y, z], yaw, pitch, "/tmp/perf_pin.txt")
    if not ls.get("ok"):
        sys.exit("REFUSED %s: scene did not load: %s" % (vn, ls.get("error")))
    open(lab.PATCH, "w").write("claude_path = /tmp/perf_pin.txt\n" + "".join("%s = %s\n" % kv for kv in BASE.items()))
    prev, since, t0 = None, time.time(), time.time()
    while time.time() - t0 < 150:
        time.sleep(1)
        cur = ident()
        if cur != prev:
            prev, since = cur, time.time()
        elif time.time() - since > 10:
            break
    print("%s settled after %.0f s: %s" % (vn, time.time() - t0, dict(zip(ID, prev))), flush=True)
    for name, over in ARMS:
        d = dict(BASE, **over)
        open(lab.PATCH, "w").write("claude_path = /tmp/perf_pin.txt\n" + "".join("%s = %s\n" % kv for kv in d.items()))
        time.sleep(6)
        xs, fr = [], []
        for i in range(4):
            time.sleep(1.5)
            st = lab.read_stats() or {}
            xs.append((st.get("pass_ms") or [0, 0, 0])[2])
            fr.append(st.get("frame_ms_avg") or 0)
        cur = ident()
        flag = "" if cur == prev else "  SCENE CHANGED %s" % (dict(zip(ID, cur)),)
        # the whole frame too (2026-10-08), so this and claude_perfprobe can
        # be held against each other on the same arm
        fm = sum(fr) / 4
        print("%-10s %-18s trace %6.2f ms (%5.2f-%5.2f)  frame %6.2f ms (%5.1f fps)%s" % (
            vn, name, sum(xs) / 4, min(xs), max(xs), fm, 1000.0 / fm if fm else 0, flag), flush=True)
