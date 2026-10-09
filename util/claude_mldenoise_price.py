#!/usr/bin/env python3
"""claude_mldenoise_price — what the learned denoiser costs IN THE ENGINE,
the live pricer's way (util/claude_live_price.py): the camera pinned at a
scoreboard pose, the scene left to stop changing and its identity printed
on every arm, arms interleaved (order rotating), every dial spelled every arm, three
rounds. Per arm: the frame (busy_ms, frame_ms_avg) and the GPU pass
times (pass_ms; the six filter passes and the learned step are named by
their position, read from this build's pipeline order below).

  python3 util/claude_mldenoise_price.py [scene ...]       (holds the GPU lock)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import claude_gpu_lock  # noqa: E402

claude_gpu_lock.hold("util/claude_mldenoise_price.py")
import claude_lab as lab  # noqa: E402
import claude_mldenoise_capture as cap  # noqa: E402

# pass_ms slots in this build (secondstage.cpp order, subpasses off):
# 0 raster3d, 1 second_stage, 2 trace, 3-8 the six claude_denoise passes
# (each skipped while the learned one runs), 9 the learned step, 10 exposure
DN = list(range(3, 9))
LEARNED = 9
ARMS = [("filter", cap.ANYTHING), ("learned", cap.LEARNED), ("honest", cap.HONEST)]
ID = ("grid_hash", "area_emitters", "far_db_blocks", "sun_lux")
OUT = os.path.expanduser("~/data/mldenoise/price")


def ident():
    st = lab.read_stats() or {}
    return tuple(st.get(k) for k in ID)


def main():
    scenes = sys.argv[1:] or list(cap.SCENES)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sc in scenes:
        cap.seat()   # a fresh game per scene, as the scoreboard
        ex = cap.truth_meta(sc)["exposure"]
        pos, yaw, pitch, tod = cap.SCENES[sc]
        pin = "/tmp/mld_price_pin.txt"
        open(pin, "w").write("0 %s %s %s %s %s\n" % (pos[0], pos[1], pos[2], yaw, pitch))
        lab.goto({"pos": pos, "yaw": yaw, "pitch": pitch, "time": tod})

        def push(d):
            open(lab.PATCH, "w").write("claude_path = %s\n" % pin + "".join("%s = %s\n" % kv for kv in d.items()))
        base = dict(cap.ANYTHING, **cap.DISPLAY, claude_exposure=ex)
        push(base)
        prev, since, t0 = None, time.time(), time.time()
        while time.time() - t0 < 150:
            time.sleep(1)
            cur = ident()
            if cur != prev:
                prev, since = cur, time.time()
            elif time.time() - since > 10:
                break
        print("%s settled after %.0f s: %s" % (sc, time.time() - t0, dict(zip(ID, prev))), flush=True)
        for rnd in range(3):
            # the order rotates each round: the trace's own time drifted with
            # the arm order on the first run (2026-10-09)
            for name, dials in ARMS[rnd % 3:] + ARMS[:rnd % 3]:
                push(dict(dials, **cap.DISPLAY, claude_exposure=ex))
                time.sleep(8)   # the dial poll (~1 Hz) and the pass EMA (0.9) settle
                xs = []
                for i in range(5):
                    time.sleep(1.2)
                    st = lab.read_stats() or {}
                    pm = st.get("pass_ms") or []
                    xs.append({"busy": st.get("busy_ms"), "frame": st.get("frame_ms_avg"),
                               "dn": sum(pm[i] for i in DN if i < len(pm)),
                               "learned": pm[LEARNED] if len(pm) > LEARNED else None,
                               "trace": pm[2] if len(pm) > 2 else None, "n": len(pm), "pass_ms": pm})
                cur = ident()
                avg = lambda k: sum(x[k] for x in xs if x[k] is not None) / max(1, sum(x[k] is not None for x in xs))
                r = {"scene": sc, "round": rnd, "arm": name, "busy_ms": avg("busy"), "frame_ms": avg("frame"),
                     "dn_ms": avg("dn"), "learned_ms": avg("learned"), "trace_ms": avg("trace"),
                     "n_passes": xs[-1]["n"], "pass_ms_last": xs[-1]["pass_ms"], "ident": dict(zip(ID, cur)),
                     "same_scene": cur == prev}
                rows.append(r)
                print("%-10s r%d %-8s busy %6.2f ms  frame %6.2f  trace %6.2f  six passes %5.3f  learned %5.3f%s" % (
                    sc, rnd, name, r["busy_ms"], r["frame_ms"], r["trace_ms"], r["dn_ms"], r["learned_ms"],
                    "" if r["same_scene"] else "  SCENE CHANGED %s" % (r["ident"],)), flush=True)
        json.dump(rows, open(os.path.join(OUT, "price.json"), "w"), indent=1)
    open(lab.PATCH, "w").write("claude_path = 0\n")


if __name__ == "__main__":
    main()
