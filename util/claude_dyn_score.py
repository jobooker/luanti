#!/usr/bin/env python3
import json, sys
import numpy as np
sys.path.insert(0, "util")
import claude_judge as J
d = json.load(open("/tmp/dyn_dumps.json"))
R, rr = J.load_dump(d["ref"])
print("reference frames", len(R))
for arm in ("realtime", "realtime-denoise"):
    T, rows = J.load_dump(d[arm])
    n = min(len(T), len(R))
    q = J.jod_video(T[:n], R[:n], 60.0, clip=40)
    p = J.pacing(rows)
    print("%-17s video JOD %.2f  fps %.1f p99 %.1f ms  brightness %.3f" % (
        arm, q, p["fps_mean"], p["frame_ms_p99"],
        np.mean([J.brightness_ratio(t, r) for t, r in zip(T[:n:8], R[:n:8])])))
    # error over time: linear RMSE per frame against the reference frame
    err = [float(np.sqrt(((J.lin(T[i]) - J.lin(R[i])) ** 2).mean())) for i in range(n)]
    line = "  RMSE by 10-frame bin: " + " ".join("%.3f" % np.mean(err[b:b + 10]) for b in range(0, n, 10))
    print(line)
