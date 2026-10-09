#!/usr/bin/env python3
"""claude_mldenoise_equaltime — the network against today's filter at EQUAL
TIME, not only equal input. In one second the network arm accumulates
floor(1000 / (honest frame ms + network ms)) frames, the filter arm
floor(1000 / anything frame ms), each frame time as the scoreboard measured
it (run 20261008-131615). Each arm is scored at the deepest captured depth
not above its frame count (conservative for both). Scores from rows.json.

  python3 util/claude_mldenoise_equaltime.py screenshots/mldenoise/test/<tag>/rows.json NET_MS [NET_MS ...]
"""
import json
import math
import sys

import numpy as np

RUN = "/home/jobooker/code/luanti/screenshots/scoreboard/runs/20261008-131615/shots.json"


def main():
    rec = json.load(open(sys.argv[1]))
    fms = {}
    for s in json.load(open(RUN))["shots"]:
        fms[(s["scene"], s["contender"])] = s["frame_ms"]
    for net_ms in [float(x) for x in sys.argv[2:]]:
        print("network cost %.1f ms/frame" % net_ms)
        print("scene      filter: ms  frames  RMSE   FLIP  | network: ms  frames  RMSE   FLIP  | RMSE filter/net")
        for sc in ["forest", "plains", "cabin", "torchroom"]:
            rows = [r for r in rec["rows"] if r["scene"] == sc]
            depths = sorted({r["frames"] for r in rows})

            def at(n, k):
                d = max([x for x in depths if x <= n] or [depths[0]])
                rr = [r for r in rows if r["frames"] == d]
                return d, np.mean([r["rmse_" + k] for r in rr]), np.mean([r.get("flip_" + k, float("nan")) for r in rr])
            fa = fms[(sc, "anything")]
            fn = fms[(sc, "honest")] + net_ms
            na, nn = max(1, math.floor(1000 / fa)), max(1, math.floor(1000 / fn))
            da, ra, pa = at(na, "filter")
            dn, rn, pn = at(nn, "net")
            print("%-10s %6.1f %4d->%-3d %.4f %.4f | %6.1f %4d->%-3d %.4f %.4f | %.2fx" % (
                sc, fa, na, da, ra, pa, fn, nn, dn, rn, pn, ra / rn))


if __name__ == "__main__":
    main()
