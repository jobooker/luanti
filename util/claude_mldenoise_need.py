"""equal time without the capture grid's coarseness: how many of the filter's
frames does the learned filter need to match the filter's RMSE AND FLIP, and
how many does it get at its measured cost?"""
import json
import math
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = json.load(open(os.path.join(REPO, "screenshots/mldenoise/test/engine/rows.json")))["rows"]
P = json.load(open(os.path.expanduser("~/data/mldenoise/price/price.json")))
med = lambda sc, arm: np.median([p["busy_ms"] for p in P if p["scene"] == sc and p["arm"] == arm])
for sc in ("forest", "plains", "cabin", "torchroom"):
    depths = sorted({r["frames"] for r in rows if r["scene"] == sc})

    def st(k, n):
        rr = [r for r in rows if r["scene"] == sc and r["frames"] == n]
        return np.mean([r["rmse_" + k] for r in rr]), np.mean([r["flip_" + k] for r in rr])
    nf = math.floor(1000 / med(sc, "filter"))
    nl = math.floor(1000 / med(sc, "learned"))
    for D in sorted({max(x for x in depths if x <= min(nf, depths[-1])), 32, 16, 4}):
        fr, ff = st("filter", D)
        need = [n for n in depths if st("learned", n)[0] <= fr and st("learned", n)[1] <= ff]
        nn = min(need) if need else None
        print("%-9s filter at %2d fr: RMSE %.4f FLIP %.3f -> learned matches both from %s fr (%.0f%% of the "
              "filter's frames); at measured cost it gets %.0f%% (%d vs %d per s)" % (
                  sc, D, fr, ff, nn, 100.0 * nn / D if nn else float("nan"), 100.0 * nl / nf, nl, nf))

print()
print("interpolated (metric linear in log frames between captured depths): the learned filter's frames needed")
for sc in ("forest", "plains", "cabin", "torchroom"):
    depths = sorted({r["frames"] for r in rows if r["scene"] == sc})
    nf = math.floor(1000 / med(sc, "filter"))
    nl = math.floor(1000 / med(sc, "learned"))

    def curve(k, j):
        return [np.mean([r[("rmse_", "flip_")[j] + k] for r in rows if r["scene"] == sc and r["frames"] == n])
                for n in depths]
    out = []
    for D in (16, 32, max(x for x in depths if x <= min(nf, depths[-1]))):
        cells = []
        for j, nm in ((0, "RMSE"), (1, "FLIP")):
            target = curve("filter", j)[depths.index(D)]
            lc = curve("learned", j)
            need = None
            for i in range(len(depths)):
                if lc[i] <= target:
                    if i == 0:
                        need = depths[0]
                    else:
                        a, b = lc[i - 1], lc[i]
                        t = (a - target) / (a - b)
                        need = math.exp(math.log(depths[i - 1]) + t * (math.log(depths[i]) - math.log(depths[i - 1])))
                    break
            cells.append("%s %s" % (nm, ("%.0f%%" % (100 * need / D)) if need else "never"))
        out.append("at %d: %s" % (D, ", ".join(cells)))
    print("%-9s gets %.0f%% of the filter's frames | needs: %s" % (sc, 100.0 * nl / nf, " | ".join(out)))
