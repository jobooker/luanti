#!/usr/bin/env python3
"""Sky-fog analytic check -- fog stays in front of the sky (2026-10-10).

WHAT IT GUARDS. claude_sky_ground (the sky's radiance is a ground-level
measurement) skips, on a segment that ends at the sky, the clear
atmosphere that measurement already holds (claude_trace skySigT:
max(sigma_t - sigma_clear, 0)). Its first version skipped ALL the air
there, which removed fog in front of the sky, and no CI arm could see it:
the only air arm (furnace-050-air) is sealed.

THE ANALYTIC ANSWER. Under the CONSTANT test sky (claude_sky_uniform L,
no body, no gradient) through a purely ABSORBING medium (claude_air_scatter
0, claude_air_absorb sigma_a), a camera ray that escapes reads

    L_seen = L exp(-sigma_sky * D)

exactly, with sigma_sky = max(sigma_a - sigma_clear, 0) (claude_sky_ground
1) and D the distance from the camera to the far-field box along the ray
(claude_trace farExitT: the box a sky-bound segment's air ends at). No
scattering, so nothing is in-scattered and nothing depends on the ground.
The camera, the box and sigma_clear come from claude_stats (cam_ray,
far_box, air_clear) at the shutter; the arm looks up, so every pixel is sky.

WHAT IT CATCHES, and by how much at the arm's settings: fog dropped in
front of the sky (the first version) reads exp(+sigma_sky D) too bright
(x2.24 in CI); the baseline not subtracted (sky_ground 0) reads exp(-sigma_clear
D) (about 1 % in CI's 128-node grid: below the tolerance, and printed).

BLIND TO: scattering (the in-scatter half of the medium), the sun and the
dome's shape (all off), geometry (none in view).

Usage: claude_skyfog_check.py IMAGE [--lsky L] --linear DUMP_PREFIX
"""
import json
import os
import sys

import numpy as np

LUM = np.array([0.2126, 0.7152, 0.0722])


def box_distance(p, d, box):
    """distance from p along unit rays d (N x 3) to the exit of the box
    (world lo corner box[:3], edge box[3]) -- farExitT"""
    lo = np.array(box[:3], float)
    hi = lo + box[3]
    room = np.where(d >= 0, hi - p, p - lo)
    return np.maximum(np.min(room / np.maximum(np.abs(d), 1e-6), axis=-1), 0.0)


def expected(stats, w, h, l_sky, sigma_sky, sub=2):
    cr = stats["cam_ray"]
    pos, fwd, right, up = (np.array(cr[i:i + 3], float) for i in (0, 3, 6, 9))
    acc = np.zeros((h, w))
    for sy in range(sub):
        for sx in range(sub):
            u = (np.arange(w) + (sx + 0.5) / sub) / w * 2 - 1
            v = 1 - (np.arange(h) + (sy + 0.5) / sub) / h * 2    # dump rows top first
            U, V = np.meshgrid(u, v)
            d = fwd + U[..., None] * right + V[..., None] * up
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            D = box_distance(pos, d.reshape(-1, 3), stats["far_box"]).reshape(h, w)
            acc += l_sky * np.exp(-sigma_sky * D) / (sub * sub)
    return acc


def main():
    img = sys.argv[1]
    l_sky = float(sys.argv[sys.argv.index("--lsky") + 1]) if "--lsky" in sys.argv else 1.0
    if "--linear" not in sys.argv:
        sys.exit("skyfog needs --linear (the radiance, not the picture)")
    prefix = sys.argv[sys.argv.index("--linear") + 1]
    st = json.load(open(img.replace(".png", ".capture.json")))
    stats = st.get("stats_at_shutter") or st.get("stats") or {}
    for k in ("cam_ray", "far_box", "air_clear", "claude_air_scatter", "claude_air_absorb"):
        if stats.get(k) is None:
            sys.exit("no %s in the capture's stats: the referee cannot speak" % k)
    if stats["claude_air_scatter"] != 0:
        sys.exit("the arm must be purely absorbing (claude_air_scatter %s)" % stats["claude_air_scatter"])
    sa, sc = float(stats["claude_air_absorb"]), float(stats["air_clear"])
    m = json.load(open(prefix + ".json"))
    lin = np.fromfile(prefix + ".f32", np.float32).reshape(m["h"], m["w"], 4)[..., :3].astype(np.float64)
    y = lin @ LUM
    h, w = y.shape
    want = {"sky_ground 1 (the rule)": max(sa - sc, 0.0),
            "sky_ground 0 (the double count)": sa,
            "no air on sky segments (the first version)": 0.0}
    meas = float(y.mean())
    se = float(y.std() / np.sqrt(y.size))
    print("skyfog  sigma_a %.6g /m  sigma_clear %.6g /m  far box %s  camera %s"
          % (sa, sc, stats["far_box"], stats["cam_ray"][:3]))
    print("source: linear %s (%dx%d %s)" % (prefix, m["w"], m["h"], m["format"]))
    for i, (tag, s) in enumerate(want.items()):
        e = float(expected(stats, w, h, l_sky, s).mean())
        line = "%s: analytic %.6f  measured %.6f  (ratio %.5f)" % (tag, e, meas, meas / e)
        if i == 0:
            print("ratio %.5f +/- %.5f  %s" % (meas / e, se / e, line))
        else:
            print("  " + line)


if __name__ == "__main__":
    main()
