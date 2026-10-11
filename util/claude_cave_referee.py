#!/usr/bin/env python3
"""claude_cave_referee -- the cave sun patch, judged analytically, and the
truth-vs-control question judged in LINEAR radiance (2026-10-10, cave-turn).

WHY. The truth guard (claude_truth_store.control_verdict) found the control
(no light sampling, uniform bounces) 2.4 % brighter than the NEE truth in the
cave lit through one 1x1 opening. Measured here: in linear radiance the two
agree (cave pose, 4096 frames x 2 seeds: control/truth 0.9975 +- 0.0014; the
guard's own depths, truth 1024 / control 18500 frames: 1.0004), and both match
the analytic floor patch below to 0.2 %. The 2.4 % is the guard's METRIC: it
averages 8-bit DISPLAY values (exposure, ACES, gamma), and the control's
per-pixel noise in that room is enormous and heavy tailed (relative sd 2 per
pixel at 4096 frames in the walls; the median pixel sits at a quarter of the
mean), so the curve's convex toe turns noise into brightness. The same linear
images read +2.3 %, +4.5 % or -6 % in display space depending only on how
many frames the control had. Inverting the display transform per pixel does
not rescue the 8-bit frames either (the sun patch sits at 249/255 where one
code step is ~10 % of radiance): -1.6 %.

  run SPEC.json OUT          arms on one seat (display dials off, a fresh demo
                             world, linear accum dumps, GPU lock held)
  referee OUT NAME...        the cave floor under the opening, analytic
  pad OUT NAME...            the open sky-furnace pad, analytic
  compare OUT [mask=ARM] A/B ...   linear ratio with its standard error (per-pixel
                             variance from the two seeds), and the guard's
                             display-space metric of the same images
  noise OUT ARM...           per-pixel variance (two seeds) times the frame
                             count: flat if the average converges as 1/N

THE REFEREE (vantage cave-skylight-noon, "Referee (2b)"). For a floor point,
the directions that pass the 1-node-thick opening are those whose crossings
of y = 13.5 and y = 14.5 both fall inside its square: a RECTANGLE in the
tangent plane (tx, tz) = (dx/dy, dz/dy). Irradiance on a horizontal receiver
from radiance L(t) is INT L(t) (1 + |t|^2)^-2 dtx dtz, so each pixel is two
summed-area-table lookups: the sun (the uniform disc the engine draws and
samples, tan(radius) = 0.07 * 1.7, radiance sun_lux / 1000 / omega) and the
dome (mix(horizon, zenith, sqrt(cos y)), the engine's sky). Checked: the full
disc gives sun_lux * (1 + cos a) / 2 to 0.003 %; the flux reaching the floor
equals the flux through the shaft to 0.005 %. Blind to: air (run the arms
with claude_air_scatter = claude_air_absorb = 0), light through an open door
(shut it: the arms' "doors": true), indirect light (claude_bounces 1).
"""
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LUM = np.array([0.2126, 0.7152, 0.0722])
EXPOSURE = 0.808885          # the playtest truths' fixed exposure (run.json)

# ---------------------------------------------------------------- geometry
FLOOR_Y = 8.5                       # top of the floor nodes (y = 8)
HOLE = (13.5, 14.5, 88.5, 89.5)     # x0 x1 z0 z1 of the opening, node (14,14,89)
SLAB = (13.5, 14.5)                 # the ceiling layer y = 14
INTERIOR = (10.5, 17.5, 85.5, 92.5)
TAN_A = 0.07 * 1.7                  # game.cpp sun_half at sun_scale 1
RHO = (186 / 255.0) ** 2.2          # gray186, cellAlbedo()
PAD = (50.5, 69.5, 100.5, 119.5)    # sky-furnace pad about (60,110), 3-node margin


class SAT:
    """summed-area table of f on a regular grid over [-R, R]^2, cell h"""
    def __init__(self, f, R, h):
        n = int(round(2 * R / h))
        c = (np.arange(n) + 0.5) * h - R
        tx, tz = np.meshgrid(c, c, indexing="ij")
        self.S = np.zeros((n + 1, n + 1))
        self.S[1:, 1:] = (f(tx, tz) * h * h).cumsum(0).cumsum(1)
        self.R, self.h, self.n = R, h, n

    def cum(self, a, b):
        ia = np.clip((a + self.R) / self.h, 0, self.n)
        ib = np.clip((b + self.R) / self.h, 0, self.n)
        i0 = np.floor(ia).astype(int).clip(0, self.n - 1)
        j0 = np.floor(ib).astype(int).clip(0, self.n - 1)
        fa, fb = ia - i0, ib - j0
        S = self.S
        return (S[i0, j0] * (1 - fa) * (1 - fb) + S[i0 + 1, j0] * fa * (1 - fb)
                + S[i0, j0 + 1] * (1 - fa) * fb + S[i0 + 1, j0 + 1] * fa * fb)

    def rect(self, a0, a1, b0, b1):
        ok = (a1 > a0) & (b1 > b0)
        a1, b1 = np.maximum(a1, a0), np.maximum(b1, b0)
        v = self.cum(a1, b1) - self.cum(a0, b1) - self.cum(a1, b0) + self.cum(a0, b0)
        return np.where(ok, v, 0.0)


def sky_tables(stats):
    cos_a = 1.0 / np.sqrt(1.0 + TAN_A ** 2)
    l_sun = stats["sun_lux"] / 1000.0 / (2 * np.pi * (1 - cos_a))
    hor = float(np.dot(stats["sky_horizon"], LUM))
    zen = float(np.dot(stats["sky_zenith"], LUM))

    def disc(tx, tz):    # zenith sun (time 0.5, no orbit tilt)
        q = 1 + tx * tx + tz * tz
        return (1 / np.sqrt(q) >= cos_a) * l_sun / q ** 2

    def dome(tx, tz):
        q = 1 + tx * tx + tz * tz
        return (hor + (zen - hor) * q ** -0.25) / q ** 2
    return SAT(disc, 0.13, 0.00025), SAT(dome, 1.6, 0.001)


def opening_rect(x, z):
    h0, h1 = SLAB[0] - FLOOR_Y, SLAB[1] - FLOOR_Y
    def axis(p, lo, hi):
        return (np.maximum((lo - p) / h0, (lo - p) / h1),
                np.minimum((hi - p) / h0, (hi - p) / h1))
    a0, a1 = axis(x, HOLE[0], HOLE[1])
    b0, b1 = axis(z, HOLE[2], HOLE[3])
    return a0, a1, b0, b1


def floor_hits(cam, w, h, box, sub=3):
    """per sub-pixel sample: world (x, z) where the camera ray meets the
    floor plane, and whether that is inside `box`. Rows TOP first, like the
    dump (claude_linear.py)."""
    pos = np.array(cam["cam_pos"], float) + np.array(cam["origin"], float)
    fwd, right, up = (np.array(cam[k], float) for k in ("fwd", "right", "up"))
    out = []
    for sy in range(sub):
        for sx in range(sub):
            u = (np.arange(w) + (sx + 0.5) / sub) / w * 2 - 1
            v = 1 - (np.arange(h) + (sy + 0.5) / sub) / h * 2
            U, V = np.meshgrid(u, v)
            d = fwd + U[..., None] * right + V[..., None] * up
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            t = (FLOOR_Y - pos[1]) / np.minimum(d[..., 1], -1e-9)
            x, z = pos[0] + t * d[..., 0], pos[2] + t * d[..., 2]
            ok = (t > 0) & (x > box[0]) & (x < box[1]) & (z > box[2]) & (z < box[3])
            out.append((x, z, ok))
    return pos, out


# ---------------------------------------------------------------- dumps
def load_y(d, name):
    j = json.load(open(os.path.join(d, name + ".json")))
    a = np.fromfile(os.path.join(d, name + ".f32"), np.float32).reshape(j["h"], j["w"], 4)
    return a[..., :3].astype(np.float64)


def runs(d):
    return {r["name"]: r for r in (json.loads(l) for l in open(os.path.join(d, "runs.jsonl")))}


def camera(d, names):
    for n in names:
        p = os.path.join(d, n + "_trace", "camera.json")
        if os.path.exists(p):
            return json.load(open(p))
    sys.exit("no camera.json (shoot one arm with \"export\": true)")


def cmd_referee(d, names):
    rs = runs(d)
    st = rs[names[0]]["stats"]
    if st.get("claude_air_scatter") or st.get("claude_air_absorb"):
        print("WARNING: air is on in these arms; the referee has no air")
    s_sun, s_dome = sky_tables(st)
    h, w = load_y(d, names[0]).shape[:2]
    pos, hits = floor_hits(camera(d, names), w, h, INTERIOR)
    e_sun, e_dome, ok = np.zeros((h, w)), np.zeros((h, w)), np.ones((h, w), bool)
    for x, z, k in hits:
        a0, a1, b0, b1 = opening_rect(x, z)
        e_sun += s_sun.rect(a0, a1, b0, b1) / len(hits)
        e_dome += s_dome.rect(a0, a1, b0, b1) / len(hits)
        ok &= k
    ref = RHO / np.pi * (e_sun + e_dome)
    peak = ref[ok].max()
    regions = [("sun core (> 0.5 peak)", ok & (ref > 0.5 * peak)),
               ("penumbra (0.05-0.5 peak)", ok & (ref > 0.05 * peak) & (ref <= 0.5 * peak)),
               ("whole lit patch", ok & (ref > 0.05 * peak)),
               ("floor the sun cannot reach", ok & (e_sun < 1e-9))]
    _report(d, names, rs, ref, regions, "camera %s; sun_lux %.0f" % (pos, st["sun_lux"]))


def box_distance(p, dirs, box):
    """claude_trace farExitT in world coords: distance from p along unit
    rays to the exit of the far-field box (world lo corner box[:3], edge
    box[3]) -- where a sky-bound segment's air ends"""
    lo = np.array(box[:3], float)
    hi = lo + box[3]
    room = np.where(dirs >= 0, hi - p, p - lo)
    return np.maximum(np.min(room / np.maximum(np.abs(dirs), 1e-6), axis=-1), 0.0)


def pad_irradiance(st, sigma_sky, x):
    """horizontal irradiance (luminance, engine units) at pad point x from
    the sun and the dome, each dimmed by exp(-sigma_sky D) along its own
    sky-bound path (D = farExitT); sigma_sky = 0 gives the plain IES sums"""
    cos_a = 1.0 / np.sqrt(1.0 + TAN_A ** 2)
    sd = np.array(st.get("sun_dir") or [0.0, 1.0, 0.0], float)
    box = st.get("far_box")
    # the disc by quadrature in its own frame (cone angle, azimuth): each
    # direction its own cos to the pad and its own path through the air (a
    # disc ray at angle theta travels D / cos theta, ~1.4 % more extinction
    # than the centre ray in the 0.5 km fog)
    l_sun = st["sun_lux"] / 1000.0 / (2 * np.pi * (1 - cos_a))
    nr, na = 200, 96
    ct = 1 - (np.arange(nr) + 0.5) / nr * (1 - cos_a)
    az = (np.arange(na) + 0.5) / na * 2 * np.pi
    CT, AZ = np.meshgrid(ct, az, indexing="ij")
    st_ = np.sqrt(1 - CT * CT)
    ta = np.array([1.0, 0.0, 0.0]) if abs(sd[1]) > 0.5 else np.array([0.0, 1.0, 0.0])
    tx = np.cross(ta, sd); tx /= np.linalg.norm(tx)
    ty = np.cross(sd, tx)
    dirs = (tx * (st_ * np.cos(AZ))[..., None] + ty * (st_ * np.sin(AZ))[..., None]
            + sd * CT[..., None]).reshape(-1, 3)
    dw = (1 - cos_a) / nr * 2 * np.pi / na
    cosn = np.maximum(dirs[:, 1], 0.0)
    tr = np.exp(-sigma_sky * box_distance(x, dirs, box)) if sigma_sky > 0 else 1.0
    e_sun = float((l_sun * cosn * tr).sum() * dw)
    # the unoccluded, unattenuated disc for the record: sun_lux (1+cos a)/2 cos z
    e_full = st["sun_lux"] / 1000.0 * (1 + cos_a) / 2 * sd[1]
    t_sun = e_sun / e_full
    hor = float(np.dot(st["sky_horizon"], LUM))
    zen = float(np.dot(st["sky_zenith"], LUM))
    # the dome by midpoint quadrature in (cos theta, phi): radiance
    # mix(H, Z, sqrt(cos)) times cos, dw = dcos dphi
    nc, nphi = 400, 360
    c = (np.arange(nc) + 0.5) / nc
    ph = (np.arange(nphi) + 0.5) / nphi * 2 * np.pi
    C, P = np.meshgrid(c, ph, indexing="ij")
    sn = np.sqrt(1 - C * C)
    dirs = np.stack([sn * np.cos(P), C, sn * np.sin(P)], -1).reshape(-1, 3)
    t = np.exp(-sigma_sky * box_distance(x, dirs, box)).reshape(C.shape) if sigma_sky > 0 else 1.0
    e_dome = float(((hor + (zen - hor) * np.sqrt(C)) * C * t).sum() * (1.0 / nc) * (2 * np.pi / nphi))
    return e_sun, e_dome, t_sun


def full_stats(r):
    """every stat at the shutter (the run log keeps a subset)"""
    try:
        return json.load(open(r["png"].replace(".png", ".capture.json")))["stats"]
    except Exception:
        return r["stats"]


def cmd_pad(d, names):
    """the open pad, direct light only (claude_bounces 1). With air: the sun
    and the dome dimmed along their sky-bound paths by the air above the
    clear baseline (claude_sky_ground 1) or by all of it (0), the camera
    segment (it ends at the pad) by all of it. Not modelled: in-scatter on
    the camera segment (sigma t_cam < 0.03 here; printed as a bound) and the
    fence/surroundings (~0.2 %)."""
    rs = runs(d)
    cam = camera(d, names)
    h, w = load_y(d, names[0]).shape[:2]
    pos, hits = floor_hits(cam, w, h, PAD, sub=1)
    x, z, ok = hits[0]
    # the camera's own path length to each pad pixel
    t_cam = np.hypot(np.hypot(x - pos[0], z - pos[2]), FLOOR_Y - pos[1])
    xc = np.array([60.0, FLOOR_Y, 110.0])
    by_arm = {}
    for n in names:
        by_arm.setdefault(rs[n]["arm"], []).append(n)
    print("camera %s; fence ignored (~0.2 %%)" % pos)
    print("%-14s %9s %9s %9s %8s %9s %9s %18s" % ("arm", "sigma_t", "sig_sky", "T_sun", "E_dome", "referee",
                                               "camT", "measured/referee"))
    for arm, ns in by_arm.items():
        st = full_stats(rs[ns[0]])
        sg = float(rs[ns[0]]["dials"].get("claude_sky_ground", 1))
        sig = float(st.get("claude_air_scatter") or 0) + float(st.get("claude_air_absorb") or 0)
        sig_sky = max(sig - float(st.get("air_clear") or 0.0), 0.0) if sg > 0.5 else sig
        e_sun, e_dome, t_sun = pad_irradiance(st, sig_sky, xc)
        ref = RHO / np.pi * (e_sun + e_dome) * np.exp(-sig * t_cam)
        ys = [load_y(d, n) @ LUM for n in ns]
        v = np.mean([y[ok].mean() for y in ys]) / ref[ok].mean()
        se = _se(ys, ok) / ref[ok].mean()
        print("%-14s %9.3g %9.3g %9.4f %8.4f %9.4f %9.4f %10.4f +- %.4f   (sun %s, sky_ground %g, sky_lux check %.4f)"
              % (arm, sig, sig_sky, t_sun, e_dome, ref[ok].mean(), float(np.exp(-sig * t_cam[ok]).mean()),
                 v, se, st.get("sun_dir"), sg, st["sky_lux"] / 1000.0))


def _report(d, names, rs, ref, regions, head):
    print(head)
    arms = {}
    for n in names:
        arms.setdefault(rs[n]["arm"], []).append(load_y(d, n) @ LUM)
    print("%-28s %8s %9s" % ("region", "pixels", "referee") + "".join(" %22s" % a for a in arms))
    for rn, m in regions:
        s = "%-28s %8d %9.4f" % (rn, m.sum(), ref[m].mean())
        for a, ims in arms.items():
            v = np.mean([x[m].mean() for x in ims]) / ref[m].mean()
            se = _se(ims, m) / ref[m].mean()
            s += " %22s" % ("%.4f +- %.4f" % (v, se))
        print(s)


def _se(ims, m):
    """standard error of the seed-averaged region mean, from the per-pixel
    two-seed variance (pixels have independent random streams)"""
    if len(ims) < 2:
        return float("nan")
    v = ((ims[0][m] - ims[1][m]) ** 2 / 2).sum() / m.sum() ** 2
    return float(np.sqrt(v / len(ims)))


def guard_metric(lin):
    """what claude_truth_store compares: exposure, Narkowicz ACES, gamma
    1/2.2, 8 bits, decoded as sRGB, luminance (claude_present + _frame)"""
    x = np.maximum(lin, 0) * EXPOSURE
    a = np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    g = np.round(a ** (1 / 2.2) * 255) / 255
    return np.where(g <= 0.04045, g / 12.92, ((g + 0.055) / 1.055) ** 2.4) @ LUM


def cmd_compare(d, args):
    regions = [("all", None)]
    if args and args[0].startswith("mask="):
        mk = np.mean([load_y(d, args[0][5:] + "_s%d" % s) @ LUM for s in (1, 2)], 0)
        patch = mk > 0.1 * np.percentile(mk, 99.9)
        regions += [("patch", patch), ("rest", ~patch)]
        args = args[1:]
    for p in args:
        a, b = p.split("/")
        A = [load_y(d, n) for n in sorted(runs(d)) if n.rsplit("_s", 1)[0] == a]
        B = [load_y(d, n) for n in sorted(runs(d)) if n.rsplit("_s", 1)[0] == b]
        s = "%-24s" % p
        for rn, m in regions:
            m = np.ones(A[0].shape[:2], bool) if m is None else m
            ya, yb = [x @ LUM for x in A], [x @ LUM for x in B]
            ma, mb = np.mean([x[m].mean() for x in ya]), np.mean([x[m].mean() for x in yb])
            r = ma / mb
            se = r * np.hypot(_se(ya, m) / ma, _se(yb, m) / mb)
            s += "  %s %.4f +- %.4f" % (rn, r, se)
        ga = np.mean([guard_metric(x).mean() for x in A])
        gb = np.mean([guard_metric(x).mean() for x in B])
        print(s + "  | guard's display metric %.4f" % (ga / gb))


def cmd_noise(d, arms):
    """does the average converge as 1/N? var * N is flat for a true running
    average; it grows for an exponential one (found 2026-10-10: the per-pixel
    count capped at 4096 made parked averages exponential past 4096 frames)"""
    rs = runs(d)
    for a in arms:
        ns = sorted(n for n in rs if n.rsplit("_s", 1)[0] == a)
        x, y = (load_y(d, n) @ LUM for n in ns[:2])
        frames = min(rs[n]["stats"]["still_frames"] for n in ns[:2])
        v = float(((x - y) ** 2 / 2).mean())
        print("%-16s frames %6d  mean %.5f  per-pixel var %.5g  var x frames %.4g"
              % (a, frames, (x.mean() + y.mean()) / 2, v, v * frames))


# ---------------------------------------------------------------- arms
DISPLAY_OFF = {"claude_denoise": 0, "claude_denoise_learned": 0, "claude_ledger": 0,
               "claude_boost": 0, "claude_split": 0, "claude_raw_frame": 0,
               "claude_reproject": 0, "claude_guide": 0, "claude_auto_exposure": 0,
               "claude_exposure": EXPOSURE, "claude_rng": 2, "claude_truth": 0,
               "claude_area_pick": 0, "claude_area_skip": 0, "claude_view": 0,
               "claude_sky_uniform": 0, "claude_torch_nee": 0, "claude_area_nee": 0,
               "claude_bounces": 24, "claude_nee": 0, "claude_bounce_uniform": 0}


def cmd_run(spec_path, out):
    """SPEC: {"pose": [x,y,z,yaw,pitch], "frames": N, "seeds": [1,2], "base": {dials},
    "arms": [{"name", "dials", "pose"?, "frames"?, "seeds"?, "doors"?, "export"?}]}.
    Every arm spells every dial (an unpushed dial does not revert on a seat);
    air-off arms go last unless "base" sets air for all."""
    spec = json.load(open(spec_path))
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, HERE)
    import claude_gpu_lock
    claude_gpu_lock.hold("claude_cave_referee run " + spec_path)
    world = subprocess.check_output([sys.executable, os.path.join(HERE, "claude_seat_world.py"),
                                     "fresh"], text=True).strip().splitlines()[-1].strip()
    os.environ["CLAUDE_SEAT_WORLD"] = world
    import claude_ci as ci
    import claude_denoise_price as price
    # a detached run ignores SIGINT; make SIGTERM stop the seat on the way out
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
    base = dict(DISPLAY_OFF, **spec.get("base", {}))
    price.start_seat()
    log = open(os.path.join(out, "runs.jsonl"), "a")
    doors = None
    try:
        for arm in spec["arms"]:
            if arm.get("doors") is not None and arm["doors"] != doors:
                ci.set_doors(bool(arm["doors"]))
                doors = arm["doors"]
                time.sleep(3)
            for seed in arm.get("seeds", spec.get("seeds", [1, 2])):
                name = "%s_s%d" % (arm["name"], seed)
                dials = dict(base, **arm.get("dials", {}), claude_rng_seed=seed)
                p = arm.get("pose", spec.get("pose"))
                cmd = [sys.executable, os.path.join(HERE, "claude_shoot.py"), "--skip-seat",
                       "--pin", "--play", "--pos"] + [str(v) for v in p[:3]] + [
                       "--yaw", str(p[3]), "--pitch", str(p[4]),
                       "--time", str(arm.get("time", spec.get("time", 0.5))),
                       "--frames", str(arm.get("frames", spec.get("frames", 4096))),
                       "--name", name, "--accum-dump", os.path.join(out, name)]
                if arm.get("export"):
                    cmd += ["--export-trace", os.path.join(out, name + "_trace")]
                for k, v in dials.items():
                    cmd += ["--dial", "%s=%s" % (k, v)]
                t0 = time.time()
                r = subprocess.run(cmd, capture_output=True, text=True)
                png = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
                rec = {"name": name, "arm": arm["name"], "seed": seed, "rc": r.returncode,
                       "secs": round(time.time() - t0, 1), "png": png, "dials": dials,
                       "pose": p, "doors": doors, "tail": r.stdout[-600:] + r.stderr[-600:]}
                try:
                    st = json.load(open(png.replace(".png", ".capture.json")))["stats"]
                    rec["stats"] = {k: st.get(k) for k in (
                        "still_frames", "claude_nee", "claude_bounces", "claude_air_scatter",
                        "claude_air_absorb", "claude_sky_uniform", "sun_lux", "sky_lux",
                        "sky_horizon", "sky_zenith", "grid_origin", "reset_why", "sun_dir",
                        "features", "air_clear", "far_box", "cam_ray")}
                except Exception as e:
                    rec["stats_err"] = str(e)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(name, "rc", r.returncode, rec["secs"], "s", flush=True)
    finally:
        ci.stop_seat()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    c, rest = sys.argv[1], sys.argv[2:]
    if c == "run":
        cmd_run(rest[0], rest[1])
    elif c == "referee":
        cmd_referee(rest[0], rest[1:])
    elif c == "pad":
        cmd_pad(rest[0], rest[1:])
    elif c == "compare":
        cmd_compare(rest[0], rest[1:])
    elif c == "noise":
        cmd_noise(rest[0], rest[1:])
    else:
        sys.exit(__doc__)
