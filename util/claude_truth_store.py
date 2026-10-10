#!/usr/bin/env python3
"""claude_truth_store -- render a playtest's truth once, keep it, and CHECK it
instead of re-rendering it (2026-10-09).

John: "preserve the truth and reuse it. We do need a way to know when the
truth changes tho. Not every change invalidates the truth..anything that
helps our guessing or ML shouldn't..but changing the rendering rules overall
would? ... checks just a handful of frames, and then only regenerates truth
if the truth frames are changed".

The truth video of a camera path (each pose held `hold` frames) was the bulk
of a motion playtest's GPU time: 15-20 minutes per scenario, re-rendered for
every run. Now:

  * KEY = the INPUTS only: the scenario, its camera path (frame keys), hold,
    scale and any dials set on every run. The engine version is deliberately
    NOT in the key: whether the engine changed the light is measured, below.
    Display caches and guesses (history, denoisers, the ledger, ML) are off
    in the truth runs by construction, so they never reach it.
  * CHECK = a few poses from the moving part of the stored path, each led by
    its predecessors so the carried history resembles the stored video's,
    rendered at the same hold with a seed the stored truth never uses
    (independent noise; the renderer is deterministic per seed, so the same
    seed would pass an unchanged engine trivially). Compared with the stored
    frames: whole-pixel error, 8x8-block error (noise averages out, a change
    in the light does not) and overall brightness.
  * FLOOR = the same check, measured right after the truth is stored. A later
    check fails when its error exceeds CHECK_K x the floor.
  * history.jsonl in the store records every store, pass and fail with its
    numbers: the data the TUNED values below are to be learned from.
"""
import hashlib
import json
import os
import shutil
import time

import numpy as np

ROOT = os.path.expanduser("~/data/luanti-truth")
CHECK_FRAMES = 4    # TUNED: poses per check | learn by: smallest count that still fails every known rule change in history.jsonl
CHECK_LEAD = 0      # predecessors before each check pose: none since TRUTH_DEF 2 (each pose starts over, so nothing carries in) | learn by: floor_blk against lead
CHECK_SEED = 101    # the stored truth renders at seed 0
FLOOR_SEEDS = (101, 103, 105)  # TUNED: the floor is the worst of these | learn by: self-test passes at unused seeds (102, 104...) in history.jsonl
CHECK_K = 1.5       # TUNED: fail above K x floor | learn by: history.jsonl passes on no-change vs fails on known changes
EPS = 1e-4          # linear units: below this an error is 8-bit rounding, not a change
BLOCK = 8
FPS_MATCH = 0.25    # TUNED: a stored path is reused when its fps is within 25 % of the measured (10 % missed: park fps swings more than that run to run, 2026-10-09) | learn by: playtest scores at fps x1.25 vs x1.0 (does the pace error move them?)
EPS_B = 0.005       # TUNED: brightness tolerance on top of K x the floor's | learn by: history.jsonl (the smallest real change it must catch)
# THE TRUTH'S DEFINITION. 2 = each pose converges on its own (game.cpp
# claude_path_hold_fresh, 2026-10-09); 1 = history from earlier poses rode
# along (up to 3 % in a turn). A truth made under another definition is
# never matched: bump this when the reference mode changes.
TRUTH_DEF = 3
# 3 (2026-10-09): THE DENOISER IS OFF in every truth run. Until then the
# truth was 256 frames per pose PLUS today's denoiser: a filtered picture,
# not a converged one (seen when one test forced the denoiser off: indoors,
# 256 raw frames are grainy). The truth is the plain average of real paths,
# held long enough to converge (claude_playtest --hold, default 4096).
# 4 (2026-10-09, John: "make sure that doesn't happen again"): the ENGINE'S
# truth mode (claude_truth=1) instead of a list of dials remembered to turn
# off. Three guards, each catching what the others miss:
#   1. truth mode forces every DISPLAY-side feature off (game.cpp
#      claudeTruthForce), and every dial the engine watches must be
#      classified in DIAL_CLASS below, a display dial must be one truth mode
#      forces (dial_problems): a new feature cannot slip in unclassified;
#   2. every frame records what rendered it (the dump row's "features"), and
#      a truth whose rows show anything but truth mode is never stored or
#      reused (feature_problems);
#   3. a deliberately dumb renderer (CONTROL_DIALS, the scoreboard's control)
#      run long at one pose must agree with the stored truth (control_verdict):
#      it catches a truth that is wrong for a reason nobody listed.
# A TRUTH_DEF 3 truth (made with the dials list) is ADMITTED to 4 only by
# passing a truth-mode check and the control, never by being 3.
TRUTH_DEF = 5
# 5 (2026-10-09, John: "we don't need the whole video ... doesn't flicker and
# is close to the truth are different things"): the truth is a handful of
# poses (truth_frames), each held as long as the arms need (hold_needed);
# flicker is measured on the real-time video alone. ~70 converged poses at
# 4096 frames became ~10 at a few hundred: hours to minutes.
TRUTH_MID = 8      # TUNED: poses through the move | learn by: the spread of the moving-frame numbers over sampled subsets
TRUTH_DIALS = ["claude_truth=1"]
FORCED_BY_TRUTH = {"claude_denoise", "claude_denoise_learned", "claude_ledger", "claude_boost",
                   "claude_split", "claude_raw_frame", "claude_reproject", "claude_bounces"}
FEATURE_OFF = ("denoise", "denoise_learned", "ledger", "boost", "split", "raw_frame", "reproject")
CONTROL_DIALS = ["claude_nee=0", "claude_bounce_uniform=1", "claude_torch_nee=0", "claude_area_nee=0",
                 "claude_guide=0", "claude_area_pick=0", "claude_area_skip=0", "claude_boost=0"]
CONTROL_HOLD = 32768   # TUNED: the scoreboard check's frame count | learn by: the control's own two-seed spread at this count
# DIAL CLASSES (filled from the shader, 2026-10-09). physics / geometry /
# estimator / eye: the same in truth as in play; display: must be in
# FORCED_BY_TRUTH; debug: instrument views and plants; dead: read, never used.
DIAL_CLASS = {
    # display: forced off by truth mode (FORCED_BY_TRUTH), or inert once it is
    "claude_denoise": "display", "claude_denoise_learned": "display", "claude_ledger": "display",
    "claude_boost": "display", "claude_split": "display", "claude_raw_frame": "display",
    "claude_reproject": "display", "claude_bounces": "display",
    "claude_denoise_young": "display-inert",   # the denoiser's fade-in: nothing to fade once it is off
    "claude_motion_alpha": "display-inert",    # the history floor while reprojecting: off with reproject
    "claude_truth": "truth",
    # estimators: other unbiased ways to sample the same light (the control catches a bias)
    "claude_rng": "estimator", "claude_nee": "estimator", "claude_torch_nee": "estimator",
    "claude_area_nee": "estimator", "claude_area_pick": "estimator", "claude_area_skip": "estimator",
    "claude_guide": "estimator", "claude_guide_deposit": "estimator", "claude_guide_keep": "estimator",
    "claude_guide_impl": "estimator", "claude_guide_alpha": "estimator", "claude_bounce_uniform": "estimator",
    # geometry: the world's shape walked differently, the same picture
    "claude_pyramid": "geometry", "claude_descend": "geometry", "claude_model_far": "geometry",
    "claude_bricks": "geometry", "claude_bricks_far": "geometry", "claude_walk_exact": "geometry",
    # physics: what the true picture is
    "claude_glass_flush": "physics", "claude_texel_colour": "physics", "claude_body_colour": "physics",
    "claude_sun_redden": "physics", "claude_air_scatter": "physics", "claude_air_absorb": "physics",
    "claude_air_g": "physics", "claude_water_absorb": "physics", "claude_units": "physics",
    "claude_flame": "physics", "claude_leaf_transmit": "physics",
    # the eye: applied identically to the truth and to play
    "claude_exposure": "eye", "claude_auto_exposure": "eye", "claude_present_guide": "eye", "claude_white_balance": "eye",
    "claude_night_vision": "eye", "claude_adapt_colour": "eye", "claude_adapt_brighter": "eye",
    "claude_adapt_darker": "eye",
    # debug: instrument views, planted defects, test skies
    "claude_grid_debug": "debug", "claude_view": "debug", "claude_sky_uniform": "debug",
    "claude_tree_plant": "debug", "claude_tree_variant": "debug", "claude_tree_dirs": "debug",
    # dead: read and pushed, read by nothing (the five-pass chain and the raster
    # path; classification 2026-10-09). Clean-up candidates.
    "claude_water_reflections": "dead", "claude_gi": "dead", "claude_gi_split": "dead",
    "claude_clay": "dead", "claude_texture": "dead", "claude_gray": "dead", "claude_pure": "dead",
    "claude_bevel": "dead", "claude_parallax": "dead", "claude_jitter": "dead",
    "claude_skybounce": "dead", "claude_sun_angle": "dead", "claude_night_sky": "dead",
    "claude_moon_gain": "dead", "claude_bounce2": "dead", "claude_cache_sky": "dead",
    "claude_bisect": "dead", "claude_nee_gate": "dead", "claude_cost": "dead",
    "claude_face_direct": "dead", "claude_tiers": "dead", "claude_bounce_stride": "dead",
    "claude_face_texels": "dead", "claude_cache_remap": "dead", "claude_far_hist": "dead",
    "claude_light_ladder": "dead", "claude_lod_dither": "dead", "claude_far_grain": "dead",
    "claude_far_fog": "dead", "claude_sky_azimuth": "dead", "claude_subvox": "dead",
    "claude_refine": "dead",
    "exposure_compensation": "dead", "golden_hour_strength": "dead", "ssao_strength": "dead",
    "bump_strength": "dead",
}
# NOT a dial, part of what "truth" means here: the trace runs at half
# resolution (claude_trace_scale 0.5) and claude_present upsamples it with a
# joint-bilateral blend of 4 texels -- identical in the truth and in play.


def dial_names(repo):
    """every dial the engine watches (game.cpp SETTING_CALLBACKS)"""
    import re
    src = open(os.path.join(repo, "src", "client", "game.cpp")).read()
    blk = src[src.index("SETTING_CALLBACKS[] = {"):]
    return re.findall(r'"([a-z0-9_]+)"', blk[:blk.index("};")])


def dial_problems(repo):
    """reasons no truth may be stored: an unclassified dial, or a display
    dial truth mode does not force off"""
    out = []
    for d in dial_names(repo):
        c = DIAL_CLASS.get(d)
        if c is None:
            out.append("%s is not classified (claude_truth_store.DIAL_CLASS)" % d)
        elif c == "display" and d not in FORCED_BY_TRUTH:
            out.append("%s is display-side and truth mode does not force it off" % d)
    return out


def held_problems(dump_dir):
    """reasons a dumped reference is not a held truth: every row must sit on
    its own path frame (0, 1, 2, ...) and have converged its full hold (a
    path cancelled mid-way dumped frames at path_frame -1, unheld)"""
    rows = _rows(dump_dir)
    out = []
    hold = max(r["still_frames"] for r in rows) if rows else 0
    for k, r in enumerate(rows):
        if r["path_frame"] != k:
            out.append("frame %d is on path frame %s" % (r["i"], r["path_frame"]))
        elif k == 0 or _pose(r) != _pose(rows[k - 1]):
            if r["still_frames"] < truth_hold_floor(rows):
                out.append("frame %d held %d frames" % (r["i"], r["still_frames"]))
        if len(out) >= 3:
            break
    return out


def truth_hold_floor(rows):
    """the hold every new pose of a reference must have reached: the most
    common still_frames over its new poses"""
    st = [r["still_frames"] for k, r in enumerate(rows) if k == 0 or _pose(r) != _pose(rows[k - 1])]
    return max(set(st), key=st.count) if st else 0


def feature_problems(dump_dir):
    """reasons a dumped video is not a truth: a row without the record, not
    in truth mode, or with a display feature on"""
    out = []
    for r in _rows(dump_dir):
        f = r.get("features")
        if f is None:
            out.append("frame %d has no feature record" % r["i"])
        elif f.get("truth") != 1:
            out.append("frame %d not in truth mode" % r["i"])
        else:
            on = [k for k in FEATURE_OFF if f.get(k, 1) != 0]
            if f.get("bounces") != 24:
                on.append("bounces=%s" % f.get("bounces"))
            if not f.get("rng", 0) >= 1:
                on.append("rng=%s" % f.get("rng"))
            if on:
                out.append("frame %d has %s on" % (r["i"], ",".join(on)))
        if len(out) >= 3:
            break
    return out


CONTROL_SEEDS = (201, 202)   # two control renders: their disagreement is the control's own noise
CONTROL_K = 2.5              # TUNED: agree within K x the control's tile noise | learn by: history.jsonl control verdicts on known-good and known-broken truths
CONTROL_KMEAN = 3.0          # TUNED: the whole-frame mean within this many sigma of the control's mean noise


def control_verdict(ctrl_dirs, ref_dir, idx):
    """the controls' frames against the stored truth's frame `idx`. The
    fixed tolerance alone (1 % mean, 5 % median 16x16 tile: the scoreboard
    rule) assumed a clean control; in a dark room lit through a 1x1 hole the
    control (no light sampling) finds the sun by luck and was 3.4x noisier
    than the truth after 32768 frames, so it "disagreed" by its own noise
    (2026-10-09, cave-turn). With two control seeds their disagreement
    measures that noise: agree when within the fixed tolerance OR within
    CONTROL_K (tiles) / CONTROL_KMEAN (mean) times the control's noise."""
    if isinstance(ctrl_dirs, str):
        ctrl_dirs = [ctrl_dirs]
    t = _rows(ref_dir)
    b = _frame(ref_dir, t[idx])
    lum = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])
    k = 16

    def tiles(x):
        l = lum(x)
        h, w = l.shape
        return l[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k).mean((1, 3))

    cs, feats = [], []
    for c in ctrl_dirs:
        cr = _rows(c)
        if not _near(_pose(cr[-1]), _pose(t[idx])):
            return False, {"why": "control pose differs"}
        cs.append(_frame(c, cr[-1]))
        feats.append(cr[-1].get("features", {}).get("truth"))
    tt = tiles(b)
    ct = [tiles(c) for c in cs]
    cm = sum(ct) / len(ct)
    mb = float(lum(b).mean())
    means = [float(lum(c).mean()) for c in cs]
    ratio = (sum(means) / len(means)) / max(mb, 1e-9)
    mad = float(np.median(np.abs(cm - tt) / np.maximum(tt, 1e-6)))
    info = {"ratio": ratio, "tile_mad": mad, "frame": idx, "controls": len(cs),
            "control_truth_mode": all(f == 1 for f in feats)}
    ok_fixed = abs(ratio - 1) < 0.01 and mad < 0.05
    ok_noise = False
    if len(ct) >= 2:
        sig_t = np.abs(ct[0] - ct[1]) / 2.0                 # noise of the mean of two
        noise_mad = float(np.median(sig_t / np.maximum(tt, 1e-6)))
        sig_m = abs(means[0] - means[1]) / 2.0 / max(mb, 1e-9)
        info.update(noise_tile=noise_mad, noise_mean=sig_m)
        ok_noise = mad <= CONTROL_K * noise_mad and abs(ratio - 1) <= max(0.01, CONTROL_KMEAN * sig_m)
    info["agree_by"] = "tolerance" if ok_fixed else ("control noise" if ok_noise else None)
    return (ok_fixed or ok_noise) and info["control_truth_mode"], info


def find(spec_base, fps):
    """the stored truth for these inputs whose path was laid out at the fps
    nearest the measured one, within FPS_MATCH; None if none"""
    best = None
    if not os.path.isdir(ROOT):
        return None
    for k in os.listdir(ROOT):
        m = os.path.join(ROOT, k, "manifest.json")
        if not os.path.exists(m):
            continue
        man = json.load(open(m))
        sp = dict(man["spec"])
        f = sp.pop("fps_path", None)
        sp.pop("keys", None)
        admit = False
        if sp.get("truth_def") == 3 and spec_base.get("truth_def") == 4:
            sp["truth_def"] = 4          # a candidate for admission, not a match
            admit = True
        if sp != spec_base or not f or not all(os.path.isdir(man[x]) for x in ("reference", "faces")):
            continue
        man["admit"] = admit
        if not all(fl.get("seeds", 1) >= len(FLOOR_SEEDS) for fl in man["floor"]):
            continue   # a floor from fewer seeds under-reads the noise
        err = abs(f / fps - 1.0)
        if err <= FPS_MATCH and (best is None or err < best[0]):
            best = (err, man)
    return best[1] if best else None


def key_for(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def lookup(key):
    m = os.path.join(ROOT, key, "manifest.json")
    if not os.path.exists(m):
        return None
    man = json.load(open(m))
    for k in ("reference", "faces"):
        if not os.path.isdir(man[k]):
            return None
    return man


def log(event):
    os.makedirs(ROOT, exist_ok=True)
    event = dict(event, t=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(os.path.join(ROOT, "history.jsonl"), "a") as f:
        f.write(json.dumps(event) + "\n")


def _rows(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
    rows.sort(key=lambda r: r["i"])
    return rows


def _frame(d, r):
    a = np.fromfile(os.path.join(d, "%05d.rgba" % r["i"]), np.uint8)
    x = a.reshape(r["h"], r["w"], 4)[::-1, :, :3].astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _pose(r):
    return (r["pos"][0], r["pos"][1], r["pos"][2], r["yaw"], r["pitch"])


def path_pose(keys, fr):
    """game.cpp claudePathPose, in the same float32 arithmetic: the exact
    pose of path frame `fr` (the dump's logged pose has 6 digits only)"""
    f32 = np.float32
    if fr <= keys[0][0]:
        return tuple(float(f32(v)) for v in keys[0][1:])
    for i in range(1, len(keys)):
        if fr <= keys[i][0]:
            a, b = [f32(v) for v in keys[i - 1]], [f32(v) for v in keys[i]]
            t = f32(f32(fr) - a[0]) / max(f32(1e-6), f32(b[0] - a[0]))
            dyaw = f32(np.fmod(f32(b[4] - a[4] + f32(540.0)), f32(360.0)) - f32(180.0))
            return tuple(float(v) for v in (a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t,
                                            a[3] + (b[3] - a[3]) * t, a[4] + dyaw * t,
                                            a[5] + (b[5] - a[5]) * t))
    return tuple(float(f32(v)) for v in keys[-1][1:])


def _near(p, q):
    return all(abs(a - b) < 1e-2 for a, b in zip(p, q))


def check_plan(ref_dir, keys):
    """the stored frames to check: evenly over the frames where the pose
    changed (a held pose keeps converging, so those frames are older than
    `hold`; the moving frames are each exactly `hold` frames old), each led
    by its CHECK_LEAD predecessors; poses exact from the path keys"""
    rows = _rows(ref_dir)
    moving = [i for i in range(1, len(rows)) if _pose(rows[i]) != _pose(rows[i - 1])] or [1]
    n = min(CHECK_FRAMES, len(moving))
    idxs = sorted({moving[int((j + 0.5) * len(moving) / n)] for j in range(n)})
    poses = []
    for i in idxs:
        for k in range(CHECK_LEAD + 1):
            r = rows[max(0, i - CHECK_LEAD + k)]
            p = path_pose(keys, r["path_frame"])
            if not _near(p, _pose(r)):
                raise RuntimeError("path_pose disagrees with the dump at frame %d: %r vs %r"
                                   % (r["i"], p, _pose(r)))
            poses.append(p)
    return idxs, poses


def write_check_path(poses, path):
    with open(path, "w") as f:
        for j, p in enumerate(poses):
            f.write("%d %r %r %r %r %r\n" % ((j,) + tuple(p)))


def _blocks(x):
    h, w = x.shape[0] // BLOCK * BLOCK, x.shape[1] // BLOCK * BLOCK
    return x[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK, 3).mean((1, 3))


def compare(check_dir, ref_dir, idxs):
    """per checked pose: the check's frame (last of its lead-in) against the
    stored frame; refuses frames with no signal (a black pair "agrees")"""
    crow, rrow = _rows(check_dir), _rows(ref_dir)
    out = []
    for c, i in enumerate(idxs):
        j = c * (CHECK_LEAD + 1) + CHECK_LEAD
        if j >= len(crow):
            return None
        a, b = _frame(check_dir, crow[j]), _frame(ref_dir, rrow[i])
        if not _near(_pose(crow[j]), _pose(rrow[i])):
            return None
        out.append({"frame": i, "px": float(np.sqrt(((a - b) ** 2).mean())),
                    "blk": float(np.sqrt(((_blocks(a) - _blocks(b)) ** 2).mean())),
                    "mean_ratio": float(a.mean() / max(b.mean(), 1e-9)),
                    "ref_mean": float(b.mean()), "still": [crow[j]["still_frames"], rrow[i]["still_frames"]]})
    return out


def worst(floors):
    """per frame, the largest error and the largest brightness deviation
    over several floor renders; None if any of them failed"""
    if not floors or any(f is None for f in floors):
        return None
    out = []
    for fr in zip(*floors):
        w = dict(fr[0])
        w["px"] = max(f["px"] for f in fr)
        w["blk"] = max(f["blk"] for f in fr)
        dev = max(fr, key=lambda f: abs(f["mean_ratio"] - 1.0))
        w["mean_ratio"] = dev["mean_ratio"]
        w["seeds"] = len(fr)
        out.append(w)
    return out


HOLD_FIRST = 1024        # TUNED: a scenario's first truth, before any arm has been measured against one | learn by: hold_needed's answers
HOLD_MIN, HOLD_MAX = 256, 16384   # TUNED: bounds on hold_needed
NOISE_SHARE = 0.32      # the truth's noise under this fraction of the best arm's error adds < 5 % to it (sqrt(1 + 0.32^2) = 1.05): a rule


def truth_frames(keys):
    """the path frames a truth renders: the start pose, TRUTH_MID poses spread
    over the move, the end pose"""
    F = int(keys[-1][0]) + 1
    poses = [path_pose(keys, f) for f in range(F)]
    moving = [f for f in range(1, F) if poses[f] != poses[f - 1]]
    mid = [moving[int((j + 0.5) * len(moving) / TRUTH_MID)] for j in range(min(TRUTH_MID, len(moving)))]
    return sorted(set([0] + mid + [F - 1]))


def truth_hold(ref_dir, default=0):
    """the hold a stored truth was rendered at (its frames' still_frames on
    moving poses), so its check renders the same"""
    rows = _rows(ref_dir)
    st = [r["still_frames"] for i, r in enumerate(rows) if i == 0 or _pose(r) != _pose(rows[i - 1])]
    return int(min(st)) if st else (default or HOLD_FIRST)


def note_arm_error(manifest_path, arm, rms):
    """the scorer records each arm's error against this truth (for hold_needed)"""
    m = json.load(open(manifest_path))
    m.setdefault("arm_rms", {})[arm] = rms
    json.dump(m, open(manifest_path, "w"), indent=1)


def hold_needed(scenario):
    """frames per pose for a NEW truth of this scenario: from the last stored
    truth of it, its noise (floor) at its hold and the best arm's error
    against it, the hold that puts the noise under NOISE_SHARE of that error
    (noise falls as 1/sqrt(frames)). HOLD_FIRST with no history."""
    best = None
    if os.path.isdir(ROOT):
        for k in os.listdir(ROOT):
            mp = os.path.join(ROOT, k, "manifest.json")
            if not os.path.exists(mp):
                continue
            m = json.load(open(mp))
            if m["spec"].get("scenario") != scenario or not m.get("arm_rms") or not m.get("floor"):
                continue
            if best is None or m["stored"] > best["stored"]:
                best = m
    if best is None:
        return HOLD_FIRST
    sig = float(np.median([f["px"] for f in best["floor"]])) / np.sqrt(2)
    e = min(best["arm_rms"].values())
    n_ref = truth_hold(best["reference"])
    need = n_ref * (sig / (NOISE_SHARE * e)) ** 2
    return int(min(HOLD_MAX, max(HOLD_MIN, 2 ** int(np.ceil(np.log2(max(need, 1)))))))


def build_hash(repo):
    """what decides the picture: the engine binary, the shaders, the block
    models (content), and the game's files (names, sizes, dates: it is
    large). The same build gives the same answer to the truth check, so a
    truth checked under this hash need not be checked again (2026-10-09:
    the check was ~5 min of every playtest scenario)."""
    h = hashlib.sha256()
    def add_file(f):
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    add_file(os.path.join(repo, "bin", "luanti"))
    for top in ("client/shaders", "util/claude_models"):
        for dp, dn, fn in sorted(os.walk(os.path.join(repo, top))):
            dn.sort()
            for f in sorted(fn):
                h.update(os.path.join(dp, f)[len(repo):].encode())
                add_file(os.path.join(dp, f))
    game = os.path.join(repo, "games", "mineclonia")
    for dp, dn, fn in sorted(os.walk(game, followlinks=True)):
        dn.sort()
        for f in sorted(fn):
            st = os.stat(os.path.join(dp, f))
            h.update(("%s %d %d" % (os.path.join(dp, f)[len(game):], st.st_size, int(st.st_mtime))).encode())
    return h.hexdigest()[:16]


def mark_verified(man, build):
    """this build passed (or made) the truth: remember it in the manifest"""
    m = os.path.join(os.path.dirname(man["reference"]), "manifest.json")
    disk = json.load(open(m))
    vb = disk.setdefault("verified_builds", [])
    if build not in vb:
        vb.append(build)
    json.dump(disk, open(m, "w"), indent=1)
    man["verified_builds"] = vb


def admit(man, control):
    """a TRUTH_DEF 3 truth that passed a truth-mode check and the control
    becomes a 4: its manifest says so, and how"""
    m = os.path.join(os.path.dirname(man["reference"]), "manifest.json")
    disk = json.load(open(m))
    disk["spec"]["truth_def"] = 4
    disk["admitted"] = {"from_def": 3, "control": control, "t": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(disk, open(m, "w"), indent=1)


def verdict(stats, floor):
    """True = the truth stands. Every pose must be within K x its floor."""
    bad = []
    for s, f in zip(stats, floor):
        for m in ("px", "blk"):
            if s[m] > CHECK_K * f[m] + EPS:
                bad.append("frame %d %s %.5f > %.1f x floor %.5f" % (s["frame"], m, s[m], CHECK_K, f[m]))
        db, fb = abs(s["mean_ratio"] - 1.0), abs(f["mean_ratio"] - 1.0)
        if db > CHECK_K * fb + EPS_B:
            bad.append("frame %d brightness x%.4f (floor x%.4f)" % (s["frame"], s["mean_ratio"], f["mean_ratio"]))
    return not bad, bad


def store(key, spec, exposure, ref_dir, faces_dir, idxs, floor, extra=None):
    """move the dumps into the store (a rename: same disk) and write the
    manifest; an older entry under the key is removed first"""
    d = os.path.join(ROOT, key)
    if os.path.isdir(d):
        shutil.rmtree(os.path.join(ROOT, key))
    os.makedirs(d)
    ref = os.path.join(d, "reference")
    faces = os.path.join(d, "faces")
    shutil.move(ref_dir, ref)
    shutil.move(faces_dir, faces)
    man = dict(spec=spec, key=key, exposure=exposure, reference=ref, faces=faces,
               check_frames=idxs, floor=floor, stored=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               check=dict(frames=CHECK_FRAMES, lead=CHECK_LEAD, seed=CHECK_SEED, k=CHECK_K, block=BLOCK),
               **(extra or {}))
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    return man
