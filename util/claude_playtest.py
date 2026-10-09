#!/usr/bin/env python3
"""claude_playtest — what John sees when he plays, measured automatically.

John, 2026-10-06, after playing: "so grainy ... it doesn't do well when you
back up and stuff fills in from out of view, like it starts off all dark",
and the frame rate was 6 fps. "You need some tests ... you should be able to
detect this yourself, automatically." Today's motion tests ran in photo
settings, at a fixed distance per frame, on the bright plains -- none of
those three things.

So, per scenario:
  * PLAY SETTINGS: the config scrubbed exactly as claude_look.sh does (the
    renderer runs the game's defaults: NEE, denoiser, units, eye...), a
    headless seat, only the capture mechanics pushed (claude_motion --play).
  * REAL SPEED: the frame rate is measured at the start pose first and the
    path laid out in SECONDS (walking 4 m/s, turning 90 deg/s), so at 6 fps a
    frame moves as far as it did for John.
  * EXPOSURE FIXED at what play's auto exposure chose at the start pose, in
    both arms, so the eye adapting over time is not counted as rendering.
  * ARMS: real time (dumped every frame) and the reference (each pose held
    to HOLD frames: the converged picture at the same pose), plus the
    face-ID view per pose for the reveal mask.

Reported, per scenario:
  fps p50 / p99, frames over 2x the median        (the 6 fps)
  JOD of the real-time video vs the reference     (the grain in motion)
  REVEALED pixels: brightness vs the truth by frames since revealed
                                                   ("starts off all dark")
  frames after the stop until JOD >= 9 per frame   (settling when you stare)
and a GIF: real time | reference, side by side.

Pass/fail thresholds are John's (objectives): this reports numbers.

THE TRUTH IS STORED (2026-10-09, util/claude_truth_store.py): rendered once
per scenario and path, then each run re-renders a few of its poses at another
seed and reuses it if they agree within the noise measured when it was stored;
otherwise it is rendered again and replaced. --fresh-truth forces a render;
--truth-check-only [--truth-check-seed N] [--truth-check-dial k=v] tests the
check itself (another seed must pass; a change to the light rules must fail).

An A/B is ONE run with several --arm: the arms share the path and the truth
(rendered once), and each starts from the same settled, reset state.

  python3 util/claude_playtest.py [--only backup ...] [--hold 256]
  python3 util/claude_playtest.py --arm off: --arm m5:claude_ledger=5
  ~/.venvs/judge/bin/python util/claude_playtest.py --score RUN_DIR
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT = os.path.join(REPO, "screenshots", "playtest")
import claude_truth_store as TS

WALK = 4.0          # m/s (Luanti's walking speed)
TURN = 90.0         # deg/s

# name: start pose (x, y, z, yaw, pitch), then moves: ("hold", s) |
# ("move", dx, dz, s) in metres | ("turn", dyaw, s)
SCENARIOS = {
    "backup": ((5, 8.5, -20, 0, -10), [("hold", 1.0), ("move", 0, -6, 1.5), ("hold", 1.5)]),
    "forward": ((5, 8.5, -32, 0, -10), [("hold", 1.0), ("move", 0, 6, 1.5), ("hold", 1.5)]),
    "turn": ((60, 44.5, 60, 45, -10), [("hold", 1.0), ("turn", 90, 1.0), ("hold", 1.5)]),
    "forest-walk": ((146.5, 8.5, 123.5, 270, -5), [("hold", 1.0), ("move", 5, 0, 1.25), ("hold", 1.5)]),
    # INDOORS, where bounce light dominates (2026-10-07): the doorway room,
    # backing 1.5 m straight away from the doorway (yaw 315 looks +x +z);
    # 3 m put the camera inside the back wall (all-black frames, 2026-10-07)
    "room-backup": ((241.7, 8.5, 231.7, 315, -12), [("hold", 1.0), ("move", -1.06, -1.06, 1.0), ("hold", 1.5)]),
}

DEFER_RE = r'defer = \[(.*?)\]'


def scrub_conf_like_look():
    """remove the renderer keys claude_look.sh removes, so the game's own
    defaults apply (its `defer` list, read from the script itself)"""
    look = open(os.path.join(HERE, "claude_look.sh")).read()
    keys = re.findall(r'"(claude_[a-z_]+)"', re.search(DEFER_RE, look, re.S).group(1))
    conf = os.path.join(REPO, "minetest.conf")
    s = open(conf).read()
    for k in keys + ["claude_pyramid", "claude_far_levels", "claude_far_plants",
                     "claude_far_leaf_medium", "claude_far_skyexit", "claude_far_only",
                     "claude_pixel_reset", "claude_world_clamp", "claude_grid_drain",
                     "claude_path_hold", "claude_dump_scale", "claude_rng_seed"]:
        s = re.sub(r"(?m)^%s\s*=.*\n?" % re.escape(k), "", s)
    open(conf, "w").write(s)
    open(os.path.join(REPO, "claude_settings_patch.conf"), "w").close()
    open(os.path.join(REPO, "worlds", "gallery", "claude_dial.conf"), "w").close()
    return keys


def path_frames(start, moves, fps):
    """keys "frame x y z yaw pitch" with the moves laid out in seconds"""
    x, y, z, yaw, pitch = start
    f = 0
    keys = [(f, x, y, z, yaw, pitch)]
    for m in moves:
        if m[0] == "hold":
            f += max(1, round(m[1] * fps))
        elif m[0] == "move":
            f += max(1, round(m[3] * fps))
            x += m[1]; z += m[2]
        elif m[0] == "turn":
            f += max(1, round(m[2] * fps))
            yaw += m[1]
        keys.append((f, x, y, z, yaw, pitch))
    return keys


def run(cmd):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip().splitlines()


def game_default(member):
    """a dial's default as the engine source declares it (game.cpp
    `float m_<member> = <v>f;`); refuses rather than guess"""
    src = open(os.path.join(REPO, "src", "client", "game.cpp")).read()
    m = re.search(r"float m_%s = ([0-9.]+)f;" % re.escape(member), src)
    if not m:
        sys.exit("REFUSED: no default for m_%s in game.cpp" % member)
    return m.group(1)


def capture(args):
    import claude_lab as lab
    global PLAY_DENOISE
    PLAY_DENOISE = ["--dial", "claude_denoise=1",
                    "--dial", "claude_denoise_learned=" + game_default("denoise_learned")]
    run_dir = os.path.join(OUT, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir)
    keys = scrub_conf_like_look()
    # --arm NAME:k=v,k=v (repeatable); --rt-dial alone is one arm named "rt"
    arms = [(a.split(":", 1)[0], [kv for kv in a.split(":", 1)[1].split(",") if kv] if ":" in a else [])
            for a in (args.arm or [])] or [("rt", list(args.rt_dial or []))]
    if len({a for a, _ in arms}) != len(arms):
        sys.exit("REFUSED: two arms share a name: %r" % [a for a, _ in arms])
    arm_keys = sorted({kv.split("=")[0].strip() for _, kvs in arms for kv in kvs})
    meta = {"deferred_to_game_defaults": keys, "scenarios": {},
            "arms": {a: kvs for a, kvs in arms}, "dials_all_runs": args.dial or []}
    first = True
    nonlocal_first = [True]
    for name, (start, moves) in SCENARIOS.items():
        first = first or nonlocal_first[0]
        nonlocal_first[0] = False
        if args.only and name not in args.only:
            continue
        d = os.path.join(run_dir, name)
        os.makedirs(d)
        x, y, z, yaw, pitch = start
        # 1. park there in play settings and read fps + the eye's exposure
        pin = os.path.join(d, "park.txt")
        open(pin, "w").write("0 %s %s %s %s %s\n" % (x, y, z, yaw, pitch))
        # extra dials for an A/B arm (--dial k=v), on every capture of the run
        extra = sum([["--dial", kv] for kv in (args.dial or [])], [])
        out = run(["python3", "util/claude_motion.py", "--play", "--path", pin, "--nodump",
                   "--frames", "120", "--name", name + "-park"] + extra + ([] if first else ["--skip-seat"]))
        if first:
            lab.rpc("abm", on=False)
            time.sleep(60)            # the world-file reader fills the far levels
        first = False
        lab.goto({"pos": [x, y, z], "yaw": yaw, "pitch": pitch})
        open(lab.PATCH, "w").write("claude_path = %s\n" % pin)
        time.sleep(6)
        st = lab.read_stats() or {}
        fps = max(2.0, float(st.get("fps") or 30.0))
        expo = (st.get("auto_exposure") or [None])[0]
        # 2. the path in seconds at that fps -- or, when a truth is stored for
        # these inputs at an fps within 10 % of it, THAT path (and its truth)
        spec_base = {"scenario": name, "start": list(start), "moves": [list(m) for m in moves],
                     "hold": args.hold, "scale": 2, "all_run_dials": sorted(args.dial or []),
                     "truth_def": TS.TRUTH_DEF}
        man = None if args.fresh_truth else TS.find(spec_base, fps)
        if man:
            fps_path = man["spec"]["fps_path"]
            pk = [tuple(k) for k in man["spec"]["keys"]]
        else:
            fps_path = round(fps, 3)
            pk = path_frames(start, moves, fps_path)
        pf = os.path.join(d, "path.txt")
        with open(pf, "w") as f:
            for k in pk:
                f.write("%d %r %r %r %r %r\n" % tuple(k))
        spec = dict(spec_base, fps_path=fps_path, keys=[list(k) for k in pk])
        key = TS.key_for(spec)
        expo_measured = expo
        if man:
            expo = man["exposure"]
        # every run at seed 0 unless it says otherwise (the truth check's own
        # seed would persist on the seat into the next run)
        # ...and the denoiser at the GAME'S defaults: the truth runs turn it
        # off and dials persist on the seat into the next scenario
        fixed = ["--dial", "claude_auto_exposure=0", "--dial", "claude_rng_seed=0", "--dial", "claude_truth=0"] + \
            PLAY_DENOISE + \
            (["--dial", "claude_exposure=%r" % expo] if expo else []) + extra
        common = ["python3", "util/claude_motion.py", "--play", "--skip-seat", "--path", pf, "--scale", "2"] + fixed
        # ARMS (2026-10-09): every arm plays the SAME path and is scored against
        # the SAME truth, rendered once per scenario. Before, each A/B arm was
        # its own playtest: its own fps reading, so a slightly different path,
        # and its own truth -- two truth renders (the bulk of the GPU time) for
        # a comparison that was not quite one-variable. An arm's dials are on
        # its real-time run only; the truth runs get every arm key at 0 (the
        # truth must not see a display cache, e.g. claude_ledger).
        rt_off = sum([["--dial", k + "=0"] for k in arm_keys], [])
        # the truth runs (and their checks): no denoiser (TRUTH_DEF 3)
        TRUTH_ON = sum([["--dial", kv] for kv in TS.TRUTH_DIALS], [])

        def truth_check(ref_dir, tag, seed=TS.CHECK_SEED, dials=()):
            """a few poses of the stored truth, re-rendered at another seed
            (claude_truth_store.py); returns (frames checked, stats or None)"""
            idxs, poses = TS.check_plan(ref_dir, pk)
            cp = os.path.join(d, "check-path-%s.txt" % tag)
            TS.write_check_path(poses, cp)
            out = run([c if c != pf else cp for c in common] + rt_off + TRUTH_ON +
                      ["--name", name + "-check-" + tag, "--dial", "claude_path_hold=%d" % args.hold,
                       "--dial", "claude_rng_seed=%d" % seed] + sum([["--dial", kv] for kv in dials], []))
            stats = TS.compare(out[-1], ref_dir, idxs) if out and os.path.isdir(out[-1]) else None
            keep = None
            if out and os.path.isdir(out[-1]) and out[-1].startswith(os.path.join(REPO, "screenshots", "dump")):
                # kept beside the store (16 frames, ~32 MB): two checks that
                # disagree are an instrument to look at, not a number
                keep = os.path.join(TS.ROOT, "checks", name, "%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), tag))
                os.makedirs(os.path.dirname(keep), exist_ok=True)
                shutil.move(out[-1], keep)
            return idxs, stats, keep

        def control(ref_dir, idxs):
            """guard 3: the scoreboard's control (the dumbest honest renderer)
            held CONTROL_HOLD frames at the middle checked pose, against the
            truth there. Its dials stay on the seat, so the seat restarts
            before the next scenario."""
            nonlocal_first[0] = True
            i = idxs[len(idxs) // 2]
            r = TS._rows(ref_dir)[i]
            cp = os.path.join(d, "control-path.txt")
            TS.write_check_path([TS.path_pose(pk, r["path_frame"])], cp)
            out = run([c if c != pf else cp for c in common] + rt_off + TRUTH_ON +
                      sum([["--dial", kv] for kv in TS.CONTROL_DIALS], []) +
                      ["--name", name + "-control", "--dial", "claude_path_hold=%d" % TS.CONTROL_HOLD])
            if not out or not os.path.isdir(out[-1]):
                return False, {"why": "control render failed"}
            ok, info = TS.control_verdict(out[-1], ref_dir, i)
            keep = os.path.join(TS.ROOT, "checks", name, "%s-control" % time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(os.path.dirname(keep), exist_ok=True)
            shutil.move(out[-1], keep)
            info["dump"] = keep
            return ok, info

        if args.truth_check_only:
            # the check's own test: an unchanged engine at another seed must
            # pass, a change to the light rules (--truth-check-dial) must fail
            if not man:
                print("%-12s no stored truth for key %s (measured %.1f fps)" % (name, key, fps), flush=True)
                continue
            idxs, stats, _ = truth_check(man["reference"], "selftest", args.truth_check_seed,
                                         args.truth_check_dial or [])
            ok, why = TS.verdict(stats, man["floor"]) if stats else (False, ["check render failed"])
            TS.log({"event": "selftest", "key": key, "scenario": name, "pass": ok, "why": why,
                    "seed": args.truth_check_seed, "dials": args.truth_check_dial or [],
                    "stats": stats, "floor": man["floor"]})
            print("%-12s self-test seed %d dials %r: %s  %s" % (name, args.truth_check_seed,
                  args.truth_check_dial or [], "PASS" if ok else "FAIL", "; ".join(why[:3])), flush=True)
            for s, f in zip(stats or [], man["floor"]):
                print("             frame %d: px %.5f (floor %.5f)  blk %.5f (floor %.5f)  brightness x%.4f"
                      % (s["frame"], s["px"], f["px"], s["blk"], f["blk"], s["mean_ratio"]), flush=True)
            continue

        rt = {}
        for ai, (arm, kvs) in enumerate(arms):
            arm_dials = sum([["--dial", kv] for kv in kvs], [])
            # every arm starts the same way: parked at the start pose with its
            # own dials, the light memory and history forgotten (a ledger kept
            # from the last arm would already know the whole route), then 6 s
            # to settle with its dials on
            run(["python3", "util/claude_motion.py", "--play", "--skip-seat", "--path", pin,
                 "--nodump", "--frames", "120", "--name", name + "-park-" + arm] + extra + arm_dials)
            lab.goto({"pos": [x, y, z], "yaw": yaw, "pitch": pitch})
            open(lab.PATCH, "w").write("claude_path = %s\nclaude_reset_accum = %s-%d\n"
                                       % (pin, arm, time.time_ns()))
            time.sleep(6)
            rt[arm] = run(common + arm_dials + ["--name", name + "-rt-" + arm])
        # THE TRUTH (2026-10-09): stored once, then checked instead of re-rendered;
        # three guards (claude_truth_store, TRUTH_DEF 4): every dial classified,
        # truth mode recorded in every frame, the dumb control agrees
        ref, fid = [], []
        truth = {"key": key}
        if man:
            fb = [] if man.get("admit") else TS.feature_problems(man["reference"])
            idxs, stats, cdir = truth_check(man["reference"], "verify")
            ok, why = TS.verdict(stats, man["floor"]) if stats else (False, ["check render failed"])
            fb += TS.feature_problems(cdir) if cdir else ["check render kept nothing"]
            if fb:
                ok, why = False, why + fb
            if ok and man.get("admit"):
                cok, cinfo = control(man["reference"], idxs)
                truth["control"] = cinfo
                if cok:
                    TS.admit(man, cinfo)
                else:
                    ok, why = False, why + ["control disagrees: ratio %s tile %s" % (cinfo.get("ratio"), cinfo.get("tile_mad"))]
            truth.update(stats=stats, floor=man["floor"], why=why)
            TS.log({"event": ("admit" if man.get("admit") else "pass") if ok else "fail", "key": key,
                    "scenario": name, "why": why, "stats": stats, "floor": man["floor"]})
            if ok:
                truth["outcome"] = ("admitted from TRUTH_DEF 3 (truth-mode check and control passed)"
                                    if man.get("admit") else "reused (stored %s)" % man["stored"])
            else:
                truth["outcome"] = "replaced: " + "; ".join(why[:3])
                man = None
        if not man:
            dial_bad = TS.dial_problems(REPO)
            if dial_bad:
                # guard 1: no truth while any dial is unclassified, or a
                # display dial escapes truth mode
                truth["outcome"] = "NO TRUTH: " + "; ".join(dial_bad[:3])
                man = {"reference": "", "faces": ""}
        if not man:
            ref = run(common + rt_off + TRUTH_ON + ["--name", name + "-ref", "--dial", "claude_path_hold=%d" % args.hold])
            # face IDs at FULL resolution: the half-res dump blends with a linear
            # filter, which would invent codes at every edge; the scorer takes
            # every second pixel exactly instead
            fid = run([c if c != "2" else "1" for c in common] + rt_off + ["--name", name + "-faces",
                      "--dial", "claude_path_hold=1", "--dial", "claude_view=22"])
            fb = TS.feature_problems(ref[-1]) if ref and os.path.isdir(ref[-1]) else ["no truth render"]
            if any("REFUSED" in l for l in ref + fid) or not (ref and fid) or fb:
                # guard 2: a video that is not truth mode in every frame is
                # not a truth, and nothing is scored against it
                truth["outcome"] = "NO TRUTH: " + "; ".join((fb or ["render refused"])[:3])
                man = {"reference": "", "faces": ""}
            else:
                # the floor: the same check, right away, against what was just
                # made, at FLOOR_SEEDS seeds; per frame the worst of them. One
                # seed under-reads the noise: a fresh seed failed at 2x a
                # one-seed floor indoors (2026-10-09 self-test)
                floors = []
                for sd in TS.FLOOR_SEEDS:
                    idxs, fl, _ = truth_check(ref[-1], "floor%d" % sd, seed=sd)
                    floors.append(fl)
                floor = TS.worst(floors)
                cok, cinfo = control(ref[-1], idxs) if floor is not None else (False, {"why": "floor failed"})
                truth["control"] = cinfo
                if floor is None or not cok:
                    truth["outcome"] = "NO TRUTH: " + ("floor check failed" if floor is None else
                                                      "control disagrees: ratio %s tile %s" % (cinfo.get("ratio"), cinfo.get("tile_mad")))
                    man = {"reference": "", "faces": ""}
                else:
                    man = TS.store(key, spec, expo, ref[-1], fid[-1], idxs, floor,
                                   {"fps_measured": fps, "control": cinfo,
                                    "engine": run(["git", "rev-parse", "--short", "HEAD"])[-1]})
                    TS.log({"event": "store", "key": key, "scenario": name, "floor": floor, "control": cinfo})
                    truth.setdefault("outcome", "stored")
                    truth["floor"] = floor
        print("%-12s truth %s: %s" % (name, key, truth.get("outcome")), flush=True)
        meta["scenarios"][name] = {"fps_at_start": fps, "fps_path": fps_path, "exposure": expo,
                                   "exposure_measured": expo_measured, "frames": pk[-1][0] + 1,
                                   "arms": {a: v[-1] for a, v in rt.items()},
                                   "reference": man["reference"], "faces": man["faces"], "truth": truth,
                                   "notes": [l for l in sum(rt.values(), []) + ref + fid if "REFUSED" in l]}
        print(name, json.dumps(meta["scenarios"][name]), flush=True)
    open(lab.PATCH, "w").write("claude_path = 0\nclaude_view = 0\n")
    lab.rpc("abm", on=True)
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=1)
    print(run_dir)


def score(run_dir):
    import numpy as np
    from PIL import Image
    import claude_judge as J
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    report = {}
    for name, sc in meta["scenarios"].items():
        if not sc.get("reference"):
            print("%-12s NOT SCORED: no truth (%s)" % (name, sc.get("truth", {}).get("outcome")))
            continue
        R, _ = J.load_dump(sc["reference"])
        # AN INSTRUMENT MUST SEE SOMETHING (2026-10-07): a path that walks the
        # camera into a solid block records black in BOTH arms, and two black
        # frames "agree" perfectly. Refuse instead of scoring them.
        black = [i for i, f in enumerate(R) if float(np.asarray(f).mean()) < 2.0]
        if black:
            print("%-12s REFUSED: %d reference frames are black (first at %d): the camera "
                  "path enters something solid" % (name, len(black), black[0]))
            continue
        Fc, _ = J.load_dump(sc["faces"])
        # every arm against the one truth (old runs: a single "realtime")
        arms = sc.get("arms") or {"rt": sc["realtime"]}
        for arm, rt_dir in arms.items():
            key = name if len(arms) == 1 else name + "/" + arm
            T, rows = J.load_dump(rt_dir)
            n = min(len(T), len(R), len(Fc))
            pace = J.pacing(rows)
            jod = J.jod_video(T[:n], R[:n], max(pace.get("fps_mean", 30.0), 1.0), clip=min(60, n))
            # the reveal curve: pixels whose face code no pixel had the frame before
            codes = [(f[::2, ::2].astype(np.int32) * np.array([1, 256, 65536])).sum(2) for f in Fc[:n]]
            sky = 255 + 255 * 256 + 255 * 65536
            lumT = [J.lin(f) @ np.array([0.2126, 0.7152, 0.0722]) for f in T[:n]]
            lumR = [J.lin(f) @ np.array([0.2126, 0.7152, 0.0722]) for f in R[:n]]
            born = np.full(codes[0].shape, -1)
            ages = {}
            for i in range(1, n):
                new = ~np.isin(codes[i], codes[i - 1]) & (codes[i] != sky)
                same = codes[i] == codes[i - 1]
                born = np.where(new, i, np.where(same, born, -1))
                age = np.where(born >= 0, i - born, -1)
                for a in range(0, 31):
                    m = age == a
                    if m.sum() < 200:
                        continue
                    t_, r_ = lumT[i][m].mean(), lumR[i][m].mean()
                    ages.setdefault(a, []).append((t_, r_, int(m.sum())))
            curve = {a: float(sum(t for t, r, c in v) / max(sum(r for t, r, c in v), 1e-9))
                     for a, v in sorted(ages.items())}
            # DARK PATCHES (what John saw: "starts off all dark"): locally
            # smoothed brightness, real time vs the truth at the same pose, and
            # the share of the frame below 80 % / 50 % of the truth. The reveal
            # curve above only catches faces that are NEW to the screen; backing
            # up showed a dark band over ground that WAS on screen the frame
            # before (its history lost while moving), which this sees.
            def smooth(x, k=8):
                h, w = x.shape
                hh, ww = h // k, w // k
                return x[:hh * k, :ww * k].reshape(hh, k, ww, k).mean((1, 3))
            dark80, dark50 = [], []
            for i in range(n):
                st, sr = smooth(lumT[i]), smooth(lumR[i])
                lit = sr > 1e-3
                ratio = np.where(lit, st / np.maximum(sr, 1e-6), 1.0)
                dark80.append(float(((ratio < 0.8) & lit).mean()))
                dark50.append(float(((ratio < 0.5) & lit).mean()))
            moving = [i for i in range(1, n) if rows[i]["pos"] != rows[i - 1]["pos"]
                      or rows[i]["yaw"] != rows[i - 1]["yaw"]]
            # settling: per-frame JOD after the last move
            last_move = max(i for i, r in enumerate(rows[:n]) if i > 0 and
                            (r["pos"] != rows[i - 1]["pos"] or r["yaw"] != rows[i - 1]["yaw"])) \
                if any(r["pos"] != rows[0]["pos"] or r["yaw"] != rows[0]["yaw"] for r in rows[:n]) else 0
            settle = None
            for i in range(last_move, n, max(1, (n - last_move) // 20)):
                q = J.jod_still(J.fit(T[i], J.display()["resolution"][::-1]), J.fit(R[i], J.display()["resolution"][::-1]))
                if q >= 9.0:
                    settle = i - last_move
                    break
            dm = lambda v, idx: float(np.mean([v[i] for i in idx])) if idx else 0.0
            after = list(range(last_move + 1, n))
            report[key] = dict(pace, jod_video=jod, reveal_brightness_by_age=curve,
                                dark80_moving=dm(dark80, moving), dark50_moving=dm(dark50, moving),
                                dark80_after_stop=dm(dark80, after[:10]), dark80_curve=dark80,
                                frames_to_jod9_after_stop=settle, frames=n,
                                fps_at_start=sc["fps_at_start"])
            print("%-20s fps %.1f (p99 frame %.1f ms, spikes %d) | JOD video %.2f | settle to JOD 9: %s frames"
                  % (key, pace.get("fps_mean", 0), pace.get("frame_ms_p99", 0), pace.get("spikes_over_2x", 0),
                     jod, settle))
            print("             dark patches (share of frame below 80%% / 50%% of truth): moving %.3f / %.3f, first 10 frames after stop %.3f"
                  % (report[key]["dark80_moving"], report[key]["dark50_moving"], report[key]["dark80_after_stop"]))
            print("             new-face reveal curve (catches only faces new to the screen): " +
                  " ".join("%d:%.2f" % (a, v) for a, v in list(curve.items())[:12]))
            # the GIF: real time | reference
            frames = []
            for i in range(0, n, max(1, n // 60)):
                a = Image.fromarray(T[i]).resize((480, 270))
                b = Image.fromarray(R[i]).resize((480, 270))
                c = Image.new("RGB", (960, 270)); c.paste(a, (0, 0)); c.paste(b, (480, 0))
                frames.append(c)
            frames[0].save(os.path.join(run_dir, key.replace("/", "-") + ".gif"), save_all=True, append_images=frames[1:],
                           duration=int(1000 / max(pace.get("fps_mean", 30), 1) * max(1, n // 60)), loop=0)
    json.dump(report, open(os.path.join(run_dir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    # the GPU lock for this tool's whole life (children inherit it): see util/claude_gpu_lock.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import claude_gpu_lock
    # scoring takes the lock too: the video score (JOD) runs on the GPU
    # through PyTorch, and ran out of memory beside a training job (2026-10-09)
    claude_gpu_lock.hold('util/claude_playtest.py')
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    # TUNED: frames per truth pose, denoiser off | learn by: playtest scores
    # against a 4096 truth vs a 16384 one on one scenario (stop when they agree)
    ap.add_argument("--hold", type=int, default=4096)
    ap.add_argument("--score")
    ap.add_argument("--dial", action="append", help="k=v on every capture (an A/B arm)")
    ap.add_argument("--rt-dial", action="append",
                    help="k=v on the real-time run only; the truth runs get k=0")
    ap.add_argument("--arm", action="append",
                    help="NAME:k=v,k=v -- one real-time run per arm, all on the same path "
                         "and scored against one truth (NAME: alone = the game as it is)")
    ap.add_argument("--fresh-truth", action="store_true",
                    help="render the truth even if a stored one exists (and store it)")
    ap.add_argument("--truth-check-only", action="store_true",
                    help="only check each scenario's stored truth (no arms): the check's self-test")
    ap.add_argument("--truth-check-seed", type=int, default=None,
                    help="seed for --truth-check-only (default the check's own)")
    ap.add_argument("--truth-check-dial", action="append",
                    help="k=v on the --truth-check-only render: a deliberate rule change it must catch")
    a = ap.parse_args()
    if a.truth_check_seed is None:
        a.truth_check_seed = TS.CHECK_SEED
    if a.truth_check_dial and not a.truth_check_only:
        sys.exit("REFUSED: --truth-check-dial is for --truth-check-only")
    if a.arm and a.rt_dial:
        sys.exit("REFUSED: --arm and --rt-dial together (put the dials in the arms)")
    if "--score" in sys.argv:
        # an empty --score (a failed capture upstream) once fell through to a
        # full capture of every scenario, without the GPU lock (scoring skips
        # it), and held the GPU for two hours (2026-10-08)
        if not a.score or not os.path.isdir(a.score):
            sys.exit("REFUSED: --score needs an existing run folder, got %r" % a.score)
        score(a.score)
    else:
        capture(a)
