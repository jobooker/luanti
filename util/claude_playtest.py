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

  python3 util/claude_playtest.py [--only backup ...] [--hold 256]
  ~/.venvs/judge/bin/python util/claude_playtest.py --score RUN_DIR
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT = os.path.join(REPO, "screenshots", "playtest")

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


def capture(args):
    import claude_lab as lab
    run_dir = os.path.join(OUT, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir)
    keys = scrub_conf_like_look()
    meta = {"deferred_to_game_defaults": keys, "scenarios": {}}
    first = True
    for name, (start, moves) in SCENARIOS.items():
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
        # 2. the path in seconds at that fps
        pk = path_frames(start, moves, fps)
        pf = os.path.join(d, "path.txt")
        with open(pf, "w") as f:
            for k in pk:
                f.write("%d %r %r %r %r %r\n" % k)
        fixed = ["--dial", "claude_auto_exposure=0"] + (["--dial", "claude_exposure=%r" % expo] if expo else []) + extra
        common = ["python3", "util/claude_motion.py", "--play", "--skip-seat", "--path", pf, "--scale", "2"] + fixed
        # --rt-dial: on the real-time run only; the truth runs get it at 0
        # (the truth must not see a display cache, e.g. claude_ledger)
        rt_only = sum([["--dial", kv] for kv in (args.rt_dial or [])], [])
        rt_off = sum([["--dial", kv.split("=")[0] + "=0"] for kv in (args.rt_dial or [])], [])
        rt = run(common + rt_only + ["--name", name + "-rt"])
        ref = run(common + rt_off + ["--name", name + "-ref", "--dial", "claude_path_hold=%d" % args.hold])
        # face IDs at FULL resolution: the half-res dump blends with a linear
        # filter, which would invent codes at every edge; the scorer takes
        # every second pixel exactly instead
        fid = run([c if c != "2" else "1" for c in common] + rt_off + ["--name", name + "-faces",
                  "--dial", "claude_path_hold=1", "--dial", "claude_view=22"])
        meta["scenarios"][name] = {"fps_at_start": fps, "exposure": expo, "frames": pk[-1][0] + 1,
                                   "realtime": rt[-1], "reference": ref[-1], "faces": fid[-1],
                                   "notes": [l for l in rt + ref + fid if "REFUSED" in l]}
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
        T, rows = J.load_dump(sc["realtime"])
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
        report[name] = dict(pace, jod_video=jod, reveal_brightness_by_age=curve,
                            dark80_moving=dm(dark80, moving), dark50_moving=dm(dark50, moving),
                            dark80_after_stop=dm(dark80, after[:10]), dark80_curve=dark80,
                            frames_to_jod9_after_stop=settle, frames=n,
                            fps_at_start=sc["fps_at_start"])
        print("%-12s fps %.1f (p99 frame %.1f ms, spikes %d) | JOD video %.2f | settle to JOD 9: %s frames"
              % (name, pace.get("fps_mean", 0), pace.get("frame_ms_p99", 0), pace.get("spikes_over_2x", 0),
                 jod, settle))
        print("             dark patches (share of frame below 80%% / 50%% of truth): moving %.3f / %.3f, first 10 frames after stop %.3f"
              % (report[name]["dark80_moving"], report[name]["dark50_moving"], report[name]["dark80_after_stop"]))
        print("             new-face reveal curve (catches only faces new to the screen): " +
              " ".join("%d:%.2f" % (a, v) for a, v in list(curve.items())[:12]))
        # the GIF: real time | reference
        frames = []
        for i in range(0, n, max(1, n // 60)):
            a = Image.fromarray(T[i]).resize((480, 270))
            b = Image.fromarray(R[i]).resize((480, 270))
            c = Image.new("RGB", (960, 270)); c.paste(a, (0, 0)); c.paste(b, (480, 0))
            frames.append(c)
        frames[0].save(os.path.join(run_dir, name + ".gif"), save_all=True, append_images=frames[1:],
                       duration=int(1000 / max(pace.get("fps_mean", 30), 1) * max(1, n // 60)), loop=0)
    json.dump(report, open(os.path.join(run_dir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    # the GPU lock for this tool's whole life (children inherit it): see util/claude_gpu_lock.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import claude_gpu_lock
    if "--score" not in sys.argv:   # scoring only reads files: no GPU lock
        claude_gpu_lock.hold('util/claude_playtest.py')
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--hold", type=int, default=256)
    ap.add_argument("--score")
    ap.add_argument("--dial", action="append", help="k=v on every capture (an A/B arm)")
    ap.add_argument("--rt-dial", action="append",
                    help="k=v on the real-time run only; the truth runs get k=0")
    a = ap.parse_args()
    if a.score:
        score(a.score)
    else:
        capture(a)
