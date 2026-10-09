#!/usr/bin/env python3
"""claude_mldenoise_capture — data for the learned-denoiser prototype
(docs-draft/mldenoise.md).

Per view, two accumulations of the HONEST contender (play settings,
denoiser off), pose pinned, each dumping the denoiser's whole input set
(claude_dump_at: accumulated radiance, its direct part, the guide =
accumulated albedo + face code, the moments, and what is shown) at several
frame depths from ONE accumulation:

  A  seed sA, auto exposure on (the scoreboard's exposure probe, 256
     frames): dumps at 1, 4, 16, 64; the settled exposure is read and
     pinned for B
  B  seed sB, exposure pinned: dumps at 1, 4, 16, 64 and REF (the
     converged answer, default 2048)

  train:  python3 util/claude_mldenoise_capture.py train [--only NAME ...]
  test:   python3 util/claude_mldenoise_capture.py test [scene ...]
          (the four scoreboard scenes, a fresh game each as the scoreboard
          does, the ANYTHING contender at the truth's exposure: its "den"
          dump is today's filter on exactly the samples the network gets)
  truth:  python3 util/claude_mldenoise_capture.py truth torchroom
          (a linear truth into this worktree when the main one is stale)
"""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_gpu_lock  # noqa: E402

DATA = os.path.expanduser("~/data/mldenoise")
MAIN_TRUTH = os.path.expanduser("~/code/luanti/screenshots/scoreboard/truth")

# name: (x, y, z, yaw, pitch, time). Every pose >= ~25 m from the four
# scoreboard poses (forest 146.5,123.5; plains 5,-25; cabin 0,-12;
# torchroom 241.7,231.7) and not looking back at them. Yaw: view
# direction (-sin yaw, 0, cos yaw).
VIEWS = {
    # forest, away from the scoreboard's forest pose
    "forest-n": (150, 8.5, 160, 0, -5, 0.5),
    "forest-s": (150, 8.5, 88, 180, -5, 0.36),
    "forest-e": (185, 8.5, 123, 270, -3, 0.62),
    "forest-ne-up": (175, 8.5, 152, 45, 20, 0.45),
    "forest-se": (180, 8.5, 95, 225, -8, 0.7),
    "forest-far": (210, 8.5, 165, 0, -5, 0.55),
    "forest-sw": (125, 8.5, 92, 135, -10, 0.3),
    "treeline-back": (100, 8.5, 123.5, 90, -10, 0.3),
    # open ground and high views
    "plains-w": (-40, 8.5, -30, 90, -6, 0.3),
    "plains-s": (10, 8.5, -60, 180, -10, 0.68),
    "plains-e": (60, 8.5, -20, 270, -8, 0.45),
    "overlook": (60, 44.5, 60, 45, -35, 0.5),
    "overlook-west": (60, 44.5, 60, 225, -25, 0.4),
    "overlook-dusk": (60, 44.5, 60, 135, -30, 0.72),
    "skypad": (60, 8.5, 104, 0, -45, 0.5),
    "skypad-low": (60, 8.5, 104, 90, -10, 0.29),
    # rooms and cabins (interiors: never in a scoreboard frame)
    "cozy-golden": (8, 8.5, 3, 90, -8, 0.73),
    "cozy-midday": (7, 8.5, 6, 149, -8, 0.5),
    "cozy-morning": (7, 8.5, 2, 55, -5, 0.3),
    "cornell": (47, 8.5, 1, 0, 12, 0.5),
    "cave-skylight": (14, 8.5, 87, 0, -15, 0.5),
    "cave-skylight-pm": (14, 8.5, 87, 180, -20, 0.64),
    "glass-lit": (2, 8.5, 173, 270, 0, 0.5),
    "glass-dark": (10, 8.5, 173, 90, 0, 0.42),
    "glassfurnace": (103, 8.5, 171, 0, 0, 0.5),
    "sealed-outside": (5, 8.5, 245, 0, 0, 0.4),
    "sealed-outside-pm": (5, 8.5, 245, 180, -10, 0.66),
    # more light: the same places at other hours, other headings
    "forest-n-dusk": (150, 8.5, 160, 30, -5, 0.72),
    "forest-e-dawn": (185, 8.5, 123, 270, 15, 0.28),
    "forest-far-w": (210, 8.5, 165, 90, -5, 0.4),
    "plains-w-dusk": (-40, 8.5, -30, 0, -6, 0.7),
    "plains-sw": (-30, 8.5, -70, 135, -8, 0.55),
    "overlook-down": (60, 44.5, 60, 0, -50, 0.5),
    "glassfurnace-pm": (103, 8.5, 171, 180, -5, 0.65),
    "cozy-golden-2": (8, 8.5, 3, 270, -8, 0.6),
    "cave-skylight-am": (14, 8.5, 87, 90, -10, 0.3),
}
HOLDOUT = ["forest-far", "plains-e", "cozy-morning", "cave-skylight-pm"]

SCENES = {   # the scoreboard's (util/claude_scoreboard.py), read-only here
    "forest": ([146.5, 8.5, 123.5], 270, -5, 0.5),
    "plains": ([5, 8.5, -25], 0, -12, 0.5),
    "cabin": ([0, 8.5, -12], 0, -5, 0.5),
    "torchroom": ([241.7, 8.5, 231.7], 315, -12, 0.5),
}
HONEST = {"claude_nee": 1, "claude_bounce_uniform": 0, "claude_denoise": 0,
          "claude_torch_nee": 1, "claude_area_nee": 1, "claude_guide": 0,
          "claude_area_pick": 0, "claude_area_skip": 0, "claude_boost": 0}
ANYTHING = dict(HONEST, claude_denoise=1)
# the anything contender's frames in one second on the scoreboard's run of
# 2026-10-08T17:16Z (reports/scoreboard.md): the equal-input depth
ONE_SECOND = {"forest": 27, "plains": 47, "cabin": 70, "torchroom": 52}
DISPLAY = {"claude_auto_exposure": 0, "claude_white_balance": 0, "claude_night_vision": 0}


def stats():
    import claude_lab as lab
    return lab.read_stats() or {}


def arm(depths, prefix):
    """claude_dump_at, proven pending before the shot that consumes it"""
    import claude_lab as lab
    for _ in range(3):
        with open(lab.PATCH, "w") as f:
            f.write("claude_dump_at = %s:%s\n" % (",".join(str(d) for d in depths), prefix))
        t0 = time.time()
        while time.time() - t0 < 6:
            time.sleep(0.5)
            if (stats().get("dump_pending") or 0) == len(depths):
                return
    raise SystemExit("REFUSED: dump schedule never pending")


def shoot(pose, dials, frames, name, depths, prefix, tries=int(os.environ.get("MLD_TRIES", 3))):
    x, y, z, yaw, pitch, tod = pose
    cmd = ["python3", os.path.join(HERE, "claude_shoot.py"), "--skip-seat", "--play", "--pin",
           "--pos", str(x), str(y), str(z), "--yaw", str(yaw), "--pitch", str(pitch),
           "--time", str(tod), "--frames", str(frames), "--name", name]
    for k, v in dials.items():
        cmd += ["--dial", "%s=%s" % (k, v)]
    # long shots outlast the shooter's default 180 s wait; short ones must
    # not wait 25 minutes on a pose that keeps restarting
    env = dict(os.environ, CLAUDE_SETTLE_MAX_S="1500") if frames > 1024 else None
    last = ""
    for _ in range(tries):
        arm(depths, prefix)
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
        last = (r.stdout.strip().splitlines() or [""])[-1]
        ok = last.endswith(".png") and os.path.exists(last)
        missing = [d for d in depths if not os.path.exists("%s_%d.json" % (prefix, d))]
        # every dump must be N clean frames: no restart since the arming
        dirty = [d for d in depths if d not in missing and
                 json.load(open("%s_%d.json" % (prefix, d))).get("resets_since_arm", 1) != 0]
        if ok and not missing and not dirty:
            return last
        for d in depths:   # nothing from a failed attempt survives
            for f in glob.glob("%s_%d.*" % (prefix, d)):
                os.remove(f)
        missing = missing + ["dirty %s" % dirty] if dirty else missing
        print("    retry %s: %s missing %s" % (name, last[:150], missing), flush=True)
        time.sleep(15)
    raise Refused("REFUSED %s: %s" % (name, last))


class Refused(Exception):
    pass


def capstats(png):
    return json.load(open(png.replace(".png", ".capture.json")))["stats"]


def ident(st):
    return {k: st.get(k) for k in ("grid_hash", "area_emitters", "far_db_blocks", "sun_lux",
                                    "sky_lux", "grid_solid", "frame_ms_avg", "busy_ms", "pass_ms")}


def seat():
    import claude_scoreboard as sb
    sb.seat()


def cmd_train(only, ref):
    seat()
    n = 0
    for name, pose in VIEWS.items():
        if only and name not in only:
            continue
        d = os.path.join(DATA, "views", name)
        if os.path.exists(os.path.join(d, "meta.json")):
            print("have", name, flush=True)
            continue
        os.makedirs(d, exist_ok=True)
        if n and n % 10 == 0:
            seat()   # a fresh game now and then: nothing piles up
        n += 1
        t0 = time.time()
        sA, sB = 11 + 2 * n, 12 + 2 * n
        try:
            pa, pb = shoot_view(name, pose, d, sA, sB, ref)
        except Refused as e:
            print("SKIP %s: %s" % (name, e), flush=True)
            continue
        if pa is None:
            continue
        sta, stb = capstats(pa), capstats(pb)
        os.replace(pb, os.path.join(d, "B.png"))
        os.replace(pa, os.path.join(d, "A.png"))
        meta = {"name": name, "pose": pose, "exposure": json.load(open(os.path.join(d, "exposure.json")))["exposure"],
                "seeds": [sA, sB], "ref": ref,
                "A": ident(sta), "B": ident(stb), "seconds": round(time.time() - t0),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print("%-18s exposure %.4g  %3d s  frame %.1f ms" % (name, meta["exposure"], meta["seconds"],
              stb.get("busy_ms") or stb.get("frame_ms_avg") or -1), flush=True)


def shoot_view(name, pose, d, sA, sB, ref):
    if True:
        pa = shoot(pose, dict(HONEST, claude_auto_exposure=1, claude_exposure=1, claude_white_balance=0,
                              claude_night_vision=0, claude_rng_seed=sA), 256, name + "-A",
                   [1, 4, 16, 64], os.path.join(d, "A"))
        sta = capstats(pa)
        ex = sta.get("auto_exposure")
        if isinstance(ex, list):
            ex = ex[0] if ex and (len(ex) < 4 or ex[3] > 0.5) else None
        if not ex or not (0 < ex < 1e9):
            print("SKIP %s: no settled exposure (%r)" % (name, ex), flush=True)
            return None, None
        json.dump({"exposure": ex}, open(os.path.join(d, "exposure.json"), "w"))
        pb = shoot(pose, dict(HONEST, **DISPLAY, claude_exposure=ex, claude_rng_seed=sB), ref, name + "-B",
                   [1, 4, 16, 64, ref], os.path.join(d, "B"))
        return pa, pb


def truth_meta(sc):
    return json.load(open(os.path.join(MAIN_TRUTH, sc, "meta.json")))


def cmd_test(scenes, depths, seeds, truth_frames):
    """per scene, ONE fresh game: the truth first (honest, seed 0, linear
    dump), then the anything contender's noisy shots, same session and pose,
    so truth and inputs see the same loaded scene (the main checkout's
    truths were rendered on partly loaded scenes: instruments lane,
    2026-10-08). Each shot's grid_hash and area_emitters are recorded and
    compared with this truth's."""
    base = depths
    for sc in scenes:
        depths = sorted(set(base) | {ONE_SECOND[sc]})
        tm = truth_meta(sc)
        pos, yaw, pitch, tod = SCENES[sc]
        pose = (pos[0], pos[1], pos[2], yaw, pitch, tod)
        d = os.path.join(DATA, "test", sc)
        os.makedirs(d, exist_ok=True)
        seat()   # a fresh game per scene, as the scoreboard does
        ex = tm["exposure"]   # the scoreboard's pinned exposure for this scene
        tpng = shoot(pose, dict(HONEST, **DISPLAY, claude_exposure=ex), truth_frames, "mldtruth-" + sc,
                     [truth_frames], os.path.join(d, "truth"))
        tst = capstats(tpng)
        os.replace(tpng, os.path.join(d, "truth.png"))
        tid = ident(tst)
        print("%-10s truth %d frames  grid_hash %s  area_emitters %s" % (sc, truth_frames, tid["grid_hash"],
              tid["area_emitters"]), flush=True)
        rec = {"scene": sc, "exposure": ex, "truth": {"frames": truth_frames, "id": tid,
               "prefix": "truth_%d" % truth_frames}, "main_truth_scene_id": tm["scene_id"], "shots": []}
        for s in seeds:
            dd = dict(ANYTHING, **DISPLAY, claude_exposure=ex, claude_rng_seed=s)
            png = shoot(pose, dd, max(depths), "%s-anything-s%d" % (sc, s), depths, os.path.join(d, "s%d" % s))
            st = capstats(png)
            os.replace(png, os.path.join(d, "s%d_%d.png" % (s, max(depths))))
            sid = ident(st)
            same = all(sid.get(k) == tid.get(k) for k in ("grid_hash", "area_emitters"))
            rec["shots"].append({"seed": s, "depths": depths, "id": sid, "same_scene": same})
            print("%-10s seed %d  %s  grid_hash %s area_emitters %s%s" % (sc, s, depths, sid["grid_hash"],
                  sid["area_emitters"], "" if same else "  SCENE DIFFERS FROM THIS TRUTH"), flush=True)
            json.dump(rec, open(os.path.join(d, "meta.json"), "w"), indent=1)


def cmd_truth(scenes, frames):
    for sc in scenes:
        tm = truth_meta(sc)
        pos, yaw, pitch, tod = SCENES[sc]
        pose = (pos[0], pos[1], pos[2], yaw, pitch, tod)
        seat()
        d = os.path.join(DATA, "truth", sc)
        os.makedirs(d, exist_ok=True)
        dd = dict(HONEST, **DISPLAY, claude_exposure=tm["exposure"])   # seed 0, as the scoreboard's truth
        png = shoot(pose, dd, frames, "mldtruth-" + sc, [frames], os.path.join(d, "truth"))
        st = capstats(png)
        os.replace(png, os.path.join(d, "truth.png"))
        json.dump({"scene": sc, "frames": frames, "exposure": tm["exposure"], "id": ident(st),
                   "main_truth_scene_id": tm["scene_id"]}, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print("truth", sc, ident(st), flush=True)


if __name__ == "__main__":
    claude_gpu_lock.hold("util/claude_mldenoise_capture.py " + " ".join(sys.argv[1:]))
    what = sys.argv[1]
    rest = sys.argv[2:]
    if what == "train":
        only = [a for a in rest if not a.startswith("--")]
        cmd_train(only, int(os.environ.get("MLD_REF", 2048)))
    elif what == "test":
        cmd_test(rest or list(SCENES), [int(x) for x in os.environ.get("MLD_DEPTHS", "1,4,16,64").split(",")],
                 [int(x) for x in os.environ.get("MLD_SEEDS", "101,102,103").split(",")],
                 int(os.environ.get("MLD_TRUTH_FRAMES", 8192)))
    elif what == "batch":
        # several jobs under ONE hold of the GPU lock (children inherit it):
        # one command per line in the file named
        for line in open(rest[0]):
            line = line.strip()
            if line and not line.startswith("#"):
                print("batch:", line, flush=True)
                subprocess.run(line, shell=True, cwd=REPO, stdin=subprocess.DEVNULL)
    elif what == "truth":
        cmd_truth(rest, int(os.environ.get("MLD_TRUTH_FRAMES", 16384)))
    else:
        print(__doc__)
