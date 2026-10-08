#!/usr/bin/env python3
"""claude_scoreboard — three contenders, one frozen truth, equal time.

John, 2026-10-08: "some sort of consistent comparisons ... a real dumb
baseline ... the second class is like everything's still just the truth
... the next one is the best sort of machine learning ... each one should
be improving over time, but ideally within each generation the smarter
option keeps pulling further ahead ... the fully retraced truth ... render
once ... and with the debug make sure our truth is still the truth."

  CONTENDERS  control   the dumbest honest renderer: the exact block walk,
                        bounces uniformly random, no light sampling, no
                        denoiser. It never gets smarter: the floor.
              honest    the best renderer that still converges to the truth
                        (today: play settings with the denoiser off).
              anything  the best renderer, anything allowed (today: play
                        settings as shipped, denoiser on).
  TRUTH       per scene, the honest renderer run very long, once; kept with
              the commit it came from. Re-render only when physics changes.
  BUDGET      the same milliseconds for everyone: one 60 fps frame, and one
              second standing still. Each contender gets as many frames as
              fit (its own measured frame time on this GPU).
  SCORE       FLIP of the picture you would see against the truth's picture
              (lower is better), every picture at the truth's exposure.
  CHECK       the control run very long against the truth, in LINEAR
              radiance: two honest renderers must converge to one picture.

  python3 util/claude_scoreboard.py truth [scene ...]
  python3 util/claude_scoreboard.py run [scene ...]
  python3 util/claude_scoreboard.py check [scene ...]
  (run and check call the judge venv themselves for scoring)
"""
import glob
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "screenshots", "scoreboard")
JUDGE_PY = os.path.expanduser("~/.venvs/judge/bin/python")
PAGE = os.path.expanduser("~/code/luanti-docs/reports/scoreboard.md")

# pos, yaw, pitch, time of day
SCENES = {
    "forest": ([146.5, 8.5, 123.5], 270, -5, 0.5),
    "plains": ([5, 8.5, -25], 0, -12, 0.5),
    "cabin": ([0, 8.5, -12], 0, -5, 0.5),   # 9.5 fell under the pin (2026-10-08)
    "torchroom": ([241.7, 8.5, 231.7], 315, -12, 0.5),
}
# Every dial any contender sets is spelled by every contender: a dial a
# shot does not push keeps the previous shot's value (it leaked three
# times on 2026-10-08).
CONTENDERS = {
    "control": {"claude_nee": 0, "claude_bounce_uniform": 1, "claude_denoise": 0,
                "claude_torch_nee": 0, "claude_area_nee": 0, "claude_guide": 0,
                "claude_area_pick": 0, "claude_area_skip": 0, "claude_boost": 0},
    "honest": {"claude_nee": 1, "claude_bounce_uniform": 0, "claude_denoise": 0,
               "claude_torch_nee": 1, "claude_area_nee": 1, "claude_guide": 0,
               "claude_area_pick": 0, "claude_area_skip": 0, "claude_boost": 0},
    "anything": {"claude_nee": 1, "claude_bounce_uniform": 0, "claude_denoise": 1,
                 "claude_torch_nee": 1, "claude_area_nee": 1, "claude_guide": 0,
                 "claude_area_pick": 0, "claude_area_skip": 0, "claude_boost": 0},
}
# display-only effects that adapt over time: off for everyone, so no
# contender is helped or hurt by them
DISPLAY = {"claude_auto_exposure": 0, "claude_white_balance": 0, "claude_night_vision": 0}
BUDGETS_MS = {"one frame": 1000.0 / 60.0, "one second": 1000.0}
TRUTH_FRAMES = int(os.environ.get("SCOREBOARD_TRUTH_FRAMES", 16384))   # smoke tests set it low
CHECK_FRAMES = int(os.environ.get("SCOREBOARD_CHECK_FRAMES", 32768))


def git_sha():
    return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def shoot(scene, dials, frames, name, dump=None, tries=3):
    pos, yaw, pitch, tod = SCENES[scene]
    cmd = ["python3", os.path.join(HERE, "claude_shoot.py"), "--skip-seat", "--play", "--pin",
           "--pos"] + [str(v) for v in pos] + ["--yaw", str(yaw), "--pitch", str(pitch),
           "--time", str(tod), "--frames", str(frames), "--name", name]
    for k, v in dials.items():
        cmd += ["--dial", "%s=%s" % (k, v)]
    if dump:
        cmd += ["--accum-dump", dump]
    # long shots (truths, checks) outlast the shooter's default 180 s wait
    env = dict(os.environ, CLAUDE_SETTLE_MAX_S="1500") if frames > 2048 else None
    last = ""
    for _ in range(tries):
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
        last = (r.stdout.strip().splitlines() or [""])[-1]
        if last.endswith(".png") and os.path.exists(last):
            return last
        err = (r.stderr.strip().splitlines() or [""])[-1]
        print("    retry %s: %s %s" % (name, last[:120], err[:120]), flush=True)
        time.sleep(10)
    raise SystemExit("REFUSED %s %s: %s" % (scene, name, last))


def stats_of(png):
    return json.load(open(png.replace(".png", ".capture.json")))["stats"]


def seat():
    sys.path.insert(0, HERE)
    from claude_playtest import scrub_conf_like_look
    import claude_denoise_price as price
    scrub_conf_like_look()
    price.start_seat()
    # a fresh game restarts its picture while it settles (measured: three
    # back-to-back shots all refused "time did not freeze"); wait it out
    import claude_lab as lab
    prev, since, t0 = None, time.time(), time.time()
    while time.time() - t0 < 90:
        time.sleep(1)
        cur = (lab.read_stats() or {}).get("accum_resets")
        if cur != prev:
            prev, since = cur, time.time()
        elif time.time() - since > 8:
            break


def cmd_truth(scenes):
    for sc in scenes:
        seat()   # a fresh game per scene: what loads must not depend on history
        d = os.path.join(OUT, "truth", sc)
        os.makedirs(d, exist_ok=True)
        # the exposure: what the eye settles on for this scene, then fixed
        probe = shoot(sc, dict(CONTENDERS["honest"], claude_auto_exposure=1, claude_exposure=1,
                               claude_white_balance=0, claude_night_vision=0), 256, "probe-" + sc)
        ex = stats_of(probe).get("auto_exposure")
        if isinstance(ex, list):   # [factor, adapted, now, written]: the factor
            ex = ex[0] if ex and (len(ex) < 4 or ex[3] > 0.5) else None
        if not ex or not (0 < ex < 1e9):
            raise SystemExit("REFUSED %s: no settled exposure (%r)" % (sc, ex))
        dials = dict(CONTENDERS["honest"], **DISPLAY, claude_exposure=ex)
        t0 = time.time()
        png = shoot(sc, dials, TRUTH_FRAMES, "truth-" + sc, dump=os.path.join(d, "truth"))
        st = stats_of(png)
        os.replace(png, os.path.join(d, "truth.png"))
        meta = {"scene": sc, "pose": SCENES[sc], "frames": TRUTH_FRAMES, "exposure": ex,
                "dials": dials, "sha": git_sha(), "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seconds": round(time.time() - t0), "scene_id": {k: st.get(k) for k in
                ("grid_hash", "area_emitters", "far_db_blocks", "sun_lux", "sky_lux")}}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print("truth %-10s exposure %.4g  %d frames in %d s  %s" % (sc, ex, TRUTH_FRAMES, meta["seconds"],
              meta["scene_id"]), flush=True)


def truth_meta(sc):
    p = os.path.join(OUT, "truth", sc, "meta.json")
    if not os.path.exists(p):
        raise SystemExit("no truth for %s: run `truth %s` first" % (sc, sc))
    return json.load(open(p))


def cmd_run(scenes):
    run = os.path.join(OUT, "runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run, exist_ok=True)
    shots = {"sha": git_sha(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shots": []}
    for sc in scenes:
        tm = truth_meta(sc)
        seat()   # a fresh game per scene, as its truth had
        for cn, cd in CONTENDERS.items():
            dials = dict(cd, **DISPLAY, claude_exposure=tm["exposure"])
            # its own frame time on this GPU, still, at this scene
            st = stats_of(shoot(sc, dials, 64, "price-%s-%s" % (sc, cn)))
            ms = st.get("busy_ms") or st.get("frame_ms_avg")
            for bn, budget in BUDGETS_MS.items():
                n = max(1, int(math.floor(budget / ms)))
                png = shoot(sc, dials, n, "%s-%s-%d" % (sc, cn, n))
                s2 = stats_of(png)
                ident = {k: s2.get(k) for k in ("grid_hash", "area_emitters", "far_db_blocks", "sun_lux")}
                same = all(ident[k] == tm["scene_id"].get(k) for k in ("area_emitters", "sun_lux"))
                shots["shots"].append({"scene": sc, "contender": cn, "budget": bn, "frame_ms": ms,
                                       "frames": n, "png": png, "scene_id": ident, "same_scene": same})
                print("%-10s %-9s %-11s %6.2f ms/frame -> %5d frames%s" % (
                    sc, cn, bn, ms, n, "" if same else "  SCENE DIFFERS FROM TRUTH %s" % ident), flush=True)
        json.dump(shots, open(os.path.join(run, "shots.json"), "w"), indent=1)
    subprocess.run([JUDGE_PY, os.path.abspath(__file__), "score", run], check=True)


def cmd_try(scene, variants):
    """the honest renderer plus one change per variant, at equal time against
    the scene's truth; logged to experiments.jsonl, not to the history.
    Every variant spells every dial any variant sets (unset ones revert to
    the honest renderer's value), so nothing leaks from one to the next."""
    tm = truth_meta(scene)
    seat()
    keys = sorted({k for v in variants.values() for k in v})
    base = dict(CONTENDERS["honest"])
    defaults = {"claude_guide": 0, "claude_guide_impl": 2, "claude_area_pick": 0, "claude_area_skip": 0,
                "claude_torch_nee": 1, "claude_area_nee": 1, "claude_boost": 0, "claude_bounces": 24}
    run = os.path.join(OUT, "tries", time.strftime("%Y%m%d-%H%M%S") + "-" + scene)
    os.makedirs(run, exist_ok=True)
    shots = {"sha": git_sha(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shots": [],
             "try": True, "variants": variants}
    for vn, over in [("honest", {})] + list(variants.items()):
        d = dict(base, **{k: base.get(k, defaults.get(k, 0)) for k in keys})
        d.update(over)
        d.update(DISPLAY, claude_exposure=tm["exposure"])
        st = stats_of(shoot(scene, d, 64, "price-%s-%s" % (scene, vn.replace(" ", "_"))))
        ms = st.get("busy_ms") or st.get("frame_ms_avg")
        for bn, budget in BUDGETS_MS.items():
            n = max(1, int(math.floor(budget / ms)))
            png = shoot(scene, d, n, "%s-%s-%d" % (scene, vn.replace(" ", "_"), n))
            s2 = stats_of(png)
            same = s2.get("area_emitters") == tm["scene_id"].get("area_emitters")
            shots["shots"].append({"scene": scene, "contender": vn, "budget": bn, "frame_ms": ms,
                                   "frames": n, "png": png, "same_scene": same, "dials": over})
            print("%-10s %-26s %-11s %6.2f ms/frame -> %5d frames%s" % (scene, vn, bn, ms, n,
                  "" if same else "  SCENE DIFFERS"), flush=True)
    json.dump(shots, open(os.path.join(run, "shots.json"), "w"), indent=1)
    subprocess.run([JUDGE_PY, os.path.abspath(__file__), "score", run], check=True)


def cmd_score(run):
    sys.path.insert(0, HERE)
    import claude_judge as J
    shots = json.load(open(os.path.join(run, "shots.json")))
    rows = []
    for s in shots["shots"]:
        ref = J.load_png(os.path.join(OUT, "truth", s["scene"], "truth.png"))
        img = J.load_png(s["png"])
        m = J.still(img, ref)
        # plain pixel error too: FLIP and JOD both rated the control WORSE
        # after 35 frames than after 1 (first run, 2026-10-08) while its
        # brightness converged; RMSE of the displayed picture cannot do that
        import numpy as np
        a = img.astype(np.float64) / 255.0
        b = ref.astype(np.float64) / 255.0
        rmse = float(np.sqrt(((a - b) ** 2).mean()))
        s.update(flip=m["flip"], jod=m["jod"], brightness=m["brightness"], rmse=rmse)
        rows.append(s)
        print("%-10s %-26s %-11s RMSE %.4f  FLIP %.4f  JOD %5.2f  bright %.4f  (%d frames)%s" % (
            s["scene"], s["contender"], s["budget"], rmse, m["flip"], m["jod"], m["brightness"], s["frames"],
            "" if s.get("same_scene", True) else "  SCENE DIFFERS"), flush=True)
    rec = {"utc": shots["utc"], "sha": shots["sha"], "run": os.path.basename(run), "rows": rows}
    if shots.get("try"):
        rec["variants"] = shots["variants"]
        with open(os.path.join(OUT, "experiments.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        return
    # rescoring a run replaces its line rather than adding a second one
    hp = os.path.join(OUT, "history.jsonl")
    old = [l for l in open(hp)] if os.path.exists(hp) else []
    old = [l for l in old if json.loads(l).get("run") != rec["run"]]
    with open(hp, "w") as f:
        f.writelines(old)
        f.write(json.dumps(rec) + "\n")
    write_page()


def write_page():
    hist = [json.loads(l) for l in open(os.path.join(OUT, "history.jsonl"))]
    last = hist[-1]
    cn = list(CONTENDERS)
    L = ["---", "title: \"Scoreboard: control, honest, anything vs the truth\"",
         "summary: \"Three renderers at equal time against a frozen converged truth, per scene; generated by util/claude_scoreboard.py.\"",
         "tags: [luanti, report]", "author: claude", "---", "",
         "# Scoreboard", "",
         "*Generated by `util/claude_scoreboard.py` (engine repo); do not edit by hand.*", "",
         "Three contenders, the same milliseconds each, scored against a converged truth rendered once per",
         "scene. Each cell: **RMSE / FLIP (frames)**, both lower is better, 0 = identical to the truth. RMSE",
         "is plain pixel error of the picture you see; FLIP is perceptual but misjudges very noisy pictures",
         "(it rated the control worse after 35 frames than after 1). Each contender gets as many frames",
         "as fit in the budget at its own measured frame time.", "",
         "- **control**: the dumbest honest renderer (random bounces, no light sampling, no denoiser). The floor.",
         "- **honest**: the best renderer that still converges to the truth (play settings, denoiser off).",
         "- **anything**: the best renderer, anything allowed (play settings as shipped).", "",
         "## Latest run: %s at `%s`" % (last["utc"], last["sha"]), ""]
    for bn in BUDGETS_MS:
        L += ["### %s" % bn, "", "| scene | " + " | ".join(cn) + " | honest vs control | anything vs honest |",
              "|---|" + "---|" * (len(cn) + 2)]
        scenes = []
        for r in last["rows"]:
            if r["scene"] not in scenes:
                scenes.append(r["scene"])
        for sc in scenes:
            v = {r["contender"]: r for r in last["rows"] if r["scene"] == sc and r["budget"] == bn}
            cells = ["%.4f / %.3f (%d fr)" % (v[c].get("rmse", float("nan")), v[c]["flip"], v[c]["frames"])
                     if c in v else "-" for c in cn]
            def gap(a, b):
                ra, rb = v.get(a, {}).get("rmse"), v.get(b, {}).get("rmse")
                return "%.2fx" % (ra / rb) if ra and rb else "-"
            bad = any(not r.get("same_scene", True) for r in v.values())
            L.append("| %s%s | %s | %s | %s |" % (sc, " **(scene differs from its truth: not valid)**" if bad else "",
                     " | ".join(cells), gap("control", "honest"), gap("honest", "anything")))
        L.append("")
    L += ["(The gap columns read \"how many times more RMSE\": control/honest, honest/anything.",
          "Above 1.00x the smarter contender is ahead.)", "", "## Over time (mean RMSE across valid scenes)", "",
          "| run | commit | budget | " + " | ".join(cn) + " |", "|---|---|---|" + "---|" * len(cn)]
    for h in hist:
        for bn in BUDGETS_MS:
            # the mean over the WHOLE scene set, or the run is marked
            # incomplete: a mean over different scenes is not a trend
            valid = {r["scene"] for r in h["rows"] if r["budget"] == bn and r.get("same_scene", True)}
            invalid = {r["scene"] for r in h["rows"] if r["budget"] == bn} - valid
            complete = set(SCENES) <= valid and not invalid
            means = []
            for c in cn:
                xs = [r.get("rmse", r["flip"]) for r in h["rows"] if r["contender"] == c and r["budget"] == bn
                      and r["scene"] in valid]
                means.append(("%.4f" % (sum(xs) / len(xs)) if xs else "-")
                             + ("" if complete else "*"))
            L.append("| %s | `%s` | %s | %s |" % (h["utc"][:16], h["sha"], bn, " | ".join(means)))
    L += ["", "\\* incomplete: not every scene was valid in that run, so it is not comparable.", "",
          "Why the control can get WORSE with more time: it never aims at the sun, so it finds it by luck, and",
          "each lucky ray is a white speck until enough frames average it down. More frames, more specks, not",
          "yet enough to dim them. Run long enough it converges (the check below).",
          "", "## Is the truth still the truth?", ""]
    chk = os.path.join(OUT, "check.jsonl")
    if os.path.exists(chk):
        L += ["The control (a different honest renderer) run very long, against each truth, in linear radiance.",
              "", "| when | scene | control frames | mean brightness ratio | tile median abs diff | verdict |",
              "|---|---|---|---|---|---|"]
        for l in open(chk):
            c = json.loads(l)
            L.append("| %s | %s | %d | %.4f | %.4f | %s |" % (c["utc"][:16], c["scene"], c["frames"],
                     c["ratio"], c["tile_mad"], c["verdict"]))
    else:
        L.append("Not run yet (`claude_scoreboard.py check`).")
    os.makedirs(os.path.dirname(PAGE), exist_ok=True)
    open(PAGE, "w").write("\n".join(L) + "\n")
    print("page:", PAGE)


def cmd_check(scenes):
    import numpy as np
    for sc in scenes:
        tm = truth_meta(sc)
        seat()
        d = os.path.join(OUT, "truth", sc)
        dump = os.path.join(d, "control-long")
        dials = dict(CONTENDERS["control"], **DISPLAY, claude_exposure=tm["exposure"])
        shoot(sc, dials, CHECK_FRAMES, "check-" + sc, dump=dump)

        def load(p):
            m = json.load(open(p + ".json"))
            return np.fromfile(p + ".f32", np.float32).reshape(m["h"], m["w"], 4)[..., :3].astype(np.float64)
        a, t = load(dump), load(os.path.join(d, "truth"))
        lum = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])
        la, lt = lum(a), lum(t)
        ratio = la.mean() / lt.mean()
        k = 16
        h, w = lt.shape
        ta = la[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k).mean((1, 3))
        tt = lt[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k).mean((1, 3))
        mad = float(np.median(np.abs(ta - tt) / np.maximum(tt, 1e-6)))
        # TUNED: 1% on the mean, 5% median tile | learn by: the spread of two
        # truths rendered with different seeds (their own disagreement)
        verdict = "agree" if abs(ratio - 1) < 0.01 and mad < 0.05 else "DISAGREE"
        rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "scene": sc, "frames": CHECK_FRAMES,
               "ratio": float(ratio), "tile_mad": mad, "verdict": verdict, "truth_sha": tm["sha"], "sha": git_sha()}
        with open(os.path.join(OUT, "check.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("check %-10s control/truth brightness %.4f  tile median |diff| %.4f  -> %s" % (sc, ratio, mad, verdict),
              flush=True)
    if os.path.exists(os.path.join(OUT, "history.jsonl")):
        write_page()


if __name__ == "__main__":
    # the GPU lock for this tool's whole life (children inherit it): see util/claude_gpu_lock.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import claude_gpu_lock
    if sys.argv[1:2] != ["score"]:   # scoring only reads files: no GPU lock
        claude_gpu_lock.hold('util/claude_scoreboard.py')
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:] or list(SCENES)
    if what == "truth":
        cmd_truth(rest)
    elif what == "run":
        cmd_run(rest)
    elif what == "score":
        cmd_score(sys.argv[2])
    elif what == "check":
        cmd_check(rest)
    elif what == "try":
        cmd_try(sys.argv[2], json.loads(sys.argv[3]))
    else:
        print(__doc__)
