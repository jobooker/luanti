#!/usr/bin/env python3
"""claude_settle_calibrate — how deep does a CI shot actually need to be?

Every CI shot has waited a fixed `settle = 60.0` WALL-CLOCK seconds since
the harness was written, and nobody had measured whether 60 s buys
anything the referee can see. This builds the evidence:

  curve   one continuous settle per arm, with the shutter fired at each
          of several accumulated-frame counts (250/500/1000/... ). The
          accumulator is a true 1/N running mean with no floor
          (game.cpp: accum_alpha = 1/(2+still_frames)), so shooting at
          N=250 on the way to N=8000 does NOT disturb the deeper frames:
          a screenshot is a present-path read, it resets nothing. One
          settle therefore yields the whole curve, and every point on it
          shares a history, which is exactly what CI does too.

  repeat  R INDEPENDENT settles to one N (full teleport-away-and-back
          reset between each), so "the knee is stable" is a measured
          claim and not one run's luck.

WHY FRAMES AND NOT SECONDS. `still_frames` is the real variable. The
same 60 s bought 3,928 frames on one CI run and 7,491 on another, because
fps moves with the scene, the build and the machine — and worse, ANY
world change clamps still_frames to 10 (game.cpp claudeVolumeSnapshot /
claudeVolumeIncremental), so a wall-clock settle interrupted in its last
second fires the shutter on an accumulator ~1 frame deep and calls it a
measurement. That is not hypothetical: it is measured.md's "Gate 4,
second half". A frames-based wait is immune by construction — a reset
merely extends the wait.

Numbers land in screenshots/settle/<runid>/curve.json; report.py-free,
`analyse` prints the tables straight from it.

  python3 util/claude_settle_calibrate.py curve  --skip-build
  python3 util/claude_settle_calibrate.py repeat --n 1500 --runs 3
  python3 util/claude_settle_calibrate.py analyse screenshots/settle/<id>
"""
import argparse
import json
import os
import shutil
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab            # noqa: E402
import claude_ci as ci              # noqa: E402
import claude_cornell_check as cornell  # noqa: E402

OUT_ROOT = os.path.join(REPO, "screenshots", "settle")
DEFAULT_TARGETS = [250, 500, 1000, 2000, 4000, 8000]
# Arms: the tightest-tolerance / highest-variance referee arm, and the
# general scene. cornell runs claude_nee = 0 with 24 bounces off a small
# panel, so it is the noise floor of the whole set by construction.
ARMS = ["cornell", "cozy-ci"]

POLL = 0.25            # s between still_frames reads (stats file is 1 Hz)
# still_frames is monotone between resets (game.cpp adds 1 per still
# frame), so ANY decrease is a world change / camera move / sun move --
# there is no noise floor to allow for. The clamp is to 10, which from a
# shallow point is a small drop, so a slack window would miss exactly the
# early resets that matter most.
ARM_ATTEMPTS = 3       # a world change mid-curve costs the whole arm
CURVE_TIMEOUT = 600.0  # s per arm attempt


# ---------------------------------------------------------------- instrument

def tile_means(png, nx=4, ny=4):
    """Mean linear luminance of an nx*ny grid over the HUD-cropped frame.

    The stand-in for "the region means the referee reads" on an arm that
    has no region referee (cozy-ci is `referee: None`). Same linear
    transform as the Cornell referee (ACES inverse + gamma 2.2 undone),
    same spirit: a mean over an area, so per-pixel noise averages down
    and what is left is the quantity a referee could actually consume.
    Cropped to the same band lab.rms_diff uses, so the tiles and the RMS
    are reading the same pixels.
    """
    import numpy as np
    from PIL import Image
    im = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64)
    h = im.shape[0]
    im = im[int(h * 0.12):int(h * 0.86)]
    lum = cornell.lin(im) @ cornell.LUMA
    H, W = lum.shape
    out = {}
    for r in range(ny):
        for c in range(nx):
            t = lum[H * r // ny:H * (r + 1) // ny, W * c // nx:W * (c + 1) // nx]
            out["t%d%d" % (r, c)] = float(t.mean())
    return out


# The Cornell REGIONS boxes are pixel-fraction boxes calibrated against
# ONE vantage's geometry (see claude_cornell_check). Running them over a
# cozy frame would produce five numbers that parse fine and mean nothing,
# which is the shape of a blind instrument. Only the cornell arms get
# them; every arm gets the tiles.
REGION_ARMS = ("cornell", "cornell-nee1")


def measure(png, with_regions):
    """Every number this calibration reads off one frame."""
    m = {"tiles": tile_means(png)}
    if with_regions:
        try:
            st = cornell.region_stats(png)
            m["regions"] = {k: v[0] for k, v in st.items()}
            m["purity"] = {k: v[2] for k, v in st.items()}
        except Exception as e:
            m["regions_error"] = str(e)
    return m


def golden_for(arm):
    rid = ci.read_golden()
    if not rid:
        return None
    p = os.path.join(ci.CI_DIR, rid, arm + ".png")
    return p if os.path.exists(p) else None


# ---------------------------------------------------------------- capture

def shutter(outdir, label):
    """Fire the shutter and file the PNG. Returns (dst, sf_after).

    still_frames is read by the CALLER before this is called (that is
    the quantity CI gates on, and the stats file is 1 Hz so it
    UNDER-reports); this returns the read taken once the PNG has landed,
    so every point on the curve carries a bracket rather than a single
    optimistic number. The gap between them is the shutter latency: the
    ~1 Hz settings-patch poll plus a ~3 MB synchronous PNG write.
    """
    before = lab.newest_shot()
    marker = "settle_%s_%d" % (label, time.time_ns())
    with open(lab.PATCH, "w") as f:
        f.write("claude_screenshot = %s\n" % marker)
    png, deadline = None, time.time() + ci.SHOT_TIMEOUT
    while time.time() < deadline:
        cur = lab.newest_shot()
        if cur and cur != before:
            png = ci.await_complete(cur)
            break
        time.sleep(ci.SHOT_POLL_INTERVAL)
    if not png:
        raise RuntimeError("no complete screenshot for %s" % label)
    sf_after = (lab.read_stats() or {}).get("still_frames")
    dst = os.path.join(outdir, label + ".png")
    shutil.copy2(png, dst)
    return dst, sf_after


def start_arm(arm, vantages, dials):
    """dials -> reset -> snapshot -> volume proven. Same order as
    claude_ci.capture(), for the same reasons (the dial channel is a
    ~1 Hz poll, and frames rendered with the previous arm's dials never
    decay out of a 1/N running mean)."""
    v, park = vantages[arm], ci.park_for(arm, vantages)
    ci.push_dials(dials, "%s_pre_%d" % (arm, time.time_ns()))
    reset_err = ci.reset_accumulation(v, park)
    aim0 = ci.read_aim()
    marker = "%s_%d" % (arm, time.time_ns())
    block = dict(dials, claude_volume_snapshot=marker)
    ci.push_dials(block, marker)
    vol = ci.await_volume(marker, block, room=ci.room_of(arm))
    return v, aim0, {"reset_error": reset_err, "volume": vol}


def walk_to(targets, outdir, prefix, log, with_regions):
    """Accumulate, firing the shutter as still_frames crosses each target.

    Returns (points, reset_at). reset_at is not None when the world
    changed under us — Mineclonia's grass ABM mutates a node inside the
    cozy bubble every 30-90 s, which correctly clamps still_frames to 10
    and makes everything after it a different, shallower curve. The
    whole arm attempt is discarded when that happens; it is not patched
    over.
    """
    points, last, deadline = [], -1, time.time() + CURVE_TIMEOUT
    todo = list(targets)
    while todo and time.time() < deadline:
        st = lab.read_stats() or {}
        sf = st.get("still_frames")
        if sf is None:
            time.sleep(POLL)
            continue
        if last >= 0 and sf < last:
            return points, {"at_still_frames": last, "dropped_to": sf,
                            "targets_done": [p["target"] for p in points]}
        last = sf
        if sf >= todo[0]:
            target = todo.pop(0)
            label = "%s_N%d" % (prefix, target)
            dst, sf_after = shutter(outdir, label)
            pt = {"target": target, "still_frames_read": sf,
                  "still_frames_after_shutter": sf_after,
                  "fps": st.get("fps"), "accum_alpha": st.get("accum_alpha"),
                  "volume_hash": st.get("volume_hash"),
                  "volume_snap_seq": st.get("volume_snap_seq"),
                  "png": os.path.basename(dst)}
            pt.update(measure(dst, with_regions))
            points.append(pt)
            log("    N=%-5d sf %s..%s  fps %.1f" % (
                target, sf, sf_after, st.get("fps") or 0.0))
            continue
        time.sleep(POLL)
    return points, None


# ---------------------------------------------------------------- commands

def seat_args(a):
    return types.SimpleNamespace(skip_build=a.skip_build, allow_debug=False,
                                 skip_deploy=a.skip_deploy)


def bring_up(a, out, rec):
    err = ci.bring_up_seat(out, rec, seat_args(a))
    if err:
        return err
    if (rec.get("build_type") or "").lower() not in ci.RELEASE_TYPES:
        return "not a Release build tree: %s" % rec.get("build_type")
    if rec["shader_failures"]:
        return "%d shader compile failures" % len(rec["shader_failures"])
    if not rec["freeze"].get("deepening"):
        return "the accumulator is not deepening: %s" % rec["freeze"]
    rec["doors_shut"] = ci.set_doors(True)
    # A referee room is a referee only while it is sealed and made of the
    # nodes its builder laid down; cornell is one of the two arms here.
    rec["room_integrity"] = ci.check_room_integrity()
    bad = [i for i in rec["room_integrity"] if i.get("off_spec")
           or i.get("error")]
    if bad:
        return "referee room off-spec: %s" % json.dumps(bad)[:500]
    return None


def run_arms(a, out, rec, targets, prefix_of, log):
    """Shared body of curve/repeat: bring up the seat, walk each arm."""
    vs = lab.load_vantages()
    dials = dict(ci.CANONICAL_DIALS)
    dials["claude_nee"] = 0
    rec["dials"] = dials
    rec["targets"] = targets
    rec["arms"] = {}
    try:
        for arm in a.arms:
            rec["arms"][arm] = []
            for rep in range(a.runs):
                for attempt in range(ARM_ATTEMPTS):
                    log("  %s rep %d attempt %d" % (arm, rep + 1, attempt + 1))
                    t0 = time.time()
                    v, aim0, meta = start_arm(arm, vs, dials)
                    pts, reset = walk_to(targets, out,
                                         prefix_of(arm, rep), log,
                                         arm in REGION_ARMS)
                    ok, detail = ci.aim_ok(aim0, ci.read_aim(), v)
                    h, herr = ci.room_hash(ci.room_of(arm))
                    entry = dict(meta, rep=rep, attempt=attempt + 1,
                                 points=pts, reset=reset, aim_ok=ok,
                                 aim_detail=detail, room_hash=h,
                                 room_hash_error=herr,
                                 wall_s=round(time.time() - t0, 1))
                    if reset:
                        log("    RESET mid-curve at sf=%s -> %s; discarding"
                            % (reset["at_still_frames"], reset["dropped_to"]))
                        rec["arms"][arm].append(entry)
                        continue
                    if not ok:
                        log("    AIM DRIFT: %s; discarding" % detail)
                        rec["arms"][arm].append(entry)
                        continue
                    entry["accepted"] = True
                    rec["arms"][arm].append(entry)
                    break
    finally:
        rec["doors_reopened"] = ci.set_doors(False)
    return rec


def write_out(out, rec):
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = os.path.join(out, "curve.json")
    json.dump(rec, open(path, "w"), indent=2)
    print(path)
    return path


def new_run(tag):
    git = ci.git_state()
    rid = "%s_%s_%s" % (time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
                        git["tag"], tag)
    out = os.path.join(OUT_ROOT, rid)
    os.makedirs(out, exist_ok=True)
    print("out: %s" % out)
    return out, dict(git, run_id=rid, kind=tag,
                     started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime()))


def cmd_curve(args):
    args.runs = 1
    out, rec = new_run("curve")
    err = bring_up(args, out, rec)
    if err:
        rec["aborted"] = err
        write_out(out, rec)
        print("ABORTED: %s" % err)
        return 1
    targets = [int(t) for t in args.targets.split(",")]
    run_arms(args, out, rec, targets, lambda arm, rep: arm, print)
    write_out(out, rec)
    return 0


def cmd_repeat(args):
    out, rec = new_run("repeat")
    err = bring_up(args, out, rec)
    if err:
        rec["aborted"] = err
        write_out(out, rec)
        print("ABORTED: %s" % err)
        return 1
    run_arms(args, out, rec, [args.n],
             lambda arm, rep: "%s_r%d" % (arm, rep + 1), print)
    write_out(out, rec)
    return 0


# ---------------------------------------------------------------- analysis

def accepted(rec, arm):
    return [e for e in rec["arms"].get(arm, []) if e.get("accepted")]


def _table(rows, hdr):
    w = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
    line = lambda r: "  ".join(str(r[i]).ljust(w[i]) for i in range(len(hdr)))
    print(line(hdr))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print(line(r))


def print_curve(rec, tol=0.0025):
    for arm, entries in rec["arms"].items():
        acc = accepted(rec, arm)
        print("\n=== %s ===" % arm)
        if not acc:
            print("  no accepted run (%d attempts)" % len(entries))
            continue
        pts = acc[0]["points"]
        deep = pts[-1]
        gold = golden_for(arm)
        keys = sorted(deep.get("regions") or {})
        tkeys = sorted(deep["tiles"])
        hdr = ["N", "sf_read", "sf_shut", "fps"] + keys + \
              ["worst_dev%", "tile_worst%", "rms_gold"]
        rows = []
        for p in pts:
            devs = [abs(p["regions"][k] / deep["regions"][k] - 1.0)
                    for k in keys] if keys else [0.0]
            tdev = max(abs(p["tiles"][k] / deep["tiles"][k] - 1.0)
                       for k in tkeys)
            rms = ""
            png = os.path.join(os.path.dirname(rec["_path"]), p["png"])
            if gold and os.path.exists(png):
                try:
                    rms = lab.rms_diff(gold, png)["rms"]
                except Exception as e:
                    rms = "err"
            rows.append([p["target"], p["still_frames_read"],
                         p["still_frames_after_shutter"],
                         round(p.get("fps") or 0, 1)]
                        + ["%.5f" % p["regions"][k] for k in keys]
                        + ["%.3f" % (100 * max(devs)), "%.3f" % (100 * tdev),
                           rms])
        _table(rows, hdr)
        knee = next((p["target"] for p, r in zip(pts, rows)
                     if float(r[-3]) <= tol * 100), None)
        tknee = next((p["target"] for p, r in zip(pts, rows)
                      if float(r[-2]) <= tol * 100), None)
        print("  knee (regions within %.2f%% of N=%d): %s"
              % (tol * 100, deep["target"], knee))
        print("  knee (tiles   within %.2f%% of N=%d): %s"
              % (tol * 100, deep["target"], tknee))


def print_repeat(rec, tol=0.0025):
    for arm in rec["arms"]:
        acc = accepted(rec, arm)
        print("\n=== %s: %d accepted run(s) ===" % (arm, len(acc)))
        if len(acc) < 2:
            continue
        pts = [e["points"][0] for e in acc]
        keys = sorted(pts[0].get("regions") or {}) + \
            ["#" + k for k in sorted(pts[0]["tiles"])]

        def val(p, k):
            return p["tiles"][k[1:]] if k.startswith("#") else p["regions"][k]
        rows = []
        for k in keys:
            vals = [val(p, k) for p in pts]
            spread = (max(vals) - min(vals)) / (sum(vals) / len(vals))
            rows.append([k] + ["%.5f" % v for v in vals]
                        + ["%.3f" % (100 * spread),
                           "ok" if spread <= tol else "OVER"])
        _table(rows, ["region"] + ["run%d" % (i + 1) for i in range(len(pts))]
               + ["spread%", ""])
        worst = max(float(r[-2]) for r in rows)
        print("  worst spread across %d runs: %.3f%% (tol %.2f%%) -> %s"
              % (len(pts), worst, tol * 100,
                 "PASS" if worst <= tol * 100 else "FAIL"))
        print("  still_frames at each shutter: %s"
              % [p["still_frames_read"] for p in pts])


def cmd_analyse(args):
    path = args.path
    if os.path.isdir(path):
        path = os.path.join(path, "curve.json")
    rec = json.load(open(path))
    rec["_path"] = path
    print("%s  %s@%s  %s" % (rec["run_id"], rec["branch"], rec["sha"],
                             rec.get("kind")))
    if rec.get("aborted"):
        print("ABORTED: %s" % rec["aborted"])
        return 1
    (print_repeat if rec.get("kind") == "repeat" else print_curve)(
        rec, tol=args.tol)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--skip-build", action="store_true")
        p.add_argument("--skip-deploy", action="store_true")
        p.add_argument("--arms", default=",".join(ARMS),
                       type=lambda s: s.split(","))
        return p

    p = common(sub.add_parser("curve"))
    p.add_argument("--targets", default=",".join(str(t) for t in DEFAULT_TARGETS))
    p.set_defaults(func=ci.with_run_lock(cmd_curve))
    p = common(sub.add_parser("repeat"))
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--runs", type=int, default=3)
    p.set_defaults(func=ci.with_run_lock(cmd_repeat))
    p = sub.add_parser("analyse")
    p.add_argument("path")
    p.add_argument("--tol", type=float, default=0.0025)
    p.set_defaults(func=cmd_analyse)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
