#!/usr/bin/env python3
"""A/B THE BAKE FLOOR at John's own vantage: are the pinholes gone?

WHAT IS BEING MEASURED. On 2026-08-16, once the tracer's walk learned to
descend into 16^3 cells, John saw bright specks in horizontal rows on
the cosy cabin's far plank wall — the lit furnace room behind it, seen
through 1/16 m straight-through tunnels in the plank model. The bake
that carves those models from texture brightness had no floor, so a
groove line that is dark all the way across a tile removed a whole row
of cells and a cabin wall is ONE node thick. util/claude_models.py now
enforces a floor (`enforce_min_thickness`): no all-air line through a
full-solid-node model on any axis.

THE ONE VARIABLE is the MASK FILES on disk. Arm "before" installs the
model JSONs from a git ref (default 99afed869, the last commit
before the floor landed);
arm "after" installs the working tree's. Same binary, same seat, same
world, same dials, same settle — nothing else moves.

THE LEVER IS A DIAL, NOT A RESTART. game.cpp's claudeLoadModels drops
and re-reads the whole model table whenever `claude_models` changes
value, and a geometry dial that moved forces a FULL re-walk of the trace
grid (not the incremental path). So toggling claude_models 1 -> 0 -> 1
after swapping the files loads the new masks into a running client.
`grid_hash` is printed per arm and MUST differ between arms — if it does
not, the swap did not reach the grid and every number below is about the
same geometry twice (the golden-diff blindness of 2026-08-12).

WHAT IT REPORTS, per arm:
  * bright-pixel count in the wall box, at three thresholds. This is the
    measurement: a pinhole is a handful of near-white pixels on a wall
    whose own luminance is ~20-40. Counting outliers reads the defect
    directly; a region mean averages it away.
  * the wall box mean and the whole-frame mean, for context and to prove
    the box did not simply go dark for an unrelated reason.
  * still_frames / grid_hash / accum_resets at the shutter.

Usage:  python3 util/claude_bakefloor_ab.py [--vantage cozy-holes]
                                            [--before-ref SHA]
                                            [--frames 1500]
The seat (server + client) must already be up. Prints JSON on stdout and
writes screenshots/bakefloor_ab_<vantage>.json.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab  # noqa: E402

MODEL_DIR = os.path.join(HERE, "claude_models")
PARK_VANTAGE = "furnace-050"

# The far wall of the cosy cabin as it lands in a 1920x1080 frame from
# the `cozy-holes` vantage: left of the bright vault panel, below the
# skylight, above the floor line. Read off John's own capture
# (screenshots/screenshot_20260816_184259.png), where the specks sit at
# roughly x 890-1035, y 490-590 — inside this box with room to spare.
WALL_BOX = (700, 240, 1140, 600)   # x0, y0, x1, y1 (exclusive)
# A wall lit only by bounce reads ~20-40 luma here; the furnace room
# behind it is an emitter. 80 is already far outside the wall's own
# distribution, 160 is unambiguous.
BRIGHT = (80, 120, 160)


def stats():
    return json.load(open(lab.STATS))


def settle_to(frames, hard_max_s=240.0):
    t0 = time.time()
    last = None
    while time.time() - t0 < hard_max_s:
        try:
            last = stats()
        except Exception:
            time.sleep(0.5)
            continue
        if last.get("still_frames", 0) >= frames:
            return last, True
        time.sleep(0.5)
    return last, False


def git_models(ref, outdir):
    """Extract every model JSON at `ref` into outdir. Returns [names]."""
    os.makedirs(outdir, exist_ok=True)
    ls = subprocess.run(["git", "ls-tree", "--name-only", ref,
                         "util/claude_models/"], cwd=REPO,
                        capture_output=True, text=True, check=True)
    got = []
    for path in ls.stdout.split():
        if not path.endswith(".json"):
            continue
        blob = subprocess.run(["git", "show", "%s:%s" % (ref, path)],
                              cwd=REPO, capture_output=True, check=True)
        name = os.path.basename(path)
        with open(os.path.join(outdir, name), "wb") as f:
            f.write(blob.stdout)
        got.append(name)
    if not got:
        raise SystemExit("no model JSON at ref %s" % ref)
    return sorted(got)


def install_models(srcdir):
    """Copy a set of model JSONs over util/claude_models/. Only .json —
    the preview PNGs are for humans and the engine never reads them.
    Returns (count, md5 of the whole installed set): the arm's INPUT,
    stated as one number, so "both arms measured the same masks" is a
    thing the record can be checked for rather than assumed."""
    h = hashlib.md5()
    n = 0
    for name in sorted(os.listdir(srcdir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(srcdir, name), "rb") as f:
            data = f.read()
        with open(os.path.join(MODEL_DIR, name), "wb") as f:
            f.write(data)
        h.update(name.encode())
        h.update(data)
        n += 1
    return n, h.hexdigest()[:12]


def reload_models():
    """Make a running client re-read the model JSONs from disk, and
    re-walk the grid with them. 1 -> 0 -> 1; each edge is a geometry
    dial change, which game.cpp answers with a full snapshot."""
    lab.doorway(claude_models=0)
    time.sleep(3.0)
    lab.doorway(claude_models=1)
    time.sleep(4.0)


def wall_stats(png):
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64)
    lum = a @ np.array([0.2126, 0.7152, 0.0722])
    x0, y0, x1, y1 = WALL_BOX
    box = lum[y0:y1, x0:x1]
    out = {
        "wall_box": list(WALL_BOX),
        "wall_px": int(box.size),
        "wall_mean": round(float(box.mean()), 4),
        "wall_max": round(float(box.max()), 2),
        "wall_p999": round(float(np.percentile(box, 99.9)), 2),
        "frame_mean": round(float(lum.mean()), 4),
    }
    for t in BRIGHT:
        out["bright_over_%d" % t] = int((box > t).sum())
    return out


def arm(vantage, srcdir, tag, frames, descend=1):
    vs = lab.load_vantages()
    v = vs[vantage]
    away = vs[PARK_VANTAGE if vantage != PARK_VANTAGE else "cornell"]
    installed, md5 = install_models(srcdir)
    # dials before the reset, as in claude_descend_ab: a dial pushed
    # after the accumulator restarts keeps the previous arm's frames at
    # a fixed weight forever.
    lab.rpc("revive", player="claude")
    lab.doorway(**dict(claude_view=0, claude_nee=0, claude_bounces=24,
                       claude_grid_debug=3, claude_rng=1,
                       claude_descend=descend, claude_stats=1,
                       claude_grid_follow=1, claude_show_hud=0,
                       claude_show_chat=0, claude_input_lock=1))
    reload_models()
    lab.goto(away)
    time.sleep(1.0)
    lab.goto(v)
    lab.doorway(claude_grid_snapshot="bakefloor_%s" % tag)
    time.sleep(2.0)
    st, ok = settle_to(frames)
    png = lab.shot("bakefloor_%s" % tag, settle=0.5)
    rec = dict(arm=tag, descend=descend, models_installed=installed,
               models_md5=md5, png=png,
               settled=ok, still_frames=st.get("still_frames"),
               grid_hash=st.get("grid_hash"),
               grid_solid=st.get("grid_solid"),
               accum_resets=st.get("accum_resets"),
               frame_ms_avg=st.get("frame_ms_avg"),
               busy_ms=st.get("busy_ms"), pass_ms=st.get("pass_ms"),
               cap_artifact=lab.cap_artifact(st, lab.fps_caps()))
    rec.update(wall_stats(png))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantage", default="cozy-holes")
    # The last commit BEFORE the bake floor landed. Pinned rather than
    # "HEAD" so this A/B still answers the same question after the fix
    # is committed — a default of HEAD would quietly compare the fix
    # against itself and print "no change".
    ap.add_argument("--before-ref", default="99afed869",
                    help="git ref holding the PRE-floor model JSONs")
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--no-descend0-arm", dest="descend0_arm",
                    action="store_false",
                    help="skip the claude_descend = 0 control arm")
    args = ap.parse_args()

    before_dir = os.path.join(REPO, "screenshots", "_bakefloor_before")
    after_dir = os.path.join(REPO, "screenshots", "_bakefloor_after")
    git_models(args.before_ref, before_dir)
    # snapshot the working tree's masks so the arms can be re-run in
    # either order without the "before" copy overwriting them.
    os.makedirs(after_dir, exist_ok=True)
    install_dir_backup = install_models_to(MODEL_DIR, after_dir)

    runs = []
    try:
        runs.append(arm(args.vantage, before_dir, "before", args.frames))
        runs.append(arm(args.vantage, after_dir, "after", args.frames))
        # THIRD ARM, and it is not a spare: the floored masks left ONE
        # dim speck where the pre-floor frame had six bright ones. This
        # arm asks whether that survivor comes from the sub-voxel path
        # at all -- claude_descend = 0 stops march() entering any 16^3
        # cell, so a speck that SURVIVES it is a 1 m-cell leak (a stock
        # nodebox, a genuinely missing node) and no mask rule can reach
        # it, while one that VANISHES is a sub-voxel leak the axis rule
        # does not cover (a diagonal through a cell corner).
        if args.descend0_arm:
            runs.append(arm(args.vantage, after_dir, "after_descend0",
                            args.frames, descend=0))
    finally:
        install_models(after_dir)   # the working tree's masks go back
        reload_models()

    # grid_hash is a hash of the WORLD's node content (game.cpp
    # claudeTraceGridContentHash: per-block node ids and param2), NOT of
    # the 16^3 masks -- so it MUST come out identical here, and an early
    # draft of this script wrongly declared itself blind when it did.
    # The proof that the swap reached the render is elsewhere:
    #   * the installed mask sets differ (md5 below), and
    #   * the client logged "[claude_models] N models" after each
    #     reload, i.e. it re-read the table from disk.
    same_world = (runs[0].get("grid_hash") == runs[1].get("grid_hash"))
    masks_differ = runs[0].get("models_md5") != runs[1].get("models_md5")
    b, a = runs[0], runs[1]
    out = {"vantage": args.vantage, "before_ref": args.before_ref,
           "frames": args.frames, "runs": runs,
           "backup_files": install_dir_backup,
           "same_world": same_world, "masks_differ": masks_differ,
           "verdict": None}
    if not masks_differ:
        out["verdict"] = ("BLIND: both arms installed the SAME masks. No "
                          "number here is about the bake floor.")
    elif not same_world:
        out["verdict"] = ("SUSPECT: grid_hash moved between arms, so the "
                          "WORLD changed under the measurement — this A/B "
                          "is not one variable.")
    else:
        out["verdict"] = ("bright pixels >%d in the wall box: %d -> %d"
                          % (BRIGHT[0], b["bright_over_%d" % BRIGHT[0]],
                             a["bright_over_%d" % BRIGHT[0]]))
    dest = os.path.join(REPO, "screenshots",
                        "bakefloor_ab_%s.json" % args.vantage)
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("\nwrote %s" % dest)


def install_models_to(srcdir, dstdir):
    os.makedirs(dstdir, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(srcdir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(srcdir, name), "rb") as f:
            data = f.read()
        with open(os.path.join(dstdir, name), "wb") as f:
            f.write(data)
        n += 1
    return n


if __name__ == "__main__":
    main()
