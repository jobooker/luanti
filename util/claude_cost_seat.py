#!/usr/bin/env python3
"""Shared seat plumbing for the DESCEND COST INSTRUMENT (2026-08-16).

THE WORK THIS SERVES. On 2026-08-16 the walker learned to step inside a
class-250 cell -- the "descend" -- and the trace pass got much more
expensive: 11.9 -> 32.2 ms in the cosy cabin, which does descend, and
11.2 -> 15.2 ms in Cornell, which has no class-250 cell at all and had
the descend dial switched OFF. This module is the plumbing three
measuring scripts share: park the camera, restart the client on a chosen
shader variant, and read the per-pass GPU times the engine already
records. It measures. It changes nothing.

WHAT `pass_ms` IS. `RenderPipeline::run` (src/client/render/pipeline.cpp)
wraps every step of the depth-1 nested pipeline in a GL_TIME_ELAPSED
query and game.cpp dumps the exponentially-smoothed per-step
milliseconds into claude_stats.json. With the traced chain built by
`addPostProcessing` (secondstage.cpp) and bloom / auto-exposure / FXAA /
MSAA all off, the slots are, in execution order:

    0  Draw3D          Luanti's own raster draw of the whole world
    1  second_stage    the vanilla final merge
    2  claude_trace    THE TRACER -- the number this instrument is about
    3  claude_present  upsample + tone map
    4  SwapTextures    the ping-pong rename, ~0

`PASS_NAMES` below is that list, and `read_passes()` refuses to name
them if the client reports a different number of steps -- a renamed slot
is exactly the kind of silent mis-attribution this repo keeps paying
for.

LAWS OBSERVED (spec/environment-laws.md):
  * Release verified from build/CMakeCache.txt before any number.
  * minetest.conf is SCRATCH: the client rewrites it on exit, so the
    conf is re-pinned after every client death and before every start.
  * A shader compile failure is SILENT; debug.txt is read after every
    client start and a failure aborts.
  * fps_max / fps_max_unfocused pinned at 200, and every sample is
    checked for the sleep artifact (frame_ms_avg = busy_ms + sleep).
  * A measurement seat is not flown: free_move and friends are pinned
    off by claude_ci.pin_conf.
  * Park at a VANTAGE, never at an offset from one, and revive first --
    a dead player renders a dialog over every frame while the stats read
    healthy.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab       # noqa: E402
import claude_ci as ci         # noqa: E402

PASS_NAMES = ["raster3d", "second_stage", "trace", "present", "swap"]
VARIANT_TOOL = os.path.join(HERE, "claude_shader_variant.py")

# The park vantage: another room the body already rests in, so returning
# to the measured vantage is a real accumulator reset and never a fall.
PARK = {"cozy-ci": "cornell", "cornell": "furnace-050",
        "furnace-050": "cornell"}


def require_release():
    t = ci.build_type()
    if str(t).lower() not in ci.RELEASE_TYPES:
        raise SystemExit("CMAKE_BUILD_TYPE is %r, not Release -- "
                         "environment-laws forbids a benchmark here" % (t,))
    return t


def variant(*args):
    r = subprocess.run([sys.executable, VARIANT_TOOL] + list(args),
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("shader variant tool failed: %s%s"
                         % (r.stdout, r.stderr))
    return r.stdout.strip()


def stop_client():
    subprocess.run(["pkill", "-TERM", "-f", ci.SEAT_PATTERNS[1]], cwd=REPO)
    deadline = time.time() + ci.SEAT_TERM_WAIT
    while time.time() < deadline:
        if subprocess.run(["pgrep", "-f", ci.SEAT_PATTERNS[1]],
                          capture_output=True).returncode != 0:
            time.sleep(1.0)     # let the conf rewrite land before re-pinning
            return True
        time.sleep(0.5)
    subprocess.run(["pkill", "-KILL", "-f", ci.SEAT_PATTERNS[1]], cwd=REPO)
    time.sleep(1.5)
    return False


def _debug_size():
    try:
        return os.path.getsize(lab.DEBUG)
    except OSError:
        return 0


def restart_client(conf_extra=None, logdir=None):
    """Kill the client, re-pin the conf, start it again, and prove it
    came up with no shader compile failure. The SERVER is left alone --
    the world, the grid bake and the bridge all survive, so a restart
    moves exactly one thing: which shader text the driver compiled."""
    stop_client()
    ci.pin_conf()
    if conf_extra:
        path = os.path.join(REPO, "minetest.conf")
        lines = open(path).read().splitlines()
        keys = set(conf_extra)
        kept = [l for l in lines if l.split("=")[0].strip() not in keys]
        kept += ["%s = %s" % (k, v) for k, v in sorted(conf_extra.items())]
        open(path, "w").write("\n".join(kept) + "\n")
    mark = _debug_size()
    logdir = logdir or os.path.join(lab.SHOTS, "cost")
    os.makedirs(logdir, exist_ok=True)
    log = open(os.path.join(logdir, "client.log"), "ab")
    subprocess.Popen(ci.CLIENT_CMD, cwd=REPO, stdout=log,
                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
    if not ci.wait_for_client():
        raise SystemExit("client never became ready")
    fails = compile_failures(mark)
    if fails:
        raise SystemExit("SHADER COMPILE FAILURE (silent by default, so "
                         "this abort is the whole point):\n" + "\n".join(fails))
    return True


def compile_failures(since=0):
    """New "Failed to compile" lines in debug.txt since byte `since`."""
    try:
        with open(lab.DEBUG, errors="replace") as f:
            f.seek(since)
            return [l.strip() for l in f if "Failed to compile" in l]
    except OSError:
        return []


def park(vantage_name, dials=None, snapshot_tag=None):
    """Revive, push dials, teleport away and back, force a grid snapshot.
    Returns the stats sample once the client is answering again."""
    vs = lab.load_vantages()
    v = vs[vantage_name]
    away = vs[PARK.get(vantage_name, "cornell")]
    lab.rpc("revive", player="claude")
    if dials:
        lab.doorway(**dials)
    lab.goto(away)
    time.sleep(0.8)
    lab.goto(v)
    if snapshot_tag:
        lab.doorway(claude_grid_snapshot=snapshot_tag)
    time.sleep(2.0)
    return lab.read_stats()


def read_passes(settle_s=6.0, samples=3, gap=1.2):
    """Mean per-pass GPU ms once the EMA has caught up.

    pass_ms is an EMA with weight 0.1 per frame, so it needs ~30 frames
    to be within a few per cent of a step change; claude_stats.json is
    rewritten once a second. settle_s waits, then `samples` reads a
    second apart are averaged -- and their spread is returned too, so a
    reading taken while the EMA was still moving is visible rather than
    silently averaged in."""
    time.sleep(settle_s)
    rows = []
    for i in range(samples):
        if i:
            time.sleep(gap)
        st = lab.read_stats()
        if not st:
            continue
        rows.append(st)
    if not rows:
        raise SystemExit("no stats")
    out = {}
    n = len(rows[-1].get("pass_ms", []))
    names = PASS_NAMES if n == len(PASS_NAMES) else \
        ["slot%d" % i for i in range(n)]
    for i, name in enumerate(names):
        vals = [r["pass_ms"][i] for r in rows if len(r.get("pass_ms", [])) > i]
        out[name] = sum(vals) / len(vals)
        out[name + "_spread"] = max(vals) - min(vals)
    for k in ("frame_ms_avg", "busy_ms", "draw_ms", "fps", "still_frames",
              "grid_hash", "grid_solid", "accum_resets"):
        v = [r.get(k) for r in rows if r.get(k) is not None]
        if not v:
            continue
        out[k] = (sum(v) / len(v)) if isinstance(v[0], (int, float)) else v[-1]
    out["pass_slots"] = n
    out["pass_named"] = (n == len(PASS_NAMES))
    out["cap_artifact"] = lab.cap_artifact(rows[-1], lab.fps_caps())
    out["samples"] = len(rows)
    return out


def freeze_time():
    lab.rpc("cmd", command="set", param="time_speed 0")
    time.sleep(0.5)
    return lab.get_time_speed()


def mmm(vals):
    """min / median / max, the shape every number in this instrument is
    reported in."""
    s = sorted(vals)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {"min": round(s[0], 3), "median": round(med, 3),
            "max": round(s[-1], 3), "n": n}


def write_json(name, obj):
    dst = os.path.join(lab.SHOTS, "cost", name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write(json.dumps(obj, indent=1))
    print("written: %s" % dst)
    return dst
