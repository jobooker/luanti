#!/usr/bin/env python3
"""claude_ci — golden-image CI for the traced renderer.

One command builds the client, brings up a clean seat, and captures the
same four vantages every time: two furnace rooms (analytic referee), the
Cornell box (bleed referee), and the cozy cabin interior (eyes only).
Every shot lands beside its .capture.json, its referee stdout, and an
RMS diff against the previous run, in a run dir named for the commit.

COMMIT-THEN-CAPTURE. A capture is evidence only if it names a sha. Run
CI on a clean tree: the run dir and every row of the gallery are keyed
to `git rev-parse --short HEAD`. A dirty tree still runs, loudly, tagged
`<sha>-dirty` — those runs are scratch, never a golden. Convergence is
carried the same way: a shot whose capture record says NOT CONVERGED
wears a warning badge in both html files rather than being silently
compared. Referee scripts that exit nonzero are data, not CI failures.

Each shot is diffed TWICE: against the previous run (what did this
commit change?) and against the pinned golden (has quality drifted by
inches?). Pin one with `golden`; without a pin the second column is —.

Quickstart:
  python3 util/claude_ci.py run                  # build, seat, 4 shots
  python3 util/claude_ci.py run --skip-build     # reuse ./bin/luanti
  open screenshots/ci/index.html                 # the master gallery

Future work (not implemented): referee calibration mode — periodically
capture a deliberately broken frame (e.g. claude_bounces=1) and assert
each referee still FAILS it; an instrument that can't see a planted
defect is blind.
"""
import argparse
import glob
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab  # noqa: E402  (same dir; shares the file-RPC channel)

# ---------------------------------------------------------------- constants
# Nothing below this line is allowed to hide in the body of the script.

SETTLE_DEFAULT = 60.0      # s of stillness before each shot (accumulator depth)
FREEZE_WAIT = 8.0          # s between the two still_frames reads in freeze
SEAT_PORT = 30000          # server port for the CI seat
SEAT_WORLD = "worlds/gallery"
SEAT_CLIENT_NAME = "claude"
SEAT_BOOT_WAIT = 6.0       # s to let the server listen before the client dials
SEAT_TERM_WAIT = 20.0      # s to wait for SIGTERM'd seat processes to exit
CLIENT_READY_TIMEOUT = 60.0  # s to wait for the client's first stats write
BUILD_CMD = ["bash", "util/ci/build.sh"]

SERVER_CMD = ["./bin/luantiserver", "--world", SEAT_WORLD,
              "--port", str(SEAT_PORT)]
CLIENT_CMD = ["./bin/luanti", "--address", "127.0.0.1", "--port",
              str(SEAT_PORT), "--name", SEAT_CLIENT_NAME, "--go"]
# pkill -f patterns for the two known seat invocations (RUNTIME only — the
# script owns the seat while it runs, and leaves it up afterwards).
SEAT_PATTERNS = ["bin/luantiserver --world " + SEAT_WORLD,
                 "bin/luanti --address 127.0.0.1"]

# Canonical photo state. Every capture is taken with exactly these dials.
CANONICAL_DIALS = {"claude_view": 0, "claude_bounces": 24}

# Pinned capture resolution AND frame pacing. Luanti SAVES its window
# size back into minetest.conf on exit, so one manual resize silently
# changes every future capture and breaks pixel comparison against the
# golden (discovered run #3: 2880x1576 vs the golden's 1920x1080).
#
# fps_max / fps_max_unfocused are here for the same reason and were
# MISSING until 2026-08-15: FpsControl::limit sleeps to fps_max when the
# window is focused and fps_max_unfocused when it is not, and both have
# hidden defaults (60 / 10). A CI seat's window is never focused, so
# every headless run before today slept to 100 ms frames and reported
# them as frame_ms_avg — a sleeping client reading as a slow renderer.
# 200 is above anything this renderer reaches, so no sleep is taken.
# Keep this list identical to the fps block in claude_seat_conf.ref.
#
# These keys are forced into minetest.conf before every seat start;
# autosave_screensize=false stops the exit-save from undoing it.
PINNED_CONF = {"screen_w": "1920", "screen_h": "1080",
               "fullscreen": "false", "window_maximized": "false",
               "autosave_screensize": "false",
               "fps_max": "200", "fps_max_unfocused": "200",
               # Without this key the console->client dial channel is
               # SILENTLY DEAD: pollSettingsPatch reads claude_dial_file,
               # an empty path means "off", and /dial then writes its
               # file and reports success from the SERVER while the
               # client never reads it (found 2026-08-15 during the NEE
               # eyeball — the same shape as the /set trap: the
               # confirmation renders on the client's screen and proves
               # nothing). It was in claude_seat_conf.ref all along, but
               # nothing installs that file.
               "claude_dial_file": SEAT_WORLD + "/claude_dial.conf"}

# The dial file is a HUMAN's channel and it persists in the world dir
# across sessions, so a leftover /dial from yesterday would be applied
# to a fresh seat on its first poll — and applied AFTER the patch file
# in the same tick, i.e. it wins. Wiring the channel (above) without
# clearing it would hand every CI run a stale shadow dial.
DIAL_HEADER = ("# claude_dial.conf: console-set client dials (/dial),\n"
               "# merged in by the client's claude_dial_file poll (~1 Hz)\n"
               "# CLEARED at seat start by claude_ci.pin_conf.\n")


def clear_dial_file():
    """Empty the /dial channel so a human's leftover dial cannot shadow a
    run's own dials. See DIAL_HEADER for why this is not optional."""
    path = os.path.join(REPO, SEAT_WORLD, "claude_dial.conf")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(DIAL_HEADER)
    except Exception as e:
        print("WARNING: could not clear the dial file: %s" % e)


def pin_conf():
    """Force PINNED_CONF keys into minetest.conf (idempotent), and clear
    the /dial channel. Both are seat hygiene: a key missing here is a
    channel silently off, and a dial left there is a setting silently on."""
    clear_dial_file()
    path = os.path.join(REPO, "minetest.conf")
    lines = open(path).read().splitlines() if os.path.exists(path) else []
    keys = set(PINNED_CONF)
    kept = [l for l in lines
            if l.split("=")[0].strip() not in keys]
    kept += ["%s = %s" % (k, v) for k, v in sorted(PINNED_CONF.items())]
    open(path, "w").write("\n".join(kept) + "\n")

CI_DIR = os.path.join(REPO, "screenshots", "ci")
# The pinned golden: one line, a run dir name. Comparing only to the
# previous run lets quality drift by inches, one invisible step per commit.
GOLDEN_FILE = os.path.join(CI_DIR, "GOLDEN")
CI_TAG = "ci"              # the key that marks a vantage as part of this set
# (vantage, referee kind, referee arg) — stable order, and the only shots
# this harness takes. Every name here must carry "ci": true in the json.
CI_VANTAGES = [("furnace-050", "furnace", "050"),
               ("furnace-073", "furnace", "073"),
               ("cornell", "cornell", None),
               ("cozy-ci", None, None)]

# Room doorways (claude_gallery_deploy layout: floor y=8, walk y=9, doors
# on the z=0 line). The referee rooms MUST be shut to be referees — an
# open door leaks the sky into an analytic furnace. Plugged before the
# vantage loop (so the last scene churn precedes every settle), reopened
# on the way out so the gallery stays walkable.
CI_DOORS = [{"pos": {"x": 20, "y": 9, "z": 0}, "name": None},   # furnace-050
            {"pos": {"x": 33, "y": 9, "z": 0}, "name": None},   # furnace-073
            {"pos": {"x": 47, "y": 9, "z": 0},
             "name": "claude_bridge:gray221"}]                  # cornell

# Furnace referee measurement patch. None = the referee's own default, a
# 200x200 block at (w//4 +/- 100, h//2 +/- 100) — resolution-independent.
# Set to (x0, y0, x1, y1) to pin it against a resolution change.
FURNACE_PATCH = None
FURNACE_RATIO_TOL = 0.15   # |measured/analytic - 1| per channel for a pass
THUMB_MAX_PX = 460         # css width cap on gallery thumbnails (no resizing)


# ---------------------------------------------------------------- helpers

def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, **kw)


def git_state():
    def g(*a):
        return sh(["git"] + list(a)).stdout.strip()
    dirty = bool(g("status", "--porcelain"))
    sha = g("rev-parse", "--short", "HEAD")
    return {"sha": sha, "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": dirty, "tag": sha + ("-dirty" if dirty else "")}


def stop_seat():
    """SIGTERM the two known invocations and wait for them to actually go."""
    for pat in SEAT_PATTERNS:
        sh(["pkill", "-TERM", "-f", pat])
    deadline = time.time() + SEAT_TERM_WAIT
    while time.time() < deadline:
        if all(sh(["pgrep", "-f", p]).returncode != 0 for p in SEAT_PATTERNS):
            return True
        time.sleep(0.5)
    for pat in SEAT_PATTERNS:
        sh(["pkill", "-KILL", "-f", pat])
    time.sleep(1.0)
    return False


def start_seat(rundir):
    logs = {}
    for tag, cmd, wait in (("server", SERVER_CMD, SEAT_BOOT_WAIT),
                           ("client", CLIENT_CMD, 0.0)):
        logs[tag] = open(os.path.join(rundir, tag + ".log"), "wb")
        subprocess.Popen(cmd, cwd=REPO, stdout=logs[tag],
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
        time.sleep(wait)


def wait_for_client(timeout=CLIENT_READY_TIMEOUT):
    """Ready == the client answers a stats request with a FRESH stats file.
    A stale claude_stats.json from a previous session reads as ready, so
    the mtime must postdate this call."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            lab.doorway(claude_stats=1)          # sleeps ~1.4 s itself
            time.sleep(0.6)
            if os.path.getmtime(lab.STATS) > t0 and lab.read_stats():
                return True
        except Exception:
            time.sleep(1.0)
    return False


def do_freeze():
    """LAB RULE #1, as data instead of an exit code: stop time and prove
    the accumulator deepens. A run that fails this is not aborted — it is
    recorded, and every shot in it will carry its own convergence line."""
    try:
        lab.rpc("cmd", command="set", param="time_speed 0")
    except Exception as e:
        return {"error": str(e), "deepening": False}
    lab.doorway(claude_stats=1)
    time.sleep(1.5)
    a = lab.read_stats() or {}
    time.sleep(FREEZE_WAIT)
    lab.doorway(claude_stats=1)
    time.sleep(1.5)
    b = lab.read_stats() or {}
    sa, sb = a.get("still_frames", 0), b.get("still_frames", 0)
    return {"time_speed": lab.get_time_speed(),
            "still_frames": [sa, sb],
            "accum_alpha": [a.get("accum_alpha"), b.get("accum_alpha")],
            "deepening": bool(sb > sa)}


def set_doors(shut):
    out = []
    for d in CI_DOORS:
        kw = dict(pos=d["pos"], shut=shut)
        if shut and d["name"]:
            kw["name"] = d["name"]
        try:
            out.append(lab.rpc("door", **kw))
        except Exception as e:
            out.append({"error": str(e), "pos": d["pos"]})
    return out


def capture(name, vantage, rundir, settle):
    """vantage -> settle -> shot, filed into the run dir under its name."""
    lab.goto(vantage)
    png = lab.shot(name, settle=settle)          # settle happens inside shot
    dst = os.path.join(rundir, name + ".png")
    shutil.copy2(png, dst)
    rec = os.path.splitext(png)[0] + ".capture.json"
    cap = {}
    if os.path.exists(rec):
        shutil.copy2(rec, os.path.join(rundir, name + ".capture.json"))
        cap = json.load(open(rec))
    st = cap.get("stats") or {}
    return dst, {"frozen": cap.get("frozen"),
                 "time_speed": cap.get("time_speed"),
                 "sha": cap.get("sha"), "dirty": cap.get("dirty"),
                 "still_frames": st.get("still_frames"),
                 "accum_alpha": st.get("accum_alpha"), "fps": st.get("fps")}


# ---------------------------------------------------------------- referees

def run_referee(kind, arg, png, rundir, name):
    script = os.path.join(HERE, "claude_%s_check.py" % kind)
    cmd = [sys.executable, script, png] + ([arg] if arg else [])
    if kind == "furnace" and FURNACE_PATCH:
        cmd += ["--patch"] + [str(v) for v in FURNACE_PATCH]
    r = sh(cmd)
    text = (r.stdout or "") + (r.stderr or "")
    with open(os.path.join(rundir, name + ".referee.txt"), "w") as f:
        f.write("$ %s\n\n%s\n[exit %d]\n" % (" ".join(cmd), text, r.returncode))
    out = {"kind": kind, "arg": arg, "returncode": r.returncode,
           "txt": name + ".referee.txt", "stdout": text}
    out.update(parse_furnace(text) if kind == "furnace" else parse_cornell(text))
    return out


def parse_furnace(t):
    """Headline: measured/analytic per channel (1.000 == correct transport)."""
    out = {"ratio_analytic": {}}
    for ch, ratio in re.findall(
            r"^([RGB])\s+measured\s+[-\d.]+.*?analytic Le/\(1-rho\)\s+[-\d.]+"
            r"\s+\(ratio\s+([-\d.]+)\)", t, re.M):
        out["ratio_analytic"][ch] = float(ratio)
    m = re.search(r"clipped\s+([-\d.]+)%", t)
    if m:
        out["clipped_pct"] = float(m.group(1))
    return out


def parse_cornell(t):
    out = {}
    m = re.search(r"R/G ratio: left\s+([-\d.]+)\s+vs right\s+([-\d.]+)\s+->\s+(.+)", t)
    if m:
        out["flank_rg"] = {"left": float(m.group(1)), "right": float(m.group(2))}
        out["bleed_verdict"] = m.group(3).strip()
    m = re.search(r"brightest 1% mean linear luminance:\s+([-\d.]+)", t)
    if m:
        out["emitter_brightest"] = float(m.group(1))
    m = re.search(r"ceiling flanks \(indirect-only\) mean:\s+([-\d.]+)", t)
    if m:
        out["ceiling_flanks"] = float(m.group(1))
    m = re.search(r"corner profile: \d+ samples, (\d+) large jumps", t)
    if m:
        out["corner_jumps"] = int(m.group(1))
    return out


def verdict(ref):
    """(mark, key number). '-' whenever the referee could not speak."""
    if not ref:
        return "-", "no referee"
    if ref["returncode"] != 0:
        return "-", "referee exit %d" % ref["returncode"]
    if ref["kind"] == "furnace":
        r = ref.get("ratio_analytic") or {}
        if len(r) != 3:
            return "-", "unparsed"
        worst = max(r.values(), key=lambda v: abs(v - 1.0))
        ok = all(abs(v - 1.0) <= FURNACE_RATIO_TOL for v in r.values())
        return ("PASS" if ok else "FAIL"), "ratio %.3f" % worst
    v = ref.get("bleed_verdict")
    if not v:
        return "-", "unparsed"
    f = ref.get("flank_rg") or {}
    key = "R/G %.3f vs %.3f" % (f.get("left", 0), f.get("right", 0))
    return ("PASS" if v.startswith("COLOR BLEED PRESENT") else "FAIL"), key


# ---------------------------------------------------------------- run dirs

def load_runs():
    """Every run dir with a run.json, newest first (the name sorts by UTC)."""
    runs = []
    for p in sorted(glob.glob(os.path.join(CI_DIR, "*", "run.json")), reverse=True):
        try:
            runs.append((os.path.basename(os.path.dirname(p)), json.load(open(p))))
        except Exception:
            pass
    return runs


def previous_run(this_id):
    for rid, _ in load_runs():
        if rid < this_id:
            return rid
    return None


def read_golden():
    try:
        rid = open(GOLDEN_FILE).read().strip()
    except Exception:
        return None
    return rid or None


def diff_against(run_id, vantage, png):
    """RMS of one shot vs the same vantage in another run dir (HUD cropped)."""
    if not run_id:
        return None
    other = os.path.join(CI_DIR, run_id, vantage + ".png")
    if not os.path.exists(other):
        return None
    try:
        return lab.rms_diff(other, png)
    except Exception as ex:
        return {"error": str(ex)}


# ---------------------------------------------------------------- html

CSS = """body{background:#101214;color:#d8d8d8;font:14px/1.5 -apple-system,
Segoe UI,Helvetica,sans-serif;margin:0;padding:24px}
a{color:#7fb2ff}h1,h2{font-weight:600;margin:0 0 6px}
h1{font-size:20px}h2{font-size:16px}
.meta{color:#8a8f96;font-size:12px;margin-bottom:18px}
.shot{display:flex;gap:18px;align-items:flex-start;border-top:1px solid #23262a;
padding:18px 0}.shot img{width:100%%;max-width:900px;border:1px solid #23262a}
pre{background:#17191c;border:1px solid #23262a;padding:10px;overflow-x:auto;
font:12px/1.45 SFMono-Regular,Menlo,monospace;color:#c8ccd0;margin:0}
.col{flex:1;min-width:280px}
.run{border-top:1px solid #23262a;padding:16px 0}
.cells{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px}
.cell{width:%dpx}.cell img{width:100%%;display:block;border:1px solid #23262a}
.cap{font-size:12px;color:#9aa0a6;margin-top:4px}
.vPASS{color:#6ee787;font-weight:700}.vFAIL{color:#ff8a8a;font-weight:700}
.warn{display:inline-block;background:#5a3a00;color:#ffd479;border:1px solid
#8a5c00;padding:2px 8px;border-radius:3px;font-size:12px;font-weight:700}
.dirty{background:#5a1010;color:#ffb4b4;border:1px solid #8a1c1c;padding:1px 6px;
border-radius:3px;font-size:11px}
""" % THUMB_MAX_PX


def _badge(shot):
    f = (shot.get("capture") or {}).get("frozen") or ""
    return ('<span class="warn">NOT CONVERGED</span> ' if "NOT CONVERGED" in f
            or "unknown" in f else "")


def _mark(shot):
    """Referee verdict as a glyph. Plain text, so it serves html and tty."""
    m = (shot.get("verdict") or ["-", ""])[0]
    return '<span class="v%s">%s</span>' % (
        m.strip("-") or "none",
        {"PASS": "&#10003;", "FAIL": "&#10007;"}.get(m, "&mdash;"))


def _one_rms(d):
    if not d:
        return "n/a"
    return "err" if d.get("error") else \
        "%.3f (worst %.1f)" % (d["rms"], d["worst_channel_delta"])


def _rms(shot):
    """Both diffs, one string — previous run AND the pinned golden."""
    return "prev %s | golden %s" % (_one_rms(shot.get("rms_vs_prev")),
                                    _one_rms(shot.get("rms_vs_golden")))


def write_run_html(rundir, run):
    e, parts = html.escape, []
    parts.append("<title>CI %s</title><style>%s</style>" % (e(run["tag"]), CSS))
    parts.append("<h1>%s <span class=meta>%s</span></h1>" % (e(run["tag"]), e(run["branch"])))
    parts.append('<div class=meta>%s &middot; settle %gs &middot; freeze: %s'
                 ' &middot; prev %s &middot; golden %s &middot; '
                 '<a href="../index.html">all runs</a></div>'
                 % (e(run["started_utc"]), run["settle"],
                    "deepening" if (run.get("freeze") or {}).get("deepening")
                    else "NOT DEEPENING", e(run.get("prev_run") or "none"),
                    e(run.get("golden_run") or "none pinned")))
    for name, _, _ in CI_VANTAGES:
        s = run["shots"].get(name) or {}
        if not s.get("png"):
            parts.append("<div class=shot><div class=col><h2>%s</h2><div "
                         "class=meta>no capture: %s</div></div></div>"
                         % (e(name), e(s.get("error", "not attempted"))))
            continue
        ref = (s.get("referee") or {}).get("stdout") or "(no referee for this vantage)"
        parts.append(
            '<div class=shot><div class=col style="flex:2"><img src="%s"></div>'
            '<div class=col><h2>%s %s</h2><div class=meta>%s %s &middot; %s</div>'
            "<pre>%s</pre></div></div>"
            % (e(s["png"]), e(name), _mark(s), _badge(s),
               e(str((s.get("capture") or {}).get("frozen"))), _rms(s), e(ref)))
    p = os.path.join(rundir, "index.html")
    open(p, "w").write("\n".join(parts))
    return p


def write_master_html():
    e, parts = html.escape, []
    gold = read_golden()
    parts.append("<title>claude_ci gallery</title><style>%s</style>" % CSS)
    parts.append("<h1>claude_ci &mdash; golden images</h1>")
    parts.append("<div class=meta>newest first. dirty runs are scratch, never "
                 "a golden. pinned golden: %s &mdash; every shot is diffed "
                 "against it AND against the run above it.</div>"
                 % e(gold or "none (claude_ci.py golden &lt;run-dir&gt;)"))
    for rid, run in load_runs():
        d = ' <span class=dirty>DIRTY</span>' if run.get("dirty") else ""
        if rid == gold:
            d += ' <span class=warn>GOLDEN</span>'
        parts.append('<div class=run><h2><a href="%s/index.html">%s</a>%s '
                     '<span class=meta>%s &middot; %s</span></h2><div class=cells>'
                     % (e(rid), e(run.get("tag", rid)), d,
                        e(run.get("branch", "?")), e(run.get("started_utc", "?"))))
        for name, _, _ in CI_VANTAGES:
            s = (run.get("shots") or {}).get(name) or {}
            if not s.get("png"):
                parts.append('<div class=cell><div class=cap>%s &mdash; %s'
                             "</div></div>"
                             % (e(name), e(s.get("error", "missing"))))
                continue
            key = (s.get("verdict") or ["-", ""])[1]
            parts.append('<div class=cell><a href="%s/%s"><img src="%s/%s"></a>'
                         "<div class=cap>%s%s %s %s<br>%s</div></div>"
                         % (e(rid), e(s["png"]), e(rid), e(s["png"]), _badge(s),
                            e(name), _mark(s), e(str(key)), _rms(s)))
        parts.append("</div></div>")
    os.makedirs(CI_DIR, exist_ok=True)
    p = os.path.join(CI_DIR, "index.html")
    open(p, "w").write("\n".join(parts))
    return p


# ---------------------------------------------------------------- the run

def cmd_run(args):
    git = git_state()
    if git["dirty"]:
        print("!" * 70)
        print("!! WORKING TREE IS DIRTY. This run is SCRATCH, not a golden.")
        print("!! Captures map to shas; %s does not name a commit." % git["tag"])
        print("!" * 70)
    run_id = "%s_%s" % (time.strftime("%Y%m%d-%H%M%S", time.gmtime()), git["tag"])
    rundir = os.path.join(CI_DIR, run_id)
    os.makedirs(rundir, exist_ok=True)
    print("run dir: %s" % rundir)

    # per-run dial state: canonical photo dials plus the estimator switch
    # (claude_nee). §8 clause 5: the dial state is recorded with the run,
    # because "nee=1 agrees with the nee=0 golden" is only evidence if
    # both runs say which they were.
    dials = dict(CANONICAL_DIALS, claude_nee=args.nee)
    run = dict(git, run_id=run_id, settle=args.settle,
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               dials=dials, shots={})

    if args.skip_build:
        run["build"] = {"ran": False}
    else:
        print("building: %s" % " ".join(BUILD_CMD))
        r = subprocess.run(BUILD_CMD, cwd=REPO)
        run["build"] = {"ran": True, "returncode": r.returncode}
        if r.returncode != 0:
            run["aborted"] = "build failed"
            json.dump(run, open(os.path.join(rundir, "run.json"), "w"), indent=2)
            print("BUILD FAILED (%d) — run aborted." % r.returncode)
            return 1

    print("seat: stopping any running luanti...")
    run["seat_clean_stop"] = stop_seat()
    pin_conf()
    run["pinned_conf"] = PINNED_CONF
    print("seat: starting server + client on port %d" % SEAT_PORT)
    start_seat(rundir)
    if not wait_for_client():
        run["aborted"] = "client never became ready"
        json.dump(run, open(os.path.join(rundir, "run.json"), "w"), indent=2)
        print("CLIENT NOT READY after %.0fs — run aborted." % CLIENT_READY_TIMEOUT)
        return 1
    print("seat: client ready")

    run["freeze"] = do_freeze()
    if not run["freeze"].get("deepening"):
        print("!! FREEZE FAILED: the accumulator is not deepening. Every shot "
              "below is suspect — read its convergence line.")
    lab.doorway(**dials)
    print("dials: %s" % dials)

    vs = lab.load_vantages()
    run["doors_shut"] = set_doors(True)
    try:
        prev, gold = previous_run(run_id), read_golden()
        run["prev_run"], run["golden_run"] = prev, gold
        for name, kind, arg in CI_VANTAGES:
            if name not in vs:
                print("%-14s MISSING from claude_vantages.json" % name)
                continue
            if not vs[name].get(CI_TAG):
                print("%-14s not tagged %r — capturing anyway" % (name, CI_TAG))
            try:
                png, cap = capture(name, vs[name], rundir, args.settle)
            except Exception as ex:
                # One dead vantage must not cost the other three.
                run["shots"][name] = {"error": str(ex)}
                print("%-14s CAPTURE FAILED: %s" % (name, ex))
                continue
            shot = {"png": os.path.basename(png), "capture": cap}
            if kind:
                shot["referee"] = run_referee(kind, arg, png, rundir, name)
            shot["verdict"] = list(verdict(shot.get("referee")))
            shot["rms_vs_prev"] = diff_against(prev, name, png)
            shot["rms_vs_golden"] = diff_against(gold, name, png)
            run["shots"][name] = shot
            print("%-14s %s %s  %s" % (name, shot["verdict"][0],
                                       shot["verdict"][1], _rms(shot)))
    finally:
        run["doors_reopened"] = set_doors(False)

    run["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # strip the bulky referee stdout out of run.json (it lives in the .txt)
    disk = json.loads(json.dumps(run))
    for s in disk["shots"].values():
        if s.get("referee"):
            s["referee"].pop("stdout", None)
    json.dump(disk, open(os.path.join(rundir, "run.json"), "w"), indent=2)
    write_run_html(rundir, run)
    master = write_master_html()

    marks = " ".join("%s=%s" % (n, run["shots"].get(n, {}).get("verdict", ["-"])[0])
                     for n, _, _ in CI_VANTAGES)
    print("%s %s@%s  %s  (prev %s, golden %s)"
          % ("DIRTY" if run["dirty"] else "clean", run["branch"], run["sha"],
             marks, run.get("prev_run") or "none",
             run.get("golden_run") or "none"))
    print(master)
    return 0


def cmd_golden(args):
    """Pin the run every later run is measured against. No arg = show it."""
    if not args.run_dir:
        print(read_golden() or "no golden pinned")
        return 0
    rid = os.path.basename(args.run_dir.rstrip("/"))
    if not os.path.exists(os.path.join(CI_DIR, rid, "run.json")):
        print("no such run: %s" % rid)
        print("known: %s" % (", ".join(r for r, _ in load_runs()) or "(none)"))
        return 1
    run = json.load(open(os.path.join(CI_DIR, rid, "run.json")))
    if run.get("dirty"):
        print("REFUSING: %s was captured from a dirty tree. A golden must "
              "name a commit." % rid)
        return 1
    open(GOLDEN_FILE, "w").write(rid + "\n")
    write_master_html()
    print("golden pinned: %s (%s@%s)" % (rid, run.get("branch"), run.get("sha")))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="build, seat, capture, referee, publish")
    p.add_argument("--settle", type=float, default=SETTLE_DEFAULT,
                   help="seconds of stillness before each shot (default %(default)s)")
    p.add_argument("--skip-build", action="store_true",
                   help="capture with the binaries already in ./bin")
    p.add_argument("--nee", type=int, default=1, choices=(0, 1),
                   help="claude_nee for this run: 1 = NEE+MIS estimator "
                        "(default), 0 = pure photo path (golden mode)")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("golden", help="pin/show the run all diffs measure against")
    p.add_argument("run_dir", nargs="?", help="a run dir name under screenshots/ci/")
    p.set_defaults(func=cmd_golden)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
