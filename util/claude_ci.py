#!/usr/bin/env python3
"""claude_ci — golden-image CI for the traced renderer. IT CAN GO RED.

One command builds the client, brings up a clean seat, and captures the
same five arms every time: two furnace rooms (analytic referee), the
Cornell box in photo mode AND with the NEE estimator on (region-ratio
referee), and the cozy cabin interior (eyes only). Every shot lands
beside its .capture.json, its referee stdout, and an RMS diff against
the previous run, in a run dir named for the commit.

RED/GREEN. Until 2026-08-15 this script was a REPORTER: referee output
was "data, not CI failures" and the process exited 0 whatever the frame
looked like, so nothing could ever go red and the measurement that
convicted NEE (roadmap 1a) lived only in a session transcript. Now every
capture feeds named ASSERTIONS, `run` prints GREEN or `RED: n failing —
names` and exits nonzero on red, and a referee that cannot speak (script
crash, missing golden) is RED, not a pass — an instrument that cannot
fail is not evidence (physics-contract §8 clause 3).

`calibrate` is the other half of that clause: it captures the referee
vantages with a deliberately broken dial and asserts every referee FAILS
the frame. Any referee that passes a planted defect is named BLIND, and
what each one is blind to goes in spec/measured.md.

COMMIT-THEN-CAPTURE. A capture is evidence only if it names a sha. Run
CI on a clean tree: the run dir and every row of the gallery are keyed
to `git rev-parse --short HEAD`. A dirty tree still runs, loudly, tagged
`<sha>-dirty` — those runs are scratch, never a golden. RELEASE ONLY:
util/ci/build.sh defaults to Debug (~4.5x slower — environment-laws
"Seat / build"), so this script asks it for Release and refuses to
capture against a Debug build tree.

Each shot is diffed TWICE: against the previous run (what did this
commit change?) and against the pinned golden (has quality drifted by
inches?). Pin one with `golden`; without a pin the second column is —
and the Cornell region assertions cannot run at all.

Quickstart:
  python3 util/claude_ci.py run                  # build, seat, 5 shots
  python3 util/claude_ci.py run --skip-build     # reuse ./bin/luanti
  python3 util/claude_ci.py calibrate            # can the referees fail?
  open screenshots/ci/index.html                 # the master gallery
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
import claude_cornell_check as cornell  # noqa: E402  (the region referee)
import claude_rooms_check as rooms  # noqa: E402  (are the rooms still rooms?)

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
# util/ci/build.sh reads CMAKE_BUILD_TYPE and DEFAULTS IT TO DEBUG, which
# is the one thing environment-laws forbids for a measurement (Release is
# ~4.5x faster; the cache was Debug for an unknown period). Ask for
# Release explicitly here AND refuse a Debug cache below — the build
# command and the cache are two different claims.
BUILD_ENV = {"CMAKE_BUILD_TYPE": "Release"}
RELEASE_TYPES = ("release", "relwithdebinfo")
CMAKE_CACHE = os.path.join(REPO, "build", "CMakeCache.txt")

SERVER_CMD = ["./bin/luantiserver", "--world", SEAT_WORLD,
              "--port", str(SEAT_PORT)]
CLIENT_CMD = ["./bin/luanti", "--address", "127.0.0.1", "--port",
              str(SEAT_PORT), "--name", SEAT_CLIENT_NAME, "--go"]
# pkill -f patterns for the two known seat invocations (RUNTIME only — the
# script owns the seat while it runs, and leaves it up afterwards).
SEAT_PATTERNS = ["bin/luantiserver --world " + SEAT_WORLD,
                 "bin/luanti --address 127.0.0.1"]

# Canonical photo state. Every capture is taken with exactly these dials,
# pushed explicitly and PROVEN from the client's own log at each shutter
# (the hidden-default class: an unset claude_* dial is a silent zero, and
# the /dial file is a second channel that WINS — environment-laws).
CANONICAL_DIALS = {
    "claude_view": 0,           # photo; 6 (clay) was left in the conf once
    "claude_bounces": 24,
    # The traced pipeline master switch (0 = raster pass-through, 3 =
    # traced). Pushed EXPLICITLY: it lived only in minetest.conf, and a
    # conf that lost the line would have had CI silently photographing
    # the raster renderer and calling it a golden.
    "claude_volume_debug": 3,
    "claude_nee": 0,            # PHOTO MODE IS THE TRUTH (§6). The
                                # estimator gets its own arm below.
    # RNG source: 1 = the counter-based PCG, the default since roadmap
    # 1a. 0 is the old hash chain, which biased the direction sampler by
    # ~9% and made photo mode 1-8% dark in Cornell — every Cornell number
    # before measured.md's "1a" section was taken with it. Pushed
    # EXPLICITLY and proven, like the rest: an unset claude_* dial is a
    # silent default (environment-laws, the hidden-default class), and
    # this one decides what the Monte Carlo integrator integrates.
    "claude_rng": 1,
    "claude_stats": 1,
    # freeze the volume bubble: the periodic re-snap is a ~295 ms hitch
    # every 3 s and a silent scene change under a settling accumulator.
    # With follow off, each capture takes its own snapshot after the
    # teleport (see capture()).
    "claude_volume_follow": 0,
    # --- overlay suppression, and it is NOT cosmetic --------------------
    # Region means and RMS compare PIXELS. The hotbar covers the only
    # floor this vantage can see, and each capture's "Saved screenshot
    # to ..." chat line lands in the ceiling-flank band: measured
    # 2026-08-15, the same ceiling box reads 1.09 against a one-chat-line
    # golden and 4.93 against a two-line one. show_hud/show_chat were
    # F1/F2-only until 629368367 made them settings.
    "claude_show_hud": 0,
    "claude_show_chat": 0,
    # A measurement seat is not drivable by hand: mouse-look and
    # movement are ignored while this is set (game.cpp
    # claudeInputLocked). Turning does not reset the accumulator, so a
    # stray hand blends two views into one "converged" frame and leaves
    # no other trace — the aim guard catches it after the fact, this
    # stops it happening.
    "claude_input_lock": 1,
    "recent_chat_messages": 0,
    "node_highlighting": "none",
}
# The dials proven per capture. The rest are pushed but not asserted;
# these three are the ones that have silently invalidated measurements.
PROVEN_DIALS = ("claude_view", "claude_nee", "claude_bounces",
                "claude_volume_debug", "claude_rng")

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

# The shot list. `name` is the arm (and the file name), `vantage` the
# saved camera, `dials` the per-arm override on CANONICAL_DIALS.
#
# cornell-nee1 is the estimator arm: the SAME vantage as cornell, with
# claude_nee = 1. Two arms of one room in one run is the only way the
# estimator can be judged against the truth on the same seat, same
# build, same settle — §6 says an estimator must agree with photo mode
# in expectation, so this pair is the test of the whole rung.
CI_SHOTS = [
    {"name": "furnace-050", "vantage": "furnace-050",
     "referee": ("furnace", "050")},
    {"name": "furnace-073", "vantage": "furnace-073",
     "referee": ("furnace", "073")},
    {"name": "cornell", "vantage": "cornell", "referee": ("cornell", None)},
    {"name": "cornell-nee1", "vantage": "cornell", "referee": ("cornell", None),
     "dials": {"claude_nee": 1}},
    {"name": "cozy-ci", "vantage": "cozy-ci", "referee": None},
]
# The golden image every Cornell region ratio is measured against: the
# PHOTO arm of the pinned golden run.
GOLDEN_REF_SHOT = "cornell"

# Room doorways (claude_gallery_deploy layout: floor y=8, walk y=9, doors
# on the z=0 line). The referee rooms MUST be shut to be referees — an
# open door leaks the sky into an analytic furnace. Plugged before the
# vantage loop (so the last scene churn precedes every settle), reopened
# on the way out so the gallery stays walkable.
CI_DOORS = [{"pos": {"x": 20, "y": 9, "z": 0}, "name": None},   # furnace-050
            {"pos": {"x": 33, "y": 9, "z": 0}, "name": None},   # furnace-073
            {"pos": {"x": 47, "y": 9, "z": 0},
             "name": "claude_bridge:gray221"}]                  # cornell

# Referee-room integrity. A room is a referee only while it is SEALED
# and made of exactly the nodes its builder laid down: one dug node at
# (47,11,8) leaked daylight into Cornell for an unknown period and
# invalidated a day of numbers, and on 2026-08-15 the same room was
# found missing four front-wall nodes AND one of its two shadow
# occluders. Every run now walks all three referee rooms node by node
# (claude_rooms_check, through OPS.scan) and asserts zero off-spec
# nodes; the room hash goes in run.json. Off-spec is RED, not a warning.

# Furnace referee measurement patch. None = the referee's own default, a
# 200x200 block at (w//4 +/- 100, h//2 +/- 100) — resolution-independent.
# Set to (x0, y0, x1, y1) to pin it against a resolution change.
FURNACE_PATCH = None

# --- tolerances, and where each number came from -------------------------
#
# FURNACE. The old FURNACE_RATIO_TOL = 0.15 could not see a 5% transport
# error — it certified the NEE bias as "exact to the instrument" while
# Cornell caught it in one run. Replaced by a per-room pinned value and a
# tolerance derived from the ROOM'S OWN reproducibility: 3x the spread of
# the ratio across two consecutive clean runs (see measured.md "CI
# red/green"). -050 is the precision energy referee (unclipped, 0.982
# across every build). -073 SATURATES the ACES inversion (100% clipped),
# so its ratio carries ~10% slop by construction and its tolerance is
# wide on purpose: it is a truncation detector, not a precision one.
# DERIVED 2026-08-15 from two consecutive clean Release runs
# (20260815-231750 and -232454, both @ 6fa72c640), at full precision
# rather than the referee's 3-decimal print:
#   furnace-050  0.982430 / 0.982430  -> spread 0.000000
#   furnace-073  1.112564 / 1.112792  -> spread 0.000228 (3x = 0.00069)
# 3x the spread is below the resolution the referee prints, so both
# tolerances are floored at 3x that resolution (3 x 0.001). The old
# blanket 0.15 is 50x looser; note what that does NOT buy, though —
# furnace-050 reads 0.982 under claude_nee 0 AND 1, so no tolerance
# makes this room able to see the NEE bias. Tightening it stops it
# CERTIFYING a broken estimator, which is what it did twice.
#
# -073 carries a caveat: its patch is 100% CLIPPED (the ACES shoulder
# saturates at rho 0.73), so its ratio is a saturation reading, not a
# transport one — reproducible to 0.0002 but not physically meaningful
# to that precision. Its own history moved with builds (0.733 fp16,
# 1.068-1.087 mid-day, 1.113 today) and it is a truncation detector.
FURNACE_PINNED = {"050": 0.982, "073": 1.113}
FURNACE_TOL = {"050": 0.003, "073": 0.003}
# Legacy name kept so old run.json rows still parse.
FURNACE_RATIO_TOL = 0.15

# CORNELL. Every region mean within 1% of the photo golden. Chosen as
# the tolerance the PHOTO arm holds at CI settle depth (measured: the
# nee-sweep photo arm sits at 0.999-1.010 of a golden four times deeper),
# and it is deliberately tight enough to fail the estimator arm: nee 1
# reads +2% to +9% by region and is RED today BY DESIGN. That red is
# roadmap step 1a's gate.
CORNELL_RATIO_TOL = 0.010
# still_frames at the shutter below which a capture is not a measurement.
CONVERGED_MIN = 100

THUMB_MAX_PX = 460         # css width cap on gallery thumbnails (no resizing)

# --- planted defects, for `calibrate` ------------------------------------
# A referee that cannot fail a broken frame is not evidence. Each defect
# is ONE dial, chosen so a correct referee must reject it:
#
#  bounces1  claude_bounces = 1 — direct light only. Every indirect
#            term in the room disappears; the furnace's L = Le/(1-rho)
#            collapses to Le, Cornell loses all bleed.
#  clay      claude_view = 6 — the photo path with every reflectance
#            clamped to CLAY_RHO = 0.5. TRANSPORT IS BROKEN AND THE
#            ENERGY IS NEARLY UNCHANGED IN THE 050 ROOM: that room's own
#            rho is (186/255)^2.2 = 0.494, so its analytic answer moves
#            ~1%, while the 073 room (rho 0.727) and Cornell's colored
#            walls move enormously. It is the energy-neutral transport
#            defect the furnace is expected to be blind to — and it is a
#            real landmine, not a hypothetical: a mid-run /dial view 6
#            silently turned two cozy arms into uniform-albedo renders on
#            2026-08-15.
PLANTED_DEFECTS = [
    {"name": "bounces1", "dials": {"claude_bounces": 1},
     "note": "direct light only — no indirect term anywhere"},
    {"name": "clay", "dials": {"claude_view": 6},
     "note": "uniform albedo 0.5 — transport broken, furnace-050 energy "
             "within ~1% by coincidence of its own rho"},
]


# ---------------------------------------------------------------- helpers

# One run owns the seat. claude_lab's bridge lock is per RPC, so two
# claude_ci processes do NOT collide on a call — they politely take
# turns teleporting the player out of each other's settle, kill each
# other's client with stop_seat, and produce two run dirs full of
# frames neither of them can account for. Demonstrated 2026-08-15 by
# launching a second run by accident (a shell heredoc swallowed the
# launch line): both processes were alive on one seat for four minutes.
# A run-level lock is the missing half of the single-slot channel.
RUN_LOCK = os.path.join(REPO, "screenshots", "ci", ".run.lock")
RUN_LOCK_STALE = 3600.0    # s; a crashed run must not wedge the seat


def run_lock_acquire():
    try:
        os.makedirs(os.path.dirname(RUN_LOCK), exist_ok=True)
        fd = os.open(RUN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, ("%d %f\n" % (os.getpid(), time.time())).encode())
        os.close(fd)
        return None
    except FileExistsError:
        try:
            age = time.time() - os.path.getmtime(RUN_LOCK)
            holder = open(RUN_LOCK).read().strip()
        except Exception:
            age, holder = 0.0, "?"
        if age > RUN_LOCK_STALE:
            os.unlink(RUN_LOCK)
            return run_lock_acquire()
        return ("another claude_ci run owns this seat: %s, %.0fs old. Two "
                "runs on one seat interleave teleports and kill each "
                "other's client — wait for it, or remove %s if you are "
                "sure it is dead." % (holder, age, RUN_LOCK))


def run_lock_release():
    try:
        os.unlink(RUN_LOCK)
    except OSError:
        pass


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, **kw)


def git_state():
    def g(*a):
        return sh(["git"] + list(a)).stdout.strip()
    dirty = bool(g("status", "--porcelain"))
    sha = g("rev-parse", "--short", "HEAD")
    return {"sha": sha, "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": dirty, "tag": sha + ("-dirty" if dirty else "")}


def build_type():
    """CMAKE_BUILD_TYPE as the configured build tree states it, or None.
    Copied from claude_nee_sweep: the build COMMAND and the build CACHE
    are different claims, and only the cache says what ./bin/luanti is."""
    try:
        for line in open(CMAKE_CACHE):
            if line.startswith("CMAKE_BUILD_TYPE:"):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return None


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


def shader_compile_failures():
    """environment-laws: shader compile failure is SILENT — the engine
    falls back to raster and keeps reporting stale stats. debug.txt is
    the only honest signal, so read it before believing any frame."""
    try:
        with open(lab.DEBUG, errors="replace") as f:
            return [l.strip() for l in f if "Failed to compile" in l][-5:]
    except Exception:
        return []


def do_freeze():
    """LAB RULE #1: stop time and prove the accumulator deepens. A moving
    sun resets the accumulator every frame, so the "converged" image is
    frame 1 repeated forever. Recorded AND asserted."""
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


def check_room_integrity():
    """One dug node turns a referee room into a lamp. Ask the world, all
    three rooms, every node — not one spot check."""
    out = []
    for room in rooms.ROOMS:
        try:
            c = rooms.check(room)
        except Exception as e:
            c = {"room": room["name"], "error": str(e), "off_spec": [None],
                 "hash": None}
        c.pop("counts", None)
        out.append(c)
    return out


# ---------------------------------------------------------------- capture

RESET_POLL_TIMEOUT = 15.0     # s to wait for still_frames to drop
RESET_POLL_INTERVAL = 0.5
RESET_STILL_FRAMES_MAX = 3
SHOT_TIMEOUT = 25.0
SHOT_POLL_INTERVAL = 0.4
SHOT_SETTLE_INTERVAL = 0.25


def reset_accumulation(vantage, park):
    """Reset still_frames by teleporting AWAY and back, and PROVE it.

    Two arms of the same room (cornell / cornell-nee1) sit at the same
    position, and game.cpp only resets the accumulator when the camera
    moves — so without this the estimator arm would settle on top of the
    photo arm's history and every number would be a blend of the two.
    Turning in place does NOT reset (measured 2026-08-15: rpc tp yaw sets
    the PLAYER's yaw, the client owns its camera, still_frames climbed
    straight through it). Returning to the vantage's stored REST position
    lands the body where it already rests, so nothing slides.
    """
    st0 = lab.read_stats() or {}
    before = st0.get("still_frames")
    try:
        lab.goto(park)
        time.sleep(0.5)
        lab.goto(vantage)
    except Exception as e:
        return "reset rpc failed: %s" % e
    deadline = time.time() + RESET_POLL_TIMEOUT
    last = None
    while time.time() < deadline:
        st = lab.read_stats()
        last = st.get("still_frames") if st else None
        if last is not None:
            if last <= RESET_STILL_FRAMES_MAX:
                return None
            if before is not None and last < before:
                return None
        time.sleep(RESET_POLL_INTERVAL)
    return ("still_frames never dropped after reset (was %s, last seen %s)"
            % (before, last))


# The traced present path stamps a 12x12 pure-green square in the
# BOTTOM-LEFT of every frame it draws (client/shaders/claude_present,
# `gl_FragCoord.x < 12 && gl_FragCoord.y < 12`). The raster
# pass-through cannot draw it. It is therefore the one proof of "the
# tracer produced these pixels" that lives IN the pixels rather than in
# a stats file — and it answers, per capture and after the fact, the
# question a human asks from the screen ("is ray tracing off?").
TRACE_MARKER_PX = 12


def trace_marker(png):
    """(ok, detail) — did the traced present path draw this frame?"""
    try:
        from PIL import Image
        import numpy as np
        im = np.asarray(Image.open(png).convert("RGB"))
        h = im.shape[0]
        q = im[h - TRACE_MARKER_PX:h, 0:TRACE_MARKER_PX].reshape(-1, 3)
        mean = q.mean(axis=0)
        ok = bool((q[:, 0] < 8).all() and (q[:, 1] > 247).all()
                  and (q[:, 2] < 8).all())
        return ok, ("bottom-left %dpx marker mean RGB %s%s"
                    % (TRACE_MARKER_PX, mean.round(1),
                       "" if ok else " — NOT the traced present path: this "
                       "frame is the raster pass-through, or the volume was "
                       "empty"))
    except Exception as e:
        return False, "marker check failed: %s" % e


def await_complete(path):
    """Return path once the PNG is fully written and decodable, else None.

    The file APPEARING is not the file being WRITTEN: a 1080p PNG is
    ~3 MB written synchronously from the render thread, and 7 of 10
    captures in the first sweep pilot were read mid-write. Wait for the
    size to stop moving, then make PIL prove it can decode the whole
    thing."""
    deadline = time.time() + SHOT_TIMEOUT
    last = -1
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        if size > 0 and size == last:
            try:
                from PIL import Image
                with Image.open(path) as im:
                    im.load()
                return path
            except Exception:
                pass
        last = size
        time.sleep(SHOT_SETTLE_INTERVAL)
    return None


def push_dials(dials, marker):
    """Write the whole dial block plus a unique per-capture marker.

    The marker matters: claudeApplyPatchFile skips a patch file whose
    bytes are unchanged, so pushing the same dials twice logs nothing the
    second time and the capture has no proof of its own dial state. The
    marker makes every block unique, so the client re-applies and re-logs
    all of it, timestamped at this capture."""
    kv = dict(dials)
    kv["claude_ci_capture"] = marker
    lab.doorway(**kv)
    return kv


def dial_state(png, expect, marker):
    """What the CLIENT actually had at this shutter, from its own
    [claude_settings_patch] log — not from what we asked for. The /dial
    file is a second channel applied AFTER ours in the same tick, so it
    WINS, and a mid-run /dial view 6 silently invalidated two arms on
    2026-08-15."""
    out = {"seen": {}, "marker_logged": False, "ok": None}
    try:
        rec = json.load(open(os.path.splitext(png)[0] + ".capture.json"))
    except Exception as e:
        out["error"] = "no capture record: %s" % e
        out["ok"] = False
        return out
    for line in rec.get("patch_log", []):
        m = re.search(r"\[claude_settings_patch\]\s+(\S+)\s+=\s+(\S+)", line)
        if not m:
            continue
        out["seen"][m.group(1)] = m.group(2)     # last write wins == state
        if m.group(1) == "claude_ci_capture" and m.group(2) == marker:
            out["marker_logged"] = True
    bad = []
    for k in PROVEN_DIALS:
        want, got = str(expect[k]), out["seen"].get(k)
        if got is None:
            bad.append("%s never logged" % k)
        elif got != want:
            bad.append("%s = %s, expected %s" % (k, got, want))
    if not out["marker_logged"]:
        bad.append("this capture's dial block was never applied "
                   "(marker %s absent from the log)" % marker)
    out["ok"] = not bad
    if bad:
        out["error"] = "; ".join(bad)
    return out


VOLUME_TIMEOUT = 25.0      # s to wait for a snapshot to become valid
VOLUME_POLL = 1.0
# How far the camera may sit from the vantage at the shutter. Turning
# does NOT reset the accumulator (measured), so one stray mouse-look
# inside a 60 s settle blends two views into a frame that looks
# converged and carries a perfectly clean dial state — 2026-08-15, a
# hand on the mouse mid-run. The aim is read again just before the
# shutter and compared with the vantage; drift is RED.
AIM_TOL_DEG = 1.0
AIM_TOL_NODES = 0.05
# The stored vantage y is where the teleport AIMS; the body then rests
# on the floor surface half a node lower, which is not drift.
AIM_REST_DROP = 0.6


def read_aim():
    try:
        return lab.rpc("aim")
    except Exception as e:
        return {"error": str(e)}


def _dang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def aim_ok(start, now, vantage):
    """(ok, detail). Two claims: the camera points where the VANTAGE
    says, and it has not moved between the teleport and the shutter.
    The second is the one that catches a hand on the mouse mid-settle,
    which does not reset the accumulator and so leaves no other trace."""
    if not start or start.get("error") or not now or now.get("error"):
        return False, "aim rpc failed: %s" % ((start or {}).get("error")
                                              or (now or {}).get("error"))
    dyaw = _dang(now["yaw"], vantage["yaw"])
    dpitch = abs(now["pitch"] - vantage["pitch"])
    p, q = now["pos"], start["pos"]
    dmove = max(abs(p[k] - q[k]) for k in "xyz")
    dturn = max(_dang(now["yaw"], start["yaw"]),
                abs(now["pitch"] - start["pitch"]))
    dpos = max(abs(p["x"] - vantage["pos"][0]), abs(p["z"] - vantage["pos"][2]))
    ok = (dyaw <= AIM_TOL_DEG and dpitch <= AIM_TOL_DEG
          and dpos <= AIM_TOL_NODES and dmove <= AIM_TOL_NODES
          and dturn <= AIM_TOL_DEG
          and -AIM_REST_DROP <= p["y"] - vantage["pos"][1] <= AIM_TOL_NODES)
    return ok, ("yaw %.2f/%.2f pitch %.2f/%.2f pos (%.2f,%.2f,%.2f) | "
                "vs vantage: %.2f deg, %.2f deg, %.2f nodes | moved since "
                "teleport: %.3f nodes, %.2f deg"
                % (now["yaw"], vantage["yaw"], now["pitch"], vantage["pitch"],
                   p["x"], p["y"], p["z"], dyaw, dpitch, dpos, dmove, dturn))


def await_volume(marker, block, tries=3):
    """Prove the tracer has something to trace BEFORE the settle starts.

    claude_volume_follow = 0 freezes the bubble, which is what a
    measurement wants — but it also means NOTHING bootstraps a volume:
    an idle seat sits at volume_valid = 0 with the traced pipeline on,
    marching an empty bubble, and the frame looks like the tracer is
    off (John, from the screen, 2026-08-15 — and he was right). Each
    capture therefore triggers its own snapshot after the teleport, and
    this waits for the client to say it took. A snapshot fired before
    the map arrived at the new vantage snaps an EMPTY bubble, so
    area_total is checked too, not just validity.
    """
    for attempt in range(tries):
        deadline = time.time() + VOLUME_TIMEOUT
        while time.time() < deadline:
            st = lab.read_stats() or {}
            if st.get("volume_valid") == 1 and (st.get("area_total") or 0) > 0:
                return {"ok": True, "attempts": attempt + 1,
                        "volume_valid": st.get("volume_valid"),
                        "area_emitters": st.get("area_emitters"),
                        "area_total": st.get("area_total"),
                        "emitters": st.get("emitters")}
            time.sleep(VOLUME_POLL)
        if attempt + 1 < tries:      # re-trigger with a fresh token
            push_dials(block, "%s_retry%d" % (marker, attempt))
    st = lab.read_stats() or {}
    return {"ok": False, "attempts": tries,
            "volume_valid": st.get("volume_valid"),
            "area_emitters": st.get("area_emitters"),
            "area_total": st.get("area_total"),
            "error": "no volume after %d snapshot requests: volume_valid=%s "
                     "area_total=%s — the tracer would be marching an empty "
                     "bubble" % (tries, st.get("volume_valid"),
                                 st.get("area_total"))}


def capture(shot, vantage, park, dials, rundir, settle):
    """dials -> park -> vantage (proven reset) -> snapshot -> re-assert
    dials -> settle -> shutter -> N read BEFORE the PNG write -> file it
    under the arm name.

    THE ARM'S DIALS GO ON BEFORE THE ACCUMULATOR IS RESET, and that
    ordering is not cosmetic. The dial channel is a ~1 Hz poll, so an
    arm that pushes its dials AFTER the reset renders its first second
    or two of frames with the PREVIOUS ARM'S dials — and the accumulator
    is a TRUE 1/N running average with no floor (game.cpp: "no floor...
    a parked camera must actually converge"), so those M stale frames
    survive to the shutter as exactly M/N of the image. They never decay
    out.

    MEASURED 2026-08-16, roadmap 1a: a claude_view 9 frame captured
    directly after a claude_view 0 frame carried 4.26% of that photo
    frame — identical to 3 significant figures in all three channels and
    in both halves of the ceiling box, where the direct-light view's own
    answer is zero. It read as "the aimed estimator manufactures direct
    light on a coplanar surface". It was this.
    """
    name = shot["name"]
    info = {}
    info["dials_preapplied"] = push_dials(
            dials, "%s_pre_%d" % (name, time.time_ns()))
    info["reset_error"] = reset_accumulation(vantage, park)
    info["aim_at_start"] = read_aim()
    # with claude_volume_follow = 0 the bubble never re-centres on its
    # own, so take one snapshot here — before the settle, since it
    # clamps still_frames — and re-assert the dials after it.
    marker = "%s_%d" % (name, time.time_ns())
    block = dict(dials, claude_volume_snapshot=marker)
    info["dials_pushed"] = push_dials(block, marker)
    # the settle clock starts only once the volume is proven present:
    # a snapshot also clamps still_frames, so waiting here costs nothing
    # and a capture over an empty bubble costs everything.
    info["volume"] = await_volume(marker, block)
    time.sleep(settle)

    # still_frames BEFORE waiting on the ~3 MB PNG write (measured.md
    # "Owed to the harness"): the record written after the write is 1-3
    # frames late. This one is up to 1 s stale in the other direction —
    # the stats file is rewritten once per second — so it UNDER-reports,
    # which is the safe side for a convergence floor.
    ok, detail = aim_ok(info.get("aim_at_start"), read_aim(), vantage)
    info["aim_at_shutter"] = {"ok": ok, "detail": detail}
    before = lab.newest_shot()
    st = lab.read_stats() or {}
    info["still_frames_at_shutter"] = st.get("still_frames")
    info["stats_at_shutter"] = {k: st.get(k) for k in
                                ("volume_valid", "area_emitters", "area_total",
                                 "emitters", "frame_ms_avg", "busy_ms",
                                 "pass_ms", "accum_alpha")}
    with open(lab.PATCH, "w") as f:
        f.write("claude_screenshot = %s\n" % marker)
    png = None
    deadline = time.time() + SHOT_TIMEOUT
    while time.time() < deadline:
        cur = lab.newest_shot()
        if cur and cur != before:
            png = await_complete(cur)
            break
        time.sleep(SHOT_POLL_INTERVAL)
    if not png:
        raise RuntimeError("no complete screenshot appeared for %s" % name)
    lab.write_capture_record(png)

    dst = os.path.join(rundir, name + ".png")
    shutil.copy2(png, dst)
    rec = os.path.splitext(png)[0] + ".capture.json"
    cap = {}
    if os.path.exists(rec):
        shutil.copy2(rec, os.path.join(rundir, name + ".capture.json"))
        cap = json.load(open(rec))
    st2 = cap.get("stats") or {}
    info.update({"frozen": cap.get("frozen"), "time_speed": cap.get("time_speed"),
                 "sha": cap.get("sha"), "dirty": cap.get("dirty"),
                 "still_frames": st2.get("still_frames"),
                 "accum_alpha": st2.get("accum_alpha"), "fps": st2.get("fps"),
                 "cap_artifact": cap.get("cap_artifact")})
    info["dial_state"] = dial_state(png, dials, marker)
    return dst, info


def park_for(vantage_name, vantages):
    """Somewhere else the player can STAND. Landing anywhere but a rest
    position starts a physics slide that resets accumulation for up to a
    minute (the cozy-ci landmine)."""
    alt = "furnace-050" if vantage_name != "furnace-050" else "cornell"
    return vantages[alt]


# ---------------------------------------------------------------- referees

def run_referee(kind, arg, png, rundir, name, golden_png=None):
    script = os.path.join(HERE, "claude_%s_check.py" % kind)
    cmd = [sys.executable, script, png] + ([arg] if arg else [])
    if kind == "furnace" and FURNACE_PATCH:
        cmd += ["--patch"] + [str(v) for v in FURNACE_PATCH]
    if kind == "cornell":
        cmd += ["--regions"]
        if golden_png:
            cmd += ["--ratio", golden_png]
    r = sh(cmd)
    text = (r.stdout or "") + (r.stderr or "")
    with open(os.path.join(rundir, name + ".referee.txt"), "w") as f:
        f.write("$ %s\n\n%s\n[exit %d]\n" % (" ".join(cmd), text, r.returncode))
    out = {"kind": kind, "arg": arg, "returncode": r.returncode,
           "txt": name + ".referee.txt", "stdout": text,
           "golden_png": golden_png}
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
    means, purity = {}, {}
    for name, val, pur in re.findall(
            r"^region (\S+)\s+mean ([-\d.]+).*purity ([-\d.]+)", t, re.M):
        means[name] = float(val)
        purity[name] = float(pur)
    if means:
        out["region_means"] = means
        out["region_purity"] = purity
    ratios = {}
    for name, val in re.findall(r"^ratio (\S+)\s+([-\d.]+)\s+\(this", t, re.M):
        ratios[name] = float(val)
    if ratios:
        out["region_ratios"] = ratios
        out["region_worst"] = max(abs(v - 1.0) for v in ratios.values())
    return out


def furnace_verdict(ref, variant):
    """PASS/FAIL against the room's own pinned ratio, not a blanket 15%."""
    if not ref or ref.get("returncode") != 0:
        return "-", "referee could not speak"
    r = ref.get("ratio_analytic") or {}
    if len(r) != 3:
        return "-", "unparsed"
    pin, tol = FURNACE_PINNED[variant], FURNACE_TOL[variant]
    worst = max(r.values(), key=lambda v: abs(v - pin))
    ok = all(abs(v - pin) <= tol for v in r.values())
    return ("PASS" if ok else "FAIL"), "ratio %.3f (pinned %.3f +/- %.3f)" % (
        worst, pin, tol)


def cornell_verdict(ref):
    """PASS/FAIL on the five region ratios vs the pinned photo golden.
    This is the measurement that convicted NEE — the whole point of the
    step. Without a golden it cannot speak, which is not a pass."""
    if not ref or ref.get("returncode") != 0:
        return "-", "referee could not speak"
    ratios = ref.get("region_ratios")
    if not ratios:
        return "-", "no golden pinned (region ratios unmeasurable)"
    worst_name = max(ratios, key=lambda k: abs(ratios[k] - 1.0))
    worst = ratios[worst_name]
    ok = abs(worst - 1.0) <= CORNELL_RATIO_TOL
    return ("PASS" if ok else "FAIL"), "%s %.4f (tol %.3f)" % (
        worst_name, worst, CORNELL_RATIO_TOL)


def verdict(shot_def, ref):
    """(mark, key number). '-' whenever the referee could not speak, and
    '-' counts as RED: a referee that cannot speak is not a pass."""
    if not shot_def.get("referee"):
        return "-", "no referee (eyes only)"
    kind, arg = shot_def["referee"]
    if not ref:
        return "-", "no referee output"
    return furnace_verdict(ref, arg) if kind == "furnace" else cornell_verdict(ref)


# ---------------------------------------------------------------- assertions

class Assertions:
    """Named claims, each true or false, with the number that decides it.
    The run's exit code is nothing more than 'did any of these fail'."""

    def __init__(self):
        self.items = []

    def add(self, name, ok, detail):
        self.items.append({"name": name, "ok": bool(ok), "detail": detail})
        print("  %-26s %-5s %s" % (name, "ok" if ok else "FAIL", detail))
        return ok

    @property
    def failed(self):
        return [a for a in self.items if not a["ok"]]

    def summary(self):
        if not self.items:
            return "RED: no assertions ran"
        if not self.failed:
            return "GREEN: %d/%d assertions pass" % (len(self.items), len(self.items))
        return "RED: %d failing — %s" % (
            len(self.failed), ", ".join(a["name"] for a in self.failed))


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


def golden_png():
    """The pinned golden run's PHOTO Cornell frame, or None."""
    rid = read_golden()
    if not rid:
        return None
    p = os.path.join(CI_DIR, rid, GOLDEN_REF_SHOT + ".png")
    return p if os.path.exists(p) else None


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
.green{background:#0f3d20;color:#6ee787;border:1px solid #1e6b39;padding:2px 8px;
border-radius:3px;font-size:12px;font-weight:700}
.red{background:#4a0f14;color:#ff8a8a;border:1px solid #8a1c25;padding:2px 8px;
border-radius:3px;font-size:12px;font-weight:700}
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


def _rg(run):
    v = run.get("verdict") or ""
    if v.startswith("GREEN"):
        return '<span class=green>GREEN</span>'
    if v.startswith("RED"):
        return '<span class=red>%s</span>' % html.escape(v)
    return ""


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
    parts.append("<h1>%s %s <span class=meta>%s</span></h1>"
                 % (e(run["tag"]), _rg(run), e(run["branch"])))
    parts.append('<div class=meta>%s &middot; settle %gs &middot; build %s'
                 ' &middot; freeze: %s &middot; prev %s &middot; golden %s'
                 ' &middot; <a href="../index.html">all runs</a></div>'
                 % (e(run["started_utc"]), run["settle"],
                    e(str(run.get("build_type"))),
                    "deepening" if (run.get("freeze") or {}).get("deepening")
                    else "NOT DEEPENING", e(run.get("prev_run") or "none"),
                    e(run.get("golden_run") or "none pinned")))
    rows = "".join("<tr><td>%s</td><td class=%s>%s</td><td>%s</td></tr>"
                   % (e(a["name"]), "vPASS" if a["ok"] else "vFAIL",
                      "ok" if a["ok"] else "FAIL", e(str(a["detail"])))
                   for a in run.get("assertions", []))
    parts.append("<h2>assertions</h2><pre>%s</pre>"
                 % e("\n".join("%-26s %-5s %s"
                               % (a["name"], "ok" if a["ok"] else "FAIL",
                                  a["detail"])
                               for a in run.get("assertions", []))))
    for shot_def in CI_SHOTS:
        name = shot_def["name"]
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
                 "against it AND against the run above it; the Cornell region "
                 "ratios are measured against its photo arm.</div>"
                 % e(gold or "none (claude_ci.py golden &lt;run-dir&gt;)"))
    for rid, run in load_runs():
        d = ' <span class=dirty>DIRTY</span>' if run.get("dirty") else ""
        if rid == gold:
            d += ' <span class=warn>GOLDEN</span>'
        parts.append('<div class=run><h2><a href="%s/index.html">%s</a>%s %s '
                     '<span class=meta>%s &middot; %s</span></h2><div class=cells>'
                     % (e(rid), e(run.get("tag", rid)), d, _rg(run),
                        e(run.get("branch", "?")), e(run.get("started_utc", "?"))))
        for shot_def in CI_SHOTS:
            name = shot_def["name"]
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


# ---------------------------------------------------------------- the seat

def bring_up_seat(rundir, run, args):
    """Build (Release), refuse a Debug cache, start the seat, freeze time.
    Returns an error string, or None when the seat is ready to measure."""
    if args.skip_build:
        run["build"] = {"ran": False}
    else:
        print("building: %s (CMAKE_BUILD_TYPE=Release)" % " ".join(BUILD_CMD))
        env = dict(os.environ, **BUILD_ENV)
        r = subprocess.run(BUILD_CMD, cwd=REPO, env=env)
        run["build"] = {"ran": True, "returncode": r.returncode,
                        "env": BUILD_ENV}
        if r.returncode != 0:
            return "build failed (%d)" % r.returncode

    bt = build_type()
    run["build_type"] = bt
    if not args.allow_debug and (bt or "").lower() not in RELEASE_TYPES:
        return ("build tree is CMAKE_BUILD_TYPE=%s. environment-laws: no "
                "measurement from a Debug binary (Release is ~4.5x faster; "
                "util/ci/build.sh defaults to Debug). Rebuild with "
                "CMAKE_BUILD_TYPE=Release, or pass --allow-debug to record "
                "a scratch run." % bt)

    print("seat: stopping any running luanti...")
    run["seat_clean_stop"] = stop_seat()
    pin_conf()
    run["pinned_conf"] = PINNED_CONF
    print("seat: starting server + client on port %d" % SEAT_PORT)
    start_seat(rundir)
    if not wait_for_client():
        return "client never became ready after %.0fs" % CLIENT_READY_TIMEOUT
    print("seat: client ready")
    run["shader_failures"] = shader_compile_failures()
    run["freeze"] = do_freeze()
    if not run["freeze"].get("deepening"):
        print("!! FREEZE FAILED: the accumulator is not deepening.")
    return None


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

    base = dict(CANONICAL_DIALS)
    base["claude_nee"] = args.nee
    run = dict(git, run_id=run_id, settle=args.settle,
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               dials=base, shots={}, assertions=[])
    A = Assertions()

    err = bring_up_seat(rundir, run, args)
    if err:
        run["aborted"] = err
        A.add("seat", False, err)
        return finish(run, rundir, A)

    A.add("build-release", (run.get("build_type") or "").lower() in RELEASE_TYPES,
          "CMAKE_BUILD_TYPE=%s" % run.get("build_type"))
    A.add("shaders-compile", not run["shader_failures"],
          "%d 'Failed to compile' lines in debug.txt"
          % len(run["shader_failures"]))
    A.add("freeze-deepening", run["freeze"].get("deepening"),
          "still_frames %s, time_speed %s" % (run["freeze"].get("still_frames"),
                                              run["freeze"].get("time_speed")))

    vs = lab.load_vantages()
    run["doors_shut"] = set_doors(True)
    run["room_integrity"] = check_room_integrity()
    for item in run["room_integrity"]:
        bad = item.get("off_spec") or []
        A.add("room-%s-onspec" % item["room"], not bad,
              item.get("error") or "hash %s, %d off-spec node(s)%s"
              % (item.get("hash"), len(bad),
                 "" if not bad else ": " + ", ".join(
                     "(%d,%d,%d) want %s got %s"
                     % (b["pos"][0], b["pos"][1], b["pos"][2], b["want"],
                        b["got"]) for b in bad[:6])))
    gold_png = golden_png()
    try:
        prev, gold = previous_run(run_id), read_golden()
        run["prev_run"], run["golden_run"] = prev, gold
        run["golden_png"] = gold_png
        for shot_def in CI_SHOTS:
            name, vname = shot_def["name"], shot_def["vantage"]
            if vname not in vs:
                A.add("%s-capture" % name, False,
                      "%s MISSING from claude_vantages.json" % vname)
                continue
            dials = dict(base, **shot_def.get("dials", {}))
            try:
                png, cap = capture(shot_def, vs[vname], park_for(vname, vs),
                                   dials, rundir, args.settle)
            except Exception as ex:
                # One dead vantage must not cost the other four.
                run["shots"][name] = {"error": str(ex)}
                A.add("%s-capture" % name, False, str(ex))
                continue
            shot = {"png": os.path.basename(png), "capture": cap}
            if shot_def["referee"]:
                kind, arg = shot_def["referee"]
                shot["referee"] = run_referee(
                    kind, arg, png, rundir, name,
                    gold_png if kind == "cornell" else None)
            shot["verdict"] = list(verdict(shot_def, shot.get("referee")))
            shot["rms_vs_prev"] = diff_against(prev, name, png)
            shot["rms_vs_golden"] = diff_against(gold, name, png)
            run["shots"][name] = shot

            ds = cap.get("dial_state") or {}
            A.add("%s-dials" % name, ds.get("ok"),
                  ds.get("error") or " ".join(
                      "%s=%s" % (k.replace("claude_", ""),
                                 ds.get("seen", {}).get(k))
                      for k in PROVEN_DIALS))
            ok, detail = trace_marker(png)
            A.add("%s-traced" % name, ok, detail)
            aim = cap.get("aim_at_shutter") or {}
            A.add("%s-aim" % name, aim.get("ok"), aim.get("detail"))
            vol = cap.get("volume") or {}
            A.add("%s-volume" % name, vol.get("ok"),
                  vol.get("error") or "volume_valid=%s area_emitters=%s/%s "
                  "(snapshot attempt %s)"
                  % (vol.get("volume_valid"), vol.get("area_emitters"),
                     vol.get("area_total"), vol.get("attempts")))
            st = (cap.get("stats_at_shutter") or {})
            sf = cap.get("still_frames_at_shutter")
            A.add("%s-converged" % name, (sf or 0) >= CONVERGED_MIN,
                  "still_frames at shutter %s (min %d)" % (sf, CONVERGED_MIN))
            if cap.get("reset_error"):
                A.add("%s-accum-reset" % name, False, cap["reset_error"])
            if shot_def["referee"]:
                A.add("%s-%s" % (name, shot_def["referee"][0]),
                      shot["verdict"][0] == "PASS", shot["verdict"][1])
    finally:
        run["doors_reopened"] = set_doors(False)
    return finish(run, rundir, A)


def finish(run, rundir, A):
    run["assertions"] = A.items
    run["verdict"] = A.summary()
    run["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    disk = json.loads(json.dumps(run))
    for s in disk.get("shots", {}).values():
        if s.get("referee"):
            s["referee"].pop("stdout", None)
    json.dump(disk, open(os.path.join(rundir, "run.json"), "w"), indent=2)
    write_run_html(rundir, run)
    master = write_master_html()
    marks = " ".join("%s=%s" % (d["name"],
                                run.get("shots", {}).get(d["name"], {})
                                .get("verdict", ["-"])[0])
                     for d in CI_SHOTS)
    print("%s %s@%s  %s" % ("DIRTY" if run["dirty"] else "clean",
                            run["branch"], run["sha"], marks))
    print(master)
    print(run["verdict"])
    return 0 if not A.failed else 1


# ------------------------------------------------------------- calibrate

def cmd_calibrate(args):
    """Plant a defect; every referee must FAIL. Physics-contract §8
    clause 3: naming a referee is not enough — one that cannot fail the
    change is not evidence. Any referee that PASSES a planted defect is
    reported BLIND by name, and this command exits nonzero."""
    git = git_state()
    run_id = "%s_%s_calibrate" % (time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
                                  git["tag"])
    rundir = os.path.join(CI_DIR, run_id)
    os.makedirs(rundir, exist_ok=True)
    print("run dir: %s" % rundir)
    base = dict(CANONICAL_DIALS)
    base["claude_nee"] = args.nee
    run = dict(git, run_id=run_id, settle=args.settle, calibrate=True,
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               dials=base, shots={}, defects=PLANTED_DEFECTS, results=[])

    err = bring_up_seat(rundir, run, args)
    if err:
        run["aborted"] = err
        print("ABORTED: %s" % err)
        json.dump(run, open(os.path.join(rundir, "run.json"), "w"), indent=2)
        return 1

    vs = lab.load_vantages()
    gold_png = golden_png()
    run["golden_png"] = gold_png
    set_doors(True)
    blind = []
    try:
        for defect in PLANTED_DEFECTS:
            for shot_def in CI_SHOTS:
                if not shot_def["referee"] or shot_def["name"] == "cornell-nee1":
                    continue      # referee vantages only, photo arm only
                name = "%s_%s" % (shot_def["name"], defect["name"])
                vname = shot_def["vantage"]
                dials = dict(base, **shot_def.get("dials", {}))
                dials.update(defect["dials"])
                print("\n=== %s: %s" % (name, defect["dials"]))
                try:
                    png, cap = capture(dict(shot_def, name=name), vs[vname],
                                       park_for(vname, vs), dials, rundir,
                                       args.settle)
                except Exception as ex:
                    print("CAPTURE FAILED: %s" % ex)
                    run["results"].append({"defect": defect["name"],
                                           "arm": shot_def["name"],
                                           "error": str(ex)})
                    continue
                kind, arg = shot_def["referee"]
                ref = run_referee(kind, arg, png, rundir, name,
                                  gold_png if kind == "cornell" else None)
                ds = cap.get("dial_state") or {}
                # every referee that speaks about this frame, by name
                seen = []
                if kind == "furnace":
                    v = furnace_verdict(ref, arg)
                    seen.append(("furnace-analytic/%s" % arg, v))
                else:
                    seen.append(("cornell-regions", cornell_verdict(ref)))
                    bl = ref.get("bleed_verdict")
                    seen.append(("cornell-legacy-bleed",
                                 (("PASS" if (bl or "").startswith(
                                     "COLOR BLEED PRESENT") else "FAIL"),
                                  bl or "unparsed")))
                for rname, (mark, detail) in seen:
                    caught = mark == "FAIL"
                    print("  %-28s %-9s %s"
                          % (rname, "CAUGHT" if caught else "BLIND", detail))
                    run["results"].append(
                        {"defect": defect["name"], "arm": shot_def["name"],
                         "referee": rname, "mark": mark, "detail": detail,
                         "caught": caught, "png": os.path.basename(png),
                         "dials_ok": ds.get("ok"),
                         "dials_error": ds.get("error")})
                    if not caught:
                        blind.append("%s vs %s (%s)"
                                     % (rname, defect["name"], detail))
    finally:
        set_doors(False)

    run["blind"] = blind
    run["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(run, open(os.path.join(rundir, "run.json"), "w"), indent=2)
    print("\n" + "=" * 70)
    if blind:
        print("BLIND REFEREES (%d) — each passed a planted defect:" % len(blind))
        for b in blind:
            print("  %s" % b)
        print("Write what each is blind to into spec/measured.md "
              "(physics-contract §8 clause 3).")
    else:
        print("ALL REFEREES CAUGHT EVERY PLANTED DEFECT.")
    print("=" * 70)
    return 1 if blind else 0


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
    if (run.get("build_type") or "").lower() not in RELEASE_TYPES:
        print("REFUSING: %s was captured against a %s build tree."
              % (rid, run.get("build_type")))
        return 1
    if (run.get("dials") or {}).get("claude_nee") not in (0, "0"):
        print("REFUSING: %s is not a PHOTO run (claude_nee=%s). The golden "
              "is the truth mode (physics-contract §6)."
              % (rid, (run.get("dials") or {}).get("claude_nee")))
        return 1
    open(GOLDEN_FILE, "w").write(rid + "\n")
    write_master_html()
    print("golden pinned: %s (%s@%s)" % (rid, run.get("branch"), run.get("sha")))
    return 0


def cmd_regions(args):
    """Print the Cornell region boxes and, optionally, measure a frame.
    Here so the boxes can be re-checked on any frame without a seat."""
    im_w, im_h = 1920, 1080
    for name, boxes in cornell.REGIONS.items():
        for b in boxes:
            print("%-15s frac (%.3f,%.3f)-(%.3f,%.3f)  px (%d,%d)-(%d,%d)"
                  % (name, b[0], b[1], b[2], b[3], b[0] * im_w, b[1] * im_h,
                     b[2] * im_w, b[3] * im_h))
    if args.image:
        cornell.print_regions(args.image)
        if args.golden:
            cornell.print_ratios(args.image, args.golden)
    return 0


def with_run_lock(fn):
    def wrapped(args):
        held = run_lock_acquire()
        if held:
            print("REFUSING: %s" % held)
            return 1
        try:
            return fn(args)
        finally:
            run_lock_release()
    return wrapped


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--settle", type=float, default=SETTLE_DEFAULT,
                       help="seconds of stillness before each shot "
                            "(default %(default)s)")
        p.add_argument("--skip-build", action="store_true",
                       help="capture with the binaries already in ./bin")
        p.add_argument("--allow-debug", action="store_true",
                       help="capture against a Debug build tree anyway "
                            "(scratch only — environment-laws forbids it)")
        p.add_argument("--nee", type=int, default=0, choices=(0, 1),
                       help="claude_nee for the base arms: 0 = the photo "
                            "path, the truth mode (default). The cornell-nee1 "
                            "arm always runs with 1.")
        return p

    p = common(sub.add_parser("run", help="build, seat, capture, referee, judge"))
    p.set_defaults(func=with_run_lock(cmd_run))
    p = common(sub.add_parser("calibrate",
                              help="plant a defect; every referee must FAIL"))
    p.set_defaults(func=with_run_lock(cmd_calibrate))
    p = sub.add_parser("golden", help="pin/show the run all diffs measure against")
    p.add_argument("run_dir", nargs="?", help="a run dir name under screenshots/ci/")
    p.set_defaults(func=cmd_golden)
    p = sub.add_parser("regions", help="show the Cornell region boxes")
    p.add_argument("image", nargs="?")
    p.add_argument("--golden")
    p.set_defaults(func=cmd_regions)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
