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

# --- the settle: ACCUMULATED FRAMES, not wall-clock seconds -------------
#
# WAS `SETTLE_DEFAULT = 60.0` seconds, from the day the harness was
# written and never measured. Two things were wrong with a clock:
#
# 1. It does not buy a fixed depth. The same 60 s bought still_frames
#    3,928 on one run and 7,491 on another (measured, run.json), because
#    fps moves with the scene, the build and the machine. Depth is the
#    quantity the image quality actually depends on.
# 2. ANY world change clamps still_frames to 10 (game.cpp
#    claudeTraceGridSnapshot / claudeTraceGridIncremental), and Mineclonia's
#    grass ABM changes a node inside the bubble every ~30-180 s. A clock
#    interrupted in its last second fires the shutter on an accumulator
#    one frame deep and calls it a measurement -- measured.md "Gate 4,
#    second half", where cozy-day-ci came in at still_frames 11 and CI
#    went red on a tree whose only change was harness hygiene.
#
# Waiting on FRAMES is immune to both: a reset merely extends the wait.
#
# 2000 is chosen from the convergence curve in spec/measured.md "Settle
# calibration", NOT from the 0.25%-of-N=8000 rule the handoff proposed --
# that rule turned out to be unanswerable, because Cornell's `floor`
# region is still moving 0.27% between 8,000 and 16,000 frames, so the
# reference it compares against is not converged either. The number that
# decides instead is the tolerance CI actually enforces:
# CORNELL_RATIO_TOL, 1% on the worst region ratio against the golden.
# Worst observed over four independent runs: 0.99% at N=250, 1.14% at
# N=500 (a genuine FAIL), 0.91% at 1000, 0.73% at 2000, 0.56% at 4000.
# At 2000 the run-to-run sigma of that quantity is 0.36%, so 1% is 2.8
# sigma. Today's 60 s buys cornell 3,928-5,781 frames, i.e. 3.0-3.3
# sigma -- but it buys cornell-nee1 only 152-2,108 (measured across
# three runs), i.e. as little as 1.0 sigma on the SAME 1% tolerance.
# A flat 2,000 frames is therefore a large net gain in reliability, and
# the arm it slightly relaxes is the one that was never at risk.
SETTLE_FRAMES = 2000
SETTLE_SECONDS_WAS = 60.0  # what SETTLE_FRAMES replaced, 2026-08-16
# Ceiling, so an arm whose world never stops changing cannot wedge a
# run. exterior-ci is unsealed and map blocks keep arriving there, each
# one a real change that correctly caps still_frames; measured depths at
# 60 s were 162 / 191 / 1,812 / 2,849. Reaching the ceiling is NOT a new
# way to go red -- the shot is taken and the existing `-converged`
# assertion (still_frames >= CONVERGED_MIN) judges it, exactly as before.
SETTLE_MAX_S = 180.0
# ...except that firing the shutter one frame after a reset is the whole
# defect being removed, so the ceiling does not apply until the
# accumulator is at least this deep. Past SETTLE_HARD_MAX_S the shot is
# taken regardless and the run says so.
SETTLE_MIN_FRAMES = 300
SETTLE_HARD_MAX_S = 300.0
SETTLE_POLL = 0.25
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
    "claude_grid_debug": 3,
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
    # SUB-VOXEL DESCENT (2026-08-16). 1 = march() steps into a class-250
    # cell's 16^3 mask; 0 = the pre-descend behaviour, where a stair is a
    # 1 m cube. Pushed EXPLICITLY and recorded in every .capture.json
    # rather than left to its source default, because this dial decides
    # what GEOMETRY the goldens are of — the previous sub-voxel dial
    # (claude_subvox) is pinned to 0 in the seat conf and nothing reads
    # it, which is exactly the silence this line exists to avoid.
    "claude_descend": 1,
    "claude_stats": 1,
    # The grid follows the camera again (2026-08-16). This was pinned
    # to 0 from 2026-08-15 because the re-snap ran on a 3 s timer and
    # cost a visible hitch plus a silent scene change under a settling
    # accumulator. It is now event-driven and incremental (game.cpp
    # claudeTraceGridIncremental; spec/measured.md "Volume re-snap fix"), so
    # a tick with no world change costs nothing and touches nothing.
    # Pinned EXPLICITLY at 1 rather than left unset: the unset value
    # happens to be 1 too, and that is exactly the hidden-default class
    # environment-laws warns about.
    #
    # This makes CI honest about one thing it could not see before: the
    # world is NOT static. Mineclonia's grass ABM turns a covered
    # dirt_with_grass into dirt every 30-90 s, and with follow on that
    # reaches the trace and caps still_frames, as a real world change
    # should. Each capture still forces its own snapshot after the
    # teleport (see capture()).
    "claude_grid_follow": 1,
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
                "claude_grid_debug", "claude_rng")

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
               # Harness precondition, not a feature (John, 2026-08-15,
               # after a creeper detonated inside the Cornell box and
               # damaged a referee room): a hostile mob wandering into a
               # sealed referee is the same class of silent damage as a
               # dug node, and this game's mob spawner reads this ONE
               # global bool at server start (mcl_mobs/spawning.lua,
               # `core.settings:get_bool("mobs_spawn", true)`, file
               # scope — not hot-reloadable, so it must be in the conf
               # BEFORE start_seat(), not pushed live).
               "mobs_spawn": "false",
               # /warp (roadmap 1b) reads/writes util/claude_vantages.json,
               # which sits outside the world dir and outside every mod
               # dir — the sandboxed `io` refuses both (environment-laws:
               # the client REWRITES minetest.conf on exit, which is how
               # this line gets silently dropped if it is only ever
               # hand-added once instead of pinned here).
               "secure.trusted_mods": "claude_bridge",
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


# THE SAME BUG, ONE CHANNEL OVER — and this one had never been closed.
# claude_settings_patch.conf persists in path_user across seat restarts,
# and claudeApplyPatchFile() applies it on the FIRST poll of a fresh
# client, because the client's "have I applied this already" memory
# starts empty. So whatever the last session happened to leave in that
# file silently reconfigures the next seat, and minetest.conf — which
# every instrument in this repo reads — does not show it.
#
# Cost, 2026-08-16: a leftover `claude_grid_follow = 0` from a baseline
# arm pushed itself onto a fresh seat that had been started explicitly to
# measure follow ON. The conf said 1, the client ran 0, and the only
# evidence was one ACTION line in debug.txt. This is exactly the
# environment-laws "/set shadows the conf" law, arriving through a file
# instead of a chat command.
PATCH_HEADER = ("# claude_settings_patch.conf: the live client dial\n"
                "# channel, polled by pollSettingsPatch() at ~1 Hz.\n"
                "# CLEARED at seat start by claude_ci.pin_conf -- a file\n"
                "# left here by the previous session applies itself to the\n"
                "# next client's FIRST poll and shows up in no conf.\n")


def clear_dial_file():
    """Empty BOTH live dial channels so a leftover setting from another
    session cannot shadow a run's own dials. See DIAL_HEADER and
    PATCH_HEADER for what each one cost."""
    for path, header in (
            (os.path.join(REPO, SEAT_WORLD, "claude_dial.conf"), DIAL_HEADER),
            (os.path.join(REPO, "claude_settings_patch.conf"), PATCH_HEADER)):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").write(header)
        except Exception as e:
            print("WARNING: could not clear %s: %s" % (path, e))


def pin_conf():
    """Force PINNED_CONF keys into minetest.conf (idempotent), and clear
    the live dial channels. Both are seat hygiene: a key missing here is a
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

    # --- gallery phase 2 (roadmap 1b): sky/sun/glass referee rooms ------
    # Content + harness only — no shader/estimator change, so none of
    # these have an image referee yet. Every one of them still gets the
    # full generic gate (dials, traced marker, aim, grid, converged,
    # room hash) — "referee: None" means "eyes only, or perf only until
    # 2b", not "unchecked". Goldens are pinned WRONG-BY-DESIGN (dark or
    # black) until 2b lands the sky/sun terms; see spec/measured.md "1b —
    # Gallery phase 2" for which is which.
    {"name": "exterior-ci", "vantage": "exterior-ci", "referee": None},
    {"name": "skyfurnace-050", "vantage": "skyfurnace-050", "referee": None},
    {"name": "cave-skylight-noon", "vantage": "cave-skylight-noon",
     "referee": None},
    {"name": "cave-skylight-night", "vantage": "cave-skylight-night",
     "referee": None},
    {"name": "cave-glass", "vantage": "cave-glass", "referee": None},
    {"name": "cozy-night-ci", "vantage": "cozy-night-ci", "referee": None},
    {"name": "cozy-day-ci", "vantage": "cozy-day-ci", "referee": None},
    # Same physical vantage as cozy-day-ci; OPS.lamps swaps the panel to
    # its unlit twin for this capture only (cmd_run restores it in a
    # finally, even on a capture exception) — sunlight-only interior,
    # honestly black until 2b lights it through the windows.
    {"name": "cozy-day-dark-ci", "vantage": "cozy-day-ci", "referee": None,
     "lamps_off": True},
]
# The golden image every Cornell region ratio is measured against: the
# PHOTO arm of the pinned golden run.
GOLDEN_REF_SHOT = "cornell"


def shots_for(args):
    """The arms this invocation will shoot. ALL THIRTEEN unless --scored.

    Only four arms carry a pass/fail verdict -- furnace-050, furnace-073,
    cornell, cornell-nee1 -- because they are the only ones with a
    referee. The other nine are captured, diffed against their goldens
    and PUT IN THE GALLERY, but `A.add` is never called with an image
    comparison for them, so no pixel they contain can turn a run red.
    --scored shoots only the four.

    THE DEFAULT DOES NOT MOVE, and that is deliberate rather than
    cautious. John's standing rule from the settle calibration is "no arm
    dropped", and the nine unscored arms are how a human sees a
    regression the referees are blind to -- the cosy rooms are where
    every geometry change has actually shown itself. --scored is a fast
    inner loop for someone iterating on Cornell, not a cheaper CI. What
    it costs in coverage is written into DECISIONS.md with the numbers
    attached; whether it should ever be the default is not this session's
    call to make.
    """
    if getattr(args, "scored", False):
        return [d for d in CI_SHOTS if d["referee"]]
    return list(CI_SHOTS)

# Room doorways (claude_gallery_deploy layout: floor y=8, walk y=9, doors
# on the z=0 line). The referee rooms MUST be shut to be referees — an
# open door leaks the sky into an analytic furnace. Plugged before the
# vantage loop (so the last scene churn precedes every settle), reopened
# on the way out so the gallery stays walkable.
# name is ALWAYS explicit now (found 2026-08-15, roadmap 1b gate): a
# None here falls back to OPS.door reading the NEIGHBOUR node's name at
# shut time, which races map generation for a freshly-emerged region —
# caught live when cave-glass's door was sealed with "ignore" (the
# engine's not-yet-generated placeholder) instead of its wall material,
# because the neighbour cell hadn't finished generating when set_doors()
# ran early in a run. The golden this shipped from had an unsealed
# referee room and was re-cut. furnace-050/furnace-073 never showed the
# symptom (their area is long-cached), but the race is the same one
# either way, so all five doors get an explicit plug now, not just
# Cornell's (which always had one).
CI_DOORS = [{"pos": {"x": 20, "y": 9, "z": 0},
             "name": "claude_bridge:gray186_lit"},               # furnace-050
            {"pos": {"x": 33, "y": 9, "z": 0},
             "name": "claude_bridge:gray221_lit"},               # furnace-073
            {"pos": {"x": 47, "y": 9, "z": 0},
             "name": "claude_bridge:gray221"},                   # cornell
            {"pos": {"x": 14, "y": 9, "z": 85},
             "name": "claude_bridge:gray186"},                   # cave-skylight
            {"pos": {"x": 34, "y": 9, "z": 85},
             "name": "claude_bridge:gray186"}]                   # cave-glass

# Room hash (roadmap 1b): "the gate that proves rebuild determinism".
# One dug node at (47,11,8) leaked daylight through Cornell for an
# unknown period and nothing in the harness noticed (measured.md, "CI
# red/green") — it surfaced only when a deep-convergence RMS drifted the
# wrong way and someone looked at the frame. OPS.scan already returns an
# order-dependent hash of every node name in a box (its ancestor, used by
# check_room_integrity() for the three original rooms' full spec check);
# this extends the SAME primitive to every CI room, including the ones
# with no hand-written Python spec, and makes it a GATE: a capture whose
# room hash disagrees with the golden's is REFUSED, not diffed. Boxes
# include the door cell, so an unshut door changes the hash too.
ROOM_BOXES = {
    "furnace-050": ((17, 8, 0), (23, 14, 6)),
    "furnace-073": ((30, 8, 0), (36, 14, 6)),
    "cornell": ((43, 8, 0), (51, 16, 8)),
    "cozy": ((0, 8, 0), (10, 17, 8)),
    "cave-skylight": ((10, 8, 85), (18, 14, 93)),
    "cave-glass": ((30, 8, 85), (38, 14, 93)),
    "sky-furnace-050": ((47, 8, 97), (73, 9, 123)),
    # exterior-ci is "no build" (the handoff's own words) — there is no
    # structure to protect, so this is a small box around the stand
    # point rather than a meaningful integrity claim. It still gets a
    # hash in every sidecar, mechanically, because every capture does.
    "exterior-ci": ((4, 8, -26), (6, 10, -24)),
}
# vantage name -> ROOM_BOXES key, for shots whose vantage name does not
# match the room name 1:1 (cozy has four vantages over one physical room;
# the two cave vantages share cave-skylight's box; sky-furnace-050 the
# vantage vs sky-furnace-050 the pad).
VANTAGE_ROOM = {
    "cozy-ci": "cozy", "cozy-night-ci": "cozy", "cozy-day-ci": "cozy",
    "cozy-day-dark-ci": "cozy",
    "cave-skylight-noon": "cave-skylight", "cave-skylight-night": "cave-skylight",
    "cave-glass": "cave-glass",
    "skyfurnace-050": "sky-furnace-050",
    "exterior-ci": "exterior-ci",
}


def room_of(vantage_name):
    return VANTAGE_ROOM.get(vantage_name, vantage_name)


ROOM_HASH_RETRIES = 4
ROOM_HASH_RETRY_WAIT = 3.0   # s


def room_hash(room_name):
    """(hash, error) for ROOM_BOXES[room_name], via the bridge's OPS.scan.

    FOUND 2026-08-15 (roadmap 1b gate, the deliberate-break demonstration
    itself): OPS.scan reports an unloaded cell as the literal string
    "unloaded" rather than failing, so a room whose chunks have not
    finished loading yet hashes to a value that mixes real node names
    with "unloaded" placeholders -- a hash that is internally consistent
    (reproducible) but not the room's true content. Caught live: a
    --skip-deploy run refused cave-glass against the golden even though
    the room was untouched; re-scanning after the chunk finished loading
    reproduced the golden hash exactly. This is the documented
    environment-laws landmine ("async block emerge races any sampler
    that reads the world") arriving through a new door. Retrying here,
    not just waiting longer once, because the failure is silent at the
    OPS.scan layer -- there is nothing to distinguish "genuinely all air"
    from "not loaded yet" without looking at the counts.
    """
    box = ROOM_BOXES.get(room_name)
    if not box:
        return None, "no ROOM_BOXES entry for %r" % room_name
    p1, p2 = box
    last_err = None
    for attempt in range(ROOM_HASH_RETRIES):
        try:
            r = lab.rpc("scan", p1={"x": p1[0], "y": p1[1], "z": p1[2]},
                        p2={"x": p2[0], "y": p2[1], "z": p2[2]})
            unloaded = (r.get("counts") or {}).get("unloaded", 0)
            if not unloaded:
                return r["hash"], None
            last_err = ("%d/%d cells unloaded at scan time (attempt %d/%d)"
                       % (unloaded, r.get("total"), attempt + 1,
                          ROOM_HASH_RETRIES))
        except Exception as e:
            last_err = str(e)
        time.sleep(ROOM_HASH_RETRY_WAIT)
    return None, last_err


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
#
# -073 RE-DERIVED 2026-08-16 at 060ac10f4, when the counter-based RNG
# became the default (roadmap 1a) and the renderer got brighter: 1.113 ->
# 1.053 / 1.069 on two consecutive clean runs. BOTH HALVES OF THAT MOVED:
# the ratio, and its own reproducibility. The patch is 100% clipped, so
# the referee is inverting an ACES shoulder — the printed spread was
# 0.000228 under the old RNG and is 0.016 under the new one, with a
# per-pixel sigma of 0.6 on a mean of 6.9. So the pin is the mean of the
# two runs and the tolerance is 3x the observed spread, which is 16x
# looser than -050's and still 12x tighter than the defects calibrate
# plants (bounces1 took this room to 0.485, clay to 0.546). It remains a
# TRUNCATION detector; nothing about it is precise.
FURNACE_PINNED = {"050": 0.982, "073": 1.061}
FURNACE_TOL = {"050": 0.003, "073": 0.048}
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


GALLERY_DEPLOY = os.path.join(HERE, "claude_gallery_deploy.py")
# Marker of "vegetation has been cleared on this world at least once".
# The clear step (claude_gallery_deploy.CLEAR) walks a large box in
# y-slabs and is slow; the ROOM BUILDS are cheap idempotent box-fills and
# must re-run every time (that is the whole point of 1b — a room a
# creeper detonated inside, or a node someone dug, is repaired by the
# next deploy, not trusted). So: full deploy (with clear) once per world,
# --skip-clear on every run after, per the handoff ("idempotent;
# --skip-clear after the first run of a seat").
GALLERY_CLEARED_MARKER = os.path.join(REPO, SEAT_WORLD,
                                      ".claude_gallery_cleared")


# Expected room hash right after a fresh deploy (door OPEN — deploy
# leaves doors open for walkability; set_doors(True) runs later, in
# cmd_run). Only for rooms with no util/claude_rooms_check.py spec
# (furnace-050/furnace-073/cornell already get a stronger node-by-node
# diff via check_room_integrity(), just later in the flow, after the
# door is shut). Without this, a silent no-op deploy is
# indistinguishable from real room damage until the NEXT capture, diffed
# against a golden from a past run — the handoff's own line: "verify by
# hash, not by faith." Values measured 2026-08-16 on a fresh
# --skip-clear deploy, one-tracer @ b9e4382df; re-derive if a room
# builder's own geometry changes on purpose.
EXPECTED_DEPLOY_HASH = {
    "cave-skylight": "ad2aaf48",
    "cave-glass": "57b6fef8",
    "sky-furnace-050": "93ddf298",
    "cozy": "af59b2a6",
    "exterior-ci": "cc32604a",
}


def check_deploy_hashes():
    """{room: (got, want)} for every EXPECTED_DEPLOY_HASH entry that
    disagrees, or {} if all match."""
    bad = {}
    for room, want in EXPECTED_DEPLOY_HASH.items():
        got, err = room_hash(room)
        if got != want:
            bad[room] = (got, want, err)
    return bad


def verify_deploy_hashes():
    """None on success, else a message distinct from "room hash mismatch
    vs golden" (that one compares against a PAST run's capture; this one
    checks the deploy that just ran against what its own builders are
    supposed to produce, deterministically, every time)."""
    bad = check_deploy_hashes()
    if not bad:
        return None
    print("deploy-hash mismatch on first check (%s) — retrying deploy once "
          "before failing" % ", ".join(bad))
    deploy_gallery()   # idempotent, --skip-clear; a silent no-op the
                       # first time looks exactly like a successful build
    bad2 = check_deploy_hashes()
    if not bad2:
        print("deploy-hash mismatch resolved on retry")
        return None
    return ("deploy did not produce the expected room(s) after a retry: "
            + "; ".join("%s want %s got %s%s"
                       % (r, want, got, (" (%s)" % err) if err else "")
                       for r, (got, want, err) in bad2.items()))


def deploy_gallery():
    """Rebuild the gallery on THIS run's seat, every run. Returns (ok, info).
    Must run AFTER the server/client are up (it talks to the bridge)."""
    skip_clear = os.path.exists(GALLERY_CLEARED_MARKER)
    cmd = [sys.executable, GALLERY_DEPLOY] + (["--skip-clear"]
                                               if skip_clear else [])
    print("gallery: deploying (%s)..."
          % ("skip-clear" if skip_clear else "FULL CLEAR, first run on this world"))
    r = sh(cmd)
    info = {"cmd": cmd, "returncode": r.returncode, "skip_clear": skip_clear,
            "stdout_tail": (r.stdout or "")[-6000:],
            "stderr_tail": (r.stderr or "")[-2000:]}
    ok = r.returncode == 0 and "MISSING NODE NAMES" not in (r.stdout or "")
    if ok and not skip_clear:
        os.makedirs(os.path.dirname(GALLERY_CLEARED_MARKER), exist_ok=True)
        open(GALLERY_CLEARED_MARKER, "w").write("cleared by run\n")
    return ok, info


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
    """Reset the accumulator by teleporting AWAY and back, and PROVE it.

    Two arms of the same room (cornell / cornell-nee1) sit at the same
    position, and game.cpp only resets the accumulator when the camera
    moves — so without this the estimator arm would settle on top of the
    photo arm's history and every number would be a blend of the two.
    Turning in place does NOT reset (measured 2026-08-15: rpc tp yaw sets
    the PLAYER's yaw, the client owns its camera, still_frames climbed
    straight through it). Returning to the vantage's stored REST position
    lands the body where it already rests, so nothing slides.

    THE PROOF IS A COUNTER, and it had to become one (2026-08-16, settle
    calibration). This used to poll still_frames for a value <= 3, or
    below whatever it read before the teleport. Both clauses look
    through a 1 Hz keyhole: claude_stats.json is rewritten once per
    second while still_frames climbs at 70-90 per second, so the window
    in which a reset frame is visible as "<= 3" is ~0.03 s wide, and the
    "below before" clause is unusable whenever `before` is ALREADY small.
    That is not a corner case — cozy-day-dark-ci enters with `before` =
    11, because the previous arm's lamp swap clamped it — and it
    produced `still_frames never dropped after reset (was 11, last seen
    1353)`: a RED for a reset the engine guarantees on any camera move
    and the aim assertion independently confirms happened. The same
    false RED is already on record in spec/measured.md "Gate 4, second
    half", inside a flake that was attributed wholly to the grass ABM.

    game.cpp now exports `accum_resets`, incremented wherever
    still_frames is ZEROED (camera move, grid rebase, sky change) and
    NOT where it is merely clamped to 10 by a world change. A counter
    cannot be missed by sampling — any two reads bracket every reset
    between them — so this is exact where a threshold was a guess.
    """
    st0 = lab.read_stats() or {}
    before_resets = st0.get("accum_resets")
    before = st0.get("still_frames")
    try:
        lab.goto(park)
        time.sleep(0.5)
        lab.goto(vantage)
    except Exception as e:
        return "reset rpc failed: %s" % e
    deadline = time.time() + RESET_POLL_TIMEOUT
    last, last_resets = None, None
    while time.time() < deadline:
        st = lab.read_stats() or {}
        last = st.get("still_frames") if st else None
        last_resets = st.get("accum_resets") if st else None
        if before_resets is not None and last_resets is not None:
            # Two teleports happened (away, then back), so the counter
            # must have moved at least twice; require only that it moved,
            # since a client that started mid-sequence is still proof.
            if last_resets > before_resets:
                return None
        elif last is not None:
            # Pre-counter client (older binary): fall back to the old,
            # keyhole-limited test rather than refusing to run at all.
            if last <= RESET_STILL_FRAMES_MAX:
                return None
            if before is not None and last < before:
                return None
        time.sleep(RESET_POLL_INTERVAL)
    if before_resets is not None and last_resets is not None:
        return ("accum_resets never moved after the teleport (%s -> %s); "
                "the camera did not move, or the client is not the one "
                "being measured" % (before_resets, last_resets))
    return ("still_frames never dropped after reset (was %s, last seen %s) "
            "— and this client exports no accum_resets, so the check is "
            "the old sampled one and may be reporting a reset it simply "
            "could not see" % (before, last))


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
                       "frame is the raster pass-through, or the grid was "
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


class Timeline:
    """Where a shot's seconds actually went.

    THE HANDOFF'S ORDER, and it is the right one: ~185 s of a 682 s run
    was not settle at all and NOBODY HAD ATTRIBUTED IT. The last time
    someone looked at a number of that shape it turned out to be 25 s per
    capture burning a timeout inside await_grid, which had read as a
    benign retry for a month. So this is built before anything is cut,
    and it stays afterwards -- a phase that grows back is then one grep
    away instead of one archaeology session away.

    Wall clock only, and deliberately dumb: a monotonic clock, one mark
    per phase, no averaging and no smoothing. The numbers land in
    run.json next to the settle they are being compared against.
    """

    def __init__(self):
        self._t0 = time.monotonic()
        self._last = self._t0
        self.phases = {}

    def mark(self, name):
        now = time.monotonic()
        self.phases[name] = round(now - self._last, 2)
        self._last = now
        return self.phases[name]

    def done(self):
        d = dict(self.phases)
        d["TOTAL"] = round(time.monotonic() - self._t0, 2)
        d["_other"] = round(d["TOTAL"] - sum(self.phases.values()), 2)
        return d


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


GRID_TIMEOUT = 25.0      # s to wait for a snapshot to become valid
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


# Rooms that are fully sealed boxes (walls/floor/ceiling close around the
# camera) get a HIGHER solid-count floor than the universal one below.
# Measured live 2026-08-16 (fresh Release binary, grid_debug=3,
# grid_follow=0, one snapshot per vantage): grid_solid ranged
# 239,200 (furnace-050, the smallest room) to 322,562 (cozy) — the 128^3
# bubble also picks up the mgflat ground plane and anything else within
# 64 nodes, so even "exterior-ci" (no build at all) read 269,377. 100,000
# is comfortably below every measured sealed room and comfortably above
# what a genuinely broken/near-empty bubble would read.
SEALED_ROOMS = {"furnace-050", "furnace-073", "cornell", "cozy",
                 "cave-skylight", "cave-glass"}
VOLUME_SOLID_FLOOR_SEALED = 100000


def await_grid(marker, block, room=None, tries=3, seq_before=None):
    """Prove the tracer has something to trace BEFORE the settle starts.

    A capture must never be taken over an empty bubble: a client with no
    grid marches nothing, and the frame looks like the tracer is off
    (John, from the screen, 2026-08-15 — and he was right). That used to
    be the normal state, because claude_grid_follow was pinned to 0 and
    nothing bootstrapped a grid. Follow is on again as of 2026-08-16,
    so the seat does bootstrap — but each capture still triggers its own
    snapshot after the teleport (the follow path re-centres on the poll,
    up to 1 s later, and a settle must not start before the grid is the
    one being photographed), and this waits for the client to say it took.

    FOUND 2026-08-15 (roadmap 1b, cave-skylight/cave-glass): this used
    to also require area_total > 0, on the theory that a snapshot fired
    before the map arrived would show up as an empty bubble. Read
    game.cpp before trusting that theory further (debugging discipline:
    read the code, not the memory of intent) — area_total is not a
    general occupancy count, it is the NEE area-LIGHT-cell count
    ("emissive cells the snapshot actually found", ClaudeTraceGrid::
    area_total, game.cpp ~149). So area_total > 0 was never "is there a
    grid" — it was "is there a LIT grid", true by coincidence for
    every room that existed before 1b (all of them had lamps) and false
    BY DESIGN for a sealed box with none.

    FOUND AGAIN 2026-08-16 (Opus review of the 1b diff): grid_valid ==
    1 alone is a TAUTOLOGY, not just an insufficient-for-unlit-rooms
    check — game.cpp sets g_claude_grid.valid = true ONCE and never
    clears it, so after the very first snapshot on a seat, grid_valid
    reads 1 forever regardless of what a later snapshot actually found.
    An all-air bubble at a brand-new vantage would pass. Fixed at the
    source: game.cpp now exports grid_solid (non-air cells the LAST
    WALK actually found, computed unconditionally every call, before
    the unchanged-content early return) and grid_snap_seq (increments
    once per call, so a caller can prove a NEW walk happened rather than
    reading a stale count from a walk at a different vantage entirely).

    MEASURED 2026-08-16 (settle calibration): seq_before must be read by
    the CALLER, BEFORE the snapshot is requested, and it was not. This
    function read it on entry -- i.e. AFTER push_dials() had already
    written the request and slept 1.4 s for the client's ~1 Hz poll. The
    poll usually lands inside that sleep, so seq_before was routinely the
    POST-snapshot value, "has a new walk happened" could never become
    true, and attempt 1 burned the full GRID_TIMEOUT before a retry
    with a fresh marker got the answer. It cost 25 s per capture: every
    shot of both runs since claude_grid_follow went back to 1 reports
    `attempts: 2` (13/13 and 12/13), against 13/13 `attempts: 1` on the
    follow-off golden run -- follow ON makes the race near-certain,
    because the follow path re-centres on arrival at the new vantage and
    bumps snap_seq on its own. ~5 minutes of every ~22 minute run.
    """
    if seq_before is None:                 # legacy callers: racy, as above
        seq_before = (lab.read_stats() or {}).get("grid_snap_seq")
    floor = VOLUME_SOLID_FLOOR_SEALED if room in SEALED_ROOMS else 1
    for attempt in range(tries):
        deadline = time.time() + GRID_TIMEOUT
        while time.time() < deadline:
            st = lab.read_stats() or {}
            solid = st.get("grid_solid") or 0
            if (st.get("grid_valid") == 1 and solid >= floor
                    and st.get("grid_snap_seq") != seq_before):
                return {"ok": True, "attempts": attempt + 1,
                        "grid_valid": st.get("grid_valid"),
                        "grid_solid": solid, "grid_floor": floor,
                        "grid_snap_seq": st.get("grid_snap_seq"),
                        "area_emitters": st.get("area_emitters"),
                        "area_total": st.get("area_total"),
                        "emitters": st.get("emitters")}
            time.sleep(VOLUME_POLL)
        if attempt + 1 < tries:      # re-trigger with a fresh token
            push_dials(block, "%s_retry%d" % (marker, attempt))
    st = lab.read_stats() or {}
    return {"ok": False, "attempts": tries,
            "grid_valid": st.get("grid_valid"),
            "grid_solid": st.get("grid_solid"), "grid_floor": floor,
            "area_emitters": st.get("area_emitters"),
            "area_total": st.get("area_total"),
            "error": "no grid after %d snapshot requests: grid_valid=%s "
                     "grid_solid=%s (floor %s), snap_seq unchanged=%s — "
                     "the tracer would be marching an empty bubble, or this "
                     "is a stale read from a different vantage's snapshot"
                     % (tries, st.get("grid_valid"), st.get("grid_solid"),
                        floor, st.get("grid_snap_seq") == seq_before)}


def await_frames(target, max_s=SETTLE_MAX_S, min_frames=SETTLE_MIN_FRAMES,
                 hard_max_s=SETTLE_HARD_MAX_S):
    """Block until the accumulator is `target` frames deep. THE SETTLE.

    Returns the record of what the wait actually did, which is the point:
    a wall-clock settle could not say how deep the frame it shot was, and
    every depth in this repo before today was an after-the-fact reading
    rather than a precondition.

    still_frames is read from claude_stats.json, which the client
    rewrites once per second, so this UNDER-reports by up to one stats
    window -- the safe direction for a floor. The shutter that follows
    adds another ~1 s (the settings-patch poll plus a ~3 MB synchronous
    PNG write), measured, so the frame taken is deeper than the number
    this returns, consistently for every arm.

    Resets are counted, not smoothed over: still_frames is monotone
    between resets, so any decrease is a real world change (or a camera
    move, which the aim guard would catch separately). Counting them is
    how a run reports what the world did to it.
    """
    t0, last, resets = time.time(), None, []
    while True:
        st = lab.read_stats() or {}
        sf = st.get("still_frames")
        el = time.time() - t0
        if sf is not None:
            if last is not None and sf < last:
                resets.append({"at_still_frames": last, "dropped_to": sf,
                               "after_s": round(el, 1)})
            last = sf
            if sf >= target:
                return {"reached": True, "still_frames": sf,
                        "wall_s": round(el, 1), "target": target,
                        "resets": resets}
            if (el >= max_s and sf >= min_frames) or el >= hard_max_s:
                return {"reached": False, "still_frames": sf,
                        "wall_s": round(el, 1), "target": target,
                        "resets": resets,
                        "why": ("hard ceiling %.0fs" % hard_max_s
                                if el >= hard_max_s
                                else "ceiling %.0fs with >= %d frames"
                                     % (max_s, min_frames))}
        time.sleep(SETTLE_POLL)


def capture(shot, vantage, park, dials, rundir, settle, vantage_name=None):
    """dials -> park -> vantage (proven reset) -> snapshot -> re-assert
    dials -> settle (N frames) -> shutter -> N read BEFORE the PNG write
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
    room = room_of(vantage_name or shot["name"])
    info = {}
    tl = Timeline()
    info["dials_preapplied"] = push_dials(
            dials, "%s_pre_%d" % (name, time.time_ns()))
    tl.mark("dials_pre")
    info["reset_error"] = reset_accumulation(vantage, park)
    tl.mark("reset")
    info["aim_at_start"] = read_aim()
    tl.mark("aim_start")
    # Take one snapshot here — before the settle, since it clamps
    # still_frames — and re-assert the dials after it. Follow being on
    # does not make this redundant: the follow path only re-centres on
    # its next 1 Hz poll, and the settle must not start against the
    # PREVIOUS vantage's bubble.
    marker = "%s_%d" % (name, time.time_ns())
    block = dict(dials, claude_grid_snapshot=marker)
    # BEFORE the request is written: the client cannot have served a
    # snapshot we have not asked for yet, and push_dials sleeps 1.4 s for
    # the ~1 Hz poll, which is long enough for the answer to arrive
    # before a read taken after it. See await_grid.
    seq_before = (lab.read_stats() or {}).get("grid_snap_seq")
    info["dials_pushed"] = push_dials(block, marker)
    tl.mark("dials_push")
    # the settle clock starts only once the grid is proven present:
    # a snapshot also clamps still_frames, so waiting here costs nothing
    # and a capture over an empty bubble costs everything.
    info["grid"] = await_grid(marker, block, room=room,
                                  seq_before=seq_before)
    tl.mark("await_grid")
    info["settle"] = await_frames(settle)
    tl.mark("settle")

    # still_frames BEFORE waiting on the ~3 MB PNG write (measured.md
    # "Owed to the harness"): the record written after the write is 1-3
    # frames late. This one is up to 1 s stale in the other direction —
    # the stats file is rewritten once per second — so it UNDER-reports,
    # which is the safe side for a convergence floor.
    ok, detail = aim_ok(info.get("aim_at_start"), read_aim(), vantage)
    info["aim_at_shutter"] = {"ok": ok, "detail": detail}
    tl.mark("aim_shutter")
    before = lab.newest_shot()
    st = lab.read_stats() or {}
    info["still_frames_at_shutter"] = st.get("still_frames")
    info["stats_at_shutter"] = {k: st.get(k) for k in
                                ("grid_valid", "grid_solid",
                                 "grid_snap_seq", "area_emitters",
                                 "area_total", "emitters", "frame_ms_avg",
                                 "busy_ms", "pass_ms", "accum_alpha")}
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
    tl.mark("shutter")
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
    tl.mark("png")
    info["dial_state"] = dial_state(png, dials, marker)
    tl.mark("dial_state")

    # Room hash (roadmap 1b): the same box every run, hashed by the
    # bridge (OPS.scan) AFTER the settle, so it reflects exactly what was
    # in frame for this shutter — not what the deploy intended. Written
    # into both this run's in-memory record and the copied sidecar, so a
    # golden pinned from this run carries its own room hash forward.
    h, herr = room_hash(room)
    info["room"] = room
    info["room_hash"] = h
    info["room_hash_error"] = herr
    try:
        capjson = os.path.join(rundir, name + ".capture.json")
        capd = json.load(open(capjson)) if os.path.exists(capjson) else {}
        capd["room"] = room
        capd["room_hash"] = h
        json.dump(capd, open(capjson, "w"), indent=2)
    except Exception as e:
        info["room_hash_error"] = "%s (also: sidecar write failed: %s)" % (herr, e)

    tl.mark("room_hash")
    info["timeline"] = tl.done()
    return dst, info


def park_for(vantage_name, vantages):
    """Somewhere else the player can STAND. Landing anywhere but a rest
    position starts a physics slide that resets accumulation for up to a
    minute (the cozy-ci landmine)."""
    alt = "furnace-050" if vantage_name != "furnace-050" else "cornell"
    return vantages[alt]


# ---------------------------------------------------------------- referees

def run_referee(kind, arg, png, rundir, name, golden_png=None,
                golden_refused=None):
    """golden_refused, when set, means a golden PNG exists but the room
    hash gate (the same one diff_against_gated uses for the RMS diff)
    refused it for this shot -- the --ratio comparison is a pixel
    comparison against that golden image, exactly like the RMS diff, and
    was NOT gated on the room hash until Opus's review of the 1b diff
    caught it: a hash-refused Cornell still printed ratio numbers,
    quietly comparing pixels from two different rooms. golden_png is
    dropped from the command in that case, and the refusal reason is
    carried through to the parsed result so cornell_verdict() can say
    REFUSED instead of "no golden pinned" (a golden IS pinned; it was
    refused, a different claim)."""
    script = os.path.join(HERE, "claude_%s_check.py" % kind)
    cmd = [sys.executable, script, png] + ([arg] if arg else [])
    if kind == "furnace" and FURNACE_PATCH:
        cmd += ["--patch"] + [str(v) for v in FURNACE_PATCH]
    use_golden = golden_png if not golden_refused else None
    if kind == "cornell":
        cmd += ["--regions"]
        if use_golden:
            cmd += ["--ratio", use_golden]
    r = sh(cmd)
    text = (r.stdout or "") + (r.stderr or "")
    with open(os.path.join(rundir, name + ".referee.txt"), "w") as f:
        f.write("$ %s\n\n%s\n[exit %d]\n" % (" ".join(cmd), text, r.returncode))
    out = {"kind": kind, "arg": arg, "returncode": r.returncode,
           "txt": name + ".referee.txt", "stdout": text,
           "golden_png": use_golden, "golden_refused": golden_refused}
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
    if ref.get("golden_refused"):
        return "-", "REFUSED: %s" % ref["golden_refused"]
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


def other_room_hash(run_id, name):
    """The room hash a PREVIOUS run recorded for shot `name`, from its own
    run.json (not the sidecar — run.json is what golden() pins)."""
    if not run_id:
        return None, "no run"
    try:
        run = json.load(open(os.path.join(CI_DIR, run_id, "run.json")))
        s = (run.get("shots") or {}).get(name) or {}
        h = (s.get("capture") or {}).get("room_hash")
        return h, None if h else "that run recorded no room_hash for %s" % name
    except Exception as e:
        return None, str(e)


def diff_against_gated(run_id, name, png, this_hash, kind):
    """diff_against, but REFUSED (not computed) when the room hash the
    comparison run recorded for this shot disagrees with this capture's
    own hash. This is the mechanism roadmap 1b exists to add: a diff
    against a room someone dug a hole in — or a creeper detonated inside
    — must be refused on sight, not silently reported as a number that
    then has to be explained after the fact (measured.md, "CI red/green":
    one dug node leaked daylight through Cornell for a day and nothing
    noticed until an RMS drifted the wrong way)."""
    other_hash, other_err = other_room_hash(run_id, name)
    if other_hash is None:
        # No comparable hash on record (no run pinned, or it predates
        # room hashing) — not a refusal, just nothing to gate on yet.
        return diff_against(run_id, name, png), None
    if this_hash is None:
        return None, ("REFUSED: this capture has no room hash of its own "
                      "(%s) — cannot compare against %s" % (kind, run_id))
    if this_hash != other_hash:
        return None, ("REFUSED: room hash mismatch vs %s %s (%s changed: "
                      "%s -> %s). The room was rebuilt, damaged, or dug "
                      "into since that capture; a pixel diff against it "
                      "would compare two different rooms." % (
                          kind, run_id, name, other_hash, this_hash))
    return diff_against(run_id, name, png), None


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
    if d.get("refused"):
        return "REFUSED (room hash)"
    if d.get("error"):
        return "err"
    return "%.3f (worst %.1f)" % (d["rms"], d["worst_channel_delta"])


def _rms(shot):
    """Both diffs, one string — previous run AND the pinned golden."""
    return "prev %s | golden %s" % (_one_rms(shot.get("rms_vs_prev")),
                                    _one_rms(shot.get("rms_vs_golden")))


def write_run_html(rundir, run):
    e, parts = html.escape, []
    parts.append("<title>CI %s</title><style>%s</style>" % (e(run["tag"]), CSS))
    parts.append("<h1>%s %s <span class=meta>%s</span></h1>"
                 % (e(run["tag"]), _rg(run), e(run["branch"])))
    parts.append('<div class=meta>%s &middot; settle %g frames &middot; build %s'
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

    if getattr(args, "skip_deploy", False):
        # --skip-deploy: capture the world AS IT SITS, no rebuild first.
        # Deploy-every-run is the feature that auto-heals a dug node or a
        # creeper's damage; it also means "dig a hole, then claude_ci
        # run" can never demonstrate the room-hash REFUSAL, because the
        # hole is repaired before the capture ever happens. This flag is
        # how the refusal half of the gate gets tested honestly, on a
        # deliberately-broken room, without the auto-heal masking it.
        #
        # Still emerge (force-load) the build region, though: bring_up_seat
        # just restarted the server, which starts with NOTHING loaded, and
        # a bridge op like OPS.door silently no-ops on an unloaded chunk --
        # confirmed live (2026-08-15): furnace-050/furnace-073's doors
        # stayed OPEN and Cornell scanned 369 "unloaded" nodes on a
        # --skip-deploy run with no emerge, which is a SECOND variable
        # riding along with the deliberately-broken node and has nothing
        # to do with the room-hash mechanism being tested.
        print("gallery: --skip-deploy, emerging only (no rebuild)")
        er = sh([sys.executable, GALLERY_DEPLOY, "--emerge-only"])
        run["gallery_deploy"] = {"skipped": True, "emerge_only": True,
                                 "returncode": er.returncode,
                                 "stdout_tail": (er.stdout or "")[-2000:]}
    else:
        ok, dep = deploy_gallery()
        run["gallery_deploy"] = dep
        if not ok:
            return ("gallery deploy failed (rc=%s): %s"
                    % (dep["returncode"], dep["stdout_tail"][-800:]
                       or dep["stderr_tail"][-800:]))
        deploy_hash_err = verify_deploy_hashes()
        run["deploy_hash_check"] = deploy_hash_err or "ok"
        if deploy_hash_err:
            return deploy_hash_err

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
               scored_only=bool(getattr(args, "scored", False)),
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               dials=base, shots={}, assertions=[])
    A = Assertions()
    rtl = Timeline()

    err = bring_up_seat(rundir, run, args)
    if err:
        run["aborted"] = err
        A.add("seat", False, err)
        return finish(run, rundir, A)

    rtl.mark("seat_up")
    A.add("build-release", (run.get("build_type") or "").lower() in RELEASE_TYPES,
          "CMAKE_BUILD_TYPE=%s" % run.get("build_type"))
    dep = run.get("gallery_deploy") or {}
    if dep.get("skipped"):
        A.add("gallery-deploy", True, "--skip-deploy: not rebuilt this run")
    else:
        A.add("gallery-deploy", dep.get("returncode") == 0
              and "MISSING NODE NAMES" not in dep.get("stdout_tail", ""),
              "skip_clear=%s rc=%s" % (dep.get("skip_clear"), dep.get("returncode")))
        dhc = run.get("deploy_hash_check")
        A.add("deploy-room-hashes", dhc == "ok",
              dhc if dhc != "ok" else "%d rooms match EXPECTED_DEPLOY_HASH"
              % len(EXPECTED_DEPLOY_HASH))
    A.add("shaders-compile", not run["shader_failures"],
          "%d 'Failed to compile' lines in debug.txt"
          % len(run["shader_failures"]))
    A.add("freeze-deepening", run["freeze"].get("deepening"),
          "still_frames %s, time_speed %s" % (run["freeze"].get("still_frames"),
                                              run["freeze"].get("time_speed")))

    vs = lab.load_vantages()
    run["doors_shut"] = set_doors(True)
    rtl.mark("doors_shut")
    run["room_integrity"] = check_room_integrity()
    rtl.mark("room_integrity")
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
        for shot_def in shots_for(args):
            name, vname = shot_def["name"], shot_def["vantage"]
            if vname not in vs:
                A.add("%s-capture" % name, False,
                      "%s MISSING from claude_vantages.json" % vname)
                continue
            dials = dict(base, **shot_def.get("dials", {}))
            lamps_off = shot_def.get("lamps_off")
            if lamps_off:
                # cozy-day-dark-ci: swap the panel to its unlit twin for
                # THIS capture only, and guarantee it goes back even if
                # the capture throws — a stuck-dark cozy room would
                # silently poison every later cozy arm's golden.
                p1, p2 = ROOM_BOXES["cozy"]
                lab.rpc("lamps", p1=dict(zip("xyz", p1)),
                       p2=dict(zip("xyz", p2)), state="off")
            try:
                try:
                    png, cap = capture(shot_def, vs[vname], park_for(vname, vs),
                                       dials, rundir, args.settle, vname)
                except Exception as ex:
                    # One dead vantage must not cost the other four.
                    run["shots"][name] = {"error": str(ex)}
                    A.add("%s-capture" % name, False, str(ex))
                    continue
            finally:
                if lamps_off:
                    p1, p2 = ROOM_BOXES["cozy"]
                    lab.rpc("lamps", p1=dict(zip("xyz", p1)),
                           p2=dict(zip("xyz", p2)), state="on")
            shot = {"png": os.path.basename(png), "capture": cap}
            # Room-hash gate BEFORE the referee runs: the Cornell
            # --ratio comparison is a pixel diff against golden_png,
            # exactly like the RMS diff below, and both must be refused
            # together — a hash-refused room must not print ratio
            # numbers just because the RMS half of the check ran second.
            this_hash = cap.get("room_hash")
            shot["rms_vs_prev"], prev_refused = diff_against_gated(
                prev, name, png, this_hash, "prev")
            shot["rms_vs_golden"], gold_refused = diff_against_gated(
                gold, name, png, this_hash, "golden")
            if prev_refused:
                shot["rms_vs_prev"] = {"refused": prev_refused}
            if gold_refused:
                shot["rms_vs_golden"] = {"refused": gold_refused}
            if shot_def["referee"]:
                kind, arg = shot_def["referee"]
                shot["referee"] = run_referee(
                    kind, arg, png, rundir, name,
                    gold_png if kind == "cornell" else None,
                    golden_refused=gold_refused if kind == "cornell" else None)
            shot["verdict"] = list(verdict(shot_def, shot.get("referee")))
            shot["shot_s"] = rtl.mark("shot_" + name)
            run["shots"][name] = shot
            A.add("%s-room-hash" % name, not gold_refused,
                  gold_refused or ("hash %s (room %s)"
                                   % (this_hash, cap.get("room"))))

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
            vol = cap.get("grid") or {}
            A.add("%s-grid" % name, vol.get("ok"),
                  vol.get("error") or "solid=%s (floor %s) area_emitters=%s/%s "
                  "(snapshot attempt %s)"
                  % (vol.get("grid_solid"), vol.get("grid_floor"),
                     vol.get("area_emitters"), vol.get("area_total"),
                     vol.get("attempts")))
            st = (cap.get("stats_at_shutter") or {})
            sf = cap.get("still_frames_at_shutter")
            se = cap.get("settle") or {}
            # The settle is now reported, not assumed: how deep it got,
            # how long that took, and every accumulator reset it rode
            # out. A wall-clock settle could say none of this, which is
            # why the reset that turned CI into a coin flip was only
            # visible after the fact, in the shutter depth.
            A.add("%s-converged" % name, (sf or 0) >= CONVERGED_MIN,
                  "still_frames at shutter %s (min %d); settle reached %s "
                  "of %s in %ss, %d reset(s)%s"
                  % (sf, CONVERGED_MIN, se.get("still_frames"),
                     se.get("target"), se.get("wall_s"),
                     len(se.get("resets") or []),
                     "" if se.get("reached") else
                     " — TARGET NOT REACHED: %s" % se.get("why")))
            if cap.get("reset_error"):
                A.add("%s-accum-reset" % name, False, cap["reset_error"])
            if shot_def["referee"]:
                A.add("%s-%s" % (name, shot_def["referee"][0]),
                      shot["verdict"][0] == "PASS", shot["verdict"][1])
    finally:
        run["doors_reopened"] = set_doors(False)
        rtl.mark("doors_reopened")
        run["timeline"] = rtl.done()
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
    if run.get("scored_only"):
        print("!! --scored: %d of %d arms shot. This run proves LESS than a "
              "full one and must not be pinned as a golden."
              % (len(run.get("shots") or {}), len(CI_SHOTS)))
    tl = run.get("timeline") or {}
    if tl:
        settle_s = sum((s.get("capture", {}).get("settle", {}) or {})
                       .get("wall_s", 0) or 0
                       for s in run.get("shots", {}).values())
        over = sum((s.get("capture", {}).get("timeline", {}) or {})
                   .get("TOTAL", 0) - ((s.get("capture", {}).get("settle", {})
                                        or {}).get("wall_s", 0) or 0)
                   for s in run.get("shots", {}).values())
        print("wall clock %.0f s = settle %.0f s + per-shot overhead %.0f s "
              "+ seat/deploy/referees %.0f s"
              % (tl.get("TOTAL", 0), settle_s, over,
                 tl.get("TOTAL", 0) - settle_s - over))
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
                                       args.settle, vname)
                except Exception as ex:
                    print("CAPTURE FAILED: %s" % ex)
                    run["results"].append({"defect": defect["name"],
                                           "arm": shot_def["name"],
                                           "error": str(ex)})
                    continue
                kind, arg = shot_def["referee"]
                # Same room-hash gate as cmd_run, keyed on the room's
                # PLAIN vantage name (shot_def["name"], e.g. "cornell"),
                # not the deformed calibrate name ("cornell_bounces1") --
                # the golden's run.json has no entry under the deformed
                # name to compare against.
                this_hash = cap.get("room_hash")
                _, gold_refused = diff_against_gated(
                    read_golden(), shot_def["name"], png, this_hash, "golden")
                ref = run_referee(kind, arg, png, rundir, name,
                                  gold_png if kind == "cornell" else None,
                                  golden_refused=gold_refused
                                  if kind == "cornell" else None)
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
    if run.get("scored_only"):
        print("REFUSING: %s was a --scored run (%d of %d arms). A golden "
              "every later run is measured against cannot be missing nine "
              "of its images." % (rid, len(run.get("shots") or {}),
                                  len(CI_SHOTS)))
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
    if args.annotate:
        # §8 clause 4: a referee box gets LOOKED AT on a real frame before
        # it is believed, not just printed as four numbers. Outlines are
        # white-on-black double strokes and each box is captioned with its
        # region name -- John is colorblind, so the drawing carries its
        # meaning in brightness, shape and words, never in hue.
        from PIL import Image, ImageDraw
        im = Image.open(args.image).convert("RGB")
        d = ImageDraw.Draw(im)
        w, h = im.size
        for name, boxes in cornell.REGIONS.items():
            for (x0, y0, x1, y1) in boxes:
                px = (x0 * w, y0 * h, x1 * w, y1 * h)
                d.rectangle(px, outline=(0, 0, 0), width=6)
                d.rectangle(px, outline=(255, 255, 255), width=2)
                d.text((px[0] + 6, max(0, px[1] - 16)), name,
                       fill=(255, 255, 255), stroke_width=3,
                       stroke_fill=(0, 0, 0))
        im.save(args.annotate)
        print("annotated: %s" % args.annotate)
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
        p.add_argument("--settle", type=int, default=SETTLE_FRAMES,
                       metavar="FRAMES",
                       help="ACCUMULATED FRAMES of stillness before each "
                            "shot (default %(default)s). Was 60.0 SECONDS "
                            "until 2026-08-16; see SETTLE_FRAMES for why "
                            "a clock was the wrong gate.")
        p.add_argument("--skip-build", action="store_true",
                       help="capture with the binaries already in ./bin")
        p.add_argument("--skip-deploy", action="store_true",
                       help="capture the world AS IT SITS, no gallery "
                            "rebuild first (scratch/diagnostic only — this "
                            "is what lets a deliberately-broken room's "
                            "damage survive to the capture instead of "
                            "being auto-healed)")
        p.add_argument("--allow-debug", action="store_true",
                       help="capture against a Debug build tree anyway "
                            "(scratch only — environment-laws forbids it)")
        p.add_argument("--scored", action="store_true",
                       help="shoot ONLY the four arms that carry a "
                            "pass/fail verdict (furnace-050, furnace-073, "
                            "cornell, cornell-nee1). NOT the default and "
                            "not a cheaper CI -- the nine it drops are how "
                            "a human sees what the referees are blind to. "
                            "A fast inner loop for Cornell work.")
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
    p.add_argument("--annotate", metavar="OUT.png",
                   help="draw the boxes on IMAGE and save, so a human can "
                        "check a referee box is on the surface it names")
    p.set_defaults(func=cmd_regions)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
