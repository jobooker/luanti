#!/usr/bin/env python3
"""claude_nee_sweep — the NEE estimator's cost/benefit A/B, per room.

roadmap step 1 (spec/roadmap.md in luanti-docs): does the NEE/MIS light
sampler (claude_nee=1) converge to the same picture as the plain photon
path (claude_nee=0) FASTER — fewer accumulated samples, or fewer wall
milliseconds — for the same RMS against a pinned photo golden? And what
does it cost per frame to run?

One dial flip, same build, same GPU, pinned resolution, frozen time,
parked camera. For each room x each mode in MODES, this script walks an
accumulation run past a ladder of sample-count checkpoints (CHECKPOINTS),
shooting a screenshot near each one and diffing it against the room's
golden photo. It also takes a settled, parked cost sample per arm
(measure_cost) so the ms/frame number is never confused with the
RMS-vs-N curve — two different questions, two different measurements,
never fused into one number.

WHY the sweep is shaped like this, not simpler:

  - N (still_frames) is the thing that actually accumulates rays, but a
    checkpoint can only be requested through the client's 1 Hz settings
    poll, so the shot lands up to one second of FRAMES after the N that
    armed it. At the renderer's own ~6-30 fps that is 2% of N=512 and
    50% of N=64 — and it is not a wash between the arms, because the two
    modes run at different frame rates, so the faster one would be
    systematically flattered. Exactly the quantity under test.
    So the frame clock is GRADED down per checkpoint (clock_for), until
    one poll is a small fraction of the target N; at the bottom of the
    ladder that is 1 fps, one poll per FRAME, and N=1..32 is addressable
    on the nose. Changing the cap is a doorway push, not a camera move,
    so it does not reset accumulation and one climbing run still serves
    the whole ladder. OVERLAP_N is then re-walked on the free-running
    clock, so the instrument shows that its own clock is not changing
    the image rather than asserting it.

  - fps_max and fps_max_unfocused are pushed TOGETHER, always. This
    is not paranoia: FpsControl::limit (renderingengine.cpp) picks
    between the two based on device->isWindowFocused(), and a CI
    seat's window focus is not this script's to control. Setting only
    one leaves the other at its hidden default and the sweep silently
    measures a sleep instead of a renderer (environment-laws, and the
    2026-08-15 handoff that finally scripted the pin).

  - Accumulation is reset by TURNING, never by moving. game.cpp resets
    still_frames on `moved > 0.05f || turned > 1e-4f` — a body that is
    not at the vantage's exact rest position after a teleport can
    physics-slide for up to ~a minute (the cozy-ci landmine recorded
    in claude_vantages.json), resetting accumulation the whole time.
    reset_accumulation() therefore yaws 90 degrees in place and comes
    straight back — same position, so nothing slides — rather than
    teleporting away and back.

  - measure_cost distrusts its own inputs. pass_ms is an EMA with
    alpha 0.1 (pipeline.cpp), so a mode flip needs ~40 frames before
    the number means anything; COST_SETTLE buys that settling time.
    And any cost sample that clusters at 1000/fps_cap is refused
    outright via lab.cap_artifact — a "cost" number that is actually
    measuring a sleep is not a warning, it is not evidence at all.

  - Every arm proves the dial actually took, by grep'ing the client's
    own [claude_settings_patch] log (lab.patch_log) for the mode it
    just set. A live A/B whose "on" arm cannot prove it was on has
    measured nothing.

Not attempted here: judging pass/fail. This script produces the curves,
tables and plots; the roadmap step's verdict is written by a human (or
a planning session) reading sweep.md/sweep.json against the gate stated
in spec/roadmap.md.

Usage:
  python3 util/claude_nee_sweep.py --rooms cornell,cozy-ci
  python3 util/claude_nee_sweep.py --order 10 --skip-build
  python3 util/claude_nee_sweep.py --no-seat   # client already running
"""
import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab  # noqa: E402  (same dir; shares the file-RPC channel)
import claude_ci as ci    # noqa: E402  (reuses the seat lifecycle)

# ---------------------------------------------------------------- constants
# Nothing below this line is allowed to hide in the body of the script.

ROOMS = ["furnace-050", "furnace-073", "cornell", "cozy-ci"]   # vantage names
MODES = (0, 1)                       # claude_nee: 0 = plain photon path,
                                      # 1 = NEE/MIS estimator

# The sample-count ladder every curve is walked against. Dense at the low
# end (where a converging estimator either separates from its rival fast
# or doesn't), sparse at the top (where both modes should have long since
# agreed, and only wall-clock cost still differs).
CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# --- the frame clock, graded per checkpoint -------------------------------
# A checkpoint is requested through the settings-patch file, which the
# client reads on its 1 Hz poll, so the shot lands up to ONE POLL after
# the N that armed it — i.e. up to one second's worth of frames later.
# At the renderer's own ~6-30 fps that slop is 6-30 frames: 2% of N=512
# and 50% of N=64. It is not a wash between the arms, either, because the
# two modes run at DIFFERENT frame rates, so the faster mode would be
# systematically flattered — a bias pointing straight at the quantity
# being measured.
#
# The fix is to slow the frame clock in proportion to the checkpoint, so
# one second of frames stays a small fraction of N. The cap is a doorway
# push and does not touch the camera, so it can change MID-RUN without
# resetting accumulation. Cost is never read on a graded clock — cost has
# its own fast-clock measurement (measure_cost), and the wall-time axis
# is built from that, never from these frames' wall time.
BRACKET_FRAC = 0.05  # keep the one-poll slop under 5% of the target N
CLOCK_FLOOR = 1      # FpsControl::limit clamps to max(fps_limit, 1.0f),
                     # so 1 fps (1 s/frame, one poll per FRAME) is the
                     # slowest clock the engine allows — and the only one
                     # that can address N = 1..16 at all.
FAST_FPS = 200       # the pinned measuring cap: no sleep is taken at this
                     # cap by any frame this renderer produces.
SLOW_FPS = CLOCK_FLOOR   # kept as a name for the low end of the ladder
OVERLAP_N = [32]         # re-measured on the FAST clock too, as the
                         # instrument's own check that the graded clock
                         # and the free-running one agree on RMS at the
                         # same N. If they disagree, the clock is doing
                         # something to the image and nothing else here
                         # is evidence.

# still_frames vs the shot, from the loop order in Game::run():
#   3043 draw_times.limit()      <- dtime, including any cap sleep
#   3049 pollSettingsPatch()     <- claude_screenshot fires HERE, and
#                                   grabs the frame rendered last pass
#   3050 claudeUpdateAccum()     <- still_frames += 1
#   3051 claudeWriteStats()      <- writes the INCREMENTED value
#   3120 updateFrame()           <- renders, using the new value
# so the stats file written in the shot's own iteration is one ahead of
# the image that was captured. N_shot = reported - 1. Recorded raw as
# well as corrected, so a later reader can re-derive rather than trust.
N_SHOT_OFFSET = -1

COST_SAMPLES = 6     # independent stats windows averaged for the cost row
COST_SETTLE = 12.0   # s parked before the cost samples are believed —
                      # pass_ms is an EMA(alpha=0.1) in pipeline.cpp, so a
                      # mode flip needs ~40 frames (well under 12s @200fps)
                      # before it reflects the NEW mode rather than a mix

# Dials pushed explicitly on EVERY arm. An unset dial is not a default —
# it is a silent zero, and this sweep must not measure one by accident.
SWEEP_DIALS = {"claude_view": 0, "claude_bounces": 24, "claude_stats": 1,
               "claude_volume_follow": 0,
               # --- HUD suppression, and it is not cosmetic ------------
               # The RMS referee compares PIXELS, so anything drawn over
               # the frame is measured as if it were the renderer. Two
               # overlays wrecked the first pilot, both found by LOOKING
               # at a capture rather than by reading its number:
               #  * every screenshot makes the client announce "Saved
               #    screenshot to ..." in chat, and the default
               #    recent_chat_messages is 6 — so eleven captures per arm
               #    stacked six lines of white text DOWN THROUGH the
               #    compared region, and RMS rose with sample count
               #    because it was counting accumulated chat, not noise.
               #  * node_highlighting draws a box on the pointed-at node,
               #    dead centre of frame, where no crop can dodge it.
               "recent_chat_messages": 0,
               "node_highlighting": "none"}

RESET_TURN_DEG = 90       # yaw kick used to force a still_frames reset
                          # in place (see reset_accumulation)
RESET_SETTLE = 0.5        # s after the turn, before turning back
RESET_POLL_TIMEOUT = 15.0  # s to wait for still_frames to drop post-reset
RESET_POLL_INTERVAL = 0.5  # s between polls of claude_stats.json
RESET_STILL_FRAMES_MAX = 3  # still_frames must fall to <= this to call
                            # the reset proven, not just requested

SHOT_TIMEOUT = 20.0        # s to wait for request_shot()'s new png
SHOT_POLL_INTERVAL = 0.4   # s between newest_shot() polls
SHOT_SETTLE_INTERVAL = 0.25  # s between size checks while a PNG is written

CHECKPOINT_TIMEOUT = 180.0  # s to wait for one checkpoint before giving up
                            # on it and recording a timeout rather than
                            # hanging the whole sweep

COST_SAMPLE_GAP = 1.5  # s between cost samples — >= the 1s stats window,
                       # so each sample is read from a NEW window

# environment-laws "Seat / build": no benchmarks from a Debug binary.
# util/ci/build.sh defaults CMAKE_BUILD_TYPE to Debug, so the honest
# default of this tree is the one the law forbids — check the cache
# rather than trust the build command.
RELEASE_TYPES = ("release", "relwithdebinfo")
CMAKE_CACHE = os.path.join(REPO, "build", "CMakeCache.txt")

SWEEP_ROOT = os.path.join(REPO, "screenshots", "nee_sweep")


# ---------------------------------------------------------------- clocks

def build_type():
    """CMAKE_BUILD_TYPE as the configured build tree states it, or None."""
    try:
        for line in open(CMAKE_CACHE):
            if line.startswith("CMAKE_BUILD_TYPE:"):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return None


def set_caps(fps):
    """Push fps_max and fps_max_unfocused TOGETHER. FpsControl::limit
    (renderingengine.cpp) chooses between the two on window FOCUS, and a
    CI window's focus state is not ours to control — setting only one
    leaves the other at its hidden default and silently measures a sleep
    instead of a renderer (the 2026-08-15 fps-cap landmine)."""
    lab.doorway(fps_max=fps, fps_max_unfocused=fps)


def slow_clock(on):
    """True -> park on the floor clock. False -> the pinned measuring cap."""
    set_caps(SLOW_FPS if on else FAST_FPS)


def clock_for(target):
    """The fps cap that keeps one poll's worth of frames under
    BRACKET_FRAC of this checkpoint. Returns FAST_FPS once the renderer
    itself is slower than the cap would be, i.e. once no sleep is taken
    anyway and grading buys nothing."""
    want = max(CLOCK_FLOOR, int(target * BRACKET_FRAC))
    return min(want, FAST_FPS)


# ---------------------------------------------------------------- accumulation

def reset_accumulation(vantage):
    """Reset still_frames to 0 WITHOUT moving the player's body.

    game.cpp resets the accumulator on `moved > 0.05f || turned > 1e-4f`
    (the same guard that protects a still camera from resetting on
    sub-pixel jitter). A position teleport risks the cozy room's
    physics-slide landmine: a body not landed at its exact rest position
    slides for up to ~a minute, resetting accumulation the whole way
    (recorded against cozy-ci in claude_vantages.json, CI run 4). So we
    TURN instead of move — yaw kicks `turned` over the reset threshold
    with zero position delta — then use lab.goto() to land back on the
    vantage's exact recorded yaw/pitch.

    Returns None on success, or a string describing the failure (never
    raises — a bad reset is data, not a crash).
    """
    yaw = vantage["yaw"]
    pitch = vantage["pitch"]
    pos = dict(x=vantage["pos"][0], y=vantage["pos"][1], z=vantage["pos"][2])
    st0 = lab.read_stats() or {}
    before = st0.get("still_frames")
    try:
        lab.rpc("tp", pos=pos, yaw=(yaw + RESET_TURN_DEG) % 360, pitch=pitch)
        time.sleep(RESET_SETTLE)
        lab.goto(vantage)
    except Exception as e:
        return "reset rpc failed: %s" % e

    # Proving the reset is harder than doing it: claude_stats.json is
    # rewritten once per >=1 s window, so at a free-running clock the
    # window where still_frames <= 3 lasts about 50 ms and is essentially
    # unobservable — the pilot reported "reset failed" for resets that had
    # plainly worked. So accept EITHER a small absolute value (the slow
    # clock, where it is genuinely observable) or a large DROP from what
    # was there before (any clock). A drop is what a reset IS.
    deadline = time.time() + RESET_POLL_TIMEOUT
    last = None
    while time.time() < deadline:
        st = lab.read_stats()
        last = st.get("still_frames") if st else None
        if last is not None:
            if last <= RESET_STILL_FRAMES_MAX:
                return None
            if before is not None and last < before / 2.0:
                return None
        time.sleep(RESET_POLL_INTERVAL)
    return ("still_frames never dropped after reset (was %s, last seen "
            "%s) — accumulation may not have been reset" % (before, last))


def request_shot(token):
    """Write ONLY claude_screenshot into the patch file and poll for a NEW
    png. Deliberately does NOT use lab.shot(): its fixed settle sleep
    would blur exactly the frame-accurate timing this sweep depends on —
    a checkpoint has to be shot the instant N crosses the target, not
    after a few more seconds of accumulation have gone by."""
    before = lab.newest_shot()
    with open(lab.PATCH, "w") as f:
        f.write("claude_screenshot = %s\n" % token)  # unique per call —
        # claudeApplyPatchFile skips a patch whose content is unchanged
    deadline = time.time() + SHOT_TIMEOUT
    while time.time() < deadline:
        cur = lab.newest_shot()
        if cur and cur != before:
            # The file APPEARING is not the file being WRITTEN. A 1080p PNG
            # is ~3 MB and the client writes it synchronously from the
            # render thread; reading at first sight got 7 of 10 captures
            # truncated in the first pilot, and PIL failed on them AFTER
            # the run rather than during it. Wait for the size to stop
            # moving, then make PIL prove it can decode the whole thing.
            return _await_complete(cur)
        time.sleep(SHOT_POLL_INTERVAL)
    return None


def _await_complete(path):
    """Return path once the PNG is fully written and decodable, else None."""
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
                    im.load()          # decodes; raises on a truncated file
                return path
            except Exception:
                pass                   # still being written — keep waiting
        last = size
        time.sleep(SHOT_SETTLE_INTERVAL)
    return None


# ---------------------------------------------------------------- curves

def capture_curve(room, vantage, mode, targets, golden_png, outdir,
                  clock=None):
    """One accumulation run, MANY checkpoints. Resets ONCE up front, then
    shoots as N climbs past each target in turn — re-resetting for every
    checkpoint would cost sum(CHECKPOINTS) frames instead of the max(
    CHECKPOINTS) a single climbing run needs.

    The frame clock is re-graded before each target (clock_for), unless
    `clock` pins it — the overlap pass pins FAST_FPS so the graded and
    free-running clocks can be compared at the same N."""
    points = []
    # THE CLOCK GOES FIRST. set_caps() sleeps ~1.4 s waiting for the
    # client's 1 Hz poll, and at the free-running clock that is ~85 frames
    # of accumulation — so grading the clock AFTER the reset burned
    # straight past the whole low end of the ladder (pilot run: every
    # "N=1" landed at N=66). Reset only once the frame clock is already
    # slow, so the frames that elapse while we work are few.
    ordered = sorted(targets)
    set_caps(clock if clock is not None else clock_for(ordered[0]))
    err = reset_accumulation(vantage)
    if err:
        return [{"room": room, "mode": mode, "target_n": None,
                 "error": "reset failed: %s" % err}]

    for target in ordered:
        fps = clock if clock is not None else clock_for(target)
        # Re-grade UPWARD only, and never below the clock the reset ran
        # at: raising the clock mid-run is safe (N is already large by
        # then), lowering it would re-open the same window this fix shut.
        set_caps(fps)
        deadline = time.time() + CHECKPOINT_TIMEOUT
        point = {"room": room, "mode": mode, "target_n": target,
                 "clock_fps": fps,
                 "clock": "graded" if clock is None else "fast"}
        shot_taken = False
        while time.time() < deadline:
            st_before = lab.read_stats() or {}
            n_before = st_before.get("still_frames")
            if n_before is None:
                time.sleep(SHOT_POLL_INTERVAL)
                continue
            # The shot lands one poll after it is armed, and on the graded
            # clock one poll is ~one frame, so arm at target-1 and let the
            # correction below name the frame that was actually captured.
            if n_before + 1 < target:
                time.sleep(SHOT_POLL_INTERVAL)
                continue
            token = "nee_%s_m%d_n%d_%d" % (room, mode, target, time.time_ns())
            png = request_shot(token)
            st_after = lab.read_stats() or {}
            n_after = st_after.get("still_frames")
            # N_SHOT_OFFSET: the stats file written in the shot's own
            # iteration is one ahead of the image (see the constant).
            n = (n_after + N_SHOT_OFFSET) if n_after is not None else None
            point.update(n_before=n_before, n_after=n_after, n=n,
                         n_raw=n_after, n_offset=N_SHOT_OFFSET)
            if png is None:
                point["error"] = "screenshot timed out"
                break
            try:
                rms = lab.rms_diff(golden_png, png)
                point["rms"] = rms["rms"]
                point["worst_channel_delta"] = rms["worst_channel_delta"]
            except Exception as e:
                point["rms_error"] = str(e)
            dst = os.path.join(outdir, os.path.basename(png))
            try:
                shutil.copy2(png, dst)
                point["png"] = os.path.basename(dst)
                lab.write_capture_record(png)
                sidecar = os.path.splitext(png)[0] + ".capture.json"
                if os.path.exists(sidecar):
                    shutil.copy2(sidecar, os.path.join(
                        outdir, os.path.basename(sidecar)))
            except Exception as e:
                point["file_error"] = str(e)
            # busy_ms is the ONLY honest cost reading on a graded clock:
            # limit() measures it before the sleep, so a cap cannot inflate
            # it. Recorded per checkpoint to check that per-frame cost does
            # not drift with convergence depth — the shader casts the same
            # ray count at every depth, and the wall-time axis rests on
            # that, so it is checked here rather than asserted.
            point["busy_ms_at_checkpoint"] = st_after.get("busy_ms")
            point["frame_ms_avg_at_checkpoint"] = st_after.get("frame_ms_avg")
            shot_taken = True
            break
        if not shot_taken and "error" not in point:
            point["error"] = "timed out waiting for N=%d (%.0fs, clock %s fps)" % (
                target, CHECKPOINT_TIMEOUT, fps)
        points.append(point)
    return points


# ---------------------------------------------------------------- cost

def measure_cost(room, mode):
    """Parked on the FAST clock, settled COST_SETTLE s, take COST_SAMPLES
    stats reads ~COST_SAMPLE_GAP apart (a new 1s stats window each). ANY
    refused sample (lab.cap_artifact) marks the WHOLE row REFUSED, loudly
    — a cost number built on even one fps-cap-contaminated sample is not
    a warning, it does not get averaged in."""
    set_caps(FAST_FPS)
    lab.doorway(claude_stats=1)
    time.sleep(COST_SETTLE)
    caps = lab.fps_caps()
    samples = []
    refusal = None
    for i in range(COST_SAMPLES):
        lab.doorway(claude_stats=1)
        time.sleep(COST_SAMPLE_GAP)
        st = lab.read_stats()
        bad = lab.cap_artifact(st, caps)
        if bad and refusal is None:
            refusal = bad
        samples.append(st)
    row = {"room": room, "mode": mode, "n_samples": len(samples),
           "fps_caps": caps}
    if refusal:
        row["REFUSED"] = refusal
        return row
    frame_avgs = [s["frame_ms_avg"] for s in samples if s and "frame_ms_avg" in s]
    busy_ms = [s["busy_ms"] for s in samples if s and "busy_ms" in s]
    frame_worst = [s["frame_ms_worst"] for s in samples if s and "frame_ms_worst" in s]
    if not frame_avgs:
        row["REFUSED"] = "no usable stats samples"
        return row
    last = samples[-1] or {}
    row.update(
        frame_ms_avg=round(statistics.median(frame_avgs), 3),
        busy_ms=round(statistics.median(busy_ms), 3) if busy_ms else None,
        frame_ms_worst=max(frame_worst) if frame_worst else None,
        # pass_ms is an EMA(alpha=0.1) in pipeline.cpp — only the LAST
        # sample (after COST_SETTLE has let it converge on this mode)
        # is meaningful; earlier samples are still blending the old mode.
        pass_ms=last.get("pass_ms"),
        # names whether this room's light is AREA (NEE can aim at it) or
        # POINT (claudeEmitter0..7 only — the shader does not connect
        # NEE to those), which is how the per-room numbers must be read.
        area_emitters=last.get("area_emitters"),
        area_total=last.get("area_total"),
        emitters=last.get("emitters"),
        volume_valid=last.get("volume_valid"),
    )
    return row


# ---------------------------------------------------------------- golden

GOLDEN_N = 2000        # still_frames for a fresh golden. The 2026-08-15
                       # CI golden was ~537 samples because a Debug binary
                       # made anything deeper cost minutes; on Release the
                       # same room runs ~4.5x faster, so a far quieter
                       # truth is simply affordable now.
GOLDEN_TIMEOUT = 300.0


def capture_golden(rooms, vantages, outdir):
    """Shoot a fresh photo-mode golden per room, UNDER SWEEP CONDITIONS.

    Why not just use the pinned CI golden (20260815-174334_7ec19dc8d):
    it was captured on a DEBUG binary, with the F5 overlay on, a
    different item in the player's hand, and no HUD suppression. Every
    one of those differences lands inside the compared crop and is
    indistinguishable from renderer noise in an RMS number. A referee
    has to differ from the thing it judges in ONE way — the sample
    count — and that one did not. The old golden stays on disk as the
    historical record; it is not a valid ruler for this measurement.
    """
    os.makedirs(outdir, exist_ok=True)
    out = {"n_target": GOLDEN_N, "rooms": {}}
    for room in rooms:
        v = vantages[room]
        rec = {"room": room}
        lab.doorway(claude_nee=0, **SWEEP_DIALS)   # nee=0 IS photo mode (§6)
        set_caps(FAST_FPS)
        lab.goto(v)
        lab.doorway(claude_volume_snapshot="golden_%s_%d" % (room, time.time_ns()))
        lab.doorway(claude_volume_follow=0)
        err = reset_accumulation(v)
        if err:
            rec["error"] = err
            out["rooms"][room] = rec
            print("%-14s golden reset failed: %s" % (room, err))
            continue
        deadline = time.time() + GOLDEN_TIMEOUT
        n = 0
        while time.time() < deadline:
            st = lab.read_stats() or {}
            n = st.get("still_frames") or 0
            if n >= GOLDEN_N:
                break
            time.sleep(1.0)
        png = request_shot("golden_%s_%d" % (room, time.time_ns()))
        if png is None:
            rec["error"] = "screenshot timed out"
            out["rooms"][room] = rec
            continue
        dst = os.path.join(outdir, room + ".png")
        shutil.copy2(png, dst)
        lab.write_capture_record(png)
        side = os.path.splitext(png)[0] + ".capture.json"
        if os.path.exists(side):
            shutil.copy2(side, os.path.join(outdir, room + ".capture.json"))
        st = lab.read_stats() or {}
        rec.update(png=room + ".png", still_frames=st.get("still_frames"),
                   accum_alpha=st.get("accum_alpha"),
                   area_emitters=st.get("area_emitters"),
                   area_total=st.get("area_total"))
        out["rooms"][room] = rec
        print("%-14s golden at N=%s -> %s" % (room, rec.get("still_frames"), dst))
    json.dump(out, open(os.path.join(outdir, "golden.json"), "w"), indent=2)
    return out


# ---------------------------------------------------------------- arms

def arm(room, mode, vantage, golden_png, outdir):
    """One (room, mode): apply dials, prove the flip, measure cost on the
    free-running clock, then walk the whole ladder once on the graded
    clock and re-walk OVERLAP_N free-running as the instrument's own
    check."""
    result = {"room": room, "mode": mode}

    lab.doorway(claude_nee=mode, **SWEEP_DIALS)

    lab.goto(vantage)
    # re-centre the 128^3 bubble on THIS room before follow is frozen —
    # with claude_volume_follow=0 the bubble never re-centres on its own,
    # and a room outside a stale bubble renders wrong.
    snap_token = "nee_snap_%s_m%d_%d" % (room, mode, time.time_ns())
    lab.doorway(claude_volume_snapshot=snap_token)
    lab.doorway(claude_volume_follow=0)  # re-assert after the snapshot

    proof = [l for l in lab.patch_log() if "claude_nee = %d" % mode in l
             or "claude_nee=%d" % mode in l]
    result["dial_proof"] = proof[-1] if proof else "PROOF MISSING"

    result["cost"] = measure_cost(room, mode)

    outdir_arm = os.path.join(outdir, "%s_m%d" % (room, mode))
    os.makedirs(outdir_arm, exist_ok=True)

    result["curve"] = []
    result["curve"] += capture_curve(room, vantage, mode, CHECKPOINTS,
                                     golden_png, outdir_arm)
    # The instrument checks itself: re-walk the overlap checkpoint on the
    # free-running clock. Same N, same room, same mode — if the RMS does
    # not agree with the graded pass, the clock is changing the image and
    # every other point on this curve is suspect.
    result["curve"] += capture_curve(room, vantage, mode, OVERLAP_N,
                                     golden_png, outdir_arm, clock=FAST_FPS)
    slow_clock(False)
    return result


# ---------------------------------------------------------------- tables

def write_csv(path, data):
    fields = ["room", "mode", "clock", "clock_fps", "target_n", "n",
              "n_raw", "n_before", "n_after", "rms", "worst_channel_delta",
              "frame_ms_avg", "wall_ms", "busy_ms", "busy_ms_at_checkpoint",
              "frame_ms_worst", "cost_refused", "error", "rms_error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for arm_result in data["arms"]:
            cost = arm_result.get("cost") or {}
            frame_ms_avg = cost.get("frame_ms_avg")
            for pt in arm_result.get("curve", []):
                row = dict(pt)
                row["frame_ms_avg"] = frame_ms_avg
                row["busy_ms"] = cost.get("busy_ms")
                row["frame_ms_worst"] = cost.get("frame_ms_worst")
                row["cost_refused"] = cost.get("REFUSED")
                n = row.get("n")
                row["wall_ms"] = (n * frame_ms_avg
                                   if n is not None and frame_ms_avg is not None
                                   else None)
                w.writerow(row)


def write_md(path, data):
    lines = ["# NEE sweep — %s" % data["run_id"], "",
             "git: %s" % json.dumps(data["git"]), ""]
    lines.append("## Cost (fast clock, parked, %d samples over %gs settle)"
                 % (COST_SAMPLES, COST_SETTLE))
    lines.append("")
    lines.append("| room | mode | frame_ms_avg | busy_ms | frame_ms_worst | "
                 "area_emitters/total | volume_valid | note |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm_result in data["arms"]:
        c = arm_result.get("cost") or {}
        if c.get("REFUSED"):
            lines.append("| %s | %s | REFUSED | | | | | %s |" % (
                arm_result["room"], arm_result["mode"], c["REFUSED"]))
            continue
        lines.append("| %s | %s | %s | %s | %s | %s/%s | %s | |" % (
            arm_result["room"], arm_result["mode"],
            c.get("frame_ms_avg"), c.get("busy_ms"), c.get("frame_ms_worst"),
            c.get("area_emitters"), c.get("area_total"), c.get("volume_valid")))
    lines.append("")
    lines.append("## RMS vs golden, per room")
    lines.append("")
    for room in data["rooms"]:
        lines.append("### %s" % room)
        lines.append("")
        lines.append("| mode | clock | target_n | n | rms | worst_ch | note |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm_result in data["arms"]:
            if arm_result["room"] != room:
                continue
            for pt in arm_result.get("curve", []):
                lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                    pt.get("mode"), pt.get("clock"), pt.get("target_n"),
                    pt.get("n"), pt.get("rms"), pt.get("worst_channel_delta"),
                    pt.get("error") or pt.get("rms_error") or ""))
        lines.append("")
    open(path, "w").write("\n".join(lines))


def print_tables(data):
    print("\n=== cost ===")
    for arm_result in data["arms"]:
        c = arm_result.get("cost") or {}
        if c.get("REFUSED"):
            print("%-14s mode=%s  REFUSED: %s" % (
                arm_result["room"], arm_result["mode"], c["REFUSED"]))
        else:
            print("%-14s mode=%s  frame_ms_avg=%s busy_ms=%s worst=%s "
                 "area=%s/%s valid=%s" % (
                arm_result["room"], arm_result["mode"], c.get("frame_ms_avg"),
                c.get("busy_ms"), c.get("frame_ms_worst"),
                c.get("area_emitters"), c.get("area_total"),
                c.get("volume_valid")))
    print("\n=== rms vs golden ===")
    for arm_result in data["arms"]:
        for pt in arm_result.get("curve", []):
            print("%-14s mode=%s clock=%-4s target=%-4s n=%-4s rms=%s %s" % (
                arm_result["room"], arm_result["mode"], pt.get("clock"),
                pt.get("target_n"), pt.get("n"), pt.get("rms"),
                pt.get("error") or pt.get("rms_error") or ""))


# ---------------------------------------------------------------- plot

def plot(rundir, data):
    """rms_vs_n.png and rms_vs_ms.png. John is colorblind: mode is
    encoded in BRIGHTNESS and SHAPE, never hue — nee=0 is solid black
    with filled circles, nee=1 is mid-grey dashed with hollow squares."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping rms_vs_n.png / rms_vs_ms.png")
        return

    STYLE = {
        0: dict(color="black", linestyle="-", marker="o",
                markerfacecolor="black", linewidth=2, label="nee 0"),
        1: dict(color="0.55", linestyle="--", marker="s",
                markerfacecolor="none", linewidth=2, label="nee 1"),
    }

    for xkey, fname, xlabel in (("n", "rms_vs_n.png", "N samples"),
                                 ("wall_ms", "rms_vs_ms.png", "wall ms (n * frame_ms_avg)")):
        rooms = data["rooms"]
        fig, axes = plt.subplots(1, len(rooms), figsize=(5 * len(rooms), 5),
                                  squeeze=False)
        for i, room in enumerate(rooms):
            ax = axes[0][i]
            for mode in MODES:
                cost = next((a["cost"] for a in data["arms"]
                            if a["room"] == room and a["mode"] == mode), {})
                frame_ms_avg = cost.get("frame_ms_avg")
                xs, ys = [], []
                for a in data["arms"]:
                    if a["room"] != room or a["mode"] != mode:
                        continue
                    for pt in a.get("curve", []):
                        n = pt.get("n")
                        rms = pt.get("rms")
                        if n is None or rms is None:
                            continue
                        x = n if xkey == "n" else (
                            n * frame_ms_avg if frame_ms_avg else None)
                        if x is None or x <= 0 or rms <= 0:
                            continue
                        xs.append(x)
                        ys.append(rms)
                if not xs:
                    continue
                pairs = sorted(zip(xs, ys))
                xs, ys = zip(*pairs)
                st = STYLE[mode]
                ax.plot(xs, ys, color=st["color"], linestyle=st["linestyle"],
                        marker=st["marker"], markerfacecolor=st["markerfacecolor"],
                        markeredgecolor=st["color"], linewidth=st["linewidth"])
                ax.annotate(st["label"], xy=(xs[-1], ys[-1]),
                            xytext=(6, 0), textcoords="offset points",
                            color=st["color"], fontsize=9, va="center")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("RMS vs pinned photo golden")
            ax.set_title("%s\n%s@%s  referee: pinned photo golden"
                         % (room, data["git"].get("branch"), data["git"].get("sha")))
        fig.tight_layout()
        out = os.path.join(rundir, fname)
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print("wrote %s" % out)


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", choices=("01", "10"), default="01",
                    help="order the two modes are visited, so run-order "
                         "drift is visible (default %(default)s)")
    ap.add_argument("--rooms", default=",".join(ROOMS),
                    help="comma list of vantage names (default: all)")
    ap.add_argument("--skip-build", action="store_true",
                    help="capture with the binaries already in ./bin")
    ap.add_argument("--settle", type=float, default=ci.SETTLE_DEFAULT,
                    help="s of stillness before the seat is considered "
                         "ready (default %(default)s)")
    ap.add_argument("--golden", default=None,
                    help="run-dir name under screenshots/ci/ to diff "
                         "against (default: ci.read_golden())")
    ap.add_argument("--golden-dir", default=None,
                    help="directory of <room>.png to diff against, instead "
                         "of a screenshots/ci run dir")
    ap.add_argument("--make-golden", action="store_true",
                    help="shoot a fresh photo-mode golden per room UNDER "
                         "SWEEP CONDITIONS first, and use it. Required "
                         "whenever the build, HUD state or wielded item "
                         "differ from the pinned CI golden — all of which "
                         "land inside the compared crop.")
    ap.add_argument("--golden-only", action="store_true",
                    help="shoot the fresh golden and stop")
    ap.add_argument("--no-seat", action="store_true",
                    help="use the client already running instead of "
                         "restarting the seat")
    args = ap.parse_args(argv)

    rooms = [r.strip() for r in args.rooms.split(",") if r.strip()]
    order = [int(c) for c in args.order]

    git = ci.git_state()
    if git["dirty"]:
        print("!" * 70)
        print("!! WORKING TREE IS DIRTY. This sweep is SCRATCH, not evidence.")
        print("!! Captures map to shas; %s does not name a commit." % git["tag"])
        print("!" * 70)

    run_id = "%s_%s_order%s" % (
        time.strftime("%Y%m%d-%H%M%S", time.gmtime()), git["tag"], args.order)
    rundir = os.path.join(SWEEP_ROOT, run_id)
    os.makedirs(rundir, exist_ok=True)
    print("run dir: %s" % rundir)

    if not args.no_seat:
        if not args.skip_build:
            print("building: %s" % " ".join(ci.BUILD_CMD))
            r = subprocess.run(ci.BUILD_CMD, cwd=REPO)
            if r.returncode != 0:
                print("BUILD FAILED (%d) — sweep aborted." % r.returncode)
                return 1
        print("seat: stopping any running luanti...")
        ci.stop_seat()
        ci.pin_conf()
        print("seat: starting server + client")
        ci.start_seat(rundir)
        if not ci.wait_for_client():
            print("CLIENT NOT READY — sweep aborted.")
            return 1
        print("seat: client ready")

    caps = lab.fps_caps()
    if not caps["pinned"]:
        print("ABORT: fps caps are not pinned (%s) — this run would measure "
              "a sleep, not a renderer." % caps)
        return 1

    btype = build_type()
    if (btype or "").lower() not in RELEASE_TYPES:
        print("ABORT: build tree is CMAKE_BUILD_TYPE=%s. environment-laws: "
              "no benchmarks from a Debug binary. Rebuild with "
              "CMAKE_BUILD_TYPE=Release bash util/ci/build.sh" % btype)
        return 1

    try:
        lab.rpc("cmd", command="set", param="time_speed 0")
    except Exception as e:
        print("ABORT: could not freeze time: %s" % e)
        return 1
    freeze = ci.do_freeze()
    print("freeze: %s" % freeze)
    if not freeze.get("deepening"):
        print("!! FREEZE FAILED: the accumulator is not deepening. Every "
              "curve below is suspect.")

    debug_tail = ""
    try:
        with open(lab.DEBUG, errors="replace") as f:
            debug_tail = f.read()
    except Exception:
        pass
    if "Failed to compile" in debug_tail:
        print("ABORT: shader compile failure in debug.txt — a raster "
              "fallback reports stats silently, as if it were the tracer.")
        return 1

    vs = lab.load_vantages()
    missing = [r for r in rooms if r not in vs]
    for r in missing:
        print("%-14s MISSING from claude_vantages.json — skipped" % r)
    rooms = [r for r in rooms if r in vs]

    golden_made = None
    if args.make_golden or args.golden_only:
        golden_dir = os.path.join(SWEEP_ROOT, "golden_%s" % git["tag"])
        print("shooting a fresh golden under sweep conditions -> %s" % golden_dir)
        golden_made = capture_golden(rooms, vs, golden_dir)
        if args.golden_only:
            print("golden only: done")
            return 0
    elif args.golden_dir:
        golden_dir = args.golden_dir
    else:
        golden_run = args.golden or ci.read_golden()
        golden_dir = os.path.join(ci.CI_DIR, golden_run) if golden_run else None
        print("!! using the pinned CI golden (%s). If this build, its HUD "
              "state or the wielded item differ from that capture, the "
              "difference is INSIDE the crop and reads as renderer noise. "
              "--make-golden is the honest option." % golden_run)
    print("golden dir: %s" % golden_dir)

    data = {"run_id": run_id, "git": git, "rooms": rooms, "order": args.order,
            "fps_caps": caps, "build_type": btype, "golden_made": golden_made,
            "freeze": freeze, "golden_dir": golden_dir,
            "sweep_dials": SWEEP_DIALS, "checkpoints": CHECKPOINTS,
            "arms": []}

    doors = ci.set_doors(True)
    data["doors_shut"] = doors
    try:
        for room in rooms:
            vantage = vs[room]
            golden_png = os.path.join(golden_dir, room + ".png") \
                if golden_dir else None
            if not golden_png or not os.path.exists(golden_png):
                print("%-14s no golden png at %s — RMS will error per point"
                     % (room, golden_png))
            for mode in order:
                print("=== arm: room=%s mode=%s ===" % (room, mode))
                try:
                    result = arm(room, mode, vantage, golden_png, rundir)
                except Exception as e:
                    result = {"room": room, "mode": mode, "error": str(e)}
                    print("%-14s mode=%s ARM FAILED: %s" % (room, mode, e))
                data["arms"].append(result)
    finally:
        slow_clock(False)  # leave the seat on the fast/measuring clock
        data["doors_reopened"] = ci.set_doors(False)

    json.dump(data, open(os.path.join(rundir, "sweep.json"), "w"), indent=2)
    write_csv(os.path.join(rundir, "sweep.csv"), data)
    write_md(os.path.join(rundir, "sweep.md"), data)
    print_tables(data)
    plot(rundir, data)

    print("\nwrote %s" % rundir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
