#!/usr/bin/env python3
"""claude_shoot — one frame-exact capture anywhere, for looking at.

Freezes time, applies CANONICAL_DIALS plus --dial overrides, stands the
player at --pos/--yaw/--pitch at --time, and fires the client's shutter
at exactly --frames frames. Prints the PNG path. No referee, no verdict.

  python3 util/claude_shoot.py --pos 146.5 9.5 123.5 --yaw 270 --pitch 4 \
      --time 0.235 --frames 1500 --dial claude_nee=1 --name forest
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_ci as ci                       # noqa: E402
import claude_lab as lab                     # noqa: E402
import claude_denoise_price as price         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", nargs=3, type=float, required=True)
    ap.add_argument("--yaw", type=float, required=True)
    ap.add_argument("--pitch", type=float, required=True)
    ap.add_argument("--time", type=float, default=0.5)
    # the truth guard shoots poses of a playtest, whose world time is the
    # frozen snapshot's: setting --time would move the sun (2026-10-10)
    ap.add_argument("--keep-time", action="store_true", help="do not set the time of day")
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--dial", action="append", default=[],
                    help="name=value, repeatable")
    ap.add_argument("--name", default="shot")
    ap.add_argument("--skip-seat", action="store_true")
    # PIN THE POSE ON THE CLIENT (claude_path, one key): a server teleport
    # lands within the aim tolerance, not on the same pixel -- two
    # 4096-frame references of the plains differed by 2 px of yaw and a
    # sliver of height (judge validation, 2026-10-06), which a
    # reference-based judge reads as a big difference. Pinned, the pose is
    # the same float every frame of every shot.
    ap.add_argument("--pin", action="store_true")
    # PLAY PHYSICS (2026-10-08, the scoreboard): push only the capture
    # mechanics and leave the renderer at the game's own defaults, the
    # physics as played (CI's photo state switches the air off, among
    # others). --dial still overrides; a caller comparing arms must spell
    # every dial any arm sets, because an unpushed key does not revert.
    ap.add_argument("--play", action="store_true")
    # the linear accumulated radiance, read BEFORE the pin is released (an
    # unpinned camera can restart the accumulation)
    ap.add_argument("--accum-dump", help="path prefix: write <p>.f32 + <p>.json")
    # the camera the shot was taken with (camera.json: grid origin, position,
    # basis), exported while the pin still holds it, for a referee that has
    # to know which world point each pixel sees (cave sun patch, 2026-10-10)
    ap.add_argument("--export-trace", help="dir: claude_export_trace (camera.json + grids)")
    args = ap.parse_args()
    if not args.skip_seat:
        price.start_seat()
    # three tries: a freshly settling seat restarts its picture now and
    # then, and one restart during the check refused whole playtests
    # (2026-10-08/09); a seat that never settles still refuses
    for _try in range(3):
        fr = ci.do_freeze()
        if fr.get("deepening"):
            break
        time.sleep(10)
    if not fr.get("deepening"):
        print("REFUSED: time did not freeze: %r" % (fr,))
        return 2
    dials = dict(ci.CANONICAL_DIALS)
    if args.play:
        dials = {k: v for k, v in ci.CANONICAL_DIALS.items()
                 if k in ("claude_input_lock", "claude_show_hud", "claude_show_chat",
                          "claude_stats", "recent_chat_messages", "node_highlighting",
                          "claude_grid_debug", "claude_grid_follow", "claude_rng", "claude_view",
                          # the capture PROVES these were applied (ci.PROVEN_DIALS);
                          # bounces 24 and sky_uniform 0 are the game defaults, and
                          # a caller sets nee itself (the canonical 0 is photo mode)
                          "claude_bounces", "claude_sky_uniform", "claude_nee")}
    for kv in args.dial:
        k, _, v = kv.partition("=")
        dials[k.strip()] = float(v)
    vant = {"pos": list(args.pos), "yaw": args.yaw, "pitch": args.pitch,
            "time": args.time}
    if args.keep_time:
        del vant["time"]
    vs = lab.load_vantages()
    if args.pin:
        # the capture resets by teleporting away and back, which a pinned
        # camera never does: reset explicitly instead, after the arm's
        # dials are on (capture calls this right after pushing them), and
        # prove it with the client's reset counter
        def pinned_reset(vantage, park):
            # THE SERVER MUST KNOW WHERE THE CAMERA IS (2026-10-06): the
            # teleport-away-and-back this replaces also moved the SERVER's
            # player, and the server sends blocks around that position. A
            # pinned camera with no teleport sat in an empty grid wherever
            # the last shot had not been (plains: grid_solid 0 from 14:45 on,
            # unnoticed by every shot until an ML export view found it).
            # Teleport, then wait for the blocks to stop arriving.
            lab.goto(vantage)
            # THE SAME SCENE EVERY TIME (2026-10-08): "wait for the blocks
            # to stop arriving" let the client's view history decide which
            # blocks, and so which lights, the grid held (the torch room:
            # 9 area lights for its truth, 11 later). load_scene re-centres
            # the grid on this pose and has the server send its whole box.
            ls = lab.load_scene(args.pos, args.yaw, args.pitch, pf)
            print("scene: %s" % json.dumps(ls), file=sys.stderr, flush=True)
            if not ls.get("ok"):
                return "scene did not load: %s" % ls.get("error")
            before = (lab.read_stats() or {}).get("accum_resets")
            with open(lab.PATCH, "w") as f:
                f.write("claude_reset_accum = %d\n" % time.time_ns())
            t0 = time.time()
            while time.time() - t0 < 5:
                time.sleep(0.2)
                now = (lab.read_stats() or {}).get("accum_resets")
                if before is not None and now is not None and now > before:
                    return None
            return "pinned reset not seen (accum_resets %s)" % before
        ci.reset_accumulation = pinned_reset
        pf = os.path.join("/tmp", "claude_pin_%d.txt" % os.getpid())
        with open(pf, "w") as f:
            f.write("0 %r %r %r %r %r\n" % (args.pos[0], args.pos[1], args.pos[2],
                                           args.yaw, args.pitch))
        with open(lab.PATCH, "w") as f:
            f.write("claude_path = %s\n" % pf)
        time.sleep(1.5)
    rundir = os.path.join(ci.REPO, "screenshots", "shoot",
                          time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(rundir, exist_ok=True)
    png, cap = ci.capture({"name": args.name}, vant, vs["furnace-050"],
                          dials, rundir, args.frames, vantage_name=args.name)
    if args.accum_dump:
        # an old dump of the same name made the wait below return at once and
        # the OLD numbers be read (2026-10-09): remove it first
        for ext in (".json", ".f32"):
            if os.path.exists(args.accum_dump + ext):
                os.remove(args.accum_dump + ext)
        with open(lab.PATCH, "w") as f:
            f.write("claude_accum_dump = %s\n" % args.accum_dump)
        t0 = time.time()
        while time.time() - t0 < 15 and not os.path.exists(args.accum_dump + ".json"):
            time.sleep(0.25)
        if not os.path.exists(args.accum_dump + ".json"):
            print("REFUSED: the linear dump did not arrive")
            return 2
    if args.export_trace:
        cj = os.path.join(args.export_trace, "camera.json")
        if os.path.exists(cj):
            os.remove(cj)
        with open(lab.PATCH, "w") as f:
            f.write("claude_export_trace = %s\n" % args.export_trace)
        t0 = time.time()
        while time.time() - t0 < 15 and not os.path.exists(cj):
            time.sleep(0.25)
        if not os.path.exists(cj):
            print("REFUSED: the trace export did not arrive")
            return 2
    if args.pin:
        with open(lab.PATCH, "w") as f:
            f.write("claude_path = 0\n")
    if cap.get("reset_error"):
        # an unproven reset means the frame may hold the previous shot
        print("REFUSED: %s" % cap["reset_error"])
        return 2
    # an empty bubble is not a picture of the place (CI asserts this; the
    # shoot tool did not, and an afternoon of plains shots were empty)
    g = cap.get("grid") or {}
    try:
        solid_now = json.load(open(png.replace(".png", ".capture.json")))["stats"].get("grid_solid")
    except Exception:
        solid_now = None
    if not g.get("ok", True) or solid_now == 0:
        print("REFUSED: empty or missing grid (grid %s, grid_solid at the shutter %s)"
              % (g.get("error") or g.get("grid_solid"), solid_now))
        return 2
    # A STILL SHOT IS STILL (2026-10-07): the doorway-room vantage stood the
    # player half a metre above the floor; it fell and was re-pinned all
    # through a 4096-frame "truth" that held 61 accumulated frames. Two
    # rules: the frames asked for were accumulated, and the camera did not
    # drift (sub-threshold drift blends instead of restarting).
    try:
        st = json.load(open(png.replace(".png", ".capture.json")))["stats"]
    except Exception:
        st = {}
    sf, drift = st.get("still_frames"), st.get("still_drift")
    if sf is not None and sf < args.frames:
        print("REFUSED: accumulated %s of %d frames (restarts by cause %s)"
              % (sf, args.frames, st.get("reset_why")))
        return 2
    if drift is not None and drift > 0.01:
        print("REFUSED: camera drifted %.3f units during the shot" % drift)
        return 2
    print("frames", (cap.get("settle") or {}).get("still_frames"))
    print(png)
    return 0


if __name__ == "__main__":
    # the GPU lock for this tool's whole life (children inherit it): see util/claude_gpu_lock.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import claude_gpu_lock
    claude_gpu_lock.hold('util/claude_shoot.py')
    sys.exit(main())
