#!/usr/bin/env python3
"""claude_water_depth — does water absorb exactly as much light as it should?

WHAT THIS IS FOR, in plain terms. Since 2026-10-05 water absorbs light at
pure water's measured rate (claude_water_absorb, game.cpp
CLAUDE_WATER_ABSORB_*: Pope & Fry 1997). Absorption is a place light can
go missing, so it gets a referee with a closed-form answer.

THE ROOM is a sealed box of black concrete (albedo ~0.005) at x 200..216,
z 228..238, far outside every CI bubble. Its floor holds two 3x3 pools,
2 and 4 nodes deep, each with a glowing floor (gray186_lit). Nothing else
in the room emits and nothing in it reflects much, so a pixel that looks
down through the water at the glowing floor carries the floor's own
light, crossing ONE air/water interface and d / cos(theta_t) metres of
water.

THE MEASUREMENT is a ratio at one aim: claude_water_absorb 1 over 0.
The interface's Fresnel loss, the floor's emission and the exposure are
the same in both frames and divide out, so per channel

    ratio = exp(-a_c * L),   L = d / cos(theta_t),
    theta_t = asin(sin(theta_i) / 1.333)

with theta_i the centre ray's angle from vertical. What does NOT divide
out is small and named: light the floor sends up, the surface reflects
back down (2 % at normal incidence) and the floor reflects again. The
tolerance is sized for that, and the two depths must also agree with
each other (ln ratio scales with L), which no constant contamination can
fake.

AND NOTHING IS BELIEVED WITHOUT SIGNAL: the absorb-0 frame's centre must
be bright (the glowing floor is actually in view) and the frame's corner
dark (the room is sealed).

Usage:
  python3 util/claude_water_depth.py              # headless seat + run
  python3 util/claude_water_depth.py --skip-seat  # a seat is already up
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_ci as ci                     # noqa: E402
import claude_lab as lab                   # noqa: E402
from claude_furnace_check import aces_inverse  # noqa: E402

import numpy as np                         # noqa: E402
from PIL import Image                      # noqa: E402

# measured constants, the same numbers game.cpp uploads (Pope & Fry 1997)
ABSORB = np.array([0.2661, 0.0554, 0.0101])
IOR_WATER = 1.333

BLACK = "mcl_colorblocks:concrete_black"
WATER = "mcl_core:water_source"
GLOW = "claude_bridge:gray186_lit"
SHELL = ((200, 0, 228), (216, 15, 238))
ROOM = ((201, 9, 229), (215, 13, 237))     # air above the floor (top y 9)
PITCH = -70.0                              # 20 deg from straight down
POOLS = {
    # x range, depth; the player stands west of the pool facing east
    "d2": {"x": (204, 206), "depth": 2},
    "d4": {"x": (210, 212), "depth": 4},
}
POOL_Z = (232, 234)
SETTLE = 400            # the floor is seen directly: little noise
EXPOSURE = 0.1          # keep the floor on the curve (as the furnaces do)
PATCH = 6               # half-size of the centre patch, px
# TUNED: relative tolerance on each channel's ratio | learn by: the
# spread over repeated runs, and the contamination bound named above
REL_TOL = 0.03
# TUNED: the d4/d2 log-ratio must equal the path ratio to this
# | learn by: as above
DEPTH_LAW_TOL = 0.05
MIN_CENTRE = 40.0       # 0..255: the glowing floor must be in view
MAX_CORNER = 20.0       # 0..255: the room must be dark


def P(x, y, z):
    return {"x": x, "y": y, "z": z}


def fill(a, b, name):
    lab.rpc("fill", p1=P(*a), p2=P(*b), name=name)


def build():
    fill(SHELL[0], SHELL[1], BLACK)
    fill(ROOM[0], ROOM[1], "air")
    for p in POOLS.values():
        x0, x1 = p["x"]
        d = p["depth"]
        fill((x0, 8 - d, POOL_Z[0]), (x1, 8 - d, POOL_Z[1]), GLOW)
        fill((x0, 9 - d, POOL_Z[0]), (x1, 8, POOL_Z[1]), WATER)


def aim(p):
    # feet on the floor (top at y 9) 0.3 m west of the pool, facing east
    return {"pos": [p["x"][0] - 0.3, 9.0, (POOL_Z[0] + POOL_Z[1]) / 2.0 + 0.5],
            "yaw": 270, "pitch": PITCH}


def path_len(depth):
    ti = math.radians(90.0 + PITCH)              # from vertical
    tt = math.asin(math.sin(ti) / IOR_WATER)
    return depth / math.cos(tt)


def shoot(tag, a, absorb):
    dials = dict(ci.CANONICAL_DIALS)
    dials["claude_water_absorb"] = absorb
    dials["claude_exposure"] = EXPOSURE
    ci.push_dials(dials, "%s_%d" % (tag, time.time_ns()))
    lab.goto(a)
    ci.await_frames(SETTLE)
    png = lab.shot(tag, settle=0.5, record=False)
    st = lab.read_stats() or {}
    return png, st


def centre(png):
    im = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64)
    H, W, _ = im.shape
    c = im[H // 2 - PATCH:H // 2 + PATCH, W // 2 - PATCH:W // 2 + PATCH]
    byte = c.reshape(-1, 3).mean(0)
    lin = aces_inverse((c / 255.0) ** 2.2).reshape(-1, 3).mean(0) / EXPOSURE
    corner = im[:H // 8, :W // 8].mean()
    return byte, lin, corner


def start_seat():
    ci.stop_seat()
    ci.pin_conf()
    for tag, cmd, wait in (("server", ci.SERVER_CMD, ci.SEAT_BOOT_WAIT),
                           ("client", ci.HEADLESS_WRAP + ci.CLIENT_CMD, 0)):
        subprocess.Popen(cmd, cwd=ci.REPO,
                         stdout=open("/tmp/claude_water_depth.%s.log" % tag, "wb"),
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
        time.sleep(wait)
    if not ci.wait_for_client():
        sys.exit("client never became ready")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-seat", action="store_true")
    args = ap.parse_args()
    if not args.skip_seat:
        start_seat()
    lab.goto(aim(POOLS["d2"]))
    time.sleep(3)          # let the blocks load before building
    build()
    time.sleep(3)
    out, ok, logs = {}, True, {}
    for name, p in POOLS.items():
        a = aim(p)
        frames = {}
        for absorb in (0, 1):
            png, st = shoot("water-%s-absorb%d" % (name, absorb), a, absorb)
            seen = st.get("claude_water_absorb")
            if seen is None or int(round(float(seen))) != absorb:
                print("REFUSED: %s absorb=%d but the client reports "
                      "claude_water_absorb=%r" % (name, absorb, seen))
                return 2
            frames[absorb] = png
        b0, l0, corner0 = centre(frames[0])
        b1, l1, _ = centre(frames[1])
        if b0.mean() < MIN_CENTRE or corner0 > MAX_CORNER:
            print("REFUSED: %s no signal (centre byte %.1f, need >= %.0f; "
                  "corner %.1f, need <= %.0f) — look at %s"
                  % (name, b0.mean(), MIN_CENTRE, corner0, MAX_CORNER,
                     frames[0]))
            return 2
        L = path_len(p["depth"])
        want = np.exp(-ABSORB * L)
        got = l1 / l0
        rel = got / want - 1.0
        good = bool(np.all(np.abs(rel) <= REL_TOL))
        ok = ok and good
        logs[name] = (np.log(got), L)
        out[name] = {"depth": p["depth"], "path_m": L,
                     "ratio": got.tolist(), "want": want.tolist(),
                     "rel_err": rel.tolist(), "centre_byte_absorb0": b0.tolist(),
                     "frames": frames}
        print("%s  path %.3f m  ratio R/G/B %.4f %.4f %.4f  want %.4f %.4f "
              "%.4f  err %+.1f%% %+.1f%% %+.1f%%  %s"
              % (name, L, *got, *want, *(100 * rel),
                 "PASS" if good else "FAIL"))
    # the depth law: ln ratio scales with the path, channel by channel
    (g2, L2), (g4, L4) = logs["d2"], logs["d4"]
    law = (g4 / g2) / (L4 / L2) - 1.0
    # blue barely absorbs over these depths; its log is noise-dominated
    law_ok = abs(law[0]) <= DEPTH_LAW_TOL and abs(law[1]) <= DEPTH_LAW_TOL
    out["depth_law_err"] = law.tolist()
    print("depth law (ln ratio d4/d2 over path ratio, -1): R %+.3f G %+.3f "
          "B %+.3f (R and G scored, tol %.2f)  %s"
          % (*law, DEPTH_LAW_TOL, "PASS" if law_ok else "FAIL"))
    ok = ok and law_ok
    json.dump(out, open("/tmp/claude_water_depth.json", "w"), indent=1)
    print("GREEN" if ok else "RED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
