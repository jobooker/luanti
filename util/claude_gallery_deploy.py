#!/usr/bin/env python3
"""Deploy the interior-program gallery into a running world.

The gallery (cozy house, two furnace referee rooms, Cornell box, the
connecting hall) is scripted content, not hand-built: this is the one
command that reproduces it on any seat. Coordinates are the ratified
layout from luanti-docs/interior-program-plan.md phase 1 — rooms on the
z=0 line, doors facing south, floors at y=8, walk level y=9.

Prereq: a server running this world with the assembled claude_bridge
mod (util/claude_seat_assemble.sh).

Usage: util/claude_gallery_deploy.py [--skip-clear]
Env: CLAUDE_WORLD (defaults to <repo>/worlds/gallery)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from claude_lab import rpc  # noqa: E402  (shares the file-RPC channel)

FLOOR = 8          # mgflat surface top; rooms sit on it
WALK = FLOOR + 1

# Vegetation clear box — light control, per the deployed-gallery record.
CLEAR = dict(p1={"x": -56, "y": WALK, "z": -40},
             p2={"x": 64, "y": 30, "z": 40})


def emerge(p1, p2):
    rpc("emerge_region", p1=p1, p2=p2)
    time.sleep(3.0)  # emerge is async; give the mapgen thread a beat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-clear", action="store_true",
                    help="leave surface vegetation alone (faster re-runs)")
    args = ap.parse_args()

    print("ping:", rpc("ping"))

    # Emerge everything the build touches before any set_node: the
    # bridge writes into loaded blocks only, and a silent no-op here
    # looks exactly like a successful build.
    print("emerging build region...")
    emerge({"x": -60, "y": 0, "z": -44}, {"x": 68, "y": 40, "z": 44})

    if not args.skip_clear:
        # Clear in y-slabs; OPS.fill caps at 60k nodes per call.
        print("clearing vegetation...")
        for y in range(WALK, 31):
            r = rpc("fill", p1={**CLEAR["p1"], "y": y},
                    p2={**CLEAR["p2"], "y": y}, name="air")
            if y == WALK:
                print("  slab y=%d: %s" % (y, r))

    print("cozy house @ (0,%d,0)..." % FLOOR)
    r = rpc("cozy", pos={"x": 0, "y": FLOOR, "z": 0})
    print("  ", r)
    if r.get("missing"):
        print("  !! MISSING NODE NAMES:", r["missing"])

    print("furnace 050 @ x17, furnace 073 @ x30...")
    print("  ", rpc("furnace", pos={"x": 17, "y": FLOOR, "z": 0},
                    size=5, variant="050"))
    print("  ", rpc("furnace", pos={"x": 30, "y": FLOOR, "z": 0},
                    size=5, variant="073"))

    print("cornell @ x43...")
    print("  ", rpc("cornell", pos={"x": 43, "y": FLOOR, "z": 0}, size=7))

    print("hall...")
    print("  ", rpc("hall", y=FLOOR, x0=-2, x1=56,
                    doors=[5, 20, 33, 47], torches=[10, 27, 44]))

    print("\ndone. vantages: util/claude_vantages.json")


if __name__ == "__main__":
    main()
