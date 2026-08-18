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
# Widened 2026-08-15 (roadmap 1b) to cover the sky/sun/glass referee
# cluster (cave pair + sky-furnace pad), which sits >=70 nodes north of
# the cabin/Cornell line on purpose — see claude_ci.py ROOM_BOXES.
# Widened again 2026-08-18 (roadmap coverage 4) for the transparency
# referee pair, which sits at z >= 170 -- far enough north that its
# emissive walls are outside the 128^3 trace bubble of every CI vantage,
# so a new lit room cannot displace a referee room's own lights in the
# cap-16 area-emitter list.
CLEAR = dict(p1={"x": -56, "y": WALK, "z": -40},
             p2={"x": 112, "y": 30, "z": 182})


def emerge(p1, p2):
    rpc("emerge_region", p1=p1, p2=p2)
    time.sleep(3.0)  # emerge is async; give the mapgen thread a beat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-clear", action="store_true",
                    help="leave surface vegetation alone (faster re-runs)")
    ap.add_argument("--emerge-only", action="store_true",
                    help="force-load the build region and stop -- no "
                         "clear, no room rebuilds. For a fresh server "
                         "that has nothing loaded yet (claude_ci "
                         "--skip-deploy): a bridge op like OPS.door "
                         "silently no-ops on an unloaded chunk, and after "
                         "a server restart NOTHING is loaded until "
                         "something forces it.")
    args = ap.parse_args()

    print("ping:", rpc("ping"))

    # Emerge everything the build touches before any set_node: the
    # bridge writes into loaded blocks only, and a silent no-op here
    # looks exactly like a successful build.
    print("emerging build region...")
    emerge({"x": -60, "y": 0, "z": -44}, {"x": 84, "y": 40, "z": 132})
    emerge({"x": -4, "y": 0, "z": 164}, {"x": 116, "y": 40, "z": 184})
    # the sealed-plank referee (2026-08-18). No CLEAR pass out here on
    # purpose: the room is a closed box and nothing outside it can reach
    # the interior, so surface vegetation is not a light-control problem
    # the way it is around the open sky pad.
    emerge({"x": -8, "y": 0, "z": 244}, {"x": 20, "y": 30, "z": 268})

    if args.emerge_only:
        print("\n--emerge-only: chunks loaded, stopping (no clear, no rebuild)")
        return

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

    # Sky/sun/glass referee cluster (roadmap 1b, gallery phase 2). North
    # of the cabin/Cornell line by >=70 nodes on purpose (measured.md
    # "Rung 2" landmine 2 / the handoff's distance constraint): these
    # rooms are referees for 2b's sky and sun terms, and must not sit in
    # the 128^3 bubble of any existing CI vantage, nor let the open
    # sky-furnace pad show up in the cabin's or Cornell's own capture.
    # No hall — the mgflat plain between here and the cabin cluster is
    # already flat and walkable, so nothing needs building to connect
    # them; each room gets its own walk-in door same as furnace/Cornell.
    print("cave-skylight @ (10,%d,85), opening glazed..." % FLOOR)
    print("  ", rpc("skycave", pos={"x": 10, "y": FLOOR, "z": 85},
                    sx=7, sy=5, sz=7, glazed=True))
    print("cave-glass @ (30,%d,85), opening plugged (opaque, by design)..."
          % FLOOR)
    print("  ", rpc("skycave", pos={"x": 30, "y": FLOOR, "z": 85},
                    sx=7, sy=5, sz=7, glazed=False))
    print("sky-furnace-050 pad @ (60,%d,110)..." % FLOOR)
    print("  ", rpc("skypad", center={"x": 60, "y": FLOOR, "z": 110},
                    half=12, fence=13))

    # Transparency referee pair (roadmap coverage 4, 2026-08-18). Built
    # with an OPAQUE partition, which is what the world holds between
    # measurements: util/claude_glass_probe.py swaps the pane per arm and
    # puts it back. So the standing world is the BEFORE half of the A/B,
    # and a probe that dies half way leaves a sealed, honest room rather
    # than a half-glazed one.
    print("glasspair @ (0,%d,170), partition opaque..." % FLOOR)
    print("  ", rpc("glasspair", pos={"x": 0, "y": FLOOR, "z": 170},
                    s=5, pane="opaque"))
    print("glassfurnace @ (100,%d,170), slab opaque..." % FLOOR)
    print("  ", rpc("glassfurnace", pos={"x": 100, "y": FLOOR, "z": 170},
                    size=5, pane="opaque"))

    # THE SEALED-PLANK REFEREE (2026-08-18, John's request). A plank box
    # inside a shell of emitters, one node of air between them, nothing
    # lighting the interior. Any non-black pixel inside is a leak through
    # the carved relief of a baked model -- the defect b945042ee fixed
    # and that nothing in CI could see.
    #
    # z = 250 for the same reason the transparency pair is at 170, only
    # more so: 562 emissive cells inside the 128^3 bubble of another
    # vantage would join that capture's cap-16 area-emitter list. 250 is
    # 74 clear of the northernmost thing here (glasspair, z <= 176).
    print("sealed-plank @ (0,%d,250), oak, no door..." % FLOOR)
    r = rpc("sealedbox", pos={"x": 0, "y": FLOOR, "z": 250},
            sx=5, sy=4, sz=5, wood="oak")
    print("  ", {k: v for k, v in r.items() if k != "probe"})
    # THE WALL IS EXACTLY ONE NODE THICK, READ BACK OFF THE MAP.
    # A two-node wall masks the defect completely (the second node's
    # solid core covers the first node's groove), so this is the one
    # parameter that silently destroys the test -- assert it here rather
    # than trust the builder, and fail the deploy loudly if it moves.
    if r.get("error"):
        print("  !! SEALED-PLANK BUILD FAILED:", r["error"])
        raise SystemExit(2)
    plank, emit = r["plank"], r["emit"]
    for axis, names in sorted(r["probe"].items()):
        n = len(names)
        want = ([emit, "air", plank] + ["air"] * (n - 6)
                + [plank, "air", emit])
        ok = names == want
        print("  probe %s: %s%s" % (axis, "1-node wall OK" if ok else "BAD",
                                    "" if ok else " got %s want %s"
                                    % (names, want)))
        if not ok:
            raise SystemExit(
                "SEALED-PLANK WALL IS NOT ONE NODE THICK on the %s axis "
                "-- the room cannot see the defect it exists for" % axis)

    print("\ndone. vantages: util/claude_vantages.json")


if __name__ == "__main__":
    main()
