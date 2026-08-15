#!/usr/bin/env python3
"""claude_rooms_check — is a referee room still a referee?

A furnace room is an analytic referee because it is SEALED, uniform and
uniformly emissive; a Cornell box is a bleed referee because its five
surfaces are exactly the colors the builder laid down. One dug node
breaks both claims silently: a single missing node at (47,11,8) leaked
daylight into Cornell for an unknown period on 2026-08-15 and only the
numbers going strange gave it away. Nothing in the harness could see it.

This walks each room's whole box through the bridge (OPS.scan), rebuilds
the SPEC from the same numbers the builders use
(util/claude_bridge_gallery.lua OPS.furnace / OPS.cornell), and prints
every position that disagrees — plus a hash of the room's node names,
which is the room's identity as one number.

  claude_rooms_check.py            # verify, name every off-spec node
  claude_rooms_check.py --repair   # re-run the builders, then verify
  claude_rooms_check.py --json     # machine-readable, for claude_ci

Doors: the builders leave a 1x2 doorway OPEN for walkability and
claude_ci plugs it before capturing. An open doorway is therefore
EXPECTED here and reported separately from a hole.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import claude_lab as lab  # noqa: E402

AIR = "air"

# The gallery layout, as claude_gallery_deploy.py deploys it.
ROOMS = [
    {"name": "furnace-050", "kind": "furnace", "pos": (17, 8, 0), "size": 5,
     "variant": "050"},
    {"name": "furnace-073", "kind": "furnace", "pos": (30, 8, 0), "size": 5,
     "variant": "073"},
    {"name": "cornell", "kind": "cornell", "pos": (43, 8, 0), "size": 7},
]


def furnace_spec(pos, size, variant):
    """OPS.furnace, in Python. Same numbers or it proves nothing."""
    ox, oy, oz = pos
    s = size
    e = s + 1
    wall = ("claude_bridge:gray221_lit" if variant == "073"
            else "claude_bridge:gray186_lit")
    spec = {}
    for x in range(ox, ox + e + 1):
        for y in range(oy, oy + e + 1):
            for z in range(oz, oz + e + 1):
                spec[(x, y, z)] = wall
    for x in range(ox + 1, ox + s + 1):
        for y in range(oy + 1, oy + s + 1):
            for z in range(oz + 1, oz + s + 1):
                spec[(x, y, z)] = AIR
    dx = ox + 1 + (s - 1) // 2
    door = [(dx, oy + 1, oz), (dx, oy + 2, oz)]
    for p in door:
        spec[p] = AIR
    return spec, set(door), (ox, oy, oz), (ox + e, oy + e, oz + e)


def cornell_spec(pos, size):
    """OPS.cornell, in Python."""
    ox, oy, oz = pos
    s = size
    e = s + 1
    spec = {}
    for x in range(ox, ox + e + 1):
        for y in range(oy, oy + e + 1):
            for z in range(oz, oz + e + 1):
                spec[(x, y, z)] = "claude_bridge:gray221"
    for y in range(oy + 1, oy + s + 1):
        for z in range(oz + 1, oz + s + 1):
            spec[(ox, y, z)] = "claude_bridge:red221"
            spec[(ox + e, y, z)] = "claude_bridge:green221"
    for x in range(ox + 1, ox + s + 1):
        for y in range(oy + 1, oy + s + 1):
            for z in range(oz + 1, oz + s + 1):
                spec[(x, y, z)] = AIR
    mid = ox + 1 + (s - 1) // 2
    mz = oz + 1 + (s - 1) // 2
    for x in range(mid - 1, mid + 2):
        for z in range(mz - 1, mz + 2):
            spec[(x, oy + e, z)] = "claude_bridge:white_lit"
    spec[(ox + 3, oy + 1, oz + 4)] = "claude_bridge:gray186"
    spec[(ox + 6, oy + 1, oz + 6)] = "claude_bridge:gray186"
    door = [(mid, oy + 1, oz), (mid, oy + 2, oz)]
    for p in door:
        spec[p] = AIR
    return spec, set(door), (ox, oy, oz), (ox + e, oy + e, oz + e)


def spec_for(room):
    if room["kind"] == "furnace":
        return furnace_spec(room["pos"], room["size"], room["variant"])
    return cornell_spec(room["pos"], room["size"])


def scan(p1, p2):
    return lab.rpc("scan", p1={"x": p1[0], "y": p1[1], "z": p1[2]},
                   p2={"x": p2[0], "y": p2[1], "z": p2[2]})


def check(room):
    spec, door, p1, p2 = spec_for(room)
    got = scan(p1, p2)
    names = got["names"]
    i = 0
    bad, doors_open = [], []
    for x in range(p1[0], p2[0] + 1):
        for y in range(p1[1], p2[1] + 1):
            for z in range(p1[2], p2[2] + 1):
                want, have = spec[(x, y, z)], names[i]
                i += 1
                if want == have:
                    continue
                rec = {"pos": [x, y, z], "want": want, "got": have}
                (doors_open if (x, y, z) in door else bad).append(rec)
    # A doorway the builder leaves open is expected; a doorway that is
    # SOLID is also fine (claude_ci plugs it). Only report it as info.
    return {"room": room["name"], "p1": list(p1), "p2": list(p2),
            "hash": got["hash"], "total": got["total"],
            "counts": got["counts"], "off_spec": bad,
            "door_state": doors_open}


def repair(room):
    if room["kind"] == "furnace":
        return lab.rpc("furnace", pos=dict(zip("xyz", room["pos"])),
                       size=room["size"], variant=room["variant"])
    return lab.rpc("cornell", pos=dict(zip("xyz", room["pos"])),
                   size=room["size"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repair", action="store_true",
                    help="re-run the room builders (idempotent), then verify")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--room", action="append", help="limit to these rooms")
    args = ap.parse_args()

    rooms = [r for r in ROOMS if not args.room or r["name"] in args.room]
    out = {"before": [], "repaired": [], "after": []}
    for room in rooms:
        out["before"].append(check(room))
    if args.repair:
        for room in rooms:
            out["repaired"].append({"room": room["name"],
                                    "result": repair(room)})
        for room in rooms:
            out["after"].append(check(room))

    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    for phase in ("before", "after"):
        if not out[phase]:
            continue
        print("=== %s repair ===" % phase if args.repair else "=== scan ===")
        for c in out[phase]:
            print("%-12s box (%s)-(%s)  %d nodes  hash %s  OFF-SPEC %d"
                  % (c["room"], c["p1"], c["p2"], c["total"], c["hash"],
                     len(c["off_spec"])))
            for b in c["off_spec"][:40]:
                print("    (%3d,%3d,%3d) want %-28s got %s"
                      % (b["pos"][0], b["pos"][1], b["pos"][2], b["want"],
                         b["got"]))
            if len(c["off_spec"]) > 40:
                print("    ... %d more" % (len(c["off_spec"]) - 40))
            for d in c["door_state"]:
                print("    doorway (%3d,%3d,%3d) is %s (builder leaves it "
                      "open; claude_ci plugs it)"
                      % (d["pos"][0], d["pos"][1], d["pos"][2], d["got"]))
    worst = out["after"] or out["before"]
    return 1 if any(c["off_spec"] for c in worst) else 0


if __name__ == "__main__":
    sys.exit(main())
