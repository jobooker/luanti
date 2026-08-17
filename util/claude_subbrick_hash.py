#!/usr/bin/env python3
"""DOES THE INCREMENTAL RE-SNAP MAINTAIN THE SUB-BRICK SUMMARY?

THE WORK, plainly. The `subbrick` branch gives every ring cell's 16^3
occupancy mask a baked 4^3 summary -- per sub-brick, EMPTY / FULL /
MIXED -- so the ray walk can cross empty space in 1/4 m jumps instead of
1/16 m ones. The summary is a BAKE. The trace grid is re-baked two ways:
a full walk of all 2.1 M cells (bootstrap and re-centre) and an
INCREMENTAL pass over just the blocks the world reported dirty (a dig, a
place, an ABM). If the incremental pass rebuilt the mask and forgot the
summary, the walk would read a stale EMPTY for a brick that now has
something in it -- a hole you can see through -- or a stale FULL for one
that is now empty -- a phantom wall. Nothing would log it and no
existing check would catch it.

WHY grid_hash CANNOT ANSWER THIS. `grid_hash` is a hash of NODE CONTENT,
combined per block. A bake that skipped the summary leaves node content
completely untouched, so grid_hash would be identical and the run would
look clean. So game.cpp exports `subbrick_hash` -- FNV-1a over the whole
512 KB summary, recomputed at the end of every bake, full or incremental
-- and this script asks the two questions that hash exists for:

  1. AN EDIT MOVES IT. Dig two nodes out of a wall inside the ring; the
     incremental path fires and the summary must CHANGE. An instrument
     that cannot fail is not evidence (physics-contract §8), and a hash
     that never moves would pass question 2 trivially.
  2. THE INCREMENTAL STATE EQUALS THE FULL-WALK STATE. Immediately
     after that incremental bake, force a FULL snapshot over the same
     origin. The summary must come back BYTE-IDENTICAL -- same hash.
     This is the check; everything else is setup.

  3. AND IT ROUND-TRIPS. Put the nodes back, let the incremental path
     fire again, and the hash must return to the value it had before
     the edit -- then force one more full walk and get the same answer.

THE WORLD IS RESTORED before the script exits, on every path, including
failure: this runs against the gallery world every CI arm is captured
in, and leaving a hole in the cosy cabin's wall would quietly re-pin
four goldens' worth of damage into the next run.

Usage:  util/claude_subbrick_hash.py [--vantage cozy-ci] [--pos X Y Z]
The SERVER and a CLIENT must already be up. Exit 0 = all three passed.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab            # noqa: E402
import claude_cost_seat as cs       # noqa: E402

# A plank wall of the cosy cabin: class 250 (it carries a 16^3 mask), so
# removing it changes real sub-brick states rather than only clearing a
# uniform 0xFF cell. Two nodes, because OPS.door writes a 2-high column.
DEFAULT_POS = {"x": 3, "y": 9, "z": 8}
# The box the world is proved to be UNCHANGED over on the way out. The
# edit writes a node back by NAME, which does not carry param2, so
# "I put it back" is a claim that needs a witness: OPS.scan hashes every
# node name in a box in a fixed order, and the hash before and after
# must match. A CI arm asserts a room hash and a silently altered cabin
# would surface as a red run days later with nothing to point at.
GUARD_BOX = ({"x": 1, "y": 9, "z": 1}, {"x": 12, "y": 11, "z": 10})

DIALS = {
    "claude_nee": 0, "claude_bounces": 24, "claude_grid_debug": 3,
    "claude_rng": 1, "claude_grid_follow": 1, "claude_models": 1,
    "claude_descend": 1, "claude_show_hud": 0, "claude_show_chat": 0,
    "claude_input_lock": 1, "claude_stats": 1,
}


def read():
    """(subbrick_hash, grid_hash, snap_seq) as the client reports them."""
    st = lab.read_stats() or {}
    return (st.get("subbrick_hash"), st.get("grid_hash"),
            st.get("grid_snap_seq"))


def wait_seq(prev_seq, what, timeout=25.0):
    """Wait for a NEW bake. Reads a COUNTER and compares two reads --
    environment-laws: a threshold on a value sampled once a second can
    only see events that last longer than a second, and a bake does
    not."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        sb, gh, seq = read()
        if seq is not None and prev_seq is not None and seq != prev_seq:
            return sb, gh, seq
        time.sleep(0.5)
    raise SystemExit("TIMEOUT: no new bake after %s (snap_seq stuck at %s)"
                     % (what, prev_seq))


def full_snapshot(prev_seq, tag):
    """Force a full walk over the same origin and wait for it.

    The tag must be UNIQUE per call: the client applies the patch file
    only when its bytes CHANGE, so re-writing the same value is a no-op
    and the wait below would then time out on a snapshot nobody asked
    for."""
    lab.doorway(claude_grid_snapshot="subbrickhash_%s" % tag)
    return wait_seq(prev_seq, "the forced full snapshot (%s)" % tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantage", default="cozy-ci")
    ap.add_argument("--pos", nargs=3, type=int, default=None,
                    help="world x y z of the 2-high column to edit")
    ap.add_argument("--out", default="subbrick_hash.json")
    a = ap.parse_args()
    pos = (dict(zip("xyz", a.pos)) if a.pos else dict(DEFAULT_POS))

    cs.require_release()
    lab.doorway(**DIALS)
    st = lab.park(a.vantage) if hasattr(lab, "park") else None
    if st is None:
        cs.park(a.vantage, dials=DIALS, snapshot_tag="subbrick_hash")
    time.sleep(2.0)

    was = lab.rpc("probe", x=pos["x"], ytop=pos["y"], ybot=pos["y"],
                  z=pos["z"])
    name = (was or {}).get("name")
    if not name or name in ("air", "ignore", "all air or unloaded"):
        raise SystemExit("nothing solid at %s to edit: %s" % (pos, was))
    print("editing (%d,%d,%d) and the node above it: %s"
          % (pos["x"], pos["y"], pos["z"], name))

    room_before = lab.rpc("scan", p1=GUARD_BOX[0], p2=GUARD_BOX[1])["hash"]
    print("room hash before: %s" % room_before)

    out = {"vantage": a.vantage, "pos": pos, "node": name,
           "room_hash_before": room_before, "steps": []}
    restored = False
    try:
        sb0, gh0, seq0 = read()
        out["steps"].append({"what": "baseline", "subbrick": sb0,
                             "grid": gh0, "seq": seq0})
        print("baseline           subbrick %s  grid %s  seq %s"
              % (sb0, gh0, seq0))

        # --- 1. an edit must MOVE the summary --------------------------
        lab.rpc("door", pos=pos, shut=False)
        sb1, gh1, seq1 = wait_seq(seq0, "the dig")
        out["steps"].append({"what": "dug (incremental)", "subbrick": sb1,
                             "grid": gh1, "seq": seq1})
        print("dug, incremental   subbrick %s  grid %s  seq %s" % (sb1, gh1, seq1))

        # --- 2. THE CHECK: incremental == full -------------------------
        sb2, gh2, seq2 = full_snapshot(seq1, "dug")
        out["steps"].append({"what": "dug (forced FULL walk)",
                             "subbrick": sb2, "grid": gh2, "seq": seq2})
        print("dug, full walk     subbrick %s  grid %s  seq %s" % (sb2, gh2, seq2))

        # --- 3. and it round-trips -------------------------------------
        lab.rpc("door", pos=pos, shut=True, name=name)
        restored = True
        sb3, gh3, seq3 = wait_seq(seq2, "the restore")
        out["steps"].append({"what": "restored (incremental)",
                             "subbrick": sb3, "grid": gh3, "seq": seq3})
        print("restored, incr     subbrick %s  grid %s  seq %s" % (sb3, gh3, seq3))
        sb4, gh4, seq4 = full_snapshot(seq3, "restored")
        out["steps"].append({"what": "restored (forced FULL walk)",
                             "subbrick": sb4, "grid": gh4, "seq": seq4})
        print("restored, full     subbrick %s  grid %s  seq %s" % (sb4, gh4, seq4))
    finally:
        if not restored:
            try:
                lab.rpc("door", pos=pos, shut=True, name=name)
                print("world restored on the way out")
            except Exception as e:      # noqa: BLE001
                print("!! COULD NOT RESTORE %s: %s" % (pos, e))

    room_after = lab.rpc("scan", p1=GUARD_BOX[0], p2=GUARD_BOX[1])["hash"]
    out["room_hash_after"] = room_after
    print("room hash after:  %s" % room_after)

    gates = [
        ("the world is back as it was (room hash round-tripped)",
         room_after == room_before),
        ("the edit MOVED the summary (the instrument can fail)",
         sb1 != sb0),
        ("INCREMENTAL == FULL WALK after the edit", sb1 == sb2),
        ("the summary ROUND-TRIPPED to its original value", sb3 == sb0),
        ("INCREMENTAL == FULL WALK after the restore", sb3 == sb4),
    ]
    print()
    for text, ok in gates:
        print("%-4s %s" % ("PASS" if ok else "FAIL", text))
    out["gates"] = [{"gate": t, "pass": bool(o)} for t, o in gates]
    out["verdict"] = "PASS" if all(o for _t, o in gates) else "FAIL"
    d = os.path.join(REPO, "screenshots", "cost")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, a.out), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwritten: %s" % os.path.join(d, a.out))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
