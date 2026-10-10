#!/usr/bin/env python3
"""claude_freeze_check -- does a frozen world hold still, and move again when released?

Demo mode (util/claude_seat_world.py, claude_freeze in the engine, OPS.freeze
in the bridge) claims a test seat's world does not change by itself. This
tests the claim on a fresh per-run copy of the snapshot:

  1. the engine half is on from the server's first step (the copy's marker);
  2. OPS.freeze{on=true} holds the mods' globalsteps;
  3. a sand block is placed in the air and the builtin falling check run on
     it (what a player's dig next to it would do): it becomes a falling
     entity; a water source is placed on the ground;
  4. frozen for HOLD_S: the entity's position (server), the sand column, the
     water's neighbourhood (OPS.scan counts) and a screenshot of the spot
     (the client: it moves entities by itself between server updates) are
     sampled;
  5. released for the same time: the same samples. The sand must land and
     the water must spread.

  python3 util/claude_freeze_check.py [--hold 15]
Prints one JSON report (and the screenshot paths). Takes the GPU lock.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SAND = "mcl_core:sand"
WATER = "mcl_core:water_source"
# an open spot: the "backup" playtest scenario's start, looking +z
START = (5.0, 8.5, -20.0)


def johns_play_running():
    """a luanti client NOT under gamescope = a human seat (claude_look.sh)"""
    r = subprocess.run(["pgrep", "-x", "luanti"], capture_output=True, text=True)
    for pid in r.stdout.split():
        try:
            ppid = open("/proc/%s/stat" % pid).read().rsplit(")", 1)[1].split()[1]
            parent = open("/proc/%s/comm" % ppid).read().strip()
        except OSError:
            continue
        if not parent.startswith("gamescope"):
            return int(pid)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=float, default=15.0, help="seconds frozen, then seconds released")
    ap.add_argument("--snapshot", help="snapshot to copy (default the newest)")
    ap.add_argument("--no-hold-motion", action="store_true",
                    help="freeze without zeroing entity motion (the A/B for what the client draws)")
    a = ap.parse_args()
    import claude_gpu_lock
    claude_gpu_lock.hold("util/claude_freeze_check.py")
    pid = johns_play_running()
    if pid:
        sys.exit("REFUSED: a human seat is running (luanti pid %d, not under gamescope)" % pid)
    import claude_seat_world as SW
    world = SW.fresh(a.snapshot)
    os.environ[SW.ENV] = world
    import claude_lab as lab
    import claude_denoise_price as price
    rep = {"world": world}
    price.start_seat()
    try:
        rep["at_start"] = lab.rpc("freeze")
        rep["freeze_on"] = lab.rpc("freeze", on=True, hold_motion=not a.no_hold_motion)
        x, y, z = START
        # the camera, pinned, looking along +z and a little up at the drop
        pin = "/tmp/claude_freeze_check.pin"
        open(pin, "w").write("0 %r %r %r 0 12\n" % (x, y, z))
        lab.rpc("tp", pos={"x": x, "y": y, "z": z}, yaw=0, pitch=12)
        open(lab.PATCH, "w").write("claude_path = %s\nclaude_stats = 1\n" % pin)
        time.sleep(4)
        # a spot ahead with ground below and 8 nodes of air above it
        spot = None
        for dz in range(6, 16):
            for dx in (0, 2, -2, 4, -4):
                cx, cz = int(round(x)) + dx, int(round(z)) + dz
                g = lab.rpc("probe", x=cx, z=cz, ytop=int(y) + 12, ybot=int(y) - 12)
                if g.get("y") is False:
                    continue
                gy = g["y"]
                col = lab.rpc("scan", p1={"x": cx, "y": gy + 1, "z": cz}, p2={"x": cx, "y": gy + 8, "z": cz})
                if col["counts"].get("air") == 8:
                    spot = (cx, gy, cz)
                    break
            if spot:
                break
        if not spot:
            raise RuntimeError("no open spot ahead of %r" % (START,))
        cx, gy, cz = spot
        sand = {"x": cx, "y": gy + 6, "z": cz}
        col = {"p1": {"x": cx, "y": gy + 1, "z": cz}, "p2": {"x": cx, "y": gy + 7, "z": cz}}
        # water two nodes to the side, on the ground, if that is open too
        wx = cx + 3
        wg = lab.rpc("probe", x=wx, z=cz, ytop=int(y) + 12, ybot=int(y) - 12)
        water = {"x": wx, "y": wg["y"] + 1, "z": cz}
        wbox = {"p1": {"x": wx - 3, "y": water["y"] - 1, "z": cz - 3},
                "p2": {"x": wx + 3, "y": water["y"], "z": cz + 3}}
        rep["spot"] = {"ground": spot, "sand": sand, "water": water}
        rep["water_box_before"] = lab.rpc("scan", **wbox)["counts"]
        # aim the pinned camera at the drop (eye 1.5 m above the feet)
        import math
        pitch = math.degrees(math.atan2(sand["y"] + 0.5 - (y + 1.5), cz + 0.0 - z))
        open(pin, "w").write("0 %r %r %r 0 %r\n" % (x, y, z, pitch))
        lab.rpc("tp", pos={"x": x, "y": y, "z": z}, yaw=0, pitch=pitch)
        rep["camera_pitch"] = pitch
        time.sleep(2)
        lab.rpc("set_node", pos=sand, name=SAND)
        rep["after_falling_check"] = lab.rpc("check_falling", pos=sand)
        lab.rpc("set_node", pos=water, name=WATER)

        def sample(tag, t0):
            ents = [e for e in lab.rpc("objects", pos=sand, radius=10) or []
                    if e.get("name") == "__builtin:falling_node"]
            s = {"t": round(time.time() - t0, 1),
                 "falling_entities": [[round(e["pos"]["x"], 3), round(e["pos"]["y"], 3), round(e["pos"]["z"], 3)]
                                      for e in ents],
                 "sand_column": lab.rpc("scan", **col)["names"],
                 "water_box": lab.rpc("scan", **wbox)["counts"]}
            print(tag, json.dumps(s), flush=True)
            return s

        def shots(tag):
            p = lab.shot("freezecheck-%s-%d" % (tag, time.time()), settle=0.0, record=False)
            return p

        t0 = time.time()
        rep["frozen"] = [sample("frozen", t0)]
        rep["frozen_shots"] = [shots("frozen-early")]
        rep["frozen"].append(sample("frozen", t0))
        while time.time() - t0 < a.hold:
            time.sleep(min(5.0, max(0.1, a.hold - (time.time() - t0))))
            rep["frozen"].append(sample("frozen", t0))
        rep["frozen_shots"].append(shots("frozen-late"))
        rep["freeze_state_late"] = lab.rpc("freeze")
        rep["freeze_off"] = lab.rpc("freeze", on=False)
        t1 = time.time()
        rep["released"] = [sample("released", t1)]
        while time.time() - t1 < a.hold:
            time.sleep(min(5.0, max(0.1, a.hold - (time.time() - t1))))
            rep["released"].append(sample("released", t1))
        rep["released_shots"] = [shots("released")]
        fr, rl = rep["frozen"], rep["released"]
        rep["verdict"] = {
            "frozen_entity_still": all(s["falling_entities"] == fr[0]["falling_entities"] for s in fr)
            and bool(fr[0]["falling_entities"]),
            "frozen_column_still": all(s["sand_column"] == fr[0]["sand_column"] for s in fr),
            "frozen_water_still": all(s["water_box"] == fr[0]["water_box"] for s in fr),
            "released_sand_landed": not rl[-1]["falling_entities"] and SAND in rl[-1]["sand_column"],
            "released_water_spread": rl[-1]["water_box"] != fr[-1]["water_box"],
        }
    finally:
        try:
            lab.rpc("freeze", on=False)
        except Exception as e:
            print("unfreeze failed: %s" % e)
        open(lab.PATCH, "w").write("claude_path = 0\n")
    log = "/tmp/claude_denoise_price.server.log"
    rep["server_log"] = [l.rstrip() for l in open(log, errors="replace")
                         if "claude_freeze" in l or "FROZEN" in l or "unfrozen" in l
                         or "demo world" in l or "motion given back" in l]
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
