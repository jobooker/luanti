#!/usr/bin/env python3
"""Far-data orchestrator: emerge + sample terrain around a center via
the claude_bridge, drop JSON files where the client ingests them
(<minetest>/claude_far/). Requires the far ops (claude_bridge_far.lua)
appended to the bridge mod and the server restarted.

Usage: claude_far_fetch.py <center_x> <center_z> <radius_m> [ybot] [ytop]
Blocks are 16 m; y range given in BLOCK coords (default -1..8 = -16..144m).
Idempotent: skips columns whose output file already exists.
"""
import sys, os, json, time, subprocess

import os as _os
WORLD = _os.environ.get("CLAUDE_WORLD", "/opt/luanti/config/worlds/world")
FARDIR = os.path.expanduser(
    "~/Library/Application Support/minetest/claude_far")
CHUNK = int(os.environ.get("CLAUDE_FAR_CHUNK", "24"))
THROTTLE = float(os.environ.get("CLAUDE_FAR_THROTTLE", "0"))  # s between calls


def bridge(op, **kw):
    req = {"id": "far%d" % time.time_ns(), "op": op}
    req.update(kw)
    subprocess.run(["ssh", "beelink", "cat > %s/claude_cmd.json" % WORLD],
        input=json.dumps(req).encode(), check=True)
    deadline = time.time() + 60
    while time.time() < deadline:
        time.sleep(0.7)
        raw = subprocess.run(
            ["ssh", "beelink", "cat %s/claude_out.json" % WORLD],
            capture_output=True).stdout
        try:
            out = json.loads(raw)
        except Exception:
            continue
        if out.get("id") == req["id"]:
            if not out.get("ok"):
                raise RuntimeError(out.get("error"))
            return out["result"]
    raise RuntimeError("bridge timeout on " + op)


def main():
    cx, cz, radius = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ybot = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    ytop = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    os.makedirs(FARDIR, exist_ok=True)
    b0x, b0z = cx // 16, cz // 16
    rb = radius // 16
    # emerge in strips (async server-side; sampling tolerates gaps —
    # rerun the script to fill anything that emerged late)
    for sz in range(-rb, rb + 1, 8):
        bridge("emerge_region",
            p1={"x": (b0x - rb) * 16, "y": ybot * 16, "z": (b0z + sz) * 16},
            p2={"x": (b0x + rb) * 16 + 15, "y": ytop * 16 + 15,
                "z": (b0z + min(sz + 7, rb)) * 16 + 15})
    print("emerge kicked; sampling…")
    cols = [(b0x + dx, b0z + dz)
            for dx in range(-rb, rb + 1) for dz in range(-rb, rb + 1)]
    done = 0
    for i in range(0, len(cols), CHUNK):
        chunk = cols[i:i + CHUNK]
        fname = os.path.join(FARDIR, "far_%d_%d_%d.json"
                % (chunk[0][0], chunk[0][1], len(chunk)))
        if os.path.exists(fname):
            done += len(chunk)
            continue
        if THROTTLE > 0:
            time.sleep(THROTTLE)  # let the server tick breathe (block
            # streaming starved during unthrottled runs — John's
            # "LOD #1 is basically not there", 2026-08-12)
        res = bridge("sample_columns", cols=[list(c) for c in chunk],
                ybot=ybot, ytop=ytop)
        blocks = res.get("blocks") or []
        with open(fname + ".tmp", "w") as f:
            json.dump({"blocks": blocks}, f, separators=(",", ":"))
        os.rename(fname + ".tmp", fname)
        done += len(chunk)
        print("\r%d/%d columns, last chunk %d blocks   "
              % (done, len(cols), len(blocks)), end="", flush=True)
    print("\ndone; client ingests within ~5 s")


if __name__ == "__main__":
    main()
