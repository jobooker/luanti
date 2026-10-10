#!/usr/bin/env python3
"""claude_play_log -- a passive record of every picture restart while John
plays (2026-10-09). He reported "underground ... the light just regenerates
as if i moved even if i haven't"; the session left no record of which
restart cause fired. Started by claude_look.sh beside the client; reads only
the client's own 1 Hz stats file (no GPU, no game interaction) and stops
when the client does. One line per second in which the picture restarted
or the scene changed: wall-clock time, the cause counters that moved, what
changed (grid box, light list, far blocks), the grid origin (the camera is
~64 cells from it), fps.

  /tmp/claude_look/restarts-YYYYMMDD-HHMMSS.log
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(os.path.dirname(HERE), "claude_stats.json")
CAUSE = ["restart", "sky", "sun/light", "grid re-centre/teleport", "camera moved", "other"]
WATCH = ("grid_origin", "grid_blocks", "area_emitters", "area_hash", "emitters", "point_emitters",
         "far_db_blocks", "summary_blocks")


def main():
    client_pid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    out = open(os.path.join("/tmp/claude_look", time.strftime("restarts-%Y%m%d-%H%M%S.log")), "w")
    prev, prev_mt = None, 0
    while client_pid is None or os.path.exists("/proc/%d" % client_pid):
        time.sleep(0.25)
        try:
            mt = os.path.getmtime(STATS)
            if mt == prev_mt:
                continue
            prev_mt = mt
            st = json.load(open(STATS))
        except Exception:
            continue
        if prev is not None:
            dr = [b - a for a, b in zip(prev.get("reset_why", []), st.get("reset_why", []))]
            why = ", ".join("%s x%d" % (CAUSE[min(i, 5)], n) for i, n in enumerate(dr) if n > 0)
            ch = [k for k in WATCH if st.get(k) != prev.get(k)]
            dropped = st.get("still_frames", 0) < prev.get("still_frames", 0)
            if why or ch or dropped:
                out.write("%s still %6d -> %6d | fps %5.1f | origin %s | restarts: %s | changed: %s\n" % (
                    time.strftime("%H:%M:%S"), int(prev.get("still_frames", 0)), int(st.get("still_frames", 0)),
                    st.get("fps", 0), st.get("grid_origin"), why or "-",
                    "; ".join("%s %s->%s" % (k, prev.get(k), st.get(k)) for k in ch) or "-"))
                out.flush()
        prev = st


if __name__ == "__main__":
    main()
