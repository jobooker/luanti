#!/usr/bin/env python3
"""claude_far_emerge — have the seat's server generate terrain out to a
radius, so the client's world-file reader (claude_far_world) can fold it.

The server only generates around players (max_block_generate_distance);
far terrain that was never generated is not in map.sqlite. This asks the
bridge's emerge_region op for it in strips and returns at once; the
server generates in the background and saves as it goes.

  python3 util/claude_far_emerge.py 60 60 512        # x z radius (nodes)
  python3 util/claude_far_emerge.py 60 60 512 -16 63 # and a y range
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_lab as lab                     # noqa: E402


def main():
    cx, cz, r = (int(v) for v in sys.argv[1:4])
    y0 = int(sys.argv[4]) if len(sys.argv) > 4 else -16
    y1 = int(sys.argv[5]) if len(sys.argv) > 5 else 63
    strip = 128
    n = 0
    for z in range(cz - r, cz + r, strip):
        lab.rpc("emerge_region",
                      p1={"x": cx - r, "y": y0, "z": z},
                      p2={"x": cx + r - 1, "y": y1,
                          "z": min(z + strip, cz + r) - 1})
        n += 1   # a failed call raises
    print("emerge kicked: %d strips, %d x %d nodes, y %d..%d"
          % (n, 2 * r, 2 * r, y0, y1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
