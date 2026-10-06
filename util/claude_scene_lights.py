#!/usr/bin/env python3
"""claude_scene_lights — two small rooms for judging real light units
(2026-10-05), far from every CI bubble (x >= 240, z >= 230):

  torch room  x 240..248, z 230..238: gray186 walls (rho 0.50), sealed,
              ONE floor torch at (244, 9, 234). Lit by nothing else.
  lava room   x 252..260, z 230..238: gray186 walls, sealed, a 3x3 lava
              pool sunk in the floor at x 255..257, z 233..235.

Aims, as claude_shoot / claude_ab --pos arguments:
  torch:  --pos 241.7 9 231.7 --yaw 315 --pitch -12
  lava:   --pos 253.7 9 231.7 --yaw 315 --pitch -25

Run with a seat up: python3 util/claude_scene_lights.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_lab as lab   # noqa: E402

P = lambda x, y, z: dict(x=x, y=y, z=z)
WALL = "claude_bridge:gray186"


def room(x0, x1, z0, z1):
    lab.rpc("fill", p1=P(x0, 7, z0), p2=P(x1, 13, z1), name=WALL)
    lab.rpc("fill", p1=P(x0 + 1, 9, z0 + 1), p2=P(x1 - 1, 12, z1 - 1), name="air")


def main():
    lab.rpc("tp", pos=P(244, 20, 234), yaw=0, pitch=0)
    time.sleep(4)
    room(240, 248, 230, 238)
    lab.rpc("set_node", pos=P(244, 9, 234), name="mcl_torches:torch")
    room(252, 260, 230, 238)
    lab.rpc("fill", p1=P(255, 8, 233), p2=P(257, 8, 235), name="mcl_core:lava_source")
    for p in ((244, 9, 234), (256, 8, 234), (241, 9, 231)):
        print(p, lab.rpc("get_node", pos=P(*p))["name"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
