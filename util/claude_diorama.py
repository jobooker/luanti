#!/usr/bin/env python3
"""Taste MOCKUP: the cozy hearth corner composed at 16x sub-voxel
resolution from util/claude_models JSONs, rendered by a toy raycaster
with distance-falloff glow from the models' own emissive voxels.

NOT the renderer — no real transport, no shadows, no referee value.
This exists so shapes/scale/emitter placement get taste judgment
BEFORE the phase 4.5 re-land pays for real integration.

Scene = the cozy house's actual hearth quadrant (OPS.cozy layout,
interior x 1..6, z 1..6): plank floor, cobble hearth pad x1-3/z3-5,
west+north walls, furnace at (1,y,4->east... placed per cozy at
(1,6) facing east, campfire (2,4), lantern (4,1), carpet x4-6/z3-5,
crafting table (6,1), flowerpot at (6,2) for visibility.
"""
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "claude_models")
N = 16


def load(name):
    d = json.load(open(os.path.join(MODELS, name + ".json")))
    pal = d["palette"]
    v = np.array(d["voxels"], dtype=np.uint16)
    return pal, v


def rot_y(v, times):
    """Rotate a [z][y][x] grid by 90-degree steps about +y (front +z
    toward +x per step)."""
    for _ in range(times % 4):
        v = np.transpose(v, (2, 1, 0))[::-1, :, :]
    return v


class Scene:
    def __init__(self, mx, my, mz):
        self.pal = [None]
        self.pmap = {}
        self.v = np.zeros((mz * N, my * N, mx * N), dtype=np.uint16)

    def pi(self, rgb, emit=0):
        key = (tuple(rgb), emit)
        if key not in self.pmap:
            self.pal.append(dict(rgb=list(rgb), emit=emit))
            self.pmap[key] = len(self.pal) - 1
        return self.pmap[key]

    def fill_cell(self, cx, cy, cz, rgb_a, rgb_b=None, stripe=4):
        """1m cube of flat color; optional horizontal stripes."""
        a = self.pi(rgb_a)
        b = self.pi(rgb_b) if rgb_b else a
        for yy in range(N):
            idx = b if (yy // stripe) % 2 else a
            self.v[cz * N:(cz + 1) * N, cy * N + yy,
                   cx * N:(cx + 1) * N] = idx
    def place(self, name, cx, cy, cz, rot=0):
        pal, mv = load(name)
        mv = rot_y(mv, rot)
        remap = np.zeros(len(pal), dtype=np.uint16)
        for i, p in enumerate(pal):
            if p:
                remap[i] = self.pi(tuple(p["rgb"]), p["emit"])
        self.v[cz * N:(cz + 1) * N, cy * N:(cy + 1) * N,
               cx * N:(cx + 1) * N] = remap[mv]


def build():
    # 6m x 4m x 6m corner: x 0..5 (0 = west wall), z 0..5 (5 = north
    # wall row), y 0 = floor slab, y 1..3 air/furniture
    sc = Scene(6, 4, 6)
    plank = (168, 128, 74)
    plank2 = (150, 112, 62)
    spruce = (104, 78, 48)
    spruce2 = (92, 68, 42)
    cobble = (128, 126, 122)
    log = (90, 66, 40)
    for x in range(6):
        for z in range(6):
            sc.fill_cell(x, 0, z, plank, plank2, stripe=4)   # floor
    for z in range(6):                                        # west wall
        for y in (1, 2, 3):
            sc.fill_cell(0, y, z, spruce, spruce2, stripe=8)
    for x in range(6):                                        # north wall
        for y in (1, 2, 3):
            sc.fill_cell(x, y, 5, spruce, spruce2, stripe=8)
    sc.fill_cell(0, 1, 5, log)                                # corner log
    sc.fill_cell(0, 2, 5, log)
    sc.fill_cell(0, 3, 5, log)
    # hearth pad (cozy: cobble floor cells x1-3 z3-5)
    for x in (1, 2, 3):
        for z in (3, 4, 5):
            if z < 5:
                sc.fill_cell(x, 0, z, cobble, (116, 114, 110), stripe=5)
    # furniture per the cozy layout (hearth quadrant), y=1 on the floor
    sc.place("furnace_baked", 1, 1, 4, rot=1)   # against west wall, faces east
    sc.place("campfire_lit", 2, 1, 3)
    sc.place("lantern_floor", 4, 1, 1)
    sc.place("carpet_white", 4, 1, 3)
    sc.place("carpet_white", 5, 1, 3)
    sc.place("carpet_white", 4, 1, 4)
    sc.place("crafting_baked", 5, 1, 1, rot=2)
    sc.place("flowerpot_poppy", 5, 1, 2)
    sc.place("torch_baked", 3, 2, 5)            # on the north wall, high
    sc.place("bookshelf_baked", 2, 1, 5)
    sc.place("bookshelf_baked", 3, 1, 5)
    return sc


def render(sc, px=760):
    Z, Y, X = sc.v.shape
    pal_rgb = np.array([[0, 0, 0]] + [p["rgb"] for p in sc.pal[1:]],
                       dtype=np.float32)
    pal_emit = np.array([0] + [p["emit"] for p in sc.pal[1:]],
                        dtype=np.float32)
    # emitter clusters: centroid of each emissive palette region
    lights = []
    for i, p in enumerate(sc.pal):
        if p and p["emit"] > 0:
            zz, yy, xx = np.nonzero(sc.v == i)
            if len(xx):
                lights.append((np.array([xx.mean(), yy.mean(),
                                         zz.mean()]),
                               p["emit"] ** 2 * len(xx) * 0.55,
                               np.array(p["rgb"]) / 255.0))
    d = np.array([-0.60, -0.38, 0.70])  # from the SE, into the corner
    d /= np.linalg.norm(d)
    u = np.cross([0, 1, 0], d); u /= np.linalg.norm(u)
    w = np.cross(d, u)
    center = np.array([X * 0.55, Y * 0.30, Z * 0.42])
    origin = center - d * 300
    span = X * 1.15
    img = np.zeros((px, px, 3), dtype=np.float32)
    shade = {(1, 0, 0): 0.85, (-1, 0, 0): 0.68, (0, 1, 0): 1.1,
             (0, -1, 0): 0.42, (0, 0, 1): 1.0, (0, 0, -1): 0.55}
    for py in range(px):
        for pxx in range(px):
            a = (pxx / px - 0.5) * span
            b = (0.5 - py / px) * span * 0.9
            ro = origin + u * a + w * b
            t0 = 0.0
            for ax, lim in ((0, X), (1, Y), (2, Z)):
                if d[ax] > 1e-6:
                    t0 = max(t0, (0 - ro[ax]) / d[ax])
                elif d[ax] < -1e-6:
                    t0 = max(t0, (lim - ro[ax]) / d[ax])
            pos = ro + d * (t0 + 1e-3)
            cell = np.floor(pos).astype(int)
            step = np.sign(d).astype(int)
            tmax = np.empty(3); tdel = np.empty(3)
            for ax in range(3):
                if abs(d[ax]) < 1e-9:
                    tmax[ax] = 1e30; tdel[ax] = 1e30
                else:
                    nxt = cell[ax] + (1 if step[ax] > 0 else 0)
                    tmax[ax] = (nxt - pos[ax]) / d[ax] + t0
                    tdel[ax] = abs(1.0 / d[ax])
            normal = (0, 1, 0)
            for _ in range(int(X + Y + Z) + 8):
                if (0 <= cell[0] < X and 0 <= cell[1] < Y
                        and 0 <= cell[2] < Z):
                    m = sc.v[cell[2], cell[1], cell[0]]
                    if m:
                        base = pal_rgb[m]
                        if pal_emit[m] > 0:
                            img[py, pxx] = np.clip(
                                base * (1.1 + pal_emit[m] / 10.0), 0, 255)
                            break
                        f = shade.get(normal, 0.7) * 0.30
                        hitp = cell + 0.5
                        nvec = np.array(normal, dtype=np.float32)
                        light = np.array([0.06, 0.055, 0.05])
                        for lp, lpow, lcol in lights:
                            lv = lp - np.array([hitp[0], hitp[1],
                                                hitp[2]])
                            d2 = float(lv @ lv) + 1.0
                            ndl = max(float((lv / np.sqrt(d2)) @ nvec),
                                      0.08)
                            light += lcol * (lpow * ndl / d2) * 0.045
                        img[py, pxx] = np.clip(
                            base * (f + light * 2.2) , 0, 255)
                        break
                ax = int(np.argmin(tmax))
                cell[ax] += step[ax]
                tmax[ax] += tdel[ax]
                normal = tuple(int(-step[k]) if k == ax else 0
                               for k in range(3))
            else:
                img[py, pxx] = (16, 17, 22)
    empty = img.sum(axis=2) == 0
    img[empty] = (16, 17, 22)
    return Image.fromarray(img.astype(np.uint8))


if __name__ == "__main__":
    sc = build()
    out = os.path.join(MODELS, "hearth_diorama_mockup.png")
    render(sc).save(out)
    print(out)
