#!/usr/bin/env python3
"""Authored 16^3 sub-voxel models — the phase 4.5 test articles
(interior-program-plan.md; ADR-0005 authored assets, ADR-0009
emission law, ADR-0010 fold operator).

Template is GENERATOR, never runtime (the circle-postmortem law):
this script emits JSON masks the snapshot bake consumes, plus
isometric preview PNGs so shapes get eyes on them before the engine
does. Voxel space: x right, y up, z toward the viewer; the FRONT of
a model (furnace mouth, chest clasp) faces +z.

Usage: claude_models.py [outdir]      (default: util/claude_models/)

JSON format per model:
  { "name": ..., "palette": [ {"rgb":[r,g,b], "emit":0..14}, ... ],
    "voxels": [z][y][x] palette index, 0 = air }
Palette index 0 is reserved for air. "emit" follows the node
light_source scale; the bake maps it through ADR-0009 emitStrength
and ADR-0010's flux fold.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

N = 16


def model_furnace():
    """Dark stone body, recessed mouth in the +z face, emissive fire
    voxels at the back of the recess + ember floor. The mouth is the
    ADR-0010 flux-fold test subject: same watts at every rung."""
    pal = [
        None,
        dict(rgb=(96, 94, 92), emit=0),     # 1 stone body
        dict(rgb=(72, 70, 68), emit=0),     # 2 dark rim / mouth frame
        dict(rgb=(52, 50, 48), emit=0),     # 3 recess interior (soot)
        dict(rgb=(255, 150, 40), emit=13),  # 4 fire
        dict(rgb=(255, 80, 16), emit=9),    # 5 embers
        dict(rgb=(120, 118, 114), emit=0),  # 6 top slab lighter stone
    ]
    v = np.ones((N, N, N), dtype=np.uint8)  # [z][y][x], start solid stone
    # subtle banding: top course lighter, base course darker
    v[:, 15, :] = 6
    v[:, 0, :] = 2
    # mouth: opening 6 wide x 5 tall in the +z face, recessed 4 deep
    #   x 5..10, y 1..5, z 12..15 carved to air; frame stays
    for z in range(12, 16):
        for y in range(1, 6):
            for x in range(5, 11):
                v[z, y, x] = 0
    # frame ring around the opening on the front face
    for y in range(0, 7):
        for x in range(4, 12):
            if (y in (0, 6) or x in (4, 11)) and v[15, y, x] != 0:
                v[15, y, x] = 2
    # recess walls get soot
    for z in range(12, 16):
        for y in range(1, 6):
            for x in range(5, 11):
                for dz, dy, dx in ((0, -1, 0), (0, 1, 0), (0, 0, -1),
                                   (0, 0, 1), (-1, 0, 0)):
                    zz, yy, xx = z + dz, y + dy, x + dx
                    if 0 <= zz < N and 0 <= yy < N and 0 <= xx < N \
                            and v[zz, yy, xx] == 1:
                        v[zz, yy, xx] = 3
    # fire: back wall of the recess (z=11 face voxels) glows
    for y in range(2, 5):
        for x in range(6, 10):
            v[11, y, x] = 4
    # embers on the recess floor
    for x in range(6, 10):
        v[12, 1, x] = 5
        v[13, 1, x] = 5 if x % 2 == 0 else 3
    return "furnace_custom", pal, v


def model_chest():
    """Oak body inset 1 voxel all round (a chest is smaller than its
    cell), darker lid seam, iron clasp PROUD by one voxel — the
    plan's literal example of real sub-voxel geometry that shadows
    itself."""
    pal = [
        None,
        dict(rgb=(152, 106, 56), emit=0),   # 1 oak body
        dict(rgb=(122, 82, 40), emit=0),    # 2 dark edge banding
        dict(rgb=(96, 62, 30), emit=0),     # 3 lid seam shadow line
        dict(rgb=(168, 168, 176), emit=0),  # 4 iron clasp
    ]
    v = np.zeros((N, N, N), dtype=np.uint8)
    # body x 1..14, z 1..14, y 0..13 (proud clasp adds z=15)
    v[1:15, 0:14, 1:15] = 1
    # edge banding: vertical corners + top/bottom rims of body and lid
    for y in (0, 8, 9, 13):
        v[1:15, y, 1] = 2
        v[1:15, y, 14] = 2
        v[1, y, 1:15] = 2
        v[14, y, 1:15] = 2
    for z, x in ((1, 1), (1, 14), (14, 1), (14, 14)):
        v[z, 0:14, x] = 2
    # lid seam: dark line all round at y=8 (lid is y 9..13)
    v[1:15, 8, 1] = 3
    v[1:15, 8, 14] = 3
    v[1, 8, 1:15] = 3
    v[14, 8, 1:15] = 3
    # clasp: 2 wide x 3 tall centered on the front face, PROUD 1 voxel
    for y in range(7, 10):
        for x in range(7, 9):
            v[14, y, x] = 4       # on the body face
            v[15, y, x] = 4       # proud by 1/16 m
    return "chest_custom", pal, v


def model_crafting():
    """Full-cube oak workbench: darker top rim, a 3x3 grid of grooves
    RECESSED one voxel into the top (real geometry that self-shadows,
    not painted lines), tool-motif banding on the sides."""
    pal = [
        None,
        dict(rgb=(150, 104, 54), emit=0),   # 1 oak body
        dict(rgb=(112, 76, 38), emit=0),    # 2 dark rim
        dict(rgb=(88, 58, 28), emit=0),     # 3 groove floor
        dict(rgb=(132, 92, 46), emit=0),    # 4 top plank light
    ]
    v = np.ones((N, N, N), dtype=np.uint8)
    v[:, 15, :] = 4                          # top surface
    # top rim
    v[0, 15, :] = 2
    v[15, 15, :] = 2
    v[:, 15, 0] = 2
    v[:, 15, 15] = 2
    # 3x3 grid grooves recessed 1: lines at x/z = 4-5 and 10-11
    for a in (4, 5, 10, 11):
        for b in range(1, 15):
            v[a, 15, b] = 0
            v[b, 15, a] = 0
            v[a, 14, b] = 3
            v[b, 14, a] = 3
    # side banding: dark waist line
    v[0, 9, :] = 2
    v[15, 9, :] = 2
    v[:, 9, 0] = 2
    v[:, 9, 15] = 2
    return "crafting_custom", pal, v


# ---- isometric preview (orthographic ray march, front-right-top) ----

def preview(pal, v, px=420):
    img = np.zeros((px, px, 3), dtype=np.float32)
    # orthographic rays along d; camera basis u (right), w (up)
    d = np.array([-0.62, -0.44, -0.65])
    d /= np.linalg.norm(d)
    u = np.cross([0, 1, 0], d)
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    origin = np.array([8.0, 8.0, 8.0]) - d * 40
    span = 26.0
    shade = {tuple([1, 0, 0]): 0.85, tuple([-1, 0, 0]): 0.70,
             tuple([0, 1, 0]): 1.25, tuple([0, -1, 0]): 0.45,
             tuple([0, 0, 1]): 1.05, tuple([0, 0, -1]): 0.60}
    for py in range(px):
        for pxx in range(px):
            a = (pxx / px - 0.5) * span
            b = (0.5 - py / px) * span
            ro = origin + u * a + w * b
            # DDA through the 16^3 grid ([x,y,z] indexing == v[z,y,x])
            t0 = 0.0
            pos = ro.copy()
            # advance to the bounding box
            for ax in range(3):
                if d[ax] > 1e-6:
                    t0 = max(t0, (0 - pos[ax]) / d[ax])
                elif d[ax] < -1e-6:
                    t0 = max(t0, (N - pos[ax]) / d[ax])
            pos = ro + d * (t0 + 1e-4)
            cell = np.floor(pos).astype(int)
            step = np.sign(d).astype(int)
            tmax = np.empty(3)
            tdelta = np.empty(3)
            for ax in range(3):
                if abs(d[ax]) < 1e-9:
                    tmax[ax] = 1e30
                    tdelta[ax] = 1e30
                else:
                    nxt = cell[ax] + (1 if step[ax] > 0 else 0)
                    tmax[ax] = (nxt - pos[ax]) / d[ax]
                    tdelta[ax] = abs(1.0 / d[ax])
            normal = -step * (np.array([1, 0, 0]))
            for _ in range(64):
                if all(0 <= cell[ax] < N for ax in range(3)):
                    m = v[cell[2], cell[1], cell[0]]
                    if m:
                        c = np.array(pal[m]["rgb"], dtype=np.float32)
                        e = pal[m]["emit"]
                        f = shade.get(tuple(normal.tolist()), 0.6)
                        col = c * f
                        if e:
                            col = c * (0.9 + e / 14.0 * 1.6)
                        img[py, pxx] = np.clip(col, 0, 255)
                        break
                ax = int(np.argmin(tmax))
                cell[ax] += step[ax]
                tmax[ax] += tdelta[ax]
                normal = np.zeros(3, dtype=int)
                normal[ax] = -step[ax]
            else:
                pass
    bg = img.sum(axis=2) == 0
    img[bg] = (24, 26, 30)
    return Image.fromarray(img.astype(np.uint8))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "claude_models")
    os.makedirs(outdir, exist_ok=True)
    for fn in (model_furnace, model_chest, model_crafting):
        name, pal, v = fn()
        data = dict(name=name,
                    palette=[None] + [dict(rgb=list(p["rgb"]),
                                           emit=p["emit"])
                                      for p in pal[1:]],
                    voxels=v.tolist())
        jp = os.path.join(outdir, name + ".json")
        with open(jp, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        pv = preview(pal, v)
        pp = os.path.join(outdir, name + "_preview.png")
        pv.save(pp)
        solid = int((v > 0).sum())
        emis = int(sum((v == i).sum() for i, p in enumerate(pal)
                       if p and p["emit"] > 0))
        print("%-16s solid %4d/4096  emissive %3d  -> %s , %s"
              % (name, solid, emis, jp, pp))


if __name__ == "__main__":
    main()
