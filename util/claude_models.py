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


# ---- texture-derived bake: the 16x16 gift ---------------------------
# Mineclonia tiles are 16x16 and our grid is 16^3: each face's texels
# become surface voxel colors; depth comes from a per-face luminance
# heightfield (dark recesses, maxdepth voxels); bright warm texels on
# designated faces become emissive voxels. Hand-authored models above
# remain the fallback for nodes whose derived depth reads wrong.

def _face_map(face, u, v, d):
    """(texel u,v, carve depth d) -> voxel x,y,z for each cube face."""
    if face == "front":   return u, 15 - v, 15 - d          # +z
    if face == "back":    return 15 - u, 15 - v, d          # -z
    if face == "right":   return 15 - d, 15 - v, u          # +x
    if face == "left":    return d, 15 - v, 15 - u          # -x
    if face == "top":     return u, 15 - d, v               # +y
    if face == "bottom":  return u, d, 15 - v               # -y
    raise ValueError(face)


def bake_from_tiles(name, tiles, maxdepth=3, emissive_faces=(),
                    emit_level=13):
    """tiles: {face: png path}. Returns (name, palette, voxels) in the
    same shape the authored models use. Palette grows per unique
    (rgb, emit) — texel-true colors, no quantization."""
    imgs = {}
    for face, path in tiles.items():
        imgs[face] = np.asarray(
            Image.open(path).convert("RGB").resize((N, N),
                                                   Image.NEAREST),
            dtype=np.float32)
    # interior filler: mean of the side-ish faces
    fillsrc = [f for f in ("side", "left", "right", "back")
               if f in tiles] or list(tiles)
    fill = tuple(int(c) for c in
                 np.mean([imgs[f].mean(axis=(0, 1)) for f in fillsrc],
                         axis=0))
    pal = [None]
    pindex = {}

    def pi(rgb, emit=0):
        key = (rgb, emit)
        if key not in pindex:
            pal.append(dict(rgb=list(rgb), emit=emit))
            pindex[key] = len(pal) - 1
        return pindex[key]

    v = np.full((N, N, N), pi(fill), dtype=np.uint16)
    # 'side' shorthand expands to the four lateral faces
    faces = {}
    for face, img in imgs.items():
        if face == "side":
            for f in ("left", "right", "back"):
                faces.setdefault(f, img)
            faces.setdefault("front", img)
        else:
            faces[face] = img
    for face, img in faces.items():
        lum = img @ np.array([0.2126, 0.7152, 0.0722])
        lo, hi = lum.min(), max(lum.max(), lum.min() + 1.0)
        med = float(np.median(lum))
        for vv in range(N):
            for u in range(N):
                r, g, b = img[vv, u]
                is_fire = (face in emissive_faces and r > 140.0
                           and r > 1.5 * b and g > 40.0)
                if is_fire:
                    d = maxdepth  # fire sits at the back of its recess
                else:
                    # STEP rule, not linear: a noisy stone texture must
                    # stay a solid block with 1-voxel grain — only
                    # near-black features (the mouth) carve deep. The
                    # linear heightfield swiss-cheesed the furnace.
                    t = (lum[vv, u] - lo) / (hi - lo)
                    if t < 0.18:
                        d = maxdepth
                    elif lum[vv, u] < med - 0.10 * (hi - lo):
                        d = 1
                    else:
                        d = 0
                for dd in range(d):
                    x, y, z = _face_map(face, u, vv, dd)
                    v[z, y, x] = 0
                x, y, z = _face_map(face, u, vv, d)
                v[z, y, x] = pi((int(r), int(g), int(b)),
                                emit_level if is_fire else 0)
    return name, pal, v


def model_furnace_baked():
    tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "games", "mineclonia", "mods", "ITEMS",
                        "mcl_furnaces", "textures")
    return bake_from_tiles(
        "furnace_baked",
        dict(front=os.path.join(tdir, "default_furnace_front_active.png"),
             side=os.path.join(tdir, "default_furnace_side.png"),
             top=os.path.join(tdir, "default_furnace_top.png"),
             bottom=os.path.join(tdir, "default_furnace_bottom.png")),
        maxdepth=3, emissive_faces=("front",), emit_level=13)


def model_crafting_baked():
    tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "games", "mineclonia", "mods", "ITEMS",
                        "mcl_crafting_table", "textures")
    return bake_from_tiles(
        "crafting_baked",
        dict(front=os.path.join(tdir, "crafting_workbench_front.png"),
             side=os.path.join(tdir, "crafting_workbench_side.png"),
             top=os.path.join(tdir, "crafting_workbench_top.png")),
        maxdepth=2)


def extrude_cutout(name, path, thick=2, emit_level=12):
    """Torch-class bake: a mostly-transparent 16x16 tile extruded into
    a `thick`-voxel standing model centered in the cell. Bright warm
    texels (the flame) become emissive voxels — the point light gets a
    real sub-voxel body instead of an analytic nub."""
    img = np.asarray(Image.open(path).convert("RGBA").resize(
        (N, N), Image.NEAREST), dtype=np.float32)
    pal = [None]
    pindex = {}

    def pi(rgb, emit=0):
        key = (rgb, emit)
        if key not in pindex:
            pal.append(dict(rgb=list(rgb), emit=emit))
            pindex[key] = len(pal) - 1
        return pindex[key]

    v = np.zeros((N, N, N), dtype=np.uint16)
    z0 = (N - thick) // 2
    for vv in range(N):
        for u in range(N):
            r, g, b, a = img[vv, u]
            if a < 128:
                continue
            is_flame = r > 180.0 and g > 100.0 and r > 1.4 * b
            idx = pi((int(r), int(g), int(b)),
                     emit_level if is_flame else 0)
            for z in range(z0, z0 + thick):
                v[z, 15 - vv, u] = idx
    return name, pal, v


def model_torch_baked():
    tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "games", "mineclonia", "mods", "ITEMS",
                        "mcl_torches", "textures")
    return extrude_cutout(
        "torch_baked",
        os.path.join(tdir, "default_torch_on_floor.png"),
        thick=2, emit_level=12)


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
    for fn in (model_furnace, model_chest, model_crafting,
               model_furnace_baked, model_crafting_baked,
               model_torch_baked):
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
