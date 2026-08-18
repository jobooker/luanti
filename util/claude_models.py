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
    # mouth: opening 6 wide x 5 tall in the +z face, recessed 3 deep
    #   x 5..10, y 1..5, z 13..15 carved to air; frame stays
    #
    # 3 AND NOT 4, corrected 2026-08-18 (0d). This said 4 (z 12..15) and
    # had said so since it was written -- but the bake floor's carve
    # shell for this model is 3 sub-voxels deep, so the z = 12 layer was
    # in NO face shell and was silently plugged on the way out. The
    # shipped mask has therefore had a 3-deep mouth all along, with the
    # fire WALLED IN behind the plug (12 of its 18 emissive cells sat at
    # z = 11, one layer behind solid stone). The new precondition in
    # enforce_opaque() is what surfaced it. Nothing on screen moves:
    # MANIFEST maps every furnace node to furnace_baked, so this
    # hand-authored model is a fallback that is not placed in the world.
    for z in range(13, 16):
        for y in range(1, 6):
            for x in range(5, 11):
                v[z, y, x] = 0
    # frame ring around the opening on the front face
    for y in range(0, 7):
        for x in range(4, 12):
            if (y in (0, 6) or x in (4, 11)) and v[15, y, x] != 0:
                v[15, y, x] = 2
    # recess walls get soot
    for z in range(13, 16):
        for y in range(1, 6):
            for x in range(5, 11):
                for dz, dy, dx in ((0, -1, 0), (0, 1, 0), (0, 0, -1),
                                   (0, 0, 1), (-1, 0, 0)):
                    zz, yy, xx = z + dz, y + dy, x + dx
                    if 0 <= zz < N and 0 <= yy < N and 0 <= xx < N \
                            and v[zz, yy, xx] == 1:
                        v[zz, yy, xx] = 3
    # fire: back wall of the recess (z=12 face voxels) glows
    for y in range(2, 5):
        for x in range(6, 10):
            v[12, y, x] = 4
    # embers on the recess floor
    for x in range(6, 10):
        v[13, 1, x] = 5
        v[14, 1, x] = 5 if x % 2 == 0 else 3
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


LUMW = np.array([0.2126, 0.7152, 0.0722])

# Physical mean albedo per material family, LINEAR (what pathAlbedo
# produces). Game textures are painted to be DISPLAYED, not to be used
# as reflectance: raw oak planks linearize to 0.136, about 3x darker
# than real wood, and interiors are bounce-dominated so that error
# compounds once per bounce (a 3-bounce corner keeps ~6% of its light).
ALBEDO_TARGET = {
    "wood": 0.35, "log": 0.30, "stone": 0.25, "cobble": 0.22,
    "wool": 0.60, "book": 0.30, "default": 0.32,
}


def _delight(img, carved, delight=1.0, target=None):
    """Turn a painted texture into a physical albedo map.

    Two corrections, both bake-time (template is generator, never
    runtime — the circle-postmortem law):

    1. DELIGHT. A groove is dark in the texture because the artist
       PAINTED a shadow there. We then carve that same texel into a
       real recess, which the tracer shadows for real -- so the
       darkness lands twice. Where we carve, lift the texel back to
       the unshadowed material tone and let geometry produce the
       shadow. (The carve DEPTH still comes from the original
       luminance; only the stored colour changes.)
    2. RENORMALIZE. Scale the face so its mean linear albedo is a
       physical value for the material.

    Emissive texels are excluded by the caller: their brightness is
    real, not painted shading.
    """
    out = img.astype(np.float32).copy()
    if delight > 0.0 and carved.any() and (~carved).any():
        base = out[~carved].mean(axis=0)          # unshadowed tone
        baselum = max(float(base @ LUMW), 1.0)
        cur = out[carved]
        curlum = np.maximum(cur @ LUMW, 1.0)
        lifted = np.clip(cur * (baselum / curlum)[:, None], 0.0, 255.0)
        out[carved] = cur * (1.0 - delight) + lifted * delight
    if target:
        lin = (out / 255.0) ** 2.2
        m = float(lin.mean())
        if m > 1e-6:
            lin = np.clip(lin * (target / m), 0.0, 1.0)
            out = (lin ** (1.0 / 2.2)) * 255.0
    return out


def bake_from_tiles(name, tiles, maxdepth=3, emissive_faces=(),
                    emit_level=13, delight=1.0, albedo=None):
    """tiles: {face: png path}. Returns (name, palette, voxels) in the
    same shape the authored models use. Palette grows per unique
    (rgb, emit) — texel-true colors, no quantization.

    delight/albedo drive _delight(): see there for why a raw game
    texture is not an albedo map. delight=0, albedo=None reproduces
    the original texel-verbatim bake."""
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
        # STEP rule + STRUCTURE constraints (John's eyes, 2026-08-13:
        # "some of the geometry doesn't seem realistically possible").
        # A dark texel is only CARVED if it belongs to a connected dark
        # feature of >= 3 texels (kills lone pits and floating studs —
        # a knot is dark, not deep), and depth-1 grain never carves the
        # outer ring (kills undercut slivers where two faces chew the
        # same cube edge). Deep carves (the near-black mouth band) are
        # trusted as-is: they are designed features, not noise.
        deep = ((lum - lo) / (hi - lo)) < 0.18
        grain = lum < (med - 0.10 * (hi - lo))
        # despeckle grain: 4-connected component size >= 3
        keep = np.zeros((N, N), dtype=bool)
        seen = np.zeros((N, N), dtype=bool)
        for sv0 in range(N):
            for su0 in range(N):
                if not grain[sv0, su0] or seen[sv0, su0]:
                    continue
                comp = [(sv0, su0)]
                seen[sv0, su0] = True
                qi = 0
                while qi < len(comp):
                    cv, cu = comp[qi]; qi += 1
                    for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nv, nu = cv + dv, cu + du
                        if 0 <= nv < N and 0 <= nu < N \
                                and grain[nv, nu] and not seen[nv, nu]:
                            seen[nv, nu] = True
                            comp.append((nv, nu))
                if len(comp) >= 3:
                    for cv, cu in comp:
                        keep[cv, cu] = True
        # Which texels become recessed geometry — computed from the
        # ORIGINAL luminance (the artist's shading is a good depth
        # signal), then handed to _delight so their painted-in shadow
        # can be removed before the colour is stored.
        interior = np.zeros((N, N), dtype=bool)
        interior[1:N - 1, 1:N - 1] = True
        fire = np.zeros((N, N), dtype=bool)
        if face in emissive_faces:
            fire = ((img[:, :, 0] > 140.0)
                    & (img[:, :, 0] > 1.5 * img[:, :, 2])
                    & (img[:, :, 1] > 40.0))
        carved = (deep | (keep & interior)) & ~fire
        alb = _delight(img, carved, delight=delight, target=albedo)
        alb[fire] = img[fire]   # emissive brightness is real, not paint

        for vv in range(N):
            for u in range(N):
                r, g, b = alb[vv, u]
                is_fire = bool(fire[vv, u])
                if is_fire:
                    d = maxdepth  # fire sits at the back of its recess
                elif deep[vv, u]:
                    d = maxdepth
                elif keep[vv, u] and 0 < vv < N - 1 and 0 < u < N - 1:
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
        maxdepth=3, emissive_faces=("front",), emit_level=13,
        albedo=ALBEDO_TARGET["stone"])


def model_crafting_baked():
    tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "games", "mineclonia", "mods", "ITEMS",
                        "mcl_crafting_table", "textures")
    return bake_from_tiles(
        "crafting_baked",
        dict(front=os.path.join(tdir, "crafting_workbench_front.png"),
             side=os.path.join(tdir, "crafting_workbench_side.png"),
             top=os.path.join(tdir, "crafting_workbench_top.png")),
        maxdepth=2, albedo=ALBEDO_TARGET["wood"])


def model_bookshelf_baked():
    mods = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "games", "mineclonia", "mods", "ITEMS")
    return bake_from_tiles(
        "bookshelf_baked",
        dict(side=os.path.join(mods, "mcl_books", "textures",
                               "default_bookshelf.png"),
             top=os.path.join(mods, "mcl_core", "textures",
                              "default_wood.png"),
             bottom=os.path.join(mods, "mcl_core", "textures",
                                 "default_wood.png")),
        maxdepth=1)  # book spines recess a single voxel


def model_bed(part):
    """Two-cell bed, authored (MCL beds are mesh nodes, no cube
    tiles). Canonical orientation: foot at low z, head at high z;
    the bake rotates by param2. Half-height per the real thing."""
    pal = [
        None,
        dict(rgb=(126, 84, 44), emit=0),    # 1 oak frame
        dict(rgb=(96, 62, 30), emit=0),     # 2 dark frame edge / legs
        dict(rgb=(178, 34, 34), emit=0),    # 3 red blanket
        dict(rgb=(210, 48, 48), emit=0),    # 4 blanket highlight fold
        dict(rgb=(235, 232, 224), emit=0),  # 5 white sheet
        dict(rgb=(250, 249, 245), emit=0),  # 6 pillow
    ]
    v = np.zeros((N, N, N), dtype=np.uint16)
    head = part == "head"
    # legs: 2x2x3 at the cell's OUTER end corners only
    zleg = (13, 14) if head else (1, 2)
    for x in (1, 2, 13, 14):
        for z in zleg:
            v[z, 0:3, x] = 2
    # base slab + side rails
    v[0:16, 3, 1:15] = 1
    v[0:16, 4, 1] = 2
    v[0:16, 4, 14] = 2
    if not head:
        v[0, 4, 1:15] = 2       # footboard lip
        v[0, 5, 1:15] = 2
    else:
        v[15, 4:8, 1:15] = 2    # headboard
    # mattress + covers
    if head:
        v[0:9, 5, 2:14] = 3     # blanket reaches partway up the bed
        v[0:8, 6, 2:14] = 4
        v[9:14, 5, 2:14] = 5    # sheet
        v[10:14, 6, 3:13] = 6   # pillow
        v[11:13, 7, 4:12] = 6
    else:
        v[1:16, 5, 2:14] = 3
        v[2:15, 6, 2:14] = 4    # folded top layer
    return "bed_red_%s" % part, pal, v


def model_lantern():
    """Floor lantern: iron cage, emissive core — the cozy house's
    fourth emitter class gets a real body."""
    pal = [
        None,
        dict(rgb=(70, 72, 80), emit=0),      # 1 iron cage
        dict(rgb=(48, 50, 56), emit=0),      # 2 dark iron base/cap
        dict(rgb=(255, 214, 120), emit=12),  # 3 glowing core
    ]
    v = np.zeros((N, N, N), dtype=np.uint16)
    v[5:11, 0, 5:11] = 2                     # base plate
    for x, z in ((5, 5), (5, 10), (10, 5), (10, 10)):
        v[z, 1:7, x] = 1                     # corner posts
    v[6:10, 1:6, 6:10] = 3                   # core
    v[5:11, 6, 5:11] = 2                     # cap
    v[7:9, 7, 7:9] = 1                       # hanger nub
    v[7:9, 8, 7:9] = 1
    return "lantern_floor", pal, v


def model_campfire():
    """Crossed logs, ember bed, low flame — the hearth's centerpiece.
    Fire voxels emissive at the campfire's light level."""
    pal = [
        None,
        dict(rgb=(110, 72, 36), emit=0),     # 1 log
        dict(rgb=(76, 48, 24), emit=0),      # 2 log end/dark
        dict(rgb=(255, 120, 30), emit=11),   # 3 embers
        dict(rgb=(255, 200, 60), emit=13),   # 4 flame
        dict(rgb=(40, 36, 32), emit=0),      # 5 char
    ]
    v = np.zeros((N, N, N), dtype=np.uint16)
    # two logs along x (bottom), two along z (crossed on top)
    for z0 in (2, 11):
        v[z0:z0 + 3, 0:3, 0:16] = 1
        v[z0:z0 + 3, 0:3, 0] = 2
        v[z0:z0 + 3, 0:3, 15] = 2
    for x0 in (2, 11):
        v[0:16, 3:6, x0:x0 + 3] = 1
        v[0, 3:6, x0:x0 + 3] = 2
        v[15, 3:6, x0:x0 + 3] = 2
    # char where flame licks the logs
    v[5:11, 3, 5:11] = 5
    # ember bed + flame column
    v[5:11, 0:2, 5:11] = 3
    v[6:10, 2:5, 6:10] = 4
    v[7:9, 5:7, 7:9] = 4
    return "campfire_lit", pal, v


def model_carpet():
    pal = [None, dict(rgb=(232, 230, 226), emit=0),
           dict(rgb=(210, 206, 200), emit=0)]
    v = np.zeros((N, N, N), dtype=np.uint16)
    v[:, 0, :] = 1
    v[0, 0, :] = 2                           # subtle woven border
    v[15, 0, :] = 2
    v[:, 0, 0] = 2
    v[:, 0, 15] = 2
    return "carpet_white", pal, v


def model_flowerpot():
    pal = [
        None,
        dict(rgb=(150, 82, 50), emit=0),    # 1 terracotta
        dict(rgb=(96, 60, 36), emit=0),     # 2 soil
        dict(rgb=(58, 120, 48), emit=0),    # 3 stem
        dict(rgb=(214, 64, 64), emit=0),    # 4 poppy bloom
    ]
    v = np.zeros((N, N, N), dtype=np.uint16)
    v[6:10, 0:4, 6:10] = 1                  # pot
    v[7:9, 3, 7:9] = 2                      # soil
    v[8, 4:8, 8] = 3                        # stem
    v[7:10, 8:10, 7:10] = 4                 # bloom
    v[8, 9, 8] = 4
    return "flowerpot_poppy", pal, v


def _mcl(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "games", "mineclonia", "mods", *parts)


def model_planks_oak():
    t = _mcl("ITEMS", "mcl_core", "textures", "default_wood.png")
    return bake_from_tiles("planks_oak_baked",
                           dict(side=t, top=t, bottom=t), maxdepth=1,
                           albedo=ALBEDO_TARGET["wood"])


def model_planks_spruce():
    t = _mcl("ITEMS", "mcl_core", "textures", "mcl_core_planks_spruce.png")
    return bake_from_tiles("planks_spruce_baked",
                           dict(side=t, top=t, bottom=t), maxdepth=1,
                           albedo=ALBEDO_TARGET["wood"])


def model_log_oak():
    return bake_from_tiles(
        "log_oak_baked",
        dict(side=_mcl("ITEMS", "mcl_core", "textures",
                       "default_tree.png"),
             top=_mcl("ITEMS", "mcl_core", "textures",
                      "default_tree_top.png"),
             bottom=_mcl("ITEMS", "mcl_core", "textures",
                         "default_tree_top.png")),
        maxdepth=1, albedo=ALBEDO_TARGET["log"])


def model_cobble():
    t = _mcl("ITEMS", "mcl_core", "textures", "default_cobble.png")
    return bake_from_tiles("cobble_baked",
                           dict(side=t, top=t, bottom=t), maxdepth=1,
                           albedo=ALBEDO_TARGET["cobble"])


# ---- the bake floor: no model may be see-through ---------------------
# Found 2026-08-16 (roadmap "HOLES IN THE WALL"). John, looking at the
# cosy cabin's far wall after the walker learned to descend into 16^3
# cells: bright specks, in horizontal rows, that move as the camera
# moves. They were the lit furnace room seen through 1/16 m tunnels in
# the plank model. A cabin wall is ONE node thick, so a straight line of
# air through the model IS a hole through the wall.
#
# The tunnels are not a bug in any one carve; they are what carving with
# no floor produces. Two ways in, both seen in the data:
#   * one face carves a whole row -- a plank's groove line is dark all
#     the way across the tile, so the entire outer layer along that row
#     goes to air (planks_oak_baked, top face, y=15 z=7);
#   * two faces carve the same cells -- an edge row belongs to both the
#     -y and -z face, and between them they cover all 16 (planks_*, the
#     y=0/z=0 corner rows).
# The rule below is written against the RESULT, not against either
# cause, so it also catches the third way in that nobody has thought of.
#
# THE 2026-08-16 RULE, AND WHY IT WAS NOT ENOUGH. It said: no line of
# cells through the model ON ANY AXIS may be all air, and it plugged one
# cell (index 7) of each offending line. It closed all 768 axis-aligned
# lines and John's specks came back anyway -- because A RAY IS NOT AXIS
# ALIGNED. The reasoning behind the one-cell plug ("a ray can only
# traverse the tunnel end to end by staying inside it") is false for a
# tunnel 3 cells DEEP: the face carves reach maxdepth = 3, so the air
# beside a plugged line is usually air too, and a ray that drifts one
# sub-voxel sideways over the crossing goes round the plug.
#
# AND THE TUNNEL CROSSES A NODE SEAM, which is why one model in
# isolation looks innocent. MEASURED 2026-08-17 on the SHIPPED masks
# with a plain DDA on the CPU: not one ray in 120,000 gets from the -x
# face of a single plank cell to its +x face. Simulate the actual WALL
# instead -- one node thick, the same mask tiled in y and z -- and rays
# DO get through: 5 in 300,000 for spruce, 19 in 300,000 for oak, all of
# them near-perpendicular. A ray leaves one cell's carved shell through
# a SIDE face and carries on inside its neighbour's. That is the sliver
# rate, and none of it is the renderer's doing: the GPU's copy of the
# spruce mask was dumped to screen the same day and matched the JSON in
# all 4096 bits, and an independent point sampler found no place where
# the walk stepped over geometry that was there.
#
# THE RULE NOW, and it is a proof rather than a patch: in a model that
# represents a FULL SOLID NODE, every air cell lies within MAXDEPTH of
# EXACTLY ONE face. Carving is a surface effect -- no face carves deeper
# than maxdepth -- so that makes the air six DISJOINT BOXES, one per
# face, each open only to its own face:
#
#     -x box = x in [0,3), y and z in [3,13)      (and five more)
#
# A 3D-DDA visits face-adjacent cells, so a ray's path through air is a
# 6-CONNECTED chain and cannot leave its box without entering a cell the
# rule has made solid. A ray that came in through the -x face is moving
# +x, so it cannot leave the way it came either. It therefore STOPS
# INSIDE THE NODE, whatever its direction. Both leaks this bake has now
# paid for are corollaries: a straight crossing (-x to +x) has to cross
# the solid core, and a CORNER CLIP (-z to +x, which is what survived
# the first attempt at this fix on 2026-08-17 and put the corner post's
# oak inside the cabin's spruce wall) has to cross a cell that is within
# maxdepth of both faces.
#
# FILL POLICY: the offending cells take the model's base material -- the
# modal solid palette entry, which for a texture bake is the interior
# fill colour the bake already writes. Equivalently, and this is the way
# to picture the cost: a face may only carve its central texels and keeps
# a SOLID RIM whose width is exactly the carve depth.
#
# **RIM WIDTH AND CARVE DEPTH ARE ONE NUMBER.** That is why the depth
# below is PER MODEL and not global (2026-08-18, DECISIONS 0d, John:
# "shallow grooves"). Until now every solid node was judged at depth 3 --
# the deepest carve in the set, which is the furnace's mouth -- so a
# plank, whose grooves are only ONE sub-voxel deep, was made to carry a
# THREE-texel rim it had no carve to justify. On screen a plank wall
# stopped reading as continuous plank lines and started reading as inset
# panels: every groove died three texels short of the block edge, so the
# joins between blocks became a raised lattice.
#
# The proof below needs only that no carve in THIS model goes deeper than
# THIS model's depth, so each model may use its own -- and every model's
# own depth is the `maxdepth` its bake was already called with. The
# planks, log, cobble and bookshelf drop 3 -> 1, so the rim drops to a
# single texel, which is the narrowest a one-deep carve can possibly
# leave. The furnace keeps 3: its mouth really is carved that deep.
# `main()` ASSERTS this rather than trusting the table -- an air cell
# deeper than the declared depth is a bake error, not a plug.
#
# Applied ONLY to SOLID_NODE_MODELS. A bed, a lantern, a campfire, a
# carpet, a torch and a flowerpot are see-through on purpose and any
# "fix" to them would be a defect.

# name -> carve depth = rim width, in sub-voxels (1/16 m each).
SOLID_NODE_MODELS = {
    # texture-derived cube bakes: the `maxdepth=` of each bake_from_tiles
    # call below, which is the deepest air any face of it can reach.
    "furnace_baked": 3,      # near-black mouth band, carved 3 deep
    "crafting_baked": 2,
    "bookshelf_baked": 1,    # book spines recess a single voxel
    "planks_oak_baked": 1,
    "planks_spruce_baked": 1,
    "log_oak_baked": 1,
    "cobble_baked": 1,
    # hand-authored full-cube nodes: the depth their own carve reaches.
    "furnace_custom": 3,     # mouth: z 12..15 air behind the +z face
    "crafting_custom": 1,    # 3x3 grid of grooves, recessed one voxel
}
# NOT here, and each for a reason: chest_custom (a chest body is inset
# from its cell -- it is a full node in the map but not a full cube of
# geometry), bed_red_foot/head, lantern_floor, campfire_lit,
# carpet_white, flowerpot_poppy, torch_baked.

def through_lines(v):
    """Every all-air line through a 16^3 mask, as (axis, a, b) with
    v indexed [z][y][x]:
        ("x", z, y)  ->  v[z, y, :]
        ("y", z, x)  ->  v[z, :, x]
        ("z", y, x)  ->  v[:, y, x]
    Empty list == the model is opaque along every axis."""
    s = np.asarray(v) > 0
    out = []
    for a in range(N):
        for b in range(N):
            if not s[a, b, :].any():
                out.append(("x", a, b))
            if not s[a, :, b].any():
                out.append(("y", a, b))
            if not s[:, a, b].any():
                out.append(("z", a, b))
    return out


def base_material_index(pal, v):
    """The model's base material: the most common non-air palette
    index. For a texture bake that is the interior fill the bake
    already writes, so a plug is made of the same stuff as the
    node's inside."""
    counts = np.bincount(np.asarray(v).ravel(),
                         minlength=len(pal)).astype(np.int64)
    counts[0] = 0
    if counts.sum() == 0:
        return 1
    return int(counts.argmax())


# The deepest carve in the whole set, kept only as the upper bound a
# per-model depth may not exceed. It is NOT the depth any model is judged
# at: SOLID_NODE_MODELS carries that, per model, and it is the rim width.
SHELL_MAX = 3


def shell_count(n=N, depth=SHELL_MAX):
    """For every cell of an n^3 model, how many of the six faces it is
    within `depth` of. 1 = it may be air; 0 or >=2 = it may not."""
    idx = np.indices((n, n, n))
    return (np.minimum(idx, n - 1 - idx) < depth).sum(axis=0)


def carve_reach(v):
    """The deepest air in the model, as a DEPTH: 0 = no air at all,
    1 = air only in the outermost layer of some face, 3 = air three
    layers in. Compared against the model's declared depth, this is what
    turns "no carve goes deeper than the shell" from an assumption into
    an assertion."""
    air = np.asarray(v) == 0
    if not air.any():
        return 0
    idx = np.indices(air.shape)
    depth = np.minimum(idx, N - 1 - idx).min(axis=0)   # distance to nearest face
    return int(depth[air].max()) + 1


def enforce_opaque(name, pal, v, depth):
    """Apply the bake floor: every air cell must sit in EXACTLY ONE
    face's carve shell, which makes the node opaque to any straight ray
    (see the block above). Returns (v, report); v is modified in place.
    report = dict(before=n, after=n, plugs=[(x,y,z)], material=i).

    `depth` is the model's own carve depth, which is also the width of
    the solid rim this leaves on every face.

    `before`/`after` still count all-air AXIS-ALIGNED lines, because
    that is the number this bake has reported since 2026-08-16 and it
    stays comparable -- but it is a corollary of the rule now, not the
    rule, and `after` is 0 by construction.
    """
    v = np.asarray(v)
    before = through_lines(v)
    reach = carve_reach(v)
    # THE PRECONDITION OF THE PROOF, checked before anything is plugged:
    # the six carve shells are disjoint boxes only while no carve reaches
    # deeper than `depth`. An air cell deeper than that would be silently
    # filled below -- geometry destroyed by a table entry that is too
    # small -- so it stops the bake instead.
    if reach > depth:
        raise SystemExit(
            "BAKE FLOOR: %s carves %d sub-voxels deep but is declared at "
            "depth %d in SOLID_NODE_MODELS. Plugging would destroy that "
            "carve; raise the declared depth (and its rim) or carve "
            "shallower." % (name, reach, depth))
    mat = base_material_index(pal, v)
    bad = (v == 0) & (shell_count(depth=depth) != 1)
    plugs = [(int(x), int(y), int(z))
             for z, y, x in np.argwhere(bad)]
    v[bad] = mat
    after = through_lines(v)
    return v, dict(before=len(before), after=len(after), plugs=plugs,
                   material=mat, depth=depth, reach=reach)


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


# model -> node substitution map: at snapshot-bake time (the phase 4.5
# re-land) cells holding these nodes take the named model's mask +
# colors + emission instead of the texture-heightfield default; the
# model is rotated by the node's param2. The cozy house already
# contains every one of these nodes, so "placement" is automatic.
MANIFEST = {
    "furnace_baked": ["mcl_furnaces:furnace_active",
                      "mcl_furnaces:furnace"],
    "chest_custom": ["mcl_chests:chest_small", "mcl_chests:chest"],
    "crafting_baked": ["mcl_crafting_table:crafting_table"],
    "torch_baked": ["mcl_torches:torch", "mcl_torches:torch_wall"],
    "bookshelf_baked": ["mcl_books:bookshelf"],
    "bed_red_foot": ["mcl_beds:bed_red_bottom"],
    "bed_red_head": ["mcl_beds:bed_red_top"],
    "lantern_floor": ["mcl_lanterns:lantern_floor"],
    "campfire_lit": ["mcl_campfires:campfire_lit"],
    "carpet_white": ["mcl_wool:white_carpet"],
    "flowerpot_poppy": ["mcl_flowerpots:flower_pot"],
    "planks_oak_baked": ["mcl_trees:wood_oak"],
    "planks_spruce_baked": ["mcl_trees:wood_spruce"],
    "log_oak_baked": ["mcl_trees:tree_oak"],
    "cobble_baked": ["mcl_core:cobble"],
}


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "claude_models")
    os.makedirs(outdir, exist_ok=True)
    print("outdir: %s" % outdir)
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(dict(rotate_param2=True, nodes=MANIFEST), f,
                  indent=1)
    for fn in (model_furnace, model_chest, model_crafting,
               model_furnace_baked, model_crafting_baked,
               model_torch_baked, model_bookshelf_baked,
               lambda: model_bed("foot"), lambda: model_bed("head"),
               model_lantern, model_campfire, model_carpet,
               model_flowerpot, model_planks_oak, model_planks_spruce,
               model_log_oak, model_cobble):
        name, pal, v = fn()
        if name in SOLID_NODE_MODELS:
            depth = SOLID_NODE_MODELS[name]
            if depth > SHELL_MAX:
                raise SystemExit("%s: declared depth %d exceeds SHELL_MAX %d"
                                 % (name, depth, SHELL_MAX))
            v, floor = enforce_opaque(name, pal, v, depth)
            holes = "  rim %d  holes %d->%d  opacity fill %4d" % (
                depth, floor["before"], floor["after"], len(floor["plugs"]))
            # The invariant, checked rather than trusted: no air
            # cell may be in two shells or in none, at THIS model's own
            # depth. That is the whole proof, it is 4096 cells to look at
            # rather than a ray budget to argue about, and it is asserted
            # on the RESULT so a future carve that reaches deeper than the
            # declared depth trips it (enforce_opaque checks the same
            # thing before it plugs anything, so a too-small entry in the
            # table stops the bake rather than eating the geometry).
            leftover = int(((np.asarray(v) == 0)
                            & (shell_count(depth=depth) != 1)).sum())
            if leftover:
                raise SystemExit(
                    "BAKE FLOOR FAILED on %s: %d air cell(s) are not in "
                    "exactly one face shell" % (name, leftover))
            if floor["after"]:
                raise SystemExit(
                    "BAKE FLOOR FAILED on %s: %d through-line(s) survived "
                    "the plug pass" % (name, floor["after"]))
        else:
            holes = "  (see-through by design)"
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
        print("%-20s solid %4d/4096  emissive %3d%s"
              % (name, solid, emis, holes))


if __name__ == "__main__":
    main()
