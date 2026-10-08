#!/usr/bin/env python3
"""claude_tree_build — ladder plan stage 1b: the tree built from what the
tracer actually walks (claude_export_trace), walked with the camera's own
rays through exact pixel centres, against the GPU's first hits
(claude_view 33 / 34). float32, as the GPU runs.

  python3 util/claude_tree_build.py EXPORT_DIR VIEW33.png VIEW34.png [--every 8] [--factors 4,4,4,4,4,2]
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

F = np.float32
BIG = F(1e30)
S = 128          # 1 m cells per axis
SUB = 16         # base pieces per 1 m (a parameter of this loader, not of the walk)
R0, R1 = 48, 80  # the ring of per-cell masks (shader SUBV_R0 / R1)


def load(d):
    g = np.fromfile(os.path.join(d, "grid.rgba"), np.uint8).reshape(S, S, S, 4)        # z, y, x
    sv = np.fromfile(os.path.join(d, "subvox.r8"), np.uint8).reshape(512, 512, 64)       # z, y, x(bytes)
    mid = np.fromfile(os.path.join(d, "model_ids.r8"), np.uint8).reshape(S, S, S)
    atlas = np.fromfile(os.path.join(d, "model_atlas.r8"), np.uint8).reshape(2048, 16, 16)
    pal = np.fromfile(os.path.join(d, "matpal.f32"), np.float32).reshape(2, 256, 4)
    cam = json.load(open(os.path.join(d, "camera.json")))
    return g, sv, mid, atlas, pal, cam


def fine_mask(cx, cy, cz, sv, mid, atlas, model_far=True):
    """the shader's fineSolid() for every piece of one 1 m cell: [x][y][z] bool"""
    if R0 <= cx < R1 and R0 <= cy < R1 and R0 <= cz < R1:
        rx, ry, rz = cx - R0, cy - R0, cz - R0
        m = np.zeros((16, 16, 16), bool)
        for sz in range(16):
            for sy in range(16):
                row = sv[rz * 16 + sz, ry * 16 + sy, rx * 2:rx * 2 + 2]       # 2 bytes = 16 bits in x
                bits = np.unpackbits(row, bitorder="little")
                m[:, sy, sz] = bits.astype(bool)
        return m
    code = int(mid[cz, cy, cx])
    if model_far and code > 3:
        mm, rot = code // 4 - 1, code % 4
        base = (mm * 4 + rot) * 16
        a = atlas[base:base + 16]                                              # [sz][sy][sx]
        return (a > 0).transpose(2, 1, 0)                                      # -> [x][y][z]
    return None


def build_base(g, sv, mid, atlas, pal):
    """sparse base rung: dict (cx,cy,cz) -> 'full' or a 16^3 mask"""
    cls = g[..., 3]
    fine = pal[0, :, 1] > 0.5                       # matPal(cls).g: fine material
    cells = {}
    zs, ys, xs = np.nonzero(cls > 0)
    for z, y, x in zip(zs, ys, xs):
        c = int(cls[z, y, x])
        m = None
        if fine[c]:
            in_ring = R0 <= x < R1 and R0 <= y < R1 and R0 <= z < R1
            if in_ring or mid[z, y, x] > 3:
                m = fine_mask(x, y, z, sv, mid, atlas)
        cells[(int(x), int(y), int(z))] = "full" if m is None else m
    return cells


class Tree:
    """occupancy per level as dicts of occupied keys; level 0 = base pieces"""
    def __init__(self, cells, factors):
        self.factors = factors
        self.size = [1]
        for f in factors:
            self.size.append(self.size[-1] * f)
        self.cells = cells
        base_per_m = SUB
        # level of the 1 m cell: the level whose size == SUB
        assert SUB in self.size, "the ladder must pass through 1 m"
        self.Lm = self.size.index(SUB)
        self.occ = [set() for _ in self.size]
        for (x, y, z), v in cells.items():
            if isinstance(v, str):
                for L in range(self.Lm, len(self.size)):
                    s = self.size[L] // SUB
                    self.occ[L].add((x // s, y // s, z // s))
            else:
                xs, ys, zs = np.nonzero(v)
                for L in range(0, self.Lm):
                    s = self.size[L]
                    for a, b, c in set(zip(xs // s, ys // s, zs // s)):
                        self.occ[L].add((x * (SUB // s) + int(a), y * (SUB // s) + int(b), z * (SUB // s) + int(c)))
                for L in range(self.Lm, len(self.size)):
                    s = self.size[L] // SUB
                    self.occ[L].add((x // s, y // s, z // s))
        # pieces of full cells are implied: level < Lm occupied if its 1 m cell is full
    def occupied(self, L, c):
        if L >= self.Lm:
            return tuple(c) in self.occ[L]
        s = SUB // self.size[L]
        cm = (c[0] // s, c[1] // s, c[2] // s)
        v = self.cells.get(cm)
        if v is None:
            return False
        if isinstance(v, str):
            return True
        return tuple(c) in self.occ[L]


def tcross(plane, ro, step, delta):
    return ((plane - ro) * step * delta + (F(1) - np.abs(step)) * BIG).astype(F)


def tie_axis(side):
    if side[0] < side[1] and side[0] < side[2]:
        return 0
    if side[1] < side[2]:
        return 1
    return 2


def walk(T, ro, rd, steps=20000):
    """the tree walk of claude_tree_equiv, over T; ro, rd in BASE units"""
    factors, size = T.factors, T.size
    LMAX = len(factors)
    NB = S * SUB
    step = np.sign(rd).astype(F)
    delta = (F(1) / np.maximum(np.abs(rd), F(1e-8))).astype(F)
    L = 0
    c = np.floor(ro).astype(np.int64)
    t = F(0)
    axis = -1
    started = False
    faces = lambda c, L: tcross(((c + (step > 0)) * size[L]).astype(F), ro, step, delta)
    side = faces(c, L)
    for i in range(steps):
        if started:
            while L > 0 and T.occupied(L, c):
                f = factors[L - 1]
                L -= 1
                cs = size[L]
                k = np.zeros(3, np.int64)
                for ax in range(3):
                    lo = c[ax] * f
                    if step[ax] == 0:
                        k[ax] = min(int(np.floor((ro[ax] - lo * cs) / cs)), f - 1)
                        continue
                    n = 0
                    for j in range(1, f):
                        jj = j if step[ax] > 0 else f - j
                        tm = tcross(F((lo + jj) * cs), ro[ax:ax + 1], step[ax:ax + 1], delta[ax:ax + 1])[0]
                        if tm < t or (tm == t and ax > axis):
                            n += 1
                        else:
                            break
                    k[ax] = n if step[ax] > 0 else f - 1 - n
                c = c * f + k
                side = faces(c, L)
            if L == 0 and T.occupied(0, c):
                return tuple(int(v) for v in c), axis
            while L < LMAX and not T.occupied(L + 1, c // factors[L]):
                c = c // factors[L]
                L += 1
                side = faces(c, L)
        started = True
        a = tie_axis(side)
        t = side[a]
        axis = a
        c[a] += int(step[a])
        if c[a] < 0 or c[a] >= NB // size[L]:
            return None, a
        side = faces(c, L)
    return None, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("v33")
    ap.add_argument("v34")
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--factors", default="4,4,4,4,4,2")
    ap.add_argument("--trace-scale", type=float, default=0.5)
    ap.add_argument("--noflip", action="store_true")
    ap.add_argument("--map", help="write a PNG of where each category falls")
    ap.add_argument("--classify", action="store_true", help="re-walk mismatches with nudged rays")
    ap.add_argument("--row-offset", type=float, default=-1.0)
    a = ap.parse_args()
    g, sv, mid, atlas, pal, cam = load(a.export)
    cells = build_base(g, sv, mid, atlas, pal)
    print("1 m cells: %d occupied, %d with 1/16 detail" % (len(cells), sum(not isinstance(v, str) for v in cells.values())))
    T = Tree(cells, [int(x) for x in a.factors.split(",")])
    print("levels (base pieces):", T.size, " occupied nodes per level:", [len(o) for o in T.occ])
    # the screenshot is top-down at window size; the trace ran at
    # trace_scale of it, each trace pixel shown as a k x k block (NEAREST):
    # take one screen pixel per trace pixel, then flip to GL's bottom-up rows
    k = int(round(1.0 / a.trace_scale))
    i33 = np.asarray(Image.open(a.v33).convert("RGB")).astype(int)[::k, ::k]
    i34 = np.asarray(Image.open(a.v34).convert("RGB")).astype(int)[::k, ::k]
    if not a.noflip:
        i33, i34 = i33[::-1], i34[::-1]
    H, W = i33.shape[:2]
    ro0 = (np.array(cam["cam_pos"], np.float32) + F(0.5)) * F(SUB)
    fwd, right, up = (np.array(cam[k], np.float32) for k in ("fwd", "right", "up"))
    cats = {}
    stats = {"boundary rounding": 0, "same": 0, "cell or face differs": 0, "piece differs": 0, "gpu none, tree hit": 0, "gpu hit, tree none": 0}
    shown = 0
    for py in range(0, H, a.every):
        for px in range(0, W, a.every):
            u = F((px + 0.5) / W) * F(2) - F(1)
            # the saved image sits exactly one row from the shader's rows
            # (found 2026-10-07: every reproducible mismatch needed -1.0 px in y)
            v = F((py + a.row_offset + 0.5) / H) * F(2) - F(1)
            d = (fwd + u * right + v * up).astype(F)
            d = (d / F(np.sqrt(np.sum(d * d, dtype=F)))).astype(F)
            hit, axis = walk(T, ro0, d)
            code, sub = i33[py, px], i34[py, px]
            def agrees(hit, axis, d):
                if hit is None:
                    return (code == 255).all()
                if (code == 255).all():
                    return False
                cm = [hit[k] // SUB for k in range(3)]
                face = axis * 2 + (1 if d[axis] < 0 else 0)
                gc = [code[k] & 127 for k in range(3)]
                gface = (code[0] >> 7) | ((code[1] >> 7) << 1) | ((code[2] >> 7) << 2)
                sx, sy, sz = [hit[k] % SUB for k in range(3)]
                return gc == cm and gface == face and sub[0] == sx * 16 + sy and sub[1] == sz
            def boundary():
                # ROUNDING, NOT DISAGREEMENT: a ray nudged by ~1e-6 (an ulp or a
                # few of the GPU's arithmetic) lands where the GPU did
                for e in (1e-6, 3e-6, 1e-5):
                    for dv in ((e, 0, 0), (-e, 0, 0), (0, e, 0), (0, -e, 0), (0, 0, e), (0, 0, -e)):
                        d2 = (d + np.array(dv, np.float32)).astype(F)
                        h2, a2 = walk(T, ro0, d2)
                        if agrees(h2, a2, d2):
                            return True
                return False
            if (code == [0, 255, 0]).all() or (sub == [0, 255, 0]).all():
                continue        # the capture's green 'traced' marker, not a surface
            if not agrees(hit, axis, d) and a.classify:
                if boundary():
                    stats["boundary rounding"] += 1; cats[(px, py)] = "boundary rounding"
                    continue
            gpu_none = (code == 255).all()
            if hit is None:
                k2 = "same" if gpu_none else "gpu hit, tree none"
                stats[k2] += 1; cats[(px, py)] = k2
                continue
            if gpu_none:
                stats["gpu none, tree hit"] += 1; cats[(px, py)] = "gpu none, tree hit"
                continue
            cm = [hit[k] // SUB for k in range(3)]
            face_ax = axis
            sg = 1 if d[face_ax] < 0 else 0                    # the face faces back at the ray
            face = face_ax * 2 + sg
            gc = [code[k] & 127 for k in range(3)]
            gface = (code[0] >> 7) | ((code[1] >> 7) << 1) | ((code[2] >> 7) << 2)
            if gc != cm or gface != face:
                stats["cell or face differs"] += 1; cats[(px, py)] = "cell or face differs"
                if shown < 5:
                    shown += 1
                    print("  px %d,%d  gpu cell %s face %d | tree cell %s face %d" % (px, py, gc, gface, cm, face))
                continue
            sx, sy, sz = [hit[k] % SUB for k in range(3)]
            if sub[0] != sx * 16 + sy or sub[1] != sz:
                stats["piece differs"] += 1; cats[(px, py)] = "piece differs"
                continue
            stats["same"] += 1; cats[(px, py)] = "same"
    if a.map:
        # where the categories fall: same = grey, cell/face = white, piece = mid,
        # gpu hit / tree none = black dots on a dark field (brightness, not hue)
        mp = np.zeros((H // a.every + 1, W // a.every + 1), np.uint8)
        for (px, py), cat in cats.items():
            mp[py // a.every, px // a.every] = {"same": 90, "cell or face differs": 255,
                                               "piece differs": 170, "gpu hit, tree none": 30,
                                               "gpu none, tree hit": 30, "boundary rounding": 120}[cat]
        Image.fromarray(mp[::-1]).resize((W // 2, H // 2), Image.NEAREST).save(a.map)
    n = sum(stats.values())
    print("pixels compared: %d (every %d-th)" % (n, a.every))
    for k, v in stats.items():
        print("  %-24s %7d  (%.3f%%)" % (k, v, 100.0 * v / max(n, 1)))


if __name__ == "__main__":
    main()
