#!/usr/bin/env python3
"""Fold demo, three panels from ONE fine voxel set:

  FINE — the 1-unit voxel terrain (the truth being folded)
  A    — folded 8-unit blocks, axis-aligned face normals (today's shading)
  B    — folded 8-unit blocks, normals folded from the fine faces

The fold is real: coarse occupancy = quantized mean of the fine columns in
each 8x8 footprint; the folded normal = area-weighted average of the fine
terrain's actual exposed cube faces (tops contribute +y, side walls
contribute +/-x,+/-z proportional to their area). No analytic gradients —
everything downstream of HF comes from the fine voxel grid.

This is NOT the renderer — it's an offline preview of the
`claude_far_normals` dial and a worked example of the FOLDING.md contract.

Run:  ~/.venvs/voice/bin/python util/claude/fold_normals_ab.py
Out:  util/claude/fold_normals_ab.png
"""

import numpy as np
from PIL import Image, ImageDraw

W, H = 880, 340          # per panel
BF, BC = 1.0, 8.0        # fine and coarse cell size
FOV = 55.0

# ---------------------------------------------------------------- fine set
# A rolling-hills height function, immediately quantized into 1-unit
# voxel columns. HF is the ONLY geometry source below this block.
def rolling(x, z):
    return (10.0 * np.sin(x * 0.021 + 1.7) * np.cos(z * 0.017)
            + 6.0 * np.sin(x * 0.043 - 0.6) * np.sin(z * 0.031 + 0.4)
            + 3.0 * np.sin(x * 0.09 + z * 0.07)
            + 0.012 * z)

X0, Z0 = -624.0, -64.0   # multiples of BC: geometry walls must align with the face-classification grid
NX, NZ = 1240, 1520                      # fine grid, 1 unit
fx = X0 + np.arange(NX) + 0.5
fz = Z0 + np.arange(NZ) + 0.5
HF = np.round(rolling(fx[:, None], fz[None, :]))     # 1-unit voxel columns

# ------------------------------------------------------------- the FOLD
NXC, NZC = NX // 8, NZ // 8
blocks = HF.reshape(NXC, 8, NZC, 8)
# occupancy fold: mean fine height per 8x8 footprint, quantized to blocks
HC = np.round(blocks.mean(axis=(1, 3)) / BC) * BC
# normal fold: area-weighted sum of the fine terrain's exposed faces.
# Each column top: area 1, normal +y. Each wall between neighbour columns:
# area |dh|, normal points from the taller column toward the shorter one,
# i.e. the signed sum along x is -sum(dh_x) (telescopes to the cell's mean
# slope — the average of real cube faces, not an analytic gradient).
dhx = np.zeros_like(HF); dhx[:-1, :] = HF[1:, :] - HF[:-1, :]
dhz = np.zeros_like(HF); dhz[:, :-1] = HF[:, 1:] - HF[:, :-1]
nxc = -dhx.reshape(NXC, 8, NZC, 8).sum(axis=(1, 3))
nzc = -dhz.reshape(NXC, 8, NZC, 8).sum(axis=(1, 3))
nyc = np.full_like(nxc, 64.0)                        # 64 tops of area 1
NC = np.stack([nxc, nyc, nzc], axis=-1)
NC /= np.linalg.norm(NC, axis=-1, keepdims=True)

def h_fine(x, z):
    ix = np.clip(np.floor(x - X0).astype(int), 0, NX - 1)
    iz = np.clip(np.floor(z - Z0).astype(int), 0, NZ - 1)
    return HF[ix, iz]

def h_coarse(x, z):
    ix = np.clip((np.floor(x - X0) // 8).astype(int), 0, NXC - 1)
    iz = np.clip((np.floor(z - Z0) // 8).astype(int), 0, NZC - 1)
    return HC[ix, iz]

def n_folded(x, z):
    ix = np.clip((np.floor(x - X0) // 8).astype(int), 0, NXC - 1)
    iz = np.clip((np.floor(z - Z0) // 8).astype(int), 0, NZC - 1)
    return NC[ix, iz]

# ---------------------------------------------------------------- camera
ro = np.array([0.0, rolling(0, 0) + 26.0, 0.0])
ys, xs = np.mgrid[0:H, 0:W]
u = (xs + 0.5) / W * 2 - 1
v = 1 - (ys + 0.5) / H * 2
t_half = np.tan(np.radians(FOV / 2))
rd = np.stack([u * t_half * (W / H), v * t_half - 0.10,
               np.ones_like(u)], axis=-1)
rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
rd = rd.reshape(-1, 3)
N = rd.shape[0]

def march(hfunc, cell):
    t = np.full(N, 4.0); hit_t = np.zeros(N); alive = np.ones(N, bool)
    for i in range(560):
        p = ro + rd * t[:, None]
        below = p[:, 1] < hfunc(p[:, 0], p[:, 2])
        hit_t[alive & below] = t[alive & below]
        alive &= ~below
        if not alive.any():
            break
        t = np.where(alive, t + np.maximum(0.35, t * 0.006), t)
    hit = hit_t > 0
    lo = np.maximum(hit_t - np.maximum(0.35, hit_t * 0.006), 0.0)
    hi = hit_t.copy()
    for i in range(14):
        mid = 0.5 * (lo + hi)
        p = ro + rd * mid[:, None]
        below = p[:, 1] < hfunc(p[:, 0], p[:, 2])
        hi = np.where(below & hit, mid, hi)
        lo = np.where(~below & hit, mid, lo)
    ph = ro + rd * hi[:, None]
    pb = ro + rd * (hi - 0.30)[:, None]
    nz_ = np.zeros((N, 3))                       # face normal classification
    cxh, czh = np.floor(ph[:, 0] / cell), np.floor(ph[:, 2] / cell)
    cxb, czb = np.floor(pb[:, 0] / cell), np.floor(pb[:, 2] / cell)
    sx = cxh != cxb; sz = (czh != czb) & ~sx; top = ~sx & ~sz
    nz_[top] = [0, 1, 0]
    nz_[sx, 0] = -np.sign(rd[sx, 0]); nz_[sz, 2] = -np.sign(rd[sz, 2])
    return hit, hi, ph, nz_

# ---------------------------------------------------------------- shading
sun = np.array([-0.55, 0.38, 0.74]); sun /= np.linalg.norm(sun)
SUN_COL = np.array([1.00, 0.93, 0.80]) * 1.25
SKY_COL = np.array([0.42, 0.55, 0.75])
ALBEDO = np.array([0.52, 0.55, 0.48])

def shade(hit, hit_t, n):
    lam = np.clip(n @ sun, 0, 1)[:, None]
    amb = (0.30 + 0.22 * np.clip(n[:, 1], 0, 1))[:, None]
    col = ALBEDO * (SUN_COL * lam + SKY_COL * amb)
    horizon = np.clip(rd[:, 1] * 2.2 + 0.35, 0, 1)[:, None]
    sky = SKY_COL * (1.05 - 0.45 * horizon) + np.array([0.30, 0.22, 0.10]) * \
        np.clip(rd @ sun, 0, 1)[:, None] ** 8
    fog = np.clip(hit_t / 950.0, 0, 1)[:, None] ** 1.4
    col = col * (1 - fog) + sky * fog
    col = np.where(hit[:, None], col, sky)
    return (np.clip(col, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8).reshape(H, W, 3)

hitF, tF, pF, nF = march(h_fine, BF)
hitC, tC, pC, nC_face = march(h_coarse, BC)
imgF = shade(hitF, tF, nF)
imgA = shade(hitC, tC, nC_face)
imgB = shade(hitC, tC, n_folded(pC[:, 0], pC[:, 2]))

PAD = 34
out = Image.new("RGB", (W, H * 3 + PAD * 3), (18, 18, 20))
d = ImageDraw.Draw(out)
for i, (img, label) in enumerate([
        (imgF, "FINE - the 1-unit voxel terrain (the truth being folded)"),
        (imgA, "A - folded 8-unit blocks, axis-aligned face normals (today)"),
        (imgB, "B - folded 8-unit blocks, normals folded from the fine cube faces")]):
    out.paste(Image.fromarray(img), (0, PAD * (i + 1) + H * i))
    d.text((10, PAD * i + H * i + 9), label, fill=(235, 235, 235))
path = __file__.replace(".py", ".png")
out.save(path)
print("wrote", path)
