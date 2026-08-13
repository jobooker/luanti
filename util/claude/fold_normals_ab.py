#!/usr/bin/env python3
"""A/B mockup: axis-aligned face normals vs folded (averaged) normals
on coarse-block terrain, same geometry, same silhouette, same light.

This is NOT the renderer — it's an offline illustration of the shading
phenomenon the `claude_far_normals` dial will control, so the taste call
can be previewed before the far irradiance cache exists.

Run:  ~/.venvs/voice/bin/python util/claude/fold_normals_ab.py
Out:  util/claude/fold_normals_ab.png  (A over B, labeled)
"""

import numpy as np
from PIL import Image, ImageDraw

W, H = 880, 420          # per panel
BLOCK = 8.0              # coarse cell size in world units ("16 m rung")
FOV = 55.0

# ---- smooth "true" terrain the blocks quantize (stand-in for fine voxels)
def smooth_h(x, z):
    return (10.0 * np.sin(x * 0.021 + 1.7) * np.cos(z * 0.017)
            + 6.0 * np.sin(x * 0.043 - 0.6) * np.sin(z * 0.031 + 0.4)
            + 3.0 * np.sin(x * 0.09 + z * 0.07)
            + 0.012 * z)  # gentle rise toward the horizon

def blocky_h(x, z):
    """Coarse-cell terrain: sample smooth field at cell centers, quantize
    height to whole blocks. This is what 'fold occupancy' produces."""
    cx = (np.floor(x / BLOCK) + 0.5) * BLOCK
    cz = (np.floor(z / BLOCK) + 0.5) * BLOCK
    return np.round(smooth_h(cx, cz) / BLOCK) * BLOCK

def folded_normal(x, z):
    """Mode B: the fold's averaged normal = gradient of the underlying
    fine terrain, sampled per coarse cell (constant within a cell, like
    a real folded channel would be)."""
    cx = (np.floor(x / BLOCK) + 0.5) * BLOCK
    cz = (np.floor(z / BLOCK) + 0.5) * BLOCK
    e = 2.0
    dhdx = (smooth_h(cx + e, cz) - smooth_h(cx - e, cz)) / (2 * e)
    dhdz = (smooth_h(cx, cz + e) - smooth_h(cx, cz - e)) / (2 * e)
    n = np.stack([-dhdx, np.ones_like(dhdx), -dhdz], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)

# ---- camera
ro = np.array([0.0, smooth_h(0, 0) + 26.0, 0.0])
pitch = -0.10
aspect = W / H
ys, xs = np.mgrid[0:H, 0:W]
u = (xs + 0.5) / W * 2 - 1
v = 1 - (ys + 0.5) / H * 2
t_half = np.tan(np.radians(FOV / 2))
rd = np.stack([u * t_half * aspect,
               v * t_half + pitch,
               np.ones_like(u)], axis=-1)
rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
rd = rd.reshape(-1, 3)
N = rd.shape[0]

# ---- march the blocky field (shared by both panels)
t = np.full(N, 4.0)
hit_t = np.zeros(N)
alive = np.ones(N, bool)
for i in range(560):
    p = ro + rd * t[:, None]
    below = p[:, 1] < blocky_h(p[:, 0], p[:, 2])
    newly = alive & below
    hit_t[newly] = t[newly]
    alive &= ~below
    if not alive.any():
        break
    dt = np.maximum(0.35, t * 0.006)          # step grows with distance
    t = np.where(alive, t + dt, t)

hit = hit_t > 0
# refine by bisection
lo = np.maximum(hit_t - np.maximum(0.35, hit_t * 0.006), 0.0)
hi = hit_t.copy()
for i in range(14):
    mid = 0.5 * (lo + hi)
    p = ro + rd * mid[:, None]
    below = p[:, 1] < blocky_h(p[:, 0], p[:, 2])
    hi = np.where(below & hit, mid, hi)
    lo = np.where(~below & hit, mid, lo)
hit_t = hi
ph = ro + rd * hit_t[:, None]

# ---- face classification for mode A (top vs side, which axis)
pb = ro + rd * (hit_t - 0.30)[:, None]        # a nudge before the hit
cxh, czh = np.floor(ph[:, 0] / BLOCK), np.floor(ph[:, 2] / BLOCK)
cxb, czb = np.floor(pb[:, 0] / BLOCK), np.floor(pb[:, 2] / BLOCK)
nA = np.zeros((N, 3))
side_x = (cxh != cxb)
side_z = (czh != czb) & ~side_x
top = ~side_x & ~side_z
nA[top] = [0.0, 1.0, 0.0]
nA[side_x, 0] = -np.sign(rd[side_x, 0])
nA[side_z, 2] = -np.sign(rd[side_z, 2])

# ---- mode B normal
nB = folded_normal(ph[:, 0], ph[:, 2])

# ---- shading (identical everywhere except the normal)
sun = np.array([-0.55, 0.38, 0.74]); sun /= np.linalg.norm(sun)
SUN_COL = np.array([1.00, 0.93, 0.80]) * 1.25
SKY_COL = np.array([0.42, 0.55, 0.75])
ALBEDO = np.array([0.52, 0.55, 0.48])

def shade(n):
    lam = np.clip((n @ sun), 0, 1)[:, None]
    amb = (0.30 + 0.22 * np.clip(n[:, 1], 0, 1))[:, None]
    col = ALBEDO * (SUN_COL * lam + SKY_COL * amb)
    # sky + distance fog
    horizon = np.clip(rd[:, 1] * 2.2 + 0.35, 0, 1)[:, None]
    sky = SKY_COL * (1.05 - 0.45 * horizon) + np.array([0.30, 0.22, 0.10]) * \
        np.clip((rd @ sun), 0, 1)[:, None] ** 8
    fog = np.clip(hit_t / 950.0, 0, 1)[:, None] ** 1.4
    col = col * (1 - fog) + sky * fog
    col = np.where(hit[:, None], col, sky)
    return (np.clip(col, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)

imgA = shade(nA).reshape(H, W, 3)
imgB = shade(nB).reshape(H, W, 3)

# ---- compose, label, save
PAD = 34
out = Image.new("RGB", (W, H * 2 + PAD * 2), (18, 18, 20))
out.paste(Image.fromarray(imgA), (0, PAD))
out.paste(Image.fromarray(imgB), (0, H + PAD * 2))
d = ImageDraw.Draw(out)
d.text((10, 9), "A — axis-aligned face normals (today): each coarse face one brightness",
       fill=(235, 235, 235))
d.text((10, H + PAD + 9), "B — folded normals (averaged from fine terrain): same blocks, same silhouette, re-lit",
       fill=(235, 235, 235))
path = __file__.replace(".py", ".png")
out.save(path)
print("wrote", path)
