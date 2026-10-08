"""claude_tree_gate_replay — ladder plan stage 2: explain every ray where the
tree walk and today's walk (marchMed) disagreed in the in-shader gate.

The gate (claude_view 35) counts verdicts on the GPU and records the first
128 disagreeing rays with their exact float bits (debug.txt lines
"[claude_tree_gate]   px ..."). Export the trace data at the same pose
(claude_export_trace = <dir>), then:

  grep -a "claude_tree_gate\\]   px" debug.txt | tail -N > records.txt
  python3 util/claude_tree_gate_replay.py <export dir> records.txt

For every record: the plain 1/16 m walk in float64 (the referee), a Python
copy of today's walk in float32 (must reproduce the GPU, or the GPU reads
different data than the export), and the same copy without the 1/512 entry
pull (marchMed's fine-entry clamp, SUBV - 1/512).
"""
import re
import struct
import sys

import numpy as np

F = np.float32
BIG = F(1e30)
d, records = sys.argv[1], sys.argv[2]
grid = np.fromfile(d + "/grid.rgba", np.uint8).reshape(128, 128, 128, 4)      # [z][y][x]
sub = np.fromfile(d + "/subvox.r8", np.uint8).reshape(512, 512, 64)
mids = np.fromfile(d + "/model_ids.r8", np.uint8).reshape(128, 128, 128)
atlas = np.fromfile(d + "/model_atlas.r8", np.uint8).reshape(2048, 16, 16)
pal = np.fromfile(d + "/matpal.f32", np.float32).reshape(2, 256, 4)


def ring(c):
    return all(48 <= v < 80 for v in c)


def cellinfo(c):
    x, y, z = c
    idx = int(grid[z, y, x, 3])
    return idx, float(pal[0, idx, 1]), int(mids[z, y, x]), ring(c)


def fine_at(c):
    idx, g, mid, rg = cellinfo(c)
    return rg or mid > 3


def solid(B):
    """the shader's rule: any non-air material stops a camera ray; a fine
    material is solid per its 1/16 m bits (ring texture, else model atlas)"""
    c = [int(v) // 16 for v in B]
    s = [int(v) % 16 for v in B]
    if any(v < 0 or v >= 128 for v in c):
        return False
    idx, g, mid, rg = cellinfo(c)
    if idx == 0:
        return False
    if not (g > 0.5 and (rg or mid > 3)):
        return True
    if rg:
        tx = (c[0] - 48) * 2 + s[0] // 8
        ty = (c[1] - 48) * 16 + s[1]
        tz = (c[2] - 48) * 16 + s[2]
        return bool((sub[tz, ty, tx] >> (s[0] % 8)) & 1)
    m = mid // 4 - 1
    layer = (m * 4 + mid % 4) * 16 + s[2]
    return 0 <= layer < 2048 and atlas[layer, s[1], s[0]] > 0


def exact(ro, rd):
    """the referee: plain walk over 1/16 m pieces, float64"""
    ro = ro.astype(np.float64) * 16
    rd = rd.astype(np.float64)
    step = np.sign(rd)
    ci = np.floor(ro)
    for _ in range(40000):
        ts = [((ci[a] + (step[a] > 0)) - ro[a]) / rd[a] if step[a] else 1e300 for a in range(3)]
        a = int(np.argmin(ts))
        ci[a] += step[a]
        if (ci < 0).any() or (ci >= 2048).any():
            return None
        if solid(ci):
            return tuple(int(v) for v in ci), a, ts[a] / 16
    return None


def tie(side):
    if side[0] < side[1] and side[0] < side[2]:
        return 0
    if side[1] < side[2]:
        return 1
    return 2


def march(ro, rd, pull=F(16 - 1 / 512)):
    """today's walk (marchMed, curMed 0), float32; pyramid leaps left out"""
    step = np.sign(rd).astype(F)
    delta = (F(1) / np.maximum(np.abs(rd), F(1e-6))).astype(F)
    ci = np.floor(ro).astype(F)
    cellHi = ci.copy()
    side = ((step * (ci - ro) + step * F(0.5) + F(0.5)) * delta).astype(F)
    lim = 128
    c0 = tuple(int(v) for v in cellHi)
    idx, g, mid, rg = cellinfo(c0)
    if idx and g > 0.5 and fine_at(c0):
        pu = np.clip((ro - cellHi) * F(16), F(0), pull).astype(F)
        ci = np.floor(pu)
        delta = (delta * F(1 / 16)).astype(F)
        side = ((step * (ci - pu) + step * F(0.5) + F(0.5)) * delta).astype(F)
        lim = 16
    for _ in range(20000):
        a = tie(side)
        t = side[a]
        side[a] = F(side[a] + delta[a])
        ci[a] += step[a]
        esc = (ci < 0).any() or (ci >= lim).any()
        if lim < 128:
            if not esc:
                B = cellHi * 16 + ci
                if solid(B):
                    return tuple(int(v) // 16 for v in B), a, float(t)
                continue
            delta = (delta * F(16)).astype(F)
            cellHi[a] += step[a]
            ci = cellHi.copy()
            lim = 128
            pl = (ro + rd * t - cellHi).astype(F)
            side = (t + (step * (-pl) + step * F(0.5) + F(0.5)) * delta).astype(F)
            esc = (ci < 0).any() or (ci >= 128).any()
        if esc:
            return None
        c = tuple(int(v) for v in ci)
        idx, g, mid, rg = cellinfo(c)
        if idx == 0:
            continue
        if not (g > 0.5 and fine_at(c)):
            return c, a, float(t)
        hi = ci.copy()
        pu = np.clip(((ro + rd * t) - hi) * F(16), F(0), pull).astype(F)
        su = np.floor(pu)
        if not solid(hi * 16 + su):
            cellHi = hi
            delta = (delta * F(1 / 16)).astype(F)
            side = (t + (step * (su - pu) + step * F(0.5) + F(0.5)) * delta).astype(F)
            ci = su
            lim = 16
            continue
        return c, a, float(t)
    return None


def u(h):
    return struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]


tally = {}
for line in open(records):
    m = re.search(r"px (\S+) verdict (\d) march cell (\S+) axis (\d) t (\S+) \| tree piece (\S+) "
                  r"axis (\d) t (\S+) .*bits (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)", line)
    g = m.groups()
    rd = np.array([u(b) for b in g[8:11]], F)
    ro = np.array([u(b) for b in g[11:]], F)
    ex = exact(ro, rd)
    tree = tuple(int(v) for v in g[5].split(","))
    gpu = (tuple(int(v) for v in g[2].split(",")), int(g[3]), float(g[4]))
    ok_tree = ex is not None and ex[0] == tree and ex[1] == int(g[6])
    r = march(ro, rd)
    ok_copy = r is not None and r[:2] == gpu[:2] and abs(r[2] - gpu[2]) < 2e-3
    n = march(ro, rd, pull=np.nextafter(F(16), F(0)))
    ok_pull = n is not None and ex is not None and n[:2] == (tuple(v // 16 for v in ex[0]), ex[1])
    k = ("tree exact" if ok_tree else "TREE WRONG",
         "copy reproduces GPU" if ok_copy else "COPY DIFFERS FROM GPU",
         "no-pull copy exact" if ok_pull else "no-pull copy still off")
    tally[k] = tally.get(k, 0) + 1
for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
    print("%4d  %s" % (v, " / ".join(k)))
