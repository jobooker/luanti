"""claude_tree_equiv — ladder plan stage 1a: an EXACT walk over a sparse tree
whose branching factor is a parameter PER LEVEL, from the 1/16 m base rung
up, against the plain walk at the base rung. float32, as the GPU runs it.

Generalises util/claude_hdda_equiv.py (factor 2, from 1 m up): descending
picks the child by counting how many of the f-1 inner planes the ray has
already crossed at the entry time t, every crossing time from the one
tcross() formula and ties broken by the plain walk's own rule (z > y > x).

  python3 util/claude_tree_equiv.py            # factors 2 and 4
"""
import numpy as np

F = np.float32
BIG = F(1e30)
rng = np.random.default_rng(11)
S = 256                       # base cells per axis (16 m at 1/16 m)

# --- a world at 1/16 m: ground, 1 m blocks, carved shapes, loose voxels
grid = np.zeros((S, S, S), np.uint8)
grid[:, :32, :] = 1                                   # 2 m of ground
for _ in range(60):                                   # whole 1 m blocks
    x, y, z = rng.integers(0, 16, 3) * 16
    grid[x:x + 16, max(y, 32):max(y, 32) + 16, z:z + 16] = 1
shape = rng.random((16, 16, 16)) < 0.45               # one carved "model"
for _ in range(40):                                   # placed many times (shared shape)
    x, z = rng.integers(0, 16, 2) * 16
    grid[x:x + 16, 32:48, z:z + 16] |= shape
for _ in range(500):                                  # loose single voxels
    x, y, z = rng.integers(0, S, 3)
    grid[x, y, z] = 1


def build(factors):
    """occupancy per level: level 0 = base cells; level l+1 = any child set"""
    lv = [grid]
    for f in factors:
        p = lv[-1]
        n = p.shape[0] // f
        lv.append(p.reshape(n, f, n, f, n, f).max(axis=(1, 3, 5)))
    size = [1]
    for f in factors:
        size.append(size[-1] * f)
    return lv, size


def tcross(plane, ro, step, delta):
    return ((plane - ro) * step * delta + (F(1) - np.abs(step)) * BIG).astype(F)


def tie_axis(side):
    if side[0] < side[1] and side[0] < side[2]:
        return 0
    if side[1] < side[2]:
        return 1
    return 2


def plain(ro, rd, steps=40000):
    step = np.sign(rd).astype(F)
    delta = (F(1) / np.maximum(np.abs(rd), F(1e-8))).astype(F)
    ci = np.floor(ro).astype(F)
    side = tcross(ci + np.maximum(step, 0), ro, step, delta)
    for i in range(steps):
        a = tie_axis(side)
        ci[a] += step[a]
        side[a] = tcross(ci + np.maximum(step, 0), ro, step, delta)[a]
        if (ci < 0).any() or (ci >= S).any():
            return ("escape", i + 1)
        c = ci.astype(int)
        if grid[c[0], c[1], c[2]]:
            return ("hit", tuple(c), a, i + 1)
    return ("capped", steps)


def tree(ro, rd, lv, size, factors, steps=40000):
    LMAX = len(factors)
    step = np.sign(rd).astype(F)
    delta = (F(1) / np.maximum(np.abs(rd), F(1e-8))).astype(F)
    L = 0
    c = np.floor(ro).astype(np.int64)
    t = F(0)
    axis = -1
    started = False

    def faces(c, L):
        return tcross(((c + (step > 0)) * size[L]).astype(F), ro, step, delta)

    side = faces(c, L)
    for i in range(steps):
        if started:
            while L > 0 and lv[L][tuple(c)]:
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
                    for j in range(1, f):                     # inner planes, in the ray's order
                        jj = j if step[ax] > 0 else f - j
                        tm = tcross(F((lo + jj) * cs), ro[ax:ax + 1], step[ax:ax + 1], delta[ax:ax + 1])[0]
                        if tm < t or (tm == t and ax > axis):
                            n += 1
                        else:
                            break
                    k[ax] = n if step[ax] > 0 else f - 1 - n
                c = c * f + k
                side = faces(c, L)
            if L == 0 and grid[tuple(c)]:
                return ("hit", tuple(int(v) for v in c), axis, i)
            while L < LMAX and not lv[L + 1][tuple(c // factors[L])]:
                c = c // factors[L]
                L += 1
                side = faces(c, L)
        started = True
        a = tie_axis(side)
        t = side[a]
        axis = a
        c[a] += int(step[a])
        if c[a] < 0 or c[a] >= S // size[L]:
            return ("escape", i + 1)
        side = faces(c, L)
    return ("capped", steps)


def unit(v):
    v = np.asarray(v, float)
    return (v / np.linalg.norm(v)).astype(F)


def rays():
    out = {"random": [], "axis-aligned": [], "integer starts, edge-on": []}
    for _ in range(1500):
        ro = rng.uniform(1, S - 1, 3).astype(F); ro[1] = F(rng.uniform(33, 200))
        out["random"].append((ro, unit(rng.normal(size=3))))
    for _ in range(300):
        ro = rng.uniform(1, S - 1, 3).astype(F); ro[1] = F(rng.uniform(33, 200))
        d = np.zeros(3); d[rng.integers(3)] = rng.choice([-1, 1])
        out["axis-aligned"].append((ro, d.astype(F)))
    for _ in range(500):
        ro = rng.integers(1, S - 1, 3).astype(F); ro[1] = F(rng.integers(33, 200))
        d = rng.integers(-3, 4, 3).astype(float)
        if not d.any():
            d[0] = 1
        out["integer starts, edge-on"].append((ro, unit(d)))
    return out


R = rays()
plain_res = {k: [plain(ro, rd) for ro, rd in v] for k, v in R.items()}
for name, factors in (("factor 2 (octree)", [2] * 8), ("factor 4 (64-tree)", [4] * 4),
                      ("mixed 4,4,2,2,2,2", [4, 4, 2, 2, 2, 2])):
    lv, size = build(factors)
    print("==", name, "levels (base cells):", size)
    for k, v in R.items():
        bad, ps, ts = 0, 0, 0
        for (ro, rd), a in zip(v, plain_res[k]):
            b = tree(ro, rd, lv, size, factors)
            ps += a[-1]; ts += b[-1]
            if a[0] != b[0] or (a[0] == "hit" and a[1:3] != b[1:3]):
                bad += 1
                if bad <= 2:
                    print("   MISMATCH", k, ro, rd, a, b)
        print("   %-26s %5d rays  mismatches %d   steps: plain %.0f  tree %.1f" % (k, len(v), bad, ps / len(v), ts / len(v)))
