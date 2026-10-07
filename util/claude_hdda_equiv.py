"""An EXACT hierarchical walk: every cell at every level comes from
comparing plane-crossing times computed by ONE formula, never from rounding a
position. Compared in float32 (as the GPU runs it) against the plain 1 m walk
on a random 128^3 grid with its occupancy pyramid, including rays that are
axis-aligned, start on integer coordinates, or pass exactly through edges."""
import numpy as np

F = np.float32
S = 128
LMAX = 5
BIG = F(1e30)
rng = np.random.default_rng(7)
grid = np.zeros((S, S, S), np.uint8)
grid[:, :20, :] = 1
for _ in range(400):
    x, y, z = rng.integers(0, S, 3)
    grid[x, y, z] = 1
grid[60:64, 20:40, 60:64] = 1                      # a pillar
pyr = [grid]
for l in range(1, LMAX + 1):
    p = pyr[-1]
    n = p.shape[0] // 2
    pyr.append(p.reshape(n, 2, n, 2, n, 2).max(axis=(1, 3, 5)))


def tcross(plane, ro, step, delta):
    """THE formula: time at which the ray crosses `plane` on each axis (an
    axis the ray does not move along: never)"""
    return ((plane - ro) * step * delta + (F(1) - np.abs(step)) * BIG).astype(F)


def tie_axis(side):
    """the plain walk's own tie rule"""
    if side[0] < side[1] and side[0] < side[2]:
        return 0
    if side[1] < side[2]:
        return 1
    return 2


def plain(ro, rd, steps=20000):
    ro, rd = ro.astype(F), rd.astype(F)
    step = np.sign(rd).astype(F)
    delta = (F(1) / np.maximum(np.abs(rd), F(1e-8))).astype(F)
    ci = np.floor(ro).astype(F)
    side = tcross(ci + np.maximum(step, 0), ro, step, delta)
    for i in range(steps):
        a = tie_axis(side)
        t = side[a]
        ci[a] += step[a]
        side[a] = tcross(ci + np.maximum(step, 0), ro, step, delta)[a]
        if (ci < 0).any() or (ci >= S).any():
            return ("escape", i + 1)
        c = ci.astype(int)
        if grid[c[0], c[1], c[2]]:
            return ("hit", tuple(c), a, i + 1)
    return ("capped", steps)


def hier(ro, rd, steps=20000):
    ro, rd = ro.astype(F), rd.astype(F)
    step = np.sign(rd).astype(F)
    delta = (F(1) / np.maximum(np.abs(rd), F(1e-8))).astype(F)
    L = 0
    c = np.floor(ro).astype(np.int64)            # block index at level L
    t = F(0)                                       # time the ray entered this block
    axis = -1
    started = False                                # the start cell is never tested
    side = tcross(((c + (step > 0)) << L).astype(F), ro, step, delta)
    for i in range(steps):
        if started:
            # an occupied block: go down to the child the ray is in at time t
            while L > 0 and pyr[L][tuple(c)]:
                L -= 1
                mid = ((2 * c + 1) << L).astype(F)          # the middle planes
                tm = tcross(mid, ro, step, delta)
                # crossed already at time t? A middle plane crossed at the
                # SAME instant as the entry face counts as crossed only if
                # its axis outranks the entry axis: the plain walk's tie rule
                # takes simultaneous crossings z, then y, then x
                rank = np.arange(3) > axis
                crossed = (tm < t) | ((tm == t) & rank)
                upper = np.where(step > 0, crossed,
                                 np.where(step < 0, ~crossed, ro >= mid))
                c = 2 * c + upper.astype(np.int64)
                side = tcross(((c + (step > 0)) << L).astype(F), ro, step, delta)
            if L == 0 and grid[tuple(c)]:
                return ("hit", tuple(int(v) for v in c), axis, i)
            # an empty block: climb while the parent is empty too
            while L < LMAX and not pyr[L + 1][tuple(c >> 1)]:
                L += 1
                c = c >> 1
                side = tcross(((c + (step > 0)) << L).astype(F), ro, step, delta)
        started = True
        a = tie_axis(side)
        t = side[a]
        axis = a
        c[a] += int(step[a])
        if c[a] < 0 or c[a] >= (S >> L):
            return ("escape", i + 1)
        side = tcross(((c + (step > 0)) << L).astype(F), ro, step, delta)
    return ("capped", steps)


def compare(rays, label):
    bad, hs, ps = 0, 0, 0
    for ro, rd in rays:
        a, b = plain(ro, rd), hier(ro, rd)
        ps += a[-1]; hs += b[-1]
        if a[0] != b[0] or (a[0] == "hit" and a[1:3] != b[1:3]):
            bad += 1
            if bad <= 3:
                print("  MISMATCH", label, "ro", ro, "rd", rd, "plain", a, "hier", b)
    print("%-28s %4d rays  mismatches %d   steps plain %.1f  hierarchical %.1f"
          % (label, len(rays), bad, ps / len(rays), hs / len(rays)))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


random_rays = []
for _ in range(3000):
    ro = rng.uniform(1, S - 1, 3); ro[1] = rng.uniform(21, 60)
    random_rays.append((ro, unit(rng.normal(size=3))))
axis_rays = []
for _ in range(400):
    ro = rng.uniform(1, S - 1, 3); ro[1] = rng.uniform(21, 60)
    d = np.zeros(3); d[rng.integers(3)] = rng.choice([-1, 1])
    axis_rays.append((ro, d))
integer_rays = []
for _ in range(600):
    ro = rng.integers(1, S - 1, 3).astype(float); ro[1] = rng.integers(21, 60)
    d = rng.integers(-3, 4, 3).astype(float)
    if not d.any():
        d[0] = 1
    integer_rays.append((ro, unit(d)))                # through edges and corners exactly
compare(random_rays, "random")
compare(axis_rays, "axis-aligned")
compare(integer_rays, "integer starts, edge-on")
