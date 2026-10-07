"""The coarse walk in float32 (as the GPU runs it), plain vs the original
leap vs the guarded leap, on a random 128^3 grid with its pyramid."""
import numpy as np

F = np.float32
S = 128
WALK_STEPS = 256
rng = np.random.default_rng(1)
grid = np.zeros((S, S, S), np.uint8)
grid[:, :20, :] = 1
for _ in range(300):
    x, y, z = rng.integers(0, S, 3)
    grid[x, y, z] = 1
pyr = [grid.copy()]
for l in range(1, 6):
    p = pyr[-1]
    n = p.shape[0] // 2
    pyr.append(p.reshape(n, 2, n, 2, n, 2).max(axis=(1, 3, 5)))


def walk(ro, rd, mode):
    ro = ro.astype(F); rd = rd.astype(F)
    step = np.sign(rd).astype(F)
    delta = (F(1) / np.maximum(np.abs(rd), F(1e-8))).astype(F)
    ci = np.floor(ro).astype(F)
    side = ((step * (ci - ro) + step * F(0.5) + F(0.5)) * delta).astype(F)
    t = F(0)
    for i in range(WALK_STEPS):
        if side[0] < side[1] and side[0] < side[2]:
            a = 0
        elif side[1] < side[2]:
            a = 1
        else:
            a = 2
        t = side[a]
        side[a] = F(side[a] + delta[a])
        ci[a] = F(ci[a] + step[a])
        if (ci < 0).any() or (ci >= S).any():
            return ("escape", i + 1)
        c = ci.astype(int)
        if grid[c[0], c[1], c[2]] != 0:
            return ("hit", tuple(c), a, i + 1)
        if mode == "plain":
            continue
        L = 0
        for l in range(1, 6):
            b = c >> l
            if pyr[l][b[0], b[1], b[2]] > 0:
                break
            L = l
        if L == 0:
            continue
        bs = F(1 << L)
        blo = (np.floor(ci / bs) * bs).astype(F)
        bhi = (blo + bs).astype(F)
        pl = np.where(step >= 0, bhi, blo).astype(F)
        tx = np.where(step != 0, (pl - ro) * step * delta, F(1e30)).astype(F)
        tex = tx.min()
        if mode == "guarded" and not (tex > t + F(1e-2)):
            continue                      # the exit is not clearly ahead
        tm = max(t, F(tex - F(1e-3)))
        cn = np.clip(np.floor(ro + rd * tm), blo, bhi - F(1)).astype(F)
        if mode == "guarded" and ((cn - ci) * step < 0).any():
            continue                      # never land behind the walk
        ci = cn
        side = ((step * (ci - ro) + step * F(0.5) + F(0.5)) * delta).astype(F)
    return ("capped", WALK_STEPS)


stats = {"leap": [0, 0, 0], "guarded": [0, 0, 0]}   # mismatches, capped, steps
plain_steps = 0
N = 4000
for k in range(N):
    ro = rng.uniform(1, S - 1, 3)
    ro[1] = rng.uniform(21, 60)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    a = walk(ro, d, "plain")
    plain_steps += a[-1]
    for m in ("leap", "guarded"):
        b = walk(ro, d, m)
        stats[m][2] += b[-1]
        if b[0] == "capped":
            stats[m][1] += 1
        if a[0] == "capped":
            continue                       # the plain walk ran out of steps: no answer to compare
        if a[0] != b[0] or (a[0] == "hit" and a[1:3] != b[1:3]):
            stats[m][0] += 1
print("plain: mean steps %.1f" % (plain_steps / N))
for m, (mm, cap, st) in stats.items():
    print("%-8s mismatches vs plain %4d / %d   capped %4d   mean steps %.1f" % (m, mm, N, cap, st / N))
