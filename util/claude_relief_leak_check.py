#!/usr/bin/env python3
"""claude_relief_leak_check -- the relief's leak rules, checked on random worlds.

The engine carves each plain cube's EXPOSED faces (game.cpp
claudeReliefShapeId). Its claim: carving joins no two spaces that were apart,
i.e. every connected piece of carved air (6-connected 1/16 m voxels, the
path a ray's DDA can follow) touches only open cells that were already one
face-connected space. This builds random small worlds of cells (SOLID plain
cubes with random height maps and random shells D 1..3, OPEN cells, PARTIAL
cells = node boxes), carves every solid cell with the rules, labels the air at
1/16 m and checks the claim, plus "no carved air touches a PARTIAL cell".

The same worlds carved with the OLD rule (air within D of exactly one open
face, partial counted open, no edge rules) must FAIL, or the instrument is
blind (it found the twisted-corner hole on the first try: see --old).

A port of the C++ rules, not the C++ itself: keep the two in step.
usage: claude_relief_leak_check.py [--worlds N] [--size S] [--old] [--seed K]
"""
import argparse
import collections
import numpy as np

N = 16
SHELL = 3   # the deepest any cell carves (CLAUDE_RELIEF_SHELL)
SOLID, OPEN, PARTIAL = 0, 1, 2
DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def edge_index():
    tab = -np.ones((6, 6), dtype=int)
    e = 0
    for i in range(6):
        for j in range(i + 1, 6):
            if i // 2 != j // 2:
                tab[i, j] = tab[j, i] = e
                e += 1
    return tab


EDGE = edge_index()


def cls_at(world, p):
    x, y, z = p
    S = world.shape[0]
    if not (0 <= x < S and 0 <= y < S and 0 <= z < S):
        return OPEN          # the world is surrounded by open air
    return world[x, y, z]


def carve(depth, D, nbr_open, nbr_partial, diag, old=False, nbr_D=None):
    """one cell's 16^3 air mask [x, y, z]. depth: [6][16][16] per face
    (face order +X -X +Y -Y +Z -Z, in-plane coords (a, b) as the C++)."""
    air = np.zeros((N, N, N), dtype=bool)
    idx = np.arange(N)
    for k in range(6):
        if not nbr_open[k]:
            continue
        d = depth[k]
        for dd in range(D):
            m = d > dd                           # [a, b]
            if k == 0:   air[15 - dd][m] = True         # +X: y=a, z=b
            elif k == 1: air[dd][m] = True
            elif k == 2: air[:, 15 - dd, :][m] = True   # +Y: x=a, z=b
            elif k == 3: air[:, dd, :][m] = True
            elif k == 4: air[:, :, 15 - dd][m] = True   # +Z: x=a, y=b
            else:        air[:, :, dd][m] = True
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    dist = [15 - X, X, 15 - Y, Y, 15 - Z, Z]
    near = [dist[k] < D for k in range(6)]
    if old:
        n_open = sum(near[k] & bool(nbr_open[k] or nbr_partial[k]) for k in range(6))
        return air & (n_open == 1)
    keep = air.copy()
    for k in range(6):                       # (0)
        if nbr_partial[k]:
            keep &= ~near[k]
    for i in range(6):                       # (a)
        for j in range(i + 1, 6):
            e = EDGE[i, j]
            if e >= 0 and nbr_open[i] and nbr_open[j] and not diag[e]:
                keep &= ~(near[i] & near[j])
    for u in range(6):                       # (b')
        if not nbr_open[u]:
            continue
        for w in range(6):
            if EDGE[u, w] < 0 or nbr_open[w] or nbr_partial[w]:
                continue
            for v in range(6):
                if v // 2 == w // 2 or v == u or not diag[EDGE[w, v]]:
                    continue
                # Y = A+w carves its v face near here: p must reach A+v too
                ok = nbr_open[v] and EDGE[u, v] >= 0 and diag[EDGE[u, v]]
                # the block across w carves its v face within ITS shell
                bad = near[u] & near[w] & (dist[v] < nbr_D[w])
                if ok:
                    bad &= ~near[v]
                keep &= ~bad
    return keep


def label6(mask):
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return lab, n
    except ImportError:
        lab = np.zeros(mask.shape, dtype=np.int32)
        n = 0
        S = mask.shape
        for start in zip(*np.nonzero(mask)):
            if lab[start]:
                continue
            n += 1
            lab[start] = n
            q = collections.deque([start])
            while q:
                x, y, z = q.popleft()
                for dx, dy, dz in DIRS:
                    a, b, c = x + dx, y + dy, z + dz
                    if 0 <= a < S[0] and 0 <= b < S[1] and 0 <= c < S[2] \
                            and mask[a, b, c] and not lab[a, b, c]:
                        lab[a, b, c] = n
                        q.append((a, b, c))
        return lab, n


def run_world(rng, S, old, p_open, p_partial):
    world = rng.choice([SOLID, OPEN, PARTIAL], size=(S, S, S),
                       p=[1 - p_open - p_partial, p_open, p_partial])
    # voxel grid with a 1-cell OPEN border (the outside)
    T = S + 2
    vox = np.zeros((T * N, T * N, T * N), dtype=bool)        # air
    part = np.zeros_like(vox)
    cellid = -np.ones((T, T, T), dtype=int)                  # open cell index
    W = np.full((T, T, T), OPEN)
    W[1:-1, 1:-1, 1:-1] = world
    Dcell = rng.integers(0, 4, size=(T, T, T))      # every cell's own shell (0 = an uncarved cube)
    for x in range(T):
        for y in range(T):
            for z in range(T):
                sl = (slice(x * N, x * N + N), slice(y * N, y * N + N),
                      slice(z * N, z * N + N))
                c = W[x, y, z]
                if c == OPEN:
                    vox[sl] = True
                elif c == PARTIAL:
                    part[sl] = True
                elif 0 < x < T - 1 and 0 < y < T - 1 and 0 < z < T - 1:
                    p = (x, y, z)
                    nopen = [W[x + d[0], y + d[1], z + d[2]] == OPEN for d in DIRS]
                    npart = [W[x + d[0], y + d[1], z + d[2]] == PARTIAL for d in DIRS]
                    diag = [False] * 12
                    for i in range(6):
                        for j in range(i + 1, 6):
                            e = EDGE[i, j]
                            if e < 0:
                                continue
                            q = tuple(np.add(np.add(p, DIRS[i]), DIRS[j]))
                            diag[e] = W[q] == OPEN
                    D = int(Dcell[x, y, z])
                    nD = [int(Dcell[x + d[0], y + d[1], z + d[2]])
                          if W[x + d[0], y + d[1], z + d[2]] == SOLID else 0 for d in DIRS]
                    depth = rng.integers(0, D + 1, size=(6, N, N))
                    # long straight grooves too (the case that cut the old models)
                    for k in range(6):
                        if rng.random() < 0.5:
                            depth[k][int(rng.integers(0, N)), :] = D
                        if rng.random() < 0.5:
                            depth[k][:, int(rng.integers(0, N))] = D
                    vox[sl] = carve(depth, D, nopen, npart, diag, old, nD)
    # open-cell components (face adjacency), the truth before carving
    opencells = list(zip(*np.nonzero(W == OPEN)))
    for i, c in enumerate(opencells):
        cellid[c] = i
    parent = list(range(len(opencells)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for c in opencells:
        for d in DIRS:
            q = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if all(0 <= q[t] < T for t in range(3)) and W[q] == OPEN:
                ra, rb = find(cellid[c]), find(cellid[q])
                if ra != rb:
                    parent[ra] = rb
    lab, n = label6(vox)
    # each air label -> the open-cell components it touches
    comps = collections.defaultdict(set)
    for c in opencells:
        l = lab[c[0] * N, c[1] * N, c[2] * N]
        comps[l].add(find(cellid[c]))
    joins = sum(1 for s in comps.values() if len(s) > 1)
    # carved air face-adjacent to a partial cell
    carved = vox.copy()
    for c in opencells:
        carved[c[0] * N:(c[0] + 1) * N, c[1] * N:(c[1] + 1) * N,
               c[2] * N:(c[2] + 1) * N] = False
    touch = 0
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax], b[ax] = slice(0, -1), slice(1, None)
        touch += int((carved[tuple(a)] & part[tuple(b)]).sum()
                     + (carved[tuple(b)] & part[tuple(a)]).sum())
    return joins, touch, int(carved.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=40)
    ap.add_argument("--size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--old", action="store_true",
                    help="the pre-seams rule (must FAIL: the instrument's control)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    bad = 0
    tot_air = 0
    for w in range(a.worlds):
        p_open = float(rng.uniform(0.2, 0.55))
        p_part = float(rng.uniform(0.0, 0.15))
        joins, touch, air = run_world(rng, a.size, a.old, p_open, p_part)
        tot_air += air
        if joins or (touch and not a.old):
            bad += 1
            print("world %d: %d air pieces join separate spaces, %d air voxels touch a partial cell"
                  % (w, joins, touch))
    print("%s rule: %d of %d worlds leak; %d carved air voxels in all (the carving was live)"
          % ("OLD" if a.old else "seams", bad, a.worlds, tot_air))
    return 1 if (bad and not a.old) or (a.old and not bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
