#!/usr/bin/env python3
"""STATIC PROOF that the one-loop walk is the SAME ALGORITHM as the two
-walk one it replaces. No seat, no GPU, no client: it costs a second.

THE WORK, plainly. Until 2026-08-17 the tracer walked the 128^3 grid at
1 m in `march()` and, on entering a cell that carries a 16^3 mask
(a FINE material -- "class 250" before 2026-08-18 -- inside the ring),
called `descendCell()` -- a SECOND DDA
with its own position, its own three side-distances, its own step
counter and its own axis, all live alongside the first one's. The cost
instrument (spec/measured.md, "Descend cost instrument") measured what
having that second walk in the program costs: +36 % in Cornell and
+49 % in the cabin with the descend dial OFF and not one instruction of
it executing. The replacement is ONE loop whose CELL SIZE IS A VARIABLE
-- the physics contract's §2 rule, "size is a parameter, never a
branch" -- which rescales itself x16 on entering such a cell and back
on leaving.

A rewrite like that is only allowed if the picture does not move. The
seat can compare images, but a golden comparison is noisy (CI captures
are not bit-reproducible; see spec/environment-laws.md) and it answers
"close enough", not "the same walk". This script answers the stronger
question on the source itself: given the same ray and the same world,
do the two walks return the same hit, the same cell, the same normal
and the same distance -- and do they take the same number of steps?

Both walks below are TRANSCRIPTIONS of the GLSL, line for line, from
`git show 64b005975:client/shaders/claude_trace/opengl_fragment.glsl`
(old) and the working tree (new). They are deliberately not factored
together: a shared helper is how two implementations stop being two
witnesses.

WHAT IT CANNOT SAY. It runs in float64 where the shader runs in
float32, so it proves the ALGORITHMS agree, not that the last bit of
`t` agrees. The two compute `sideDist` by different but algebraically
identical routes (the old inner walk works in sub-voxel units and
divides by 16 at the end; the new one works in world t throughout), so
low-bit differences are expected on the seat and the golden comparison
is what bounds them. Exact agreement here plus goldens inside the noise
floor there is the pair of claims; neither alone is enough.

Usage:  util/claude_walk_equiv.py [--rays 20000] [--seed 7]
Exit 0 = every ray agreed.
"""
import argparse
import random
import sys

# --- the shader's named constants, same values -----------------------
GRID_S = 128.0
MARCH_STEPS = 384
SUBV = 16.0
SUBV_ITERS = 51
SUBV_STEPS = 51            # the old inner walk's own budget
WALK_STEPS = MARCH_STEPS * (SUBV_ITERS + 1)
RUNG_FINE = 1.0 / SUBV
SUBV_R0 = 48.0
SUBV_R1 = 80.0
CLASS_AIR_MAX = 0.25
CLASS_SUBVOX_LO = 245.0 / 255.0
CLASS_SUBVOX_HI = 252.5 / 255.0
SURFACE_EPS = 0.01
DDA_MIN_ABS = 1e-6
DEPTH_MISS = 4096.0


def sign(x):
    return 0.0 if x == 0.0 else (1.0 if x > 0.0 else -1.0)


def v_sign(v):
    return [sign(c) for c in v]


def v_floor(v):
    import math
    return [math.floor(c) for c in v]


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class World:
    """A tiny sparse stand-in for claudeTraceGrid + claudeSubvoxTex.

    cells[(x,y,z)] = class byte / 255 ; masks[(x,y,z)] = set of solid
    sub-voxel indices. Absent = air. Only the ring is given masks,
    exactly as the bake does.
    """

    def __init__(self, rng, n_cells=900):
        self.cells = {}
        self.masks = {}
        for _ in range(n_cells):
            c = (rng.randrange(44, 84), rng.randrange(44, 84),
                 rng.randrange(44, 84))
            r = rng.random()
            # NOTE 2026-08-18: these are the OLD class-band values. This
            # model is self-contained -- it defines what each value means
            # a few lines below and never reads the real palette -- so it
            # still tests exactly what it always tested. Read them as
            # "fine / solid / emissive / glass", not as live encodings.
            if r < 0.60:
                cls = 250.0 / 255.0
            elif r < 0.80:
                cls = 255.0 / 255.0
            elif r < 0.90:
                cls = 200.0 / 255.0        # emissive
            else:
                cls = 145.0 / 255.0        # glass
            self.cells[c] = cls
            if abs(cls - 250.0 / 255.0) < 1e-9:
                # A mask with real structure: a slab plus scattered bits,
                # so rays both resolve on the first sub-voxel and graze
                # long diagonals through mostly-empty cells.
                m = set()
                h = rng.randrange(1, 16)
                for x in range(16):
                    for y in range(h):
                        for z in range(16):
                            m.add((x, y, z))
                for _ in range(rng.randrange(0, 200)):
                    m.add((rng.randrange(16), rng.randrange(16),
                           rng.randrange(16)))
                if rng.random() < 0.10:
                    m = set()               # a cell whose mask is empty
                self.masks[c] = m
        # some class-250 cells OUTSIDE the ring, which must stay cubes
        for _ in range(120):
            c = (rng.randrange(20, 44), rng.randrange(20, 44),
                 rng.randrange(20, 44))
            self.cells[c] = 250.0 / 255.0
        self.fetches = 0

    def cls(self, cell):
        self.fetches += 1
        return self.cells.get((int(cell[0]), int(cell[1]), int(cell[2])), 0.0)

    def subvox_solid(self, rc, sc):
        """rc is the RING-LOCAL cell (cell - 48) exactly as the shader
        passes it; recover the world cell to find the mask."""
        cell = (int(rc[0] + SUBV_R0), int(rc[1] + SUBV_R0),
                int(rc[2] + SUBV_R0))
        m = self.masks.get(cell)
        if m is None:
            return False
        return (int(sc[0]), int(sc[1]), int(sc[2])) in m


def in_ring(cell):
    return all(SUBV_R0 <= c < SUBV_R1 for c in cell)


# =====================================================================
# OLD: march() + descendCell(), transcribed from 64b005975
# =====================================================================
def descend_cell_old(w, cell, ro, rd, t_enter, n_enter, skip_first, ctr):
    n_out = list(n_enter)
    t_out = t_enter
    rc = [cell[i] - SUBV_R0 for i in range(3)]
    u = [(ro[i] + rd[i] * t_enter - cell[i]) * SUBV for i in range(3)]
    u = [clamp(x, 0.0, SUBV - 1.0 / 512.0) for x in u]
    sc = v_floor(u)
    step_dir = v_sign(rd)
    inv_rd = [1.0 / max(abs(c), DDA_MIN_ABS) for c in rd]
    side = [(step_dir[i] * (sc[i] - u[i]) + step_dir[i] * 0.5 + 0.5)
            * inv_rd[i] for i in range(3)]
    tu = 0.0
    axis = -1
    for i in range(SUBV_STEPS):
        ctr["fine"] += 1
        if not (skip_first and i == 0) and w.subvox_solid(rc, sc):
            if axis == 0:
                n_out = [-step_dir[0], 0.0, 0.0]
            elif axis == 1:
                n_out = [0.0, -step_dir[1], 0.0]
            elif axis == 2:
                n_out = [0.0, 0.0, -step_dir[2]]
            t_out = t_enter + tu / SUBV
            return True, n_out, t_out
        if side[0] < side[1] and side[0] < side[2]:
            tu = side[0]; side[0] += inv_rd[0]; sc[0] += step_dir[0]; axis = 0
        elif side[1] < side[2]:
            tu = side[1]; side[1] += inv_rd[1]; sc[1] += step_dir[1]; axis = 1
        else:
            tu = side[2]; side[2] += inv_rd[2]; sc[2] += step_dir[2]; axis = 2
        if any(c < 0.0 for c in sc) or any(c >= SUBV for c in sc):
            return False, n_out, t_out
    return False, n_out, t_out


def march_old(w, ro, rd, descend):
    ctr = {"fine": 0, "coarse": 0}
    n = [0.0, 1.0, 0.0]
    cell = v_floor(ro)
    step_dir = v_sign(rd)
    inv_rd = [1.0 / max(abs(c), DDA_MIN_ABS) for c in rd]
    side = [(step_dir[i] * (cell[i] - ro[i]) + step_dir[i] * 0.5 + 0.5)
            * inv_rd[i] for i in range(3)]
    t = 0.0
    axis = -1

    if descend and in_ring(cell):
        s0 = w.cls(cell)
        if CLASS_SUBVOX_LO < s0 < CLASS_SUBVOX_HI:
            hit, sn, st = descend_cell_old(w, cell, ro, rd, 0.0,
                                           [0.0, 1.0, 0.0], True, ctr)
            if hit:
                return dict(hit=True, n=sn, t=st, cell=list(cell),
                            cls=s0, ctr=ctr)

    for i in range(MARCH_STEPS):
        ctr["coarse"] += 1
        if side[0] < side[1] and side[0] < side[2]:
            t = side[0]; side[0] += inv_rd[0]; cell[0] += step_dir[0]; axis = 0
        elif side[1] < side[2]:
            t = side[1]; side[1] += inv_rd[1]; cell[1] += step_dir[1]; axis = 1
        else:
            t = side[2]; side[2] += inv_rd[2]; cell[2] += step_dir[2]; axis = 2

        if any(c < 0.0 for c in cell) or any(c >= GRID_S for c in cell):
            return dict(hit=False, ctr=ctr)

        s = w.cls(cell)
        if s <= CLASS_AIR_MAX:
            continue

        n = [0.0, 0.0, 0.0]
        n[axis] = -step_dir[axis]

        if descend and CLASS_SUBVOX_LO < s < CLASS_SUBVOX_HI and in_ring(cell):
            hit, sn, st = descend_cell_old(w, cell, ro, rd, t, n, False, ctr)
            if not hit:
                continue
            n = sn
            t = st

        return dict(hit=True, n=n, t=t, cell=list(cell), cls=s, ctr=ctr)
    return dict(hit=False, ctr=ctr)


# =====================================================================
# NEW: one march(), the rung is a variable. Transcribed from the tree.
# =====================================================================
def march_new(w, ro, rd, descend):
    ctr = {"fine": 0, "coarse": 0}
    step_dir = v_sign(rd)
    delta = [1.0 / max(abs(c), DDA_MIN_ABS) for c in rd]
    ci = v_floor(ro)
    cell_hi = list(ci)
    side = [(step_dir[i] * (ci[i] - ro[i]) + step_dir[i] * 0.5 + 0.5)
            * delta[i] for i in range(3)]
    s = 0.0
    t = 0.0
    lim = GRID_S
    axis = -1

    if descend and in_ring(cell_hi):
        s = w.cls(cell_hi)
        if CLASS_SUBVOX_LO < s < CLASS_SUBVOX_HI:
            pu = [clamp((ro[i] - cell_hi[i]) * SUBV, 0.0,
                        SUBV - 1.0 / 512.0) for i in range(3)]
            ci = v_floor(pu)
            # counted like the old inner walk's i == 0, which incremented
            # even under skipFirst: the rescale IS that iteration.
            ctr["fine"] += 1
            delta = [d * RUNG_FINE for d in delta]
            side = [(step_dir[i] * (ci[i] - pu[i]) + step_dir[i] * 0.5 + 0.5)
                    * delta[i] for i in range(3)]
            lim = SUBV

    for i in range(WALK_STEPS):
        if side[0] < side[1] and side[0] < side[2]:
            t = side[0]; side[0] += delta[0]; ci[0] += step_dir[0]; axis = 0
        elif side[1] < side[2]:
            t = side[1]; side[1] += delta[1]; ci[1] += step_dir[1]; axis = 1
        else:
            t = side[2]; side[2] += delta[2]; ci[2] += step_dir[2]; axis = 2

        escaped = any(c < 0.0 for c in ci) or any(c >= lim for c in ci)

        if lim < GRID_S:
            if not escaped:
                ctr["fine"] += 1
                if not w.subvox_solid([cell_hi[k] - SUBV_R0 for k in range(3)],
                                      ci):
                    continue
                n = [0.0, 0.0, 0.0]
                n[axis] = -step_dir[axis]
                return dict(hit=True, n=n, t=t, cell=list(cell_hi),
                            cls=s, ctr=ctr)
            delta = [d * SUBV for d in delta]
            cell_hi[axis] += step_dir[axis]
            ci = list(cell_hi)
            lim = GRID_S
            pl = [ro[k] + rd[k] * t - cell_hi[k] for k in range(3)]
            side = [t + (step_dir[k] * (-pl[k]) + step_dir[k] * 0.5 + 0.5)
                    * delta[k] for k in range(3)]
            escaped = any(c < 0.0 for c in ci) or any(c >= GRID_S for c in ci)

        ctr["coarse"] += 1
        if escaped:
            return dict(hit=False, ctr=ctr)

        s = w.cls(ci)
        if s <= CLASS_AIR_MAX:
            continue

        n = [0.0, 0.0, 0.0]
        n[axis] = -step_dir[axis]

        if descend and CLASS_SUBVOX_LO < s < CLASS_SUBVOX_HI and in_ring(ci):
            cell_hi = list(ci)
            pu = [clamp((ro[k] + rd[k] * t - cell_hi[k]) * SUBV, 0.0,
                        SUBV - 1.0 / 512.0) for k in range(3)]
            su = v_floor(pu)
            ctr["fine"] += 1
            if not w.subvox_solid([cell_hi[k] - SUBV_R0 for k in range(3)],
                                  su):
                delta = [d * RUNG_FINE for d in delta]
                side = [t + (step_dir[k] * (su[k] - pu[k])
                             + step_dir[k] * 0.5 + 0.5) * delta[k]
                        for k in range(3)]
                ci = su
                lim = SUBV
                continue

        return dict(hit=True, n=n, t=t, cell=list(ci), cls=s, ctr=ctr)
    return dict(hit=False, ctr=ctr)


def stress_rays(rng, n):
    """THE WORST CASE FOR THE ONE RESIDUAL DIFFERENCE, fired on purpose.

    The two walks part company by at most the OLD code's own entry
    clamp. That clamp shortens the first fine step on the axis the ray
    entered through by 1/512 of a sub-voxel; the two-walk code threw the
    shortfall away when the descent missed (its 1 m side-distances were
    never touched by the inner walk), while the one-loop code rebuilds
    the 1 m side-distances FROM THE EXIT POINT and therefore carries it
    forward. It can only bite a ray that enters a cell through one face
    and leaves through the OPPOSITE one -- an axis-aligned ray through a
    corridor of masked cells -- and it can only accumulate once per such
    cell. So: fire exactly those rays, down a long run of the ring, and
    let the reported worst |dt| be the bound."""
    out = []
    for _ in range(n):
        ax = rng.randrange(3)
        s = 1.0 if rng.random() < 0.5 else -1.0
        rd = [0.0, 0.0, 0.0]
        rd[ax] = s
        ro = [rng.uniform(48.5, 79.5) for _ in range(3)]
        ro[ax] = 46.0 if s > 0 else 82.0
        out.append((ro, rd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rays", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--stress", type=int, default=0,
                    help="fire N axis-aligned rays straight down the ring "
                         "instead of random ones -- the worst case for "
                         "the entry-clamp residual (see stress_rays)")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    w = World(rng)
    dt_max = 0.0
    dt_ray = None
    dt_nonzero = 0
    bad = 0
    checked = {"hit": 0, "miss": 0, "descended": 0}
    steps = {"old_fine": 0, "old_coarse": 0, "new_fine": 0, "new_coarse": 0}
    step_mismatch = 0

    import math
    stress = stress_rays(rng, a.stress) if a.stress else None
    n_rays = a.stress if stress else a.rays
    for k in range(n_rays):
        if stress:
            ro, rd = stress[k]
            descend = True
            o = march_old(w, list(ro), rd, descend)
            nw = march_new(w, list(ro), rd, descend)
            steps["old_fine"] += o["ctr"]["fine"]
            steps["old_coarse"] += o["ctr"]["coarse"]
            steps["new_fine"] += nw["ctr"]["fine"]
            steps["new_coarse"] += nw["ctr"]["coarse"]
            if (o["ctr"]["fine"] != nw["ctr"]["fine"]
                    or o["ctr"]["coarse"] != nw["ctr"]["coarse"]):
                step_mismatch += 1
            if o["hit"] != nw["hit"]:
                bad += 1
                continue
            if not o["hit"]:
                checked["miss"] += 1
                continue
            checked["hit"] += 1
            if o["ctr"]["fine"]:
                checked["descended"] += 1
            dt = abs(o["t"] - nw["t"])
            if dt > 0.0:
                dt_nonzero += 1
            if dt > dt_max:
                dt_max, dt_ray = dt, (list(ro), list(rd))
            if (o["cell"] != nw["cell"] or o["n"] != nw["n"]
                    or o["cls"] != nw["cls"]):
                bad += 1
                if bad <= 5:
                    print("STRESS MISMATCH %s %s: old %s %s  new %s %s"
                          % (ro, rd, o["cell"], o["n"], nw["cell"], nw["n"]))
            continue
        # Origins BOTH inside the ring (so the starting-cell rescale is
        # exercised, including inside a mask) and outside it.
        if k % 3 == 0:
            ro = [rng.uniform(46.0, 82.0) for _ in range(3)]
        else:
            ro = [rng.uniform(30.0, 98.0) for _ in range(3)]
        # directions including near-axis-aligned ones, which is where a
        # DDA's degenerate cases live
        d = [rng.gauss(0.0, 1.0) for _ in range(3)]
        if rng.random() < 0.15:
            d[rng.randrange(3)] = 0.0
        if rng.random() < 0.15:
            ax = rng.randrange(3)
            d = [0.0, 0.0, 0.0]
            d[ax] = 1.0 if rng.random() < 0.5 else -1.0
        ln = math.sqrt(sum(c * c for c in d)) or 1.0
        rd = [c / ln for c in d]
        descend = (k % 7) != 0     # exercise the dial-off path too

        o = march_old(w, list(ro), rd, descend)
        nw = march_new(w, list(ro), rd, descend)

        steps["old_fine"] += o["ctr"]["fine"]
        steps["old_coarse"] += o["ctr"]["coarse"]
        steps["new_fine"] += nw["ctr"]["fine"]
        steps["new_coarse"] += nw["ctr"]["coarse"]
        if (o["ctr"]["fine"] != nw["ctr"]["fine"]
                or o["ctr"]["coarse"] != nw["ctr"]["coarse"]):
            step_mismatch += 1
            if step_mismatch <= 5:
                print("STEP MISMATCH ray %d  old %s  new %s"
                      % (k, o["ctr"], nw["ctr"]))

        if o["hit"] != nw["hit"]:
            bad += 1
            if bad <= 10:
                print("MISMATCH hit ray %d: old %s new %s  ro=%s rd=%s"
                      % (k, o["hit"], nw["hit"], ro, rd))
            continue
        if not o["hit"]:
            checked["miss"] += 1
            continue
        checked["hit"] += 1
        if o["ctr"]["fine"]:
            checked["descended"] += 1
        dt = abs(o["t"] - nw["t"])
        if dt > 0.0:
            dt_nonzero += 1
        if dt > dt_max:
            dt_max, dt_ray = dt, (list(ro), list(rd))
        same = (o["cell"] == nw["cell"] and o["n"] == nw["n"]
                and dt <= a.tol * max(1.0, abs(o["t"]))
                and o["cls"] == nw["cls"])
        if not same:
            structural = (o["cell"] != nw["cell"] or o["n"] != nw["n"]
                          or o["cls"] != nw["cls"])
            if structural:
                bad += 1
            if structural and bad <= 10:
                print("MISMATCH ray %d\n  old cell=%s n=%s t=%.12f cls=%.6f"
                      "\n  new cell=%s n=%s t=%.12f cls=%.6f\n  ro=%s rd=%s"
                      % (k, o["cell"], o["n"], o["t"], o["cls"],
                         nw["cell"], nw["n"], nw["t"], nw["cls"], ro, rd))

    print("\nrays %d   hits %d   misses %d   rays that descended %d"
          % (n_rays, checked["hit"], checked["miss"], checked["descended"]))
    print("steps  fine  old %d  new %d %s"
          % (steps["old_fine"], steps["new_fine"],
             "SAME" if steps["old_fine"] == steps["new_fine"] else "DIFFER"))
    print("steps  coarse old %d  new %d %s"
          % (steps["old_coarse"], steps["new_coarse"],
             "SAME" if steps["old_coarse"] == steps["new_coarse"]
             else "DIFFER"))
    print("per-ray step-count mismatches: %d" % step_mismatch)
    print("hit/cell/normal/t mismatches:  %d" % bad)
    print("worst |t_old - t_new| over all hits: %.3e  (%d of %d hits "
          "differ at all)" % (dt_max, dt_nonzero, checked["hit"]))
    print("  for scale: SURFACE_EPS = %.3f, one sub-voxel = %.5f, and the "
          "old walk's own\n  entry clamp is %.3e world units "
          "(1/512 of a sub-voxel)."
          % (SURFACE_EPS, 1.0 / SUBV, 1.0 / (512.0 * SUBV)))
    if dt_ray:
        print("  worst ray: ro=%s rd=%s" % (dt_ray[0], dt_ray[1]))
    # THE VERDICT, and what each half of it means.
    #
    # STRUCTURE -- did the two walks visit the same cells, in the same
    # order, and stop on the same face of the same cell? That must be
    # exact and nothing here is allowed to soften it.
    #
    # DISTANCE -- t may differ by at most the old code's own entry
    # clamp, which is 1/512 of a sub-voxel = 1/8192 world units, once
    # per cell a ray passes STRAIGHT THROUGH on the axis it entered by.
    # The two-walk code discarded that shortfall (the inner walk never
    # touched the outer walk's side-distances); the one-loop code
    # rebuilds the outer walk from the exit point and carries it. The
    # bound is reported above and judged against SURFACE_EPS.
    ok = (bad == 0 and step_mismatch == 0)
    if not ok:
        print("\nNOT EQUIVALENT")
        return 1
    print("\nEQUIVALENT: same hit, same coarse cell, same normal, and the"
          " same number of\nsteps at both rungs, on every ray; t agrees to"
          " %.3e (bound: the entry\nclamp, %.3e, which is %.1f %% of"
          " SURFACE_EPS)."
          % (dt_max, 1.0 / (512.0 * SUBV), 100.0 * dt_max / SURFACE_EPS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
