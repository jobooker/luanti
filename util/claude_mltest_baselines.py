#!/usr/bin/env python3
"""claude_mltest_baselines — step 3 of spec/ml-light-test-plan.md: the rule
baselines any network has to beat, scored on places held out from fitting.

  B0  one number: the mean bounce of the training faces
  B1  six numbers: the mean per face direction
  B2  sky openness: the share of the face's cosine-weighted hemisphere whose
      rays leave the grid upward through nothing but air (marched through the
      dumped grid in quarter-cell steps: approximate at cell corners), and a
      per-direction linear fit of log bounce on it

Scores (per held-out face, weighted by pixel count): median and 90th
percentile relative error of luminance, and R^2 of log luminance; next to the
split-half floor (the two halves of the truth against each other).

  python3 util/claude_mltest_baselines.py RUN_DIR [--test v1 v2 ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from claude_mltest_faces import FACES, read_grid  # noqa: E402

RAYS = 64
STEP = 0.25
MAXT = 96.0
LUM = np.array([0.2126, 0.7152, 0.0722])


def hemisphere(n, k, rng):
    """k cosine-weighted directions about axis normal n (one of the six)"""
    u1, u2 = rng.random(k), rng.random(k)
    r, phi = np.sqrt(u1), 2 * np.pi * u2
    local = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], 1)
    a = int(np.argmax(np.abs(n)))
    t1 = np.eye(3)[(a + 1) % 3]
    t2 = np.eye(3)[(a + 2) % 3]
    return local[:, :1] * t1 + local[:, 1:2] * t2 + local[:, 2:3] * n


def sky_openness(cells, faces, occ, pal, seed=0):
    S = occ.shape[0]
    mat = occ[..., 3]
    blocks = (mat != 0) & ~(pal[mat, 2] > 0.5)          # opaque to a sky ray
    rng = np.random.default_rng(seed)
    out = np.zeros(len(cells))
    for f in range(6):
        idx = np.nonzero(faces == f)[0]
        if len(idx) == 0:
            continue
        n = FACES[f].astype(float)
        dirs = hemisphere(n, RAYS, rng)                   # same set per face direction
        o = cells[idx].astype(float) + 0.5 + 0.5 * n + 1e-3 * n   # just outside the face
        p = o[:, None, :] + 0 * dirs[None]                # (faces, rays, 3)
        alive = np.ones(p.shape[:2], bool)
        sky = np.zeros(p.shape[:2], bool)
        for _ in range(int(MAXT / STEP)):
            p = p + STEP * dirs[None]
            c = np.floor(p).astype(int)
            out_of = ((c < 0) | (c >= S)).any(2)
            sky |= alive & out_of & (dirs[None, :, 1] > 0)
            alive &= ~out_of
            cz = np.clip(c, 0, S - 1)
            hit = blocks[cz[..., 2], cz[..., 1], cz[..., 0]] & alive
            alive &= ~hit
            if not alive.any():
                break
        sky |= alive & (dirs[None, :, 1] > 0)             # still flying at MAXT, upward
        out[idx] = sky.mean(1)
    return out


def score(pred, truth, w, name):
    lp, lt = np.log(np.maximum(pred @ LUM, 1e-5)), np.log(np.maximum(truth @ LUM, 1e-5))
    rel = np.abs(np.exp(lp - lt) - 1)
    order = np.argsort(rel)
    cw = np.cumsum(w[order]) / w.sum()
    med, p90 = rel[order][np.searchsorted(cw, 0.5)], rel[order][np.searchsorted(cw, 0.9)]
    r2 = 1 - np.sum(w * (lp - lt) ** 2) / np.sum(w * (lt - np.average(lt, weights=w)) ** 2)
    print("%-26s median rel err %6.1f%%   90th %6.1f%%   R2(log) %6.3f" % (name, 100 * med, 100 * p90, r2))
    return {"median": float(med), "p90": float(p90), "r2": float(r2)}


def main():
    run = sys.argv[1]
    test = sys.argv[sys.argv.index("--test") + 1:] if "--test" in sys.argv else None
    z = np.load(os.path.join(run, "faces.npz"))
    names = sorted({k.split("/")[0] for k in z.files})
    if not test:
        test = names[::4]                                  # every 4th place held out
    data = {}
    for nm in names:
        S, origin, occ, pal = read_grid(os.path.join(run, nm, "grid.bin"))
        cells, faces = z[nm + "/cell"], z[nm + "/face"]
        data[nm] = dict(cell=cells, face=faces, y=z[nm + "/bounce"], w=z[nm + "/pixels"].astype(float),
                        split=z[nm + "/split"], sky=sky_openness(cells, faces, occ, pal))
        print("%-18s faces %6d  mean sky openness %.2f" % (nm, len(cells), data[nm]["sky"].mean()), flush=True)
    tr = [n for n in names if n not in test]
    cat = lambda key, ns: np.concatenate([data[n][key] for n in ns])
    ytr, yte = cat("y", tr), cat("y", test)
    ftr, fte = cat("face", tr), cat("face", test)
    str_, ste = cat("sky", tr), cat("sky", test)
    wte = cat("w", test)
    print("train places %s\ntest places  %s\ntrain faces %d, test faces %d" % (tr, test, len(ytr), len(yte)))
    res = {}
    # the floor: the halves of the truth disagree by `split` (relative)
    sp = cat("split", test)
    print("%-26s median rel err %6.1f%%   (the two halves of the truth)" % ("split-half floor", 100 * np.median(sp) / 2))
    res["B0"] = score(np.tile(ytr.mean(0), (len(yte), 1)), yte, wte, "B0 one mean")
    m1 = np.stack([ytr[ftr == f].mean(0) if (ftr == f).any() else ytr.mean(0) for f in range(6)])
    res["B1"] = score(m1[fte], yte, wte, "B1 mean per direction")
    p2 = np.zeros_like(yte)
    for f in range(6):
        a, b = ftr == f, fte == f
        if not a.any() or not b.any():
            p2[b] = m1[f]
            continue
        X = np.stack([str_[a], np.ones(a.sum())], 1)
        for ch in range(3):
            coef, *_ = np.linalg.lstsq(X, np.log(np.maximum(ytr[a, ch], 1e-5)), rcond=None)
            p2[b, ch] = np.exp(np.stack([ste[b], np.ones(b.sum())], 1) @ coef)
    res["B2"] = score(p2, yte, wte, "B2 sky openness fit")
    json.dump({"test": test, "train": tr, "scores": res}, open(os.path.join(run, "baselines.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
