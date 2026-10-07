#!/usr/bin/env python3
"""claude_mltest_faces — step 0 and step 2 of spec/ml-light-test-plan.md:
decode the face views, prove they are real, and build the per-face table.

  python3 util/claude_mltest_faces.py RUN_DIR     -> RUN_DIR/faces.npz + a report
  python3 util/claude_mltest_faces.py --selftest  (no data needed)

Per vantage it reads faces.png (claude_view 22), bounce_a/b.png (view 23,
stored as x/(1+x)) and grid.bin (claude_export_grid), and keeps a pixel only
if it is TRUSTWORTHY:
  * its 2x2 neighbourhood decodes to the same code (the tracer runs at half
    resolution and the present pass upsamples, which blends codes at edges)
  * the decoded cell is NOT AIR in the dumped grid, and the cell across the
    decoded face IS air or transmissive (it is a surface the camera can see)
Per (cell, face) it averages the decoded bounce of each half, and records
the pixel count and the split-half disagreement (the noise floor).
"""
import json
import os
import sys

import numpy as np
from PIL import Image

FACES = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
MIN_PIXELS = 8          # TUNED: faces seen by fewer pixels are too noisy | learn by: split-half spread vs count
MAX_SPLIT = 0.20        # TUNED: halves disagreeing more = unconverged | learn by: the split-half spread itself


def read_grid(path):
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"\n"):
            head += f.read(1)
        _, S, ox, oy, oz = head.decode().split()
        S = int(S)
        occ = np.frombuffer(f.read(S ** 3 * 4), np.uint8).reshape(S, S, S, 4)  # [z, y, x, c]
        pal = np.frombuffer(f.read(256 * 8 * 4), np.float32).reshape(256, 8)
    return S, (int(ox), int(oy), int(oz)), occ, pal


def decode_faces(img):
    """HxWx3 uint8 -> (cells HxWx3 int, face HxW int, valid HxW bool)"""
    a = img.astype(np.int32)
    cells = a & 127
    fbits = a >> 7
    face = fbits[..., 0] | (fbits[..., 1] << 1) | (fbits[..., 2] << 2)
    none = (a == 255).all(axis=2)
    valid = ~none & (face <= 5)
    return cells, face, valid


def decode_bounce(img):
    v = img.astype(np.float64) / 255.0
    v = np.clip(v, 0.0, 254.5 / 255.0)
    return v / (1.0 - v)


def stable_2x2(img):
    """True where the pixel equals its right, lower and lower-right
    neighbours (codes are exact or garbage, never close)"""
    a = img.astype(np.int32)
    same = np.ones(a.shape[:2], bool)
    same[:-1, :-1] = ((a[:-1, :-1] == a[1:, :-1]).all(2) & (a[:-1, :-1] == a[:-1, 1:]).all(2)
                      & (a[:-1, :-1] == a[1:, 1:]).all(2))
    same[-1, :] = False
    same[:, -1] = False
    return same


def vantage_faces(d):
    S, origin, occ, pal = read_grid(os.path.join(d, "grid.bin"))
    fimg = np.asarray(Image.open(os.path.join(d, "faces.png")).convert("RGB"))
    ba = decode_bounce(np.asarray(Image.open(os.path.join(d, "bounce_a.png")).convert("RGB")))
    bb = decode_bounce(np.asarray(Image.open(os.path.join(d, "bounce_b.png")).convert("RGB")))
    cells, face, valid = decode_faces(fimg)
    n_any = int(valid.sum())
    keep = valid & stable_2x2(fimg)
    n_stable = int(keep.sum())
    # the grid check: decoded cell not air, the cell across the face open
    mat = occ[..., 3]
    transmits = pal[:, 2] > 0.5
    ys, xs = np.nonzero(keep)
    c = cells[ys, xs]                         # (x, y, z)
    f = face[ys, xs]
    nb = c + FACES[f]
    inside = (nb >= 0).all(1) & (nb < S).all(1)
    solid = mat[c[:, 2], c[:, 1], c[:, 0]] != 0
    nbz = np.clip(nb, 0, S - 1)
    open_ = mat[nbz[:, 2], nbz[:, 1], nbz[:, 0]]
    open_ = (open_ == 0) | transmits[open_]
    ok = inside & solid & open_
    report = {"pixels_coded": n_any, "pixels_2x2_stable": n_stable,
              "pixels_grid_valid": int(ok.sum()),
              "grid_valid_share_of_stable": float(ok.mean()) if len(ok) else 0.0}
    ys, xs, c, f = ys[ok], xs[ok], c[ok], f[ok]
    key = ((c[:, 0] * S + c[:, 1]) * S + c[:, 2]) * 6 + f
    uk, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    sa = np.zeros((len(uk), 3)); sb = np.zeros((len(uk), 3))
    np.add.at(sa, inv, ba[ys, xs]); np.add.at(sb, inv, bb[ys, xs])
    ma, mb = sa / counts[:, None], sb / counts[:, None]
    la, lb = ma.mean(1), mb.mean(1)
    split = np.abs(la - lb) / np.maximum((la + lb) / 2, 1e-6)
    good = (counts >= MIN_PIXELS) & (split <= MAX_SPLIT)
    report.update(faces=int(len(uk)), faces_kept=int(good.sum()),
                  split_median=float(np.median(split[good])) if good.any() else None)
    cell = np.stack([(uk // 6) // (S * S), ((uk // 6) // S) % S, (uk // 6) % S], 1)
    return {"cell": cell[good], "face": (uk % 6)[good], "bounce": ((ma + mb) / 2)[good],
            "split": split[good], "pixels": counts[good], "origin": origin}, report


def selftest():
    """a made-up 128^3 world with one floor, a coded view of it, and
    bounce images that agree: every check must pass, and corrupting the
    code must be caught"""
    import tempfile
    S = 128
    occ = np.zeros((S, S, S, 4), np.uint8)
    occ[:, 10, :, 3] = 1                          # a floor at y = 10, material 1
    occ[:, 10, :, :3] = 120
    pal = np.zeros((256, 8), np.float32)
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "grid.bin"), "wb") as fo:
        fo.write(b"claude_grid 128 0 0 0\n")
        fo.write(occ.tobytes()); fo.write(pal.tobytes())
    H, W = 64, 64
    img = np.full((H, W, 3), 255, np.uint8)
    for yy in range(H):
        for xx in range(W):
            x, z = 20 + xx // 4, 30 + yy // 4      # 4x4 pixels per face
            fc = 3                                 # +y face
            img[yy, xx] = [x | ((fc & 1) << 7), 10 | (((fc >> 1) & 1) << 7), z | (((fc >> 2) & 1) << 7)]
    Image.fromarray(img).save(os.path.join(d, "faces.png"))
    val = 0.3 / 1.3
    b = np.full((H, W, 3), int(round(val * 255)), np.uint8)
    Image.fromarray(b).save(os.path.join(d, "bounce_a.png"))
    Image.fromarray(b).save(os.path.join(d, "bounce_b.png"))
    t, rep = vantage_faces(d)
    assert rep["grid_valid_share_of_stable"] == 1.0, rep
    assert t["face"].tolist() and set(t["face"].tolist()) == {3}, t["face"]
    assert np.allclose(t["bounce"], 0.3, atol=0.01), t["bounce"][:3]
    # corrupt: point the codes at air (y = 40): the grid check must reject
    img2 = img.copy(); img2[..., 1] = np.where(img[..., 1] == 255, 255, 40 | 128)
    Image.fromarray(img2).save(os.path.join(d, "faces.png"))
    t2, rep2 = vantage_faces(d)
    assert rep2["pixels_grid_valid"] == 0, rep2
    print("selftest ok:", rep, "| corrupted:", rep2["pixels_grid_valid"], "valid")


def main():
    if sys.argv[1] == "--selftest":
        selftest(); return
    run = sys.argv[1]
    tables, reports = {}, {}
    for name in sorted(os.listdir(run)):
        d = os.path.join(run, name)
        if not os.path.isdir(d):
            continue
        t, rep = vantage_faces(d)
        tables[name] = t
        reports[name] = rep
        print("%-18s %s" % (name, json.dumps(rep)), flush=True)
    np.savez_compressed(os.path.join(run, "faces.npz"),
                        **{"%s/%s" % (n, k): v for n, t in tables.items() for k, v in t.items()})
    json.dump(reports, open(os.path.join(run, "faces_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
