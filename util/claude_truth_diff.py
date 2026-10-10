#!/usr/bin/env python3
"""claude_truth_diff A B [OUT.jpg] -- where two truth-check renders differ.

Two checks of one stored truth at the SAME seed should be the same picture
(the renderer is deterministic per seed). When they are not, the scene
differed between the two sessions; this shows where (2026-10-09: the room
check differed 2x between sessions while the outdoor one did not).

Per checked pose: A | B | |A - B| x 8, the difference in brightness (grey
levels, not colour), and the 8x8 blocks that carry most of it.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_truth_store as TS


def main():
    a, b = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/truth_diff.jpg"
    ra, rb = TS._rows(a), TS._rows(b)
    n = min(len(ra), len(rb)) // (TS.CHECK_LEAD + 1)
    tiles = []
    for c in range(n):
        j = c * (TS.CHECK_LEAD + 1) + TS.CHECK_LEAD
        if not TS._near(TS._pose(ra[j]), TS._pose(rb[j])):
            print("pose %d differs: %r vs %r" % (c, TS._pose(ra[j]), TS._pose(rb[j])))
            continue
        fa, fb = TS._frame(a, ra[j]), TS._frame(b, rb[j])
        d = np.abs(fa - fb).mean(axis=2)
        blk = TS._blocks(np.abs(fa - fb)).mean(axis=2)
        h, w = blk.shape
        top = np.argsort(blk.ravel())[::-1][:5]
        print("pose %d (frame %d): rmse %.5f, mean A %.4f B %.4f (x%.4f); worst blocks (row, col of %dx%d): %s"
              % (c, j, float(np.sqrt(((fa - fb) ** 2).mean())), fa.mean(), fb.mean(), fb.mean() / max(fa.mean(), 1e-9),
                 h, w, ", ".join("(%d,%d) %.4f" % (i // w, i % w, blk.ravel()[i]) for i in top)))
        enc = lambda x: (np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        diff = np.clip(d * 8.0, 0, 1)
        row = np.concatenate([enc(fa), enc(fb), np.repeat((diff ** (1 / 2.2) * 255).astype(np.uint8)[:, :, None], 3, 2)], 1)
        tiles.append(row)
    if tiles:
        Image.fromarray(np.concatenate(tiles, 0)).save(out, quality=88)
        print(out)


if __name__ == "__main__":
    main()
