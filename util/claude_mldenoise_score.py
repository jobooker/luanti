#!/usr/bin/env python3
"""claude_mldenoise_score — FLIP (the scoreboard's judge) for every picture
claude_mldenoise.py test wrote, plus the side-by-side per scene.

  ~/.venvs/judge/bin/python util/claude_mldenoise_score.py screenshots/mldenoise/test/<tag>/rows.json
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import claude_judge as J  # noqa: E402

ONE_SECOND = {"forest": 27, "plains": 47, "cabin": 70, "torchroom": 52}


def main(path):
    rec = json.load(open(path))
    out_dir = os.path.dirname(path)
    for r in rec["rows"]:
        ref = J.load_png(r["png_truth"])
        for k in ("noisy", "filter", "net"):
            if "flip_" + k not in r:
                r["flip_" + k] = J.flip_still(J.load_png(r["png_" + k]), ref)
        print("%-10s seed %d %3d fr  FLIP noisy %.4f filter %.4f net %.4f | RMSE %.4f %.4f %.4f" % (
            r["scene"], r["seed"], r["frames"], r["flip_noisy"], r["flip_filter"], r["flip_net"],
            r["rmse_noisy"], r["rmse_filter"], r["rmse_net"]), flush=True)
    json.dump(rec, open(path, "w"), indent=1)
    # side by side: noisy | today's filter | network | truth, at 1 frame and one second, first seed
    for sc in sorted({r["scene"] for r in rec["rows"]}):
        rows = [r for r in rec["rows"] if r["scene"] == sc]
        seed = min(r["seed"] for r in rows)
        tiles = []
        for n in (1, ONE_SECOND.get(sc, 64)):
            rr = [r for r in rows if r["seed"] == seed and r["frames"] == n]
            if not rr:
                continue
            r = rr[0]
            ims = [Image.open(r["png_" + k]).convert("RGB") for k in ("noisy", "filter", "net")] + \
                  [Image.open(r["png_truth"]).convert("RGB")]
            labels = ["noisy, %d frame%s (RMSE %.4f)" % (n, "" if n == 1 else "s", r["rmse_noisy"]),
                      "today's filter (RMSE %.4f)" % r["rmse_filter"],
                      "network (RMSE %.4f)" % r["rmse_net"], "truth"]
            w, h = ims[0].size
            row = Image.new("RGB", (w * 4, h + 22), (0, 0, 0))
            dr = ImageDraw.Draw(row)
            for i, (im, lb) in enumerate(zip(ims, labels)):
                row.paste(im, (i * w, 22))
                dr.text((i * w + 6, 4), lb, fill=(255, 255, 255))
            tiles.append(row)
        if tiles:
            W = tiles[0].size[0]
            Ht = sum(t.size[1] for t in tiles)
            sheet = Image.new("RGB", (W, Ht))
            y = 0
            for t in tiles:
                sheet.paste(t, (0, y))
                y += t.size[1]
            p = os.path.join(out_dir, "side-by-side-%s.png" % sc)
            sheet.save(p)
            print(p)


if __name__ == "__main__":
    main(sys.argv[1])
