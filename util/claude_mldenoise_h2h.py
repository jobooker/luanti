#!/usr/bin/env python3
"""claude_mldenoise_h2h — the head-to-head: today's filter, the U-Net (L1),
the U-Net with the gradient term, and the weight network on today's passes,
on the four held-out scenes, at equal input and at equal time.

Equal time: in the one-second budget each arm accumulates
floor(1000 / its frame ms) frames, scored at the deepest captured depth not
above that (conservative for every arm but today's filter, whose counts were
captured exactly). Frame times are the scoreboard's (run 20261008-131615):
  filter       anything frame ms
  U-Nets       honest frame ms + the network's ms
  weight net   anything frame ms + the network's ms + a surcharge for the
               filter passes' extra reads, ESTIMATED as today's passes' engine
               ms x (PyTorch weighted / PyTorch today - 1)
Also writes the side-by-sides: noisy | today's filter | U-Net | U-Net+grad |
weight net | truth.

  ~/.venvs/judge/bin/python util/claude_mldenoise_h2h.py
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.expanduser("~/data/mldenoise")
TEST = os.path.join(REPO, "screenshots", "mldenoise", "test")
RUN = os.path.expanduser("~/code/luanti/screenshots/scoreboard/runs/20261008-131615/shots.json")
SCENES = ["forest", "plains", "cabin", "torchroom"]
ONE = {"forest": 27, "plains": 47, "cabin": 70, "torchroom": 52}
ARMS = [("filter", "unet", "filter"), ("U-Net", "unet", "net"), ("U-Net+grad", "unetg", "net"),
        ("weight net", "wnet", "net")]


def main():
    rows = {tag: json.load(open(os.path.join(TEST, tag, "rows.json")))["rows"] for tag in ("unet", "unetg", "wnet")}
    tm = {t: json.load(open(os.path.join(DATA, "time", f))) for t, f in
          (("unet", "careful_unet_half.json"), ("unetg", "careful_unetg.json"), ("wnet", "wnet_atrous.json"))}
    net_ms = {"unet": tm["unet"]["unet"]["fp16_wall_ms"],
              "unetg": tm["unetg"]["unetg"]["fp16_wall_ms"],
              "wnet": tm["wnet"]["compiled_fp16_features_plus_net_ms"]}
    surcharge_ratio = tm["wnet"]["compiled_port_weighted_ms"] / tm["wnet"]["compiled_port_today_ms"] - 1
    fms, filt = {}, {}
    for s in json.load(open(RUN))["shots"]:
        fms[(s["scene"], s["contender"])] = s["frame_ms"]
        if s["budget"] == "one second" and s["contender"] in ("honest", "anything"):
            pm = json.load(open(s["png"].replace(".png", ".capture.json")))["stats"]["pass_ms"]
            filt[(s["scene"], s["contender"])] = sum(pm[3:9])

    def stat(tag, key, sc, n):
        rr = [r for r in rows[tag] if r["scene"] == sc and r["frames"] == n]
        return np.mean([r["rmse_" + key] for r in rr]), np.mean([r.get("flip_" + key, np.nan) for r in rr])

    L = []
    L.append("cost (ms per 960x540 frame): U-Net %.2f, U-Net+grad %.2f, weight net %.2f (+ filter surcharge est. x%.2f of "
             "today's passes)" % (net_ms["unet"], net_ms["unetg"], net_ms["wnet"], surcharge_ratio))
    L.append("today's filter in the engine (pass_ms 3..8, filter on): " + ", ".join(
        "%s %.2f" % (sc, filt[(sc, "anything")]) for sc in SCENES))
    L.append("")
    L.append("EQUAL INPUT (mean of 3 seeds): RMSE / FLIP")
    L.append("| scene | frames | noisy | today's filter | U-Net | U-Net+grad | weight net |")
    L.append("|---|---|---|---|---|---|---|")
    for sc in SCENES:
        for n in (1, ONE[sc]):
            cells = ["%.4f / %.3f" % stat("unet", "noisy", sc, n)]
            for name, tag, key in ARMS:
                cells.append("%.4f / %.3f" % stat(tag, key, sc, n))
            L.append("| %s | %d | %s |" % (sc, n, " | ".join(cells)))
    L.append("")
    L.append("EQUAL TIME (one second): frames (scored at) RMSE / FLIP")
    L.append("| scene | today's filter | U-Net | U-Net+grad | weight net |")
    L.append("|---|---|---|---|---|")
    for sc in SCENES:
        depths = sorted({r["frames"] for r in rows["unet"] if r["scene"] == sc})
        cells = []
        for name, tag, key in ARMS:
            if name == "filter":
                ms = fms[(sc, "anything")]
            elif tag in ("unet", "unetg"):
                ms = fms[(sc, "honest")] + net_ms[tag]
            else:
                ms = fms[(sc, "anything")] + net_ms["wnet"] + surcharge_ratio * filt[(sc, "anything")]
            n = max(1, math.floor(1000 / ms))
            d = max([x for x in depths if x <= n] or [depths[0]])
            r, f = stat(tag, key, sc, d)
            cells.append("%d (%d) %.4f / %.3f" % (n, d, r, f))
        L.append("| %s | %s |" % (sc, " | ".join(cells)))
    txt = "\n".join(L)
    print(txt)
    out = os.path.join(TEST, "h2h")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "h2h.md"), "w").write(txt + "\n")
    # side by sides
    for sc in SCENES:
        tiles = []
        for n in (1, ONE[sc]):
            pick = lambda tag, key: [r for r in rows[tag] if r["scene"] == sc and r["frames"] == n and r["seed"] == 101][0]
            panels = [("noisy", pick("unet", "noisy"), "noisy"), ("today's filter", pick("unet", "filter"), "filter"),
                      ("U-Net", pick("unet", "net"), "net"), ("U-Net+grad", pick("unetg", "net"), "net"),
                      ("weight net", pick("wnet", "net"), "net")]
            ims = [Image.open(r["png_" + k]).convert("RGB") for _, r, k in panels] + \
                  [Image.open(panels[0][1]["png_truth"]).convert("RGB")]
            labels = ["%s, %d fr (RMSE %.4f)" % (lb, n, r["rmse_" + k]) for lb, r, k in panels] + ["truth"]
            w, h = ims[0].size
            row = Image.new("RGB", (w * 6, h + 22))
            dr = ImageDraw.Draw(row)
            for i, (im, lb) in enumerate(zip(ims, labels)):
                row.paste(im, (i * w, 22))
                dr.text((i * w + 6, 4), lb, fill=(255, 255, 255))
            tiles.append(row)
        sheet = Image.new("RGB", (tiles[0].size[0], sum(t.size[1] for t in tiles)))
        y = 0
        for t in tiles:
            sheet.paste(t, (0, y))
            y += t.size[1]
        p = os.path.join(out, "side-by-side-%s.png" % sc)
        sheet.save(p)
        print(p)


if __name__ == "__main__":
    main()
