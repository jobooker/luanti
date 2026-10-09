#!/usr/bin/env python3
"""claude_mldenoise_engine_score — the in-engine head-to-head.

  step 1 (rtm-torch venv, CPU):  pictures + RMSE for every arm
      ~/.venvs/rtm-torch/bin/python util/claude_mldenoise_engine_score.py pictures
  step 2 (judge venv):           FLIP, the tables (equal input, equal time
                                 with the engine's own measured frame times), side-by-sides
      ~/.venvs/judge/bin/python util/claude_mldenoise_engine_score.py tables

Arms, all on the SAME session's truth (rendered after the shots, same pose):
  noisy     the filter arm's accumulated radiance (the input)
  filter    today's filter, as the engine displayed it (den, learned 0)
  learned   the learned filter, as the engine displayed it (den, learned 1)
  unetg     the U-Net with the gradient term (PyTorch, CPU) on the filter
            arm's inputs, for reference
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DATA = os.environ.get("MLD_DATA") or os.path.expanduser("~/data/mldenoise")
OUT = os.path.join(REPO, "screenshots", "mldenoise", "test", "engine")
SCENES = ["forest", "plains", "cabin", "torchroom"]
ONE = {"forest": 27, "plains": 47, "cabin": 70, "torchroom": 52}
ARMS = ["noisy", "filter", "learned", "unetg"]


def pictures():
    import claude_mldenoise as M
    from PIL import Image
    torch, nn, F, features, Net, radiance, tonemap = M.torch_bits()
    torch.set_num_threads(16)
    st = torch.load(M.ckpt_path("unetg"), map_location="cpu")
    unet = M.make_net(Net, st["cfg"])
    unet.load_state_dict(st["net"])
    unet.eval()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sc in SCENES:
        d = os.path.join(DATA, "test_engine", sc)
        meta = json.load(open(d + "/meta.json"))
        ex = meta["exposure"]
        tp = os.path.join(d, meta["truth"]["prefix"])
        j = json.load(open(tp + ".json"))
        tl = np.fromfile(tp + ".accum.f32", np.float32).reshape(j["h"], j["w"], 4)[..., :3]
        T = M.tonemap_np(tl, ex)
        Image.fromarray(M.to_u8(T)).save(os.path.join(OUT, "%s_truth.png" % sc))
        seeds = sorted({s["seed"] for s in meta["shots"]})
        depths = meta["shots"][0]["depths"]
        for s in seeds:
            for n in depths:
                pf = os.path.join(d, "s%d_filter_%d" % (s, n))
                pl = os.path.join(d, "s%d_learned_%d" % (s, n))
                raw, code, _ = M.load_set(pf)
                with torch.no_grad():
                    r = torch.from_numpy(raw).permute(2, 0, 1)[None].float()
                    c = torch.from_numpy(code)[None, None]
                    u = radiance(unet, r, c, torch.tensor([ex]))[0].permute(1, 2, 0).numpy()
                imgs = {"noisy": raw[..., :3], "filter": M.load_den(pf), "learned": M.load_den(pl), "unetg": u}
                row = {"scene": sc, "seed": s, "frames": n, "truth": tp}
                for k, v in imgs.items():
                    D = M.tonemap_np(v, ex)
                    row["rmse_" + k] = M.rmse_disp(D, T)
                    pp = os.path.join(OUT, "%s_s%d_%d_%s.png" % (sc, s, n, k))
                    Image.fromarray(M.to_u8(D)).save(pp)
                    row["png_" + k] = pp
                row["png_truth"] = os.path.join(OUT, "%s_truth.png" % sc)
                rows.append(row)
                print("%-10s seed %d %3d fr  RMSE noisy %.4f filter %.4f learned %.4f unetg %.4f" % (
                    sc, s, n, row["rmse_noisy"], row["rmse_filter"], row["rmse_learned"], row["rmse_unetg"]),
                    flush=True)
        json.dump({"rows": rows}, open(os.path.join(OUT, "rows.json"), "w"), indent=1)


def tables():
    import claude_judge as J
    from PIL import Image, ImageDraw
    path = os.path.join(OUT, "rows.json")
    rec = json.load(open(path))
    rows = rec["rows"]
    for r in rows:
        if "flip_learned" in r:
            continue
        ref = J.load_png(r["png_truth"])
        for k in ARMS:
            r["flip_" + k] = J.flip_still(J.load_png(r["png_" + k]), ref)
    json.dump(rec, open(path, "w"), indent=1)
    price = json.load(open(os.path.join(DATA, "price", "price.json")))
    ms = {}
    for sc in SCENES:
        for arm in ("filter", "learned", "honest"):
            xs = [p for p in price if p["scene"] == sc and p["arm"] == arm]
            ms[(sc, arm)] = {k: float(np.median([p[k] for p in xs])) for k in ("busy_ms", "frame_ms", "dn_ms",
                                                                               "learned_ms", "trace_ms")}
            ms[(sc, arm)]["same_scene"] = all(p["same_scene"] for p in xs)
    unet_ms = json.load(open(os.path.join(DATA, "time", "careful_unetg.json")))["unetg"]["fp16_wall_ms"]

    def stat(sc, k, n):
        rr = [r for r in rows if r["scene"] == sc and r["frames"] == n]
        return np.mean([r["rmse_" + k] for r in rr]), np.mean([r["flip_" + k] for r in rr])
    L = ["## In-engine cost (ms per frame, pinned, 6 rounds, arm order rotating, medians; busy = the whole frame)", "",
         "| scene | honest busy | filter busy | learned busy | six passes (filter arm) | learned step (learned arm) | learned - filter (busy) |",
         "|---|---|---|---|---|---|---|"]
    for sc in SCENES:
        h, f, l = ms[(sc, "honest")], ms[(sc, "filter")], ms[(sc, "learned")]
        L.append("| %s | %.2f | %.2f | %.2f | %.3f | %.3f | %+.2f |" % (
            sc, h["busy_ms"], f["busy_ms"], l["busy_ms"], f["dn_ms"], l["learned_ms"], l["busy_ms"] - f["busy_ms"]))
    L += ["", "## Equal input (mean of 3 seeds): RMSE / FLIP", "",
          "| scene | frames | noisy | today's filter | learned (engine) | U-Net + grad (PyTorch) |", "|---|---|---|---|---|---|"]
    for sc in SCENES:
        for n in (1, ONE[sc]):
            L.append("| %s | %d | %s |" % (sc, n, " | ".join("%.4f / %.3f" % stat(sc, k, n) for k in ARMS)))
    L += ["", "## Equal time, one second, with the engine's measured busy ms: frames (scored at) RMSE / FLIP", "",
          "| scene | today's filter | learned (engine) | U-Net + grad (honest busy + %.1f ms PyTorch) |" % unet_ms,
          "|---|---|---|---|"]
    for sc in SCENES:
        depths = sorted({r["frames"] for r in rows if r["scene"] == sc})
        cells = []
        for k, fm in (("filter", ms[(sc, "filter")]["busy_ms"]), ("learned", ms[(sc, "learned")]["busy_ms"]),
                      ("unetg", ms[(sc, "honest")]["busy_ms"] + unet_ms)):
            n = max(1, math.floor(1000 / fm))
            dd = max([x for x in depths if x <= n] or [depths[0]])
            r, f = stat(sc, k, dd)
            cells.append("%d (%d) %.4f / %.3f" % (n, dd, r, f))
        L.append("| %s | %s |" % (sc, " | ".join(cells)))
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(OUT, "h2h_engine.md"), "w").write(txt + "\n")
    json.dump({"%s|%s" % k: v for k, v in ms.items()}, open(os.path.join(OUT, "price_summary.json"), "w"), indent=1)
    for sc in SCENES:
        tiles = []
        for n in (1, ONE[sc]):
            r = [x for x in rows if x["scene"] == sc and x["frames"] == n and x["seed"] == 101][0]
            ims = [Image.open(r["png_" + k]).convert("RGB") for k in ARMS] + [Image.open(r["png_truth"]).convert("RGB")]
            labels = ["%s, %d fr (RMSE %.4f, FLIP %.3f)" % (k, n, r["rmse_" + k], r["flip_" + k]) for k in ARMS] + ["truth"]
            w, h = ims[0].size
            row = Image.new("RGB", (w * 5, h + 22))
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
        p = os.path.join(OUT, "side-by-side-%s.png" % sc)
        sheet.save(p)
        print(p)


if __name__ == "__main__":
    {"pictures": pictures, "tables": tables}[sys.argv[1]]()
