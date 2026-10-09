#!/usr/bin/env python3
"""claude_mldenoise — the learned-denoiser prototype (docs-draft/mldenoise.md).

Data: util/claude_mldenoise_capture.py (dumps of the denoiser's inputs at
several frame depths, plus a converged answer per view).

  ~/.venvs/rtm-torch/bin/python util/claude_mldenoise.py train --minutes 25 [--tag NAME]
  ~/.venvs/rtm-torch/bin/python util/claude_mldenoise.py eval [--tag NAME]     (held-out views)
  ~/.venvs/rtm-torch/bin/python util/claude_mldenoise.py test [--tag NAME]     (scoreboard scenes)
  ~/.venvs/rtm-torch/bin/python util/claude_mldenoise.py time [--tag NAME]     (ms per 960x540 frame)

SPACES. Input: radiance divided by the accumulated albedo (the light on the
surface, smooth where texture is not), times the view's pinned exposure,
through log(1+x): HDR made small and scale-free, so a torch room and the
noon sun look alike to the network. Output: the same log irradiance; the
albedo goes back in and the result is LINEAR radiance, which is what the
engine's present pass, auto exposure and every referee consume. Loss: L1
in DISPLAY space (exposure, ACES, gamma: claude_present's transform), the
space the scoreboard scores and the eye sees, so the network spends its
capacity where error is visible, and the sun's x40 cannot dominate.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DATA = os.environ.get("MLD_DATA") or os.path.expanduser("~/data/mldenoise")
IMG = os.path.join(REPO, "screenshots", "mldenoise")
MAIN_TRUTH = os.path.expanduser("~/code/luanti/screenshots/scoreboard/truth")

SKY = 1.0 + 6.0 * 65536.0     # claude_trace SKY_FACE_CODE
ALB_MIN = 0.005               # claude_trace ALBEDO_FLOOR
DEPTHS = [1, 4, 16, 64]
DEV = os.environ.get("MLD_DEV", "cuda")


# ---------------------------------------------------------------- data
def load_set(prefix):
    j = json.load(open(prefix + ".json"))
    h, w = j["h"], j["w"]

    def rd(n):
        return np.fromfile("%s.%s.f32" % (prefix, n), np.float32).reshape(h, w, 4)
    acc, dr, gb, mo = rd("accum"), rd("direct"), rd("gbuf"), rd("mom")
    # the moments become the noise of the mean HERE, in float32: the raw
    # second moment reaches 4e9 in a dusk interior and overflows the fp16
    # the samples are kept in (it made the first training run NaN)
    m2, vfac, m1 = (np.nan_to_num(mo[..., i].astype(np.float64)) for i in range(3))
    sd = np.sqrt(np.maximum(m2 - m1 * m1, 0) * np.clip(vfac, 0, 1))
    mo2 = np.stack([np.clip(sd, 0, 6e4), np.clip(vfac, 0, 1), np.zeros_like(sd)], -1).astype(np.float32)
    # 13 channels: radiance rgb, packed distance, direct rgb, albedo rgb, (sd of the mean, vfac, 0)
    raw = np.concatenate([acc, dr[..., :3], gb[..., :3], mo2], -1)
    raw = np.nan_to_num(raw, nan=0.0, posinf=6e4, neginf=0.0)
    raw[..., :3] = np.clip(raw[..., :3], 0, 6e4)
    raw[..., 4:7] = np.clip(raw[..., 4:7], 0, 6e4)
    return raw, gb[..., 3].copy(), j


def load_den(prefix):
    j = json.load(open(prefix + ".json"))
    return np.fromfile(prefix + ".den.f32", np.float32).reshape(j["h"], j["w"], 4)[..., :3]


def view_dirs():
    return sorted(d for d in glob.glob(os.path.join(DATA, "views", "*")) if os.path.exists(d + "/meta.json"))


def load_view(d):
    m = json.load(open(d + "/meta.json"))
    ref = m["ref"]
    tgt, _, _ = load_set(os.path.join(d, "B_%d" % ref))
    samples = []
    for run in ("A", "B"):
        for n in DEPTHS:
            p = os.path.join(d, "%s_%d" % (run, n))
            if os.path.exists(p + ".json"):
                raw, code, _ = load_set(p)
                samples.append((run, n, raw, code))
    return m, tgt[..., :3], samples


# ---------------------------------------------------------------- model
def torch_bits():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def features(raw, code, ex):
        """raw [B,13,H,W], code [B,1,H,W], ex [B] -> [B,20,H,W]"""
        acc, dist, dr = raw[:, 0:3], raw[:, 3:4], raw[:, 4:7]
        alb = raw[:, 7:10].clamp(min=ALB_MIN)
        mo = raw[:, 10:13]
        e = ex.view(-1, 1, 1, 1)
        f_irr = torch.log1p(acc.clamp(min=0) / alb * e)
        f_dir = torch.log1p(dr.clamp(min=0) / alb * e)
        c = code.abs()
        sky = (c - SKY).abs() < 0.5
        none = c < 0.5
        face = (~sky) & (~none)
        k = torch.floor((c - 1.0) / 65536.0).clamp(0, 5)
        ax = torch.floor(k / 2)
        sg = (k - 2 * ax) * 2 - 1
        nrm = torch.cat([(ax == i).float() * sg for i in range(3)], 1) * face.float()
        mixed = (code < -0.5).float()
        f_d = torch.log2(1 + dist.clamp(min=0) * 4096.0) / 12.0
        same_r = torch.zeros_like(code)
        same_r[..., :, :-1] = ((code[..., :, 1:] - code[..., :, :-1]).abs() < 0.5).float()
        same_d = torch.zeros_like(code)
        same_d[..., :-1, :] = ((code[..., 1:, :] - code[..., :-1, :]).abs() < 0.5).float()
        f_sd = torch.log1p(mo[:, 0:1].clamp(min=0) * e)
        f_n = -torch.log2(mo[:, 1:2].clamp(min=1e-6)) / 11.0
        return torch.cat([f_irr, f_dir, alb.sqrt(), nrm, sky.float(), none.float(), mixed, f_d,
                          same_r, same_d, f_sd, f_n], 1)

    def conv(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))

    class Net(nn.Module):
        """compact U-Net, 3 downsamplings; predicts a residual on log irradiance.
        unshuffle=True runs the whole U-Net at half resolution (pixel
        unshuffle in, pixel shuffle out): ~4x cheaper."""

        def __init__(self, cin=20, ch=(24, 32, 48, 64), unshuffle=False):
            super().__init__()
            self.us = unshuffle
            c0, c1, c2, c3 = ch
            ci = cin * 4 if unshuffle else cin
            self.e0 = nn.Sequential(conv(ci, c0), conv(c0, c0))
            self.e1 = nn.Sequential(conv(c0, c1), conv(c1, c1))
            self.e2 = nn.Sequential(conv(c1, c2), conv(c2, c2))
            self.b = nn.Sequential(conv(c2, c3), conv(c3, c3))
            self.d2 = nn.Sequential(conv(c3 + c2, c2), conv(c2, c2))
            self.d1 = nn.Sequential(conv(c2 + c1, c1), conv(c1, c1))
            self.d0 = nn.Sequential(conv(c1 + c0, c0), conv(c0, c0))
            self.out = nn.Conv2d(c0, 12 if unshuffle else 3, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, f):
            x = F.pixel_unshuffle(f, 2) if self.us else f
            a = self.e0(x)
            b = self.e1(F.avg_pool2d(a, 2))
            c = self.e2(F.avg_pool2d(b, 2))
            d = self.b(F.avg_pool2d(c, 2))
            up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")
            c = self.d2(torch.cat([up(d), c], 1))
            b = self.d1(torch.cat([up(c), b], 1))
            a = self.d0(torch.cat([up(b), a], 1))
            r = self.out(a)
            if self.us:
                r = F.pixel_shuffle(r, 2)
            return f[:, 0:3] + r        # log irradiance x exposure

    def radiance(net, raw, code, ex):
        """-> linear radiance [B,3,H,W] (pads to a multiple of 16 internally)"""
        H, W = raw.shape[-2:]
        ph, pw = (-H) % 16, (-W) % 16
        f = features(raw, code, ex)
        if ph or pw:
            f = F.pad(f, (0, pw, 0, ph), mode="replicate")
        out = net(f)[..., :H, :W]
        alb = raw[:, 7:10].clamp(min=ALB_MIN)
        return torch.expm1(out.float()).clamp(min=0) * alb / ex.view(-1, 1, 1, 1)

    def tonemap(x):
        """claude_present: ACES (Narkowicz) then gamma 1/2.2; x already exposed"""
        x = x.clamp(min=0)
        a = (x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0, 1)
        return (a + 1e-5) ** (1 / 2.2)

    return torch, nn, F, features, Net, radiance, tonemap


def tonemap_np(lin, ex):
    x = np.maximum(lin, 0) * ex
    a = np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    return a ** (1 / 2.2)


def to_u8(disp):
    return (np.clip(disp, 0, 1) * 255 + 0.5).astype(np.uint8)


def ckpt_path(tag):
    return os.path.join(DATA, "ckpt", tag + ".pt")


def make_net(Net, cfg):
    return Net(ch=tuple(cfg["ch"]), unshuffle=cfg["unshuffle"])


# ---------------------------------------------------------------- train
def cmd_train(a):
    dev = a.dev
    if dev != "cpu":
        import claude_gpu_lock
        claude_gpu_lock.hold("util/claude_mldenoise.py train " + a.tag)
    torch, nn, F, features, Net, radiance, tonemap = torch_bits()
    torch.manual_seed(0)
    np.random.seed(int(time.time()) % 100000)
    holdout = set(json.load(open(os.path.join(DATA, "holdout.json"))))
    raws, codes, tgts, exs, vidx = [], [], [], [], []
    tlist = []
    t0 = time.time()
    for d in view_dirs():
        name = os.path.basename(d)
        if name in holdout:
            continue
        m, tgt, samples = load_view(d)
        ti = len(tlist)
        tlist.append(torch.from_numpy(tgt).permute(2, 0, 1).half())
        if not np.isfinite(tgt).all() or np.abs(tgt).max() > 6e4:
            raise SystemExit("REFUSED: target of %s not finite or past fp16" % name)
        for run, n, raw, code in samples:
            if not np.isfinite(raw).all() or np.abs(raw).max() > 6.5e4:
                raise SystemExit("REFUSED: %s %s_%d not finite or past fp16 (max %g)" % (name, run, n, np.abs(raw).max()))
            raws.append(torch.from_numpy(raw).permute(2, 0, 1).half())
            codes.append(torch.from_numpy(code)[None])
            exs.append(m["exposure"])
            vidx.append(ti)
    print("loaded %d samples from %d views in %.0f s" % (len(raws), len(tlist), time.time() - t0), flush=True)
    RAW = torch.stack(raws).to(dev)          # fp16 on the GPU
    CODE = torch.stack(codes).to(dev)        # fp32 (codes are exact integers to 4e5)
    TGT = torch.stack(tlist).to(dev)
    EX = torch.tensor(exs, device=dev)
    VI = torch.tensor(vidx, device=dev)
    del raws, codes, tlist

    N, _, H, W = RAW.shape
    p = ckpt_path(a.tag)
    cfg = {"ch": [int(x) for x in a.ch.split(",")], "unshuffle": bool(a.unshuffle)}
    net = None
    state = None
    if os.path.exists(p):
        state = torch.load(p, map_location=dev)
        cfg = state["cfg"]
    net = make_net(Net, cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    step = 0
    if state:
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        print("resumed %s at step %d" % (a.tag, step), flush=True)
    total = a.total_steps
    sched = lambda s: a.lr * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    P, B = a.crop, a.batch
    scaler_ok = True
    t_end = time.time() + a.minutes * 60
    losses = []
    while time.time() < t_end and step < total:
        idx = torch.randint(0, N, (B,), device=dev)
        ys = torch.randint(0, (H - P) // 2, (B,)) * 2
        xs = torch.randint(0, (W - P) // 2, (B,)) * 2
        r = torch.stack([RAW[i, :, y:y + P, x:x + P] for i, y, x in zip(idx.tolist(), ys.tolist(), xs.tolist())]).float()
        c = torch.stack([CODE[i, :, y:y + P, x:x + P] for i, y, x in zip(idx.tolist(), ys.tolist(), xs.tolist())])
        t = torch.stack([TGT[VI[i], :, y:y + P, x:x + P] for i, y, x in zip(idx.tolist(), ys.tolist(), xs.tolist())]).float()
        ex = EX[idx]
        if np.random.rand() < 0.5:
            r, c, t = r.flip(-1), c.flip(-1), t.flip(-1)
        for g in opt.param_groups:
            g["lr"] = sched(step)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=bool(a.amp)):
            pred = radiance(net, r, c, ex)
        e = ex.view(-1, 1, 1, 1)
        loss = (tonemap(pred.float() * e) - tonemap(t * e)).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if not torch.isfinite(loss):
            raise SystemExit("REFUSED: loss went non-finite at step %d" % step)
        losses.append(loss.item())
        step += 1
        if step % a.log_every == 0:
            print("step %6d  loss %.5f  lr %.2e  %.0f s left" % (step, np.mean(losses[-a.log_every:]), sched(step),
                  t_end - time.time()), flush=True)
        if step % 2000 == 0:
            save(torch, net, opt, step, cfg, p)
    save(torch, net, opt, step, cfg, p)
    print("saved %s at step %d" % (p, step), flush=True)


def save(torch, net, opt, step, cfg, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg}, p + ".tmp")
    os.replace(p + ".tmp", p)


def load_net(tag, dev=None):
    dev = dev or DEV
    torch, nn, F, features, Net, radiance, tonemap = torch_bits()
    st = torch.load(ckpt_path(tag), map_location=dev)
    net = make_net(Net, st["cfg"]).to(dev)
    net.load_state_dict(st["net"])
    net.eval()
    return net, st


def infer(tag_net, raw, code, ex):
    torch, nn, F, features, Net, radiance, tonemap = torch_bits()
    net = tag_net
    with torch.no_grad():
        r = torch.from_numpy(raw).permute(2, 0, 1)[None].float().to(DEV)
        c = torch.from_numpy(code)[None, None].to(DEV)
        out = radiance(net, r, c, torch.tensor([ex], device=DEV).float())
    return out[0].permute(1, 2, 0).cpu().numpy()


def rmse_disp(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


# ---------------------------------------------------------------- eval (held-out views)
def cmd_eval(a):
    import claude_gpu_lock
    if DEV != "cpu":
        claude_gpu_lock.hold("util/claude_mldenoise.py eval " + a.tag)
    net, st = load_net(a.tag)
    holdout = json.load(open(os.path.join(DATA, "holdout.json")))
    rows = []
    for name in holdout:
        d = os.path.join(DATA, "views", name)
        if not os.path.exists(d + "/meta.json"):
            continue
        m, tgt, samples = load_view(d)
        ex = m["exposure"]
        T = tonemap_np(tgt, ex)
        for run, n, raw, code in samples:
            if run != "A":
                continue      # B's shallow depths share samples with its own target
            out = infer(net, raw, code, ex)
            rn = rmse_disp(tonemap_np(raw[..., :3], ex), T)
            rm = rmse_disp(tonemap_np(out, ex), T)
            rows.append({"view": name, "frames": n, "noisy": rn, "net": rm})
            print("%-18s %3d fr  noisy %.4f  net %.4f  (%.2fx)" % (name, n, rn, rm, rn / rm), flush=True)
    os.makedirs(os.path.join(DATA, "eval"), exist_ok=True)
    json.dump({"tag": a.tag, "step": st["step"], "rows": rows},
              open(os.path.join(DATA, "eval", a.tag + ".json"), "w"), indent=1)


# ---------------------------------------------------------------- test (scoreboard scenes)
def truth_lin(sc):
    """the linear truth rendered in the SAME session and pose as the test
    inputs (claude_mldenoise_capture.py test)"""
    d = os.path.join(DATA, "test", sc)
    m = json.load(open(d + "/meta.json"))
    p = os.path.join(d, m["truth"]["prefix"])
    j = json.load(open(p + ".json"))
    return np.fromfile(p + ".accum.f32", np.float32).reshape(j["h"], j["w"], 4)[..., :3], p


def cmd_test(a):
    import claude_gpu_lock
    from PIL import Image
    if DEV != "cpu":
        claude_gpu_lock.hold("util/claude_mldenoise.py test " + a.tag)
    net, st = load_net(a.tag)
    out_dir = os.path.join(IMG, "test", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for sc in a.scenes:
        d = os.path.join(DATA, "test", sc)
        meta = json.load(open(d + "/meta.json"))
        ex = meta["exposure"]
        tl, tsrc = truth_lin(sc)
        T = tonemap_np(tl, ex)
        Image.fromarray(to_u8(T)).save(os.path.join(out_dir, "%s_truth.png" % sc))
        for shot in meta["shots"]:
            s = shot["seed"]
            for n in shot["depths"]:
                pre = os.path.join(d, "s%d_%d" % (s, n))
                raw, code, j = load_set(pre)
                den = load_den(pre)
                t0 = time.time()
                out = infer(net, raw, code, ex)
                imgs = {"noisy": raw[..., :3], "filter": den, "net": out}
                row = {"scene": sc, "seed": s, "frames": n, "still_frames": j.get("still_frames"),
                       "same_scene": shot["same_scene"], "truth": tsrc}
                for k, v in imgs.items():
                    D = tonemap_np(v, ex)
                    row["rmse_" + k] = rmse_disp(D, T)
                    pp = os.path.join(out_dir, "%s_s%d_%d_%s.png" % (sc, s, n, k))
                    Image.fromarray(to_u8(D)).save(pp)
                    row["png_" + k] = pp
                    # linear error too (relative, per pixel luminance): what the engine's referees read
                    row["lin_" + k] = float(np.abs(v - tl).mean() / max(tl.mean(), 1e-9))
                row["png_truth"] = os.path.join(out_dir, "%s_truth.png" % sc)
                rows.append(row)
                print("%-10s seed %d %3d fr  RMSE noisy %.4f  filter %.4f  net %.4f" % (
                    sc, s, n, row["rmse_noisy"], row["rmse_filter"], row["rmse_net"]), flush=True)
    json.dump({"tag": a.tag, "step": st["step"], "rows": rows}, open(os.path.join(out_dir, "rows.json"), "w"),
              indent=1)
    print(os.path.join(out_dir, "rows.json"))


# ---------------------------------------------------------------- timing
def cmd_time(a):
    """ms per 960x540 frame. Two ways: (1) back-to-back, 50 frames between
    two GPU events, the GPU never idle (the steady state inside a render
    loop that keeps the GPU busy); (2) one frame at a time with a sync after
    each, which on this card came out bimodal (p10 2 ms, median 14 ms on
    2026-10-09: the clocks drop whenever the GPU idles between frames)."""
    import claude_gpu_lock
    claude_gpu_lock.hold("util/claude_mldenoise.py time " + a.tag)
    torch, nn, F, features, Net, radiance, tonemap = torch_bits()
    net, st = load_net(a.tag)
    res = {"cfg": st["cfg"]}
    raw = torch.rand(1, 13, 540, 960, device="cuda") * 2
    code = (torch.randint(0, 6, (1, 1, 540, 960), device="cuda") * 65536 + 1000).float()
    ex = torch.tensor([0.3], device="cuda")
    f = features(raw, code, ex)
    fp = F.pad(f, (0, 0, 0, 4), mode="replicate")

    def bench(fn, n=50, reps=7):
        for _ in range(30):
            fn()
        torch.cuda.synchronize()
        loops = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(n):
                fn()
            e.record()
            torch.cuda.synchronize()
            loops.append(s.elapsed_time(e) / n)
        single = []
        for _ in range(60):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            single.append(s.elapsed_time(e))
        return {"back_to_back_ms": float(np.median(loops)), "back_to_back_min": float(np.min(loops)),
                "single_median_ms": float(np.median(single)), "single_p10_ms": float(np.percentile(single, 10))}

    with torch.no_grad():
        for dtype, name in ((torch.float32, "fp32"), (torch.float16, "fp16")):
            for cl in (False, True):
                m = make_net(Net, st["cfg"]).cuda().eval()
                m.load_state_dict(net.state_dict())
                m = m.to(dtype)
                x = fp.to(dtype)
                if cl:
                    m = m.to(memory_format=torch.channels_last)
                    x = x.contiguous(memory_format=torch.channels_last)
                k = name + ("+channels_last" if cl else "")
                res[k] = bench(lambda: m(x))
                print("%-22s net only: back to back %.3f ms/frame (min %.3f); one at a time median %.3f, p10 %.3f"
                      % (k, res[k]["back_to_back_ms"], res[k]["back_to_back_min"], res[k]["single_median_ms"],
                         res[k]["single_p10_ms"]), flush=True)
        m = make_net(Net, st["cfg"]).cuda().eval().half()
        m.load_state_dict({k: v.half() for k, v in net.state_dict().items()})

        def e2e():
            f = features(raw, code, ex).half()
            f = F.pad(f, (0, 0, 0, 4), mode="replicate")
            o = m(f)[..., :540, :]
            return torch.expm1(o.float()) * raw[:, 7:10] / 0.3
        res["fp16_end_to_end"] = bench(e2e)
        print("fp16 end to end (features + net + albedo back in): back to back %.3f ms/frame" %
              res["fp16_end_to_end"]["back_to_back_ms"], flush=True)
    res["params"] = sum(p.numel() for p in net.parameters())
    res["device"] = torch.cuda.get_device_name(0)
    print("params", res["params"])
    os.makedirs(os.path.join(DATA, "time"), exist_ok=True)
    json.dump(res, open(os.path.join(DATA, "time", a.tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("scenes", nargs="*", default=["forest", "plains", "cabin", "torchroom"])
    ap.add_argument("--tag", default="unet")
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--total-steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ch", default="24,32,48,64")
    ap.add_argument("--unshuffle", type=int, default=0)
    ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--log-every", type=int, default=500)
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "test": cmd_test, "time": cmd_time}[a.what](a)
