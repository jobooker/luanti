#!/usr/bin/env python3
"""claude_mldenoise_atrous — today's a-trous filter (client/shaders/claude_denoise)
ported to PyTorch pass for pass, and a small network that supplies its weights
(docs-draft/mldenoise.md, "idea 2").

THE PORT is exact by construction, not by fit: the same six passes, the same
taps and kernel (B3 spline 1/16 1/4 3/8 1/4 1/16, steps 1 2 4 8 16), the same
same-face test (claude_trace's face code), the same noise estimate (moments
when the history holds >= 4 samples, the 7x7 same-face spread otherwise), the
same 3x3 variance smoothing, SIGMA_L 4, the same young-pixel fade (kappa, over
claude_denoise_young = 64 samples, the game's default), albedo divided out and
put back. `check` compares it with what the engine displayed (`den` dumps).

THE NETWORK may only re-weigh real samples; it never paints a colour:
  sig   per pixel and pass, a factor on the luminance edge-stopping width
  feat  per pixel, a few affinity features: a tap's weight is multiplied by
        exp(-|f_p - f_q|^2) (the learned "same surface, same light" test)
  mix   per pixel, softmax weights over the six stages (unfiltered, after
        pass 1..5): how far to blur here
So the output is a convex blend of same-face averages of the pixel's own
neighbourhood: it cannot make light that no sample carried. At zero weights it
IS today's filter (mix starts on stage 5 with weight 0.99997).

  ~/.venvs/rtm-torch/bin/python util/claude_mldenoise_atrous.py check            (CPU: port vs engine)
  ~/.venvs/rtm-torch/bin/python util/claude_mldenoise_atrous.py train --tag wnet --minutes 25
  ~/.venvs/rtm-torch/bin/python util/claude_mldenoise_atrous.py test --tag wnet
  ~/.venvs/rtm-torch/bin/python util/claude_mldenoise_atrous.py time --tag wnet
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_mldenoise as M  # noqa: E402

ALB_MIN = 0.005
SIGMA_L = 4.0          # claude_denoise SIGMA_L (literature value, SVGF)
VFAC_YOUNG = 0.25      # claude_denoise VFAC_YOUNG
YOUNG = 64.0           # claude_denoise_young, the game's default (game.cpp m_denoise_young)
H5 = [1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16]
NFEAT = 4
MIX_BIAS = 12.0        # stage 5 at 0.99997 when the network says nothing (Adam does not mind the small gradients)


def bits():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def lum(x):
        return x[:, 0:1] * 0.2126 + x[:, 1:2] * 0.7152 + x[:, 2:3] * 0.0722

    def pad(x, p, v=0.0):
        return F.pad(x, (p, p, p, p), value=v)

    def atrous(raw, code, mod=None, young=YOUNG):
        """raw [B,13,H,W] in claude_mldenoise's layout (10 = sd of the mean,
        11 = vfac, 12 = the pixel's sample count), code [B,1,H,W] -> linear
        radiance [B,3,H,W]. mod: None = today's filter."""
        H, W = raw.shape[-2:]
        acc = raw[:, 0:3]
        alb = raw[:, 7:10].clamp(min=ALB_MIN)
        sd, vfac, nsamp = raw[:, 10:11], raw[:, 11:12], raw[:, 12:13]
        acode = code.abs()
        valid = acode >= 0.5
        eps = 1e-20 if torch.is_grad_enabled() else 0.0

        def tap_ok(cq):
            return ((cq > 0.5) & ((cq - acode).abs() < 0.5)).to(raw.dtype)

        def sl(t, p, dy, dx):
            return t[..., p + dy:p + dy + H, p + dx:p + dx + W]

        irr = acc / alb
        # ---- pass 0: the noise estimate
        li = lum(irr)
        cpd, lpd = pad(code, 3), pad(li, 3)
        s1 = torch.zeros_like(li)
        s2 = torch.zeros_like(li)
        n = torch.zeros_like(li)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ok = torch.ones_like(li) if (dy == 0 and dx == 0) else tap_ok(sl(cpd, 3, dy, dx))
                lq = sl(lpd, 3, dy, dx)
                s1 = s1 + lq * ok
                s2 = s2 + lq * lq * ok
                n = n + ok
        m1 = s1 / n
        var_s = (s2 / n - m1 * m1).clamp(min=0)
        var = torch.where(vfac <= VFAC_YOUNG, sd * sd, var_s)
        rgb = torch.where(valid, irr, acc)
        v = torch.where(valid, var, torch.zeros_like(var))
        stages = [rgb]
        for it in range(1, 6):
            step = 1 << (it - 1)
            # the noise estimate, smoothed 3x3 on the same face
            vpd, cpd = pad(v, 1), pad(code, 1)
            vs = torch.zeros_like(v)
            vw = torch.zeros_like(v)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ok = torch.ones_like(v) if (dy == 0 and dx == 0) else tap_ok(sl(cpd, 1, dy, dx))
                    k = (0.5 if dx == 0 else 0.25) * (0.5 if dy == 0 else 0.25)
                    vs = vs + sl(vpd, 1, dy, dx) * k * ok
                    vw = vw + k * ok
            sig = SIGMA_L * torch.sqrt((vs / vw).clamp(min=0) + eps) + 1e-8
            if mod is not None:
                sig = sig * mod["sig"][:, it - 1:it]
            lp = lum(rgb)
            neff = nsamp if it == 5 else 1.0 / vfac.clamp(min=1e-4)
            kappa = ((neff - 1.0) / young).clamp(0, 1) if young > 0.5 else torch.ones_like(lp)
            P = 2 * step
            rpd, vpd, cpd, lpd = pad(rgb, P), pad(v, P), pad(code, P), pad(lp, P)
            fpd = pad(mod["feat"], P) if mod is not None else None
            s = torch.zeros_like(rgb)
            ws = torch.zeros_like(lp)
            vsum = torch.zeros_like(lp)
            for j, dy in enumerate(range(-2, 3)):
                for i, dx in enumerate(range(-2, 3)):
                    oy, ox = dy * step, dx * step
                    ok = torch.ones_like(lp) if (dy == 0 and dx == 0) else tap_ok(sl(cpd, P, oy, ox))
                    w = H5[i] * H5[j] * torch.exp(-kappa * (sl(lpd, P, oy, ox) - lp).abs() / sig) * ok
                    if fpd is not None and not (dy == 0 and dx == 0):
                        w = w * torch.exp(-((sl(fpd, P, oy, ox) - mod["feat"]) ** 2).sum(1, keepdim=True))
                    s = s + sl(rpd, P, oy, ox) * w
                    ws = ws + w
                    vsum = vsum + w * w * sl(vpd, P, oy, ox)
            rgb = torch.where(valid, s / ws, rgb)
            v = torch.where(valid, vsum / (ws * ws), v)
            stages.append(rgb)
        if mod is not None:
            mix = mod["mix"]
            out = sum(mix[:, k:k + 1] * stages[k] for k in range(6))
        else:
            out = rgb
        return torch.where(valid, out * alb, acc)

    def conv(i, o, d=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=d, dilation=d), nn.LeakyReLU(0.1, inplace=True))

    class Head(nn.Module):
        """a per-pixel layer as a matrix product on channels-last data: MIOpen's
        1x1 convolution took 1.0 ms here where the product takes 0.3 (bench)"""

        def __init__(self, i, o):
            super().__init__()
            self.lin = nn.Linear(i, o)

        def forward(self, x):
            return self.lin(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    class WeightNet(nn.Module):
        """ALL AT 1/4 RESOLUTION: the per-pixel edges stay the filter's own
        job (the exact same-face test and the brightness test); the network
        only says, per 4x4 block, how wide the brightness test should be per
        pass, a few affinity features, and how far to blur. Its maps are
        upsampled bilinearly (free in a shader's texture fetch). A full-
        resolution head cost 2.5-3 ms in PyTorch here (profile 2026-10-09:
        permute copies and per-pixel products), more than the budget.
        Plain 3x3 convs only: a dilated one took 12-16 ms in MIOpen."""

        def __init__(self, cin=20, c=24):
            super().__init__()
            self.a = nn.Sequential(conv(cin, c), conv(c, c))
            self.b = conv(c, c)
            self.c = conv(c, c)
            self.h = nn.Conv2d(c, 5 + NFEAT + 6, 1)
            nn.init.zeros_(self.h.weight)
            nn.init.zeros_(self.h.bias)
            self.register_buffer("mix_bias", torch.tensor([0, 0, 0, 0, 0, MIX_BIAS], dtype=torch.float32).view(1, 6, 1, 1))

        def low(self, pooled):
            """pooled features [B,C,H/4,W/4] -> maps at 1/4 resolution, activated"""
            q = self.a(pooled)
            e = self.b(F.avg_pool2d(q, 2))
            q = self.c(q + F.interpolate(e, size=q.shape[-2:], mode="nearest"))
            o = self.h(q).float()
            return torch.cat([torch.exp(o[:, 0:5].clamp(-4, 4)), o[:, 5:5 + NFEAT],
                              torch.softmax(o[:, 5 + NFEAT:] + self.mix_bias, 1)], 1)

        def forward(self, f):
            """f: full-resolution features [B,C,H,W] -> the mod dict the port reads"""
            a = F.interpolate(self.low(F.avg_pool2d(f, 4)), size=f.shape[-2:], mode="bilinear", align_corners=False)
            return {"sig": a[:, 0:5], "feat": a[:, 5:5 + NFEAT], "mix": a[:, 5 + NFEAT:]}

    return torch, nn, F, atrous, WeightNet


def grad_l1(torch, a, b):
    """L1 of the difference of horizontal and vertical neighbour differences:
    punishes grain the truth does not have (and edges it has but a blur lost)"""
    dxa, dxb = a[..., :, 1:] - a[..., :, :-1], b[..., :, 1:] - b[..., :, :-1]
    dya, dyb = a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :]
    return (dxa - dxb).abs().mean() + (dya - dyb).abs().mean()


# ---------------------------------------------------------------- check
def cmd_check(a):
    """the port against what the engine displayed, on the anything dumps"""
    torch, nn, F, atrous, WeightNet = bits()
    rows = []
    for sc in a.scenes:
        d = os.path.join(M.DATA, "test", sc)
        meta = json.load(open(d + "/meta.json"))
        ex = meta["exposure"]
        s = meta["shots"][0]["seed"]
        for n in meta["shots"][0]["depths"]:
            if n not in (1, 4, 16, 64) and n != max(meta["shots"][0]["depths"]):
                continue
            pre = os.path.join(d, "s%d_%d" % (s, n))
            raw, code, _ = M.load_set(pre, clip=False)
            den = M.load_den(pre)
            with torch.no_grad():
                r = torch.from_numpy(raw).permute(2, 0, 1)[None].double()
                c = torch.from_numpy(code)[None, None].double()
                out = atrous(r, c)[0].permute(1, 2, 0).numpy()
            rel = np.abs(out - den) / np.maximum(np.abs(den), 1e-3)
            dd = M.rmse_disp(M.tonemap_np(out, ex), M.tonemap_np(den, ex))
            rows.append({"scene": sc, "frames": n, "max_rel": float(rel.max()), "p999_rel": float(np.percentile(rel, 99.9)),
                         "share_rel_gt_1e-3": float((rel > 1e-3).mean()), "display_rmse_port_vs_engine": dd,
                         "display_rmse_engine_vs_noisy": M.rmse_disp(M.tonemap_np(raw[..., :3], ex), M.tonemap_np(den, ex))})
            r_ = rows[-1]
            print("%-10s %3d fr  port vs engine: display RMSE %.6f (engine vs noisy %.4f)  rel err p99.9 %.2e  max %.2e  "
                  "share >1e-3 %.5f" % (sc, n, dd, r_["display_rmse_engine_vs_noisy"], r_["p999_rel"], r_["max_rel"],
                                        r_["share_rel_gt_1e-3"]), flush=True)
    os.makedirs(os.path.join(M.DATA, "eval"), exist_ok=True)
    json.dump(rows, open(os.path.join(M.DATA, "eval", "atrous_port_check.json"), "w"), indent=1)


# ---------------------------------------------------------------- train
def cmd_train(a):
    import claude_gpu_lock
    claude_gpu_lock.hold("util/claude_mldenoise_atrous.py train " + a.tag)
    torch, nn, F, atrous, WeightNet = bits()
    _, _, _, features, _, _, tonemap = M.torch_bits()
    dev = "cuda"
    torch.manual_seed(0)
    np.random.seed(int(time.time()) % 100000)
    holdout = set(json.load(open(os.path.join(M.DATA, "holdout.json"))))
    raws, codes, tl, exs, vidx = [], [], [], [], []
    t0 = time.time()
    for d in M.view_dirs():
        name = os.path.basename(d)
        if name in holdout:
            continue
        m, tgt, samples = M.load_view(d)
        ti = len(tl)
        tl.append(torch.from_numpy(tgt).permute(2, 0, 1).half())
        for run, nn_, raw, code in samples:
            raws.append(torch.from_numpy(raw).permute(2, 0, 1).half())
            codes.append(torch.from_numpy(code)[None])
            exs.append(m["exposure"])
            vidx.append(ti)
    print("loaded %d samples from %d views in %.0f s" % (len(raws), len(tl), time.time() - t0), flush=True)
    RAW = torch.stack(raws).to(dev)
    CODE = torch.stack(codes).to(dev)
    TGT = torch.stack(tl).to(dev)
    EX = torch.tensor(exs, device=dev)
    VI = torch.tensor(vidx, device=dev)
    del raws, codes, tl
    N, _, H, W = RAW.shape
    p = M.ckpt_path(a.tag)
    net = WeightNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    step = 0
    if os.path.exists(p):
        st = torch.load(p, map_location=dev)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print("resumed at", step, flush=True)
    total = a.total_steps
    sched = lambda s: a.lr * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    P, B = a.crop, a.batch
    t_end = time.time() + a.minutes * 60
    losses = []
    while time.time() < t_end and step < total:
        idx = torch.randint(0, N, (B,))
        ys = (torch.randint(0, (H - P) // 4, (B,)) * 4).tolist()
        xs = (torch.randint(0, (W - P) // 4, (B,)) * 4).tolist()
        il = idx.tolist()
        r = torch.stack([RAW[i, :, y:y + P, x:x + P] for i, y, x in zip(il, ys, xs)]).float()
        c = torch.stack([CODE[i, :, y:y + P, x:x + P] for i, y, x in zip(il, ys, xs)])
        t = torch.stack([TGT[VI[i], :, y:y + P, x:x + P] for i, y, x in zip(il, ys, xs)]).float()
        ex = EX[idx.to(dev)]
        if np.random.rand() < 0.5:
            r, c, t = r.flip(-1), c.flip(-1), t.flip(-1)
        for g in opt.param_groups:
            g["lr"] = sched(step)
        mod = net(features(r, c, ex))
        pred = atrous(r, c, mod)
        e = ex.view(-1, 1, 1, 1)
        dp, dt = tonemap(pred * e), tonemap(t * e)
        loss = (dp - dt).abs().mean() + a.grad * grad_l1(torch, dp, dt)
        if not torch.isfinite(loss):
            raise SystemExit("REFUSED: loss went non-finite at step %d" % step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        step += 1
        if step % a.log_every == 0:
            print("step %6d  loss %.5f  lr %.2e  %.0f s left" % (step, np.mean(losses[-a.log_every:]), sched(step),
                  t_end - time.time()), flush=True)
        if step % 1000 == 0:
            save(torch, net, opt, step, p)
    save(torch, net, opt, step, p)
    print("saved %s at step %d" % (p, step), flush=True)


def save(torch, net, opt, step, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step}, p + ".tmp")
    os.replace(p + ".tmp", p)


# ---------------------------------------------------------------- test
def cmd_test(a):
    import claude_gpu_lock
    from PIL import Image
    claude_gpu_lock.hold("util/claude_mldenoise_atrous.py test " + a.tag)
    torch, nn, F, atrous, WeightNet = bits()
    _, _, _, features, _, _, _ = M.torch_bits()
    net = WeightNet().cuda().eval()
    st = torch.load(M.ckpt_path(a.tag), map_location="cuda")
    net.load_state_dict(st["net"])
    out_dir = os.path.join(M.IMG, "test", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for sc in a.scenes:
        d = os.path.join(M.DATA, "test", sc)
        meta = json.load(open(d + "/meta.json"))
        ex = meta["exposure"]
        tl, tsrc = M.truth_lin(sc)
        T = M.tonemap_np(tl, ex)
        Image.fromarray(M.to_u8(T)).save(os.path.join(out_dir, "%s_truth.png" % sc))
        for shot in meta["shots"]:
            s = shot["seed"]
            for n in shot["depths"]:
                pre = os.path.join(d, "s%d_%d" % (s, n))
                raw, code, j = M.load_set(pre)
                den = M.load_den(pre)
                with torch.no_grad():
                    r = torch.from_numpy(raw).permute(2, 0, 1)[None].float().cuda()
                    c = torch.from_numpy(code)[None, None].cuda()
                    e = torch.tensor([ex], device="cuda").float()
                    mod = net(features(r, c, e))
                    out = atrous(r, c, mod)[0].permute(1, 2, 0).cpu().numpy()
                    if a.save_mod and s == shot["seed"] and n in (1, max(shot["depths"])):
                        np.savez_compressed(os.path.join(out_dir, "%s_s%d_%d_mod.npz" % (sc, s, n)),
                                            **{k: v[0].cpu().numpy().astype(np.float16) for k, v in mod.items()})
                imgs = {"noisy": raw[..., :3], "filter": den, "net": out}
                row = {"scene": sc, "seed": s, "frames": n, "same_scene": shot["same_scene"], "truth": tsrc}
                for k, v in imgs.items():
                    D = M.tonemap_np(v, ex)
                    row["rmse_" + k] = M.rmse_disp(D, T)
                    pp = os.path.join(out_dir, "%s_s%d_%d_%s.png" % (sc, s, n, k))
                    Image.fromarray(M.to_u8(D)).save(pp)
                    row["png_" + k] = pp
                row["png_truth"] = os.path.join(out_dir, "%s_truth.png" % sc)
                rows.append(row)
                print("%-10s seed %d %3d fr  RMSE noisy %.4f  filter %.4f  net %.4f" % (
                    sc, s, n, row["rmse_noisy"], row["rmse_filter"], row["rmse_net"]), flush=True)
    json.dump({"tag": a.tag, "step": st["step"], "rows": rows}, open(os.path.join(out_dir, "rows.json"), "w"), indent=1)
    print(os.path.join(out_dir, "rows.json"))


# ---------------------------------------------------------------- time
def cmd_time(a):
    """sustained wall clock, 200 frames back to back, as claude_mldenoise_time.py"""
    import claude_gpu_lock
    claude_gpu_lock.hold("util/claude_mldenoise_atrous.py time " + a.tag)
    torch, nn, F, atrous, WeightNet = bits()
    _, _, _, features, _, _, _ = M.torch_bits()
    net = WeightNet().cuda().eval()
    if os.path.exists(M.ckpt_path(a.tag)):
        net.load_state_dict(torch.load(M.ckpt_path(a.tag), map_location="cuda")["net"])
    H, W = 540, 960
    raw = torch.rand(1, 13, H, W, device="cuda") * 2
    raw[:, 11] = 0.05
    raw[:, 12] = 20
    code = (torch.randint(0, 6, (1, 1, H, W), device="cuda") * 65536 + 1000).float()
    ex = torch.tensor([0.3], device="cuda")

    def wall(fn, n=200):
        with torch.no_grad():
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                for _ in range(n):
                    fn()
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) / n * 1000)
            return float(np.median(ts))
    res = {}
    f = features(raw, code, ex)
    mod = net(f)
    res["eager_features_ms"] = wall(lambda: features(raw, code, ex))
    res["eager_net_ms"] = wall(lambda: net(f))
    res["eager_port_today_ms"] = wall(lambda: atrous(raw, code), 20)
    res["eager_port_weighted_ms"] = wall(lambda: atrous(raw, code, mod), 20)
    # fused: torch.compile (inductor) writes one kernel where eager runs dozens,
    # which is what a shader in the engine would do too
    try:
        cf = torch.compile(features)
        cn = torch.compile(net)
        ca = torch.compile(atrous)
        res["compiled_features_ms"] = wall(lambda: cf(raw, code, ex))
        res["compiled_net_ms"] = wall(lambda: cn(f))
        res["compiled_features_plus_net_ms"] = wall(lambda: cn(cf(raw, code, ex)))
        neth = WeightNet().cuda().eval().half()
        neth.load_state_dict({k: v.half() for k, v in net.state_dict().items()})

        def fe16(raw, code, ex):
            # features, pooled, the network, its maps upsampled to full resolution
            f = features(raw, code, ex).half()
            return F.interpolate(neth.low(F.avg_pool2d(f, 4)), size=f.shape[-2:], mode="bilinear",
                                 align_corners=False)

        def fe16lo(raw, code, ex):
            # the same without the final upsample (a shader's bilinear fetch does it)
            return neth.low(F.avg_pool2d(features(raw, code, ex).half(), 4))
        c16lo = torch.compile(fe16lo)
        res["compiled_fp16_lowres_maps_ms"] = wall(lambda: c16lo(raw, code, ex))
        c16 = torch.compile(fe16)
        res["compiled_fp16_features_plus_net_ms"] = wall(lambda: c16(raw, code, ex))
        res["compiled_port_today_ms"] = wall(lambda: ca(raw, code), 50)
        res["compiled_port_weighted_ms"] = wall(lambda: ca(raw, code, mod), 50)
        res["compiled_whole_ms"] = wall(lambda: ca(raw, code, cn(cf(raw, code, ex))), 50)
    except Exception as e:
        res["compile_error"] = repr(e)[:300]
    res["params"] = sum(p.numel() for p in net.parameters())
    for k, v in res.items():
        print("%-24s %s" % (k, ("%.3f" % v) if isinstance(v, float) else v), flush=True)
    os.makedirs(os.path.join(M.DATA, "time"), exist_ok=True)
    json.dump(res, open(os.path.join(M.DATA, "time", a.tag + "_atrous.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("scenes", nargs="*", default=["forest", "plains", "cabin", "torchroom"])
    ap.add_argument("--tag", default="wnet")
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--total-steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    # TUNED: weight of the gradient term (1.0) | learn by: FLIP on the held-out views, 0 / 0.5 / 1 / 2
    ap.add_argument("--grad", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--save-mod", type=int, default=1)
    a = ap.parse_args()
    {"check": cmd_check, "train": cmd_train, "test": cmd_test, "time": cmd_time}[a.what](a)
