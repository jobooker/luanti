"""the GLSL network (learned.comp.glsl K_CONV/K_B/K_C) re-done in numpy from
the weights file, index formula for index formula: if it matches the engine,
the port's formulas differ from PyTorch; if it matches PyTorch, the GPU code
does something the formulas do not say."""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_mldenoise as M  # noqa: E402
import claude_mldenoise_atrous as A  # noqa: E402

torch, nn, F, atrous, WeightNet = A.bits()
pre = sys.argv[1]
blob = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_mldenoise_weights.bin"), "rb").read()
cin, c, nf, n = struct.unpack("<4i", blob[8:24])
w = np.frombuffer(blob[24:], "<f4").astype(np.float64)
o = 0
off = {}
for k, size in (("a0w", c * cin * 9), ("a0b", c), ("a1w", c * c * 9), ("a1b", c), ("bw", c * c * 9), ("bb", c),
                ("cw", c * c * 9), ("cb", c), ("hw", 15 * c), ("hb", 15), ("mb", 6)):
    off[k] = w[o:o + size]
    o += size
leaky = lambda x: np.where(x > 0, x, 0.1 * x)


def conv(x, W, b, ci, co):
    Hh, Ww = x.shape[:2]
    Wt = W.reshape(co, ci, 3, 3)
    xp = np.zeros((Hh + 2, Ww + 2, ci))
    xp[1:-1, 1:-1] = x
    out = np.tile(b, (Hh, Ww, 1))
    for ky in range(3):
        for kx in range(3):
            out += xp[ky:ky + Hh, kx:kx + Ww] @ Wt[:, :, ky, kx].T
    return out


eF = np.fromfile(pre + ".lF.f32", np.float32).reshape(135, 240, 20).astype(np.float64)
eM = np.fromfile(pre + ".lM.f32", np.float32).reshape(135, 240, 16)
h = leaky(conv(eF, off["a0w"], off["a0b"], cin, c))
q = leaky(conv(h, off["a1w"], off["a1b"], c, c))
He, We = 67, 120
pq = q[:2 * He, :2 * We].reshape(He, 2, We, 2, c).mean((1, 3))
e = leaky(conv(pq, off["bw"], off["bb"], c, c))
iy = np.minimum(np.floor(np.arange(135) * np.float32(67 / 135)).astype(int), He - 1)
ix = np.minimum(np.floor(np.arange(240) * np.float32(120 / 240)).astype(int), We - 1)
s = q + e[iy][:, ix]
cc = leaky(conv(s, off["cw"], off["cb"], c, c))
hh = cc @ off["hw"].reshape(15, c).T + off["hb"]
sig = np.exp(np.clip(hh[..., :5], -4, 4))
lg = hh[..., 9:] + off["mb"]
lg = np.exp(lg - lg.max(-1, keepdims=True))
mix = lg / lg.sum(-1, keepdims=True)
rep = np.concatenate([sig, hh[..., 5:9], mix], -1)
net = WeightNet()
net.load_state_dict(torch.load(M.ckpt_path(sys.argv[2] if len(sys.argv) > 2 else "wnet2"), map_location="cpu")["net"])
net.eval()
with torch.no_grad():
    pt = net.low(torch.from_numpy(eF.astype(np.float32)).permute(2, 0, 1)[None])[0].permute(1, 2, 0).numpy()
for name, a, b in (("replica vs engine", rep, eM[..., :15]), ("replica vs PyTorch", rep, pt[..., :15]),
                   ("engine vs PyTorch", eM[..., :15], pt[..., :15])):
    d = np.abs(a - b)
    print("%-20s max %.2e mean %.2e  (sig1 max %.2e, mix5 max %.2e)" % (name, d.max(), d.mean(), d[..., 0].max(), d[..., 14].max()))
if os.path.exists(pre + ".lH.f32"):
    for nm, ref in (("H (conv a0)", h), ("Q (conv a1)", q), ("S (q + e)", s)):
        eng = np.fromfile(pre + ".l%s.f32" % nm[0], np.float32).reshape(135, 240, c)
        dd = np.abs(eng - ref)
        print("%-12s engine vs replica: max %.2e mean %.2e   (values mean |x| %.3f)" % (nm, dd.max(), dd.mean(), np.abs(ref).mean()))
    eng_h = np.fromfile(pre + ".lH.f32", np.float32).reshape(135, 240, c)
    q_from_eng_h = leaky(conv(eng_h.astype(np.float64), off["a1w"], off["a1b"], c, c))
    eng_q = np.fromfile(pre + ".lQ.f32", np.float32).reshape(135, 240, c)
    print("Q from engine H vs engine Q: max %.2e" % np.abs(q_from_eng_h - eng_q).max())
    eng_s = np.fromfile(pre + ".lS.f32", np.float32).reshape(135, 240, c)
    e_eng = eng_s - eng_q                               # the engine's upsampled e
    e_full = e[iy][:, ix]
    print("e: engine vs replica max %.2e mean %.2e" % (np.abs(e_eng - e_full).max(), np.abs(e_eng - e_full).mean()))
    # is the engine's e the replica's e from somewhere else? try every parent shift and pooling variants
    best = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = np.clip(iy + dy, 0, 66); xx = np.clip(ix + dx, 0, 119)
            best.append((np.abs(e_eng - e[yy][:, xx]).mean(), dy, dx))
    print("parent shifts (mean err, dy, dx):", sorted(best)[:3])
    # e computed WITHOUT pooling (q sampled at 2r) or with q itself (no pool)
    pq2 = q[0:134:2, 0:240:2]
    e2 = leaky(conv(pq2, off["bw"], off["bb"], c, c))
    print("no-average pooling variant: mean err %.2e" % np.abs(e_eng - e2[iy][:, ix]).mean())
    # zero padding vs something else at the 1/8 border: errors by parent row
    err_par = np.abs(e_eng - e_full).mean((1, 2))
    print("err by 1/4 row (first 12, last 12):", np.round(err_par[:12], 3), np.round(err_par[-12:], 3))
    err_c = np.abs(e_eng - e_full).mean((0, 2))
    print("err by 1/4 col (first 12, last 12):", np.round(err_c[:12], 3), np.round(err_c[-12:], 3))
