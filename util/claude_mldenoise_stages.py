"""the engine's network stage by stage against the PyTorch twin, on one dump
(lF = pooled features, lM = activated maps; claudeLearnedDump)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_mldenoise as M  # noqa: E402
import claude_mldenoise_atrous as A  # noqa: E402

torch, nn, F, atrous, WeightNet = A.bits()
_, _, _, features, _, _, _ = M.torch_bits()
net = WeightNet()
net.load_state_dict(torch.load(M.ckpt_path(sys.argv[2] if len(sys.argv) > 2 else "wnet2"), map_location="cpu")["net"])
net.eval()
pre = sys.argv[1]
d = os.path.dirname(pre)
ex = json.load(open(os.path.join(d, "meta.json")))["exposure"]
rc, code, _ = M.load_set(pre)
t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float()
c = torch.from_numpy(code)[None, None]
eF = np.fromfile(pre + ".lF.f32", np.float32).reshape(135, 240, 20)
eM = np.fromfile(pre + ".lM.f32", np.float32).reshape(135, 240, 16)
with torch.no_grad():
    pf = F.avg_pool2d(features(t(rc), c, torch.tensor([ex])), 4)[0].permute(1, 2, 0).numpy()
    pm = net.low(F.avg_pool2d(features(t(rc), c, torch.tensor([ex])), 4))[0].permute(1, 2, 0).numpy()
    pm_from_engine_f = net.low(torch.from_numpy(eF).permute(2, 0, 1)[None])[0].permute(1, 2, 0).numpy()
names = ["irr r", "irr g", "irr b", "dir r", "dir g", "dir b", "salb r", "salb g", "salb b", "n x", "n y", "n z",
         "sky", "none", "mixed", "dist", "same r", "same d", "sd", "N"]
print("pooled features, engine vs PyTorch (max abs diff, where):")
for i, n in enumerate(names):
    dd = np.abs(eF[..., i] - pf[..., i])
    y, x = np.unravel_index(dd.argmax(), dd.shape)
    print("  %-7s max %.2e at (row %d, col %d)  mean %.2e   engine %.4f torch %.4f" % (n, dd.max(), y, x, dd.mean(),
          eF[y, x, i], pf[y, x, i]))
# the engine's maps are packed: M0 = sig1-4, M1 = (sig5, f0, f1, f2), M2 = (f3, mix0-2), M3 = (mix3-5, 0)
pm_packed = pm[..., :15]
mn = ["sig1", "sig2", "sig3", "sig4", "sig5", "f0", "f1", "f2", "f3", "mix0", "mix1", "mix2", "mix3", "mix4", "mix5"]
print("maps, engine vs PyTorch-from-PyTorch-features | vs PyTorch-from-ENGINE-features:")
for i, n in enumerate(mn):
    a = np.abs(eM[..., i] - pm_packed[..., i])
    b = np.abs(eM[..., i] - pm_from_engine_f[..., i])
    print("  %-5s max %.2e mean %.2e | max %.2e mean %.2e" % (n, a.max(), a.mean(), b.max(), b.mean()))
err = np.abs(eM[..., :15] - pm_from_engine_f).max(2)
bad = err > 1e-3
ys, xs = np.nonzero(bad)
print("bad texels", bad.sum(), "of", bad.size)
print("rows with bad:", np.unique(ys)[:40], "... count per row max", np.bincount(ys).max() if len(ys) else 0)
print("cols with bad:", np.unique(xs)[:40])
# intermediate: is it the 1/8 branch? compare a fresh torch run with the e-branch zeroed vs engine is impossible; print row pattern mod 2
print("bad rows mod 2:", np.bincount(ys % 2), " cols mod 2:", np.bincount(xs % 2))
