"""mode 2 (lean) against mode 1 in PyTorch, on the in-engine test's inputs:
how much does dropping the affinity features and rounding the maps to fp16
change the picture and its error?"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_mldenoise as M  # noqa: E402
import claude_mldenoise_atrous as A  # noqa: E402

torch, nn, F, atrous, WeightNet = A.bits()
_, _, _, features, _, _, _ = M.torch_bits()
torch.set_num_threads(8)
net = WeightNet()
net.load_state_dict(torch.load(M.ckpt_path("wnet2"), map_location="cpu")["net"])
net.eval()
for sc in ("forest", "plains", "cabin", "torchroom"):
    d = os.path.join(M.DATA, "test_engine", sc)
    meta = json.load(open(d + "/meta.json"))
    ex = meta["exposure"]
    tp = os.path.join(d, meta["truth"]["prefix"])
    j = json.load(open(tp + ".json"))
    T = M.tonemap_np(np.fromfile(tp + ".accum.f32", np.float32).reshape(j["h"], j["w"], 4)[..., :3], ex)
    for n in (1, 16, 64):
        pre = os.path.join(d, "s101_filter_%d" % n)
        rc, code, _ = M.load_set(pre)
        ru, _, _ = M.load_set(pre, clip=False)
        t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float()
        c = torch.from_numpy(code)[None, None]
        with torch.no_grad():
            f = features(t(rc), c, torch.tensor([ex]))
            o1 = atrous(t(ru), c, net(f))[0].permute(1, 2, 0).numpy()
            o2 = atrous(t(ru), c, A.lean_mod(net, f))[0].permute(1, 2, 0).numpy()
        D1, D2 = M.tonemap_np(o1, ex), M.tonemap_np(o2, ex)
        print("%-10s %2d fr  mode1 RMSE %.5f  mode2 RMSE %.5f  | mode2 vs mode1 display RMSE %.6f" % (
            sc, n, M.rmse_disp(D1, T), M.rmse_disp(D2, T), M.rmse_disp(D1, D2)), flush=True)
