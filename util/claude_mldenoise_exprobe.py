"""which exposure did the engine's features use? (the forest gap instrument)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_mldenoise as M  # noqa: E402
import claude_mldenoise_atrous as A  # noqa: E402

torch, nn, F, atrous, WeightNet = A.bits()
_, _, _, features, _, _, _ = M.torch_bits()
torch.set_num_threads(16)
net = WeightNet()
net.load_state_dict(torch.load(M.ckpt_path("wnet2"), map_location="cpu")["net"])
net.eval()
sc = sys.argv[1]
d = os.path.join(M.DATA, "test_engine", sc)
ex = json.load(open(d + "/meta.json"))["exposure"]
pre = os.path.join(d, "s101_learned_4")
rc, code, _ = M.load_set(pre)
ru, _, _ = M.load_set(pre, clip=False)
den = M.load_den(pre)
t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float()
c = torch.from_numpy(code)[None, None]
for k in [float(x) for x in sys.argv[2:]] or (0.5, 1.0, 2.0, 1.0 / ex):
    with torch.no_grad():
        out = atrous(t(ru), c, net(features(t(rc), c, torch.tensor([ex * k]))))[0].permute(1, 2, 0).numpy()
    rel = np.abs(out - den) / np.maximum(np.abs(den), 1e-3)
    print("%-10s ex x %.4f (= %.5f): engine vs PyTorch display RMSE %.6f, rel p50 %.1e" % (
        sc, k, ex * k, M.rmse_disp(M.tonemap_np(out, ex), M.tonemap_np(den, ex)), float(np.median(rel))), flush=True)
