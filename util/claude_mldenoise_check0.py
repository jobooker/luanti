"""dial 0 in this build: today's six passes (now wrapped) against the exact
PyTorch port, on the filter arm of the in-engine test. The same check before
the change matched to 5e-6."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_mldenoise as M  # noqa: E402
import claude_mldenoise_atrous as A  # noqa: E402

torch, nn, F, atrous, WeightNet = A.bits()
for sc in ("forest", "plains", "cabin", "torchroom"):
    d = os.path.join(M.DATA, "test_engine", sc)
    ex = json.load(open(d + "/meta.json"))["exposure"]
    for n in (1, 16, 64):
        pre = os.path.join(d, "s101_filter_%d" % n)
        raw, code, _ = M.load_set(pre, clip=False)
        den = M.load_den(pre)
        with torch.no_grad():
            out = atrous(torch.from_numpy(raw).permute(2, 0, 1)[None].double(),
                         torch.from_numpy(code)[None, None].double())[0].permute(1, 2, 0).numpy()
        rel = np.abs(out - den) / np.maximum(np.abs(den), 1e-3)
        print("%-10s %2d fr  dial 0 vs the port of today's filter: max rel %.1e, display RMSE %.6f" % (
            sc, n, rel.max(), M.rmse_disp(M.tonemap_np(out, ex), M.tonemap_np(den, ex))), flush=True)
