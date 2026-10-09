"""held-out views on the CPU: today's filter (the port) against the weight
net, display RMSE and the gradient term, plus what the net asks for"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_mldenoise as M  # noqa: E402
import claude_mldenoise_atrous as A  # noqa: E402

torch, nn, F, atrous, WeightNet = A.bits()
_, _, _, features, _, _, tonemap = M.torch_bits()
torch.set_num_threads(8)
net = WeightNet()
st = torch.load(M.ckpt_path(sys.argv[1]), map_location="cpu")
net.load_state_dict(st["net"])
net.eval()
print("step", st["step"])
for name in json.load(open(os.path.join(M.DATA, "holdout.json"))):
    m, tgt, samples = M.load_view(os.path.join(M.DATA, "views", name))
    ex = m["exposure"]
    T = torch.from_numpy(tgt).permute(2, 0, 1)[None]
    for run, n, raw, code in samples:
        if run != "A":
            continue
        r = torch.from_numpy(raw).permute(2, 0, 1)[None]
        c = torch.from_numpy(code)[None, None]
        e = torch.tensor([ex])
        with torch.no_grad():
            b = atrous(r, c)
            mod = net(features(r, c, e))
            w = atrous(r, c, mod)
            dt, db, dw = tonemap(T * ex), tonemap(b * ex), tonemap(w * ex)
            print("%-17s %3d fr  RMSE filter %.4f  wnet %.4f | grad term filter %.4f wnet %.4f | sig %s mix %s" % (
                name, n, float(((db - dt) ** 2).mean().sqrt()), float(((dw - dt) ** 2).mean().sqrt()),
                float(A.grad_l1(torch, db, dt)), float(A.grad_l1(torch, dw, dt)),
                np.round(mod["sig"].mean((0, 2, 3)).numpy(), 2), np.round(mod["mix"].mean((0, 2, 3)).numpy(), 3)),
                flush=True)
