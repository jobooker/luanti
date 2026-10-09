"""how sensitive is the learned filter to float-rounding-sized changes in its
features? (separates precision from a port bug in the engine/PyTorch gap)"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_mldenoise as M, claude_mldenoise_atrous as A
torch, nn, F, atrous, WeightNet = A.bits()
_, _, _, features, _, _, _ = M.torch_bits()
torch.set_num_threads(16)
net = WeightNet(); net.load_state_dict(torch.load(M.ckpt_path("wnet2"), map_location="cpu")["net"]); net.eval()
for sc in sys.argv[1:]:
    d = os.path.join(M.DATA, "test_engine", sc); ex = json.load(open(d + "/meta.json"))["exposure"]
    pre = os.path.join(d, "s101_learned_4")
    rc, code, _ = M.load_set(pre); ru, _, _ = M.load_set(pre, clip=False)
    t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float()
    c = torch.from_numpy(code)[None, None]
    with torch.no_grad():
        f = features(t(rc), c, torch.tensor([ex]))
        mod = net(f); out = atrous(t(ru), c, mod)[0].permute(1, 2, 0).numpy()
        g = torch.Generator().manual_seed(0)
        for eps in (1e-6, 1e-5):
            f2 = f * (1 + eps * torch.randn(f.shape, generator=g))
            mod2 = net(f2); out2 = atrous(t(ru), c, mod2)[0].permute(1, 2, 0).numpy()
            rel = np.abs(out2 - out) / np.maximum(np.abs(out), 1e-3)
            print("%-10s features x(1 + %.0e noise): display RMSE change %.6f, share >1%% %.5f, feat range %.1f..%.1f" % (
                sc, eps, M.rmse_disp(M.tonemap_np(out2, ex), M.tonemap_np(out, ex)), float((rel.max(2) > 1e-2).mean()),
                float(mod["feat"].min()), float(mod["feat"].max())), flush=True)
