"""forest: is the engine/PyTorch gap the moments (fp32 in the engine, fp64 in load_set)?"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_mldenoise as M, claude_mldenoise_atrous as A
torch, nn, F, atrous, WeightNet = A.bits()
_, _, _, features, _, _, _ = M.torch_bits()
torch.set_num_threads(16)
net = WeightNet(); net.load_state_dict(torch.load(M.ckpt_path("wnet2"), map_location="cpu")["net"]); net.eval()
sc = sys.argv[1] if len(sys.argv) > 1 else "forest"
d = os.path.join(M.DATA, "test_engine", sc); meta = json.load(open(d + "/meta.json")); ex = meta["exposure"]
pre = os.path.join(d, "s101_learned_4")
rc, code, _ = M.load_set(pre); ru, _, _ = M.load_set(pre, clip=False); den = M.load_den(pre)
mo = np.fromfile(pre + ".mom.f32", np.float32).reshape(540, 960, 4)
m2, vf, m1 = mo[..., 0], mo[..., 1], mo[..., 2]
sd32 = np.sqrt(np.maximum(m2 - m1 * m1, np.float32(0)) * np.clip(vf, 0, 1)).astype(np.float32)
print("sd fp64 vs fp32: share differing >1%:", float((np.abs(rc[..., 10] - np.clip(sd32, 0, 6e4)) > 0.01 * np.abs(rc[..., 10]) + 1e-6).mean()))
t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float()
c = torch.from_numpy(code)[None, None]
for name, r_feat in (("fp64 sd (as trained)", rc), ("fp32 sd (as the engine)", np.concatenate([rc[..., :10], np.clip(sd32, 0, 6e4)[..., None], rc[..., 11:]], -1))):
    with torch.no_grad():
        out = atrous(t(ru), c, net(features(t(r_feat), c, torch.tensor([ex]))))[0].permute(1, 2, 0).numpy()
    rel = np.abs(out - den) / np.maximum(np.abs(den), 1e-3)
    print("%-26s engine vs PyTorch display RMSE %.6f  share >1%% %.5f" % (name, M.rmse_disp(M.tonemap_np(out, ex), M.tonemap_np(den, ex)), float((rel.max(2) > 1e-2).mean())))
bad = rel.max(2) > 1e-2
cc = np.abs(code)
cats = {"sky": np.abs(cc - (1 + 6 * 65536)) < 0.5, "none(code 0)": cc < 0.5, "mixed": code < -0.5}
cats["face"] = ~(cats["sky"] | cats["none(code 0)"] | cats["mixed"])
for k, m in cats.items():
    print("%-13s pixels %6d  bad %6d (%.3f)" % (k, m.sum(), (bad & m).sum(), (bad & m).sum() / max(1, m.sum())))
ys, xs = np.nonzero(bad)
print("bad rows by band of 54:", np.histogram(ys, bins=10, range=(0, 540))[0])
print("bad cols by band of 96:", np.histogram(xs, bins=10, range=(0, 960))[0])
