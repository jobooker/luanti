import sys, os, json, numpy as np
sys.path.insert(0, "util")
os.environ["MLD_DATA"]="/tmp/mldsmoke"
import claude_mldenoise as M, claude_mldenoise_atrous as A
torch, nn, F, atrous, WeightNet = A.bits()
_, _, _, features, _, _, _ = M.torch_bits()
net = WeightNet(); net.load_state_dict(torch.load(M.ckpt_path("wnet"), map_location="cpu")["net"]); net.eval()
pre = "/tmp/mldsmoke/test_engine/cabin/s101_learned_4"
rc, code, _ = M.load_set(pre); ru, _, _ = M.load_set(pre, clip=False); den = M.load_den(pre)
t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float()
c = torch.from_numpy(code)[None, None]
with torch.no_grad():
    mod = net(features(t(rc), c, torch.tensor([0.4]))) if False else None
meta=json.load(open("/tmp/mldsmoke/test_engine/cabin/meta.json")); ex=meta["exposure"]
with torch.no_grad():
    mod = net(features(t(rc), c, torch.tensor([ex])))
    base = atrous(t(ru), c)[0].permute(1,2,0).numpy()
    out = atrous(t(ru), c, mod)[0].permute(1, 2, 0).numpy()
rel = (np.abs(out - den) / np.maximum(np.abs(den), 1e-3)).max(2)
bad = rel > 1e-2
ys, xs = np.nonzero(bad)
print("bad", bad.sum(), "rows hist", np.histogram(ys, bins=[0,4,8,100,270,440,532,536,540])[0], "cols", np.histogram(xs, bins=[0,4,8,480,952,956,960])[0])
cc = np.abs(code)
mixed = code < -0.5
print("bad on mixed pixels", (bad & mixed).sum(), "on sky", (bad & (np.abs(cc-(1+6*65536))<0.5)).sum(), "invalid", (bad & (cc<0.5)).sum())
# are learned maps differing? compare engine den vs pytorch base filter
relb = (np.abs(base - den) / np.maximum(np.abs(den), 1e-3)).max(2)
print("engine learned vs pytorch TODAY filter: median rel", np.median(relb))
