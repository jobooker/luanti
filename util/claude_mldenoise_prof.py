import os, sys
sys.path.insert(0, os.path.expanduser("~/code/luanti-wt/mldenoise/util"))
import claude_gpu_lock
claude_gpu_lock.hold("mldenoise profile")
import torch
import claude_mldenoise as M
import claude_mldenoise_atrous as A
torch_, nn, F, atrous, WeightNet = A.bits()
_, _, _, features, _, _, _ = M.torch_bits()
H, W = 540, 960
raw = torch.rand(1, 13, H, W, device="cuda") * 2
raw[:, 11] = 0.05
code = (torch.randint(0, 6, (1, 1, H, W), device="cuda") * 65536 + 1000).float()
ex = torch.tensor([0.3], device="cuda")
net = WeightNet().cuda().eval().half()
mode = sys.argv[1] if len(sys.argv) > 1 else "eager"


def fe(raw, code, ex):
    return net.activate(net.raw_out(features(raw, code, ex).half().permute(0, 2, 3, 1).contiguous()))


fn = torch.compile(fe) if mode == "compiled" else fe
with torch.no_grad():
    for _ in range(20):
        fn(raw, code, ex)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(20):
            fn(raw, code, ex)
        torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=60))
