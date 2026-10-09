import os, sys, time
sys.path.insert(0, os.path.expanduser("~/code/luanti-wt/mldenoise/util"))
import claude_gpu_lock
claude_gpu_lock.hold("mldenoise micro-benchmark")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.backends.cudnn.benchmark = True


def wall(fn, n=100):
    with torch.no_grad():
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000


H, W = 540, 960
for dt in (torch.float32, torch.float16):
    x = torch.rand(1, 44, H, W, device="cuda", dtype=dt)
    c1 = nn.Conv2d(44, 16, 1).cuda().to(dt)
    print(dt, "conv1x1 44->16 full res NCHW %.3f ms" % wall(lambda: c1(x)))
    xl = x.permute(0, 2, 3, 1).contiguous()
    wt = c1.weight.view(16, 44).t().contiguous()
    print(dt, "matmul 44->16 full res NHWC %.3f ms" % wall(lambda: xl @ wt))
    xc = x.contiguous(memory_format=torch.channels_last)
    c1c = c1.to(memory_format=torch.channels_last)
    print(dt, "conv1x1 channels_last %.3f ms" % wall(lambda: c1c(xc)))
    lo = torch.rand(1, 24, H // 4, W // 4, device="cuda", dtype=dt)
    c3 = nn.Conv2d(24, 24, 3, padding=2, dilation=2).cuda().to(dt)
    print(dt, "conv3x3 dil2 24ch quarter res %.3f ms" % wall(lambda: c3(lo)))
    c3b = nn.Conv2d(24, 24, 3, padding=1).cuda().to(dt)
    print(dt, "conv3x3 24ch quarter res %.3f ms" % wall(lambda: c3b(lo)))
    f = torch.rand(1, 24, H, W, device="cuda", dtype=dt)
    c3f = nn.Conv2d(24, 24, 3, padding=1).cuda().to(dt)
    print(dt, "conv3x3 24ch FULL res %.3f ms" % wall(lambda: c3f(f)))
    up = lambda: F.interpolate(lo, size=(H, W), mode="bilinear", align_corners=False)
    print(dt, "bilinear upsample 24ch %.3f ms" % wall(up))
    print(dt, "avgpool4 20ch %.3f ms" % wall(lambda: F.avg_pool2d(x[:, :20], 4)))
    print(dt, "elementwise exp 3ch full %.3f ms" % wall(lambda: torch.exp(x[:, :3])))
print("MIOPEN_FIND_MODE", os.environ.get("MIOPEN_FIND_MODE"))
