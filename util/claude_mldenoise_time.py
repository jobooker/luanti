"""careful timing: wall clock over many frames, with and without a GPU graph
(the graph removes Python's per-kernel launch cost, which an in-engine
implementation would not pay)"""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser("~/code/luanti-wt/mldenoise/util"))
import claude_gpu_lock
claude_gpu_lock.hold("mldenoise careful timing")
import numpy as np
import claude_mldenoise as M
torch, nn, F, features, Net, radiance, tonemap = M.torch_bits()
out = {}
for tag in sys.argv[1:]:
    net, st = M.load_net(tag)
    res = {}
    for dtype, dn in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        m = M.make_net(Net, st["cfg"]).cuda().eval().to(dtype)
        m.load_state_dict({k: v.to(dtype) for k, v in net.state_dict().items()})
        x = torch.rand(1, 20, 544, 960, device="cuda", dtype=dtype)
        with torch.no_grad():
            for _ in range(50):
                m(x)
            torch.cuda.synchronize()
            for rep in range(3):
                t0 = time.perf_counter()
                for _ in range(200):
                    m(x)
                torch.cuda.synchronize()
                wall = (time.perf_counter() - t0) / 200 * 1000
            res[dn + "_wall_ms"] = wall
            # graph
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(5):
                        y = m(x)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    y = m(x)
                for _ in range(20):
                    g.replay()
                torch.cuda.synchronize()
                ts = []
                for rep in range(5):
                    t0 = time.perf_counter()
                    for _ in range(200):
                        g.replay()
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) / 200 * 1000)
                res[dn + "_graph_ms"] = float(np.median(ts))
            except Exception as e:
                res[dn + "_graph_ms"] = "failed: %s" % e
        print(tag, dn, res, flush=True)
    out[tag] = res
json.dump(out, open(os.path.join(M.DATA, "time", "careful.json"), "w"), indent=1)
