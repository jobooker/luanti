#!/usr/bin/env python3
"""claude_ml_guide_train — can a small network fill a guide table from one
noisy 8-frame window plus the voxels around the block, better than the
counts themselves? Held-out PLACE (never seen in training).

Score: importance-sampling efficiency of the mixture the shader samples,
p = 0.5 / 64 + 0.5 g (g = the table's distribution), against the target t
(the average of the other windows): eff = 1 / sum_b t_b^2 / p_b, 1.0 when
p = t. Weighted over tables by the light they carry.

  ~/.venvs/rtm-torch/bin/python util/claude_ml_guide_train.py DATA_DIR --test doorway-room
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

S, GW = 128, 8   # grid size; feature window, in coarse cells of 4


def read_windows(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "w*.bin"))):
        b = open(f, "rb").read()
        for chunk in [b]:
            nl = chunk.index(b"\n")
            hdr = chunk[:nl].decode().split()
            ox, oy, oz = int(hdr[3]), int(hdr[4]), int(hdr[5])
            cnt = int(hdr[9])
            rec = np.frombuffer(chunk[nl + 1:nl + 1 + cnt * 324], dtype=np.uint8).reshape(cnt, 324)
            idx = rec[:, :4].copy().view(np.int32)[:, 0]
            tab = rec[:, 4:].copy().view(np.uint32).reshape(cnt, 80)[:, :64].astype(np.float64)
            out.append(((ox, oy, oz), dict(zip(idx.tolist(), tab))))
    return out


def read_grid(d):
    b = open(os.path.join(d, "grid.bin"), "rb").read()
    nl = b.index(b"\n")
    h = b[:nl].decode().split()
    occ = np.frombuffer(b[nl + 1:nl + 1 + S ** 3 * 4], dtype=np.uint8).reshape(S, S, S, 4)   # z, y, x, rgba
    mat = np.frombuffer(b[nl + 1 + S ** 3 * 4:], dtype=np.float32).reshape(256, 8)
    solid = (occ[..., 3] > 0).astype(np.float32)
    emit = mat[occ[..., 3], 0] * (occ[..., :3].astype(np.float32).mean(-1) / 255.0)
    return (int(h[2]), int(h[3]), int(h[4])), solid, emit


def table_block(t, origin):
    face = t % 6
    b = t // 6
    m = np.array([b % 32, (b // 32) % 32, b // 1024])          # x, y, z (mod 32)
    o4 = np.floor(np.array(origin) / 4.0).astype(int)
    wb = o4 + ((m - o4) % 32)                                    # the world block inside this grid
    local = wb * 4 - np.array(origin)                            # its first cell, grid-local
    return face, local


def features(solid, emit, local, face):
    # a 32-cell cube centred on the block, pooled to 8^3 (4-cell coarse cells)
    c = local + 2 - 16
    f = np.zeros((2, 32, 32, 32), np.float32)
    lo = np.maximum(c, 0)
    hi = np.minimum(c + 32, S)
    if (hi > lo).all():
        f[0, lo[2] - c[2]:hi[2] - c[2], lo[1] - c[1]:hi[1] - c[1], lo[0] - c[0]:hi[0] - c[0]] = \
            solid[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
        f[1, lo[2] - c[2]:hi[2] - c[2], lo[1] - c[1]:hi[1] - c[1], lo[0] - c[0]:hi[0] - c[0]] = \
            emit[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
    g = f.reshape(2, 8, 4, 8, 4, 8, 4).mean((2, 4, 6)).reshape(-1)
    oh = np.zeros(6, np.float32)
    oh[face] = 1.0
    return np.concatenate([g, oh])


def build(d):
    wins = read_windows(d)
    origin, solid, emit = read_grid(d)
    rows = []
    for wi, (org, tabs) in enumerate(wins):
        if org != origin:
            continue          # the grid moved under this window; skip it
        for t, counts in tabs.items():
            # target: the other windows' average for this table
            others = [w[1][t] for k, w in enumerate(wins) if k != wi and t in w[1] and w[0] == origin]
            if len(others) < 8:
                continue
            tgt = np.sum(others, 0)
            if tgt.sum() <= 0:
                continue
            face, local = table_block(t, origin)
            rows.append((features(solid, emit, local, face), counts, tgt))
    return rows


def eff(t, g):
    t = t / t.sum()
    p = 0.5 / 64 + 0.5 * g
    return 1.0 / np.sum(t * t / p)


class Net(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(nin + 65, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                               nn.Linear(256, 64))

    def forward(self, x, c):
        tot = c.sum(1, keepdim=True)
        cn = c / tot.clamp(min=1e-9)
        return torch.log_softmax(self.m(torch.cat([x, cn, torch.log1p(tot)], 1)), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--test", default="doorway-room")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    places = sorted(p for p in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, p)))
    data = {p: build(os.path.join(a.data, p)) for p in places}
    for p in places:
        print("%-14s %6d (window, table) rows" % (p, len(data[p])))
    tr = [r for p in places if p != a.test for r in data[p]]
    te = data[a.test]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(np.array([r[0] for r in tr]), device=dev)
    C = torch.tensor(np.array([r[1] for r in tr]), dtype=torch.float32, device=dev)
    T = torch.tensor(np.array([r[2] / r[2].sum() for r in tr]), dtype=torch.float32, device=dev)
    W = torch.log1p(torch.tensor(np.array([r[2].sum() for r in tr]), dtype=torch.float32, device=dev))
    net = Net(X.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(a.epochs):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 512):
            j = perm[i:i + 512]
            lp = net(X[j], C[j])
            loss = -(W[j] * (T[j] * lp).sum(1)).mean()      # cross-entropy to the target, light-weighted
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        Xt = torch.tensor(np.array([r[0] for r in te]), device=dev)
        Ct = torch.tensor(np.array([r[1] for r in te]), dtype=torch.float32, device=dev)
        G = net(Xt, Ct).exp().cpu().numpy()
    res = {"uniform (no guide)": [], "counts, one window (today's guide)": [], "network": [], "target itself": []}
    wts = []
    for r, g in zip(te, G):
        t, c = r[2], r[1]
        wts.append(t.sum())
        res["uniform (no guide)"].append(eff(t, np.full(64, 1 / 64)))
        res["counts, one window (today's guide)"].append(eff(t, c / c.sum() if c.sum() > 0 else np.full(64, 1 / 64)))
        res["network"].append(eff(t, g))
        res["target itself"].append(eff(t, t / t.sum()))
    wts = np.array(wts)
    print("held-out place: %s (%d rows); importance-sampling efficiency, light-weighted (1.0 = perfect):" % (a.test, len(te)))
    for k, v in res.items():
        print("  %-38s %.4f" % (k, float(np.sum(np.array(v) * wts) / wts.sum())))
    torch.save(net.state_dict(), os.path.join(a.data, "guide_net_%s.pt" % a.test))


if __name__ == "__main__":
    main()
