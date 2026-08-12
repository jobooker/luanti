#!/usr/bin/env python3
"""claude_lab (local seat) — render-testing harness for the traced
Luanti client, talking to a claude_bridge server on THIS machine.
Port of jobooker/luanti-server client/claude_lab.py: the rpc goes
through the local world dir instead of ssh'ing to the beelink.

Subcommands:
  rpc OP [k=v ...]           raw bridge call (JSON values accepted)
  stats                      read the client's rolling frame stats
  shot NAME [--settle S]     capture one screenshot, print its path
  set K=V [K=V ...]          write doorway settings (live, ~1 s)
  vantage NAME               teleport + aim at a saved vantage
  tour [--set NAME]          visit saved vantages, capture each
  compare A B                RMS difference between two screenshots

Vantages live in util/claude_vantages.json (same shape as the beelink
vantages.json, plus optional "time" applied before the shot).
Env: CLAUDE_MT_DIR (user dir; defaults to the repo root — RUN_IN_PLACE),
     CLAUDE_WORLD (world dir; defaults to <repo>/worlds/gallery).
"""
import argparse, json, os, sys, time, glob

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MT = os.environ.get("CLAUDE_MT_DIR") or REPO
WORLD = os.environ.get("CLAUDE_WORLD") or os.path.join(REPO, "worlds", "gallery")
PATCH = os.path.join(MT, "claude_settings_patch.conf")
SHOTS = os.path.join(MT, "screenshots")
STATS = os.path.join(MT, "claude_stats.json")
VANTAGES = os.path.join(HERE, "claude_vantages.json")


def doorway(**kv):
    """Apply live client settings via the settings-patch channel."""
    with open(PATCH, "w") as f:
        for k, v in kv.items():
            f.write("%s = %s\n" % (k, v))
    time.sleep(1.4)  # poller runs at ~1 Hz


def rpc(op, **kw):
    """One bridge call over the world-dir file protocol."""
    req = dict(id="lab%d" % time.time_ns(), op=op, **kw)
    with open(os.path.join(WORLD, "claude_cmd.json"), "w") as f:
        json.dump(req, f)
    deadline = time.time() + 30
    outp = os.path.join(WORLD, "claude_out.json")
    while time.time() < deadline:
        time.sleep(0.3)
        try:
            with open(outp) as f:
                out = json.load(f)
        except Exception:
            continue
        if out.get("id") == req["id"]:
            if not out.get("ok"):
                raise RuntimeError("%s: %s" % (op, out.get("error")))
            return out.get("result")
    raise RuntimeError("bridge timeout on " + op)


def newest_shot():
    files = glob.glob(os.path.join(SHOTS, "*.png"))
    return max(files, key=os.path.getmtime) if files else None


def shot(token=None, settle=2.5):
    before = newest_shot()
    time.sleep(settle)
    doorway(claude_screenshot=token or ("t%d" % int(time.time())))
    for _ in range(20):
        cur = newest_shot()
        if cur and cur != before:
            time.sleep(0.4)  # let the write finish
            return cur
        time.sleep(0.5)
    raise RuntimeError("no screenshot appeared (client running? mode on?)")


def rms_diff(a, b, crop_hud=True):
    from PIL import Image
    import numpy as np
    ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
    ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
    if ia.shape != ib.shape:
        raise RuntimeError("size mismatch %s vs %s" % (ia.shape, ib.shape))
    if crop_hud:
        h = ia.shape[0]
        ia = ia[int(h * 0.12):int(h * 0.86)]
        ib = ib[int(h * 0.12):int(h * 0.86)]
    d = ia - ib
    return {"rms": round(float((d * d).mean() ** 0.5), 3),
            "worst_channel_delta": round(float(abs(d).max()), 1)}


def load_vantages():
    return json.load(open(VANTAGES)) if os.path.exists(VANTAGES) else {}


def goto(v):
    if "time" in v:
        rpc("time", set=v["time"])
    rpc("tp", pos=dict(x=v["pos"][0], y=v["pos"][1], z=v["pos"][2]),
        yaw=v["yaw"], pitch=v["pitch"])
    time.sleep(1.0)


def parse_kv(pairs):
    kv = {}
    for pair in pairs:
        k, _, v = pair.partition("=")
        try:
            kv[k.strip()] = json.loads(v.strip())
        except Exception:
            kv[k.strip()] = v.strip()
    return kv


def cmd_rpc(args):
    print(json.dumps(rpc(args.op, **parse_kv(args.kv)), indent=2))


def cmd_stats(args):
    doorway(claude_stats=1)
    time.sleep(1.5)
    print(open(STATS).read().strip())


def cmd_shot(args):
    print(shot(args.name, settle=args.settle))


def cmd_set(args):
    kv = parse_kv(args.kv)
    doorway(**kv)
    print("applied: %s" % kv)


def cmd_vantage(args):
    vs = load_vantages()
    if args.name not in vs:
        print("known vantages: %s" % ", ".join(sorted(vs)))
        return
    goto(vs[args.name])
    print("at %s" % args.name)


def cmd_tour(args):
    vs = load_vantages()
    names = [n for n in sorted(vs)
             if not args.set or vs[n].get("set") == args.set]
    out = []
    for n in names:
        goto(vs[n])
        doorway(claude_volume_snapshot="tour_" + n)
        p = shot("tour_" + n, settle=args.settle)
        out.append({"vantage": n, "shot": p})
        print("%-24s -> %s" % (n, os.path.basename(p)))
    print(json.dumps(out, indent=2))


def cmd_compare(args):
    print(json.dumps(rms_diff(args.a, args.b), indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rpc"); p.add_argument("op")
    p.add_argument("kv", nargs="*"); p.set_defaults(func=cmd_rpc)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    p = sub.add_parser("shot"); p.add_argument("name", nargs="?")
    p.add_argument("--settle", type=float, default=2.5)
    p.set_defaults(func=cmd_shot)
    p = sub.add_parser("set"); p.add_argument("kv", nargs="+")
    p.set_defaults(func=cmd_set)
    p = sub.add_parser("vantage"); p.add_argument("name")
    p.set_defaults(func=cmd_vantage)
    p = sub.add_parser("tour"); p.add_argument("--set")
    p.add_argument("--settle", type=float, default=3.0)
    p.set_defaults(func=cmd_tour)
    p = sub.add_parser("compare"); p.add_argument("a"); p.add_argument("b")
    p.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
