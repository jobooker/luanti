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
  freeze                     time_speed 0 + prove the accumulator deepens
  vantage NAME               teleport + aim at a saved vantage
  tour [--set NAME]          visit saved vantages, capture each
  compare A B                RMS difference between two screenshots

Every `shot` also writes <shot>.capture.json beside the PNG: build sha,
branch, the conf's claude_* dials, the live settings-patch lines from
debug.txt, the frame stats, and time_speed. Evidence expires per build
AND per config — a golden without its config record is not evidence
(2026-08-14: a whole session judged frames with no config record at all).

Vantages live in util/claude_vantages.json (same shape as the beelink
vantages.json, plus optional "time" applied before the shot).
Env: CLAUDE_MT_DIR (user dir; defaults to the repo root — RUN_IN_PLACE),
     CLAUDE_WORLD (world dir; defaults to <repo>/worlds/gallery).
"""
import argparse, json, os, subprocess, sys, time, glob

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MT = os.environ.get("CLAUDE_MT_DIR") or REPO
WORLD = os.environ.get("CLAUDE_WORLD") or os.path.join(REPO, "worlds", "gallery")
PATCH = os.path.join(MT, "claude_settings_patch.conf")
SHOTS = os.path.join(MT, "screenshots")
STATS = os.path.join(MT, "claude_stats.json")
CONF = os.path.join(MT, "minetest.conf")
DEBUG = os.path.join(MT, "debug.txt")
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


def shot(token=None, settle=2.5, record=True):
    before = newest_shot()
    time.sleep(settle)
    doorway(claude_screenshot=token or ("t%d" % int(time.time())))
    for _ in range(20):
        cur = newest_shot()
        if cur and cur != before:
            time.sleep(0.4)  # let the write finish
            return (write_capture_record(cur) or cur) if record else cur
        time.sleep(0.5)
    raise RuntimeError("no screenshot appeared (client running? mode on?)")


# ---------------------------------------------------------------- evidence

def _git(*a):
    try:
        return subprocess.run(("git",) + a, cwd=REPO, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def conf_dials(path=CONF):
    """The claude_* lines of a conf file, as a dict."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line.startswith("claude_") and "=" in line:
                    k, _, v = line.partition("=")
                    out[k.strip()] = v.strip()
    except Exception:
        pass
    return out


def patch_log(limit=60):
    """Live dial overrides, from the client's own [claude_settings_patch]
    log. That log is ground truth: doorway() truncates the patch file, so
    the file shows only the LAST write while g_settings keeps them all."""
    try:
        with open(DEBUG, errors="replace") as f:
            lines = [l.strip() for l in f if "[claude_settings_patch]" in l]
        return lines[-limit:]
    except Exception:
        return []


def read_stats():
    try:
        return json.load(open(STATS))
    except Exception:
        return None


def get_time_speed():
    """Ask the server for time_speed. A runtime /set shadows the conf, so
    the conf value is not evidence — the server's is."""
    try:
        r = rpc("cmd", command="set", param="time_speed")
        return (r or {}).get("msg")
    except Exception:
        return None


def capture_record():
    st = read_stats()
    return {
        "shot_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "sha": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "binary_mtime": (time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(os.path.getmtime(os.path.join(REPO, "bin", "luanti"))))
            if os.path.exists(os.path.join(REPO, "bin", "luanti")) else None),
        "conf_dials": conf_dials(),
        "patch_log": patch_log(),
        "stats": st,
        "time_speed": get_time_speed(),
        "frozen": _frozen_verdict(st),
    }


def _frozen_verdict(st):
    """still_frames == 1 means the accumulator is being reset every frame
    (a moving sun does it), so the frame is #1 repeated, not converged."""
    if not st:
        return "unknown (no stats)"
    sf = st.get("still_frames")
    al = st.get("accum_alpha")
    if sf is None:
        return "unknown"
    if sf <= 2:
        return "NOT CONVERGED (still_frames %s) — freeze time" % sf
    if al is not None and al > 0.08:
        return "converging (still_frames %s, alpha %s)" % (sf, al)
    return "converged (still_frames %s, alpha %s)" % (sf, al)


def write_capture_record(png):
    """Write <shot>.capture.json beside a PNG. Never fails a capture."""
    try:
        rec = capture_record()
        with open(os.path.splitext(png)[0] + ".capture.json", "w") as f:
            json.dump(rec, f, indent=2)
    except Exception as e:
        print("WARNING: capture record failed: %s" % e, file=sys.stderr)
    return png


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
    p = shot(args.name, settle=args.settle)
    print(p)
    rec = os.path.splitext(p)[0] + ".capture.json"
    if os.path.exists(rec):
        r = json.load(open(rec))
        print("  %s@%s  time_speed=%s  %s"
              % (r.get("branch"), r.get("sha"), r.get("time_speed"),
                 r.get("frozen")))


def cmd_freeze(args):
    """LAB RULE #1: freeze time before judging anything. A sun that moves
    zeroes still_frames every frame, so the accumulator never deepens and
    photo mode never engages — every capture is frame 1, repeated."""
    rpc("cmd", command="set", param="time_speed 0")
    print("set: %s" % get_time_speed())
    doorway(claude_stats=1)
    time.sleep(1.5)
    a = read_stats()
    if a is None:
        print("NO STATS: is claude_stats enabled and the client running?")
        sys.exit(2)
    print("still_frames %s, accum_alpha %s -> waiting %.0fs"
          % (a.get("still_frames"), a.get("accum_alpha"), args.wait))
    time.sleep(args.wait)
    doorway(claude_stats=1)
    time.sleep(1.5)
    b = read_stats() or {}
    sa, sb = a.get("still_frames", 0), b.get("still_frames", 0)
    print("still_frames %s -> %s, accum_alpha %s -> %s"
          % (sa, sb, a.get("accum_alpha"), b.get("accum_alpha")))
    if sb <= sa:
        print("FAIL: the accumulator is NOT deepening. Something is still "
              "invalidating it (sun? camera? scene churn?). Do not judge "
              "any capture until this climbs.")
        sys.exit(1)
    print("OK: accumulator deepening.")


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
    p = sub.add_parser("freeze"); p.add_argument("--wait", type=float,
        default=6.0); p.set_defaults(func=cmd_freeze)
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
