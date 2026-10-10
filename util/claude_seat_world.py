#!/usr/bin/env python3
"""claude_seat_world -- which world the test seat runs on, and DEMO MODE copies.

A test run should be a pure function of (build, world snapshot, path,
dials) (2026-10-09). The seat used to run on worlds/gallery itself, which
every run, every deploy and every playing session changes a little, and
the server kept simulating on it while a test ran (a row of roof blocks
once rendered as a smooth tube in one run: a falling block, most likely).
Two halves:

  * THE WORLD: a frozen SNAPSHOT of the gallery world, made once while no
    server runs on it (a btrfs reflink copy, so ~free), and a fresh per-run
    reflink copy of it for each test run. Nothing ever runs on the snapshot
    itself (its files are made read-only).
  * THE SERVER: the per-run copy carries the file claude_demo_freeze, which
    makes the bridge set the engine setting claude_freeze from the server's
    first step; the harness then calls the bridge's OPS.freeze to hold the
    mods' globalsteps too (util/claude_bridge_gallery.lua).

THE SWITCH: CLAUDE_SEAT_WORLD (env) is the world dir the seat's server runs
on, read by claude_ci (SEAT_WORLD, SERVER_CMD, the dial file) and
claude_lab (the bridge's file protocol lives in the world dir), and so by
everything built on them (motion, shoot, playtest). Unset = worlds/gallery,
exactly as before.

  python3 util/claude_seat_world.py which              # what the seat would run
  python3 util/claude_seat_world.py snapshot           # make a snapshot (no server may run on it)
  python3 util/claude_seat_world.py fresh [--snapshot DIR]   # a per-run copy; prints its path
  python3 util/claude_seat_world.py list
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_WORLD = "worlds/gallery"          # relative to the repo, as the seat always ran it
ENV = "CLAUDE_SEAT_WORLD"
SNAP_ROOT = os.path.expanduser("~/data/luanti-worlds")
RUNS = os.path.join(SNAP_ROOT, "runs")
SNAP_GLOB = "gallery-snap-*"
DEMO_MARK = "claude_demo_freeze"           # read by the bridge at load (claude_bridge_gallery.lua)
KEEP_RUNS = 6                              # per-run copies kept for looking at afterwards
# the bridge's per-session files: a copy must not inherit a stale request,
# answer or lock (a stale lock blocks every rpc for RPC_LOCK_STALE s)
SESSION_FILES = ("claude_cmd.json", "claude_out.json", "claude_rpc.lock")


def seat_world():
    """the seat's world as the server is given it (relative to the repo or absolute)"""
    return os.environ.get(ENV) or DEFAULT_WORLD


def seat_world_abs():
    return os.path.join(REPO, seat_world())


def servers_on(world):
    """PIDs of running luantiserver processes whose --world is `world`"""
    want = os.path.realpath(world)
    out = []
    r = subprocess.run(["pgrep", "-x", "luantiserver"], capture_output=True, text=True)
    for pid in r.stdout.split():
        try:
            argv = open("/proc/%s/cmdline" % pid, "rb").read().split(b"\0")
            cwd = os.readlink("/proc/%s/cwd" % pid)
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in argv]
        if "--world" in argv and argv.index("--world") + 1 < len(argv):
            w = os.path.realpath(os.path.join(cwd, argv[argv.index("--world") + 1]))
            if w == want:
                out.append(int(pid))
    return out


def seat_pids(world):
    """exact PIDs of the seat on `world`: its servers, and the headless clients
    (luanti under gamescope) started from this checkout, with their gamescope
    wrappers"""
    pids = servers_on(world)
    r = subprocess.run(["pgrep", "-x", "luanti"], capture_output=True, text=True)
    for pid in r.stdout.split():
        try:
            cwd = os.readlink("/proc/%s/cwd" % pid)
            chain, p = [], pid
            for _ in range(3):      # luanti <- gamescopereaper <- gamescope
                p = open("/proc/%s/stat" % p).read().rsplit(")", 1)[1].split()[1]
                if not open("/proc/%s/comm" % p).read().strip().startswith("gamescope"):
                    break
                chain.append(int(p))
        except OSError:
            continue
        if os.path.realpath(cwd) == os.path.realpath(REPO) and chain:
            pids += [int(pid)] + chain
    return pids


def stop_seat_on(world, wait=20.0):
    """A DEMO SEAT MUST NOT OUTLIVE ITS RUN (2026-10-10). Every other
    checkout's claude_ci.stop_seat finds the seat by "--world worlds/gallery",
    so a server on a per-run copy is invisible to them: one left up held port
    30000 and every other seat start failed. Stops the seat on `world` by
    exact PID (SIGTERM, then SIGKILL after `wait` s); returns the PIDs."""
    import signal
    pids = seat_pids(world)
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    t0 = time.time()
    while time.time() - t0 < wait and any(os.path.exists("/proc/%d" % p) for p in pids):
        time.sleep(0.5)
    for p in pids:
        if os.path.exists("/proc/%d" % p):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
    return pids


def snapshots():
    return sorted(d for d in glob.glob(os.path.join(SNAP_ROOT, SNAP_GLOB)) if os.path.isdir(d))


def make_snapshot(src=None, name=None):
    """a frozen reflink copy of the gallery world; refuses while a server runs on it"""
    src = os.path.realpath(src or os.path.join(REPO, DEFAULT_WORLD))
    busy = servers_on(src)
    if busy:
        sys.exit("REFUSED: luantiserver %s is running on %s; a snapshot of a live world "
                 "is not a snapshot" % (busy, src))
    dst = os.path.join(SNAP_ROOT, name or "gallery-snap-" + time.strftime("%Y%m%d-%H%M%S"))
    if os.path.exists(dst):
        sys.exit("REFUSED: %s exists" % dst)
    os.makedirs(SNAP_ROOT, exist_ok=True)
    subprocess.run(["cp", "-a", "--reflink=always", src, dst], check=True)
    if servers_on(src):
        # a server started during the copy: the copy may be torn
        shutil.rmtree(dst)
        sys.exit("REFUSED: a server started on %s during the copy" % src)
    for f in SESSION_FILES:
        if os.path.exists(os.path.join(dst, f)):
            os.unlink(os.path.join(dst, f))
    info = {"source": src, "made": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "files": {os.path.relpath(p, dst): os.path.getsize(p)
                      for p in glob.glob(os.path.join(dst, "*")) if os.path.isfile(p)}}
    json.dump(info, open(os.path.join(dst, "SNAPSHOT.json"), "w"), indent=1)
    # read-only: nothing runs on the snapshot itself (a server would fail to
    # open its map read-write, loudly, instead of quietly changing it)
    subprocess.run(["chmod", "-R", "a-w", dst], check=True)
    return dst


def _prune():
    """drop the oldest per-run copies beyond KEEP_RUNS (never one a server runs on)"""
    runs = sorted((d for d in glob.glob(os.path.join(RUNS, "*")) if os.path.isdir(d)),
                  key=os.path.getmtime)
    for d in runs[:-KEEP_RUNS] if len(runs) > KEEP_RUNS else []:
        real = os.path.realpath(d)
        if os.path.dirname(real) != os.path.realpath(RUNS) or not os.path.exists(os.path.join(real, DEMO_MARK)):
            print("claude_seat_world: not pruning %s (not a per-run copy)" % d, file=sys.stderr)
            continue
        if servers_on(real):
            continue
        shutil.rmtree(real)


def fresh(snapshot=None):
    """a per-run reflink copy of the snapshot (default: CLAUDE_SEAT_SNAPSHOT, else the
    newest), marked for the demo freeze; returns its absolute path"""
    snap = snapshot or os.environ.get("CLAUDE_SEAT_SNAPSHOT") or (snapshots() or [None])[-1]
    if not snap or not os.path.isfile(os.path.join(snap, "world.mt")):
        sys.exit("REFUSED: no world snapshot (%r); make one: python3 util/claude_seat_world.py snapshot"
                 % snap)
    os.makedirs(RUNS, exist_ok=True)
    _prune()
    dst = os.path.join(RUNS, "%s-%s-%d" % (os.path.basename(snap.rstrip("/")),
                                           time.strftime("%Y%m%d-%H%M%S"), os.getpid()))
    subprocess.run(["cp", "-a", "--reflink=always", snap.rstrip("/"), dst], check=True)
    subprocess.run(["chmod", "-R", "u+w", dst], check=True)
    for f in SESSION_FILES:
        if os.path.exists(os.path.join(dst, f)):
            os.unlink(os.path.join(dst, f))
    open(os.path.join(dst, DEMO_MARK), "w").write(
        "demo mode: the bridge sets claude_freeze from the server's first step\n"
        "snapshot: %s\n" % os.path.realpath(snap))
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["which", "snapshot", "fresh", "list"])
    ap.add_argument("--snapshot", help="fresh: the snapshot to copy (default the newest)")
    ap.add_argument("--src", help="snapshot: the world to snapshot (default worlds/gallery)")
    a = ap.parse_args()
    if a.cmd == "which":
        print(seat_world_abs())
    elif a.cmd == "snapshot":
        print(make_snapshot(a.src))
    elif a.cmd == "fresh":
        print(fresh(a.snapshot))
    else:
        for s in snapshots():
            print(s)
        for r in sorted(glob.glob(os.path.join(RUNS, "*"))):
            print("  run copy:", r, "(server %s)" % servers_on(r) if servers_on(r) else "")
