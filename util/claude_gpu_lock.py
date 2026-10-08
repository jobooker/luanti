"""claude_gpu_lock — one game on the GPU at a time, across every checkout.

Starting a seat stops whatever game is running, and timings are only
honest with the GPU to themselves. Several worktrees (parallel work,
2026-10-08) each have their own tools and their own CI lock, so the lock
that matters lives outside all of them: /tmp/claude_gpu.lock. hold() blocks
until it is free and keeps it until this process exits (an flock dies with
its process, so a crash cannot wedge it). /tmp/claude_gpu.owner says who has it.
"""
import fcntl
import os
import sys
import time

LOCK = "/tmp/claude_gpu.lock"
OWNER = "/tmp/claude_gpu.owner"
_held = None


def hold(what=""):
    global _held
    if _held is not None:
        return
    f = open(LOCK, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            who = open(OWNER).read().strip()
        except OSError:
            who = "?"
        print("[gpu-lock] waiting for: %s" % who, file=sys.stderr, flush=True)
        fcntl.flock(f, fcntl.LOCK_EX)
    _held = f
    open(OWNER, "w").write("pid %d since %s in %s: %s\n" % (
        os.getpid(), time.strftime("%H:%M:%S"), os.getcwd(), what or " ".join(sys.argv)))
