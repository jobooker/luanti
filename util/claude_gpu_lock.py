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
    """Take the lock for this process's whole life and hand it down: child
    processes see CLAUDE_GPU_LOCK_HELD and do not try to take it again (an
    flock is per open file, so a child taking it would wait for its own
    parent). Call it at the top of every tool that drives the game, not only
    where the game is started: a tool that starts the game in a short child
    and keeps using it would otherwise release the lock mid-run (found
    2026-10-08)."""
    global _held
    if _held is not None or os.environ.get("CLAUDE_GPU_LOCK_HELD"):
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
    os.environ["CLAUDE_GPU_LOCK_HELD"] = str(os.getpid())   # children inherit it
    open(OWNER, "w").write("pid %d since %s in %s: %s\n" % (
        os.getpid(), time.strftime("%H:%M:%S"), os.getcwd(), what or " ".join(sys.argv)))
