"""claude_gpu_lock — one game on the GPU at a time, across every checkout.

Starting a seat stops whatever game is running, and timings are only
honest with the GPU to themselves. Several worktrees (parallel work,
2026-10-08) each have their own tools and their own CI lock, so the lock
that matters lives outside all of them: /tmp/claude_gpu.lock. hold() blocks
until it is free and keeps it until this process exits (an flock dies with
its process, so a crash cannot wedge it). /tmp/claude_gpu.owner says who has it.
"""
import atexit
import fcntl
import os
import sys
import time

LOCK = "/tmp/claude_gpu.lock"
QUEUE = "/tmp/claude_gpu.queue"
OWNER = "/tmp/claude_gpu.owner"
_held = None


def _alive(ticket):
    try:
        os.kill(int(ticket.split("-")[1]), 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


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
    # FIRST COME, FIRST SERVED (2026-10-08): a bare flock wakes waiters in
    # no particular order, and a lane that launches a new job every few
    # minutes starved one that had waited 50. Each waiter leaves a ticket
    # (arrival time, pid) in QUEUE; only the oldest LIVE ticket may take the
    # lock. A crashed waiter's ticket is skipped (its pid is gone).
    os.makedirs(QUEUE, exist_ok=True)
    ticket = "%020d-%d" % (time.time_ns(), os.getpid())
    tpath = os.path.join(QUEUE, ticket)
    open(tpath, "w").write(what or " ".join(sys.argv))
    atexit.register(lambda: os.path.exists(tpath) and os.unlink(tpath))
    f = open(LOCK, "a+")
    told = False
    while True:
        ahead = [t for t in os.listdir(QUEUE) if t < ticket and _alive(t)]
        if not ahead:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
        if not told:
            try:
                who = open(OWNER).read().strip()
            except OSError:
                who = "?"
            print("[gpu-lock] waiting (%d ahead in the queue) for: %s" % (len(ahead), who),
                  file=sys.stderr, flush=True)
            told = True
        time.sleep(1.0)
    os.unlink(tpath)
    _held = f
    os.environ["CLAUDE_GPU_LOCK_HELD"] = str(os.getpid())   # children inherit it
    open(OWNER, "w").write("pid %d since %s in %s: %s\n" % (
        os.getpid(), time.strftime("%H:%M:%S"), os.getcwd(), what or " ".join(sys.argv)))
