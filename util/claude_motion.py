#!/usr/bin/env python3
"""claude_motion — drive a camera path frame by frame and dump what the
player saw, for the motion judge.

The client walks the path by FRAME INDEX (claude_path), so two runs see
the same poses; the dump (claude_dump) reads frames back asynchronously.
Prints the dump directory and the frame pacing measured by the client.

  python3 util/claude_motion.py --path util/paths/plains-walk.txt \
      --frames 600 --name plains-walk
  python3 util/claude_motion.py ... --nodump      # same run, no dump (perf arm)

Path file: one key per line, "frame x y z yaw pitch" (vantage convention,
pitch up positive); poses between keys are linear, the last is held.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_ci as ci                       # noqa: E402
import claude_lab as lab                     # noqa: E402
import claude_denoise_price as price         # noqa: E402

DUMP_ROOT = os.path.join(ci.REPO, "screenshots", "dump")


def path_len(path):
    keys = [l.split() for l in open(path) if l.strip()]
    return int(float(keys[-1][0])) + 1


def pctl(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--frames", type=int, default=0,
                    help="frames to dump (default: the whole path)")
    ap.add_argument("--dial", action="append", default=[])
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--name", default="motion")
    ap.add_argument("--skip-seat", action="store_true")
    ap.add_argument("--nodump", action="store_true")
    ap.add_argument("--mover", help="claude_mover file: a node moved on the path clock")
    ap.add_argument("--play", action="store_true",
                    help="PLAY settings: push only the capture mechanics, so the "
                         "renderer runs the game's defaults (as claude_look.sh)")
    args = ap.parse_args()
    path = os.path.abspath(args.path)
    n = args.frames or path_len(path)
    if not args.skip_seat:
        price.start_seat()
    # three tries: a freshly settling seat restarts its picture now and
    # then, and one restart during the check refused whole playtests
    # (2026-10-08/09); a seat that never settles still refuses
    for _try in range(3):
        fr = ci.do_freeze()
        if fr.get("deepening"):
            break
        time.sleep(10)
    if not fr.get("deepening"):
        print("REFUSED: time did not freeze: %r" % (fr,))
        return 2
    dials = dict(ci.CANONICAL_DIALS) if not args.play else {
        k: ci.CANONICAL_DIALS[k] for k in ("claude_input_lock", "claude_show_hud",
                                           "claude_show_chat", "claude_stats",
                                           "recent_chat_messages", "node_highlighting",
                                           # a debug view set by an earlier capture (the
                                           # playtest's face-ID pass) must not leak into
                                           # the next scenario: it did, 2026-10-08
                                           "claude_view")
        if k in ci.CANONICAL_DIALS}
    for kv in args.dial:
        k, _, v = kv.partition("=")
        dials[k.strip()] = float(v)
    dials["claude_dump_scale"] = args.scale
    # dials persist on the seat: a reference run's hold would turn the
    # next real-time arm into a half-reference (seen 2026-10-06)
    dials.setdefault("claude_path_hold", 0)
    token = "%s_%s" % (args.name, time.strftime("%Y%m%d-%H%M%S"))
    ci.push_dials(dials, "motion_%d" % time.time_ns())
    out = os.path.join(DUMP_ROOT, token)
    with open(lab.PATCH, "w") as f:
        f.write("claude_path = %s\n" % path)
        if args.mover:
            f.write("claude_mover = %s\n" % os.path.abspath(args.mover))
        if not args.nodump:
            f.write("claude_dump = %d:%s\n" % (n, token))
    # the client's own per-second stats, for the no-dump arm and for both
    stats = []
    t0 = time.time()
    meta = os.path.join(out, "meta.jsonl")
    last_mt = 0
    hold = dials.get("claude_path_hold", 0)
    # reference mode holds each pose for `hold` frames (~100 fps)
    budget = 30 + n / 20.0 + n * hold / 60.0
    while time.time() - t0 < budget:
        time.sleep(0.25)
        mt = os.path.getmtime(lab.STATS)
        if mt != last_mt:
            last_mt = mt
            st = lab.read_stats() or {}
            stats.append(st.get("frame_ms_avg", 0))
        if args.nodump:
            if time.time() - t0 > n / 60.0 + 3:
                break
        elif os.path.exists(meta) and sum(1 for _ in open(meta)) >= n:
            break
    with open(lab.PATCH, "w") as f:
        f.write("claude_path = 0\n")
        if args.mover:
            f.write("claude_mover = 0\n")
    rep = {"token": token, "path": path, "frames": n, "dials": dials,
           "stats_frame_ms": stats}
    if not args.nodump:
        rows = [json.loads(l) for l in open(meta)]
        rows.sort(key=lambda r: r["i"])
        pf = [r["path_frame"] for r in rows]
        # THE PATH MUST BE THE PATH: every dumped frame on it, in order
        if min(pf) < 0 or any(b != a + 1 for a, b in zip(pf, pf[1:])):
            bad = next(i for i, (a, b) in enumerate(zip(pf, pf[1:])) if b != a + 1 or a < 0)
            print("REFUSED: path frames not consecutive from dump frame %d (%r)"
                  % (bad, pf[max(0, bad - 2):bad + 3]))
            rep["refused"] = True
        dts = [r["dt_us"] / 1000.0 for r in rows[1:] if r["dt_us"] > 0]
        rep["dumped"] = len(rows)
        rep["frame_ms"] = {"mean": statistics.mean(dts), "p50": pctl(dts, 0.5),
                           "p99": pctl(dts, 0.99), "max": max(dts)}
        json.dump(rep, open(os.path.join(out, "run.json"), "w"), indent=1)
        print("dumped %d/%d frames  frame ms mean %.2f p50 %.2f p99 %.2f max %.1f"
              % (len(rows), n, rep["frame_ms"]["mean"], rep["frame_ms"]["p50"],
                 rep["frame_ms"]["p99"], rep["frame_ms"]["max"]))
    if stats:
        print("client 1 Hz frame_ms_avg: median %.2f over %d s"
              % (statistics.median(stats[1:] or stats), len(stats)))
    print(out)
    return 0


if __name__ == "__main__":
    # the GPU lock for this tool's whole life (children inherit it): see util/claude_gpu_lock.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import claude_gpu_lock
    claude_gpu_lock.hold('util/claude_motion.py')
    sys.exit(main())
