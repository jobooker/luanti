#!/usr/bin/env python3
"""claude_still_world — does claude_abm actually stop the world moving,
and does a world edit reach the tracer?

Two questions, one seat session, because both are about the same thing:
whether a change to the map arrives where it is supposed to and nowhere
else.

  still   The gate from spec/handoffs/2026-08-16-still-the-world-abm.md.
          Scan a box inside the cosy bubble, wait, scan it again, diff
          the node names. Once with ABMs LIVE (the conviction, re-run
          rather than assumed) and once with claude_abm = 0. The claim
          under test is "zero differences with ABMs off", and it is worth
          nothing unless the ABM-live window actually fires -- so this
          reports BOTH windows and says plainly when the live one was
          quiet, instead of counting a quiet control as a pass.

  edits   The roadmap RED "world edits do not reach the tracer while John
          drives", both halves:
            A. tracer ON, dig one node, look for an
               `[claude_grid] incremental` line within a second. This is
               the half nobody had tested.
            B. tracer OFF (claude_grid_debug = 0, i.e. after pressing O),
               dig one node, tracer back ON. Before the fix the dirty
               list was drained and discarded and the grid kept its old
               `valid` flag, so the edit was lost until the next
               re-centre. After the fix the drain invalidates and the
               next consumer tick does a full walk.
          Judged on grid_hash and grid_snap_seq -- COUNTERS and content
          hashes, never a sampled threshold (environment-laws), plus the
          client's own log lines.

Both write JSON into screenshots/still/<runid>/ and print their tables.

  python3 util/claude_still_world.py still --window 300 --skip-build
  python3 util/claude_still_world.py edits --skip-build
"""
import argparse
import json
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab     # noqa: E402
import claude_ci as ci       # noqa: E402

OUT_ROOT = os.path.join(REPO, "screenshots", "still")

# The scan box. OPS.scan refuses more than 20,000 cells, so this is a
# 36x7x36 slab (9,072 cells) over the cosy cabin AND the ground around
# it: the grass ABM works on `dirt_with_grass` with something above it,
# which is ground, not cabin, so a box covering only the room's interior
# is blind to the thing it is looking for. The first attempt was
# 26x8x26 from (-8,5,-8) and it missed (21,6,7) -- the exact cell the
# original conviction named -- by three nodes in x.
SCAN_BOX = ((-10, 6, -10), (25, 12, 25))
SCAN_ARM = "cozy-ci"        # the vantage to park at; ABMs run in ACTIVE
                            # blocks only, i.e. near a player


def scan():
    p1, p2 = SCAN_BOX
    return lab.rpc("scan",
                   p1={"x": p1[0], "y": p1[1], "z": p1[2]},
                   p2={"x": p2[0], "y": p2[1], "z": p2[2]})


def diff_names(a, b):
    """[(index, was, now)] for every cell whose node name changed."""
    out = []
    for i, (x, y) in enumerate(zip(a["names"], b["names"])):
        if x != y:
            out.append((i, x, y))
    return out


def cell_of(scan_result, index):
    p1, p2 = scan_result["p1"], scan_result["p2"]
    ny = p2["y"] - p1["y"] + 1
    nz = p2["z"] - p1["z"] + 1
    x = index // (ny * nz)
    y = (index // nz) % ny
    z = index % nz
    return {"x": p1["x"] + x, "y": p1["y"] + y, "z": p1["z"] + z}


def abm(on=None):
    return lab.rpc("abm", **({} if on is None else {"on": on}))


def log_tail(n=4000):
    try:
        with open(lab.DEBUG, errors="replace") as f:
            return f.read()[-n:]
    except Exception:
        return ""


def log_mark():
    """Byte offset in debug.txt, so a later read can quote only what
    happened AFTER the thing being tested. Reading the whole file and
    grepping would find last week's line and call it evidence."""
    try:
        return os.path.getsize(lab.DEBUG)
    except Exception:
        return 0


def log_since(mark, needles):
    try:
        with open(lab.DEBUG, errors="replace") as f:
            f.seek(mark)
            body = f.read()
    except Exception:
        return []
    return [l.strip() for l in body.splitlines()
            if any(n in l for n in needles)]


def bring_up(args, out, rec):
    err = ci.bring_up_seat(out, rec, types.SimpleNamespace(
        skip_build=args.skip_build, allow_debug=False,
        skip_deploy=args.skip_deploy))
    if err:
        return err
    if (rec.get("build_type") or "").lower() not in ci.RELEASE_TYPES:
        return "not a Release build tree: %s" % rec.get("build_type")
    if rec["shader_failures"]:
        return "%d shader compile failures" % len(rec["shader_failures"])
    return None


def new_run(tag):
    git = ci.git_state()
    rid = "%s_%s_%s" % (time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
                        git["tag"], tag)
    out = os.path.join(OUT_ROOT, rid)
    os.makedirs(out, exist_ok=True)
    print("out: %s" % out)
    return out, dict(git, run_id=rid, kind=tag,
                     started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime()))


def write_out(out, rec):
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = os.path.join(out, rec["kind"] + ".json")
    json.dump(rec, open(path, "w"), indent=2)
    print(path)
    return path


# ---------------------------------------------------------------- still

def one_window(label, window, log):
    """Scan, wait, scan -- and COUNT THE FOLDS, which is the real answer.

    The scan diff names the CAUSE (which node changed) and the fold count
    names the COST. They are different questions and the second is the
    one this step is being bought for: every `[claude_grid] incremental`
    line with a changed block is the client re-walking part of the grid,
    which clamps still_frames and restarts a settle. The golden run's own
    client.log carries four of them inside the cosy arms alone.

    A scan diff can also come back empty for an honest reason -- an ABM
    that has already converted every node it can convert has nothing left
    to do -- and a quiet control window proves nothing about the dial. So
    both numbers are reported and neither is allowed to stand in for the
    other.
    """
    eff = abm()
    log("  %s: claude_abm = %s (asked the server, not the conf)"
        % (label, eff.get("claude_abm")))
    mark = log_mark()
    st0 = lab.read_stats() or {}
    a = scan()
    t0 = time.time()
    log("  %s: scan 1 hash %s (%d cells)" % (label, a["hash"], a["total"]))
    while time.time() - t0 < window:
        time.sleep(5.0)
    b = scan()
    st1 = lab.read_stats() or {}
    folds = log_since(mark, ["[claude_grid] incremental"])
    d = diff_names(a, b)
    log("  %s: scan 2 hash %s after %.0f s -> %d cell(s) changed, "
        "%d grid fold(s), accum_resets %s -> %s"
        % (label, b["hash"], time.time() - t0, len(d), len(folds),
           st0.get("accum_resets"), st1.get("accum_resets")))
    for i, was, now in d[:20]:
        log("    %s  %s -> %s" % (cell_of(a, i), was, now))
    for l in folds[:10]:
        log("    FOLD %s" % l)
    return {"claude_abm": eff.get("claude_abm"), "window_s": round(window, 1),
            "hash_before": a["hash"], "hash_after": b["hash"],
            "cells": a["total"], "grid_folds": len(folds),
            "fold_lines": folds,
            "accum_resets": [st0.get("accum_resets"), st1.get("accum_resets")],
            "still_frames": [st0.get("still_frames"), st1.get("still_frames")],
            "changed": [{"pos": cell_of(a, i), "was": w, "now": n}
                        for i, w, n in d]}


def cmd_still(args):
    out, rec = new_run("still")
    err = bring_up(args, out, rec)
    if err:
        rec["aborted"] = err
        write_out(out, rec)
        print("ABORTED: %s" % err)
        return 1
    log = print
    vs = lab.load_vantages()
    # The tracer must be RUNNING for the fold count to mean anything: the
    # incremental path only executes when a grid consumer is on.
    with open(lab.PATCH, "w") as f:
        f.write("claude_grid_debug = 3\nclaude_grid_follow = 1\n")
    time.sleep(2.0)
    lab.goto(vs[SCAN_ARM])
    time.sleep(2.0)
    rec["parked_at"] = SCAN_ARM
    rec["scan_box"] = SCAN_BOX
    try:
        abm(True)
        rec["live"] = one_window("ABMs LIVE", args.window, log)
        abm(False)
        rec["stilled"] = one_window("claude_abm = 0", args.window, log)
    finally:
        r = abm(True)
        log("restored: claude_abm = %s" % r.get("claude_abm"))
        rec["restored"] = r.get("claude_abm")
    rec["abm_log"] = log_since(0, ["[claude_abm]"])[-6:]
    live, still = rec["live"], rec["stilled"]
    quiet_control = not live["changed"] and not live["grid_folds"]
    rec["verdict"] = (
        "FAIL: the world changed with claude_abm = 0"
        if (still["changed"] or still["grid_folds"])
        else "PASS, but the ABM-LIVE CONTROL WINDOW WAS QUIET TOO (0 node "
             "changes, 0 grid folds in %.0f s), so this run does not show "
             "the dial doing anything -- it shows this world sitting still "
             "on its own" % live["window_s"]
        if quiet_control
        else "PASS: %d node change(s) and %d grid fold(s) with ABMs live, "
             "zero of both with claude_abm = 0"
             % (len(live["changed"]), live["grid_folds"]))
    log("\nVERDICT: %s" % rec["verdict"])
    write_out(out, rec)
    return 0


# ---------------------------------------------------------------- edits

# A wall node of the cosy cabin: solid, inside the trace bubble, and
# replaced wholesale by the next gallery deploy, so digging it costs
# nothing permanent. Read off the room scan rather than assumed -- the
# op reports what it actually dug.
DIG_POS = {"x": 0, "y": 10, "z": 4}


def grid_state():
    st = lab.read_stats() or {}
    return {"grid_hash": st.get("grid_hash"), "grid_valid": st.get("grid_valid"),
            "grid_snap_seq": st.get("grid_snap_seq"),
            "grid_solid": st.get("grid_solid")}


def set_dial(**kv):
    """Push dials and wait out the client's ~1 Hz patch poll."""
    with open(lab.PATCH, "w") as f:
        for k, v in kv.items():
            f.write("%s = %s\n" % (k, v))
    time.sleep(2.0)


def cmd_edits(args):
    out, rec = new_run("edits")
    err = bring_up(args, out, rec)
    if err:
        rec["aborted"] = err
        write_out(out, rec)
        print("ABORTED: %s" % err)
        return 1
    log = print
    vs = lab.load_vantages()
    lab.goto(vs["cozy-ci"])
    time.sleep(2.0)
    abm(False)   # one variable per experiment: no ABM edits in the middle
    rec["claude_abm"] = abm().get("claude_abm")
    rec["dig_pos"] = DIG_POS
    try:
        for case in ("tracer_on", "tracer_off_then_on"):
            log("\n=== %s ===" % case)
            # Put the room back and rebuild the grid from scratch, so each
            # case starts from the same place.
            ci.deploy_gallery()
            set_dial(claude_grid_debug=3, claude_grid_follow=1,
                     claude_grid_snapshot="still_%d" % time.time_ns())
            time.sleep(3.0)
            before = grid_state()
            log("  before: %s" % before)
            if case == "tracer_off_then_on":
                set_dial(claude_grid_debug=0)
                time.sleep(2.0)
            mark = log_mark()
            dug = lab.rpc("dig", **DIG_POS)
            log("  dug: %s" % dug)
            time.sleep(3.0)
            mid = grid_state()
            mid_lines = log_since(mark, ["[claude_grid]"])
            log("  after dig (%d s): %s" % (3, mid))
            for l in mid_lines:
                log("    LOG %s" % l)
            back_lines = []
            if case == "tracer_off_then_on":
                mark2 = log_mark()
                set_dial(claude_grid_debug=3)
                time.sleep(4.0)
                back_lines = log_since(mark2, ["[claude_grid]"])
                for l in back_lines:
                    log("    LOG(back) %s" % l)
            after = grid_state()
            log("  after: %s" % after)
            rec[case] = {"before": before, "after_dig": mid, "after": after,
                         "dug": dug, "log_after_dig": mid_lines,
                         "log_after_consumer_back": back_lines,
                         "hash_moved": before["grid_hash"] != after["grid_hash"],
                         "seq_moved": (after["grid_snap_seq"] or 0)
                                      > (before["grid_snap_seq"] or 0)}
    finally:
        set_dial(claude_grid_debug=3, claude_grid_follow=1)
        ci.deploy_gallery()
        abm(True)
        log("restored: room redeployed, claude_abm = %s"
            % abm().get("claude_abm"))
    write_out(out, rec)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("still", cmd_still), ("edits", cmd_edits)):
        p = sub.add_parser(name)
        p.add_argument("--skip-build", action="store_true")
        p.add_argument("--skip-deploy", action="store_true")
        if name == "still":
            p.add_argument("--window", type=float, default=300.0,
                           help="seconds between the two scans (default "
                                "%(default)s; the grass ABM fires every "
                                "30-180 s, so a short window can be quiet "
                                "for honest reasons)")
        p.set_defaults(func=fn)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
