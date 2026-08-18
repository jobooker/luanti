#!/usr/bin/env python3
"""How many MATERIALS does this game actually need? Count before building.

THE QUESTION, plainly. Every cell of the traced world carries one byte
saying what it is. Since 2026-08-18 that byte is an INDEX into a material
table, which caps the world at 256 distinct materials. Before committing
to that shape, the handoff required the count to be measured rather than
assumed: enumerate the distinct NON-COLOUR property tuples over every
node Mineclonia registers, and if it overflows 256, fall back to 7 bits
of index plus an explicit fine-shape bit. Overflow is a legitimate
outcome, not a failure.

MEASURED 2026-08-18 (M4 Air, Mineclonia, 3,194 registered nodes):
    distinct (kind, light_source, fine)                  31
    distinct (kind, light_source, fine, texture_alpha)   46
    distinct (drawtype, light, node_box, liquid, alpha)  66   <- upper bound
The palette holds 151 slots for the whole key space and 105 are spare.

WHY THE THREE COUNTS. The first is the vocabulary the renderer actually
uses today. The second adds the one property most likely to become a
material distinction next (transparency). The third is the pessimistic
bound: it gives a separate material to every combination of CPU-side
inputs that game.cpp's drawtype chain can tell apart, whether or not the
renderer currently cares. Even the pessimistic bound is a quarter of the
byte, which is what makes an INDEX the right answer rather than a
bitfield.

COLOUR IS NOT IN ANY OF THESE, and that is the split the design rests
on: colour varies per CELL (param2 tinting, per-node texture averages)
and stays in the grid's RGB, while the non-colour profile is a table
lookup. That split is Minecraft RTX's, not an invention.

HOW IT MEASURES. Luanti's node registry lives in the game's Lua, so the
honest way to enumerate it is to ask a running server. This script drops
a one-shot mod into the world, starts luantiserver on a spare port with
no client at all (no GPU, no seat, ~20 s), reads the dump it writes, and
takes the mod back out. It contends with nothing.

Usage:  python3 util/claude_matcount.py [--port 30001] [--keep]
"""
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD = os.path.join(REPO, "worlds", "gallery")
MODDIR = os.path.join(REPO, "mods", "claude_matcount")
DUMP = os.path.join(WORLD, "claude_matcount.tsv")

MOD_LUA = '''-- Written by util/claude_matcount.py; removed again when it finishes.
core.register_on_mods_loaded(function()
\tlocal rows, n = {}, 0
\tfor name, d in pairs(core.registered_nodes) do
\t\tn = n + 1
\t\trows[#rows + 1] = table.concat({name,
\t\t\t\td.drawtype or "normal",
\t\t\t\td.light_source or 0,
\t\t\t\t(d.node_box and d.node_box.type) or "none",
\t\t\t\td.liquidtype or "none",
\t\t\t\td.paramtype2 or "none",
\t\t\t\t(d.walkable ~= false) and 1 or 0,
\t\t\t\td.use_texture_alpha or "none"}, "\\t")
\tend
\ttable.sort(rows)
\tlocal f = io.open(core.get_worldpath() .. "/claude_matcount.tsv", "w")
\tf:write("# nodes=" .. n .. "\\n")
\tf:write(table.concat(rows, "\\n") .. "\\n")
\tf:close()
\tcore.log("action", "[claude_matcount] wrote " .. n .. " nodes")
end)
'''

# game.cpp's drawtype chain, mirrored. Keep these in step with
# claudeTraceGridWalkBlock -- if they drift, this script is measuring a
# vocabulary the renderer does not have.
GLASS = {"glasslike", "glasslike_framed", "glasslike_framed_optional"}
LEAVES = {"allfaces", "allfaces_optional"}
# non-occluding decorations: game.cpp skips these entirely unless they emit
SKIPPED = {"plantlike", "plantlike_rooted", "firelike", "signlike", "raillike"}
FINE = {"mesh", "torchlike"}     # authored models; plus a non-regular nodebox


def install():
    os.makedirs(MODDIR, exist_ok=True)
    with open(os.path.join(MODDIR, "mod.conf"), "w") as f:
        f.write("name = claude_matcount\n"
                "description = one-shot node-registry dump (temporary)\n")
    with open(os.path.join(MODDIR, "init.lua"), "w") as f:
        f.write(MOD_LUA)
    wmt = os.path.join(WORLD, "world.mt")
    txt = open(wmt).read()
    if "load_mod_claude_matcount" not in txt:
        open(wmt, "a").write("load_mod_claude_matcount = true\n")


def remove():
    shutil.rmtree(MODDIR, ignore_errors=True)
    wmt = os.path.join(WORLD, "world.mt")
    txt = open(wmt).read()
    out = re.sub(r"^load_mod_claude_matcount\s*=.*\n", "", txt,
                 flags=re.M)
    if out != txt:
        open(wmt, "w").write(out)


def dump(port):
    if os.path.exists(DUMP):
        os.remove(DUMP)
    p = subprocess.Popen(
        [os.path.join(REPO, "bin", "luantiserver"), "--world", WORLD,
         "--port", str(port)],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(90):
            if os.path.exists(DUMP):
                time.sleep(0.5)
                return
            time.sleep(1.0)
        raise RuntimeError("the server never wrote %s" % DUMP)
    finally:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()


def kind(dt, lt):
    if lt != "none":
        return "liquid"
    if dt in LEAVES:
        return "leaves"
    if dt in GLASS:
        return "glass"
    return "solid"


def fine(dt, nb):
    if dt == "nodebox" and nb not in ("regular", "none"):
        return 1
    return 1 if dt in FINE else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--keep", action="store_true",
                    help="leave the dump and the mod in place")
    a = ap.parse_args()

    install()
    try:
        dump(a.port)
        rows = [l.rstrip("\n").split("\t")
                for l in open(DUMP) if not l.startswith("#")]
    finally:
        if not a.keep:
            remove()

    narrow, wide, bound = Counter(), Counter(), Counter()
    for name, dt, ls, nb, lt, par, walk, uta in rows:
        ls = min(int(ls), 14)
        if dt == "airlike":
            continue                       # not a cell
        if dt in SKIPPED and ls == 0:
            continue                       # game.cpp skips these
        narrow[(kind(dt, lt), ls, fine(dt, nb))] += 1
        wide[(kind(dt, lt), ls, fine(dt, nb), uta)] += 1
        bound[(dt, ls, nb, lt, uta)] += 1

    print("registered nodes: %d  (cells: %d)" % (len(rows), sum(narrow.values())))
    print("distinct (kind, light_source, fine)                    : %d"
          % len(narrow))
    print("distinct (kind, light_source, fine, use_texture_alpha) : %d"
          % len(wide))
    print("distinct (drawtype, light, node_box, liquid, alpha)    : %d"
          "   <- pessimistic upper bound" % len(bound))
    print()
    for k, v in sorted(narrow.items(), key=lambda kv: -kv[1]):
        print("   %-7s light=%-3d fine=%d   %5d nodes" % (k[0], k[1], k[2], v))
    print()
    over = max(len(narrow), len(wide), len(bound)) > 256
    print("VERDICT: %s" % (
        "OVERFLOW -- fall back to 7 bits of index plus an explicit fine bit"
        if over else
        "one byte of index is enough, with room to spare"))
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
