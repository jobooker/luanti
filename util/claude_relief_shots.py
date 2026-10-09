#!/usr/bin/env python3
"""claude_relief_shots — relief off vs on, side by side (2026-10-08).

Holds the GPU lock for its whole life. Starts the game with the first
shot, finds a tree trunk near the forest vantage with the bridge's scan
(no world edits), then shoots every view with claude_relief 0 and 1
(interleaved, every arm spelling the dial), stitches pairs, and prices the
trace pass with claude_live_price.  Output: screenshots/relief/.
"""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import claude_gpu_lock
claude_gpu_lock.hold("util/claude_relief_shots.py")
import claude_lab as lab
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "screenshots", "relief")
os.makedirs(OUT, exist_ok=True)
FRAMES = int(os.environ.get("RELIEF_FRAMES", "1024"))
DEPTHS = [int(d) for d in os.environ.get("RELIEF_DEPTHS", "0,1").split(",")]
started = False
log = open(os.path.join(OUT, "run.log"), "a")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n"); log.flush()


def shoot(name, pos, yaw, pitch, t, depth, extra=None, label=None):
    """depth = claude_relief; extra = more dials (every arm spells every
    relief dial, so nothing leaks from the arm before)."""
    global started
    extra = dict({"claude_relief_flat": 0, "claude_relief_frac": 0.35}, **(extra or {}))
    label = label or "d%d" % depth
    cmd = [sys.executable, "util/claude_shoot.py", "--play", "--pin",
           "--pos", *map(str, pos), "--yaw", str(yaw), "--pitch", str(pitch),
           "--time", str(t), "--frames", str(FRAMES),
           "--name", "relief-%s-%s" % (name, label),
           "--dial", "claude_nee=1", "--dial", "claude_relief=%d" % depth]
    for k, v in extra.items():
        cmd += ["--dial", "%s=%s" % (k, v)]
    if started:
        cmd.append("--skip-seat")
    for attempt in range(3):
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or [""])[-1]
        if last.endswith(".png") and os.path.exists(last):
            started = True
            st = lab.read_stats() or {}
            say("%-14s %s %s relief_cells=%s shapes=%s still=%s" % (
                name, label, last, st.get("relief_cells"), st.get("relief_shapes"),
                st.get("still_frames")))
            dst = os.path.join(OUT, "%s-%s.png" % (name, label))
            Image.open(last).save(dst)
            return dst
        say("shot %s %s attempt %d failed: %s | %s" % (name, label, attempt,
            last, r.stderr.strip().splitlines()[-3:]))
        started = started or "REFUSED" in last
        time.sleep(20)
    raise SystemExit("two-miss: %s never shot" % name)


def stitch(name, paths, crop=None, labels=None):
    ims = [Image.open(p).convert("RGB") for p in paths]
    if crop:
        ims = [im.crop(crop) for im in ims]
    w, h = ims[0].size
    out = Image.new("RGB", (w * len(ims) + 8 * (len(ims) - 1), h + 40), "white")
    d = ImageDraw.Draw(out)
    labels = labels or ["claude_relief = %d (%s)" % (
        dep, "off, today" if dep == 0 else "grooves %d/16 m deep" % dep) for dep in DEPTHS]
    for k, (im, lab_) in enumerate(zip(ims, labels)):
        out.paste(im, (k * (w + 8), 40))
        d.text((k * (w + 8) + 10, 10), lab_, fill="black")
    p = os.path.join(OUT, "%s-compare.png" % name)
    out.save(p)
    say("stitched", p)


def find_trunk(c):
    """A log column (logs at y 9 and 10) with a clear 2.5 m approach on
    some side and ground under the camera; prefer the camera looking along
    z (the morning sun grazes the face from +x)."""
    x0, z0 = int(c[0]) - 14, int(c[2]) - 14
    r = lab.rpc("scan", p1=dict(x=x0, y=8, z=z0), p2=dict(x=x0 + 28, y=11, z=z0 + 28))
    names = r["names"]
    ny, nz = 4, 29
    def at(x, y, z):
        i = ((x - x0) * ny + (y - 8)) * nz + (z - z0)
        return names[i] if 0 <= x - x0 < 29 and 0 <= y - 8 < 4 and 0 <= z - z0 < 29 else "?"
    say("scan counts:", json.dumps({k: v for k, v in r["counts"].items() if v > 3}))
    best = None
    for x in range(x0, x0 + 29):
        for z in range(z0, z0 + 29):
            if "tree" not in at(x, 9, z) or "tree" not in at(x, 10, z):
                continue
            # camera 2 cells away along +-z first, then +-x: (dx, dz, yaw)
            for dx, dz, yaw in ((0, -3, 0), (0, 3, 180), (3, 0, 90), (-3, 0, 270)):
                cx, cz = x + dx, z + dz
                path = [(x + dx * k // 3, z + dz * k // 3) for k in (1, 2, 3)]
                clear = all(at(px, 9, pz) == "air" and at(px, 10, pz) == "air" for px, pz in path)
                ground = at(cx, 8, cz) not in ("air", "?") and "water" not in at(cx, 8, cz)
                if clear and ground:
                    d2 = (x - c[0]) ** 2 + (z - c[2]) ** 2 + (0 if dz else 1000)
                    if best is None or d2 < best[0]:
                        best = (d2, (cx + 0.5, 8.5, cz + 0.5), yaw, (x, z), at(x, 9, z))
    return best


FOREST = (146.5, 8.5, 123.5)
if os.environ.get("RELIEF_RUN") == "2":
    # what the grooves add beyond the texel colour (flat control), and the
    # two knobs: depth, and the share of the tile that sinks
    ARMS = [("today", 0, {}, "today: relief 0 (oak = baked model, one colour per cell)"),
            ("flat", 1, {"claude_relief_flat": 1}, "control: plain textured cube, no grooves"),
            ("d1", 1, {}, "relief 1: darkest 35% sink 1/16 m"),
            ("d2", 2, {}, "relief 2: darkest 35% sink up to 2/16 m"),
            ("d1f65", 1, {"claude_relief_frac": 0.65}, "relief 1, frac 0.65: brightest 35% stand proud")]
    VIEWS = [("trunk2", (151.5, 8.5, 132.5), 180, 8, 0.30, (760, 160, 1400, 1000)),
             ("ground2-side", FOREST, 0, -28, 0.30, (480, 540, 1440, 1080)),
             ("ground2-into-sun", FOREST, 270, -28, 0.30, (480, 540, 1440, 1080))]
    for name, pos, yaw, pitch, t, crop in VIEWS:
        ps = [shoot(name, pos, yaw, pitch, t, dep, ex, lab_) for lab_, dep, ex, _ in ARMS]
        stitch(name, ps, labels=[a[3] for a in ARMS])
        stitch(name + "-crop", ps, crop=crop, labels=[a[3] for a in ARMS])
    say("done")
    sys.exit(0)

pairs = {}
# 1. the wide forest view; this starts the game
for t, tag in ((0.30, "forest-morning"), (0.5, "forest-noon")):
    pairs[tag] = [shoot(tag, FOREST, 270, -5, t, d) for d in DEPTHS]
    stitch(tag, pairs[tag])
# 2. a trunk at ~2 m
tr = find_trunk(FOREST)
say("trunk:", tr)
if tr:
    _, cam, yaw, cell, nm = tr
    for t, tag in ((0.30, "trunk-morning"), (0.5, "trunk-noon")):
        pairs[tag] = [shoot(tag, cam, yaw, 8, t, d) for d in DEPTHS]
        stitch(tag, pairs[tag])
# 3. the ground at a grazing angle, morning sun from the side and from behind
for yaw, tag in ((0, "ground-side"), (270, "ground-into-sun")):
    pairs[tag] = [shoot(tag, FOREST, yaw, -28, 0.30, d) for d in DEPTHS]
    stitch(tag, pairs[tag])
    stitch(tag + "-crop", pairs[tag], crop=(480, 540, 1440, 1080))
# 4. price: trace pass, relief off / on, twice each, interleaved
arms = [["off", {"claude_relief": 0}], ["on", {"claude_relief": 1}],
        ["off2", {"claude_relief": 0}], ["on2", {"claude_relief": 1}]]
r = subprocess.run([sys.executable, "util/claude_live_price.py", json.dumps(arms), "forest"],
                   cwd=REPO, capture_output=True, text=True)
say("PRICE:\n" + r.stdout + r.stderr[-800:])
say("done")
