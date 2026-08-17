#!/usr/bin/env python3
"""PROBE 1 of the cost investigation: IS IT DIVERGENCE?

THE QUESTION. Two experiments have now killed the two leading
explanations of what the descend costs. `lean-descend` (2026-08-17)
deleted the second walk's entire live state and the present-but-off tax
did not move -- so it is not those registers. `subbrick` (2026-08-17)
cut 1/16 m steps per pixel by 56 % and the trace pass did not move
(33.70 -> 33.85 ms) -- so the running cost is NOT proportional to the
steps taken. Something else pays the bill.

THE HYPOTHESIS THIS SCRIPT TESTS. A GPU runs pixels in rows of 32 in
LOCKSTEP: every lane of the row executes the same instruction, and a
lane that has finished its loop waits for the lane that has not. So the
time a row takes is set by its WORST member, not its average one. If
that is what is happening, halving the MEAN fine-step count buys nothing
while the per-tile MAXIMUM barely moves -- because every tile still
contains at least one grazing ray that walks a long way.

WHAT IT COMPUTES. Over the per-pixel counter PNGs the `counters` shader
variant already wrote (util/claude_cost_steps.py, claude_view 12-16),
for the two-rung walk (one-tracer) and the three-rung walk (subbrick):

  * per-PIXEL mean of the count -- the number the subbrick experiment
    moved by 56 %;
  * per-TILE MAX of the count, and the mean / p50 / p90 / p99 of that
    tile-max across the frame -- the number the lockstep story says
    the hardware actually pays;
  * the fraction of tiles in which at least one pixel descends at all.

TILE SHAPE IS NOT KNOWN. Apple does not document how it packs pixels
into a SIMD group, so every plausible shape is reported rather than one
guessed at: 8x4, 4x8 and 16x2 (all 32 pixels), plus the two degenerate
bounds 32x1 and 1x32. If the answer is the same shape in all five, the
conclusion does not depend on the guess.

READ IT LIKE THIS. If the tile-max mean falls far less than the pixel
mean -- e.g. -10 % against -56 % -- then the hardware's bill barely
moved even though the work halved, and divergence SURVIVES as the
explanation for the running cost. If the tile-max mean fell as far as
the pixel mean did and the time still did not move, divergence is dead
too and the cost is somewhere nobody has looked.

NO SEAT TIME. This reads PNGs already on disk. It changes nothing, it
starts no client, and it cannot be run against a shader that is not the
one that wrote the images -- the expected per-pixel statistics of every
input are asserted below, against the numbers spec/measured.md published
for them, so a wrong file is a crash and not a quiet wrong answer.

THE MARKER CORNER. claude_present stamps a 12 px green square in the
bottom-left to prove the traced present path ran. claude_cost_steps.py
crops 16 px off the bottom and the left before describing the pixels --
which is right for a histogram and WRONG here, because cropping shifts
every tile boundary off the framebuffer's own grid. So this script keeps
the full frame, aligns tiles to (0,0), and DROPS any tile that overlaps
the marker.

Usage:  util/claude_cost_divergence.py [--out screenshots/cost/divergence.json]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SHOTS = os.path.join(REPO, "screenshots")

MARKER = 16          # px, bottom-left corner claude_present stamps

# Tile shapes as (width, height). All five hold 32 pixels: the three the
# probe was asked for, plus the two degenerate bounds, because "which
# shape does Apple use" has no public answer and a conclusion that holds
# for all five does not need one.
TILES = [(8, 4), (4, 8), (16, 2), (32, 1), (1, 32)]

# THE INPUTS, AND THEIR OWN ASSERTIONS.
#
# Every entry names the capture, the arm it belongs to, and the per-pixel
# statistics spec/measured.md already published for it. `check` is
# (mean, p90, p99, max) and is asserted before anything is computed. A
# capture whose distribution does not reproduce is a capture of a
# different shader, a different room or a different grid, and this probe
# would compare two things nobody measured.
#
# two-rung  = one-tracer's shipped walk (1 m and 1/16 m), captured
#             2026-08-17 00:23 for the lean-descend section. grid_solid
#             210,750.
# three-rung = branch `subbrick` (1 m, 1/4 m, 1/16 m), captured
#             2026-08-17 07:24 for the subbrick section. grid_solid
#             208,670. NOTE the two runs parked on different grid
#             origins; see "what this cannot hold still" below.
SETS = {
    "two-rung": {
        "grid_solid": 210750,
        "shots": {
            "primary_fine":   ("screenshot_20260817_002312.png",
                               (3.376, 8, 34, 52)),
            "primary_coarse": ("screenshot_20260817_002317.png",
                               (8.713, 12, 12, 13)),
            "path_fine":      ("screenshot_20260817_002323.png",
                               (21.740, 59, 113, 314)),
            "path_coarse":    ("screenshot_20260817_002328.png",
                               (29.010, 42, 53, 215)),
        },
    },
    "three-rung": {
        "grid_solid": 208670,
        "shots": {
            "primary_mid":    ("screenshot_20260817_072354.png",
                               (1.488, 3, 9, 14)),
            "path_mid":       ("screenshot_20260817_072359.png",
                               (7.711, 17, 30, 82)),
            "primary_fine":   ("screenshot_20260817_072405.png",
                               (1.278, 3, 8, 29)),
            "primary_coarse": ("screenshot_20260817_072410.png",
                               (8.713, 12, 12, 13)),
            "path_fine":      ("screenshot_20260817_072416.png",
                               (9.641, 19, 39, 163)),
            "path_coarse":    ("screenshot_20260817_072421.png",
                               (29.005, 42, 53, 211)),
        },
    },
}


def decode(png):
    """The counter plane, full frame, no crop. R/G are a 16-bit
    little-endian count; k/255 quantises back to k through an 8-bit
    framebuffer, so these are the exact integers the shader counted.
    B is the `descended` flag on view 12 and unused elsewhere."""
    a = np.asarray(Image.open(png).convert("RGB")).astype(np.int32)
    return a[:, :, 0] + 256 * a[:, :, 1], a[:, :, 2]


def marker_mask(shape):
    """True where a pixel is REAL, False over the present marker."""
    m = np.ones(shape, dtype=bool)
    m[shape[0] - MARKER:, :MARKER] = False
    return m


def tile_reduce(arr, valid, tw, th, fn):
    """Reduce `arr` over (th x tw) tiles aligned to (0,0), dropping any
    tile that overlaps an invalid pixel. Returns the per-tile values and
    how many tiles were dropped."""
    h, w = arr.shape
    ny, nx = h // th, w // tw
    a = arr[:ny * th, :nx * tw].reshape(ny, th, nx, tw)
    v = valid[:ny * th, :nx * tw].reshape(ny, th, nx, tw)
    keep = v.all(axis=(1, 3))
    out = fn(a, axis=(1, 3))
    return out[keep], int((~keep).sum())


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return {
        "mean": round(float(v.mean()), 3),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
        "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
        "n": int(v.size),
    }


def check_pixels(name, counts, expect):
    """Assert the capture reproduces the distribution spec/measured.md
    published for it -- ON THE SAME CROP THAT PUBLISHED IT.
    claude_cost_steps.py describes `counts[:-16, 16:]`, and that crop is
    not statistically the same frame: it removes the bottom 16 rows,
    which at the cosy vantage are the near floor and carry the longest
    walks (per-pixel mean 3.376 cropped vs 3.644 whole). So the check
    uses the crop and the TILING uses the whole frame, because a tile
    grid has to be aligned to the framebuffer's own (0,0) or it is not
    the hardware's grid."""
    f = counts[:-MARKER, MARKER:].astype(np.float64).ravel()
    got = (round(float(f.mean()), 3), float(np.percentile(f, 90)),
           float(np.percentile(f, 99)), float(f.max()))
    want = tuple(float(x) for x in expect)
    # mean to 0.001 (the published figure is rounded to three places),
    # the three order statistics exactly.
    if abs(got[0] - want[0]) > 0.0011 or got[1:] != want[1:]:
        raise SystemExit(
            "%s does not reproduce its published distribution.\n"
            "  got  mean %.3f p90 %g p99 %g max %g\n"
            "  want mean %.3f p90 %g p99 %g max %g\n"
            "This capture is not the one spec/measured.md describes; "
            "comparing it would measure something nobody took."
            % ((name,) + got + want))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "screenshots", "cost",
                                                  "divergence.json"))
    args = ap.parse_args()

    result = {"tiles": ["%dx%d" % t for t in TILES], "arms": {}}

    for arm, spec in SETS.items():
        entry = {"grid_solid": spec["grid_solid"], "shots": {}}
        planes = {}
        for view, (fname, expect) in spec["shots"].items():
            png = os.path.join(SHOTS, fname)
            if not os.path.exists(png):
                raise SystemExit("missing capture %s (%s / %s)"
                                 % (png, arm, view))
            counts, blue = decode(png)
            valid = marker_mask(counts.shape)
            check_pixels("%s/%s" % (arm, view), counts, expect)
            planes[view] = (counts, blue, valid)
            entry["shots"][view] = fname

        # The two work measures. `path_fine` is the 1/16 m steps alone --
        # the number subbrick cut by 56 %. `path_submetre` adds the 1/4 m
        # sub-brick visits, because a MIXED sub-brick still has to be
        # entered and a fetch is a fetch: it is the 20 % number.
        work = {"path_fine": planes["path_fine"][0],
                "primary_fine": planes["primary_fine"][0],
                # THE NULL CONTROL. The outer 1 m walk is unchanged
                # between the two arms to four significant figures
                # (29.010 -> 29.005 per pixel), so its tile-max mean
                # must also be unchanged. An instrument that reports a
                # large change on the arm that did change AND a large
                # change on the arm that did not is not measuring what
                # it claims. (Global CLAUDE.md: instruments can be
                # blind; a numeric comparison sanity-checks its inputs.)
                "path_coarse_CONTROL": planes["path_coarse"][0]}
        if "path_mid" in planes:
            work["path_submetre"] = (planes["path_fine"][0]
                                     + planes["path_mid"][0])
            work["primary_submetre"] = (planes["primary_fine"][0]
                                        + planes["primary_mid"][0])
        else:
            work["path_submetre"] = planes["path_fine"][0]
            work["primary_submetre"] = planes["primary_fine"][0]

        valid = marker_mask(planes["path_fine"][0].shape)
        entry["dropped_tiles_note"] = ("tiles overlapping the %d px present "
                                       "marker are dropped" % MARKER)
        entry["metrics"] = {}
        for wname, arr in work.items():
            # `pixel` is over the whole frame minus the marker -- the
            # same population the tiles are built from. `pixel_cropped`
            # is over spec/measured.md's crop, so the two records can be
            # tied together.
            m = {"pixel": stats(arr[valid]),
                 "pixel_cropped": stats(arr[:-MARKER, MARKER:])}
            for tw, th in TILES:
                vals, dropped = tile_reduce(arr, valid, tw, th, np.max)
                m["tilemax_%dx%d" % (tw, th)] = dict(stats(vals),
                                                     dropped=dropped)
                # the mean of tile MEANS is the pixel mean by
                # construction; the useful companion is how often a tile
                # has any work in it at all.
                any_, _ = tile_reduce((arr > 0).astype(np.int32), valid,
                                      tw, th, np.max)
                m["frac_tiles_touched_%dx%d" % (tw, th)] = round(
                    float(any_.mean()), 5)
            entry["metrics"][wname] = m

        # the primary ray's own descend flag, straight off view 12's blue
        # channel, as a cross-check on "does every tile contain a
        # descending pixel".
        pf, blue, _ = planes["primary_fine"]
        desc = (blue > 127).astype(np.int32)
        entry["primary_descend_frac_pixels"] = round(
            float(desc[valid].mean()), 5)
        for tw, th in TILES:
            any_, _ = tile_reduce(desc, valid, tw, th, np.max)
            entry["primary_descend_frac_tiles_%dx%d" % (tw, th)] = round(
                float(any_.mean()), 5)

        result["arms"][arm] = entry

    # --- the comparison, computed here so nobody has to divide by hand
    two, three = result["arms"]["two-rung"], result["arms"]["three-rung"]
    cmp_ = {}
    for wname in ("path_fine", "path_submetre", "path_coarse_CONTROL",
                  "primary_fine", "primary_submetre"):
        row = {"pixel_mean": [two["metrics"][wname]["pixel"]["mean"],
                              three["metrics"][wname]["pixel"]["mean"]]}
        a, b = row["pixel_mean"]
        row["pixel_mean_change_pct"] = round(100.0 * (b - a) / a, 1) if a else None
        for tw, th in TILES:
            k = "tilemax_%dx%d" % (tw, th)
            a = two["metrics"][wname][k]["mean"]
            b = three["metrics"][wname][k]["mean"]
            row[k + "_mean"] = [a, b]
            row[k + "_change_pct"] = round(100.0 * (b - a) / a, 1) if a else None
        # DIVERGENCE RATIO: how many times the average pixel's work the
        # slowest lane in its tile does. 1.0 would be a perfectly
        # coherent tile (every lane the same length of walk); a big
        # number is a tile whose cost, under lockstep, is set by one
        # lane. Reported for both arms so "did the walk become more or
        # less divergent" has a number rather than an impression.
        for tw, th in TILES:
            k = "tilemax_%dx%d" % (tw, th)
            pm = row["pixel_mean"]
            row[k + "_over_pixelmean"] = [
                round(row[k + "_mean"][i] / pm[i], 3) if pm[i] else None
                for i in (0, 1)]
        cmp_[wname] = row
    result["comparison"] = cmp_

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    # --- the table, printed
    w = sys.stdout.write
    w("\nPER-PIXEL MEAN vs PER-TILE MAX MEAN  (cozy-ci, whole path)\n")
    w("a tile is 32 pixels; the hardware runs 32 lanes in lockstep and\n"
      "waits for the slowest one.\n\n")
    # PRIMARY first: it is ONE march() call, so its tile max IS the
    # lockstep cost of that loop. The whole-path rows sum 24 bounces and
    # every shadow ray, and the max of a sum is only a LOWER bound on the
    # sum of the per-segment maxima -- true but weaker. Read the primary
    # rows as the clean test and the path rows as the corroboration.
    for wname in ("primary_fine", "primary_submetre",
                  "path_fine", "path_submetre", "path_coarse_CONTROL"):
        r = cmp_[wname]
        w("  %s\n" % wname)
        w("    %-14s %8s %8s %8s\n" % ("reduction", "two-rung", "three-rung",
                                       "change"))
        w("    %-14s %8.3f %8.3f %+7.1f %%\n"
          % ("per pixel", r["pixel_mean"][0], r["pixel_mean"][1],
             r["pixel_mean_change_pct"]))
        for tw, th in TILES:
            k = "tilemax_%dx%d" % (tw, th)
            w("    %-14s %8.3f %8.3f %+7.1f %%\n"
              % ("tile max %dx%d" % (tw, th), r[k + "_mean"][0],
                 r[k + "_mean"][1], r[k + "_change_pct"]))
        w("    %-14s %8.3f %8.3f\n"
          % ("divergence x", r["tilemax_8x4_over_pixelmean"][0],
             r["tilemax_8x4_over_pixelmean"][1]))
        w("\n")
    w("TILE-MAX DISTRIBUTION, 8x4, whole-path fine steps\n")
    for arm in ("two-rung", "three-rung"):
        s = result["arms"][arm]["metrics"]["path_fine"]["tilemax_8x4"]
        w("  %-11s mean %7.3f  p50 %5g  p90 %5g  p99 %5g  max %5g\n"
          % (arm, s["mean"], s["p50"], s["p90"], s["p99"], s["max"]))
    w("\nFRACTION OF 8x4 TILES CONTAINING AT LEAST ONE DESCENDING PIXEL\n")
    for arm in ("two-rung", "three-rung"):
        e = result["arms"][arm]
        w("  %-11s whole path %.5f   primary ray %.5f (pixels %.5f)\n"
          % (arm,
             e["metrics"]["path_fine"]["frac_tiles_touched_8x4"],
             e["primary_descend_frac_tiles_8x4"],
             e["primary_descend_frac_pixels"]))
    w("\nwrote %s\n" % args.out)


if __name__ == "__main__":
    main()
