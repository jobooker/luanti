#!/usr/bin/env python3
"""Cornell-box diagnostics from the canonical golden (vantage `cornell`:
feet (47,9,1), yaw 0, pitch +12, fov 72, 1920x1080 — west/red wall on
the LEFT of frame, east/green on the RIGHT).

Reports, all in inverted linear radiance (ACES Narkowicz + gamma 2.2
undone, same transform as claude_furnace_check.py):

  bleed   mean linear RGB of a strip on each side of the frame + the
          red/green channel ratio between them. LEGACY AND BLIND — see
          the note on BLEED_STRIPS below; kept only so old runs stay
          comparable. Use --regions/--ratio for anything that judges.
  corner  luminance profile along the bottom-left -> corner diagonal,
          flagging stair-steps (face-quilt) vs continuous falloff.
  emitter brightest region + ceiling-away-from-emitter mean.

  --regions        the five named region means (REGIONS below), the
                   measurement that convicted NEE in roadmap step 1a.
  --ratio GOLD.png the same five as this/golden, one line each. This is
                   the estimator test: photo mode is the definition of
                   correct (physics-contract §6), so an estimator that
                   moves a region mean off 1.000 is a bug in the
                   estimator.

Usage: claude_cornell_check.py IMAGE [--regions] [--ratio GOLDEN.png]
"""
import argparse
import numpy as np
from PIL import Image


# ---------------------------------------------------------------- regions
# Pixel-FRACTION boxes (x0, y0, x1, y1), so they survive a resolution
# change. Every box lies inside ONE surface of the Cornell room and is
# clear of the wall-wall seams, the emitter, the crosshair, the held
# item, the (47,11,8) sky hole, and the chat/HUD bands.
#
# The room (util/claude_bridge_gallery.lua OPS.cornell @ (43,8,0), s=7):
# shell (43,8,0)-(51,16,8) gray221; red wall x=43, green wall x=51;
# 3x3 white_lit panel in the CEILING PLANE at y=16, x46..48, z3..5.
# From the vantage (eye ~y10.1, z=1, pitch +12, vfov 72) the panel is
# just above the top of frame, the ceiling/back-wall seam falls at
# y~150 px, the floor/back-wall seam at y~899 px, and the only floor
# NOT hidden behind the hotbar is the wedge in the bottom-left corner —
# which is why the floor box is small and low.
#
# CALIBRATED 2026-08-15 on screenshots/nee_sweep (both run orders,
# nee 0 and nee 1 at N~545, against golden_6f394f250/cornell.png).
# Verified by eye on an annotated frame, per physics-contract §8 clause
# 4, and by a purity test: 100% of the red_wall box is red-dominant
# pixels and 100% of the green_wall box green-dominant.
#
# What they read (m1/golden, order10 / order01 — deterministic to 4dp):
#   ceiling_flanks 1.0937 / 1.0948   back_wall 1.0232 / 1.0234
#   red_wall       1.0837 / 1.0838   green_wall 1.0385 / 1.0390
#   floor          1.0823 / 1.0831   ... and nee 0: 1.006-1.010 all five.
REGIONS = {
    # the ceiling BEYOND the panel (z 5.9..7.5), coplanar with it, so a
    # correct estimator lights it by bounce only. Two boxes, left and
    # right of frame centre, between the chat band (ends y<0.074) and
    # the ceiling/back-wall seam (y~0.139).
    "ceiling_flanks": [(0.33, 0.078, 0.44, 0.128),
                       (0.56, 0.078, 0.67, 0.128)],
    # back wall, above the (47,11,8) hole and the crosshair
    "back_wall": [(0.36, 0.20, 0.64, 0.45)],
    # the colored walls, full visible height between the seams. Both
    # carry a strong grazing gradient (the ratio runs ~1.17 at the near
    # frame edge to ~1.00 at the far corner), so these are SURFACE
    # means, not spot samples, and moving the box moves the number.
    "red_wall": [(0.02, 0.10, 0.24, 0.78)],
    "green_wall": [(0.74, 0.10, 0.98, 0.66)],
    # THE FLOOR, THREE BOXES, ~62,000 px. It was ONE box of 7,465 px in
    # the bottom-left wedge, and that box alone set how deep every CI shot
    # in the run had to be.
    #
    # Why it was small: the comment above says "the only floor a HUD-on
    # frame shows", and that was true when it was written. CI has pushed
    # claude_show_hud = 0 since 2026-08-15 (CANONICAL_DIALS -- the hotbar
    # is pixels inside a measured crop), so the hotbar has not covered
    # anything for a day and the floor wedge visible to a referee is
    # eight times larger than the box drawn over it.
    #
    # Why it mattered (spec/measured.md, "THE KNEE RULE HAS NO ANSWER"):
    # `floor` is the noisiest surface in the room per pixel -- dark, seen
    # at a grazing angle, lit almost entirely by indirect -- AND it had
    # the smallest box by 3-42x. The two multiply: 19x the run-to-run sd
    # of back_wall, 0.876 % at N=250 falling as 1/sqrt(N) with no
    # systematic part. Four of the five regions are settled by N=250;
    # this one is not settled at N=16,000, so it forced SETTLE_FRAMES =
    # 2000 on all thirteen arms.
    #
    # MEASURED before any seat time was spent, by re-reading the 46
    # cornell.png files already on disk in screenshots/ci: over the three
    # clean runs at 3dec2a935 the old box's sd is 0.2152 % and the new
    # one's 0.0147 % -- and over the six at b9e4382df, 0.1309 % against
    # 0.0374 %. A 3.5-14.6x reduction, more than the sqrt(8.3) = 2.9x the
    # area alone predicts, because three boxes on three parts of the
    # floor decorrelate what one box could not.
    #
    # WHAT THE NEW REGION CAN AND CANNOT SEE (physics-contract §8.3).
    # CAN: indirect light on the floor across the width of the room, both
    # occluders' contact shadows, and the near-to-far grazing gradient.
    # It still contains the old box's surface, so its answer is a
    # superset of the old one's, not a different question.
    # CANNOT: the emitter, the ceiling, either coloured wall (all outside
    # every box, and the purity test would fail if one crept in); the
    # occluder faces themselves; and -- unchanged from before -- anything
    # about specular response, since the renderer has none.
    #
    # The boxes are clear of every seam by >= 14 px at 1920x1080 and were
    # checked by eye on an annotated frame, per §8 clause 4.
    "floor": [(0.458, 0.843, 0.599, 0.990),    # between the two occluders
              (0.182, 0.917, 0.232, 0.990),    # the old bottom-left wedge
              (0.747, 0.944, 0.838, 0.990)],   # right of the far occluder
}

# Legacy bleed strips. MEASURED 2026-08-15: at this vantage these two
# boxes are NOT floor — the left one is the red wall and the right one
# the green wall, so "R/G left > R/G right" is true whenever the walls
# are colored at all, with or without transport. The check is therefore
# BLIND to the thing it is named for (physics-contract §8 clause 3);
# claude_ci calibrate demonstrates it against a planted defect. Left in
# place, unchanged, because old run.json files parse it.
BLEED_STRIPS = {"y": (0.58, 0.72), "left": (0.06, 0.22),
                "right": (0.78, 0.94)}


def aces_inverse(y):
    y = np.clip(y, 0.0, 0.9999)
    a = 2.51 - 2.43 * y
    b = 0.03 - 0.59 * y
    c = -0.14 * y
    return (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def lin(img):
    return aces_inverse((img / 255.0) ** 2.2)


LUMA = np.array([0.2126, 0.7152, 0.0722])


def load(path):
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return im, lin(im)


def region_stats(path):
    """{name: (lum_mean, rgb_mean, purity)} for the five REGIONS.

    purity is the fraction of pixels that still look like the named
    surface — red-dominant in red_wall, green-dominant in green_wall,
    near-neutral elsewhere. It is the box's own alarm: a HUD element, a
    chat line or a seam creeping into a box drops it below 1.0 long
    before it moves the mean enough to notice.
    """
    im, L = load(path)
    h, w = im.shape[:2]
    lum = L @ LUMA
    out = {}
    for name, boxes in REGIONS.items():
        pix, raw = [], []
        for (x0, y0, x1, y1) in boxes:
            sl = (slice(int(h * y0), int(h * y1)), slice(int(w * x0), int(w * x1)))
            pix.append(L[sl].reshape(-1, 3))
            raw.append(im[sl].reshape(-1, 3))
        pix = np.concatenate(pix)
        raw = np.concatenate(raw)
        r, g, b = raw[:, 0], raw[:, 1], raw[:, 2]
        if name == "red_wall":
            pure = float((r > 3.0 * np.maximum(g, 1.0)).mean())
        elif name == "green_wall":
            pure = float((g > 3.0 * np.maximum(r, 1.0)).mean())
        else:
            pure = float((np.abs(r - g) < 0.6 * np.maximum(r, 1.0)).mean())
        out[name] = (float((pix @ LUMA).mean()), pix.mean(axis=0), pure)
    return out


def print_regions(path, tag="region"):
    st = region_stats(path)
    for name in REGIONS:
        m, rgb, pure = st[name]
        print("%s %-15s mean %.5f  rgb %s  purity %.3f"
              % (tag, name, m, np.round(rgb, 4), pure))
    return st


def print_ratios(path, golden):
    this = region_stats(path)
    gold = region_stats(golden)
    print("ratio golden: %s" % golden)
    worst = 0.0
    for name in REGIONS:
        r = this[name][0] / max(gold[name][0], 1e-12)
        worst = max(worst, abs(r - 1.0))
        print("ratio %-15s %.4f  (this %.5f / golden %.5f)  purity %.3f"
              % (name, r, this[name][0], gold[name][0], this[name][2]))
    print("ratio worst deviation: %.4f" % worst)
    return this, gold


def legacy(path):
    im, L = load(path)
    h, w = im.shape[:2]
    y0, y1 = int(h * BLEED_STRIPS["y"][0]), int(h * BLEED_STRIPS["y"][1])
    lx = [int(w * v) for v in BLEED_STRIPS["left"]]
    rx = [int(w * v) for v in BLEED_STRIPS["right"]]
    left = L[y0:y1, lx[0]:lx[1]].mean(axis=(0, 1))
    right = L[y0:y1, rx[0]:rx[1]].mean(axis=(0, 1))
    print("floor strip LEFT (red side):   RGB %s" % np.round(left, 4))
    print("floor strip RIGHT (green side): RGB %s" % np.round(right, 4))
    rg_l = left[0] / max(left[1], 1e-6)
    rg_r = right[0] / max(right[1], 1e-6)
    print("R/G ratio: left %.3f vs right %.3f  -> %s" % (rg_l, rg_r,
          "COLOR BLEED PRESENT" if rg_l > rg_r * 1.15 else
          "NO CLEAR BLEED (ratios within 15%)"))

    lum = L @ LUMA
    row = lum[int(h * 0.60), int(w * 0.02):int(w * 0.30)]
    d = np.diff(row)
    big = np.abs(d) > (row.mean() * 0.08)
    print("corner profile: %d samples, %d large jumps (%s)"
          % (len(row), int(big.sum()),
             "quilt suspect" if big.sum() > 6 else "continuous"))

    thresh = np.percentile(lum, 99)
    print("brightest 1%% mean linear luminance: %.3f" % lum[lum >= thresh].mean())
    ceil = np.concatenate([
        lum[:int(h * 0.12), :int(w * 0.25)].ravel(),
        lum[:int(h * 0.12), int(w * 0.75):].ravel()])
    print("ceiling flanks (indirect-only) mean: %.4f" % ceil.mean())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--regions", action="store_true",
                    help="print the five named region means")
    ap.add_argument("--ratio", metavar="GOLDEN",
                    help="print the five regions as this/golden")
    ap.add_argument("--no-legacy", action="store_true",
                    help="skip the legacy bleed/corner/emitter block")
    args = ap.parse_args()
    if not args.no_legacy:
        legacy(args.image)
    if args.regions or not args.ratio:
        print_regions(args.image)
    if args.ratio:
        print_ratios(args.image, args.ratio)


if __name__ == "__main__":
    main()
