#!/usr/bin/env python3
"""Cornell-box diagnostics from the canonical golden (vantage `cornell`:
feet (47,9,1), yaw 0, pitch +12, fov 72, 1920x1080 — west/red wall on
the LEFT of frame, east/green on the RIGHT).

Reports, all in inverted linear radiance (ACES Narkowicz + gamma 2.2
undone, same transform as claude_furnace_check.py):

  bleed   mean linear RGB of a floor strip adjacent to each colored
          wall + the red/green channel ratio between strips. Working
          one-bounce transport separates the ratios decisively.
  corner  luminance profile along the bottom-left -> corner diagonal,
          flagging stair-steps (face-quilt) vs continuous falloff.
  emitter brightest region + ceiling-away-from-emitter mean (that part
          faces away from the panel; its light is genuinely indirect).

Usage: claude_cornell_check.py IMAGE
"""
import sys
import numpy as np
from PIL import Image


def aces_inverse(y):
    y = np.clip(y, 0.0, 0.9999)
    a = 2.51 - 2.43 * y
    b = 0.03 - 0.59 * y
    c = -0.14 * y
    return (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def lin(img):
    return aces_inverse((img / 255.0) ** 2.2)


def main():
    im = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float64)
    h, w = im.shape[:2]
    L = lin(im)

    # floor strips flanking the frame bottom third (left = red side,
    # right = green side), clear of the HUD (bottom 22%) and hand
    y0, y1 = int(h * 0.58), int(h * 0.72)
    left = L[y0:y1, int(w * 0.06):int(w * 0.22)].mean(axis=(0, 1))
    right = L[y0:y1, int(w * 0.78):int(w * 0.94)].mean(axis=(0, 1))
    print("floor strip LEFT (red side):   RGB %s" % np.round(left, 4))
    print("floor strip RIGHT (green side): RGB %s" % np.round(right, 4))
    rg_l = left[0] / max(left[1], 1e-6)
    rg_r = right[0] / max(right[1], 1e-6)
    print("R/G ratio: left %.3f vs right %.3f  -> %s" % (rg_l, rg_r,
          "COLOR BLEED PRESENT" if rg_l > rg_r * 1.15 else
          "NO CLEAR BLEED (ratios within 15%)"))

    # corner gradient: luminance along a row into the bottom-left corner
    lum = L @ np.array([0.2126, 0.7152, 0.0722])
    row = lum[int(h * 0.60), int(w * 0.02):int(w * 0.30)]
    d = np.diff(row)
    # stair-step: successive-difference sign flips with large steps
    big = np.abs(d) > (row.mean() * 0.08)
    print("corner profile: %d samples, %d large jumps (%s)"
          % (len(row), int(big.sum()),
             "quilt suspect" if big.sum() > 6 else "continuous"))

    # emitter: brightest 1% vs ceiling flanks (top corners of frame)
    thresh = np.percentile(lum, 99)
    print("brightest 1%% mean linear luminance: %.3f" % lum[lum >= thresh].mean())
    ceil = np.concatenate([
        lum[:int(h * 0.12), :int(w * 0.25)].ravel(),
        lum[:int(h * 0.12), int(w * 0.75):].ravel()])
    print("ceiling flanks (indirect-only) mean: %.4f" % ceil.mean())


if __name__ == "__main__":
    main()
