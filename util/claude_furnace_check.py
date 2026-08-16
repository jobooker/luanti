#!/usr/bin/env python3
"""Furnace-room analytic check (interior-program-plan.md phase 2).

The furnace room is sealed, every interior surface the same albedo rho
and uniformly emissive with radiance Le. Correct transport must show
L = Le / (1 - rho) from anywhere inside. This script reads a converged
photo-mode screenshot from inside the shut room, inverts the display
transform (claude_present: ACES Narkowicz fit, then gamma 1/2.2 — accum
is linear radiance), and compares per channel.

Engine-side constants (claude_accum/opengl_fragment.glsl @ a9c07ba52,
game.cpp claudeTraceGridSnapshot):
  albedo      rho = pow(c/255, 2.2) of the stored cell color
  warm force  emissive cells store r=255, g=max(g,200), b=max(b,120)
  emissive class a = (170 + 5*light_source)/255; e=(a-0.65)/0.29 clamped
  Le (as seen by light transport / photoMarch) = rho_vec * (0.4 + 2*e)
  eye-hit glow (primary ray, no transport)     = rho_vec * (0.5 + 5*e)
The last two DIFFER (2.4x vs 5.5x at e=1) — reported alongside, since
whichever one the measurement matches names the defect.

Usage: claude_furnace_check.py IMAGE {050|073} [--patch X0 Y0 X1 Y1]
Default patch: a 200x200 block left-of-center (avoids crosshair + hand).
"""
import sys
import numpy as np
from PIL import Image

TEX = {"050": 186.0, "073": 221.0}


def aces_inverse(y):
    """Invert y = x(2.51x+.03)/(x(2.43x+.59)+.14), the Narkowicz fit."""
    y = np.clip(y, 0.0, 0.9999)
    a = 2.51 - 2.43 * y
    b = 0.03 - 0.59 * y
    c = -0.14 * y
    return (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def main():
    img_path, variant = sys.argv[1], sys.argv[2]
    tex = TEX[variant]
    im = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float64)
    h, w = im.shape[:2]
    if "--patch" in sys.argv:
        i = sys.argv.index("--patch")
        x0, y0, x1, y1 = map(int, sys.argv[i + 1:i + 5])
    else:
        x0, y0 = w // 4 - 100, h // 2 - 100
        x1, y1 = w // 4 + 100, h // 2 + 100
    patch = im[y0:y1, x0:x1] / 255.0

    # measured: screenshot -> linear radiance
    meas = aces_inverse(patch ** 2.2)
    mean = meas.mean(axis=(0, 1))
    std = meas.std(axis=(0, 1))
    clipped = float((patch >= 254.0 / 255.0).mean())

    # authored color survives the snapshot since ADR-0009 #3 (40720f6+)
    stored = np.array([tex, tex, tex])
    rho = (stored / 255.0) ** 2.2
    e = 1.0  # light_source 14 -> class 240 -> e clamps to 1
    le = rho * (0.4 + 2.0 * e)          # ADR-0009 emitStrength(e)
    eyehit = rho * (0.5 + 5.0 * e)      # legacy eye-hit constant
    analytic = np.where(rho < 0.999,
                        le / (1.0 - rho),
                        le * sum(rho ** k for k in range(4)))
    # photoPath caps at 4 bounces (eye hit + 4): the honest expectation
    # for a correct-but-truncated integrator
    trunc5 = le * sum(rho ** k for k in range(5))

    print("furnace %s  patch(%d,%d)-(%d,%d)  clipped %.1f%%"
          % (variant, x0, y0, x1, y1, clipped * 100))
    print("stored color: %s   rho: %s" % (stored.astype(int), np.round(rho, 3)))
    for i, ch in enumerate("RGB"):
        note = "  [rho=1: 4-bounce truncated sum]" if rho[i] > 0.999 else ""
        print("%s  measured %.3f +/- %.3f | analytic Le/(1-rho) %.3f "
              "(ratio %.3f) | 5-event trunc %.3f (ratio %.3f) | "
              "eye-hit-only %.3f (ratio %.3f)%s"
              % (ch, mean[i], std[i], analytic[i], mean[i] / analytic[i],
                 trunc5[i], mean[i] / trunc5[i],
                 eyehit[i], mean[i] / eyehit[i], note))
    print("Le (transport) per channel: %s" % np.round(le, 3))


if __name__ == "__main__":
    main()
