#!/usr/bin/env python3
"""Sky-furnace analytic check — the ONLY referee that can see a
constant-factor error in sky radiance.

WHAT THIS ROOM IS. A flat pad of albedo rho = 0.50 (claude_bridge:gray186),
24x24, laid on the open mgflat plain with no walls and no roof, ringed by
a 1-node fence 13 nodes out. It exists for exactly this measurement and
was pinned as a black golden on 2026-08-15 waiting for the sky term.

WHY IT IS NEEDED. Under a sky of CONSTANT radiance L_sky, an unoccluded
Lambertian plane reads

    E = pi * L_sky          (irradiance from the whole upper hemisphere)
    L = rho * E / pi = rho * L_sky

exactly -- and a flat plane sees no part of itself, so there is no
interreflection term to truncate. That makes the answer a NUMBER YOU
COMPUTE, not a golden you pin, which is the same family as the furnace's
L = Le/(1-rho).

Nothing else in this harness can catch a constant factor in sky radiance:
the sealed rooms (Cornell, both furnaces, cave-glass) see no sky at all,
so they pass whatever the sky says, and a uniformly hot outdoors just
looks like a bright day. Hence the claude_sky_uniform test dial, which
replaces the whole sky with one constant in every direction -- no sun, no
moon, no gradient, no ground term.

WHAT IT IS BLIND TO (physics-contract §8 clause 3, and the furnace's own
history is why this list is here):
  * ANY ERROR IN THE SKY'S SHAPE. It measures one constant. A wrong
    horizon/zenith gradient, a wrong sun size, a wrong moon, a wrong
    ground term -- all invisible, because the test dial turns every one
    of them off. That is the price of having an analytic answer.
  * GEOMETRY AND COLOUR, like every furnace. It is an energy referee.
  * OCCLUSION BY THE FENCE, which is real and is why the measured ratio
    sits slightly BELOW 1.0 rather than at it: the fence subtends about
    4.8 deg above the horizon from the pad centre, so it removes roughly
    sin^2(4.8 deg) ~ 0.7 % of the cosine-weighted hemisphere, and returns
    some of it as a bounce. The pin is the measured value, exactly the
    way furnace-050 is pinned at 0.982 rather than at 1.000.

Usage: claude_skyfurnace_check.py IMAGE {050} [--lsky L] [--patch x0 y0 x1 y1]
"""
import sys

import numpy as np
from PIL import Image

# stored cell colour of the pad, per room. rho = (c/255)^2.2, the ONE
# colour-byte law this renderer has (claude_trace cellAlbedo()).
TEX = {"050": 186.0}

# The measurement patch, in image pixels, PINNED rather than derived from
# the resolution: it must sit entirely on the pad and clear of the fence,
# the horizon and the crosshair. Chosen from a real frame at 1920x1080,
# camera (60,9,104) pitch -45 yaw 0.
DEFAULT_PATCH = (760, 700, 1160, 1000)


def aces_inverse(y):
    """Invert y = x(2.51x+.03)/(x(2.43x+.59)+.14), the Narkowicz fit that
    claude_present applies. Identical to claude_furnace_check's, because
    the two referees must invert the SAME transform -- a second copy of
    the display transform is a second thing to keep honest."""
    y = np.clip(y, 0.0, 0.9999)
    a = 2.51 - 2.43 * y
    b = 0.03 - 0.59 * y
    c = -0.14 * y
    return (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def main():
    img_path, variant = sys.argv[1], sys.argv[2]
    l_sky = 1.0
    if "--lsky" in sys.argv:
        l_sky = float(sys.argv[sys.argv.index("--lsky") + 1])
    im = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float64)
    h, w = im.shape[:2]
    if "--patch" in sys.argv:
        i = sys.argv.index("--patch")
        x0, y0, x1, y1 = map(int, sys.argv[i + 1:i + 5])
    else:
        x0, y0, x1, y1 = DEFAULT_PATCH
    patch = im[y0:y1, x0:x1] / 255.0

    meas = aces_inverse(patch ** 2.2)
    mean = meas.mean(axis=(0, 1))
    std = meas.std(axis=(0, 1))
    clipped = float((patch >= 254.0 / 255.0).mean())

    stored = np.array([TEX[variant]] * 3)
    rho = (stored / 255.0) ** 2.2
    analytic = rho * l_sky

    print("skyfurnace %s  patch(%d,%d)-(%d,%d)  clipped %.1f%%  L_sky %.4f"
          % (variant, x0, y0, x1, y1, clipped * 100, l_sky))
    print("stored color: %s   rho: %s" % (stored.astype(int), np.round(rho, 3)))
    for i, ch in enumerate("RGB"):
        print("%s  measured %.4f +/- %.4f | analytic rho*L_sky %.4f "
              "(ratio %.4f)"
              % (ch, mean[i], std[i], analytic[i], mean[i] / analytic[i]))


if __name__ == "__main__":
    main()
