#!/usr/bin/env python3
"""Named region means for the arms that have no analytic answer.

WHAT THIS IS FOR. Two of the sky step's gates are DIFFERENTIAL rather
than analytic: they compare the same box of pixels across arms that share
a camera and differ in exactly one thing.

  * cave-skylight-noon / cave-skylight-night / cave-glass are the SAME
    7x5x7 room at the SAME vantage. Two have a 1x1 ceiling opening; the
    third has that opening plugged with the wall node. So the pair says
    "sky arrived through the hole" and the plugged one says "sky leaked
    through the wall" -- which no single room can tell apart, and which
    is why 1b built three rooms instead of one.

  * cozy-day-ci and cozy-day-dark-ci are the same cabin from the same
    camera with the vault panel swapped for its unlit twin, so the second
    is a sunlight-only interior. On the floor sits one stair node whose
    TREAD and RISER are two boxes on one object: the step line is a
    brightness edge, and its ratio is the cheapest possible check that
    sub-metre geometry is being LIT as geometry rather than painted.

It prints LINEAR radiance, not display bytes: the numbers have to be
comparable across arms whose brightness differs by orders of magnitude,
and the ACES shoulder is not linear. The inverse transform is the same
one claude_furnace_check and claude_cornell_check apply, for the same
reason -- a second copy of the display transform is a second thing to
keep honest.

THIS IS NOT A REFEREE. It reports; it returns no verdict. The only
analytic verdict in this family is claude_skyfurnace_check.

Usage: claude_regions.py IMAGE ARM
       claude_regions.py --list
"""
import sys

import numpy as np
from PIL import Image

# Boxes are (x0, y0, x1, y1) in image pixels at the pinned CI resolution
# of 1920x1080 (claude_ci.PINNED_CONF). They are hand-placed off real
# frames and they are PINNED, not derived from the resolution: a box that
# moves with the window is a box that measures something different after
# a resize.
REGIONS = {
    # The cave rooms. One camera, three rooms, three boxes.
    #   skylight_patch  the floor directly under the ceiling opening --
    #                   where light that came THROUGH THE HOLE lands
    #   far_wall        the wall opposite the door, which the opening
    #                   does not illuminate directly: indirect only
    #   ceiling         the underside of the roof, which in the plugged
    #                   room is the surface a leak would show on first
    "cave": {
        "skylight_patch": [(820, 780, 1120, 1010)],
        "far_wall": [(300, 300, 700, 560)],
        "ceiling": [(700, 60, 1250, 240)],
    },
    # The cabin. The stair is the point of the first two.
    #   stair_tread  the horizontal top of the LOWER step: faces up, so
    #                it sees the vault panel and the doorway sky
    #   stair_riser  the vertical face of the UPPER step, facing the
    #                camera and away from both: the dark half of the step
    #   carpet       the white floor patch, the room's brightest diffuse
    #                surface and the most sensitive to any new light
    #   dark_corner  the NE corner with no emitter within 4 m
    "cozy": {
        "stair_tread": [(1010, 608, 1110, 626)],
        "stair_riser": [(990, 548, 1075, 588)],
        "carpet": [(850, 800, 1400, 980)],
        "dark_corner": [(60, 640, 300, 900)],
    },
}

# arm name -> region set.
#
# A BOX IS A PROPERTY OF A VANTAGE, NOT OF A ROOM, and that is the whole
# reason this mapping is written out arm by arm rather than derived from
# claude_ci.VANTAGE_ROOM. The cabin has FOUR vantages over one physical
# room: cozy-ci stands mid-aisle facing +Z, cozy-night-ci stands at
# (7,9,2) facing yaw 55, and only cozy-day-ci and cozy-day-dark-ci share
# a camera. Handing all four the same pixel boxes would have compared the
# stair's tread against whatever happens to sit at those pixels from a
# different pose -- which is a comparison that looks like a measurement
# and is not one. Caught 2026-08-17 in the first run that printed them:
# "cozy-night-ci stair_riser" was reading a wall.
#
# The three cave arms DO share one camera (14,9,87 / 34,9,87 is the same
# pose in the geometrically identical twin room), which is what makes
# them the differential referee they were built to be.
ARM_REGIONS = {
    "cave-skylight-noon": "cave",
    "cave-skylight-night": "cave",
    "cave-glass": "cave",
    "cozy-day-ci": "cozy",
    "cozy-day-dark-ci": "cozy",
}


def aces_inverse(y):
    y = np.clip(y, 0.0, 0.9999)
    a = 2.51 - 2.43 * y
    b = 0.03 - 0.59 * y
    c = -0.14 * y
    return (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def measure(path, arm):
    """{region: {"mean": [r,g,b], "lum": float, "sd": float}} in LINEAR
    radiance, or {} for an arm with no boxes."""
    key = ARM_REGIONS.get(arm)
    if not key:
        return {}
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    lin = aces_inverse(im ** 2.2)
    out = {}
    for name, boxes in REGIONS[key].items():
        vals = [lin[y0:y1, x0:x1].reshape(-1, 3) for x0, y0, x1, y1 in boxes]
        v = np.concatenate(vals, axis=0)
        mean = v.mean(axis=0)
        out[name] = {
            "mean": [float(x) for x in mean],
            # Rec.709 luminance: ONE number per region, so a ratio across
            # arms is a ratio and not three. John is colorblind; the
            # colour triple is kept for the record, the luminance is what
            # gets compared.
            "lum": float(0.2126 * mean[0] + 0.7152 * mean[1]
                         + 0.0722 * mean[2]),
            "sd": float(v.mean(axis=1).std()),
            "px": int(v.shape[0]),
        }
    return out


def main():
    if "--list" in sys.argv:
        for arm, key in sorted(ARM_REGIONS.items()):
            print("%-22s -> %s: %s" % (arm, key,
                                       ", ".join(sorted(REGIONS[key]))))
        return
    path, arm = sys.argv[1], sys.argv[2]
    r = measure(path, arm)
    if not r:
        print("no regions defined for arm %r" % arm)
        return
    print("regions %s  (LINEAR radiance)" % arm)
    for name in sorted(r):
        d = r[name]
        print("region %-14s lum %.6f  rgb %s  sd %.6f  %d px"
              % (name, d["lum"], np.round(d["mean"], 6), d["sd"], d["px"]))


if __name__ == "__main__":
    main()
