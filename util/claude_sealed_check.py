#!/usr/bin/env python3
"""Sealed-room check — the referee with no tolerance in it.

WHAT THIS ASKS. The sealed-plank room (util/claude_bridge_gallery.lua,
OPS.sealedbox) is an oak-plank box ONE NODE THICK, inside a one-node air
gap, inside a shell of emitters. Nothing lights its interior. So the
correct interior frame is EXACTLY BLACK, and the referee is a COUNT:

    how many pixels are not black?  The answer must be 0.

No tolerance, no golden, no region boxes, no analytic inversion. This is
the cheapest referee in the harness and the sharpest -- cave-glass's own
history is the argument: on 2026-08-17 that room leaked 444 pixels while
its three region means still read 0.000000 and its RMS sat inside
furnace-050's noise floor. A mean cannot see a rare event; a count can.

WHY THE ROOM IS MADE OF PLANKS. The defect this exists to catch lived in
the CARVED RELIEF of baked 16^3 models -- grooves in one block meeting
grooves in the next across a single-node wall, 19 rays in 300,000 for oak
(spruce 5), fixed by b945042ee. cave-glass is the only other sealed arm
and it is built from a PLAIN wall node, so it contains no relief at all
and could never have caught it.

COUNT, DO NOT THRESHOLD A MEAN. The accumulator is a true 1/N running
average, so a rare leak event gets DIMMER the longer the camera stands
still (measured.md, "The sub-voxel sliver") -- averaging is the one
operation that hides exactly this defect. A pixel that was ever lit stays
non-zero, because gamma pulls a 1/2000-weight event up to a byte value of
about 11. So the count is the durable signal and the mean is not.

AND IT MASKS THE HARNESS'S OWN MARKER. claude_ci paints a 12x12 traced
marker square into the bottom-left of every capture to prove the present
path drew it -- 144 deliberately non-black pixels in every frame this
harness takes. The first pass of the origin-cell leak probe counted them
and reported "144 leaked pixels"; see environment-laws.md.

Usage: claude_sealed_check.py IMAGE [--marker-px N]
Prints, and always exits 0 -- the VERDICT belongs to the caller
(claude_ci.sealed_verdict), because "the referee could not speak" and
"the referee said zero" must not be the same exit code.
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# The marker's size, defaulted from claude_ci so the two cannot drift
# apart, but overridable so this script can be run against a frame taken
# by something else.
try:
    import claude_ci as _ci
    MARKER_PX = _ci.TRACE_MARKER_PX
except Exception:                                    # standalone use
    MARKER_PX = 12


def unmarked(png, marker_px=MARKER_PX):
    """The frame with claude_ci's traced-marker square blacked out."""
    im = np.asarray(Image.open(png).convert("RGB")).copy()
    h = im.shape[0]
    im[h - marker_px:h, 0:marker_px] = 0
    return im


def count(png, marker_px=MARKER_PX):
    """{nonzero_px, total_px, max_byte, brightest_at, marker_px}.

    Display bytes, not linear radiance: "exactly black" is a statement
    about the image, and inverting a tone curve first would only add
    arithmetic to a test that needs none.
    """
    im = unmarked(png, marker_px)
    v = im.max(axis=2)
    nz = v > 0
    n = int(nz.sum())
    at = None
    if n:
        y, x = np.unravel_index(int(np.argmax(v)), v.shape)
        at = [int(x), int(y)]
    return {"nonzero_px": n, "total_px": int(v.size),
            "max_byte": int(v.max()), "brightest_at": at,
            "marker_px": marker_px}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    png = sys.argv[1]
    mpx = MARKER_PX
    if "--marker-px" in sys.argv:
        mpx = int(sys.argv[sys.argv.index("--marker-px") + 1])
    c = count(png, mpx)
    print("sealed %s" % os.path.basename(png))
    print("marker masked: %dx%d px, bottom-left" % (mpx, mpx))
    print("nonzero pixels: %d of %d   max byte %d   brightest at %s"
          % (c["nonzero_px"], c["total_px"], c["max_byte"],
             c["brightest_at"]))
    print("verdict input: LEAKED %d" % c["nonzero_px"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
