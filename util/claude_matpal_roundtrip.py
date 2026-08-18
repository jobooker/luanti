#!/usr/bin/env python3
"""The material-index round trip, read off the GPU. THRESHOLD-FREE.

WHAT THIS IS ABOUT, before any number. Every cell of the traced world
carries one byte saying what it is. Until 2026-08-18 that byte was a set
of numeric BANDS with hand-placed edges (0 air, 100 water, 145 glass,
170 + light_source*5 emissive, 250 "I have a finer shape", 255 solid),
and the edges were deliberately put at half-byte midpoints so that an
8-bit texture round-trip could not drift a cell across one. The byte is
now an INDEX into a material table, and an index tolerates NO drift at
all: one least-significant bit is a different material. So the exactness
has to be proven rather than assumed, over all 256 values -- an
instrument that only exercises the values this world happens to contain
is the blind kind, and this project has paid for five of those.

WHAT IT MEASURES. claude_view 19 draws the test: the frame is split into
256 vertical columns, column i reads texel i of claudeMatProbe (a
256x1x1 RGBA8 texture in the SAME internal format and with the same
NEAREST/CLAMP parameters as the trace grid, carrying the byte i), runs
the walk's own matIndex() over it, and paints the column WHITE if the
answer was exactly i and BLACK if it was not. The bottom eighth draws
the palette's emission column as a brightness ramp, so a palette that
uploaded as zeros is visible in the same picture. Brightness, not hue:
a red/green pass-fail would be unreadable to John.

There is no tolerance to argue about. 256 white columns or a defect.

THE OTHER HALF OF THE SAME CLAIM, and why both exist: game.cpp's
claudeMatPalUpload() runs the identical 256 values through GL at startup
(upload -> glGetTexImage -> the decode expression in float32) and
publishes the score as `matpal_roundtrip` in claude_stats.json, which
claude_ci asserts on every capture. That half is mechanical and runs in
CI; this half is the one that goes through the REAL shader, the real
sampler and the real fragment program. Two independent readings.

Usage:  python3 util/claude_matpal_roundtrip.py
Requires a live seat (util/claude_look.sh). Takes its own screenshot
through the client, so it needs no screen-recording permission.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_lab as L  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The pass/fail screen occupies uv.y >= 0.125 in the shader; the ramp is
# below it. Rows are sampled well inside each band, and the HUD/hotbar
# live at the very bottom of the window, which the ramp band avoids by
# being read from the shader's own coordinates rather than the window's.
BAND_TOP = 0.20     # fraction of image height: inside the pass/fail screen
BAND_BOT = 0.70
RAMP_TOP = 0.90     # inside the emission ramp (shader draws it at uv.y<0.125,
RAMP_BOT = 0.96     # which is the BOTTOM of the image: uv.y is flipped)


def main():
    import numpy as np
    from PIL import Image

    L.doorway(claude_view=19, claude_show_hud=0, claude_show_chat=0)
    png = L.shot(token="matpal", settle=3.0, record=False)
    img = np.asarray(Image.open(png).convert("L"), dtype=np.int16)
    h, w = img.shape
    print("frame %dx%d  %s" % (w, h, os.path.basename(png)))

    band = img[int(h * BAND_TOP):int(h * BAND_BOT), :]
    # One reading per column of the 256, taken from the middle of the
    # column so a half-pixel of column edge cannot be mistaken for a fail.
    bad = []
    for i in range(256):
        x0 = int(round(i * w / 256.0))
        x1 = int(round((i + 1) * w / 256.0))
        mid = (x0 + x1) // 2
        v = float(band[:, mid].mean())
        if v < 128.0:
            bad.append((i, round(v, 1)))

    # THE GUARD: a screen that is uniformly white for a reason other than
    # the test passing (a black frame, a failed present path, the wrong
    # view) must not read as a pass. The ramp has to show STRUCTURE --
    # the emission column runs 0 for most indices and 2.4 for the
    # brightest emitter -- and the pass/fail screen has to be bright.
    ramp = img[int(h * RAMP_TOP):int(h * RAMP_BOT), :]
    ramp_span = int(ramp.max()) - int(ramp.min())
    print("pass/fail screen: min %d  max %d  mean %.1f"
          % (band.min(), band.max(), band.mean()))
    print("emission ramp:    min %d  max %d  span %d"
          % (ramp.min(), ramp.max(), ramp_span))

    if band.max() < 128:
        print("\nBLIND: the pass/fail screen is not being drawn at all "
              "(max %d). This is not a pass and it is not a fail -- the "
              "instrument has no signal. Check claude_view reached the "
              "client and that the traced pipeline is on." % band.max())
        return 2
    if ramp_span == 0:
        print("\nBLIND: the emission ramp is FLAT, so the palette texture "
              "is either all zeros or not being sampled. A round trip "
              "through a texture that carries nothing proves nothing.")
        return 2

    print("\nMATERIAL INDICES THAT DID NOT SURVIVE: %d" % len(bad))
    if bad:
        print("  " + ", ".join("%d(%.1f)" % b for b in bad[:24]))
    print("PASS -- 256/256" if not bad
          else "FAIL -- the per-cell byte is not a reliable index")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
