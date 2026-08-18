#!/usr/bin/env python3
"""The interface ladder, scored against the closed form. THRESHOLD-SMALL.

WHAT THIS IS ABOUT, before any number. As of 2026-08-18 a ray in this
renderer can cross a surface instead of stopping at it: glass and water
are transmissive, and at every such surface the tracer takes ONE decision
-- how much of the ray reflects, and which way the rest of it bends.
Both halves are closed-form functions of the incidence angle and the two
indices of refraction, and NEITHER IS VISIBLE IN A PICTURE. A wrong
Fresnel term still draws plausible glass; a wrong index of refraction
still draws plausible water. So they get an instrument that needs no
scene at all.

WHAT IT MEASURES. claude_view 20 draws the shader's OWN
fresnelDielectric() and the shader's OWN refract(), as brightness,
across every incidence angle, in four horizontal bands. x is
cos(theta_i): grazing at the left edge, normal incidence at the right.

  band 0 (top)     R, air -> glass     eta = 1/1.52
  band 1           R, glass -> air     eta = 1.52   (TIR at the left)
  band 2           sin(theta_t), air -> glass
  band 3 (bottom)  sin(theta_t), glass -> air, 0 inside TIR

This script computes the same four curves off-GPU in double precision
and reports the largest disagreement. Bands 0 and 1 are the ENERGY
claim: R decides the split and T is 1 - R by construction in the path
loop, so scoring R IS scoring the split. Bands 2 and 3 are the GEOMETRY
claim -- Snell's law, and with it the index of refraction that went into
it. Band 3's zero crossing is the CRITICAL ANGLE, which is the sharpest
single reading of an IOR there is: 41.14 deg for n = 1.52, 48.61 for
water's 1.333, and nothing in between looks like either.

WHY THIS IS NOT A TAUTOLOGY, since one nearly was. The path loop never
computes T, so a check that "R + T = 1" cannot fail and would be the
sixth blind instrument this project has paid for. The reference here is
external: the exact unpolarised Fresnel expression, evaluated in Python.
A Schlick approximation, a swapped eta, or an IOR of 1.33 where 1.52 was
meant all show up as a curve that misses.

WHAT IT IS BLIND TO. Whether the PALETTE carries the IOR this ladder is
drawn at -- the ladder uses the shader's own IOR_LADDER constant on
purpose, so that it measures the arithmetic rather than the upload. It
also says nothing about transport: that a lossless interface does not
move a sealed furnace's radiance is util/claude_glass_probe.py's gate,
and it is a different claim.

Usage:  python3 util/claude_fresnel_check.py
Requires a live seat (util/claude_look.sh). Takes its own screenshot
through the client, so it needs no screen-recording permission.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_lab as L  # noqa: E402

# The IOR the shader's ladder is drawn at (claude_trace IOR_LADDER). It
# is a constant on BOTH sides on purpose: this instrument scores the
# interface arithmetic, not the palette upload.
IOR = 1.52
# Water's, for the palette strip below. It is NOT drawn as a ladder --
# the arithmetic is index-agnostic, so a second set of bands would
# measure the same four lines again -- but it IS the number the shader
# must have received, and the strip is where that is checked.
IOR_WATER = 1.333

# Rows sampled well inside each band. The trace runs at half resolution
# and claude_present bilinearly upsamples it, so a band boundary is
# blurred across ~2 screen rows; the middle of a band is not. The four
# bands occupy the top seven eighths of the frame; the bottom eighth is
# the palette strip.
BAND_FRACS = (0.1094, 0.3281, 0.5469, 0.7656)
PAL_FRAC = 0.94          # inside the palette strip

# The two material indices this renderer's transmissive materials land
# on. claudeMatIndex(kind, fine, light) = 1 + ((kind-1)*2 + fine)*15 +
# light with kinds air=0, solid=1, liquid=2, leaves=3, glass=4, nub=5 --
# so unlit glass is 1 + (3*2)*15 = 91 and unlit liquid is 1 + (1*2)*15 =
# 31. Written out rather than imported because the point of reading them
# HERE is to check the number that reached the GPU, and a check that
# recomputes the thing it is checking is not one.
PAL_GLASS = 91
PAL_WATER = 31

# Columns near uv.x = 0 are grazing incidence, where R has a vertical
# tangent; columns at uv.x = 1 are the right edge, where the upsample has
# nothing to its right to interpolate with. Both are named rather than
# quietly trimmed.
X_SKIP = 0.02
X_SKIP_HI = 0.005

# THE CRITICAL ANGLE IS A STEP, AND A STEP IS THE ONE THING THIS SCREEN
# CANNOT DRAW. Above it the glass -> air curves jump: R from 0.13 to 1,
# sin_t from 1 to 0. claude_trace runs at HALF resolution and
# claude_present bilinearly upsamples it, so a discontinuity is smeared
# across about two screen columns and any |value - closed form| taken
# inside that smear reads up to the full height of the step (0.50 on
# band 3) no matter how right the arithmetic is. Measured 2026-08-18:
# excluding this window takes band 3's worst disagreement from 0.501 to
# well under a display step, and band 1's from 0.129.
#
# So the step's HEIGHT is not the measurement; its POSITION is, and it is
# the sharpest number on this screen -- the critical angle, reported
# separately below and agreeing with asin(1/n) to 0.07 deg, which is
# about one and a half screen columns.
CRIT_SKIP = 0.012     # in cos(theta_i), ~23 columns at 1920 wide

# THE READ IS HALF A SCREEN PIXEL TO THE RIGHT OF THE COLUMN IT SITS IN,
# and that is measured rather than assumed. claude_trace runs at half
# resolution and claude_present resolves the full-res pixel from the four
# nearest half-res texels, so the value in column x is the curve
# evaluated half a full-res pixel along from that column's own centre.
# Scored 2026-08-18 by sweeping the offset over the same frame:
#
#   offset   band0    band1    band2    band3      (worst |delta|)
#   -0.5 px  0.00537  0.01214  0.00584  0.01345
#    0.0 px  0.00418  0.00845  0.00436  0.00973
#   +0.5 px  0.00311  0.00485  0.00285  0.00591   <- minimises all four
#   +1.0 px  0.00428  0.00724  0.00413  0.00723
#
# All four bands agree on the same offset, which is what makes it a
# geometry fact about the upsample rather than a fudge fitted to one
# curve. It matters most where the curve is steep: sin_t's slope near
# normal incidence is ~14 per unit uv, so half a pixel there is 0.008 --
# the entire residual this correction removes.
U_OFFSET_PX = 0.5

# 8-bit display quantisation is 1/255 = 0.0039, so half a step is the
# floor any reading of this screen can have, and claude_present's write
# truncates rather than rounds, which costs another. The gate is 2 steps.
# That is well under the smallest defect worth catching: Schlick misses
# the exact term by ~0.01-0.02 near grazing, and swapping 1.52 for 1.333
# moves R at normal incidence by 0.023 (6 display steps).
TOL = 0.008


def fresnel(cos_i, eta):
    """Exact unpolarised dielectric reflectance. eta = n_i / n_t."""
    s2 = eta * eta * (1.0 - cos_i * cos_i)
    if s2 >= 1.0:
        return 1.0
    cos_t = math.sqrt(1.0 - s2)
    rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t)
    rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t)
    return 0.5 * (rs * rs + rp * rp)


def sin_t(cos_i, eta):
    """sin(theta_t) by Snell, 0 inside total internal reflection."""
    si = math.sqrt(max(0.0, 1.0 - cos_i * cos_i))
    st = eta * si
    return 0.0 if st > 1.0 else st


def reference(band, cos_i):
    eta = (1.0 / IOR) if band in (0, 2) else IOR
    return fresnel(cos_i, eta) if band < 2 else sin_t(cos_i, eta)


def main():
    import numpy as np
    from PIL import Image

    L.doorway(claude_view=20, claude_show_hud=0, claude_show_chat=0)
    png = L.shot(token="fresnel", settle=3.0, record=False)
    img = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64)
    h, w = img.shape[:2]
    # claude_ci's traced marker is 12x12 deliberately-green pixels in the
    # bottom-left of EVERY frame this harness takes. Band 3 runs across
    # that corner, so it is blanked before anything is read -- the fifth
    # blind instrument here was a probe that counted it.
    img[h - 12:h, 0:12] = np.nan
    gray = img.mean(axis=2) / 255.0

    print("frame %dx%d  %s" % (w, h, os.path.basename(png)))
    print("ladder IOR %.3f, tolerance %.4f (8-bit step is %.4f)"
          % (IOR, TOL, 1.0 / 255.0))
    print()
    cos_crit = math.sqrt(max(0.0, 1.0 - (1.0 / IOR) ** 2))
    print("  band  what                       max|delta|   at cos_i   n")
    worst_all = 0.0
    rows = []
    for band, frac in enumerate(BAND_FRACS):
        row = int(h * frac)
        # three adjacent rows, averaged: the upsample is bilinear, so a
        # single row is one interpolation and three is the same one
        line = np.nanmean(gray[row - 1:row + 2, :], axis=0)
        worst, worst_x, n = 0.0, 0.0, 0
        for x in range(w):
            u = (x + 0.5 + U_OFFSET_PX) / w
            if u < X_SKIP or u > 1.0 - X_SKIP_HI:
                continue
            # the glass -> air bands carry a step at the critical angle
            if band in (1, 3) and abs(u - cos_crit) < CRIT_SKIP:
                continue
            v = line[x]
            if not np.isfinite(v):
                continue
            d = abs(v - reference(band, u))
            n += 1
            if d > worst:
                worst, worst_x = d, u
        worst_all = max(worst_all, worst)
        what = ["R  air -> glass", "R  glass -> air",
                "sin_t  air -> glass", "sin_t  glass -> air"][band]
        rows.append((band, what, worst, worst_x, n))
        print("  %4d  %-24s   %.5f     %.4f   %d"
              % (band, what, worst, worst_x, n))
    print("  (bands 1 and 3 skip |cos_i - %.4f| < %.3f: the critical "
          "angle is a STEP and a half-res upsample smears it)"
          % (cos_crit, CRIT_SKIP))

    # THE GUARD. A screen that is uniformly anything -- a black frame, a
    # failed present path, the wrong view -- must not read as a pass just
    # because the reference happens to be small somewhere. Two structural
    # facts the real ladder has and a flat frame does not: band 0 spans
    # nearly the whole range (R goes from 0.043 at normal incidence to 1
    # at grazing), and band 3 has a HARD zero region on the left, which is
    # total internal reflection and nothing else.
    b0 = np.nanmean(gray[int(h * BAND_FRACS[0]) - 1:
                         int(h * BAND_FRACS[0]) + 2, :], axis=0)
    b3 = np.nanmean(gray[int(h * BAND_FRACS[3]) - 1:
                         int(h * BAND_FRACS[3]) + 2, :], axis=0)
    span = float(np.nanmax(b0) - np.nanmin(b0))
    # the measured critical angle: the last column of band 3 that is dark
    dark = [x for x in range(w) if np.isfinite(b3[x]) and b3[x] < 0.5 / 255.0]
    crit_meas = None
    if dark:
        u = (max(dark) + 0.5) / w
        crit_meas = math.degrees(math.acos(min(1.0, u)))
    crit_true = math.degrees(math.asin(1.0 / IOR))
    print()
    print("  band 0 span            %.4f  (a real ladder spans ~0.96)" % span)
    print("  critical angle, band 3 %s vs %.2f deg closed form"
          % (("%.2f deg" % crit_meas) if crit_meas else "NOT FOUND",
             crit_true))

    if span < 0.5:
        print("\nBLIND: band 0 is flat, so this frame is not the ladder. "
              "Not a pass and not a fail -- the instrument has no signal. "
              "Check claude_view reached the client and that the traced "
              "pipeline is on.")
        return 2
    # ---- the palette's own IOR column, read through the same shader ---
    # The four bands above are drawn at a CONSTANT in the shader, so on
    # their own they prove the arithmetic and nothing about what game.cpp
    # uploaded. This strip is the other half: column i is the palette's
    # ior/2 where its transmission column says the material is a
    # dielectric, and black where it does not.
    strip = np.nanmean(gray[int(h * PAL_FRAC) - 3:int(h * PAL_FRAC) + 3, :],
                       axis=0)

    def pal_ior(i):
        x0 = int(round(i * w / 256.0))
        x1 = int(round((i + 1) * w / 256.0))
        return float(strip[(x0 + x1) // 2]) * 2.0

    lit = [i for i in range(256)
           if pal_ior(i) > 0.02]
    print()
    print("  palette IOR column: %d of 256 materials are dielectric"
          % len(lit))
    print("  glass (index %d) reads %.4f, water (index %d) reads %.4f"
          % (PAL_GLASS, pal_ior(PAL_GLASS), PAL_WATER, pal_ior(PAL_WATER)))
    pal_tol = 3.0 / 255.0 * 2.0     # 3 display steps, doubled by the /2
    pal_ok = (abs(pal_ior(PAL_GLASS) - IOR) <= pal_tol
              and abs(pal_ior(PAL_WATER) - IOR_WATER) <= pal_tol)
    print("  against %.3f and %.3f, tolerance %.4f: %s"
          % (IOR, IOR_WATER, pal_tol, "PASS" if pal_ok else "FAIL"))

    if crit_meas is None:
        print("\nBLIND: band 3 has no dark region, so total internal "
              "reflection is not being drawn at all and the geometry half "
              "of this screen is meaningless.")
        return 2

    ok = worst_all <= TOL and pal_ok
    print("\nWORST DISAGREEMENT ACROSS ALL FOUR BANDS: %.5f (tol %.5f)"
          % (worst_all, TOL))
    print("PASS -- the interface arithmetic matches the closed form" if ok
          else "FAIL -- the shader's Fresnel/Snell is not the closed form")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
