#!/usr/bin/env python3
"""claude_1a_direct — INSTRUMENT A for roadmap step 1a (the NEE bias).

The defect: claude_nee = 1 converges 2-9% hot by region against photo
mode in Cornell (spec/measured.md "CI red/green"). MIS adds two
estimates of the direct term at every vertex — w_l * (aimed light
sample) + w_b * (Le the BSDF ray found) — and the sum is supposed to be
exactly one copy of it. Three things can be wrong: the light half, the
BSDF half, or the weights. A referee that only sees the SUM cannot say
which, and every referee we own only sees the sum.

So this captures the SAME vantage three times with claude_view 9, 10 and
11, which are three independent estimates of ONE quantity — the direct
radiance leaving the primary hit (see the INSTRUMENT A block in
client/shaders/claude_trace/opengl_fragment.glsl):

  view  9  the light-sampling half alone, w_l forced to 1 (neeDirect()
           itself, one argument different, so it cannot drift from the
           estimator it is judging)
  view 10  the BSDF half alone, w_b forced to 1 — one cosine sample and
           whatever Le it lands on, i.e. `rho * Le`, no pdf arithmetic
  view 11  ANALYTIC: Lambert's contour formula for the Cornell ceiling
           panel, times rho/PI. No rays, no RNG, no light list.

CONVERGED, 9 AND 10 MUST EACH EQUAL 11. Whichever disagrees names the
broken half; if both match, the halves are sound and the bug is in the
weights. Two extra arms are captured for context: view 0 at claude_nee 0
(photo, the truth) and view 0 at claude_nee 1 (the defect itself), so
one run holds both the symptom and the diagnosis.

Numbers come from claude_cornell_check.py's five region boxes — the same
boxes and the same inverse transform that judge every other Cornell
frame. NOTHING here is hand-computed. Ratio images (9/11, 10/11, 9/10)
are written as GRAYSCALE, mid-gray = 1.000, because John is colorblind
and a ratio is a brightness question.

WHAT THIS INSTRUMENT IS BLIND TO (physics-contract §8 clause 3):
 * view 11 has no shadow ray: Cornell's two gray186 occluders shadow
   parts of the floor, and 9 and 10 read darker there BY BEING RIGHT.
   Those patches are dark in the ratio images and are not the defect.
 * view 11 is CORNELL ONLY (hard-coded panel rectangle) and is exact
   only where the panel lies wholly above the receiver's horizon — true
   of every wall/floor/ceiling of this room, false on the side faces of
   the two occluder blocks.
 * 9 and 10 are Monte Carlo. A disagreement at low still_frames is
   noise. The converged-frames assertion is the guard; the two runs
   must also agree with each other.
 * The region boxes cannot see anything outside the five boxes, or any
   defect that scales all of them together.

Usage:
  python3 util/claude_1a_direct.py            # build, seat, 5 arms
  python3 util/claude_1a_direct.py --skip-build --settle 120
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab            # noqa: E402
import claude_ci as ci              # noqa: E402  (seat, capture, guards)
import claude_cornell_check as cornell  # noqa: E402

VANTAGE = "cornell"

# The arms. `dials` overrides ci.CANONICAL_DIALS, which already pins the
# photo state, the overlay suppression and the input lock.
ARMS = [
    {"name": "photo", "dials": {"claude_view": 0, "claude_nee": 0},
     "what": "the truth (physics-contract §6)"},
    {"name": "nee1", "dials": {"claude_view": 0, "claude_nee": 1},
     "what": "the defect, for context in the same run"},
    {"name": "v09-aimed", "dials": {"claude_view": 9, "claude_nee": 1},
     "what": "light-sampling half, w_l = 1"},
    {"name": "v10-bsdf", "dials": {"claude_view": 10, "claude_nee": 1},
     "what": "BSDF half, w_b = 1"},
    {"name": "v11-analytic", "dials": {"claude_view": 11, "claude_nee": 1},
     "what": "analytic rho/PI * E, Cornell panel"},
    # FORENSICS, off by default (--only v12-forensics,v13-forensics).
    # Not radiance: each channel is a mean over frames of a fact about
    # the aimed sample, so channel 2 and 3 must be divided by channel 1
    # (the hit rate) to read as a mean over SUCCESSFUL samples.
    {"name": "v12-forensics", "dials": {"claude_view": 12, "claude_nee": 1},
     "what": "R=hit rate, G=emitter world y/64, B=cos_x", "forensic": True},
    {"name": "v13-forensics", "dials": {"claude_view": 13, "claude_nee": 1},
     "what": "R=hit rate, G=slot/16, B=k/6", "forensic": True},
    {"name": "v14-rng", "dials": {"claude_view": 14, "claude_nee": 1},
     "what": "R=5*[u1<.1] G=25*[u1<.02] B=5*[u2<.1]; ALL EXPECT 0.500",
     "forensic": True},
    {"name": "v15-rng", "dials": {"claude_view": 15, "claude_nee": 1},
     "what": "R=u1 G=sqrt(1-u1) B=2*u1^2; EXPECT 0.500 / 0.6667 / 0.6667",
     "forensic": True},
    {"name": "v16-bsdf-exact", "dials": {"claude_view": 16, "claude_nee": 1},
     "what": "view 10 with the DDA replaced by a closed-form panel hit",
     "forensic": True},
]
DEFAULT_ARMS = [a["name"] for a in ARMS if not a.get("forensic")]
# Pairs judged as ratio images + region ratios. (this, golden-ish).
PAIRS = [("v09-aimed", "v11-analytic"), ("v10-bsdf", "v11-analytic"),
         ("v09-aimed", "v10-bsdf"), ("v16-bsdf-exact", "v11-analytic"),
         ("v16-bsdf-exact", "v10-bsdf")]

RATIO_MID = 0.5      # mid-gray in the ratio image == ratio 1.000
RATIO_SPAN = 2.0     # gray 1.0 == ratio 2.0, gray 0.0 == ratio 0.0
RATIO_FLOOR = 1e-4   # denominator below this: undefined, drawn as pure
                     # black with a checker so it cannot read as "0.0"


def ratio_image(a_png, b_png, out_png):
    """Grayscale a/b in linear luminance. Mid-gray IS 1.000.

    Undefined pixels (denominator under the floor — e.g. the coplanar
    ceiling, where the analytic answer is zero by construction) are
    stamped with a 4px checker so "no signal" cannot be misread as
    "ratio 0", which is a different claim.
    """
    import numpy as np
    from PIL import Image
    _, la = cornell.load(a_png)
    _, lb = cornell.load(b_png)
    ya = la @ cornell.LUMA
    yb = lb @ cornell.LUMA
    ok = yb > RATIO_FLOOR
    r = np.where(ok, ya / np.maximum(yb, RATIO_FLOOR), 0.0)
    # gray = ratio / RATIO_SPAN, so ratio 1.0 lands on RATIO_MID exactly
    g = np.clip(r / RATIO_SPAN, 0.0, 1.0)
    img = (g * 255.0).astype(np.uint8)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((xx // 4) + (yy // 4)) % 2 == 0)
    img = np.where(ok, img, np.where(checker, 40, 0).astype(np.uint8))
    Image.fromarray(img, mode="L").save(out_png)
    inside = ok & (ya > RATIO_FLOOR)
    return {"defined_frac": float(ok.mean()),
            "median_ratio_where_defined": float(np.median(r[inside]))
            if inside.any() else None}


def region_table(pngs):
    """{arm: {region: (lum, rgb, purity)}} straight from the referee."""
    return {name: cornell.region_stats(p) for name, p in pngs.items()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settle", type=float, default=90.0,
                    help="s of stillness per arm (views 9/10 are Monte "
                         "Carlo; default %(default)s)")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--allow-debug", action="store_true")
    ap.add_argument("--only", default=",".join(DEFAULT_ARMS),
                    help="comma-separated arm names (default: %(default)s)")
    ap.add_argument("--nee", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    want = [a.strip() for a in args.only.split(",") if a.strip()]
    arms = [a for a in ARMS if a["name"] in want]
    if not arms:
        print("no such arm(s): %s" % args.only)
        return 1

    git = ci.git_state()
    run_id = "%s_%s_1a-direct" % (
        time.strftime("%Y%m%d-%H%M%S", time.gmtime()), git["tag"])
    rundir = os.path.join(ci.CI_DIR, run_id)
    os.makedirs(rundir, exist_ok=True)
    print("run dir: %s" % rundir)
    if git["dirty"]:
        print("!! WORKING TREE IS DIRTY — scratch run, %s names no commit"
              % git["tag"])

    run = dict(git, run_id=run_id, settle=args.settle, arms=arms,
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               shots={}, problems=[])

    err = ci.bring_up_seat(rundir, run, args)
    if err:
        run["aborted"] = err
        print("ABORTED: %s" % err)
        json.dump(run, open(os.path.join(rundir, "run.json"), "w"), indent=2)
        return 1
    if run.get("shader_failures"):
        run["problems"].append("shader compile failures: %s"
                               % run["shader_failures"])
        print("!! SHADER COMPILE FAILURES — the frames below are raster:")
        for line in run["shader_failures"]:
            print("   %s" % line)
    if not run["freeze"].get("deepening"):
        run["problems"].append("accumulator not deepening")

    vs = lab.load_vantages()
    ci.set_doors(True)
    run["room_integrity"] = ci.check_room_integrity()
    for item in run["room_integrity"]:
        bad = item.get("off_spec") or []
        if bad:
            run["problems"].append("%s: %d off-spec node(s)"
                                   % (item["room"], len(bad)))
    pngs = {}
    try:
        for arm in arms:
            dials = dict(ci.CANONICAL_DIALS, **arm["dials"])
            print("\n=== %s (%s): %s" % (arm["name"], arm["what"], arm["dials"]))
            png, cap = ci.capture({"name": arm["name"]}, vs[VANTAGE],
                                  ci.park_for(VANTAGE, vs), dials, rundir,
                                  args.settle)
            pngs[arm["name"]] = png
            ok_marker, marker_detail = ci.trace_marker(png)
            ds = cap.get("dial_state") or {}
            aim = cap.get("aim_at_shutter") or {}
            vol = cap.get("volume") or {}
            sf = cap.get("still_frames_at_shutter")
            row = {"png": os.path.basename(png), "dials": arm["dials"],
                   "still_frames": sf, "traced": ok_marker,
                   "traced_detail": marker_detail,
                   "dials_ok": ds.get("ok"), "dials_error": ds.get("error"),
                   "dials_seen": {k: ds.get("seen", {}).get(k)
                                  for k in ci.PROVEN_DIALS},
                   "aim_ok": aim.get("ok"), "aim_detail": aim.get("detail"),
                   "volume_ok": vol.get("ok"),
                   "area_emitters": vol.get("area_emitters"),
                   "area_total": vol.get("area_total")}
            run["shots"][arm["name"]] = row
            for claim, ok in (("dials", ds.get("ok")), ("traced", ok_marker),
                              ("aim", aim.get("ok")), ("volume", vol.get("ok")),
                              ("converged", (sf or 0) >= ci.CONVERGED_MIN)):
                if not ok:
                    run["problems"].append("%s-%s" % (arm["name"], claim))
            print("  still_frames %s | traced %s | dials %s | aim %s"
                  % (sf, ok_marker, ds.get("ok"), aim.get("ok")))
    finally:
        ci.set_doors(False)

    # ------------------------------------------------------------ numbers
    print("\n" + "=" * 72)
    print("REGION MEANS (linear luminance, claude_cornell_check boxes)")
    print("=" * 72)
    stats = region_table(pngs)
    hdr = "%-16s" % "region" + "".join("%14s" % a["name"] for a in arms)
    print(hdr)
    for rname in cornell.REGIONS:
        line = "%-16s" % rname
        for a in arms:
            st = stats.get(a["name"])
            line += "%14.6f" % st[rname][0] if st else "%14s" % "-"
        print(line)
    run["region_means"] = {a: {r: stats[a][r][0] for r in cornell.REGIONS}
                           for a in stats}
    run["region_rgb"] = {a: {r: list(map(float, stats[a][r][1]))
                             for r in cornell.REGIONS} for a in stats}

    print("\n" + "=" * 72)
    print("RATIOS — 9 and 10 must EACH equal 11")
    print("=" * 72)
    run["ratios"] = {}
    for this, gold in PAIRS:
        if this not in pngs or gold not in pngs:
            continue
        key = "%s_over_%s" % (this, gold)
        print("\n--- %s / %s" % (this, gold))
        vals = {}
        for rname in cornell.REGIONS:
            r = stats[this][rname][0] / max(stats[gold][rname][0], 1e-12)
            vals[rname] = float(r)
            print("ratio %-15s %.4f  (this %.6f / that %.6f)"
                  % (rname, r, stats[this][rname][0], stats[gold][rname][0]))
        out = os.path.join(rundir, "ratio_%s.png" % key)
        try:
            vals["_image"] = ratio_image(pngs[this], pngs[gold], out)
            print("ratio image: %s (mid-gray 128 = 1.000, white = %.1f, "
                  "checkered = denominator under %.0e)"
                  % (out, RATIO_SPAN, RATIO_FLOOR))
        except Exception as e:
            print("ratio image failed: %s" % e)
        run["ratios"][key] = vals

    # the coplanar-ceiling claim, stated as its own line
    try:
        c9 = stats["v09-aimed"]["ceiling_flanks"][0]
        c11 = stats["v11-analytic"]["ceiling_flanks"][0]
        cph = stats["photo"]["ceiling_flanks"][0]
        print("\nCOPLANAR CEILING (direct term is exactly zero there): "
              "view 9 %.6f | view 11 %.6f | photo total %.6f | "
              "9 as %% of the photo ceiling: %.2f%%"
              % (c9, c11, cph, 100.0 * c9 / max(cph, 1e-12)))
        run["coplanar_ceiling"] = {"v09": c9, "v11": c11, "photo": cph,
                                   "v09_pct_of_photo": 100.0 * c9 / max(cph, 1e-12)}
    except Exception as e:
        print("coplanar ceiling line failed: %s" % e)

    run["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(run, open(os.path.join(rundir, "run.json"), "w"), indent=2)
    print("\n%s" % rundir)
    if run["problems"]:
        print("PROBLEMS (%d): %s" % (len(run["problems"]),
                                     ", ".join(run["problems"])))
        return 1
    print("all per-capture guards passed")
    return 0


if __name__ == "__main__":
    sys.exit(ci.with_run_lock(lambda _a: main())(None))
