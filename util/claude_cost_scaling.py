#!/usr/bin/env python3
"""EXPERIMENT D of the descend cost instrument: what shape is the tax?

There is no GPU profiler for OpenGL on macOS, so nobody here can read a
register count off the driver. What CAN be done is to ask what SHAPE the
cost has, and the two candidate explanations predict different shapes:

  a fixed OCCUPANCY tax -- the shader needs more registers per thread,
  fewer threads run at once, and every unit of work is uniformly slower.
  That multiplies: the descent-present time should be a constant RATIO
  of the descent-absent time as the amount of work changes.

  a per-step cost -- extra instructions executed on some steps. That
  adds: a constant DIFFERENCE in milliseconds, independent of how much
  other work there is.

So: measure the two shaders (descent COMPILED OUT vs present-but-off) in
`cornell`, where the descent can never run, while changing the amount of
work two independent ways --

  RESOLUTION   claude_trace_scale 0.5 vs 1.0 (a quarter of the pixels vs
               all of them). Read at pipeline construction, so each
               value needs its own client start.
  PATH LENGTH  claude_bounces 1, 4, 24 (a live dial, no restart).

and report both the ratio and the difference at every point. A ratio
that holds while the difference moves says occupancy; a difference that
holds while the ratio moves says per-step work; neither holding says the
two-explanation frame is too simple and the report must say so.

IT ALSO COUNTS THE SHADER. Static, free, and the only register-adjacent
evidence available: source lines and DECLARED LOCALS in march() plus
descendCell(), for the current shader and for the pre-descend one
(`git show 8c9e87de2:client/shaders/claude_trace/opengl_fragment.glsl`).
Declared locals are NOT registers -- a compiler coalesces, rematerialises
and spills -- so this is an indication of direction and size, never a
count. Stated that way in the output.

Usage:  util/claude_cost_scaling.py [--reps 3]
The SERVER must already be up; the CLIENT is restarted four times.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import claude_lab as lab            # noqa: E402
import claude_cost_seat as cs       # noqa: E402

ROOM = "cornell"
SCALES = ["0.5", "1.0"]
BOUNCES = [1, 4, 24]
BASE_DIALS = {
    "claude_view": 0, "claude_nee": 0, "claude_grid_debug": 3,
    "claude_rng": 1, "claude_grid_follow": 1, "claude_models": 1,
    "claude_descend": 0, "claude_show_hud": 0, "claude_show_chat": 0,
    "claude_input_lock": 1, "claude_stats": 1,
}

PRE_DESCEND_REV = "8c9e87de2"
DECL = re.compile(r"^\s*(vec[234]|float|int|bool|uint|ivec[234]|bvec[234])\s"
                  r"+[A-Za-z_]")


def func_body(text, sig):
    i = text.index(sig)
    j = text.index("{", i)
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    return text[i:k + 1]


def count_body(text, sig):
    try:
        b = func_body(text, sig)
    except ValueError:
        return {"lines": 0, "declared_locals": 0, "absent": True}
    lines = [l for l in b.splitlines() if l.strip()
             and not l.strip().startswith("//")]
    return {"lines": len(lines),
            "declared_locals": sum(1 for l in lines if DECL.match(l)),
            "texture_fetches": b.count("texture3D(") + b.count("texture2D(")}


def static_shader_census():
    live = open(os.path.join(REPO, "client", "shaders", "claude_trace",
                             "opengl_fragment.glsl")).read()
    old = subprocess.run(
        ["git", "show", "%s:client/shaders/claude_trace/opengl_fragment.glsl"
         % PRE_DESCEND_REV], cwd=REPO, capture_output=True, text=True).stdout
    nod = open(os.path.join(HERE, "claude_shader_variants",
                            "nodescend_trace.glsl")).read()

    def code_lines(t):
        return sum(1 for l in t.splitlines()
                   if l.strip() and not l.strip().startswith("//"))
    out = {}
    for name, t in (("pre_descend_%s" % PRE_DESCEND_REV, old),
                    ("nodescend_variant", nod), ("live", t2 := live)):
        out[name] = {
            "file_code_lines": code_lines(t),
            "march": count_body(t, "bool march(vec3 ro, vec3 rd,"),
            "descendCell": count_body(
                t, "bool descendCell(vec3 cell, vec3 ro, vec3 rd,"),
            "uniform_count": len(re.findall(r"^uniform ", t, re.M)),
            "sampler_count": len(re.findall(r"^uniform sampler", t, re.M)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--settle", type=float, default=6.0)
    a = ap.parse_args()

    build = cs.require_release()
    if os.path.exists(os.path.join(HERE, "claude_shader_variants",
                                   ".installed.json")):
        cs.variant("restore")
    cs.variant("build")
    out = {"build_type": build, "room": ROOM, "readings": [],
           "static": static_shader_census(),
           "caveat": "declared locals are not registers; no GPU profiler "
                     "exists for OpenGL on macOS, so nothing here counts "
                     "registers -- only the SHAPE of the cost is measured"}
    print("static shader census:")
    for k, v in out["static"].items():
        print("  %-24s file %4d code lines | march %3d lines/%2d locals | "
              "descendCell %3d lines/%2d locals | %d uniforms"
              % (k, v["file_code_lines"], v["march"]["lines"],
                 v["march"]["declared_locals"], v["descendCell"]["lines"],
                 v["descendCell"]["declared_locals"], v["uniform_count"]))

    try:
        for variant in ("live", "nodescend"):
            if variant != "live":
                cs.variant("install", variant)
            for scale in SCALES:
                print("\n=== %s @ trace_scale %s ===" % (variant, scale),
                      flush=True)
                cs.restart_client({"claude_trace_scale": scale})
                lab.doorway(**BASE_DIALS)
                cs.freeze_time()
                cs.park(ROOM, dials=BASE_DIALS,
                        snapshot_tag="scal_%s_%s" % (variant, scale))
                for rep in range(a.reps):
                    for b in BOUNCES:
                        lab.doorway(claude_bounces=b)
                        p = cs.read_passes(settle_s=a.settle)
                        p.update(variant=variant,
                                 arm="absent" if variant == "nodescend"
                                     else "off",
                                 scale=scale, bounces=b, rep=rep)
                        out["readings"].append(p)
                        print("  %-9s scale %s bounces %2d rep%d  trace %7.3f"
                              % (variant, scale, b, rep, p["trace"]),
                              flush=True)
            if variant != "live":
                cs.variant("restore")
    finally:
        try:
            cs.variant("restore")
        except SystemExit:
            pass
        print(cs.variant("status"))

    grid = {}
    for scale in SCALES:
        for b in BOUNCES:
            cell = {}
            for arm in ("absent", "off"):
                v = [r["trace"] for r in out["readings"]
                     if r["scale"] == scale and r["bounces"] == b
                     and r["arm"] == arm]
                if v:
                    cell[arm] = cs.mmm(v)
            if "absent" in cell and "off" in cell:
                cell["ratio_off_over_absent"] = round(
                    cell["off"]["median"] / cell["absent"]["median"], 4)
                cell["difference_ms"] = round(
                    cell["off"]["median"] - cell["absent"]["median"], 4)
            grid["scale%s/bounces%d" % (scale, b)] = cell
    out["grid"] = grid
    print("\n%-22s %10s %10s %8s %8s"
          % ("point", "absent", "off", "ratio", "diff ms"))
    for k, v in grid.items():
        if "ratio_off_over_absent" not in v:
            continue
        print("%-22s %10.3f %10.3f %8.3f %8.3f"
              % (k, v["absent"]["median"], v["off"]["median"],
                 v["ratio_off_over_absent"], v["difference_ms"]))
    cs.write_json("scaling.json", out)


if __name__ == "__main__":
    main()
