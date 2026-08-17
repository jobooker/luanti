#!/usr/bin/env python3
"""Build MEASUREMENT-ONLY variants of the tracer's fragment shaders, and
install/restore them under the live shader path.

WHY THIS EXISTS. On 2026-08-16 the walker learned to step inside a
class-250 cell ("descend"). The trace pass then cost 2.7x more in the
cosy cabin -- expected, it does more work -- and ALSO 35-59 % more in
Cornell, a room that has no class-250 cell and where the descent can
never execute, with the dial `claude_descend` set to 0. A dial that is
off cannot cost time by running, so the suspicion is that the mere
PRESENCE of the code costs: a bigger, branchier shader needs more
registers per thread, the GPU can keep fewer threads in flight, and
every ray in the program gets slower whether or not it takes the branch.

That suspicion cannot be tested with a dial. It needs a shader with the
descent code ABSENT, compared against the same shader with it present
and switched off, on the same binary, same seat, same room. This script
makes that shader.

HOW, and why not with #ifdef. The engine compiles these files at runtime
and prepends its own header; there is no supported way to inject a
`#define` from a setting without a C++ change and a rebuild, and a
rebuild would break "same binary". So the variant is produced by TEXT
SURGERY on the committed shader, driven by anchors that are asserted to
occur exactly once, and installed by copying the file into place and
restarting the client. That is the same mechanism the descend session
used to A/B against the parent shader.

SAFETY. `install` saves the live files to `<path>.orig` and prints the
sha256 of what it saved; `restore` puts them back and REFUSES to report
success unless the restored bytes hash to the value recorded at install.
Nothing here edits the committed shader in place.

VARIANTS

  live       the committed shader, unmodified (the "present" arm; the
             dial then chooses off/on)
  twowalk    the SHADER THIS REWRITE REPLACED, verbatim, pinned at the
             commit (PRE_DESCEND_COMMIT). march() plus a separate
             descendCell() holding a second DDA's worth of locals. It is
             here so "did folding the two walks into one pay" can be
             asked inside ONE client session rather than across days.
  nodescend  descent COMPILED OUT: descendCell() and both of its call
             sites are deleted. inSubvoxRing()/subvoxSolid() are left in
             the text and become unreferenced, so the GLSL compiler
             removes them -- and `claudeSubvoxTex` therefore reports as
             a DEAD uniform in the startup census. That is expected for
             this arm and is not a defect (environment-laws: "a uniform
             that is declared but never read is stripped").
  counters   the committed shader PLUS per-pixel step counters, exposed
             through five debug views. THIS VARIANT IS SLOWER THAN THE
             SHADER IT MEASURES and must never be used for a timing
             number -- it exists only to count steps. It also installs a
             matching claude_present that passes the counter views
             through with a single NEAREST tap instead of the
             joint-bilateral upsample, because a blended byte plane is
             not a number.

COUNTER VIEWS (counters variant only). These numbers REPLACE the NEE
forensics/RNG-density views 12-16 that the committed shader puts there;
the variant is a throwaway measuring shader, not a new instrument in the
engine, so it reuses the numbers rather than needing a C++ range change
(`claude_view` is clamped to 0..16 in game.cpp).

  12  PRIMARY ray: R = fine steps low byte, G = fine steps high byte,
      B = 1.0 if the primary ray descended into any class-250 cell
  13  PRIMARY ray: R/G = coarse (1 m) steps, 16-bit little-endian
  14  WHOLE PATH incl. every bounce and every shadow/NEE ray:
      R/G = fine steps, 16-bit
  15  WHOLE PATH: R/G = coarse steps, 16-bit
  16  THE HUMAN VIEW -- a GRAY brightness ladder, log2 of whole-path
      fine steps: black = 0 fine steps, and each doubling of the count
      is one step up an 8-rung ramp to white. Gray, not colour, on
      purpose (John is colorblind).

Each 16-bit channel is `mod(n,256)/255` and `floor(n/256)/255`, which
survives the 8-bit framebuffer exactly: k/255 quantises back to k.

Usage:
    util/claude_shader_variant.py build                # write variants
    util/claude_shader_variant.py install nodescend
    util/claude_shader_variant.py install counters
    util/claude_shader_variant.py restore
    util/claude_shader_variant.py status
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TRACE = os.path.join(REPO, "client", "shaders", "claude_trace",
                     "opengl_fragment.glsl")
PRESENT = os.path.join(REPO, "client", "shaders", "claude_present",
                       "opengl_fragment.glsl")
VDIR = os.path.join(HERE, "claude_shader_variants")
STATE = os.path.join(VDIR, ".installed.json")

# The committed shaders this script was written against. A mismatch is
# not a warning: the anchors below are line-for-line assumptions about
# these files, and silently transforming a shader they no longer
# describe is how a measurement gets taken of something nobody wrote.
TRACE_SHA = "652f04699fe0a77043096a01069cdd2dc6ed9c46319c4175998a73980b161181"
PRESENT_SHA = "9c3bd80a9a83f1fb441f2d7a55064b36ff8d5b79d0d04a7495e03e9f4d353f25"


def sha(path_or_text):
    if os.path.exists(str(path_or_text)):
        data = open(path_or_text, "rb").read()
    else:
        data = path_or_text.encode()
    return hashlib.sha256(data).hexdigest()


def cut_block(text, anchor, name):
    """Delete the brace-matched block that starts at `anchor` (which must
    contain the opening `{` or be followed by one). Asserts the anchor is
    unique."""
    n = text.count(anchor)
    if n != 1:
        raise SystemExit("anchor %r for %s occurs %d times, expected 1"
                         % (anchor[:60], name, n))
    i = text.index(anchor)
    j = text.index("{", i)
    depth = 0
    k = j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    if depth != 0:
        raise SystemExit("unbalanced braces cutting %s" % name)
    # swallow the trailing newline of the closing brace
    end = k + 1
    if end < len(text) and text[end] == "\n":
        end += 1
    return text[:i] + text[end:]


def replace_once(text, old, new, name):
    n = text.count(old)
    if n != 1:
        raise SystemExit("anchor %r for %s occurs %d times, expected 1"
                         % (old[:60], name, n))
    return text.replace(old, new)


# THE `absent` ARM IS PINNED AT A COMMIT, NOT DERIVED FROM THE LIVE FILE.
# `nodescend` answers "what did this shader cost before the descent
# existed", and that arm's cost is the target the lean-descend
# experiment (2026-08-17) has to beat -- so the two tables in
# spec/measured.md, the one taken against the two-walk shader and the
# one taken against the one-loop shader, MUST share one baseline text.
# Deriving it from whatever is checked out would give each session its
# own `absent` and quietly make the two taxes incomparable. The output
# is asserted byte-for-byte (below the 4-line header) against the
# nodescend_trace.glsl the first table was measured with.
PRE_DESCEND_COMMIT = "64b005975"
# sha256 (first 16) of the generated body, BELOW the banner. Asserted on
# every build. Verified 2026-08-17 to be byte-identical to the
# nodescend_trace.glsl the 2026-08-16/17 tax table was measured with.
PRE_BODY_SHA = "4054431f2b5fb28a"


def pre_descend_source():
    """The two-walk shader as committed, straight out of git."""
    import subprocess
    r = subprocess.run(["git", "show", "%s:client/shaders/claude_trace/"
                        "opengl_fragment.glsl" % PRE_DESCEND_COMMIT],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("cannot read the pinned baseline shader at %s:\n%s"
                         % (PRE_DESCEND_COMMIT, r.stderr))
    return r.stdout


HDR = ("// GENERATED by util/claude_shader_variant.py -- DO NOT EDIT.\n"
       "// Measurement-only variant of client/shaders/%s.\n"
       "// Source sha256 %s\n"
       "// %s\n")

# The pinned-baseline arm names the COMMIT it came from, not the sha of
# a live file it deliberately ignores.
PRE_HDR = ("// GENERATED by util/claude_shader_variant.py -- DO NOT EDIT.\n"
           "// Measurement-only variant of "
           "client/shaders/claude_trace/opengl_fragment.glsl\n"
           "// PINNED baseline: the two-walk shader at %s, with the\n"
           "// descent compiled out -- descendCell() and both of its call\n"
           "// sites deleted. This is the `absent` arm and it must NOT\n"
           "// track the live file. Body sha256 %s\n")


def build_nodescend(_src_ignored):
    """Descent compiled out of the PINNED two-walk shader (see
    PRE_DESCEND_COMMIT). Its anchors are that file's, not the live
    file's -- which is the point: this arm must not move when march()
    is rewritten."""
    t = pre_descend_source()
    t = cut_block(t, "bool descendCell(vec3 cell, vec3 ro, vec3 rd, "
                     "float tEnter, vec3 nEnter,", "descendCell()")
    t = cut_block(t, "\tif (claudeDescend > 0.5 && inSubvoxRing(cell)) {",
                  "march() starting-cell descend")
    t = cut_block(t, "\t\tif (claudeDescend > 0.5 && s.a > CLASS_SUBVOX_LO",
                  "march() in-loop descend")
    # Comments may still SAY descendCell -- they are the record of what
    # this room used to do and deleting them would make the variant
    # harder to read, not more honest. No CODE line may reference it.
    live = [l for l in t.splitlines()
            if "descendCell" in l and not l.lstrip().startswith("//")]
    if live:
        raise SystemExit("nodescend variant still CALLS descendCell:\n  "
                         + "\n  ".join(live))
    body = sha(t)[:16]
    if body != PRE_BODY_SHA:
        raise SystemExit(
            "the pinned `absent` arm MOVED: body sha %s, want %s.\n"
            "That arm is the pre-descend cost every table in\n"
            "spec/measured.md is measured against; if it changes, the\n"
            "old and new taxes stop being comparable. Fix the anchors\n"
            "or re-pin deliberately, with a note saying why."
            % (body, PRE_BODY_SHA))
    return (PRE_HDR % (PRE_DESCEND_COMMIT, body)) + t


# --- the counters variant --------------------------------------------
# Globals, declared just above march(). They are plain adds in the walk;
# that makes this shader measurably SLOWER than the one it counts for,
# which is why no timing number is ever taken from it.
#
# WHERE THE INCREMENTS GO, and this is the whole reason the numbers stay
# comparable with the 2026-08-16 histogram. The walk is now ONE loop and
# a step's rung is a variable, so "which loop was this" is no longer the
# answer -- the increments are placed to reproduce, event for event,
# what the two-walk shader counted:
#   g_fine   one per SUB-VOXEL VISIT. The old inner loop incremented
#            once per iteration, including its i == 0 (which tests the
#            entry sub-voxel, or skips the test under skipFirst). In the
#            one-loop walk that iteration IS the rescale, so both
#            rescale sites increment, and the fine-rung loop body
#            increments only when it actually lands on a sub-voxel --
#            the iteration that steps OUT of the cell was, in the old
#            code, the tail of an iteration already counted.
#   g_coarse one per 1 M STEP. The old outer loop incremented once per
#            iteration, and every iteration either arrived at a cell or
#            escaped the grid; so the increment sits at that junction,
#            which a coarse-rung step reaches directly and a rescale
#            back to 1 m reaches after the same face crossing.
COUNTER_GLOBALS = """
// --- STEP COUNTERS (measurement variant only) ------------------------
// g_fine   1/16 m sub-voxel visits, all rays, all depths
// g_coarse 1 m steps of the walk, all rays, all depths
// g_desc   1.0 once any march() has descended into a class-250 cell
float g_fine = 0.0;
float g_coarse = 0.0;
float g_desc = 0.0;

// 16-bit little-endian byte pair, exact through an 8-bit framebuffer:
// k/255 quantises back to k, so the reader gets the integer it counted.
vec2 pack16(float n)
{
	n = clamp(floor(n + 0.5), 0.0, 65535.0);
	return vec2(mod(n, 256.0) / 255.0, floor(n / 256.0) / 255.0);
}

"""

COUNTER_VIEWS = """
	// --- STEP COUNTER VIEWS 12-16 (measurement variant only) ---------
	// These REPLACE the committed shader's NEE-forensics and
	// RNG-density views at the same numbers. See the header of
	// util/claude_shader_variant.py for the encoding.
	if (view >= 12 && view <= 16) {
		vec3 dbg = vec3(0.0);
		if (view == 12)
			dbg = vec3(pack16(g_primFine), g_primDesc);
		else if (view == 13)
			dbg = vec3(pack16(g_primCoarse), 0.0);
		else if (view == 14)
			dbg = vec3(pack16(g_fine), 0.0);
		else if (view == 15)
			dbg = vec3(pack16(g_coarse), 0.0);
		else {
			// THE HUMAN VIEW. Gray ladder, eight rungs, one per
			// doubling of the whole-path fine-step count. Black is
			// "this pixel never stepped inside a cell"; white is
			// 256 fine steps or more. Brightness only -- no colour.
			float g = g_fine < 0.5 ? 0.0
					: clamp(1.0 + floor(log2(g_fine)), 1.0, 8.0) / 8.0;
			dbg = vec3(g);
		}
		gl_FragColor = vec4(dbg, tPack);
		return;
	}
"""


def build_counters(src):
    t = src
    # globals + pack16, in front of march()
    t = replace_once(
        t,
        "bool march(vec3 ro, vec3 rd, out vec3 hp, out vec3 n, out vec3 alb,",
        COUNTER_GLOBALS
        + "bool march(vec3 ro, vec3 rd, out vec3 hp, out vec3 n, "
          "out vec3 alb,",
        "counter globals")
    # a sub-voxel visit on the fine rung
    t = replace_once(
        t,
        "\t\t\tif (!escaped) {\n",
        "\t\t\tif (!escaped) {\n\t\t\t\tg_fine += 1.0;\n",
        "fine step counter")
    # the 1 m step / escape junction
    t = replace_once(
        t,
        "\t\tif (escaped)\n",
        "\t\tg_coarse += 1.0;\n\t\tif (escaped)\n",
        "coarse step counter")
    # the two rescale sites: each is the old inner loop's i == 0, and
    # each is where the ray becomes one that descended
    t = replace_once(
        t,
        "\t\t\tvec3 pu = clamp((ro - cellHi) * SUBV, vec3(0.0),\n",
        "\t\t\tg_fine += 1.0;\n\t\t\tg_desc = 1.0;\n"
        "\t\t\tvec3 pu = clamp((ro - cellHi) * SUBV, vec3(0.0),\n",
        "descend flag + entry visit, starting cell")
    t = replace_once(
        t,
        "\t\t\tvec3 su = floor(pu);\n",
        "\t\t\tvec3 su = floor(pu);\n"
        "\t\t\tg_fine += 1.0;\n\t\t\tg_desc = 1.0;\n",
        "descend flag + entry visit, in-loop")
    # per-pixel primary snapshot, declared alongside the other primary*
    t = replace_once(
        t,
        "\tbool primaryHit = false;\n",
        "\tbool primaryHit = false;\n"
        "\t// the counters AS OF THE END OF THE PRIMARY RAY, kept apart\n"
        "\t// from the running totals so one view can answer \"what did\n"
        "\t// the camera ray alone cost\" and another \"what did the\n"
        "\t// whole path cost\".\n"
        "\tfloat g_primFine = 0.0;\n"
        "\tfloat g_primCoarse = 0.0;\n"
        "\tfloat g_primDesc = 0.0;\n",
        "primary counter locals")
    # zero the globals at the top of the path (after the RNG seeding)
    t = replace_once(
        t,
        "\t// --- the path ----------------------------------------------------\n",
        "\tg_fine = 0.0;\n\tg_coarse = 0.0;\n\tg_desc = 0.0;\n"
        "\t// --- the path ----------------------------------------------------\n",
        "counter reset")
    # snapshot after the primary march returns
    t = replace_once(
        t,
        "\t\tif (seg == 0) {\n\t\t\tprimaryHit = true;\n",
        "\t\tif (seg == 0) {\n"
        "\t\t\tg_primFine = g_fine;\n"
        "\t\t\tg_primCoarse = g_coarse;\n"
        "\t\t\tg_primDesc = g_desc;\n"
        "\t\t\tprimaryHit = true;\n",
        "primary counter snapshot")
    # A primary ray that MISSES never reaches the snapshot above (march()
    # returned false and the loop broke), so take it again unconditionally
    # right before the views. tPack is computed there, which is also
    # where the view block goes.
    t = replace_once(
        t,
        "\tfloat tPack = primaryHit\n",
        "\tif (!primaryHit) {\n"
        "\t\tg_primFine = g_fine;\n"
        "\t\tg_primCoarse = g_coarse;\n"
        "\t\tg_primDesc = g_desc;\n"
        "\t}\n"
        "\tfloat tPack = primaryHit\n",
        "primary counter snapshot on miss")
    t = replace_once(
        t,
        "\t// --- diagnostic views --------------------------------------------\n",
        COUNTER_VIEWS
        + "\t// --- diagnostic views --------------------------------------------\n",
        "counter view block")
    # The 9-16 instrument block would otherwise swallow views 12-16
    # before the path ever runs. Narrow it to 9-11 in this variant.
    t = replace_once(
        t,
        "\tif (view >= 9 && view <= 16) {\n",
        "\tif (view >= 9 && view <= 11) {\n",
        "narrow instrument-A range")
    return (HDR % ("claude_trace/opengl_fragment.glsl", sha(TRACE),
                   "STEP COUNTERS + views 12-16. SLOWER than the shader "
                   "it counts for: never take a timing number here."
                   )) + t


def build_present_counters(src):
    """claude_present that hands the counter views back untouched.

    The committed present upsamples the half-resolution traced buffer
    with a 4-tap joint-bilateral filter -- correct for radiance, fatal
    for a byte plane, since the average of two byte codes is a third
    byte code that nobody counted. So views >= 11.5 take ONE nearest tap
    (the accum sampler is already NEAREST) and skip the sky
    pass-through, which would otherwise paste the raster image over
    every sky pixel's step count."""
    t = replace_once(
        src,
        "\t// full-res raster depth -> linear distance in node units.\n",
        "\t// STEP COUNTER VIEWS (12-16, measurement variant): one\n"
        "\t// nearest tap, no bilateral blend, no sky pass-through, no\n"
        "\t// tone map. The value IS the message and it is an integer.\n"
        "\tif (claudeView > 11.5) {\n"
        "\t\tgl_FragColor = vec4(clamp(texture2D(accum, uv).rgb,\n"
        "\t\t\t\t0.0, 1.0), 1.0);\n"
        "\t\treturn;\n"
        "\t}\n\n"
        "\t// full-res raster depth -> linear distance in node units.\n",
        "present counter passthrough")
    return (HDR % ("claude_present/opengl_fragment.glsl", sha(PRESENT),
                   "counter views passed through with one NEAREST tap."
                   )) + t


SAMPLER_ONLY = """		// SAMPLER-ONLY CONTROL (measurement variant, experiment E).
		// The `absent` arm deletes descendCell() -- and with it the
		// only read of claudeSubvoxTex, so the compiler strips that
		// sampler too. "Descent code absent" and "one fewer 3D
		// sampler" therefore move together in that arm, and a tax
		// caused by the SAMPLER would be indistinguishable from one
		// caused by the CODE. This arm separates them: the sampler is
		// read, from a reachable branch, with none of the inner DDA
		// compiled. claudeDescend is only ever 0 or 1, so the body
		// never runs -- and it cannot be folded away either, because
		// the condition is a uniform the compiler must respect. The
		// branch is uniform across the whole draw, so at run time the
		// hardware skips it once for every thread.
		if (claudeDescend > 1.5)
			s.rgb += texture3D(claudeSubvoxTex,
					(cell + 0.5) / vec3(64.0, 512.0, 512.0)).rrr;
"""


def build_sampleronly(src):
    t = build_nodescend(None)
    t = replace_once(
        t,
        "\t\tvec4 s = texture3D(claudeTraceGrid, (cell + 0.5) / GRID_S);\n",
        "\t\tvec4 s = texture3D(claudeTraceGrid, (cell + 0.5) / GRID_S);\n"
        + SAMPLER_ONLY,
        "sampler-only control")
    # replace the nodescend banner with this variant's own
    return t.replace(
        "DESCENT COMPILED OUT: descendCell() and both call sites deleted.",
        "DESCENT COMPILED OUT, but claudeSubvoxTex kept live behind an "
        "impossible uniform test.")


TWO_HDR = ("// GENERATED by util/claude_shader_variant.py -- DO NOT EDIT.\n"
           "// Measurement-only variant of "
           "client/shaders/claude_trace/opengl_fragment.glsl\n"
           "// PINNED: the TWO-WALK shader at %s, VERBATIM -- march()\n"
           "// plus a separate descendCell() running a second DDA. Nothing\n"
           "// is cut. This arm exists so the one-loop rewrite can be timed\n"
           "// against the code it replaces INSIDE ONE SESSION, on one\n"
           "// binary, one server and one grid. Across restarts the seat\n"
           "// drifts ~5 %% and that drift lands entirely on the comparison\n"
           "// (spec/measured.md, \"What this instrument cannot hold\n"
           "// still\"), which is exactly what a same-session arm removes.\n"
           "// Body sha256 %s\n")


def build_twowalk(_src_ignored):
    """The pre-rewrite shader, unmodified, as a measurement arm."""
    t = pre_descend_source()
    if "bool descendCell(" not in t:
        raise SystemExit("the pinned two-walk shader has no descendCell()")
    return (TWO_HDR % (PRE_DESCEND_COMMIT, sha(t)[:16])) + t


CHEAP_BIT = """	// CHEAP-BIT CONTROL (measurement variant, lean-descend 2026-08-17).
	// The one-loop rewrite deleted the second DDA's entire live state
	// and did NOT move the present-but-off tax (spec/measured.md,
	// "lean-descend experiment"). So the tax is not the second copy of
	// the walk's registers, and the next suspect in the same block of
	// code is this function's ARITHMETIC: a coarse step is one fetch
	// and two compares, while a sub-voxel test fetches and then digs a
	// bit out with floor(raw*255+0.5), exp2(mod(sc.x,8)) and mod(.,2).
	// This arm keeps the address maths and the fetch and throws the
	// bit extraction away. THE IMAGE IS WRONG HERE ON PURPOSE -- every
	// sub-voxel whose byte is nonzero reads as solid -- so this arm is
	// only ever timed with claude_descend = 0, where nothing in it
	// runs and the only thing being measured is what compiling it in
	// costs.
	return texture3D(claudeSubvoxTex,
			(texel + 0.5) / vec3(64.0, 512.0, 512.0)).r > 0.5;
"""


def build_cheapbit(src):
    """The live walk with subvoxSolid()'s bit extraction removed."""
    i = src.index("bool subvoxSolid(")
    j = src.index("{", i)
    k = src.index("\n}\n", j)
    body_start = src.index("\tfloat raw = texture3D(claudeSubvoxTex,", j)
    t = src[:body_start] + CHEAP_BIT + src[k + 1:]   # keeps the "}\n"
    # CODE lines only -- this variant's own comment quotes the maths it
    # deletes, and a guard that cannot tell code from comment would trip
    # on the sentence explaining itself.
    live = [l for l in t.splitlines()
            if not l.lstrip().startswith("//")
            and ("exp2(mod(sc.x" in l or "float byte = floor(raw" in l)]
    if live:
        raise SystemExit("cheapbit still EXTRACTS the bit:\n  "
                         + "\n  ".join(live))
    return (HDR % ("claude_trace/opengl_fragment.glsl", sha(TRACE),
                   "subvoxSolid() BIT EXTRACTION REMOVED -- fetch kept, "
                   "maths dropped. Cost arm only; the image is wrong."
                   )) + t


VARIANTS = {
    "nodescend": {"trace": build_nodescend},
    "cheapbit": {"trace": build_cheapbit},
    "twowalk": {"trace": build_twowalk},
    "sampleronly": {"trace": build_sampleronly},
    "counters": {"trace": build_counters, "present": build_present_counters},
}


def check_sources():
    got_t, got_p = sha(TRACE), sha(PRESENT)
    if got_t != TRACE_SHA or got_p != PRESENT_SHA:
        raise SystemExit(
            "the live shaders are not the ones this script was written\n"
            "against (or a variant is still installed -- run `restore`).\n"
            "  trace   got %s\n          want %s\n"
            "  present got %s\n          want %s" %
            (got_t, TRACE_SHA, got_p, PRESENT_SHA))


def cmd_build(_):
    check_sources()
    os.makedirs(VDIR, exist_ok=True)
    src_t = open(TRACE).read()
    src_p = open(PRESENT).read()
    for name, spec in VARIANTS.items():
        out = os.path.join(VDIR, "%s_trace.glsl" % name)
        open(out, "w").write(spec["trace"](src_t))
        print("%-10s trace   -> %s  (%s)" % (name, out, sha(out)[:12]))
        if "present" in spec:
            out = os.path.join(VDIR, "%s_present.glsl" % name)
            open(out, "w").write(spec["present"](src_p))
            print("%-10s present -> %s  (%s)" % (name, out, sha(out)[:12]))


def cmd_install(args):
    if os.path.exists(STATE):
        raise SystemExit("a variant is already installed (%s) -- restore first"
                         % json.load(open(STATE)).get("variant"))
    check_sources()
    spec = VARIANTS[args.variant]
    state = {"variant": args.variant, "saved": {}}
    pairs = [("trace", TRACE)] + ([("present", PRESENT)]
                                  if "present" in spec else [])
    for key, live in pairs:
        v = os.path.join(VDIR, "%s_%s.glsl" % (args.variant, key))
        if not os.path.exists(v):
            raise SystemExit("variant file missing: %s (run `build`)" % v)
        # copy2, not copyfile: claude_ci's `binary-current` assertion
        # compares bin/luanti's mtime against the newest file under src/
        # and client/, so a byte-identical rewrite with a fresh
        # timestamp reads as "the binary predates this source" and turns
        # the whole run RED. The content round-trips exactly; the mtime
        # has to as well.
        shutil.copy2(live, live + ".orig")
        state["saved"][live] = sha(live)
        shutil.copyfile(v, live)
        print("installed %s -> %s" % (v, live))
    open(STATE, "w").write(json.dumps(state, indent=1))


def cmd_restore(_):
    if not os.path.exists(STATE):
        print("nothing installed")
        return
    state = json.load(open(STATE))
    ok = True
    for live, want in state["saved"].items():
        shutil.copy2(live + ".orig", live)   # content AND mtime
        os.remove(live + ".orig")
        got = sha(live)
        print("restored %s  %s" % (live, "OK" if got == want else
                                   "MISMATCH got %s want %s" % (got, want)))
        ok = ok and got == want
    os.remove(STATE)
    if not ok:
        raise SystemExit("RESTORE DID NOT ROUND-TRIP -- fix by hand")
    check_sources()
    print("live shaders are the committed ones again")


def cmd_status(_):
    print("trace   %s %s" % (sha(TRACE), "(committed)"
                             if sha(TRACE) == TRACE_SHA else "(MODIFIED)"))
    print("present %s %s" % (sha(PRESENT), "(committed)"
                             if sha(PRESENT) == PRESENT_SHA else "(MODIFIED)"))
    print("installed: %s" % (json.load(open(STATE))["variant"]
                             if os.path.exists(STATE) else "none"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    p = sub.add_parser("install")
    p.add_argument("variant", choices=sorted(VARIANTS))
    p.set_defaults(fn=cmd_install)
    sub.add_parser("restore").set_defaults(fn=cmd_restore)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
