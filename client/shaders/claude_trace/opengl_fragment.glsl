// =====================================================================
// claude_trace — THE TRUTH RENDERER (rung 1)
// =====================================================================
//
// physics-contract.md §6: "Photo mode is the definition of correct:
// pure path tracing of emissive geometry, no next-event estimation, no
// caches, no temporal tricks, one Le, one BSDF, converged. It is slow
// and that is fine."
//
// This shader IS that definition. It replaces the five-pass chain
// (claude_radiance -> claude_faces -> claude_nfaces -> claude_accum ->
// claude_denoise) with one pass: primary ray, DDA, Lambertian bounce,
// repeat. Every estimator built after this one is judged against it and
// a disagreement is a bug in the estimator, never a reason to retune
// what is here (§6).
//
// ---------------------------------------------------------------------
// WHAT IT DOES
// ---------------------------------------------------------------------
// One sample per pixel per frame, sub-pixel jittered. 3D-DDA through the
// 128^3 occupancy volume (texture unit 10). Every non-air cell is an
// opaque Lambertian surface with a cardinal normal (§2: "cardinal
// normals are law"). Emissive classes also carry Le, and they EMIT AND
// REFLECT — one surface, both terms (§4). Cosine-weighted hemisphere
// scatter. Ray escapes the volume: contributes nothing. The frame is
// blended into the persistent history buffer under the CPU's accumAlpha,
// so a parked camera converges by 1/N to the reference image.
//
// ---------------------------------------------------------------------
// THE PUNT LIST — documented absences, not quiet hacks (§3)
// ---------------------------------------------------------------------
//  * sun/sky: DOCUMENTED PUNT, RUNG 1. A ray that leaves the 128^3
//    volume returns black. There is no sun, no sky dome, no ambient.
//    Rung 1's target scenes are a SEALED Cornell box and a SEALED
//    furnace room, where no ray escapes; outdoors this renders night.
//  * next-event estimation: absent by design (§6 forbids it in truth
//    mode). Emitters are lit by being HIT. Penumbra therefore comes
//    from an emitter's real extent (§4: "a light is not a point").
//  * irradiance caches, face caches, radiance lattice: deleted.
//  * spatial denoising: deleted. Noise is resolved by convergence only.
//  * reprojection: not done. History is read at the SAME uv. Camera
//    motion is handled entirely by accumAlpha (0.5 moving, 1.0 on
//    teleport/volume-rebase), so motion smears over ~2 frames and rest
//    converges exactly. Photo mode is a parked-camera instrument.
//  * cell sizes other than 1 m: rung 1 is 1 m only. Sub-voxel bits
//    (class 250 / unit 7) and the authored 16^3 models are NOT
//    traversed; a 250 cell is a plain opaque cube here. This is a
//    KNOWN §2 contract violation, scheduled for rung 2 — recorded
//    rather than hidden.
//  * transmissive materials: water (100), leaves (130), glass (145) are
//    opaque Lambertian in rung 1. §3 lists them as out of scope.
//  * the point-light "nub" class (165) is opaque and NON-emissive here:
//    its energy used to flow through NEE, which rung 1 does not have.
//  * specular/gloss: none, at all. §3, observed as a defect 2026-08-15:
//    "a Cornell box must have no specular response at all."
//  * textures, bevel, relief, parallax, per-block jitter, clay/gray
//    remap: all gone. Albedo is the cell's stored colour, linearised.
//  * LOD / cascades / far field: gone. Nothing exists past 128^3.
//  * KNOWN INSTRUMENT SIDE-EFFECT: views 1-4 write their deterministic
//    image into the ping-pong history, so switching back to view 0 at a
//    parked camera leaves that frame inside the running average for a
//    long time (accumAlpha is ~1/N at rest). Reset it the way the CPU
//    already knows how: move the camera, or re-run a vantage teleport
//    (>20 nodes forces accumAlpha = 1.0). Deliberately NOT papered over
//    with a hidden view-change detector.
//
// ---------------------------------------------------------------------
// NO HIDDEN CONSTANTS
// ---------------------------------------------------------------------
// Every literal that changes an image is named as a const below with the
// reason it holds that value. The only unnamed numbers left are hash
// magic (Dave Hoskins' constants, arbitrary by construction) and the
// 2.2 display gamma of the stored colour bytes.
//
// ---------------------------------------------------------------------
// GLSL PROFILE
// ---------------------------------------------------------------------
// No #version here: the engine prepends "#version 150" on the desktop
// core profile (src/client/shader.cpp), plus "#define texture2D texture"
// and "#define gl_FragColor outFragColor". Same conventions claude_accum
// used. texture3D is not covered by that header, so it is defined below
// exactly as claude_accum defined it. GL 4.1 core: no image store, no
// compute, no layout(location=) on fragment outputs — none used here.
// =====================================================================

#define history texture0

uniform sampler2D history;      // previous frame's accumulated radiance
uniform sampler3D claudeVolume; // unit 10: RGBA8 128^3, rgb = cell colour,
                                // a = class byte / 255
uniform vec2 texelSize0;        // one texel of the trace-res target
uniform lowp float volumeDebug; // pipeline master switch: <2.5 = raster

uniform vec3 volumeCamPos;   // camera in volume-local node units
uniform vec3 volumeCamFwd;   // unit look direction
uniform vec3 volumeCamRight; // camera right, pre-scaled by tan(fovX/2)
uniform vec3 volumeCamUp;    // camera up, pre-scaled by tan(fovY/2)

uniform float animationTimer; // seconds; per-frame RNG decorrelation
uniform lowp float accumAlpha; // CPU: 1.0 hard reset, 0.5 moving,
                               // 1/(2+still_frames) at rest (min 0.02)

// DIAGNOSTIC SUITE (claude_view). 0 = photo, and the photo path must
// stay the truth renderer — no debug term is evaluated inside the path
// loop, the views only read facts the loop already recorded.
//   0 photo | 1 normal ladder | 2 albedo | 3 Le | 4 distance | 5 bounces
//   6 clay: the photo path with every reflectance clamped to CLAY_RHO —
//   uniform albedo shows pure transport (the ray-traced "wireframe").
//   Emission keeps its true Le; only rho is clamped. Accumulates and
//   tonemaps exactly like view 0.
// Views are GRAY LADDERS, not RGB: John is colorblind, and cardinal
// normals mean there are only six possible normals, so a wrong normal
// reads as a wrong BRIGHTNESS patch. See VIEW_N_* below for the ladder.
uniform float claudeView;
// Path-depth cap. 0 = primary emission only (you see only what emits),
// 1 = direct light only, 24 = full transport. The direct/indirect
// separation switch — no extra view modes needed.
uniform float claudeBounces;

CENTROID_ VARYING_ mediump vec2 varTexCoord;

#if __VERSION__ >= 130
#define texture3D texture
#endif

// ---------------------------------------------------------------------
// NAMED CONSTANTS
// ---------------------------------------------------------------------

// The volume is 128 cells on a side (game.cpp claudeVolumeSnapshot).
const float VOL_S = 128.0;

// A ray crosses at most one cell boundary per DDA step, and at most 128
// boundaries per axis, so 3*128 is the exact worst case for a diagonal
// crossing of the volume. Any smaller bound would silently truncate a
// ray and darken the image, which is the one failure a truth renderer
// may not have.
const int MARCH_STEPS = 384;

// Hard path-depth cap. claudeBounces is clamped to this range CPU-side;
// the loop bound must be a compile-time constant anyway.
const int BOUNCE_CAP = 24;

// Russian roulette begins at this path segment: segments 0..RR_START-1
// always survive, so the first four bounces are termination-noise-free
// and RR starts *after bounce 4*. RR is unbiased (survivors are divided
// by their survival probability), which is why the cap can be 24 rather
// than the old 4 — MEASURED: a 4-bounce cap loses about a fifth of the
// light at rho = 0.73, which is exactly the Cornell white wall.
const int RR_START = 5;

// RR survival probability is the path throughput, clamped. The floor
// keeps a dark path from being killed with certainty (which would bias
// dark materials to black); the ceiling guarantees SOME termination
// probability so the expected path length stays finite.
const float RR_Q_MIN = 0.05;
const float RR_Q_MAX = 0.95;

// Numerical albedo floor. NOT an aesthetic lift: pow(0,2.2) is 0 and a
// zero-albedo surface makes throughput exactly zero, which is fine, but
// it also makes the RR clamp the only thing keeping the loop alive.
// This is claude_accum's pathAlbedo() floor, preserved verbatim so the
// furnace referee's rho stays the same number it always was.
const float ALBEDO_FLOOR = 0.005;

// CLASS BANDS (game.cpp claudeVolumeSnapshot writes the class byte into
// the volume's alpha; the shader sees byte/255).
//   0 air | 100 water | 130 leaves | 145 glass | 165 point-light nub
//   170..240 emissive as 170 + light_source*5 | 250 authored model
//   255 solid
// Air test: claude_accum's own guard band. There is no class between 0
// and 100, so this splits air from everything else exactly.
const float CLASS_AIR_MAX = 0.25;
// Emissive band, taken at the HALF-BYTE midpoints either side of the
// 170..240 run so an 8-bit texture round-trip cannot move a cell across
// the edge. 167.5/255 sits between the nub (165) and the first emissive
// class (170); 245/255 sits between the last emissive class (240) and
// the authored-model class (250).
const float CLASS_EMIT_LO = 167.5 / 255.0;
const float CLASS_EMIT_HI = 245.0 / 255.0;

// THE EMISSION LAW (ADR-0009 #1, claude_accum emitStrength()). ONE Le,
// used by primary rays, bounce rays and every future consumer:
//     e  = clamp((class/255 - EMIT_E_BIAS) / EMIT_E_SPAN, 0, 1)
//     Le = albedo * (EMIT_BASE + EMIT_GAIN * e)
// The band maps light_source 0..14 onto e ~ 0.057..1.0. Cross-checked
// against util/claude_furnace_check.py, which asserts exactly
// le = rho * (0.4 + 2.0 * e) with e = 1 for class 240 — so the analytic
// referee L = Le/(1-rho) stays valid against this shader unchanged.
const float EMIT_E_BIAS = 0.65;
const float EMIT_E_SPAN = 0.29;
const float EMIT_BASE = 0.4;
const float EMIT_GAIN = 2.0;

// Depth packing for claude_present's joint-bilateral upsample and its
// sky test: alpha carries tHit/DEPTH_SCALE, and a miss sits at the top
// of the range. claude_present reads the same 4096; do not change one
// without the other.
const float DEPTH_SCALE = 4096.0;
const float DEPTH_MISS = 4096.0;
const float DEPTH_MAX_HIT = 4090.0;

// Ray restart offset off a surface, in cell units. The DDA advances
// before it tests, so the starting cell is never re-hit; this only
// keeps floor() on the correct side of the face.
const float SURFACE_EPS = 0.01;

// Guard against a division blowing up on an axis-aligned ray.
const float DDA_MIN_ABS = 1e-6;

const float PI2 = 6.28318530717958647692;

// --- diagnostic view constants ---------------------------------------
// Six cardinal normals, six gray steps. Read the image as brightness:
// a face showing 0.60 where 1.00 belongs is a +Z normal on a +Y face.
const float VIEW_N_PY = 1.00; // +Y  (up)
const float VIEW_N_PX = 0.75; // +X
const float VIEW_N_PZ = 0.60; // +Z
const float VIEW_N_NX = 0.45; // -X
const float VIEW_N_NZ = 0.30; // -Z
const float VIEW_N_NY = 0.15; // -Y  (down)
// Le display gain for view 3. The brightest rung-1 emitter is
// albedo*(0.4+2*1) <= 2.4, so a third of it stays below clip.
const float VIEW_EMIT_SCALE = 1.0 / 3.0;
// Distance display range for view 4: the volume's body diagonal,
// 128*sqrt(3), log-scaled so near geometry is not all one black step.
const float VIEW_DIST_MAX = 221.7;

// ---------------------------------------------------------------------
// RNG — Dave Hoskins' sine-free hashes
// ---------------------------------------------------------------------
// claude_accum seeded its path directions from the HIT POSITION, so two
// different paths landing on the same point drew the same direction —
// a correlation a truth renderer cannot carry. This one is seeded per
// pixel per frame and advanced per draw, so every dimension of every
// path is independent.
//
// Sine-free on purpose: fract(sin(x)) degrades badly once x is large,
// and gl_FragCoord * frame gets large.

float hash11(float p)
{
	p = fract(p * 0.1031);
	p *= p + 33.33;
	p *= p + p;
	return fract(p);
}

float hash13(vec3 p3)
{
	p3 = fract(p3 * 0.1031);
	p3 += dot(p3, p3.zyx + 31.32);
	return fract((p3.x + p3.y) * p3.z);
}

float g_rngState;

// NOTE: every call site assigns to a named local first. GLSL does not
// define the evaluation order of constructor/function arguments, so
// vec2(rnd1(), rnd1()) would be a real (silent, driver-specific) bug.
float rnd1()
{
	// The golden-ratio increment spreads successive states across the
	// hash's input range instead of walking one neighbourhood.
	g_rngState = hash11(g_rngState + 0.61803398875);
	return g_rngState;
}

// ---------------------------------------------------------------------
// MATERIAL
// ---------------------------------------------------------------------

// Stored cell colour -> linear albedo rho. This is claude_accum's
// pathAlbedo() with the grayWorld/clay remap removed (rung 1 has no
// material dials), so it is exactly what claude_furnace_check.py
// computes: rho = (stored/255)^2.2.
vec3 cellAlbedo(vec3 raw)
{
	return max(pow(raw, vec3(2.2)), vec3(ALBEDO_FLOOR));
}

// Clay mode (view 6): the one reflectance every surface gets. Mid-gray,
// chosen to match the gray186 lab material (rho 0.5) so clay frames are
// directly comparable to the furnace-050 room. Applied AFTER emission is
// derived, so lights keep their true Le.
const vec3 CLAY_RHO = vec3(0.5);

// THE EMISSION LAW. An emissive voxel is a surface with BOTH Le and rho
// (§4) — the caller adds this and then continues the path with rho,
// which is what makes L = Le/(1-rho) expressible in a sealed room.
vec3 cellEmission(float cls, vec3 albedo)
{
	if (cls <= CLASS_EMIT_LO || cls >= CLASS_EMIT_HI)
		return vec3(0.0);
	float e = clamp((cls - EMIT_E_BIAS) / EMIT_E_SPAN, 0.0, 1.0);
	return albedo * (EMIT_BASE + EMIT_GAIN * e);
}

// ---------------------------------------------------------------------
// TRAVERSAL — plain branchless 3D-DDA, no acceleration structure
// ---------------------------------------------------------------------
// Deliberately NOT using the occupancy MIP pyramid (unit 11). Rung 1 is
// the reference, its scenes are two small sealed rooms, and "slow is
// fine". One traversal, no second code path to keep in agreement.
//
// Returns true on an opaque hit and fills hp / n / alb / le / tHit.
// False = the ray left the volume (or ran out of steps, which the
// MARCH_STEPS bound makes unreachable inside a 128^3 grid).

bool march(vec3 ro, vec3 rd, out vec3 hp, out vec3 n, out vec3 alb,
		out vec3 le, out float tHit)
{
	hp = ro;
	n = vec3(0.0, 1.0, 0.0);
	alb = vec3(0.0);
	le = vec3(0.0);
	tHit = DEPTH_MISS;

	vec3 cell = floor(ro);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(DDA_MIN_ABS));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	float t = 0.0;
	int axis = -1;

	for (int i = 0; i < MARCH_STEPS; i++) {
		// advance to the next cell boundary, then test the cell we
		// entered: the starting cell is never tested, which is also
		// what keeps a bounce ray off its own surface
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x; sideDist.x += invRd.x;
			cell.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y; sideDist.y += invRd.y;
			cell.y += stepDir.y; axis = 1;
		} else {
			t = sideDist.z; sideDist.z += invRd.z;
			cell.z += stepDir.z; axis = 2;
		}

		if (any(lessThan(cell, vec3(0.0)))
				|| any(greaterThanEqual(cell, vec3(VOL_S))))
			return false; // escaped: contributes nothing (sky is punted)

		vec4 s = texture3D(claudeVolume, (cell + 0.5) / VOL_S);
		if (s.a <= CLASS_AIR_MAX)
			continue; // air

		// every other class is one thing: an opaque Lambertian surface
		// with a cardinal normal, which may also emit
		n = vec3(0.0);
		if (axis == 0) n.x = -stepDir.x;
		else if (axis == 1) n.y = -stepDir.y;
		else n.z = -stepDir.z;
		hp = ro + rd * t + n * SURFACE_EPS;
		alb = cellAlbedo(s.rgb);
		le = cellEmission(s.a, alb);
		tHit = t;
		return true;
	}
	return false;
}

// Cosine-weighted hemisphere direction about a cardinal normal n.
// Exact (Malley's method), not the normalize(n + cube_point) shortcut
// claude_accum used, which is not uniform on the sphere and so is not
// cosine-weighted. With pdf = cos/PI and a Lambertian BRDF of rho/PI,
// the estimator weight is exactly rho — which is why the path loop's
// only throughput update is `tp *= alb`.
vec3 cosineHemisphere(vec3 n, float u1, float u2)
{
	float r = sqrt(u1);
	float phi = PI2 * u2;
	// cardinal normals make the tangent frame trivial and degenerate-free
	vec3 t = abs(n.y) > 0.5 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	vec3 tx = normalize(cross(t, n));
	vec3 ty = cross(n, tx);
	return normalize(tx * (r * cos(phi)) + ty * (r * sin(phi))
			+ n * sqrt(max(0.0, 1.0 - u1)));
}

// ---------------------------------------------------------------------

void main(void)
{
	vec2 uv = varTexCoord.st;
	if (volumeDebug < 2.5) {
		// traced mode off: carry history through untouched
		gl_FragColor = texture2D(history, uv);
		return;
	}

	int view = int(claudeView + 0.5);
	int maxBounces = int(claudeBounces + 0.5);

	g_rngState = hash13(vec3(gl_FragCoord.xy,
			// animationTimer is unbounded seconds; wrapped by an
			// irrational multiplier so consecutive frames land far
			// apart in the hash's domain and no frame rate aliases it
			fract(animationTimer * 91.7) * 1024.0));

	// sub-pixel jitter: free anti-aliasing through the running average.
	// Off in the diagnostic views, which do not accumulate and would
	// otherwise flicker along every silhouette.
	float j0 = rnd1();
	float j1 = rnd1();
	vec2 jit = (vec2(j0, j1) - 0.5) * texelSize0;
	if (view != 0 && view != 6)
		jit = vec2(0.0);

	vec2 ndc = (uv + jit) * 2.0 - 1.0;
	vec3 rd = normalize(volumeCamFwd
			+ ndc.x * volumeCamRight + ndc.y * volumeCamUp);
	// +0.5: volumeCamPos is node-CENTRED (game.cpp: cam/BS - origin),
	// the DDA works in cell-index space where cell i spans [i, i+1)
	vec3 ro = volumeCamPos + 0.5;

	// --- the path ----------------------------------------------------
	vec3 L = vec3(0.0);
	vec3 tp = vec3(1.0);
	vec3 p = ro;
	vec3 dir = rd;

	float primaryT = DEPTH_MISS; // for the depth channel + view 4
	vec3 primaryN = vec3(0.0);   // view 1
	vec3 primaryAlb = vec3(0.0); // view 2
	vec3 primaryLe = vec3(0.0);  // view 3
	bool primaryHit = false;
	float pathBounces = 0.0;     // view 5: scatters actually taken

	for (int seg = 0; seg <= BOUNCE_CAP; seg++) {
		if (seg > maxBounces)
			break;

		vec3 hp, n, alb, le;
		float tHit;
		if (!march(p, dir, hp, n, alb, le, tHit))
			break; // escaped the volume: nothing to add

		// clay: march computed le from the TRUE albedo above; clamping
		// rho afterward changes reflectance only, never the lights
		if (view == 6)
			alb = CLAY_RHO;

		if (seg == 0) {
			primaryHit = true;
			primaryT = tHit;
			primaryN = n;
			primaryAlb = alb;
			primaryLe = le;
		}

		// §4: the surface EMITS and REFLECTS. Collect Le, keep going.
		L += tp * le;

		if (seg == maxBounces)
			break; // depth cap: no scatter from this vertex
		if (view >= 1 && view <= 4)
			break; // first-hit views need nothing past the primary

		tp *= alb;

		// Russian roulette, unbiased: survivors carry 1/q.
		if (seg + 1 >= RR_START) {
			float q = clamp(max(tp.r, max(tp.g, tp.b)),
					RR_Q_MIN, RR_Q_MAX);
			float u = rnd1();
			if (u > q)
				break;
			tp /= q;
		}

		float u1 = rnd1();
		float u2 = rnd1();
		dir = cosineHemisphere(n, u1, u2);
		p = hp;
		pathBounces += 1.0;
	}

	float tPack = primaryHit
			? min(primaryT, DEPTH_MAX_HIT) / DEPTH_SCALE : 1.0;

	// --- diagnostic views --------------------------------------------
	// Deterministic and un-accumulated (1-4); view 5 averages, because a
	// MEAN bounce count is the meaningful quantity. claude_present
	// passes 1-5 through linearly — no ACES, no gamma. View 6 (clay) is
	// lit radiance: it skips this block and accumulates/tonemaps as photo.
	if (view != 0 && view != 6) {
		vec3 dbg = vec3(0.0);
		if (view == 1) {
			float g = 0.0;
			if (primaryHit) {
				if (primaryN.y > 0.5) g = VIEW_N_PY;
				else if (primaryN.y < -0.5) g = VIEW_N_NY;
				else if (primaryN.x > 0.5) g = VIEW_N_PX;
				else if (primaryN.x < -0.5) g = VIEW_N_NX;
				else if (primaryN.z > 0.5) g = VIEW_N_PZ;
				else g = VIEW_N_NZ;
			}
			dbg = vec3(g);
		} else if (view == 2) {
			// raw stored cell colour, unlit and un-linearised
			dbg = primaryHit ? pow(primaryAlb, vec3(1.0 / 2.2)) : vec3(0.0);
		} else if (view == 3) {
			dbg = primaryLe * VIEW_EMIT_SCALE;
		} else if (view == 4) {
			float g = 0.0;
			if (primaryHit)
				g = log2(1.0 + primaryT) / log2(1.0 + VIEW_DIST_MAX);
			dbg = vec3(clamp(g, 0.0, 1.0));
		} else {
			dbg = vec3(pathBounces / max(float(maxBounces), 1.0));
		}
		if (view != 5) {
			gl_FragColor = vec4(dbg, tPack);
			return;
		}
		L = dbg; // view 5 falls through into the accumulator
	}

	// --- accumulate ---------------------------------------------------
	// history is LINEAR radiance in a 16F target; averaging must happen
	// in linear light or jittered noise converges biased dark.
	// accumAlpha is the whole reset mechanism: 1.0 on teleport or a
	// volume rebase (history discarded), 0.5 while moving, 1/(2+N) at
	// rest — a true running average, so a parked camera converges by
	// 1/N rather than sitting at an EMA's perpetual noise floor.
	vec3 fresh = max(L, vec3(0.0));
	vec3 prev = fresh;
	float a = 1.0;
	if (accumAlpha < 0.999) {
		vec4 h = texture2D(history, uv);
		// An uninitialised (or once-NaN) history texel must not poison
		// the average forever — mix() propagates NaN, and abs(NaN)<x is
		// false, so this rejects both NaN and Inf.
		if (all(lessThan(abs(h.rgb), vec3(1e6)))) {
			prev = max(h.rgb, vec3(0.0));
			a = accumAlpha;
		}
	}

	gl_FragColor = vec4(mix(prev, fresh, a), tPack);
}
