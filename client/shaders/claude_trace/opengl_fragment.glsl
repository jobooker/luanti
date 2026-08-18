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
// 128^3 occupancy grid (texture unit 10). Every non-air cell is an
// opaque Lambertian surface with a cardinal normal (§2: "cardinal
// normals are law"). Emissive classes also carry Le, and they EMIT AND
// REFLECT — one surface, both terms (§4). Cosine-weighted hemisphere
// scatter. Ray escapes the grid: contributes nothing. The frame is
// blended into the persistent history buffer under the CPU's accumAlpha,
// so a parked camera converges by 1/N to the reference image.
//
// ---------------------------------------------------------------------
// RUNG 2 — NEXT-EVENT ESTIMATION WITH MIS (claudeNee)
// ---------------------------------------------------------------------
// claudeNee = 0 is the rung-1 path above, unchanged: no light sampling,
// no MIS weight, not one extra RNG draw. That is photo mode, it is the
// definition of correct (§6), and it is the A/B partner of everything
// below. claudeNee = 1 turns on the estimator described here, which must
// agree with it IN EXPECTATION at a parked camera. A disagreement is a
// bug in this block, never a reason to retune the pure path (§6).
//
// THE INTEGRAL. At a vertex x with normal n_x and Lambertian BRDF
// f_r = rho/PI, outgoing radiance is
//
//     L(x) = Le(x) + INT_H f_r * L(x,wi) * cos_x dwi
//
// Rung 1 estimates the integral with one strategy: sample wi from the
// cosine hemisphere and recurse. Rung 2 adds a second: pick a point y on
// an emissive surface and connect. Both estimate the SAME integral, so
// adding them at full strength would double the direct light — the
// classic bug, and the one the old renderer's claude_pure comment was
// written in the blood of.
//
// THE TWO PDFs, both expressed in SOLID ANGLE about x so they can be
// compared:
//
//   BSDF sampling   p_b(wi) = cos_x / PI            [cosineHemisphere()]
//   light sampling  p_l(wi) = p_A(y) * d^2 / cos_y  [area -> solid angle]
//
// where d = |y - x| and cos_y = dot(n_y, -wi) at the light. The light
// sampler is UNIFORM OVER AREA (§4: "a light is not a point" — extent is
// what makes penumbra, and a point light casts no soft shadow at any
// quality setting):
//
//   p_A(y) = 1 / (N * k)
//
// N = claudeAreaCount emissive cells in the list; k = the number of
// faces of the chosen cell that are BOTH air-exposed (game.cpp's mask)
// AND turned toward x. Each face is a unit square, so its area is 1 and
// the area density on a chosen face is 1. Choosing among only the faces
// that face x is not an optimisation with a hidden cost: it is a change
// of p_A, and neePdfSa() below computes the identical quantity for the
// BSDF side, which is what keeps the two consistent.
//
// THE WEIGHTS — balance heuristic, one sample per strategy:
//
//   w_l = p_l / (p_l + p_b)        added at the vertex, by neeDirect()
//   w_b = p_b / (p_b + p_l)        added at the NEXT vertex's Le
//
// w_l + w_b = 1 for every direction both strategies can produce, so the
// sum is exactly one copy of the direct term. Where only one strategy
// can produce a direction the other's pdf is 0 and that strategy's
// weight is 1, which is what makes the following all correct rather than
// merely tolerable:
//
//  * AN EMITTER NOT IN THE LIST (the 16-slot cap overflowed, or the cell
//    is sealed inside solid). p_l = 0 there, so w_b = 1 and its full
//    radiance arrives through BSDF sampling. Truncating the list costs
//    variance, never energy.
//  * N = 0, or claudeNee = 0. p_l = 0 everywhere, w_b = 1 everywhere:
//    the estimator collapses back onto the pure path, exactly.
//  * A UNIFORM-EMISSIVE FURNACE. Every surface emits, so nearly every
//    BSDF bounce lands on an emitter and both strategies fire on the
//    same Watt every time. This is the case that punishes a missing
//    weight hardest and the reason the analytic L = Le/(1-rho) referee
//    is the sharpest instrument here.
//
// AT AN E-SURFACE. An emissive voxel emits AND reflects (§4), so its Le
// is collected at every vertex that lands on it, and the surface then
// scatters with the same rho as any other. The only difference rung 2
// makes is the weight on that Le:
//   - seg 0 (the camera ray): weight 1, ALWAYS. A camera ray is not a
//     sampling strategy the light sampler competes with — nothing did or
//     could aim at that surface on the eye's behalf. Both modes add it
//     identically, which is why claudeBounces = 0 is byte-identical
//     under either dial.
//   - seg > 0: weight w_b, using the pdfs of the ray that arrived.
// Le itself is never recomputed: the shadow ray's own march() hit
// carries it, through the same cellEmission() the eye ray runs. There is
// one Le in this file and NEE does not add a second (§4).
//
// WHAT NEE DOES NOT CHANGE. Russian roulette still runs after the NEE
// term (survivors carry 1/q, so E[BSDF half] is untouched). The depth
// cap still truncates both halves at the same vertex, so claudeBounces
// means the same thing in both modes. Views 1-5 are untouched; view 6
// (clay) runs whichever transport the dial says, and clay+nee is a legal
// and useful combination — uniform rho makes a bad MIS weight glaring.
//
// ---------------------------------------------------------------------
// THE SKY IS A MISS FUNCTION (roadmap coverage 3, 2026-08-17)
// ---------------------------------------------------------------------
// skyRadiance(direction) is ONE function with TWO roles, and §4's one-Le
// law is why it may not be two:
//
//   1. BACKGROUND — what a camera ray sees when it leaves the 128^3 grid.
//   2. LIGHT      — what a bounce ray collects when it leaves, and what
//                   NEE importance-samples like any other emitter.
//
// The sky a camera ray sees and the sky a shadow ray samples are the same
// evaluation of the same function, so the image cannot be inconsistent
// with its own lighting. Until today the DDA's escape branch returned
// BLACK: there was no sun, no sky dome, no ambient, and volumeSunDir was
// consumed zero times by this shader. Outdoors rendered night.
//
// IT IS FED FROM LUANTI'S OWN SKY, not from an analytic stand-in
// (game.cpp, src/client/sky.cpp): the horizon and zenith colours the Sky
// class computes for this time of day, the sun's and the moon's real
// directions, their real drawn angular sizes, and — for the moon — the
// game's own sprite, which under Mineclonia is mcl_moon's phase-correct
// 8-frame sheet seeded from the world seed. THE BLUE-SUN INCIDENT is why
// (claude_present carried the note until today): substituting an analytic
// disc drew the moon as a blue SUN, because at night the engine's
// "sun direction" IS the moon's and the disc reused the sun's multiplier.
// A sun and a moon are two bodies here, with two directions, two sizes,
// two radiances and one of them textured — never one body with a dial.
//
// WHAT IS STILL ABSENT, and it is an absence rather than a hack (§3):
//  * CLOUDS AND STARS. They were never lights and they are not drawn.
//    Before today claude_present pasted the raster sky (clouds, stars,
//    sunrise glow) wherever raster depth was empty; that composite is
//    gone, because a background the trace does not own is a second sky.
//  * THE FAR FIELD past 128^3 is one number, skyGroundCol: a flat
//    Lambertian ground at a typical terrain albedo, lit by this frame's
//    dome and body. No shape, no colour variation, no distance. It
//    exists because black below the horizon drew a black band across the
//    horizon of every outdoor frame, and because the ground really does
//    bounce light back up.
//  * Aerial perspective / scattering along the ray: §3, out of scope.
//
// ---------------------------------------------------------------------
// THE PUNT LIST — documented absences, not quiet hacks (§3)
// ---------------------------------------------------------------------
//  * sun/sky: LANDED 2026-08-17, see the block above. What remains is
//    listed there: clouds, stars, aerial perspective, and a far field
//    that is one flat number instead of a world.
//  * NEE beyond 16 area emitters: the uniform list is capped (game.cpp
//    ClaudeTraceGrid::AREA_CAP). Past that, emitters are lit by being HIT,
//    which is correct and noisier — see the weights above.
//  * NEE for the point-light list (claudeEmitter0..7): NOT connected.
//    Those are class 165/250 cells, which cellEmission() gives no Le, so
//    there is nothing to sample and nothing to double-count.
//  * irradiance caches, face caches, radiance lattice: deleted.
//  * spatial denoising: deleted. Noise is resolved by convergence only.
//  * reprojection: not done. History is read at the SAME uv. Camera
//    motion is handled entirely by accumAlpha (0.5 moving, 1.0 on
//    teleport/grid-rebase), so motion smears over ~2 frames and rest
//    converges exactly. Photo mode is a parked-camera instrument.
//  * cell sizes other than 1 m: LANDED 2026-08-16 for the sub-voxel
//    ring. march() descends into a class-250 cell and continues the
//    same DDA through its 16^3 mask at 1/16 m (unit 7), so stairs,
//    slabs, beds and every authored model are real sub-metre geometry
//    for EVERY ray type — eye, bounce and shadow — because there is
//    still exactly one traversal. What remains punted:
//      - outside the ring [48,80)^3 the mask does not exist, so a 250
//        cell is still a 1 m cube. That is a §2 ladder (a coarser rung
//        at distance), which §7 permits; it is not a different light
//        law.
//      - SUB-VOXEL CELLS EMIT NOTHING. cellEmission() returns zero for
//        cls >= CLASS_EMIT_HI = 245/255 and 250 is above it, so the
//        campfire, the lantern and every modelled torch now have their
//        true SHAPE and no glow. They had no glow before this change
//        either. The emission data exists (the model palette carries
//        emit/15 in its alpha) but wiring it in changes the DOMAIN of
//        the one emission law, and §4 admits exactly one law — that is
//        a spec decision, not a shader edit. Deliberately not done
//        here, and deliberately not worked around with a second
//        emission path.
//      - per-sub-voxel COLOUR is likewise not read. A sub-voxel hit
//        takes its cell's stored colour, so a chest is chest-shaped in
//        one albedo. claudeModelAtlas / claudeModelPal stay unread.
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
//
// The rung-2 block uses integer bitwise ops (>>, &) on the face mask.
// Those are core GLSL since 1.30, so 150 has them; they are the only
// construct in this file newer than what rung 1 used.
//
// NO ARRAY UNIFORMS AND NO DYNAMIC INDEXING. claudeArea0..15 are sixteen
// separate vec4 uniforms read through an explicit if-chain, exactly the
// shape claudeEmitter0..7 has always had. `uniform vec4 a[16]` would be
// legal GLSL 150 to index dynamically, but the ENGINE side is the
// problem: getPixelShaderConstantID() string-compares against the name
// glGetActiveUniform returns, and drivers disagree on whether that is
// "a" or "a[0]" — so the array would resolve on one machine and
// silently deliver nothing on the next, which for a light list means an
// image that is quietly noisier rather than an error.
// =====================================================================

#define history texture0

uniform sampler2D history;      // previous frame's accumulated radiance
uniform sampler3D claudeTraceGrid; // unit 10: RGBA8 128^3, rgb = cell colour,
                                // a = class byte / 255
// THE SUB-VOXEL RING, and until 2026-08-16 nothing read it. unit 7,
// R8 64x512x512, one bit per 1/16 m voxel for the 32^3 cell ring at
// grid-local [48,80)^3. Layout is game.cpp claudeTraceGridBakeSubvox,
// verbatim, and it is the SAME 512-byte mask an authored model and a
// converted node_box both produce (src/client/claude_nodebox.h):
//
//   texel.x = (cell.x-48)*2 + (sx >> 3)      64 wide, one byte = 8 sx
//   texel.y = (cell.y-48)*16 + sy
//   texel.z = (cell.z-48)*16 + sz
//   bit     = 1 << (sx & 7)
//
// R8 read as a float and unpacked with floor/mod, NOT a usampler3D:
// an integer sampler silently kills the Irrlicht material
// (environment-laws.md, the flat-blue outage).
uniform sampler3D claudeSubvoxTex;
// 1 = march() descends into class-250 cells (the default); 0 = the
// pre-2026-08-16 behaviour, in which a 250 cell is an opaque 1 m cube.
// The A/B partner for the energy and cost gates.
uniform float claudeDescend;

// --- THE SKY, as Luanti's own Sky class computes it this frame ---------
// All colours are LINEAR radiance. game.cpp linearises the engine's
// stored sky bytes with the SAME (c/255)^2.2 law cellAlbedo() uses for a
// cell's colour — this renderer has exactly one colour-byte law, and the
// sky does not get a second one.
uniform vec3 skyHorizonCol;  // Sky::getBgColor(),  the dome at y = 0
uniform vec3 skyZenithCol;   // Sky::getSkyColor(), the dome at y = 1
// THE FAR FIELD, below the horizon, and it is the crudest one there is:
// a flat Lambertian ground of a typical terrain albedo, lit by the dome
// and the body, computed CPU-side so it cannot disagree with them. It is
// not decoration — the 128^3 grid ends long before the world does, so
// every ray that leaves it heading even slightly downward was returning
// BLACK and painting a black band along the horizon of every outdoor
// frame. Zero under the uniform test sky, which keeps the analytic
// referee exact.
uniform vec3 skyGroundCol;
// The two BODIES. Each is a cone about its direction; cos = 2.0 is the
// "not in the sky" sentinel (no direction can satisfy dot >= 2), so a
// body below the horizon or switched off by the game costs one compare
// and contributes nothing — no branch keyed on time of day lives here.
uniform vec3 skySunDir;
uniform vec3 skySunCol;
uniform float skySunCos;
uniform vec3 skyMoonDir;
uniform vec3 skyMoonCol;
uniform float skyMoonCos;
// The moon quad's tangent frame (Sky::place_sky_body's own rotation), so
// the sprite can be sampled in the right orientation and the phase reads
// as a phase.
uniform vec3 skyMoonU;
uniform vec3 skyMoonV;
uniform float skyMoonTexOn;  // 1 = claudeMoonTex is live, 0 = flat disc
uniform sampler2D claudeMoonTex; // unit 19: the game's current moon sprite
// TEST DIAL (claude_sky_uniform). > 0 replaces the whole sky with a
// CONSTANT radiance of this value in every direction — no sun, no moon,
// no gradient. It exists for one referee: an unoccluded Lambertian plane
// under a uniform sky reads L = rho * L_sky exactly, an ANALYTIC number,
// and it is the only instrument in this repo that can see a constant
// factor error in sky radiance (sealed rooms see no sky at all, and a
// uniformly hot outdoors just looks like a bright day).
uniform float claudeSkyUniform;

uniform vec2 texelSize0;        // one texel of the trace-res target
uniform lowp float gridDebug; // pipeline master switch: <2.5 = raster

// WORLD node coords of grid cell (0,0,0). game.cpp sets it in the same
// block as gridCamPos, so it is live whenever the trace runs. Used ONLY
// by the roadmap-1a instrument views (9/10/11), which have to name a
// world-space rectangle in the DDA's cell-index space.
uniform vec3 gridOrigin;
uniform vec3 gridCamPos;   // camera in grid-local node units
uniform vec3 gridCamFwd;   // unit look direction
uniform vec3 gridCamRight; // camera right, pre-scaled by tan(fovX/2)
uniform vec3 gridCamUp;    // camera up, pre-scaled by tan(fovY/2)

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
//   17: THE SKY ITSELF — skyRadiance() charted over the full sphere as
//   a lat-long image, no geometry and no transport. The isolating
//   instrument for anything that goes wrong with the miss function.
//   9/10/11: INSTRUMENT A of roadmap step 1a — three independent
//   estimates of ONE quantity, the DIRECT-light radiance leaving the
//   primary hit. See the block above panelDirect() for the whole design.
//   They are inert at claude_view 0 and cost the photo path nothing.
// Views are GRAY LADDERS, not RGB: John is colorblind, and cardinal
// normals mean there are only six possible normals, so a wrong normal
// reads as a wrong BRIGHTNESS patch. See VIEW_N_* below for the ladder.
uniform float claudeView;
// Path-depth cap. 0 = primary emission only (you see only what emits),
// 1 = direct light only, 24 = full transport. The direct/indirect
// separation switch — no extra view modes needed.
uniform float claudeBounces;
// Transport mode. 0 = the pure photo path (§6 truth mode: no light
// sampling, no MIS weight, not one extra RNG draw). 1 = next-event
// estimation with multiple importance sampling. See the rung-2 header.
uniform float claudeNee;
// RNG SOURCE. 1 (DEFAULT since roadmap 1a) = the counter-based PCG
// below, keyed on (pixel, frame, draw index). 0 = the rung-1 hash CHAIN
// (g_rngState = hash11(g_rngState + phi)) — kept reachable for one
// release as the A/B partner, and it is THE OLD, BIASED ONE.
//
// An iterated float hash is a trajectory, not a generator. Whatever
// structure its successive outputs carry becomes a DETERMINISTIC bias in
// every estimator that consumes them, and this one biased
// cosineHemisphere() by ~9% against a compact overhead light. The
// marginals looked fine (E[u1] = 0.4993, E[sqrt(1-u1)] = 0.6643 vs
// 0.6667, P(u1<0.1) = 0.1010) — it is JOINT structure between successive
// draws, which no test of one draw at a time can see.
//
// It reproduced to four decimals on two separate seats, which is what
// made it look like an estimator bug rather than noise. It was in PHOTO
// MODE: the pure path is nothing but this sampler, so the truth renderer
// itself ran 1-8% dark in Cornell and the "NEE is hot" finding was the
// estimator being RIGHT. spec/measured.md "1a".
uniform float claudeRng;

// AREA-EMITTER LIST for next-event estimation (game.cpp
// claudeTraceGridSnapshot, ClaudeTraceGrid::area). One emissive CELL per slot:
//   xyz = integer grid-cell coords, the DDA's own cell space
//   w   = air-exposed face mask, bit 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z
// NO RADIANCE RIDES ALONG, on purpose: THE LAW is one Le (§4), so the
// shadow ray's own march() hit supplies it via cellEmission(). A second
// copy of Le on the CPU is the divergence the contract forbids.
uniform float claudeAreaCount; // live slots, 0..AREA_CAP; 0 = no NEE
uniform vec4 claudeArea0;
uniform vec4 claudeArea1;
uniform vec4 claudeArea2;
uniform vec4 claudeArea3;
uniform vec4 claudeArea4;
uniform vec4 claudeArea5;
uniform vec4 claudeArea6;
uniform vec4 claudeArea7;
uniform vec4 claudeArea8;
uniform vec4 claudeArea9;
uniform vec4 claudeArea10;
uniform vec4 claudeArea11;
uniform vec4 claudeArea12;
uniform vec4 claudeArea13;
uniform vec4 claudeArea14;
uniform vec4 claudeArea15;

CENTROID_ VARYING_ mediump vec2 varTexCoord;

#if __VERSION__ >= 130
#define texture3D texture
#endif

// ---------------------------------------------------------------------
// NAMED CONSTANTS
// ---------------------------------------------------------------------

// The grid is 128 cells on a side (game.cpp claudeTraceGridSnapshot).
const float GRID_S = 128.0;

// A ray crosses at most one cell boundary per DDA step, and at most 128
// boundaries per axis, so 3*128 is the exact worst case for a diagonal
// crossing of the grid. Any smaller bound would silently truncate a
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

// CLASS BANDS (game.cpp claudeTraceGridSnapshot writes the class byte into
// the grid's alpha; the shader sees byte/255).
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
// THE SUB-VOXEL CLASS, 250: "do not stop at my 1 m wall, look at my real
// shape". Banded at the half-byte midpoints either side of it for the
// same reason as the emissive band: 245 sits between the last emissive
// class (240) and 250, 252.5 between 250 and plain solid (255). A cell
// in this band and inside the ring has a 16^3 mask; one OUTSIDE the ring
// has none, and is an opaque cube — see march()'s rescale.
const float CLASS_SUBVOX_LO = 245.0 / 255.0;
const float CLASS_SUBVOX_HI = 252.5 / 255.0;

// THE SUB-VOXEL RING. game.cpp ClaudeTraceGrid::NBOX_R0/NBOX_R1, and the
// same two numbers gate the BAKE — a third copy is how a mask ends up
// read one cell away from the cell that owns it.
const float SUBV_R0 = 48.0;
const float SUBV_R1 = 80.0;
// Sub-voxels per cell edge. 1/16 m is the rendered detail size (§2).
const float SUBV = 16.0;
// THE TWO RUNGS, as the walk carries them: the size of one cell of the
// rung, in world units. march() holds exactly one of these at a time in
// a variable and rescales between them; 1/16 is a power of two, so the
// rescale round-trips bit-exactly.
const float RUNG_FINE = 1.0 / SUBV;

// THE STEP BUDGET, and it is ONE budget because there is now ONE loop.
// The bound still has to make a descent unable to starve the 1 m walk:
// if a fine step could spend what a coarse step would have spent, a ray
// through a few stairs would exhaust the bound, march() would return
// false, and the caller reads false as "escaped the grid" — i.e. BLACK,
// with no error anywhere. That failure mode is why the pre-2026-08-17
// code gave the inner walk a separate budget.
//
// So the single bound is the PRODUCT, not a shared pool. The walk
// visits at most MARCH_STEPS coarse cells. Inside one of them a
// diagonal crossing of a 16^3 mask crosses 3*16 = 48 boundaries, and
// the entry sub-voxel is tested by the rescale itself rather than by an
// iteration, so 49 iterations is the exact worst case per cell;
// SUBV_ITERS is that plus slack for a ray entering exactly on a corner.
// Every coarse cell may therefore cost 1 + SUBV_ITERS iterations, and
// WALK_STEPS = 384 * 52 = 19,968 is as unreachable as MARCH_STEPS = 384
// was on its own. The guarantee is preserved; only its arithmetic moved.
const int SUBV_ITERS = 51;
const int WALK_STEPS = MARCH_STEPS * (SUBV_ITERS + 1);

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
// before it tests, so the starting cell is never re-hit; this pushes the
// restart clear of the face it left, along that face's own normal.
//
// IT IS NOT ON ITS OWN ENOUGH TO KEEP floor() HONEST, and believing it
// was is what leaked a sealed room for as long as the walk has existed
// (2026-08-17). An epsilon along n says nothing about the other two
// axes, and it is the other two that go wrong. See restartPoint().
const float SURFACE_EPS = 0.01;

// The margin the restart point keeps from the walls of the cell it is
// clamped into, as a fraction of that cell's own size — so it is the
// same rule at 1 m and at 1/16 m, which is §2's "size is a parameter,
// never a branch" applied to this constant too. It is not a tuned
// tolerance: the only thing it has to beat is a float32 ulp at grid
// coordinates (1.5e-5 at 128), and 1/512 clears that by 128x while
// moving a bounce origin by at most 2 mm — a fifth of what SURFACE_EPS
// already moves it. It is also the constant the sub-voxel entry clamp
// has always used, in that rung's own units.
const float CELL_IN = 1.0 / 512.0;

// Guard against a division blowing up on an axis-aligned ray.
const float DDA_MIN_ABS = 1e-6;

const float PI = 3.14159265358979323846;
const float PI2 = 6.28318530717958647692;

// Slots in the area-emitter list. MUST equal ClaudeTraceGrid::AREA_CAP in
// game.cpp: the shader trusts claudeAreaCount as the size of the set it
// samples uniformly, and a mismatch would make the 1/N in the pdf a
// different N from the one the selection actually used — which is not a
// dimmer image, it is a WRONGLY SCALED one.
const int AREA_CAP = 16;

// A sampled light point sits exactly on a face plane, so the shadow
// ray's own hit lands in the emitter cell and the test is "is the first
// opaque cell the target cell", not a distance compare with a tuned
// epsilon. This is the only tolerance in that test: cell indices are
// integers, so half a cell separates any two of them.
const float CELL_MATCH_EPS = 0.5;

// SKY DOME SHAPE. The dome is a one-parameter blend from the horizon
// colour to the zenith colour in the ray's elevation. The exponent is
// below 1 so the horizon band stays NARROW and most of the visible dome
// reads as the zenith colour, which is the shape Luanti's own raster
// skybox has; at 1.0 the whole upper hemisphere is a smooth ramp and the
// horizon glow spreads halfway to the top. It is a named number because
// it changes an image (see "NO HIDDEN CONSTANTS" above).
const float SKY_DOME_POW = 0.5;
// The sentinel that means "this body is not in the sky". No unit vector
// pair has a dot product of 2, so the cone test can never fire.
const float SKY_BODY_OFF = 1.5; // any cos above this is the sentinel

// Grazing floor on cos_y at the light. p_l carries a 1/cos_y, so a face
// seen exactly edge-on drives it to infinity and inf/inf is a NaN in the
// balance weight. Declining those directions costs NO energy, because
// BOTH sides of the weight use this same threshold: neeDirect() returns
// black there and neePdfSa() returns 0, which hands the BSDF sample the
// full weight instead. The contribution the light sampler gives up is
// proportional to 1/p_l, i.e. it was heading to zero anyway.
const float NEE_COS_MIN = 1e-6;

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
// Distance display range for view 4: the grid's body diagonal,
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

// --- the RNG (claudeRng = 1, the default) ------------------------------
// PCG output-permuted LCG on a 32-bit word. Unlike the chain above, the
// draw index enters as DATA rather than as iteration count, so draw n
// and draw n+1 are two hashes of two different inputs and share no
// trajectory. Integer ops only, no float round-off in the state.
uint g_rngKey;   // per pixel, per frame
uint g_rngCtr;   // draw index within this pixel-frame

uint pcgHash(uint v)
{
	uint st = v * 747796405u + 2891336453u;
	uint wd = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;
	return (wd >> 22u) ^ wd;
}

// NOTE: every call site assigns to a named local first. GLSL does not
// define the evaluation order of constructor/function arguments, so
// vec2(rnd1(), rnd1()) would be a real (silent, driver-specific) bug.
float rnd1()
{
	if (claudeRng > 0.5) {
		g_rngCtr += 1u;
		return float(pcgHash(g_rngKey ^ pcgHash(g_rngCtr)))
				* (1.0 / 4294967296.0);
	}
	// The golden-ratio increment spreads successive states across the
	// hash's input range instead of walking one neighbourhood.
	g_rngState = hash11(g_rngState + 0.61803398875);
	return g_rngState;
}

// --- THE SKY SAMPLER'S OWN STREAM, and it is disjoint on purpose -------
// The sky light sampler (neeSky() below) is a NEW consumer of random
// numbers at every vertex. Drawing from the main chain would shift every
// subsequent draw in the path by two, which changes every scattered
// direction in every scene — including the SEALED rooms, where the sky
// is provably a no-op and the goldens must not move. Gate 1 of this step
// is exactly that claim, so the sampler is given its own counter range
// instead: draw index SKY_CTR_BASE + n, which the main chain (a few
// hundred draws per pixel at most) can never reach. The key is the same,
// so it is still one generator per pixel-frame; only the sub-stream is
// reserved. This is the whole reason the sealed arms come back
// BIT-IDENTICAL rather than merely within tolerance.
//
// It always uses the counter-based PCG, never the claudeRng = 0 hash
// chain: that dial is the A/B partner for the DIRECTION sampler's bias
// (measured.md "1a"), and an iterated chain has no disjoint sub-stream to
// give.
const uint SKY_CTR_BASE = 1u << 24u;
uint g_skyCtr;

float rndSky()
{
	g_skyCtr += 1u;
	return float(pcgHash(g_rngKey ^ pcgHash(SKY_CTR_BASE + g_skyCtr)))
			* (1.0 / 4294967296.0);
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
// THE MISS FUNCTION — sky(direction) -> radiance
// ---------------------------------------------------------------------
// ONE function, TWO roles (background on escape, and light), because §4
// admits one emission law and the sky is an emitter. Every caller in this
// file goes through skyRadiance(): the camera ray's escape, a bounce
// ray's escape, and the NEE shadow ray that misses. There is no second
// formula and no CPU-side copy of the answer.
//
// The moon's SHAPE is its sprite, sampled from the texture the game is
// already drawing (Mineclonia: mcl_moon's 8-phase sheet, seeded from the
// world seed and swapped as the days pass). A moon rendered as a flat
// disc of moon-coloured light is a dim sun with a different colour, which
// is the exact defect the blue-sun note in claude_present recorded.
//
// The sun is a flat disc: its sprite carries no information a phase-less
// glowing circle does not, and one sampler is cheaper than two. Stated
// rather than left to be discovered.
float skyBodyMask(vec3 d, vec3 bdir, float bcos, vec3 bu, vec3 bv)
{
	if (skyMoonTexOn < 0.5)
		return 1.0;
	// The body is a quad at unit distance; the ray crosses its plane at
	// p = d / dot(d, bdir). r = tan(angular radius) is the quad's half
	// extent, recovered from the cone the sampler and the pdf both use,
	// so the drawn shape and the sampled region cannot drift apart.
	float ct = max(dot(d, bdir), 1e-6);
	float r = sqrt(max(1.0 - bcos * bcos, 0.0)) / max(bcos, 1e-6);
	vec3 p = d / ct;
	vec2 q = vec2(dot(p, bu), dot(p, bv)) / r;
	vec4 t = texture2D(claudeMoonTex, q * 0.5 + 0.5);
	// The sprite is premultiplied by nothing: alpha is the cut-out and rgb
	// is the lit side. Luminance-weighted so the crescent's terminator is
	// a brightness edge, not a hue edge (John is colorblind).
	return t.a * dot(t.rgb, vec3(0.2126, 0.7152, 0.0722));
}

// THE DOME half: everything the light sampler does NOT aim at. Split out
// for one reason, and it is a requirement of MIS rather than a
// convenience: the balance weight is per LIGHT, so a direction that lands
// on the sun carries w_b for the sun AND the full dome radiance behind
// it. skyRadiance() below is still the one function every caller sees;
// the two halves exist so that dome + body is exactly it, always.
vec3 skyDome(vec3 d)
{
	if (claudeSkyUniform > 0.0)
		return vec3(claudeSkyUniform);
	// Below the horizon: the far field. The split is the horizontal
	// plane, the same plane the dome is measured from -- one
	// discriminator, no blend band, because a blend would be a third
	// thing to justify.
	if (d.y <= 0.0)
		return skyGroundCol;
	return mix(skyHorizonCol, skyZenithCol, pow(d.y, SKY_DOME_POW));
}

// THE BODY half: the sun's disc or the moon's sprite. Zero everywhere
// else. The uniform test sky has no body at all, which is what makes
// L = rho * L_sky exact for the analytic referee.
vec3 skyBody(vec3 d)
{
	if (claudeSkyUniform > 0.0)
		return vec3(0.0);
	vec3 L = vec3(0.0);
	// dot >= cos(angular radius) IS the disc; the sentinel keeps a body
	// that is below the horizon or switched off by the game out of both
	// the image and the pdf, with no second flag to keep in agreement.
	if (dot(d, skySunDir) >= skySunCos)
		L += skySunCol;
	if (dot(d, skyMoonDir) >= skyMoonCos)
		L += skyMoonCol
				* skyBodyMask(d, skyMoonDir, skyMoonCos, skyMoonU, skyMoonV);
	return L;
}

// THE MISS FUNCTION ITSELF. Every consumer that wants "the sky in this
// direction" — the camera ray, the debug view, any future one — calls
// this and nothing else.
vec3 skyRadiance(vec3 d)
{
	return skyDome(d) + skyBody(d);
}

// ---------------------------------------------------------------------
// TRAVERSAL — plain branchless 3D-DDA, no acceleration structure
// ---------------------------------------------------------------------
// Deliberately NOT using the occupancy MIP pyramid (unit 11). Rung 1 is
// the reference, its scenes are two small sealed rooms, and "slow is
// fine". One traversal, no second code path to keep in agreement.
// march() IS that traversal, and it is directly below these two helpers;
// what it returns and what it guarantees are documented on it.

// Is this cell inside the sub-voxel ring? OUTSIDE IT A CLASS-250 CELL
// HAS NO BITS — the bake only fills [48,80)^3, and game.cpp's
// authored-model branch tags a cell 250 anywhere in the 128^3 grid. A
// descent that skipped this test would find an all-zero mask and turn
// every distant chest, bed and campfire INVISIBLE.
//
// This is a §2 ladder — a coarser rung at distance — and it is written
// down as one: beyond 16 cells from the grid centre a sub-metre shape
// renders as the 1 m cell it occupies. It is the same ladder game.cpp
// already applies to point-light models (game.cpp: "outside the subvox
// ring a class-250 cell has no bits to express").
bool inSubvoxRing(vec3 cell)
{
	return all(greaterThanEqual(cell, vec3(SUBV_R0)))
			&& all(lessThan(cell, vec3(SUBV_R1)));
}

// One bit of the ring texture. rc = ring-local cell (cell - 48),
// sc = sub-voxel index inside it, both already known to be in range.
bool subvoxSolid(vec3 rc, vec3 sc)
{
	vec3 texel = vec3(rc.x * 2.0 + floor(sc.x / 8.0),
			rc.y * SUBV + sc.y,
			rc.z * SUBV + sc.z);
	float raw = texture3D(claudeSubvoxTex,
			(texel + 0.5) / vec3(64.0, 512.0, 512.0)).r;
	// +0.5 before floor: an R8 byte comes back as n/255 in float32 and
	// n/255*255 can land at n - epsilon, which floor() would drop a
	// whole bit-plane on.
	float byte = floor(raw * 255.0 + 0.5);
	float bit = mod(floor(byte / exp2(mod(sc.x, 8.0))), 2.0);
	return bit >= 0.5;
}

// ---------------------------------------------------------------------
// THE RESTART POINT — where a ray that just hit something starts next
// ---------------------------------------------------------------------
// A bounce ray, a shadow ray and the next segment of a path all begin on
// a surface, and the whole ray-origin exclusion rests on WHERE. The walk
// never tests the cell a ray starts in — that is what keeps a bounce ray
// off the face it just left — so the starting cell had better be the
// cell the ray just came THROUGH, which the walk has already tested and
// found empty. If it is any other cell, and that cell is solid, the
// exclusion waves the ray straight through a wall.
//
// IT WAS ANY OTHER CELL, AND THAT WAS THE LEAK (2026-08-17). cave-glass
// — a sealed 7x5x7 room with no opening at all — put 668 pixels of sky
// on screen under a 50x test sky. The restart was computed as
// hit + n * SURFACE_EPS and handed to floor(), and floor() and the walk
// disagreed about which cell that is.
//
// MEASURED with claude_view 18, not theorised, and it is none of the
// three things it looked like. In 100 % of events, in both plain-cube
// rooms, the cell floor() chose was the cell across the hit face PLUS a
// step SIDEWAYS — into a neighbour of the cell that was hit — and the
// restart point sat BIT-EXACTLY on that neighbour's boundary plane (its
// distance to the nearest face was 0 to the instrument's 1e-12 floor).
// The epsilon moves along n and along n only, so a tangential
// disagreement cannot be an epsilon that is too short, a grazing angle,
// or a normal pointing the wrong way. It is a hit landing on the EDGE of
// a face: the hit point's tangential coordinate is within half a float32
// ulp of an integer, floor() rounds it into the next cell, and at a
// concave corner — where a floor meets a wall — that cell is the wall.
//
// So the restart is CONSTRAINED, not merely offset: clamped into the
// cell the walk came through, whose corner is `lo` and whose size is
// `h`. floor() of the result IS that cell, by construction, at either
// rung. The ordinal rule ("skip the cell the ray starts in") and the
// identity rule ("skip the cell the ray just left") then name the same
// cell, always — which is what the exclusion's own comment has claimed
// since it was written.
//
// The clamp can only bite tangentially: along n the epsilon has already
// placed the point 0.01 inside, five times deeper than CELL_IN's margin.
// It costs two vector min/max per HIT, not per step.
vec3 restartPoint(vec3 phit, vec3 n, vec3 lo, float h)
{
	float m = h * CELL_IN;
	return clamp(phit + n * SURFACE_EPS, lo + m, lo + (h - m));
}

// ---------------------------------------------------------------------
// THE WALK — ONE 3D-DDA, and the cell size is a PARAMETER of it
// ---------------------------------------------------------------------
// §2 of the physics contract: "size is a parameter, never a branch. One
// traversal, one lighting law, one emission law, one material law,
// regardless of cell size." Taken literally, that is this function: one
// loop, one index, one set of side-distances, one step budget. On
// entering a class-250 cell inside the ring the walk RESCALES ITSELF by
// 16 — the same variables, now measured in 1/16 m — and on leaving the
// cell it rescales back and carries on. There is no second walk, no
// second position, no second side-distance triple and no callee holding
// its own copy of the four things the walk already has.
//
// (Until 2026-08-17 the descent was a separate function, `descendCell()`,
// running a second DDA with its own locals live alongside these. The
// cost instrument measured what that cost: +36 % in Cornell and +49 % in
// the cabin with the dial OFF and not one instruction of it executing —
// see spec/measured.md "Descend cost instrument". This is the fix that
// section ranked first, and it is the same algorithm, not a new one.)
//
// THE RUNG lives in exactly three variables:
//   delta   world t to cross one cell of the current rung, per axis
//           (= 1/|rd| at 1 m, /16 at 1/16 m — an exact power-of-two
//           rescale, so it round-trips)
//   lim     the index bound of the rung: GRID_S coarse, SUBV fine. It
//           is also the flag that says WHICH rung the walk is on.
//   ci      the index at the current rung: the 1 m cell, or the
//           sub-voxel inside cellHi.
// cellHi is the 1 m cell the walk is inside while it is on the fine
// rung — the mask's owner, and the cell a fine hit reports as cellOut.
//
// t is world parametric distance THROUGHOUT, at both rungs. That is
// what keeps every t in this file comparable (depth packing, NEE
// distance, the RR schedule) without a conversion at the boundary.
//
// Returns true on an opaque hit and fills hp / n / alb / le / tHit /
// cellOut. False = the ray left the grid (or ran out of steps, which
// the WALK_STEPS bound makes unreachable — see its derivation above).
//
// SHADOW RAYS USE THIS FUNCTION, not a lighter copy of it. §2 lists "a
// voxel that exists for eye rays but not for shadow, bounce, or emitter
// rays" as a contract violation, and the cheapest way to never commit it
// is to have exactly one traversal. cellOut exists for those rays: a
// visibility test that compares the first opaque CELL against the
// emitter cell needs no epsilon and cannot self-shadow the light — so
// cellOut is the COARSE cell of the hit at both rungs.
//
// THE RESCALE, both ways, is the only arithmetic that is new. For a
// world point p on the ray at parametric distance t, a rung of size h,
// and the index ci of the cell containing p measured from the corner of
// the 1 m cell (0 at the coarse rung, the sub-voxel index at the fine
// one), the distance to the next boundary on each axis is
//
//   sideDist = t + (stepDir*(ci*h - pl) + (stepDir*0.5 + 0.5)*h) / |rd|
//
// with pl = p - cellHi. At h = 1, ci = 0 that is the walk's own opening
// line; at h = 1/16 it is exactly what the old inner walk computed in
// sub-voxel units and then divided by 16. Rescaling DOWN keeps the
// coarse state nowhere, and rescaling UP rebuilds it from the exit
// point — which is why nothing has to be saved across a descent. The
// two are equivalent because the fine walk leaves the cell through the
// same face, at the same t, on the same axis, as the coarse step it
// replaces.
bool march(vec3 ro, vec3 rd, out vec3 hp, out vec3 n, out vec3 alb,
		out vec3 le, out float tHit, out vec3 cellOut)
{
	hp = ro;
	n = vec3(0.0, 1.0, 0.0);
	alb = vec3(0.0);
	le = vec3(0.0);
	tHit = DEPTH_MISS;
	cellOut = vec3(-1.0);

	vec3 stepDir = sign(rd);
	vec3 delta = 1.0 / max(abs(rd), vec3(DDA_MIN_ABS));
	vec3 ci = floor(ro);
	vec3 cellHi = ci;
	vec3 sideDist = (stepDir * (ci - ro) + stepDir * 0.5 + 0.5) * delta;
	vec4 s = vec4(0.0);
	float t = 0.0;
	float lim = GRID_S;
	int axis = -1;

	// THE STARTING CELL, and it is tested for exactly one thing. The
	// loop below never tests the cell the ray starts in, which is what
	// keeps a bounce ray off its own surface.
	//
	// THAT RULE IS ONLY SAFE BECAUSE OF restartPoint(). "The cell the ray
	// starts in" is a statement about floor(ro), and until 2026-08-17
	// nothing made floor(ro) agree with the walk that produced ro — so a
	// restart landing one ulp over a cell boundary began the walk inside
	// a WALL and this rule waved it through. It is now clamped into the
	// cell the previous walk came through, so the cell skipped here is
	// provably the cell the ray just left, at either rung. See
	// restartPoint() for the measurement that named it.
	//
	// Once cells have interiors
	// that rule is too coarse: a ray leaving one sub-voxel would escape
	// the other 4095 for free — a stair would not self-shadow, a chest
	// lid would not shadow its own body, and sub-metre geometry would
	// look right and light flat, which is §2's first listed violation
	// coming back one rung finer. So when the starting cell carries
	// sub-voxel bits, the walk rescales into it WITHOUT testing the
	// sub-voxel the ray starts in — the ray-origin exclusion, moved one
	// rung down — and every other class is left alone. A 255 cell, an
	// emitter, air: unchanged, not tested, exactly as before. (That
	// origin sub-voxel is now air by CONSTRUCTION rather than by an
	// epsilon's good behaviour: a fine hit's restart is clamped into the
	// sub-voxel the walk came through, and the walk only walks through
	// empty ones. It was "SURFACE_EPS pushes the restart 0.16 sub-voxels
	// off the face it left, belt and braces, stated rather than relied
	// upon" — and at 1 m the same reasoning turned out to be wrong.)
	if (claudeDescend > 0.5 && inSubvoxRing(cellHi)) {
		s = texture3D(claudeTraceGrid, (cellHi + 0.5) / GRID_S);
		if (s.a > CLASS_SUBVOX_LO && s.a < CLASS_SUBVOX_HI) {
			// entry point in SUB-VOXEL units, clamped INSIDE the
			// cell: the walk lands exactly on a cell plane and floor()
			// of an exact boundary can fall either side of it. The
			// clamp is on the POSITION, not on the index, and the
			// side-distances are measured from the clamped position —
			// preserved exactly from the two-walk code, because a
			// tidier clamp here moves t by ~2e-4 on rays that enter
			// through a face and that would be a second variable.
			vec3 pu = clamp((ro - cellHi) * SUBV, vec3(0.0),
					vec3(SUBV - 1.0 / 512.0));
			ci = floor(pu);
			delta *= RUNG_FINE;
			sideDist = (stepDir * (ci - pu) + stepDir * 0.5 + 0.5) * delta;
			lim = SUBV;
		}
	}

	for (int i = 0; i < WALK_STEPS; i++) {
		// ONE STEP OF THE WALK, at whatever size the walk is currently
		// set to. These are the same three lines at 1 m and at 1/16 m;
		// the rung is in delta and lim, not in a branch. Advance first,
		// then test what was entered: the cell the ray starts in is
		// never tested, which is what keeps a bounce ray off its own
		// surface.
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x; sideDist.x += delta.x;
			ci.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y; sideDist.y += delta.y;
			ci.y += stepDir.y; axis = 1;
		} else {
			t = sideDist.z; sideDist.z += delta.z;
			ci.z += stepDir.z; axis = 2;
		}

		bool escaped = any(lessThan(ci, vec3(0.0)))
				|| any(greaterThanEqual(ci, vec3(lim)));

		if (lim < GRID_S) {
			// ---- the walk is on the 1/16 m rung, inside cellHi ----
			if (!escaped) {
				if (!subvoxSolid(cellHi - vec3(SUBV_R0), ci))
					continue; // this sub-voxel is empty: step again
				n = vec3(0.0);
				if (axis == 0) n.x = -stepDir.x;
				else if (axis == 1) n.y = -stepDir.y;
				else n.z = -stepDir.z;
				// ci + n is the SUB-VOXEL the walk came through, and it
				// is empty: the loop only continues past sub-voxels it
				// found empty, and the origin one is excluded. So the
				// next ray starts inside a known-empty 1/16 m cell of
				// cellHi — the same guarantee as at 1 m, one rung down.
				hp = restartPoint(ro + rd * t, n,
						cellHi + (ci + n) * RUNG_FINE, RUNG_FINE);
				alb = cellAlbedo(s.rgb);
				le = cellEmission(s.a, alb);
				tHit = t;
				cellOut = cellHi;   // the COARSE cell, for neeDirect
				return true;
			}
			// LEFT THE CELL. The face just crossed is also the 1 m face
			// the coarse walk would have crossed, at the same t and on
			// the same axis — so rescale the walk back to 1 m, put it in
			// the neighbour across that face, rebuild sideDist from the
			// exit point, and fall straight into the arrival test below.
			// Nothing was saved across the descent and nothing is
			// restored: the state is recomputed, which is why the fine
			// rung costs no live registers of its own.
			delta *= SUBV;
			if (axis == 0) cellHi.x += stepDir.x;
			else if (axis == 1) cellHi.y += stepDir.y;
			else cellHi.z += stepDir.z;
			ci = cellHi;
			lim = GRID_S;
			vec3 pl = ro + rd * t - cellHi;
			sideDist = t + (stepDir * (-pl) + stepDir * 0.5 + 0.5) * delta;
			escaped = any(lessThan(ci, vec3(0.0)))
					|| any(greaterThanEqual(ci, vec3(GRID_S)));
		}

		if (escaped)
			return false; // escaped: contributes nothing (sky is punted)

		// ARRIVAL AT A 1 M CELL. s stays live across a descent — a fine
		// hit takes its colour and its class from the cell that owns
		// the mask — and it is only ever written here, on the coarse
		// rung, where nothing is depending on the old value.
		s = texture3D(claudeTraceGrid, (ci + 0.5) / GRID_S);
		if (s.a <= CLASS_AIR_MAX)
			continue; // air

		n = vec3(0.0);
		if (axis == 0) n.x = -stepDir.x;
		else if (axis == 1) n.y = -stepDir.y;
		else n.z = -stepDir.z;

		// CLASS 250 SAYS "DO NOT STOP AT MY 1 M WALL". Rescale the walk
		// to 1/16 m and keep going, against this cell's 16^3 mask. The
		// sub-voxel the ray enters through is tested here, by the
		// rescale, because the loop only ever tests what it stepped INTO
		// and no step has been taken inside the cell yet; if it is
		// solid, the surface the ray met is the 1 m face it came
		// through and n already holds that face's normal. A miss means
		// the ray passed THROUGH this cell and the walk resumes at 1 m
		// from the far face. Ring-gated, because outside [48,80)^3 a 250
		// cell has no mask and must stay the cube it is today.
		//
		// Shadow and bounce rays get this for free and that is the
		// point: §2 forbids a voxel that exists for eye rays but not
		// for shadow rays, and one traversal is the cheapest way never
		// to commit it. There is no lighter copy of this walk.
		if (claudeDescend > 0.5 && s.a > CLASS_SUBVOX_LO
				&& s.a < CLASS_SUBVOX_HI && inSubvoxRing(ci)) {
			cellHi = ci;
			// entry point in SUB-VOXEL units, clamped INSIDE the cell
			// (see the starting-cell block for why the clamp is on the
			// position rather than on the index)
			vec3 pu = clamp((ro + rd * t - cellHi) * SUBV, vec3(0.0),
					vec3(SUBV - 1.0 / 512.0));
			vec3 su = floor(pu);
			if (!subvoxSolid(cellHi - vec3(SUBV_R0), su)) {
				delta *= RUNG_FINE;
				sideDist = t + (stepDir * (su - pu)
						+ stepDir * 0.5 + 0.5) * delta;
				ci = su;
				lim = SUBV;
				continue;
			}
		}

		// every other class is one thing: an opaque Lambertian surface
		// with a cardinal normal, which may also emit
		//
		// ci + n is the cell the walk came through — tested and found
		// air on the step before this one, or the cell the ray started
		// in, which the origin exclusion guarantees was air for the same
		// reason one rung up. That is what makes restartPoint()'s clamp
		// target the right cell rather than merely a nearby one.
		hp = restartPoint(ro + rd * t, n, ci + n, 1.0);
		alb = cellAlbedo(s.rgb);
		le = cellEmission(s.a, alb);
		tHit = t;
		cellOut = ci;
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
// NEXT-EVENT ESTIMATION + MIS (rung 2). See the header for the math.
// Every function below is dead code when claudeNee = 0: main() never
// calls one, and the pure path does not lose or gain a single RNG draw.
// ---------------------------------------------------------------------

// The one place the sixteen slots are indexed. An if-chain, not an
// array — see the GLSL PROFILE note about array uniform names.
vec4 areaEmitter(int i)
{
	if (i == 0) return claudeArea0;
	if (i == 1) return claudeArea1;
	if (i == 2) return claudeArea2;
	if (i == 3) return claudeArea3;
	if (i == 4) return claudeArea4;
	if (i == 5) return claudeArea5;
	if (i == 6) return claudeArea6;
	if (i == 7) return claudeArea7;
	if (i == 8) return claudeArea8;
	if (i == 9) return claudeArea9;
	if (i == 10) return claudeArea10;
	if (i == 11) return claudeArea11;
	if (i == 12) return claudeArea12;
	if (i == 13) return claudeArea13;
	if (i == 14) return claudeArea14;
	return claudeArea15;
}

// Face index -> outward cardinal normal. The order IS the bit order of
// game.cpp's mask; the two must be read together or the light sampler
// aims at faces the snapshot called sealed.
vec3 faceNormal(int f)
{
	if (f == 0) return vec3(1.0, 0.0, 0.0);
	if (f == 1) return vec3(-1.0, 0.0, 0.0);
	if (f == 2) return vec3(0.0, 1.0, 0.0);
	if (f == 3) return vec3(0.0, -1.0, 0.0);
	if (f == 4) return vec3(0.0, 0.0, 1.0);
	return vec3(0.0, 0.0, -1.0);
}

// The inverse, for a normal march() already produced. Cardinal normals
// are law (§2), so this is exact, not a nearest-axis guess.
int faceIndex(vec3 n)
{
	if (n.x > 0.5) return 0;
	if (n.x < -0.5) return 1;
	if (n.y > 0.5) return 2;
	if (n.y < -0.5) return 3;
	if (n.z > 0.5) return 4;
	return 5;
}

// Can the light sampler at x put a sample on face f of cell c? Two
// conditions, and BOTH sides of the MIS weight ask this same question:
//   (1) the face is air-exposed, per the snapshot's mask. Every non-air
//       class is opaque in rung 1, so an unexposed face is one no ray
//       can reach — it belongs in neither pdf.
//   (2) the face is turned toward x. Sampling the far side of a cube
//       would spend half the samples on cos_y <= 0, and excluding it is
//       a change of p_A, not a free optimisation — which is exactly why
//       this predicate, and not two similar ones, answers for both.
bool faceCandidate(vec3 c, int mask, int f, vec3 x)
{
	if (((mask >> f) & 1) == 0)
		return false;
	vec3 nf = faceNormal(f);
	// cell centre c + 0.5, face centre half a cell further along nf
	return dot(nf, x - (c + vec3(0.5) + nf * 0.5)) > 0.0;
}

// p_l for a point the BSDF sampler found, in SOLID ANGLE about x — the
// density neeDirect() WOULD have had, had it aimed at this exact face
// from this exact x. Zero when the light sampler cannot generate the
// direction at all (cell absent from the list because the cap
// overflowed, or the face not a candidate), and that zero is load
// bearing: it makes w_b = 1 there, so a truncated list costs variance
// and never energy.
float neePdfSa(vec3 cellHit, vec3 nHit, vec3 x, float dist, float cosY,
		int nLights)
{
	if (nLights <= 0 || cosY <= NEE_COS_MIN)
		return 0.0;
	int mask = 0;
	bool listed = false;
	for (int i = 0; i < AREA_CAP; i++) {
		if (i >= nLights)
			break;
		vec4 e = areaEmitter(i);
		if (all(lessThan(abs(e.xyz - cellHit), vec3(CELL_MATCH_EPS)))) {
			mask = int(e.w + 0.5);
			listed = true;
			break;
		}
	}
	if (!listed)
		return 0.0;
	if (!faceCandidate(cellHit, mask, faceIndex(nHit), x))
		return 0.0;
	int k = 0;
	for (int f = 0; f < 6; f++) {
		if (faceCandidate(cellHit, mask, f, x))
			k++;
	}
	if (k == 0)
		return 0.0; // unreachable: the hit face itself passed the test
	// p_A = 1/(N*k) over unit-square faces, converted to solid angle
	return (dist * dist) / (float(nLights) * float(k) * cosY);
}

// One light sample at vertex (x, nx) with reflectance rho. Returns the
// MIS-weighted direct contribution WITHOUT the path throughput, which
// the caller multiplies in.
//
// Cost: exactly four RNG draws and one shadow march when it runs to
// completion — fewer on an early out, which is fine, the draws are a
// per-pixel chain and not a fixed budget.
//
// misOn is 1.0 for the transport path and is the ONLY thing the
// instrument views change: at 0.0 the balance weight w_l is replaced by
// 1.0, which turns this function into the plain light-sampling estimator
// of the direct term (roadmap 1a instrument A, claude_view 9). Every
// other line — the same list, the same k, the same face, the same
// shadow march, the same pdf — is shared with the transport path by
// construction, which is the point: an instrument that re-derives the
// thing it is measuring measures its own copy.
// FORENSICS for roadmap 1a, filled by neeDirect() on every call and read
// by claude_view 12/13. It is written, never read, inside the transport
// path — the dead store is the entire cost, and it buys an instrument
// that reports on THE SAME CALL that produced the pixel rather than on a
// re-derived copy of it.
//   x = 1.0 when this call produced a nonzero contribution, else 0
//   y = the WORLD y of the emitter cell it aimed at
//   z = cos_x at the receiver
//   w = vec2(slot index, k) packed as slot + 16*k is not needed: view 13
//       reads slot and k out of two separate accumulations instead.
vec4 g_neeDiag;
float g_neeDiagK;

vec3 neeDirect(vec3 x, vec3 nx, vec3 rho, int nLights, float misOn)
{
	g_neeDiag = vec4(0.0);
	g_neeDiagK = 0.0;
	// uniform over the list. Not importance-weighted by distance or
	// power: p_A must be reproducible by neePdfSa() from the hit alone,
	// and 1/(N*k) is.
	float us = rnd1();
	int li = min(int(float(nLights) * us), nLights - 1);
	vec4 e = areaEmitter(li);
	vec3 c = e.xyz;
	int mask = int(e.w + 0.5);

	int k = 0;
	for (int f = 0; f < 6; f++) {
		if (faceCandidate(c, mask, f, x))
			k++;
	}
	if (k == 0)
		return vec3(0.0); // this emitter shows x nothing

	float uf = rnd1();
	int pick = min(int(float(k) * uf), k - 1);
	int face = 5;
	int seen = 0;
	for (int f = 0; f < 6; f++) {
		if (!faceCandidate(c, mask, f, x))
			continue;
		if (seen == pick) {
			face = f;
			break;
		}
		seen++;
	}

	// AREA, NOT A POINT (§4). A uniform point on the unit face — this is
	// the whole reason penumbra exists in this renderer, and the wrapped
	// cosine point light that came before it could not produce one at any
	// sample count.
	vec3 nL = faceNormal(face);
	float u1 = rnd1();
	float u2 = rnd1();
	vec3 ta = abs(nL.y) > 0.5 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	vec3 tx = cross(ta, nL); // cardinal x cardinal: already unit
	vec3 ty = cross(nL, tx);
	vec3 y = c + vec3(0.5) + nL * 0.5
			+ tx * (u1 - 0.5) + ty * (u2 - 0.5);

	vec3 d = y - x;
	float dist2 = dot(d, d);
	if (dist2 < 1e-8)
		return vec3(0.0);
	float dist = sqrt(dist2);
	vec3 wi = d / dist;
	float cosX = dot(nx, wi);
	float cosY = dot(nL, -wi);
	if (cosX <= 0.0 || cosY <= NEE_COS_MIN)
		return vec3(0.0); // same threshold neePdfSa() uses — see the const

	// VISIBILITY through the one traversal. Visible iff the first opaque
	// cell on the way IS the emitter cell — no epsilon on t, and no way
	// for the emitter to shadow itself.
	vec3 shp, shn, shalb, shle, shcell;
	float sht;
	if (!march(x, wi, shp, shn, shalb, shle, sht, shcell))
		return vec3(0.0);
	if (any(greaterThanEqual(abs(shcell - c), vec3(CELL_MATCH_EPS))))
		return vec3(0.0); // occluded

	// THE LAW: one Le. shle came out of march(), which ran the same
	// cellEmission() the eye ray runs — no second formula here, and no
	// CPU-side copy of Le to drift from it. Deliberately NOT guarded
	// against shle == 0: a non-emissive cell in the list would have to
	// be a snapshot bug, and multiplying it through returns black on its
	// own. An early-out here would instead hide it, and would make
	// neeDirect() decline a direction neePdfSa() still prices — the one
	// asymmetry that actually loses energy.
	float pdfL = dist2 / (float(nLights) * float(k) * cosY); // p_l, sa
	float pdfB = cosX / PI;                                  // p_b, sa
	float w = 1.0;
	if (misOn > 0.5)
		w = pdfL / (pdfL + pdfB);                        // balance
	// forensics: this call is about to contribute. Record WHAT it aimed
	// at and at what grazing angle, in world coords, for views 12/13.
	g_neeDiag = vec4(1.0, c.y + gridOrigin.y, cosX, float(li));
	g_neeDiagK = float(k);
	// f_r = rho/PI for a Lambertian; estimator = w * f_r * Le * cos_x/p_l
	return w * (rho / PI) * shle * (cosX / pdfL);
}

// =====================================================================
// NEE FOR THE SKY — the sun/moon disc as a sampled light
// =====================================================================
//
// The sky is a light, so NEE samples it. Without this, the only way a
// path finds the sun is for a cosine-sampled bounce to land inside a
// cone of a few square degrees carrying hundreds of times the dome's
// radiance — the textbook high-variance case, and outdoors it is the
// whole image.
//
// IT IS A SEPARATE LIGHT, NOT A SEVENTEENTH SLOT IN THE AREA LIST, and
// that is what keeps this step's gate 1 honest. The direct-lighting
// integral splits by light:
//
//     L_direct = SUM_j INT f * L_j * cos dwi
//
// and each light j is MIS-combined with BSDF sampling on its own:
// w_{j,l} = p_{j,l} / (p_{j,l} + p_b) at the light sample, and
// w_{j,b} = p_b / (p_b + p_{j,l}) when the BSDF ray lands on light j.
// So adding a light does not touch any other light's weights: the area
// list's pdf, its k, its faces and its MIS weight are all EXACTLY what
// they were before today. Folding the sky into the uniform-over-N
// selection instead would have divided every emitter's p_A by (N+1) and
// moved every sealed room's estimator — the rooms whose whole job this
// week is to prove they did not move.
//
// The sampled light is the BODY only (skyBody). The dome is a light with
// no light-sampling technique, so its weight is 1 wherever a BSDF ray
// finds it — which is correct and is why skyDome and skyBody are two
// functions that sum to the one miss function.
//
// The technique is UNGATED by the receiver's normal on purpose. It emits
// directions uniformly in the body's cone whatever the surface faces; a
// direction pointing into the surface is rejected by cos_x <= 0 BEFORE
// the shadow ray, so it costs two random numbers and no traversal. That
// makes p_sky a function of the DIRECTION alone, which means the escape
// branch can price it without carrying the previous vertex's normal.
//
// Cost when it runs to completion: two draws from the reserved sky
// stream and one shadow march.

// Which body is the sampler aiming at? At most one: the sun and the moon
// are antipodal in Luanti (sky.cpp differs only 90 vs 270), so when both
// carry a live cone it is a horizon crossing and the sun is the brighter.
// A body the game has switched off, or one below the horizon, arrives
// with the SKY_BODY_OFF sentinel in its cos and is not aimed at.
bool skyNeeBody(out vec3 bdir, out float bcos)
{
	bdir = skySunDir;
	bcos = skySunCos;
	// The uniform TEST sky has no body, so the technique has no pdf.
	// Answered here rather than at the two call sites, so the sampler and
	// the pdf cannot disagree about whether the technique exists.
	if (claudeSkyUniform > 0.0)
		return false;
	if (bcos >= SKY_BODY_OFF) {
		bdir = skyMoonDir;
		bcos = skyMoonCos;
	}
	return bcos < SKY_BODY_OFF;
}

// p_sky(wi) in SOLID ANGLE: uniform over the body's cone. Zero outside
// it, and that zero is load bearing exactly as neePdfSa()'s is — it makes
// w_b = 1 for every direction the sky sampler cannot produce, so the dome
// and any second body arrive at full strength through BSDF sampling.
float skyPdfSa(vec3 wi)
{
	vec3 bdir;
	float bcos;
	if (!skyNeeBody(bdir, bcos))
		return 0.0;
	if (dot(wi, bdir) < bcos)
		return 0.0;
	// solid angle of a cone of half-angle acos(bcos)
	return 1.0 / (PI2 * (1.0 - bcos));
}

// One sky-light sample at vertex (x, nx) with reflectance rho. Returns
// the MIS-weighted direct contribution WITHOUT the path throughput.
vec3 neeSky(vec3 x, vec3 nx, vec3 rho)
{
	vec3 bdir;
	float bcos;
	if (!skyNeeBody(bdir, bcos))
		return vec3(0.0);

	// uniform on the cone
	float u1 = rndSky();
	float u2 = rndSky();
	float cosT = 1.0 - u1 * (1.0 - bcos);
	float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
	float phi = PI2 * u2;
	vec3 ta = abs(bdir.y) > 0.5 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	vec3 tx = normalize(cross(ta, bdir));
	vec3 ty = cross(bdir, tx);
	vec3 wi = normalize(tx * (sinT * cos(phi)) + ty * (sinT * sin(phi))
			+ bdir * cosT);

	float cosX = dot(nx, wi);
	if (cosX <= 0.0)
		return vec3(0.0); // the body is behind this surface: no march

	// VISIBILITY through the one traversal (§2: no lighter copy of the
	// walk for shadow rays). The sky is visible iff the ray LEAVES the
	// grid — march() returning false is exactly that test, and it is the
	// same false the escape branch in main() reads.
	vec3 shp, shn, shalb, shle, shcell;
	float sht;
	if (march(x, wi, shp, shn, shalb, shle, sht, shcell))
		return vec3(0.0); // occluded

	// THE LAW: one sky. skyBody() here is the same evaluation the camera
	// ray's escape runs — there is no second radiance for the shadow ray.
	float pdfL = 1.0 / (PI2 * (1.0 - bcos)); // p_sky, sa
	float pdfB = cosX / PI;                  // p_b, sa
	float w = pdfL / (pdfL + pdfB);          // balance heuristic
	return w * (rho / PI) * skyBody(wi) * (cosX / pdfL);
}

// =====================================================================
// INSTRUMENT A (roadmap step 1a) — the direct-light referee
// =====================================================================
//
// The defect: claude_nee = 1 converges 2-9% hot by region against photo
// mode in Cornell (measured.md "CI red/green"). MIS adds TWO estimates
// of the direct term at every vertex, w_l * (light sample) + w_b * (Le
// found by the BSDF ray), and the sum is supposed to be exactly one copy
// of it. Three things can be wrong: the light half, the BSDF half, or
// the weights. A referee that can only see the SUM cannot say which.
//
// So: three claude_view modes, all measuring the SAME scalar per pixel —
// the direct-light radiance leaving the PRIMARY hit, one bounce, nothing
// else — by three routes that share as little machinery as possible:
//
//   view  9  the light-sampling half alone, w_l forced to 1. This is
//            neeDirect() itself, one extra argument, so it cannot drift
//            from the estimator it is judging.
//   view 10  the BSDF half alone, w_b forced to 1: one cosine-hemisphere
//            sample from the primary hit, and whatever Le it lands on.
//            With p_b = cos/PI and f_r = rho/PI the estimator is exactly
//            `rho * Le(hit)` — no pdf arithmetic at all, which is why it
//            is the useful partner: it shares no pdf code with view 9.
//   view 11  ANALYTIC. No rays, no RNG, no light list: Lambert's contour
//            formula for the irradiance from the Cornell ceiling panel,
//            times rho/PI. This is the only one of the three that cannot
//            be wrong in the same way as the other two.
//
// CONVERGED, ALL THREE MUST AGREE. Whichever disagrees with 11 names the
// broken half; if 9 and 10 both match 11, the halves are sound and the
// bug is in the weights.
//
// WHAT VIEW 11 IS BLIND TO (physics-contract §8 clause 3):
//  * OCCLUSION. It is the unshadowed analytic answer. Cornell's two
//    gray186 blocks shadow parts of the FLOOR, and views 9 and 10 (which
//    both trace) will read darker there. The ratio image shows those as
//    black patches; they are the instrument, not the defect.
//  * ANY ROOM BUT CORNELL. The rectangle below is a hard-coded world
//    -space constant taken from util/claude_bridge_gallery.lua.
//  * A RECEIVER WHOSE HORIZON PLANE CUTS THE PANEL. The contour formula
//    is exact only for a polygon entirely above the receiver's horizon.
//    Every interior surface of Cornell satisfies that (the panel sits
//    inside the x/z span of every wall and above the floor); the SIDE
//    faces of the two occluder blocks do not, and are wrong there.
//  * Views 9 and 10 are Monte Carlo and need depth; view 11 is exact at
//    one sample. A disagreement at low N is noise, not a finding.
//
// Views 9-11 IGNORE claudeNee on purpose: they are the halves of the
// estimator, not the transport dial, and a forgotten claude_nee = 1
// would otherwise render view 9 silently black — a blind instrument.
// They also ignore claudeBounces: depth is fixed at one bounce.
// They ACCUMULATE and tonemap exactly like view 0, deliberately: the
// quantity is linear radiance, and claude_cornell_check.py's referee
// inverts precisely that transform, so the region ratios are read by
// the same instrument that judges every other Cornell frame, with no
// second code path to keep honest.

// CORNELL-ONLY CONSTANT, and it is flagged everywhere it is used.
// util/claude_bridge_gallery.lua OPS.cornell at pos (43,8,0), size 7:
// the 3x3 white_lit panel is FLUSH in the ceiling layer at world nodes
// x 46..48, y 16, z 3..5. Its one air-exposed downward face is the
// rectangle x in [46,49], z in [3,6] in the plane y = 16, WORLD node
// coords — the DDA's cell-index space is that minus gridOrigin.
const vec3 PANEL_WMIN = vec3(46.0, 16.0, 3.0);
const vec3 PANEL_WMAX = vec3(49.0, 16.0, 6.0);
// A cell of the panel, world node coords, for reading Le off the grid.
const vec3 PANEL_WCELL = vec3(47.0, 16.0, 4.0);

// THE LAW IS ONE Le (§4): read it out of the same texture through the
// same cellEmission() the eye ray runs, rather than restating 2.4 here.
vec3 panelLe()
{
	vec3 c = PANEL_WCELL - gridOrigin;
	vec4 s = texture3D(claudeTraceGrid, (c + 0.5) / GRID_S);
	return cellEmission(s.a, cellAlbedo(s.rgb));
}

// Lambert's contour formula: the irradiance at x with normal n from a
// uniform-radiance polygon of unit radiance,
//     E = (1/2) * SUM_i gamma_i * dot(n, Gamma_i)
// gamma_i = the angle edge i subtends at x, Gamma_i = the unit normal of
// the plane through x and edge i. Exact for a polygon wholly above the
// horizon at x (see the blind list above). Vertices are wound so the
// result is positive for a receiver BELOW a downward-facing rectangle.
float rectIrradiance(vec3 x, vec3 n, vec3 p0, vec3 p1, vec3 p2, vec3 p3)
{
	vec3 r0 = p0 - x, r1 = p1 - x, r2 = p2 - x, r3 = p3 - x;
	vec3 u0 = normalize(r0), u1 = normalize(r1);
	vec3 u2 = normalize(r2), u3 = normalize(r3);
	float e = acos(clamp(dot(u0, u1), -1.0, 1.0))
			* dot(n, normalize(cross(r0, r1)));
	e += acos(clamp(dot(u1, u2), -1.0, 1.0))
			* dot(n, normalize(cross(r1, r2)));
	e += acos(clamp(dot(u2, u3), -1.0, 1.0))
			* dot(n, normalize(cross(r2, r3)));
	e += acos(clamp(dot(u3, u0), -1.0, 1.0))
			* dot(n, normalize(cross(r3, r0)));
	return max(0.5 * e, 0.0);
}

// view 11: rho/PI * E from the Cornell panel. CORNELL ONLY.
vec3 panelDirect(vec3 x, vec3 nx, vec3 rho)
{
	vec3 pmin = PANEL_WMIN - gridOrigin;
	vec3 pmax = PANEL_WMAX - gridOrigin;
	// the panel emits DOWNWARD only (its -Y face is the exposed one), so
	// a receiver on or above its plane sees nothing from it
	if (x.y >= pmin.y)
		return vec3(0.0);
	float y = pmin.y;
	float e = rectIrradiance(x, nx,
			vec3(pmin.x, y, pmin.z), vec3(pmin.x, y, pmax.z),
			vec3(pmax.x, y, pmax.z), vec3(pmax.x, y, pmin.z));
	return (rho / PI) * panelLe() * e;
}

// ---------------------------------------------------------------------

void main(void)
{
	vec2 uv = varTexCoord.st;
	if (gridDebug < 2.5) {
		// traced mode off: carry history through untouched
		gl_FragColor = texture2D(history, uv);
		return;
	}

	int view = int(claudeView + 0.5);
	int maxBounces = int(claudeBounces + 0.5);
	// nLights is the ONE gate on the whole rung-2 block. 0 means the pure
	// path: with claudeNee = 0, or an empty/invalid list, not a single
	// line below behaves differently from rung 1 — including the RNG draw
	// order, which is what makes the A/B a real A/B and not two images
	// that merely look alike.
	int nLights = claudeNee > 0.5
			? min(int(claudeAreaCount + 0.5), AREA_CAP) : 0;
	// The sky's own light-sampling technique rides the SAME dial and
	// nothing else. It is deliberately NOT gated on nLights: outdoors
	// there are frequently no listed emitters at all, and that is exactly
	// the scene where sampling the sun matters most.
	bool skyNee = claudeNee > 0.5;

	g_rngState = hash13(vec3(gl_FragCoord.xy,
			// animationTimer is unbounded seconds; wrapped by an
			// irrational multiplier so consecutive frames land far
			// apart in the hash's domain and no frame rate aliases it
			fract(animationTimer * 91.7) * 1024.0));
	// the counter-based key, seeded from the same three facts
	g_rngKey = pcgHash(uint(gl_FragCoord.x)
			^ (uint(gl_FragCoord.y) << 11u)
			^ (uint(fract(animationTimer * 91.7) * 65536.0) << 22u));
	g_rngCtr = 0u;
	g_skyCtr = 0u;

	// sub-pixel jitter: free anti-aliasing through the running average.
	// Off in views 1-5, which do not accumulate and would otherwise
	// flicker along every silhouette.
	float j0 = rnd1();
	float j1 = rnd1();
	vec2 jit = (vec2(j0, j1) - 0.5) * texelSize0;
	// views 1-5 are deterministic ladders and would flicker along every
	// silhouette; 0, 6 and the 9-11 instruments all accumulate radiance
	// and want the free anti-aliasing (and want it identically, so the
	// three instrument views can be divided pixel by pixel).
	if (view >= 1 && view <= 5)
		jit = vec2(0.0);

	vec2 ndc = (uv + jit) * 2.0 - 1.0;
	vec3 rd = normalize(gridCamFwd
			+ ndc.x * gridCamRight + ndc.y * gridCamUp);
	// +0.5: gridCamPos is node-CENTRED (game.cpp: cam/BS - origin),
	// the DDA works in cell-index space where cell i spans [i, i+1)
	vec3 ro = gridCamPos + 0.5;

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

	// MIS bookkeeping for the BSDF strategy: the vertex the current ray
	// left from, and that ray's solid-angle density there. misArmed is
	// false on the camera ray on purpose — an eye ray is not a strategy
	// the light sampler competes with, so a directly-visible emitter is
	// added at full strength in BOTH modes. All three are inert when
	// nLights is 0.
	vec3 prevX = vec3(0.0);
	float prevPdfB = 0.0;
	bool misArmed = false;

	// --- INSTRUMENT: claude_view 18, DOES A BOUNCE RAY START INSIDE A
	// SOLID CELL? ----------------------------------------------------
	// Built 2026-08-17 for a defect the sky REVEALED rather than caused:
	// cave-glass, a sealed 7x5x7 box whose golden is black, put a few
	// hundred pixels of sky on screen. THIS IS NO LONGER A HYPOTHESIS —
	// the view found the mechanism, then named the cause, and the fix is
	// restartPoint(). What follows is what it was.
	//
	// The view asks one question and nothing else: take the primary hit,
	// draw one cosine-hemisphere bounce the way the path loop does, and
	// report whether the cell that ray STARTS in is solid. That matters
	// because march() never tests the cell a ray starts in — the
	// exclusion that keeps a bounce ray off its own face — so a ray
	// starting inside a wall is waved straight out through it.
	//
	// WHAT IT WAS, measured 2026-08-17 (util/claude_leak_probe.py), both
	// plain-cube rooms, marker excluded:
	//   rate      cave-glass 3.7e-07 / 784 px ever positive
	//             cornell    9.9e-07 / 1812 px
	//   sideways  1.000 in BOTH rooms
	//   across    1.000 in BOTH rooms
	// i.e. in 100 % of events the cell floor() chose was the cell across
	// the hit face PLUS a step sideways, into a neighbour of the cell
	// that was hit — and the restart point sat BIT-EXACTLY on that
	// neighbour's boundary plane (depth 0 to the instrument's 1e-12
	// floor, measured before these two channels replaced it). The
	// epsilon moves along n and only along n, so a tangential
	// disagreement is not a short epsilon, not a grazing angle and not a
	// wrong normal: it is a hit on the EDGE of a face, where the
	// tangential coordinate is within half a float32 ulp of an integer
	// and floor() rounds it into the wall next door.
	//
	// The verdict was never this rate — a rate can be made rarer by a bad
	// fix and still leak given enough frames. The verdict is cave-glass
	// under a 50x uniform test sky being EXACTLY black at every bounce
	// depth, which is analytic: a room with no opening receives no sky.
	// The ladder that got here, one variable at a time, at a 2000-frame
	// settle:
	//   claudeBounces 0  -> 0 leaking pixels, from twelve directions.
	//                       No EYE ray escapes; the room is sealed.
	//   claudeBounces 1  -> 144 px. The FIRST bounce is enough.
	//                 2  -> 320 px.   4 -> 1072 px. More depth, more
	//                       chances at the same mechanism.
	//   claudeDescend 0  -> unchanged. Not the sub-voxel walk.
	//   leak is LINEAR in sky radiance, so it is rays getting out.
	//
	// AND THE COUNT DEPENDS ON HOW LONG YOU LOOK, which is the sharpest
	// reason a rate is not a verdict: at a 500-frame settle the SAME
	// build reads 0 px at claudeBounces 1 and 516 at 4. Each leaked pixel
	// is one frame's escaped sample buried in a running average, so a
	// shallow settle hides the defect without changing it.
	//
	// WHAT THIS INSTRUMENT IS BLIND TO: a CLASS-250 cell is "solid" to
	// this test and is not a defect. The walk is SUPPOSED to be inside
	// one -- that is what descending means -- so every sub-voxel hit
	// reports positive, which is why the cabin, full of authored models,
	// reads 46 % of the frame (3.3e-03 mean, 963124 px). Read this view
	// only in rooms built from plain cubes until it learns to ask the
	// 16^3 mask the same question.
	if (view == 18) {
		vec3 hp, n, alb, le, cell;
		float tHit;
		float bad = 0.0;      // R: the event happened
		float sideways = 0.0; // G: c0 also differs TANGENTIALLY to the face
		float across = 0.0;   // B: c0 differs by exactly +n on the face's
		                      //    own axis (i.e. it stepped through it)
		if (march(ro, rd, hp, n, alb, le, tHit, cell)) {
			float u1 = rnd1();
			float u2 = rnd1();
			vec3 wi = cosineHemisphere(n, u1, u2);
			// the walk's own opening line, verbatim
			vec3 c0 = floor(hp);
			if (all(greaterThanEqual(c0, vec3(0.0)))
					&& all(lessThan(c0, vec3(GRID_S)))) {
				vec4 s0 = texture3D(claudeTraceGrid, (c0 + 0.5) / GRID_S);
				if (s0.a > CLASS_AIR_MAX)
					bad = 1.0;
			}
			// WHICH WAY c0 DISAGREES with the cell that was hit. The
			// restart moved along n and along n ONLY, so the walk's own
			// answer for "which cell is the ray in now" is cell + n --
			// the cell it came THROUGH, which it tested and found empty.
			// Splitting the disagreement on n's axis from the two
			// tangential ones is what separates "the epsilon failed to
			// clear the face" from "the point slid sideways into a
			// neighbour of the cell it hit", i.e. a concave corner.
			vec3 d = c0 - cell;
			across = bad * ((abs(dot(d, n) - 1.0) < 0.5) ? 1.0 : 0.0);
			vec3 dt = d - n * dot(d, n);
			sideways = bad * ((dot(abs(dt), vec3(1.0)) > 0.5) ? 1.0 : 0.0);
			// wi is drawn but unused except to keep the RNG consumption
			// identical to the path loop's, so the two see the same
			// directions at the same pixels.
			bad *= (dot(wi, n) >= 0.0) ? 1.0 : 1.0;
		}
		// Accumulates like view 5, so the per-pixel MEAN over frames is
		// the RATE -- the number worth having for an event this rare.
		// x1000 because a rate of 1e-6 would otherwise land below the
		// first display step; divide the inverted linear value by 1000
		// to read the rate back. All three channels carry the same
		// scale, so G/R and B/R are read as plain fractions of the
		// events and need no scale of their own.
		L = vec3(bad, sideways, across) * 1000.0;
		maxBounces = -1;
	}

	// --- INSTRUMENT: claude_view 17, THE WHOLE SKY IN ONE FRAME -------
	// skyRadiance() alone, over the FULL SPHERE, with no geometry, no
	// camera and no transport: the screen is a lat-long chart of the miss
	// function. x is azimuth 0..360 deg, y is elevation -90 (bottom) to
	// +90 (top), so the horizon is the middle row and the zenith is the
	// top edge.
	//
	// It exists BEFORE it is needed, on purpose. The two-miss rule says
	// two failed fixes on one sky bug and no third patch until an
	// isolating instrument exists; this is that instrument, built with
	// the feature rather than after the second miss. It answers "what
	// does the sky function actually return" without a scene, a settle or
	// a shadow ray in the way — and because it is the SAME function the
	// path calls, it cannot measure a private copy.
	//
	// Read it as BRIGHTNESS, not colour: the sun is the small blown-out
	// spot, the moon the smaller one opposite, the dome a vertical ramp,
	// and the bottom half is black because a ray below the horizon leaves
	// with nothing.
	if (view == 17) {
		float az = uv.x * PI2;
		float el = (uv.y - 0.5) * PI;
		float ce = cos(el);
		vec3 d = vec3(ce * sin(az), sin(el), ce * cos(az));
		gl_FragColor = vec4(skyRadiance(d), 1.0);
		return;
	}

	// --- INSTRUMENT A (roadmap 1a): claude_view 9 / 10 / 11 -----------
	// One primary hit, one direct-light term, no path. Design, and the
	// list of what each of the three is blind to, above panelDirect().
	// Nothing here runs at claude_view 0: the truth path is untouched.
	if (view >= 9 && view <= 16) {
		vec3 hp, n, alb, le, cell;
		float tHit;
		if (march(ro, rd, hp, n, alb, le, tHit, cell)) {
			primaryHit = true;
			primaryT = tHit;
			// the light list, read INDEPENDENTLY of claudeNee: view 9 is
			// the estimator's own half, not the transport dial, and a
			// forgotten dial must not render it silently black.
			int nInstr = min(int(claudeAreaCount + 0.5), AREA_CAP);
			if (view == 9 || view == 12 || view == 13) {
				// the light-sampling half, w_l forced to 1
				if (nInstr > 0)
					L = neeDirect(hp, n, alb, nInstr, 0.0);
				// FORENSICS (12/13). Scratch views: they answer "what did
				// the aimed sampler aim at, from this pixel" when the
				// answer is not readable off the code. Each channel is a
				// MEAN over frames, so divide the 2nd and 3rd by the 1st
				// (the hit rate) to get the mean over SUCCESSFUL samples.
				//   12 = (hit rate, emitter world y / 64, cos_x)
				//   13 = (hit rate, slot index / 16, k / 6)
				if (view == 12)
					L = vec3(g_neeDiag.x, g_neeDiag.y / 64.0, g_neeDiag.z);
				else if (view == 13)
					L = vec3(g_neeDiag.x, g_neeDiag.w / 16.0,
							g_neeDiagK / 6.0);
			} else if (view == 10) {
				// the BSDF half, w_b forced to 1. p_b = cos/PI against
				// f_r = rho/PI leaves exactly rho: no pdf arithmetic,
				// which is what makes it an independent witness.
				float u1 = rnd1();
				float u2 = rnd1();
				vec3 wi = cosineHemisphere(n, u1, u2);
				vec3 shp, shn, shalb, shle, shcell;
				float sht;
				if (march(hp, wi, shp, shn, shalb, shle, sht, shcell))
					L = alb * shle;
			} else if (view == 16) {
				// view 10 WITH THE TRAVERSAL TAKEN OUT. Same draws, same
				// cosineHemisphere(), same rho*Le estimator — but the
				// "did this ray find the light" question is answered by
				// intersecting the panel rectangle in closed form
				// instead of by march(). One variable between 10 and 16.
				// 16 == 11 and 10 low  =>  the DDA is losing hits.
				// 16 == 10             =>  the direction sampler is.
				// CORNELL ONLY, and unshadowed, exactly like view 11.
				float u1 = rnd1();
				float u2 = rnd1();
				vec3 wi = cosineHemisphere(n, u1, u2);
				vec3 pmin = PANEL_WMIN - gridOrigin;
				vec3 pmax = PANEL_WMAX - gridOrigin;
				if (wi.y > 1e-6 && hp.y < pmin.y) {
					float t = (pmin.y - hp.y) / wi.y;
					vec3 q = hp + wi * t;
					if (q.x >= pmin.x && q.x <= pmax.x
							&& q.z >= pmin.z && q.z <= pmax.z)
						L = alb * panelLe();
				}
			} else if (view == 11) {
				L = panelDirect(hp, n, alb); // CORNELL ONLY
			} else {
				// RNG DENSITY (14/15). The BSDF half of MIS estimates
				// the direct term as rho * Le(cosine-sampled hit), with
				// NO pdf arithmetic — so if it disagrees with the
				// analytic answer, either the traversal misses hits or
				// the DIRECTION SAMPLER's density is wrong. These two
				// views measure the sampler's own numbers and nothing
				// else: same draw positions as view 10 (draws 3 and 4 of
				// the per-pixel chain, straight after the two jitter
				// draws), each channel scaled so its expected value is a
				// number in the middle of the display range rather than
				// a tail that quantisation swallows.
				//   14 = (5*[u1<0.1], 25*[u1<0.02], 5*[u2<0.1])
				//        all three EXPECT 0.500 for a uniform draw.
				//        cos_theta = sqrt(1-u1), so SMALL u1 is the
				//        zenith — which is where Cornell's panel is from
				//        the floor, i.e. exactly the tail that decides
				//        whether the BSDF ray finds the light.
				//   15 = (u1, sqrt(1-u1), 2*u1*u1)
				//        EXPECT 0.500, 0.6667 (E[cos] over a cosine
				//        hemisphere), 0.6667 (2*E[u1^2] = 2/3).
				float u1 = rnd1();
				float u2 = rnd1();
				if (view == 14)
					L = vec3(u1 < 0.1 ? 5.0 : 0.0,
							u1 < 0.02 ? 25.0 : 0.0,
							u2 < 0.1 ? 5.0 : 0.0);
				else
					L = vec3(u1, sqrt(max(0.0, 1.0 - u1)),
							2.0 * u1 * u1);
			}
		}
		// skip the path loop entirely rather than re-indent it: seg 0 is
		// already past a cap of -1, so the loop below runs zero times.
		maxBounces = -1;
	}

	for (int seg = 0; seg <= BOUNCE_CAP; seg++) {
		if (seg > maxBounces)
			break;

		vec3 hp, n, alb, le, cell;
		float tHit;
		if (!march(p, dir, hp, n, alb, le, tHit, cell)) {
			// ESCAPED THE GRID — and since 2026-08-17 that is not black.
			// The ray sees the sky, through the same skyRadiance() the
			// camera ray and the NEE shadow ray use.
			//
			// The dome and the body carry DIFFERENT MIS weights and that
			// is not a special case, it is the per-light rule: the body
			// is a light the sampler aims at, so a BSDF ray that lands on
			// it gets w_b = p_b/(p_b + p_sky); the dome is a light with
			// no sampler, so it gets 1. misArmed is false on the camera
			// ray, so a directly-VIEWED sky is added at full strength in
			// both modes — nothing aimed at it on the eye's behalf.
			float misW = 1.0;
			if (misArmed && skyNee) {
				float pdfL = skyPdfSa(dir);
				float denom = prevPdfB + pdfL;
				misW = denom > 0.0 ? prevPdfB / denom : 1.0;
			}
			L += tp * (skyDome(dir) + misW * skyBody(dir));
			break;
		}

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
		//
		// Under NEE this Le arrived by the BSDF strategy, and the light
		// sampler at the previous vertex was already paid its share of
		// the same Watt — so it carries the balance weight w_b here.
		// Adding it at full strength on top of the NEE term is THE
		// double count, the one failure mode this block exists to avoid.
		// misW stays exactly 1.0 when nothing competed: the camera ray,
		// an unlisted emitter, an empty list, claudeNee = 0.
		float misW = 1.0;
		if (misArmed && any(greaterThan(le, vec3(0.0)))) {
			float cosY = dot(n, -dir);
			float pdfL = neePdfSa(cell, n, prevX, tHit, cosY, nLights);
			// A zero denominator means neither strategy claims a density
			// for this direction, which can only happen at a degenerate
			// cos; fall back to 1 rather than let a NaN into the history.
			float denom = prevPdfB + pdfL;
			misW = denom > 0.0 ? prevPdfB / denom : 1.0;
		}
		L += tp * misW * le;

		if (seg == maxBounces)
			break; // depth cap: no scatter from this vertex
		if (view >= 1 && view <= 4)
			break; // first-hit views need nothing past the primary

		// NEXT-EVENT ESTIMATION at this vertex, before the throughput
		// absorbs alb: neeDirect() carries its own rho/PI, and it must be
		// the rho this vertex actually reflects with — which in clay
		// (view 6) is CLAY_RHO, clamped above. The depth cap cuts this
		// term at the same vertex it cuts the BSDF half, so claudeBounces
		// means the same thing under either dial.
		if (nLights > 0)
			L += tp * neeDirect(hp, n, alb, nLights, 1.0);
		// The sky's own light sample, at the same vertex and under the
		// same depth cap, drawing from the RESERVED counter range so the
		// path's own random sequence is untouched (see rndSky()).
		if (skyNee)
			L += tp * neeSky(hp, n, alb);

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
		// Arm the BSDF half of the MIS pair. Russian roulette above does
		// not enter these pdfs: it scales the estimate by 1/q on the
		// survivors, which leaves the SAMPLING DENSITY of the direction
		// untouched, and the weights are densities.
		prevX = hp;
		prevPdfB = max(dot(n, dir), 0.0) / PI;
		// Armed when ANY light sampler ran at this vertex: the area list,
		// the sky, or both. Outdoors the list is often empty and the sky
		// is the only light there is — leaving this at `nLights > 0`
		// would have given the sun's BSDF half a weight of 1 while the
		// sky sampler was also paying it, i.e. the double count.
		misArmed = nLights > 0 || skyNee;
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
	// views 1-5 only: 6 (clay) and the 9-11 instruments are radiance
	// and fall through to the accumulator below.
	if (view >= 1 && view <= 5) {
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
	// grid rebase (history discarded), 0.5 while moving, 1/(2+N) at
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
