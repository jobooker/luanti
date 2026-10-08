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
//    Those cells are the nub and the fine materials, whose palette
//    emission column is 0, so there is nothing to sample and nothing to
//    double-count.
//  * irradiance caches, face caches, radiance lattice: deleted.
//  * spatial denoising: deleted. Noise is resolved by convergence only.
//  * reprojection: not done. History is read at the SAME uv. Camera
//    motion is handled entirely by accumAlpha (0.5 moving, 1.0 on
//    teleport/grid-rebase), so motion smears over ~2 frames and rest
//    converges exactly. Photo mode is a parked-camera instrument.
//  * cell sizes other than 1 m: LANDED 2026-08-16 for the sub-voxel
//    ring. march() descends into a FINE material's cell and continues the
//    same DDA through its 16^3 mask at 1/16 m (unit 7), so stairs,
//    slabs, beds and every authored model are real sub-metre geometry
//    for EVERY ray type — eye, bounce and shadow — because there is
//    still exactly one traversal. What remains punted:
//      - outside the ring [48,80)^3 the mask does not exist, so a fine
//        cell is still a 1 m cube. That is a §2 ladder (a coarser rung
//        at distance), which §7 permits; it is not a different light
//        law.
//      - SUB-VOXEL CELLS EMIT NOTHING, and since 2026-08-18 that is a
//        ZERO IN ONE COLUMN OF ONE TABLE rather than a consequence of
//        where 250 fell between two thresholds. game.cpp's
//        claudeMatEmission() returns 0 for a fine material, so the
//        campfire, the lantern and every modelled torch have their true
//        SHAPE and no glow. Changing that is a change to the emission
//        law's DOMAIN (§4 admits exactly one law), so it is a decision
//        and it gets its own commit — not a shader edit, and never a
//        second emission path.
//      - per-sub-voxel COLOUR is likewise not read. A sub-voxel hit
//        takes its cell's stored colour, so a chest is chest-shaped in
//        one albedo. claudeModelAtlas / claudeModelPal stay unread.
//  * transmissive materials: water, leaves and glass are distinct
//    MATERIALS in the palette but their transmission column is 0, so they
//    are opaque Lambertian in rung 1. §3 lists them as out of scope, and
//    the palette is where they stop being so.
//  * the point-light "nub" material is opaque and NON-emissive here:
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

uniform sampler2D history;
// THE DIRECT/BOUNCED SPLIT (claude_split, 2026-10-05, John: "not sure if
// the first frame could be rendered with one bounce, then more?"). A
// second running average holding only the light that reached the camera
// after at most ONE diffuse or air scatter, written to colour attachment 1
// and read back here as texture1. claude_present shows direct + bounced x
// a weight that ramps 0 -> 1 over claude_split frames of stillness, so the
// clean part is there at once and the slow part fades in; past the ramp
// the display is exactly the total, as before. The total is untouched.
#define historyDirect texture1
uniform sampler2D historyDirect;
// THE DENOISER'S INPUTS (claude_denoise, 2026-10-05), two more running
// averages on the same blend weight as the radiance, written to colour
// attachments 2 and 3:
//   historyGbuf  rgb = the primary hit's albedo (accumulated, so a pixel
//                      straddling two texels divides by the same mix it
//                      averaged); a = WHICH VOXEL FACE the pixel sees,
//                      this frame: 1 + (axis*2 + sign)*4096 + plane in
//                      1/16 m, or 0 = never filter (sky, glass, water)
//   historyMom   r = mean of the squared luminance of this frame's
//                    texture-free (demodulated) sample; g = how much of
//                    one sample's variance is still in the average, for
//                    ANY blend schedule: v' = (1-a)^2 v + a^2 (1/N for a
//                    true average, a/(2-a) for a moving EMA floor)
// Every surface in this world is an axis-aligned voxel face at a 1/16 m
// plane, so "same surface" is an exact test here, not a depth heuristic.
#define historyGbuf texture2
uniform sampler2D historyGbuf;
#define historyMom texture3
uniform sampler2D historyMom;
#ifdef CLAUDE_MRT_OK
layout(location = 1) out vec4 outDirect;
layout(location = 2) out vec4 outGbuf;
layout(location = 3) out vec4 outMom;
#define CLAUDE_SPLIT_OUT 1
#endif      // previous frame's accumulated radiance
// EMPTY-SPACE SKIPPING (claude_pyramid, 2026-10-06). game.cpp has built
// and uploaded an occupancy pyramid since 2026-08-12 -- level 0 = "this
// cell's material is not air" (the same byte the walk tests), levels 1..5
// = "anything non-air in this 2^L block" -- and until now nothing read it:
// the walk crossed every empty cell one at a time (~200 steps per ray,
// claude_view 21). John: "why does a ray have to cross each empty block of
// air? It's supposed to skip ahead."
uniform sampler3D claudeCoarse;   // unit 11: R8 128^3, mips 0..5
uniform float claudePyramid;      // 1 = leap across empty blocks
uniform sampler3D claudeTraceGrid; // unit 10: RGBA8 128^3, rgb = cell colour,
                                // a = MATERIAL INDEX / 255 (see matIndex)
// THE MATERIAL PALETTE, unit 20: 256x1 RGBA32F, one texel per material
// index. This is what the per-cell byte points AT — see matPal() for the
// channel layout and for why the fetch is not in the walk's hot path.
uniform sampler2D claudeMatPal;
// THE ROUND-TRIP PROBE, unit 21: 256x1x1 RGBA8, texel n carrying the byte
// n, in the SAME format and with the same NEAREST/CLAMP parameters as the
// trace grid. Read ONLY by claude_view 19, which is what keeps it out of
// every other frame and what keeps this sampler out of the census's dead
// list — a declared-but-unread sampler is stripped and reports as DEAD.
uniform sampler3D claudeMatProbe;
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
// 1 = march() descends into a fine material's cell (the default); 0 = the
// pre-2026-08-16 behaviour, in which such a cell is an opaque 1 m cube.
// The A/B partner for the energy and cost gates.
uniform float claudeDescend;
// GLASS SITS FLUSH (claude_glass_flush, 2026-10-04, John's call). 1 = a
// ray travelling in glass that arrives at a carved OPAQUE cell stops at
// that cell's 1 m face instead of descending into its relief. See the
// fineHere gate in marchMed().
uniform float claudeGlassFlush;
// WATER ABSORBS (claude_water_absorb, 2026-10-05). 1 = a ray inside a
// medium loses exp(-a d) of its throughput over the d metres it travels
// there, a = the palette's row 1 (game.cpp CLAUDE_WATER_ABSORB_*: pure
// water, measured). 0 = lossless media, as before.
uniform float claudeWaterAbsorb;
// REAL LIGHT UNITS (claude_units, 2026-10-05). Radiance unit = 1000 cd/m2.
// HOT THINGS glow by their temperature (luanti-docs
// spec/photometric-sources.md §5-6; colours = Planck through the CIE 1931
// observer into linear sRGB, normalised to unit luminance, negative blue
// clamped to 0):
//   FLAME  a candle's measured bright-zone luminance, 1-2e4 cd/m2
//          (Hollan / Schlyter), at its ~1900 K colour. A soot flame is
//          thin: a 1900 K blackbody would be 2.4e5, so its emissivity is
//          ~0.06. ONE measured flame stands for every flame (torch,
//          lantern, campfire, furnace, fire): no wood-fire luminance was
//          found.
//   LAVA   Kilauea's thermocouple 1140-1150 C, emissivity 0.95 (Fresnel,
//          basaltic glass n = 1.55): 3,069 cd/m2. It REFLECTS 4.65 %.
uniform float claudeUnits;
// LEAVES TRANSMIT (claude_leaf_transmit, 2026-10-05). A leaf's
// transmittance spectrum closely matches its reflectance spectrum (Xu &
// Ye 2023, Sci. Rep. 13:4972, "solar spectral reflectance and
// transmittance of natural leaves exhibit dramatic similarity"; after
// Knipling 1970). So a fine leaf voxel is a thin Lambertian sheet with
// tau = rho = its albedo (clamped to 0.5 so rho + tau <= 1). See the path
// loop. Only inside the sub-voxel ring: a coarse 1 m leaf cube is not a
// sheet and stays opaque.
uniform float claudeLeafTransmit;
// the denoiser's face code for "this pixel sees the sky" (see outGbuf)
const float SKY_FACE_CODE = 1.0 + 6.0 * 65536.0;
// MIS: the BSDF half's density at an NEE vertex, as a multiple of cos/pi
// (0.5 at a leaf, which sends half its paths to each side)
float g_neeBScale = 1.0;
const vec3 FLAME_RGB = vec3(2.6389, 0.6135, 0.0);
const float FLAME_L = 15.0;
const vec3 LAVA_RGB = vec3(3.42, 0.3886, 0.0);
const float LAVA_L = 3.069;
const float LAVA_RHO = 0.0465;
// FACE-TILE COLOUR (claude_texel_colour, 2026-10-04). claudeMaterials is
// the per-cell material id (unit 12, R8 128^3, 0 = none); claudeAtlas
// (unit 13, 256x768) holds three 16x16 tiles per id — top, bottom, side —
// each texel a RATIO x64 to the cell colour (game.cpp
// claudeAtlasFaceTiles). Read only at a plain cube's hit: carved models
// keep one colour per cell until their own delit palettes are wired
// (roadmap SUB-VOXEL COLOUR; a raw tile on a carved groove darkens it
// twice — bf83f1108).
uniform sampler3D claudeMaterials;
uniform sampler2D claudeAtlas;
uniform float claudeTexel;
// AIR (2026-10-04, John: "add air/haze as step one"). A homogeneous medium
// filling every AIR cell of the grid: scattering and absorption
// coefficients in 1/m, and the Henyey-Greenstein asymmetry g (0 = even,
// toward 1 = forward, like haze). 0 + 0 = no medium, and then nothing in
// this file draws a random number for it, so an air-free frame is the
// frame it always was. Beyond the grid there is no medium (the far field
// is one flat number; aerial perspective past 64 m is not modelled).
uniform float claudeAirScatter;
uniform float claudeAirAbsorb;
uniform float claudeAirG;
// FLAME-ONLY EMISSION (claude_flame, 2026-10-05, DECISIONS 0e). The model
// tables uploaded since August and read by nothing until now: per-cell
// model id ((model << 2) | rot, unit 16), the 16x16x1024 palette-index
// atlas (layer = (model*4 + rot)*16 + sz, unit 17) and the 256x64 palette
// (alpha = emit/15; slot 0 carries K*256 in R,G — game.cpp, unit 18).
uniform sampler3D claudeModelIds;
uniform sampler3D claudeModelAtlas;
uniform sampler2D claudeModelPal;
uniform float claudeFlame;


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
// REPROJECTION (claude_reproject, 2026-10-05). The previous frame's camera
// in the CURRENT grid's coordinates (game.cpp keeps it in world units, so a
// grid rebase cannot confuse it), same pre-scaling as above. While the
// camera MOVES, each pixel finds where its primary hit was on the previous
// frame's screen, checks the stored primary distance agrees, and carries
// that history forward with weight max(1/(N+1), claudeMotionAlphaMin) — a
// bounded memory (~1/alpha frames) so moving light still updates. N, the
// pixel's own sample count, rides in the direct buffer's alpha. A PARKED
// camera takes the old path exactly, so photo mode and every capture are
// what they were.
uniform vec3 claudePrevCamPos;
uniform vec3 claudePrevCamFwd;
uniform vec3 claudePrevCamRight;
uniform vec3 claudePrevCamUp;
uniform float claudeReproject;
uniform float claudeMotionAlphaMin;

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
// The frame index claudeRng = 2 keys on (game.cpp m_rng_frame_pixel).
uniform float claudeRngFrame;

// AREA-EMITTER LIST for next-event estimation (game.cpp
// claudeTraceGridSnapshot, ClaudeTraceGrid::area). One emissive CELL per slot:
//   xyz = integer grid-cell coords, the DDA's own cell space
//   w   = air-exposed face mask, bit 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z
// NO RADIANCE RIDES ALONG, on purpose: THE LAW is one Le (§4), so the
// shadow ray's own march() hit supplies it via cellEmission(). A second
// copy of Le on the CPU is the divergence the contract forbids.
uniform float claudeAreaCount; // live slots, 0..AREA_CAP; 0 = no NEE
// SMALL EMITTERS FOR NEE (2026-10-05): the 8 nearest point emitters --
// torch and lantern flames, campfires, a lit furnace's fire -- each as
// (centre in grid cells, radius of a sphere holding EVERY emitting voxel).
// See neePoint(). claudePointCount = live slots; 0 = off.
uniform vec4 claudeEmitter0;
uniform vec4 claudeEmitter1;
uniform vec4 claudeEmitter2;
uniform vec4 claudeEmitter3;
uniform vec4 claudeEmitter4;
uniform vec4 claudeEmitter5;
uniform vec4 claudeEmitter6;
uniform vec4 claudeEmitter7;
uniform float claudePointCount;
uniform float claudeTorchNee;   // claude_torch_nee: 1 = aim at flames
// claude_subpasses (2026-10-07): this program is extra-sample pass k when
// CLAUDE_SUBPASS = k > 0 (secondstage.cpp compiles it twice more)
#ifndef CLAUDE_SUBPASS
#define CLAUDE_SUBPASS 0
#endif
uniform float claudeGuideImpl;    // claude_guide_impl: 1 = v1 (walk, 3 atomics, every path writes), 2 = v2 (alias, bin only, claude_guide_keep)
uniform float claudeGuideKeep;    // claude_guide_keep: share of paths that write to the tables
uniform float claudeGuideDeposit; // claude_guide_deposit: 0 = the guide reads its tables but writes nothing (price instrument)
#if defined(GL_ARB_shader_storage_buffer_object) && defined(GL_ARB_gpu_shader5) && defined(GL_ARB_shading_language_420pack)
// THE GENERAL LADDER'S TREE (ladder-plan stage 2, 2026-10-07): 3 uints per
// node (child mask lo, hi, first child or FULL), root at 0. Built by
// game.cpp claudeTreeBuild() from the tracer's own textures.
layout(std430, binding = 3) readonly buffer ClaudeTreeBuf { uint claudeTree[]; };
// the gate's tally, counted HERE and not from a screenshot (stage 1b's
// lesson: the display pass blends neighbours, so a picture of verdicts is
// not the verdicts). [0..6] one count per verdict, [8] disagreements seen,
// [9 + 19 k ..] the first 128 of them in full. Cleared by game.cpp per frame.
layout(std430, binding = 4) buffer ClaudeTreeCountBuf { uint claudeTreeCount[]; };
#define CLAUDE_TREE_OK 1
#endif
// planted defect for the gate (claude_tree_plant 1): the tree walk starts
// 1/64 of a base piece off, so the tally MUST show disagreements
uniform float claudeTreePlant;
uniform float claudeRawFrame;   // claude_raw_frame: 1 = no history, every frame shows only its own rays (the layered comparison)
uniform float claudeBoost;      // claude_boost: 1 = extra-sample passes give young pixels more paths
uniform float claudeGuide;      // claude_guide: 1 = bounce directions guided by per-block tallies (roadmap 3d-i)
#ifdef GL_ARB_shader_image_load_store
layout(r32ui) uniform uimage2D claudeGuideW;           // this epoch's tallies (written)
layout(r32ui) readonly uniform uimage2D claudeGuideR;  // last epoch's (read: sampling and pdf)
layout(r32ui) readonly uniform uimage2D claudeGuideA;  // its alias table (claude_guide_finalize, once per epoch)
#define CLAUDE_GUIDE_OK 1
#endif
uniform float claudeAreaPick;   // claude_area_pick: 0 = uniform over the list, 1 = by each light's physical bound (light lists stage 1), 2 = bound x learned visibility (stage 2)
uniform float claudeAreaSkip;   // claude_area_skip: 1 = a block that rarely sees its lights samples them less often (stage 2, needs pick 2)
#ifdef GL_ARB_shader_image_load_store
layout(r32ui) uniform uimage2D claudeVisW;             // this epoch's shadow-ray tallies (tries, hits) per block and list slot
layout(r32ui) readonly uniform uimage2D claudeVisR;    // last epoch's
layout(r32f) readonly uniform image2D claudeVisP;      // per block: 16 visibility estimates + the call probability, closed once per epoch
#endif
uniform float claudeAreaNee;    // claude_area_nee: 0 = no area-light samples (A/B; nLights is the one gate, so MIS stays consistent)
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

// The IOR claude_view 20's ladder is drawn at. It is glass's, and it is a
// CONSTANT here on purpose: the ladder is an instrument for the interface
// ARITHMETIC, so it must not depend on what game.cpp happened to upload
// into the palette this run. The palette's own value is what the picture
// uses, and claude_fresnel_check.py reads both.
const float IOR_LADDER = 1.52;

// THE MATERIAL INDEX (2026-08-18). game.cpp writes ONE byte per cell into
// the grid's alpha and that byte is an INDEX into claudeMatPal. This file
// decodes nothing else from it: no bands, no midpoints, no arithmetic.
//
// What it replaced, so the old numbers are still readable in a git log:
//   0 air | 100 water | 130 leaves | 145 glass | 165 point-light nub
//   170..240 emissive as 170 + light_source*5 | 250 authored model
//   255 solid
// Those ranges were mutually exclusive, so a cell could be emissive OR
// fine-shaped and never both — which is why the cabin's campfire, floor
// lantern and torches had their exact shape and emitted nothing.
//
// Index 0 is air and nothing else. The old air test was `a <= 0.25`
// (byte <= 63), which worked because no class lived between 0 and 100;
// material indices start at 1, so the test is now "is the byte zero".
const float MAT_AIR_MAX = 0.5 / 255.0;

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

// THE EMISSION LAW (ADR-0009 #1, claude_accum emitStrength()) now lives
// in the PALETTE, one float per material:
//     Le = albedo * claudeMatPal[index].r
// game.cpp's claudeMatEmission() computes that column and still holds
// EMIT_BASE = 0.4 and EMIT_GAIN = 2.0, so util/claude_furnace_check.py's
// analytic L = Le/(1-rho) is unchanged and the referee stays valid. What
// has GONE is the pair of constants that existed only to undo the
// `170 + light_source*5` packing (0.65 and 0.29) and the two band edges
// that decided whether a cell was ALLOWED to emit — a decision that now
// belongs to a column of a table rather than to where a number happened
// to land between two thresholds.

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
// the flames' light sample draws from its own reserved range, for the
// same reason the sky's does: the path's own sequence is untouched
const uint PT_CTR_BASE = 1u << 25u;
uint g_ptCtr;
float rndPt()
{
	g_ptCtr += 1u;
	return float(pcgHash(g_rngKey ^ pcgHash(PT_CTR_BASE + g_ptCtr)))
			* (1.0 / 4294967296.0);
}

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

// THE BYTE -> INDEX CONVERSION, and it is the only arithmetic this file
// performs on the class byte. A NEAREST fetch of an RGBA8 texel is
// n/255.0 by the GL spec and float32 carries that exactly for every n, so
// floor(v*255 + 0.5) returns n. The claim is not assumed: game.cpp
// round-trips ALL 256 values through a texture of the same format and
// parameters at startup and publishes the score as `matpal_roundtrip` in
// claude_stats.json, and claude_view 19 draws the same test through this
// very decode. The band scheme this replaced was deliberately built to
// tolerate a drift of +/-2 (its edges sit at half-byte midpoints); an
// index tolerates none, which is why the proof is mechanical.
float matIndex(float a)
{
	return floor(a * 255.0 + 0.5);
}

// THE PALETTE LOOKUP — what the index points at.
//   .r = emission scale (multiplies albedo; 0 = not an emitter)
//   .g = 1 if this material has a 16^3 sub-voxel shape (the old class 250)
//   .b = transmission: > 0.5 means a ray is SUPPOSED to cross this
//   .a = index of refraction, read only where .b says to
// Sampled on ARRIVAL AT A NON-AIR CELL, never per step of the walk: the
// dead-weight probe priced a per-step dependent read at +2.2 ms (+14 %)
// and this is deliberately not one.
//
// TWO ENTRY POINTS FOR ONE FETCH. matPalIdx() takes the index the walk
// already decoded; matPal() takes the raw byte and decodes it first. The
// walk has the index in hand at every point that matters, and paying for
// matIndex() twice on the same texel is the kind of small dishonesty that
// makes a cost table wrong.
vec4 matPalIdx(float idx)
{
	return texture2D(claudeMatPal, vec2((idx + 0.5) / 256.0, 0.25));
}
vec4 matPal(float a)
{
	return matPalIdx(matIndex(a));
}

// TRANSMISSION, READ AS A SWITCH. game.cpp stores a fraction so a mixture
// BSDF can arrive later without moving the byte; today every material is
// 0.0 or 1.0 and this file does not implement a partial interface,
// because a fractional one is two lobes with a stochastic choice between
// them and §6 prices an estimator change.
//
// AIR IS FULLY TRANSMISSIVE AND IT IS NOT IN THE TABLE. Index 0's palette
// row is all zeros — it has to be, because "emits nothing" is what makes
// the air test cheap — so the two helpers below answer for index 0
// explicitly rather than by looking it up. Air's IOR is 1.0 and its
// transmission is 1.0, and a ray travelling through it is travelling
// through a medium exactly like any other.
bool matTransmits(float idx, vec4 pal)
{
	return idx < 0.5 || pal.b > 0.5;
}
float matIor(float idx, vec4 pal)
{
	return idx < 0.5 ? 1.0 : pal.a;
}

// THE FRESNEL TERM FOR A DIELECTRIC INTERFACE — exact, not Schlick.
//
// eta = n_incident / n_transmitted, cosI = |cos| of the incidence angle
// measured against the facing normal. Returns the unpolarised
// reflectance, the average of the s and p amplitudes squared. Total
// internal reflection returns exactly 1.0, which is the same branch the
// caller needs anyway (refract() returns the zero vector there).
//
// SCHLICK WAS NOT USED AND THAT IS A DECISION, not an accident. Its error
// against this expression peaks around 1-2 % near grazing at n = 1.5 —
// small, invisible in a picture, and about the size of the Cornell
// referee's whole tolerance. claude_view 20 scores this function against
// the closed form computed off-GPU, so an approximation here would be a
// difference the instrument reports rather than a difference nobody can
// see; there is no reason to spend the accuracy.
//
// R + T = 1 IS NOT ASSERTED ANYWHERE AND MUST NOT BE, because in this
// implementation it is a TAUTOLOGY: the caller reflects with probability
// R carrying weight R/R and refracts with probability 1-R carrying weight
// (1-R)/(1-R), so no code path ever computes T. An instrument that
// "checked" it would be the fifth blind one. The real energy claims are
// (a) this function agreeing with the closed form, which view 20 scores,
// and (b) a lossless slab not moving a sealed furnace's analytic
// radiance, which is transport rather than arithmetic.
float fresnelDielectric(float cosI, float eta)
{
	float s2 = eta * eta * (1.0 - cosI * cosI);
	if (s2 >= 1.0)
		return 1.0;               // total internal reflection
	float cosT = sqrt(1.0 - s2);
	float rs = (eta * cosI - cosT) / (eta * cosI + cosT);
	float rp = (cosI - eta * cosT) / (cosI + eta * cosT);
	return 0.5 * (rs * rs + rp * rp);
}

// Does this material carry a finer shape? The old "is the class byte in
// the band around 250", asked of the table instead. Still ring-gated by
// every caller: outside [48,80)^3 no mask is baked, so a fine material is
// an opaque 1 m cube there (§7 — geometry may be laddered with distance).
bool matFine(vec4 pal)
{
	return pal.g > 0.5;
}

// THE EMISSION LAW. An emissive voxel is a surface with BOTH Le and rho
// (§4) — the caller adds this and then continues the path with rho,
// which is what makes L = Le/(1-rho) expressible in a sealed room. ONE
// law, one table: game.cpp's area-emitter list asks the same column
// through claudeMatEmits(), so the set of cells NEE aims at and the set
// the transport finds emissive cannot drift apart.
vec3 cellEmission(float cls, vec3 albedo)
{
	return albedo * matPal(cls).r;
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

// Is this cell inside the sub-voxel ring? OUTSIDE IT A FINE-MATERIAL CELL
// HAS NO BITS — the bake only fills [48,80)^3, and game.cpp's
// authored-model branch gives a cell a fine material anywhere in the grid. A
// descent that skipped this test would find an all-zero mask and turn
// every distant chest, bed and campfire INVISIBLE.
//
// This is a §2 ladder — a coarser rung at distance — and it is written
// down as one: beyond 16 cells from the grid centre a sub-metre shape
// renders as the 1 m cell it occupies. It is the same ladder game.cpp
// already applies to point-light models (game.cpp: "outside the subvox
// ring a fine material has no bits to express").
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
// entering a fine material's cell inside the ring the walk RESCALES ITSELF by
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
//
// THE MEDIUM THE RAY IS TRAVELLING IN — curMed, added 2026-08-18 with
// transparency, and it is the whole of the change to this walk.
//
// It is the MATERIAL INDEX of the stuff around the ray: 0 for air (and
// for every caller that has no opinion), the glass's own index for a ray
// inside a pane, the water's for a ray inside a tank. The arrival test
// used to be "is this cell air"; it is now "is this cell a DIFFERENT
// material from the one I am in", which is the same test when curMed is
// 0 and is what lets a ray inside glass keep going until it reaches the
// far face. Nothing else in the walk knows about transparency: there is
// no second traversal, no per-material branch and no transmissive walk
// beside the opaque one.
//
// palOut / idxOut are the material the walk ARRIVED at. The caller needs
// both to decide what kind of interface it is standing on, and handing
// them back costs nothing — the walk fetched the palette on arrival
// anyway, and returning a value it already holds is cheaper than making
// the caller fetch it again.
//
// AND THE INTERFACE IS NOT ALWAYS THE 1 M WALL (PANES, 2026-08-23).
// Inside a fine transmissive cell — a glass pane is a 2/16 m slab in a
// 1 m cell — the boundary a ray refracts at is where the 16^3 MASK goes
// 0 -> 1 or 1 -> 0 along the ray, and the sub-voxels the mask does not
// claim are AIR, the surrounding medium. So the walk answers about a
// sub-voxel-sized region, and a caller that has to restart on the FAR
// side of the interface cannot rebuild that region from `cellOut` any
// more: `restartPoint(phit, -n, cellOut, 1.0)` would drop the ray up to
// a metre past a 1/16 m pane. hpFar is that restart point, computed at
// whichever rung the interface was actually found on, by the only code
// that knows which rung that was. `hp` is unchanged: the near side, in
// the region the ray came THROUGH.
//
// At the coarse rung hpFar is `restartPoint(phit, -n, ci, 1.0)`, which is
// the expression the path loop used to evaluate itself — same inputs,
// same order, so an opaque scene and a full-cube glass block are
// bit-identical to what they were.
// Emission scale for sub-voxel `sv` (0..15 each) of grid cell `cell`: K on
// an emitting voxel of a modelled cell, 0 on its other voxels, 1 for any
// cell without a model.
float modelVoxelEmitScale(vec3 cell, vec3 sv)
{
	float mid = floor(texture3D(claudeModelIds,
			(cell + 0.5) / GRID_S).r * 255.0 + 0.5);
	if (mid < 3.5)
		return 1.0;
	float m = floor(mid / 4.0) - 1.0;
	float rot = mod(mid, 4.0);
	float layer = (m * 4.0 + rot) * 16.0 + sv.z;
	float pidx = floor(texture3D(claudeModelAtlas,
			(vec3(sv.x, sv.y, layer) + 0.5) / vec3(16.0, 16.0, 2048.0)).r
			* 255.0 + 0.5);
	vec4 pe = texture2D(claudeModelPal,
			(vec2(pidx, m) + 0.5) / vec2(256.0, 64.0));
	if (pe.a < 0.5 / 255.0)
		return 0.0;
	vec4 p0 = texture2D(claudeModelPal, (vec2(0.0, m) + 0.5) / vec2(256.0, 64.0));
	return (floor(p0.r * 255.0 + 0.5) * 256.0 + floor(p0.g * 255.0 + 0.5))
			/ 256.0;
}

// LAVA BY TEMPERATURE, TEXEL BY TEXEL (2026-10-05). The texture's
// brightness is read as a temperature map: a texel's linear luminance
// relative to the texture's average, r, spans 0.252..4.227 on Mineclonia's
// default_lava_source_animated.png (all frames, measured), and that range
// is mapped in log(r) onto 850..1150 C: the crust-to-core span of
// Hawaiian lava (Pinkerton et al. 2002; core 1140-1150 C by thermocouple;
// spec/photometric-sources.md §5). That MAPPING is judgement (the texture
// says which texel is hotter, not by how much): TUNED, learn by John's eye
// / a photo of real pahoehoe. The rest is physics: luminance and colour
// are Planck at that temperature through the CIE 1931 observer, x 0.95
// emissivity, tabulated every 50 C (colour-science 0.4.7; log-luminance
// and chroma interpolated linearly). 850 C glows ~90x dimmer and deep
// red; 1150 C is 3,069 cd/m2, orange-yellow.
const float LAVA_LOGL[7] = float[7](-3.38051, -2.47741, -1.64501, -0.87539, -0.16159, 0.50227, 1.12126);
const float LAVA_CR[7] = float[7](4.1901, 4.0410, 3.9010, 3.7695, 3.6458, 3.5295, 3.4200);
const float LAVA_CG[7] = float[7](0.1613, 0.2055, 0.2470, 0.2858, 0.3223, 0.3565, 0.3886);
vec3 lavaLe(float r)
{
	float f = clamp((log(max(r, 1e-4)) - log(0.252))
			/ (log(4.227) - log(0.252)), 0.0, 1.0) * 6.0;
	int i = min(int(f), 5);
	float w = f - float(i);
	float ll = mix(LAVA_LOGL[i], LAVA_LOGL[i + 1], w);
	vec3 ch = vec3(mix(LAVA_CR[i], LAVA_CR[i + 1], w),
			mix(LAVA_CG[i], LAVA_CG[i + 1], w), 0.0);
	return ch * exp(ll);
}

// HOT: 0 = the emission law, 1 = lava, 2 = flame (palette row 1 .a)
float matHot(float idx)
{
	return texture2D(claudeMatPal, vec2((idx + 0.5) / 256.0, 0.75)).a;
}

// Is this cell a model whose emitting voxels are FLAME? (slot 0 alpha)
bool modelFlame(vec3 cell)
{
	float mid = floor(texture3D(claudeModelIds,
			(cell + 0.5) / GRID_S).r * 255.0 + 0.5);
	if (mid < 3.5)
		return false;
	float m = floor(mid / 4.0) - 1.0;
	return texture2D(claudeModelPal,
			(vec2(0.0, m) + 0.5) / vec2(256.0, 64.0)).a > 0.5;
}

// MODELS PAST THE RING (claude_model_far, 2026-10-06). The ring's bits
// cost memory per CELL, but an authored model's shape is shared: its 16^3
// voxels sit in claudeModelAtlas once per model and rotation. So a model
// cell anywhere in the grid can be walked at 1/16 m from the atlas, the
// same voxels the ring bake copies (claude_models JSON -> both). Node-box
// shapes (stairs, slabs, panes) are per-cell and stay ring-only.
bool modelCell(vec3 cell)
{
	return floor(texture3D(claudeModelIds, (cell + 0.5) / GRID_S).r
			* 255.0 + 0.5) > 3.5;
}
bool modelVoxelSolid(vec3 cell, vec3 sv)
{
	float mid = floor(texture3D(claudeModelIds,
			(cell + 0.5) / GRID_S).r * 255.0 + 0.5);
	float m = floor(mid / 4.0) - 1.0;
	float rot = mod(mid, 4.0);
	float layer = (m * 4.0 + rot) * 16.0 + sv.z;
	return texture3D(claudeModelAtlas,
			(vec3(sv.x, sv.y, layer) + 0.5) / vec3(16.0, 16.0, 2048.0)).r
			> 0.5 / 255.0;
}

// The hot law, if it applies: rewrites le (and lava's albedo). `fine`
// = the hit is a sub-voxel of `cell` at `sv`.
void hotLaw(float idx, vec3 cell, bool fine, vec3 sv, inout vec3 alb,
		inout vec3 le)
{
	if (claudeUnits < 0.5 || idx < 0.5)
		return;
	float hot = matHot(idx);
	if (hot > 1.5 && hot < 2.5) {
		le = FLAME_RGB * FLAME_L;
	} else if (hot > 0.5) {
		alb = vec3(LAVA_RHO);
		le = LAVA_RGB * LAVA_L;
	} else if (fine && modelFlame(cell)) {
		le = modelVoxelEmitScale(cell, sv) > 0.0
				? FLAME_RGB * FLAME_L : vec3(0.0);
	}
}

// The VOXEL's own palette colour, for models flagged "colour": "palette"
// (flowers, 2026-10-05). False (and `rgb` untouched) for every other cell.
bool modelVoxelColour(vec3 cell, vec3 sv, out vec3 rgb)
{
	rgb = vec3(0.0);
	float mid = floor(texture3D(claudeModelIds,
			(cell + 0.5) / GRID_S).r * 255.0 + 0.5);
	if (mid < 3.5)
		return false;
	float m = floor(mid / 4.0) - 1.0;
	vec4 p0 = texture2D(claudeModelPal, (vec2(0.0, m) + 0.5) / vec2(256.0, 64.0));
	if (p0.b < 0.5)
		return false;
	float rot = mod(mid, 4.0);
	float layer = (m * 4.0 + rot) * 16.0 + sv.z;
	float pidx = floor(texture3D(claudeModelAtlas,
			(vec3(sv.x, sv.y, layer) + 0.5) / vec3(16.0, 16.0, 2048.0)).r
			* 255.0 + 0.5);
	if (pidx < 0.5)
		return false;
	rgb = texture2D(claudeModelPal, (vec2(pidx, m) + 0.5) / vec2(256.0, 64.0)).rgb;
	return true;
}

// THE MEDIUM'S ABSORPTION over `dist` metres: the palette's row 1 (per
// metre, R/G/B; 0 for every material that is not a medium). Air is not
// in the table and has its own helpers below. Multiplies, never adds:
// a medium can only remove light.
vec3 medTr(float idx, float dist)
{
	if (idx < 0.5 || claudeWaterAbsorb < 0.5)
		return vec3(1.0);
	vec3 a = texture2D(claudeMatPal, vec2((idx + 0.5) / 256.0, 0.75)).rgb;
	return exp(-a * dist);
}

// ---- AIR helpers -----------------------------------------------------------
float airSigT() { return claudeAirScatter + claudeAirAbsorb; }

// distance from p along unit d to the edge of the trace grid, where the
// medium ends
float airExitT(vec3 p, vec3 d)
{
	vec3 a = abs(d);
	vec3 room = mix(p, vec3(GRID_S) - p, step(0.0, d));
	vec3 tt = room / max(a, vec3(1e-6));
	return max(min(tt.x, min(tt.y, tt.z)), 0.0);
}

// transmittance of `dist` metres of air (1 when there is no medium)
float airTr(float dist) { return exp(-airSigT() * dist); }

// Henyey-Greenstein phase function, per steradian. c = cosine between the
// path's travel direction and the new one (same sign convention for the
// light sample: c = dot(dir, wi), wi toward the light).
float hgPhase(float c, float g)
{
	float g2 = g * g;
	return (1.0 - g2) / (4.0 * 3.14159265
			* pow(max(1.0 + g2 - 2.0 * g * c, 1e-6), 1.5));
}

vec3 hgSample(vec3 w, float g, float u1, float u2)
{
	float c;
	if (abs(g) < 1e-3) {
		c = 1.0 - 2.0 * u1;
	} else {
		float q = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
		c = (1.0 + g * g - q * q) / (2.0 * g);
	}
	c = clamp(c, -1.0, 1.0);
	float sn = sqrt(max(0.0, 1.0 - c * c));
	float ph = 6.28318531 * u2;
	vec3 ta = abs(w.y) > 0.5 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	vec3 tx = normalize(cross(ta, w));
	vec3 ty = cross(w, tx);
	return normalize(tx * (sn * cos(ph)) + ty * (sn * sin(ph)) + w * c);
}

// The face-tile ratio under a plain cube's hit point (claudeTexel).
vec3 faceTileRatio(vec3 cell, vec3 phit, vec3 n)
{
	float mid = floor(texture3D(claudeMaterials,
			(cell + 0.5) / GRID_S).r * 255.0 + 0.5);
	if (mid < 0.5)
		return vec3(1.0);
	vec3 l = clamp(phit - cell, vec3(0.0), vec3(0.99999));
	float face;
	vec2 uv;
	if (abs(n.y) > 0.5) {
		face = n.y > 0.0 ? 0.0 : 1.0;
		uv = vec2(l.x, l.z);
	} else if (abs(n.x) > 0.5) {
		face = 2.0;
		uv = vec2(n.x > 0.0 ? 1.0 - l.z : l.z, 1.0 - l.y);
	} else {
		face = 2.0;
		uv = vec2(n.z > 0.0 ? l.x : 1.0 - l.x, 1.0 - l.y);
	}
	vec2 tx = floor(uv * 16.0);
	vec2 at = vec2(mod(mid, 16.0) * 16.0 + tx.x,
			face * 256.0 + floor(mid / 16.0) * 16.0 + tx.y);
	return texture2D(claudeAtlas, (at + 0.5) / vec2(256.0, 768.0)).rgb
			* (255.0 / 64.0);
}

// Does this cell have 1/16 m bits to walk, and is this sub-voxel solid?
// Ring cells: the ring's bits (as always). Elsewhere: a model cell's
// atlas voxels, when claude_model_far is on.
uniform float claudeModelFar;
// THE PIECE POOL (ladder B2, DECISIONS 0x, 2026-10-07; claude_bricks 1):
// a piece id per 1 m block (unit 22, RG8: 0 none, 1 solid, 2 empty, then
// shapes) and every distinct 16^3 shape once (unit 23, the ring's byte
// layout, 64 shapes per 16-deep slab). Same decisions as the ring and the
// atlas (game.cpp claudeTraceGridBakeBricks), so the walk must not change
// by one pixel: claude_view 39 checks exactly that.
uniform sampler3D claudeBrickIds;
uniform sampler3D claudeBrickPool;
uniform float claudeBricks;
uniform float claudeBricksFar;   // B3: node-box shapes past the ring too
bool g_bricksOff = false;   // view 39 walks once with, once without
float brickId(vec3 cell)
{
	vec2 v = texture3D(claudeBrickIds, (cell + 0.5) / GRID_S).rg;
	return floor(v.r * 255.0 + 0.5) + 256.0 * floor(v.g * 255.0 + 0.5);
}
bool brickSolid(float id, vec3 sv)
{
	if (claudeTreePlant > 0.5)   // the gate's planted defect: shapes mirrored
		sv.x = 15.0 - sv.x;
	vec3 texel = vec3(mod(id, 64.0) * 2.0 + floor(sv.x / 8.0), sv.y,
			floor(id / 64.0) * 16.0 + sv.z);
	float raw = texture3D(claudeBrickPool,
			(texel + 0.5) / vec3(128.0, 16.0, 1104.0)).r;
	float byte = floor(raw * 255.0 + 0.5);
	return mod(floor(byte / exp2(mod(sv.x, 8.0))), 2.0) >= 0.5;
}
bool fineAt(vec3 cell)
{
	if (claudeBricks > 0.5 && !g_bricksOff)
		return brickId(cell) > 0.5 && (inSubvoxRing(cell) || claudeModelFar > 0.5
				|| claudeBricksFar > 0.5);
	return inSubvoxRing(cell) || (claudeModelFar > 0.5 && modelCell(cell));
}
// IS THIS BLOCK WALKED AT 1/16 m? Today: a fine material (the snapshot
// marks ring node boxes and models fine) with bits to walk. B3 adds: any
// block whose piece is a real shape (id > 2: not solid, not empty) -- a
// stair past the ring keeps its plain material and gains its shape.
bool fineHereAt(vec4 pal, vec3 cell)
{
	if (claudeBricks > 0.5 && claudeBricksFar > 0.5 && !g_bricksOff
			&& brickId(cell) > 2.5)
		return true;
	return matFine(pal) && fineAt(cell);
}
bool fineSolid(vec3 cell, vec3 sv)
{
	if (claudeBricks > 0.5 && !g_bricksOff)
		return brickSolid(brickId(cell), sv);
	return inSubvoxRing(cell) ? subvoxSolid(cell - vec3(SUBV_R0), sv)
			: modelVoxelSolid(cell, sv);
}

// INSTRUMENT (claude_view 21, 2026-10-06): every walk step and every ray
// this pixel spends in one frame, all bounces and shadow rays included.
// John: "the math for the axis-aligned blocks is supposed to be so simple
// that caching isn't that much different than a lookup table" -- each step
// IS about a lookup; this counts how many lookups a frame actually takes.
float g_steps = 0.0;
float g_rays = 0.0;
// 3d-0 (roadmap: measure before building any learned sampling): per pixel
// per frame, bounce rays after which the path gathered no more light (the
// ceiling of what guiding could recover), and light samples that returned
// nothing (the ceiling of a better light choice). claude_view 27 / 28.
float g_bounces = 0.0, g_bounceWasted = 0.0, g_neeCalls = 0.0, g_neeZero = 0.0;
vec3 g_neeCallsT = vec3(0.0), g_neeZeroT = vec3(0.0); // area, sky, flame

// THE ONE FORMULA of the ladder walk (2026-10-06): the time at which the ray
// crosses an integer plane, per axis. Every cell at every size is found by
// comparing times from this function -- never by rounding a position, which
// is the class of bug the first empty-space leap had (it rounded a point to
// a cell, landed behind itself, and looped; util/claude_hdda_equiv.py proves
// the ladder against the plain walk in float32, edge-on rays included). An
// axis the ray does not move along gets a huge time ADDED: mix(1e30, x, 1)
// rounds x away on this GPU.
vec3 tcross(vec3 plane, vec3 ro, vec3 stepDir, vec3 delta0)
{
	return (plane - ro) * stepDir * delta0 + (vec3(1.0) - abs(stepDir)) * 1e30;
}


// =====================================================================
// THE TREE WALK (ladder-plan stage 2; util/claude_tree_equiv.py is its
// float32 proof, exact at any branching factor). Levels in BASE units
// (1/16 m): 1, 4, 16 (1 m), 64, 256, 1024, 2048 (the 128 m grid, factor 2
// at the top). A level-l node covers TREE_F[l]^3 children of level l-1;
// level 0 has no nodes, its occupancy is the level-1 node's mask bits.
// The walk keeps the node it last used at every level (tnode / tcoord), so
// a step inside the same parent never searches from the root.
// =====================================================================
#ifdef CLAUDE_TREE_OK
const int TREE_TOP = 6;
const uint TREE_FULL = 0xFFFFFFFFu;
int treeF(int l) { return l == 6 ? 2 : 4; }
int treeS(int l) { return l == 0 ? 1 : l == 1 ? 4 : l == 2 ? 16 : l == 3 ? 64 : l == 4 ? 256 : l == 5 ? 1024 : 2048; }
int tnode[7];      // node index at each level: >= 0 a node, -2 full, -3 empty, -1 unknown
ivec3 tcoord[7];   // its coordinates, in that level's cells
int g_treeIters = 0;   // walk iterations, for the price views
int g_treeLoads = 0;   // tree nodes read from memory, likewise

// the node AT level L whose coordinates are c (level-L units)
int treeNodeAt(int L, ivec3 c)
{
	int l = L;
	for (; l < TREE_TOP; l++) {
		ivec3 al = (c * treeS(L)) / treeS(l);
		if (tnode[l] != -1 && all(equal(tcoord[l], al)))
			break;
	}
	int node = tnode[l];
	for (int k = l; k > L; k--) {
		ivec3 childc = (c * treeS(L)) / treeS(k - 1);
		int nn;
		if (node == -3)
			nn = -3;
		else if (node == -2)
			nn = -2;
		else {
			int f = treeF(k);
			ivec3 local = childc - (childc / f) * f;
			int bit = (local.z * f + local.y) * f + local.x;
			uint lo = claudeTree[3 * node], hi = claudeTree[3 * node + 1];
			uint ch = claudeTree[3 * node + 2];
			bool set = bit < 32 ? ((lo >> uint(bit)) & 1u) != 0u
					: ((hi >> uint(bit - 32)) & 1u) != 0u;
			if (!set)
				nn = -3;
			else if (ch == TREE_FULL)
				nn = -2;
			else {
				int cnt = bit < 32 ? bitCount(lo & ((1u << uint(bit)) - 1u))
						: bitCount(lo) + bitCount(hi & ((1u << uint(bit - 32)) - 1u));
				nn = int(ch) + cnt;
			}
		}
		tnode[k - 1] = nn;
		tcoord[k - 1] = childc;
		node = nn;
	}
	return node;
}

// ABLATION (claude_tree_variant 1, 2026-10-07): the same lookup with NO
// path arrays, descending from the root every time. If this is faster
// than the cached walk, the arrays (indexed at run time) are the cost.
uniform float claudeTreeVariant;
uniform float claudeTreeDirs;
int treeNodeAtRoot(int L, ivec3 c)
{
	int node = 0;
	for (int k = TREE_TOP; k > L; k--) {
		if (node < 0)
			return node;
		ivec3 childc = (c * treeS(L)) / treeS(k - 1);
		int f = treeF(k);
		ivec3 local = childc - (childc / f) * f;
		int bit = (local.z * f + local.y) * f + local.x;
		uint lo = claudeTree[3 * node], hi = claudeTree[3 * node + 1];
		uint ch = claudeTree[3 * node + 2];
		bool set = bit < 32 ? ((lo >> uint(bit)) & 1u) != 0u
				: ((hi >> uint(bit - 32)) & 1u) != 0u;
		if (!set)
			return -3;
		if (ch == TREE_FULL)
			return -2;
		int cnt = bit < 32 ? bitCount(lo & ((1u << uint(bit)) - 1u))
				: bitCount(lo) + bitCount(hi & ((1u << uint(bit - 32)) - 1u));
		node = int(ch) + cnt;
	}
	return node;
}

// is the level-L cell c occupied (anything solid under it)?
bool treeOcc(int L, ivec3 c)
{
	if (L >= TREE_TOP)
		return (claudeTree[0] | claudeTree[1]) != 0u;
	int f = treeF(L + 1);
	int pn = claudeTreeVariant > 0.5 ? treeNodeAtRoot(L + 1, c / f) : treeNodeAt(L + 1, c / f);
	if (pn == -3)
		return false;
	if (pn == -2)
		return true;
	ivec3 local = c - (c / f) * f;
	int bit = (local.z * f + local.y) * f + local.x;
	uint w = bit < 32 ? claudeTree[3 * pn] : claudeTree[3 * pn + 1];
	return ((w >> uint(bit < 32 ? bit : bit - 32)) & 1u) != 0u;
}

int treeTie(vec3 side)
{
	if (side.x < side.y && side.x < side.z)
		return 0;
	if (side.y < side.z)
		return 1;
	return 2;
}

// first solid base piece along the ray: ro in BASE units; t in base units
bool treeWalk(vec3 ro, vec3 rd, out ivec3 hitB, out int hitAxis, out float hitT)
{
	for (int l = 0; l < TREE_TOP; l++)
		tnode[l] = -1;
	tnode[TREE_TOP] = 0;
	tcoord[TREE_TOP] = ivec3(0);
	vec3 stepDir = sign(rd);
	vec3 delta0 = 1.0 / max(abs(rd), vec3(1e-8));
	int L = 0;
	ivec3 c = ivec3(floor(ro));
	float t = 0.0;
	int axis = -1;
	bool started = false;
	vec3 side = tcross(vec3((c + ivec3(greaterThan(stepDir, vec3(0.0)))) * treeS(L)), ro, stepDir, delta0);
	hitB = ivec3(0); hitAxis = -1; hitT = 0.0;
	for (int i = 0; i < 4096; i++) {
		g_treeIters++;
		if (started) {
			for (int g = 0; g < 8 && L > 0 && treeOcc(L, c); g++) {
				int f = treeF(L);
				L -= 1;
				int cs = treeS(L);
				ivec3 k = ivec3(0);
				for (int ax = 0; ax < 3; ax++) {
					int lo = c[ax] * f;
					if (stepDir[ax] == 0.0) {
						k[ax] = clamp(int(floor((ro[ax] - float(lo * cs)) / float(cs))), 0, f - 1);
						continue;
					}
					int n = 0;
					for (int j = 1; j < 4; j++) {
						if (j >= f)
							break;
						int jj = stepDir[ax] > 0.0 ? j : f - j;
						float tm = (float((lo + jj) * cs) - ro[ax]) * stepDir[ax] * delta0[ax];
						if (tm < t || (tm == t && ax > axis))
							n++;
						else
							break;
					}
					k[ax] = stepDir[ax] > 0.0 ? n : f - 1 - n;
				}
				c = c * f + k;
				side = tcross(vec3((c + ivec3(greaterThan(stepDir, vec3(0.0)))) * treeS(L)), ro, stepDir, delta0);
			}
			if (L == 0 && treeOcc(0, c)) {
				hitB = c; hitAxis = axis; hitT = t;
				return true;
			}
			for (int g = 0; g < 8 && L < TREE_TOP && !treeOcc(L + 1, c / treeF(L + 1)); g++) {
				c = c / treeF(L + 1);
				L += 1;
				side = tcross(vec3((c + ivec3(greaterThan(stepDir, vec3(0.0)))) * treeS(L)), ro, stepDir, delta0);
			}
		}
		started = true;
		int a = treeTie(side);
		t = side[a];
		axis = a;
		c[a] += int(stepDir[a]);
		if (c[a] < 0 || c[a] >= 2048 / treeS(L))
			return false;
		side = tcross(vec3((c + ivec3(greaterThan(stepDir, vec3(0.0)))) * treeS(L)), ro, stepDir, delta0);
	}
	return false;
}

// THE FAST TREE WALK (claude_tree_variant 2, 2026-10-07). The same cells in
// the same order as treeWalk() (so the same exactness), organised the way
// the published 64-tree walks are (dubiousconst282, 2024): the parent
// node's 64-bit mask lives in registers and an ordinary step is ONE bit
// test; the tree is read again only when the ray leaves the parent, from
// the lowest ancestor that still contains it. Measured why: the first
// port re-asked "is this occupied?" up to three times per step, each a
// descent with divisions -- 7-14x the cost of today's walk per frame.
int treeLg(int l) { return l == 6 ? 11 : 2 * l; }
void treeLoad(int n, out uint lo, out uint hi, out uint ch)
{
	if (n == -2) {
		lo = TREE_FULL; hi = TREE_FULL; ch = TREE_FULL;
	} else {
		g_treeLoads++;
		lo = claudeTree[3 * n]; hi = claudeTree[3 * n + 1]; ch = claudeTree[3 * n + 2];
	}
}
int treeBit(ivec3 c, int f)
{
	ivec3 lc = c & ivec3(f - 1);
	return f == 4 ? (lc.z << 4) | (lc.y << 2) | lc.x : (lc.z << 2) | (lc.y << 1) | lc.x;
}
bool treeBitSet(uint lo, uint hi, int bit)
{
	return bit < 32 ? ((lo >> uint(bit)) & 1u) != 0u : ((hi >> uint(bit - 32)) & 1u) != 0u;
}
int treeChild(uint lo, uint hi, uint ch, int bit)
{
	if (ch == TREE_FULL)
		return -2;
	int cnt = bit < 32 ? bitCount(lo & ((1u << uint(bit)) - 1u))
			: bitCount(lo) + bitCount(hi & ((1u << uint(bit - 32)) - 1u));
	return int(ch) + cnt;
}

bool treeWalk2(vec3 ro, vec3 rd, out ivec3 hitB, out int hitAxis, out float hitT)
{
	vec3 stepDir = sign(rd);
	vec3 delta0 = 1.0 / max(abs(rd), vec3(1e-8));
	ivec3 up = ivec3(greaterThan(stepDir, vec3(0.0)));
	int anc[7];          // node covering the ray's cell at each level (valid from r up)
	anc[6] = 0;
	int r = 6;           // lowest level whose anc[] is valid for the current cell
	int L = 0;
	ivec3 c = ivec3(floor(ro));
	float t = 0.0;
	int axis = -1;
	uint pLo = 0u, pHi = 0u, pCh = 0u;   // the parent (level L+1) node, when r == L+1
	vec3 side = tcross(vec3(c + up), ro, stepDir, delta0);
	hitB = ivec3(0); hitAxis = -1; hitT = 0.0;
	for (int i = 0; i < 4096; i++) {
		g_treeIters++;
		if (i > 0) {
			// the ray left its parent: re-find the chain from the lowest
			// ancestor that still holds it; stop at the first empty cell
			// (the treeWalk() climb, found from above)
			if (r > L + 1) {
				bool climbed = false;
				for (int k = r; k > L + 1; k--) {
					uint lo, hi, ch;
					treeLoad(anc[k], lo, hi, ch);
					ivec3 cc = c >> (treeLg(k - 1) - treeLg(L));
					int bit = treeBit(cc, k == 6 ? 2 : 4);
					if (!treeBitSet(lo, hi, bit)) {
						L = k - 1;
						c = cc;
						pLo = lo; pHi = hi; pCh = ch;
						r = k;
						side = tcross(vec3((c + up) << treeLg(L)), ro, stepDir, delta0);
						climbed = true;
						break;
					}
					anc[k - 1] = treeChild(lo, hi, ch, bit);
				}
				if (!climbed) {
					treeLoad(anc[L + 1], pLo, pHi, pCh);
					r = L + 1;
				}
			}
			// descend while the cell is occupied (inner planes crossed,
			// counted with tcross and the tie rule, exactly as treeWalk)
			for (int g = 0; g < 8 && L > 0; g++) {
				int f = L == 6 ? 2 : 4;
				int pf = L + 1 == 6 ? 2 : 4;
				int bit = treeBit(c, pf);
				if (!treeBitSet(pLo, pHi, bit))
					break;
				int child = treeChild(pLo, pHi, pCh, bit);
				anc[L] = child;
				L -= 1;
				float cs = float(1 << treeLg(L));
				ivec3 k = ivec3(0);
				for (int ax = 0; ax < 3; ax++) {
					float lo = float(c[ax] * f) * cs;
					if (stepDir[ax] == 0.0) {
						k[ax] = clamp(int(floor((ro[ax] - lo) / cs)), 0, f - 1);
						continue;
					}
					int n = 0;
					for (int j = 1; j < 4; j++) {
						if (j >= f)
							break;
						int jj = stepDir[ax] > 0.0 ? j : f - j;
						float tm = (lo + float(jj) * cs - ro[ax]) * stepDir[ax] * delta0[ax];
						if (tm < t || (tm == t && ax > axis))
							n++;
						else
							break;
					}
					k[ax] = stepDir[ax] > 0.0 ? n : f - 1 - n;
				}
				c = c * f + k;
				treeLoad(child, pLo, pHi, pCh);
				r = L + 1;
				side = tcross(vec3((c + up) << treeLg(L)), ro, stepDir, delta0);
			}
			if (L == 0 && treeBitSet(pLo, pHi, treeBit(c, 4))) {
				hitB = c; hitAxis = axis; hitT = t;
				return true;
			}
		}
		int a = treeTie(side);
		t = side[a];
		axis = a;
		int old = c[a];
		c[a] += int(stepDir[a]);
		if (c[a] < 0 || c[a] >= (2048 >> treeLg(L)))
			return false;
		// only the stepped axis's next plane moved; tcross is per
		// component, so this is the same float as recomputing all three
		side[a] = (float((c[a] + up[a]) << treeLg(L)) - ro[a]) * stepDir[a] * delta0[a];
		// lowest level whose cell still holds both the old and new cell
		int m = L + 1;
		while (m < 6 && (old >> (treeLg(m) - treeLg(L))) != (c[a] >> (treeLg(m) - treeLg(L))))
			m++;
		r = max(r, m);
	}
	return false;
}
#endif

// THE EXACT FINE WALK (claude_walk_exact 1, 2026-10-07). The ladder
// stage 2 gate found today's walk wrong on 75 forest camera rays that the
// tree walk gets right (float64 referee): it re-derived positions where
// the tree uses the ONE formula. 64 came from the fine-entry pull
// (SUBV - 1/512), the rest from running sums. With this dial every crossing
// time comes from tcross() and the entry piece is found by counting the
// piece planes already crossed at the entry time, with the walk's own tie
// rule (z > y > x) -- the tree walk's descent, at factor 16. Planes are
// cell + k/16, exact in float, so t is the tree's t / 16 exactly.
uniform float claudeWalkExact;
bool fineCrossed(float tm, float t, int ax, int axis)
{
	return tm < t || (tm == t && ax > axis);
}
// One axis of the entry piece. The rounded position is off by at most one
// piece (its error is float-sized against 1/16 m), so one check each way.
float fineEntryAxis(float hiA, float k, float roA, float sA, float dA,
		float t, int ax, int axis)
{
	if (ax == axis)
		return sA > 0.0 ? 0.0 : SUBV - 1.0;
	if (sA == 0.0)
		return k;
	k = clamp(k, 0.0, SUBV - 1.0);
	float tLo = (hiA + k * RUNG_FINE - roA) * sA * dA;
	float tHi = (hiA + (k + 1.0) * RUNG_FINE - roA) * sA * dA;
	if (sA > 0.0) {
		if (k > 0.0 && !fineCrossed(tLo, t, ax, axis))
			k -= 1.0;
		else if (k < SUBV - 1.0 && fineCrossed(tHi, t, ax, axis))
			k += 1.0;
	} else {
		if (k < SUBV - 1.0 && !fineCrossed(tHi, t, ax, axis))
			k += 1.0;
		else if (k > 0.0 && fineCrossed(tLo, t, ax, axis))
			k -= 1.0;
	}
	return k;
}
vec3 fineEntryExact(vec3 hi, vec3 su, vec3 ro, vec3 stepDir, vec3 delta0,
		float t, int axis)
{
	return vec3(fineEntryAxis(hi.x, su.x, ro.x, stepDir.x, delta0.x, t, 0, axis),
			fineEntryAxis(hi.y, su.y, ro.y, stepDir.y, delta0.y, t, 1, axis),
			fineEntryAxis(hi.z, su.z, ro.z, stepDir.z, delta0.z, t, 2, axis));
}

bool marchMed(vec3 ro, vec3 rd, float curMed, out vec3 hp, out vec3 n,
		out vec3 alb, out vec3 le, out float tHit, out vec3 cellOut,
		out vec4 palOut, out float idxOut, out vec3 hpFar)
{
	palOut = vec4(0.0);
	idxOut = 0.0;
	hp = ro;
	hpFar = ro;
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
	// the material behind s.a, fetched once per non-air arrival
	vec4 pal = vec4(0.0);
	float t = 0.0;
	float lim = GRID_S;
	int axis = -1;
	// the ladder above 1 m (claude_pyramid): the walk is on cells of 2^L m,
	// ci in units of that size; L = 0 is the plain 1 m walk
	int L = 0;
	vec3 delta0 = 1.0 / max(abs(rd), vec3(DDA_MIN_ABS));
	vec3 walkUp = step(0.0, stepDir);   // claude_walk_exact: the far plane's offset
	vec3 walkSd = stepDir * delta0;     // ... and its slope (exact: a sign flip)

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
	if (claudeDescend > 0.5 && fineAt(cellHi)) {
		s = texture3D(claudeTraceGrid, (cellHi + 0.5) / GRID_S);
		// THE MATERIAL OF THE CELL THE RAY STARTS IN, reported even
		// though the walk is about to skip the cell itself. It has to
		// be: a hit on the fine rung inside this cell IS a hit on this
		// material, and a caller handed idxOut = 0 would read it as AIR
		// -- which since 2026-08-18 means "an interface a ray passes
		// through", so a solid sub-voxel of a stair would be crossed
		// rather than stopped at. Found by reading the transparency
		// change back against this block rather than by a frame.
		//
		// `pal` is deliberately NOT set from here, and that is a
		// PRE-EXISTING hole left standing rather than fixed in the same
		// commit: a fine hit inside the starting cell reports le = 0, so
		// a bounce ray leaving a torch and landing on the torch's own
		// far side sees no emission. Fixing it changes what the cabin
		// arms photograph, which is coverage item 5's variable, not this
		// step's. Written into spec/roadmap.md instead.
		idxOut = matIndex(s.a);
		palOut = matPalIdx(idxOut);
		if (fineHereAt(palOut, cellHi)) {
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
			if (claudeWalkExact > 0.5)
				sideDist = tcross(cellHi + (ci + step(0.0, stepDir)) * RUNG_FINE,
						ro, stepDir, delta0);
			lim = SUBV;
		}
	}

	for (int i = 0; i < WALK_STEPS; i++) {
		g_steps += 1.0;
		// ONE STEP OF THE WALK, at whatever size the walk is currently
		// set to. These are the same three lines at 1 m and at 1/16 m;
		// the rung is in delta and lim, not in a branch. Advance first,
		// then test what was entered: the cell the ray starts in is
		// never tested, which is what keeps a bounce ray off its own
		// surface.
		// claude_walk_exact: the stepped axis's next plane from the one
		// formula, (plane - ro) * (stepDir * delta0), the same float as
		// tcross(); planes are cell + k/16, exact. Else the running sum.
		bool exactStep = claudeWalkExact > 0.5 && L == 0;
		bool fineRung = lim < GRID_S;
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x;
			ci.x += stepDir.x; axis = 0;
			sideDist.x = exactStep ? ((fineRung ? cellHi.x + (ci.x + walkUp.x) * RUNG_FINE
					: ci.x + walkUp.x) - ro.x) * walkSd.x : sideDist.x + delta.x;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y;
			ci.y += stepDir.y; axis = 1;
			sideDist.y = exactStep ? ((fineRung ? cellHi.y + (ci.y + walkUp.y) * RUNG_FINE
					: ci.y + walkUp.y) - ro.y) * walkSd.y : sideDist.y + delta.y;
		} else {
			t = sideDist.z;
			ci.z += stepDir.z; axis = 2;
			sideDist.z = exactStep ? ((fineRung ? cellHi.z + (ci.z + walkUp.z) * RUNG_FINE
					: ci.z + walkUp.z) - ro.z) * walkSd.z : sideDist.z + delta.z;
		}

		if (L > 0)
			sideDist = tcross((ci + step(0.0, stepDir)) * float(1 << L), ro,
					stepDir, delta0);
		bool escaped = any(lessThan(ci, vec3(0.0)))
				|| any(greaterThanEqual(ci, vec3(L > 0 ? GRID_S / float(1 << L) : lim)));

		if (lim < GRID_S) {
			// ---- the walk is on the 1/16 m rung, inside cellHi ----
			if (!escaped) {
				// THE MATERIAL OF THE SUB-VOXEL JUST ENTERED, and this
				// one line is the whole of PANES in the walk. A fine
				// cell's mask says WHERE its material is; where the mask
				// is 0 the sub-voxel is AIR — the surrounding medium —
				// not "nothing". So the fine rung asks the same question
				// the coarse rung asks, against the same `curMed`:
				// is what I just entered a different material from the
				// one I am travelling in?
				//
				// IT IS THE OLD TEST WHEREVER THE OLD TEST APPLIED. With
				// curMed = 0 (every opaque walk, every shadow ray, every
				// eye ray in air) an empty sub-voxel gives subIdx = 0 ==
				// curMed and steps again, and a solid one gives
				// subIdx = the cell's index != 0 and stops — which is
				// `if (!subvoxSolid(...)) continue;` exactly. No stair,
				// slab, bed or authored model changes by one bit.
				//
				// AND IT IS WHERE THE STALE-curMed X-RAY DIES (measured.md
				// "Defect 2 — glass X-rays the opaque block behind it").
				// A ray that refracted into glass and then descended into
				// an opaque FINE cell — a carved plank — used to walk the
				// air sub-voxels of that plank's relief without noticing
				// that it had left the glass, and every ray leaving that
				// vertex inherited curMed = glass. Here that first air
				// sub-voxel IS an interface: glass -> air, curMed goes
				// back to 0 at the crossing, and no later ray carries a
				// medium it is not in.
				float subIdx = fineSolid(cellHi, ci)
						? idxOut : 0.0;
				if (subIdx == curMed)
					continue; // same medium: step again
				n = vec3(0.0);
				if (axis == 0) n.x = -stepDir.x;
				else if (axis == 1) n.y = -stepDir.y;
				else n.z = -stepDir.z;
				// ci + n is the SUB-VOXEL the walk came through, and it
				// holds the medium the ray is in: the loop only continues
				// past sub-voxels whose material matched curMed, and the
				// origin one is excluded. So the next ray starts inside a
				// known-curMed 1/16 m cell of cellHi — the same guarantee
				// as at 1 m, one rung down.
				hp = restartPoint(ro + rd * t, n,
						cellHi + (ci + n) * RUNG_FINE, RUNG_FINE);
				// the FAR side, in the sub-voxel that was entered
				hpFar = restartPoint(ro + rd * t, -n,
						cellHi + ci * RUNG_FINE, RUNG_FINE);
				if (subIdx < 0.5) {
					// LEFT the cell's material into one of its air
					// sub-voxels. Report AIR, because that is what the
					// ray arrived at — reporting the cell here would
					// hand the caller glass on both sides of a pane and
					// it would never refract out. alb/le are what an
					// ARRIVAL AT AIR reports one rung up, verbatim:
					// cellAlbedo of a zero texel (the floor) and no
					// emission, since air's palette row is all zeros.
					palOut = vec4(0.0);
					idxOut = 0.0;
					alb = cellAlbedo(vec3(0.0));
					le = vec3(0.0);
				} else {
					alb = cellAlbedo(s.rgb);
					vec3 vrgb;
					if (modelVoxelColour(cellHi, ci, vrgb))
						alb = cellAlbedo(vrgb);
					le = alb * pal.r;   // the palette's emission column (§4: one Le)
					if (claudeFlame > 0.5 && pal.r > 0.0)
						le *= modelVoxelEmitScale(cellHi, ci);
					if (pal.r > 0.0)
						hotLaw(idxOut, cellHi, true, ci, alb, le);
				}
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
			if (claudeWalkExact > 0.5)
				sideDist = tcross(ci + step(0.0, stepDir), ro, stepDir, delta0);
			escaped = any(lessThan(ci, vec3(0.0)))
					|| any(greaterThanEqual(ci, vec3(GRID_S)));
		}

		if (escaped)
			return false; // escaped: contributes nothing (sky is punted)

		if (L > 0) {
			if (texelFetch(claudeCoarse, ivec3(ci), L).r == 0.0) {
				// empty: climb while the parent is empty too, then go on
				while (L < 5 && texelFetch(claudeCoarse, ivec3(ci) >> 1, L + 1).r == 0.0) {
					L++;
					ci = floor(ci * 0.5);
				}
				sideDist = tcross((ci + step(0.0, stepDir)) * float(1 << L), ro,
						stepDir, delta0);
				continue;
			}
			// occupied: down to the child the ray is in at its entry time t.
			// Per axis: has the ray crossed the middle plane by t? A middle
			// plane crossed at the SAME instant as the entry face counts only
			// if its axis outranks the entry axis -- the plain walk's own tie
			// rule (z, then y, then x)
			vec3 rank = vec3(greaterThan(vec3(0.0, 1.0, 2.0), vec3(float(axis))));
			while (L > 0 && texelFetch(claudeCoarse, ivec3(ci), L).r > 0.0) {
				L--;
				vec3 mid = (2.0 * ci + 1.0) * float(1 << L);
				vec3 tm = tcross(mid, ro, stepDir, delta0);
				vec3 crossed = vec3(lessThan(tm, vec3(t)))
						+ vec3(equal(tm, vec3(t))) * rank;
				crossed = min(crossed, vec3(1.0));
				vec3 up = stepDir * (2.0 * crossed - 1.0) * 0.5 + 0.5;   // +: crossed, -: not crossed
				up = mix(up, vec3(greaterThanEqual(ro, mid)), vec3(equal(stepDir, vec3(0.0))));
				ci = 2.0 * ci + up;
			}
			sideDist = tcross((ci + step(0.0, stepDir)) * float(1 << L), ro,
					stepDir, delta0);
			if (L > 0)
				continue;   // an empty child: walk on at its size
			// L == 0: ci is the 1 m cell the ray is in at t; the arrival
			// test below takes it from here
		}

		// ARRIVAL AT A 1 M CELL. s stays live across a descent — a fine
		// hit takes its colour and its class from the cell that owns
		// the mask — and it is only ever written here, on the coarse
		// rung, where nothing is depending on the old value.
		s = texture3D(claudeTraceGrid, (ci + 0.5) / GRID_S);
		// THE ARRIVAL TEST, and since 2026-08-18 it is a comparison rather
		// than a threshold: a cell whose material is the one the ray is
		// already travelling through is not a boundary, so the walk goes
		// on. With curMed = 0 that is exactly the old `s.a <= MAT_AIR_MAX`
		// air test — index 0 is air and nothing else — and every caller
		// that never entered a medium takes precisely the path it took
		// before. With curMed = a pane's index, it is what carries a
		// refracted ray across the inside of the pane to its far face.
		float idx = matIndex(s.a);
		// AIR INSIDE AIR — the hot path, and it must stay one compare and
		// NO palette fetch. Empty space is most of every walk and the
		// dead-weight probe priced a per-step dependent read at +2.2 ms;
		// this early-out is why the fetch below is per ARRIVAL and not
		// per step. With curMed = 0, idx == curMed means idx == 0 means
		// air, so this is the old test verbatim for every caller that
		// never entered a medium.
		if (idx == curMed && curMed < 0.5) {
			// EMPTY SPACE: climb the ladder while the parent block is
			// empty, and walk on at that size (see tcross and the block
			// arrival above). Only air-in-air on the 1 m rung.
			if (claudePyramid > 0.5
					&& texelFetch(claudeCoarse, ivec3(ci) >> 1, 1).r == 0.0) {
				while (L < 5 && texelFetch(claudeCoarse, ivec3(ci) >> 1, L + 1).r == 0.0) {
					L++;
					ci = floor(ci * 0.5);
				}
				sideDist = tcross((ci + step(0.0, stepDir)) * float(1 << L), ro,
						stepDir, delta0);
			}
			continue;
		}
		// A boundary, or a cell of the ray's own medium that may still
		// have an INTERIOR. Ask the table which. ONE palette fetch per
		// arrival, kept live across a descent for the same reason `s` is:
		// a fine hit takes its material from the cell that owns the mask.
		pal = matPalIdx(idx);
		bool fineHere = claudeDescend > 0.5 && fineHereAt(pal, ci);
		// GLASS SITS FLUSH AGAINST CARVED WOOD (2026-10-04, John: "glass
		// sits flush"). A ray still IN a medium (curMed != 0, and only a
		// transmissive material is ever a medium) that arrives at an
		// opaque fine cell meets that cell's flat 1 m face: no descent, so
		// it never meets the 1/16 m air pockets of the relief. Measured
		// before this (spec/measured.md 2026-10-04, "glass beside carved
		// wood"): a ray inside glass met a groove's air sub-voxel at a
		// grazing angle, totally internally reflected, and carried the
		// window's outside view into every groove of the reveal — bars on
		// 7.7 % of the frame at a glass-block window, 1.2 % at a pane.
		// Correct optics for a metre of glass pressed against grooved
		// wood; the wrong picture, because glazing is sealed to its frame.
		// curMed = 0 (every opaque scene, every ray in air) can never take
		// this branch, so nothing without glass changes by one bit.
		if (fineHere && claudeGlassFlush > 0.5 && curMed > 0.5
				&& !matTransmits(idx, pal))
			fineHere = false;
		// SAME MATERIAL IS NOT THE SAME AS NO INTERFACE, once a material
		// can be fine (PANES, 2026-08-23). A ray inside a pane crossing
		// into the NEXT pane cell of the same window meets the same
		// material index — but that cell's connected nodebox may have no
		// arm along this ray, so the glass ends inside it. Skipping the
		// cell because the index matched would carry the ray through
		// geometry that is not there. A fine cell is therefore always
		// descended into and the mask decides; only a homogeneous 1 m
		// cell of the ray's own medium is stepped over.
		if (idx == curMed && !fineHere)
			continue;
		palOut = pal;
		idxOut = idx;

		n = vec3(0.0);
		if (axis == 0) n.x = -stepDir.x;
		else if (axis == 1) n.y = -stepDir.y;
		else n.z = -stepDir.z;

		// The region a crossing ray would restart INSIDE, defaulted to
		// the 1 m cell and narrowed to a sub-voxel by the fine gate
		// below. See hpFar on the signature.
		vec3 farLo = ci;
		float farH = 1.0;
		// "the far side of this face is one of a fine cell's AIR
		// sub-voxels" — the one arrival at the coarse rung whose material
		// is not the cell's own. Set only inside the fine gate.
		bool hitAir = false;
		// the entry sub-voxel, when this arrival is a fine cell's solid one
		vec3 suHit = vec3(-1.0);

		// A FINE MATERIAL SAYS "DO NOT STOP AT MY 1 M WALL". Rescale the
		// to 1/16 m and keep going, against this cell's 16^3 mask. The
		// sub-voxel the ray enters through is tested here, by the
		// rescale, because the loop only ever tests what it stepped INTO
		// and no step has been taken inside the cell yet; if it is
		// solid, the surface the ray met is the 1 m face it came
		// through and n already holds that face's normal. A miss means
		// the ray passed THROUGH this cell and the walk resumes at 1 m
		// from the far face. Ring-gated, because outside [48,80)^3 a fine
		// material has no mask and must stay the cube it is today.
		//
		// Shadow and bounce rays get this for free and that is the
		// point: §2 forbids a voxel that exists for eye rays but not
		// for shadow rays, and one traversal is the cheapest way never
		// to commit it. There is no lighter copy of this walk.
		if (fineHere) {
			vec3 hi = ci;
			// entry point in SUB-VOXEL units, clamped INSIDE the cell
			// (see the starting-cell block for why the clamp is on the
			// position rather than on the index)
			vec3 pu = clamp((ro + rd * t - hi) * SUBV, vec3(0.0),
					vec3(SUBV - 1.0 / 512.0));
			vec3 su = floor(pu);
			// (> 1.5: an ablation step while pricing, 2026-10-08; 2 = all)
			if (claudeWalkExact > 0.5)
				su = fineEntryExact(hi, su, ro, stepDir, delta0, t, axis);
			// the entry sub-voxel's material, by the same rule the fine
			// rung uses: the cell's own where the mask is set, AIR where
			// it is not
			float subIdx = fineSolid(hi, su) ? idx : 0.0;
			if (subIdx == curMed) {
				// no interface at the 1 m face — the ray is still in its
				// own medium. Descend; the mask boundary inside the cell
				// is where the interface is. (With curMed = 0 this is the
				// old `if (!subvoxSolid(...))`, unchanged.)
				cellHi = hi;
				delta *= RUNG_FINE;
				sideDist = t + (stepDir * (su - pu)
						+ stepDir * 0.5 + 0.5) * delta;
				if (claudeWalkExact > 0.5)
					sideDist = tcross(hi + (su + step(0.0, stepDir)) * RUNG_FINE,
							ro, stepDir, delta0);
				ci = su;
				lim = SUBV;
				continue;
			}
			// AN INTERFACE AT THE 1 M FACE. The surface the ray met is
			// the face it came through and n already holds that face's
			// normal — but the region on the FAR side of it is the entry
			// SUB-VOXEL, not the metre cell, so a crossing ray restarts
			// 1/16 m in and not up to a metre in.
			farLo = hi + su * RUNG_FINE;
			farH = RUNG_FINE;
			suHit = su;
			if (subIdx < 0.5) {
				// entering one of this cell's air sub-voxels while
				// inside a medium: the medium ends here, at the wall
				palOut = vec4(0.0);
				idxOut = 0.0;
				hitAir = true;
			}
		}

		// every other class is one thing: an opaque Lambertian surface
		// with a cardinal normal, which may also emit
		//
		// ci + n is the cell the walk came through — tested and found to
		// hold the ray's own medium on the step before this one, or the
		// cell the ray started in, which the origin exclusion guarantees
		// was that medium for the same reason one rung up. That is what
		// makes restartPoint()'s clamp target the right cell rather than
		// merely a nearby one.
		hp = restartPoint(ro + rd * t, n, ci + n, 1.0);
		hpFar = restartPoint(ro + rd * t, -n, farLo, farH);
		// hitAir is false everywhere the walk could already reach, and
		// there `cellAlbedo(s.rgb)` on an air texel is `cellAlbedo(0)`
		// anyway — so this reports exactly what an arrival at air has
		// always reported, at either rung.
		alb = hitAir ? cellAlbedo(vec3(0.0)) : cellAlbedo(s.rgb);
		le = hitAir ? vec3(0.0) : alb * pal.r; // emission column (§4: one Le)
		if (claudeFlame > 0.5 && !hitAir && suHit.x >= 0.0 && pal.r > 0.0)
			le *= modelVoxelEmitScale(ci, suHit);
		if (!hitAir && pal.r > 0.0)
			hotLaw(idxOut, ci, suHit.x >= 0.0, suHit, alb, le);
		// LAVA'S TEXTURE IS ITS TEMPERATURE MAP (2026-10-05): see lavaLe().
		// HERE, in the walk, so a shadow ray and an eye ray see the same
		// lava and NEE stays consistent.
		if (claudeUnits > 0.5 && claudeTexel > 0.5 && !hitAir
				&& suHit.x < 0.0 && pal.r > 0.0) {
			float hot = matHot(idxOut);
			if (hot > 0.5 && hot < 1.5)
				le = lavaLe(dot(pow(faceTileRatio(ci, hp, n), vec3(2.2)),
						vec3(0.2126, 0.7152, 0.0722)));
		}

		tHit = t;
		cellOut = ci;
		return true;
	}
	return false;
}

// THE VACUUM WALK — every caller that has no medium to declare.
//
// It exists so that the seven call sites that predate transparency are
// byte-identical to what they were: an instrument view, a shadow ray, the
// path loop's first march all start in air, and `curMed = 0` makes the
// arrival test above the same air test it always was. The two extra
// results are dropped here rather than at each call site.
bool march(vec3 ro, vec3 rd, out vec3 hp, out vec3 n, out vec3 alb,
		out vec3 le, out float tHit, out vec3 cellOut)
{
	vec4 pal;
	float idx;
	vec3 far;
	g_rays += 1.0;
	return marchMed(ro, rd, 0.0, hp, n, alb, le, tHit, cellOut, pal, idx,
			far);
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


// =====================================================================
// GUIDED BOUNCES (claude_guide, roadmap 3d-i, 2026-10-07)
// ---------------------------------------------------------------------
// Each 4 m block keeps, per face direction (6), a table of where light
// came back from: 8x8 bins over the square that the concentric map turns
// into a cosine-weighted hemisphere. Every bin is therefore equally likely
// under plain matte bouncing, and because every surface here is matte
// with a cardinal normal, sampling a bin in proportion to its incoming
// light IS sampling light x reflection (the "product" RCPG works for).
//
// THE RULES (exact, not tuned):
//  - mixture pdf: a bounce is plain cosine with probability 1 - ALPHA,
//    guided otherwise, so pdf/cosine-pdf = (1 - ALPHA) + 64 ALPHA g(bin)
//    and no lit direction ever has zero probability (unbiased).
//  - the table read is LAST epoch's, never written this frame, so the pdf
//    used to sample and the pdf every light sampler's MIS weight
//    evaluates are the same number.
//  - tallies are importance-weighted (light / pdf factor), so a bin's sum
//    estimates its light no matter how often the guide picked it.
// A stale or wrong table costs noise, never wrong light.
// =====================================================================
// claude_guide_alpha: the share of bounces drawn from the table. MEASURED
// 2026-10-07 on live tables at three places (util/claude_ml_guide_train.py
// data, importance-sampling efficiency): 0.5 -> 0.9 lifts doorway room 0.836
// -> 0.946, lamp room 0.783 -> 0.959, plains 0.945 -> 0.986. Default stays
// 0.5 until the in-engine A/B (still and moving) confirms it.
uniform float claudeGuideAlpha;
#define GUIDE_ALPHA claudeGuideAlpha
const int GUIDE_BLOCK = 4;     // TUNED: cells per table side | learn by: same sweep at 2/4/8
const int GUIDE_GRID = 32;     // tables per axis, toroidal in world cells: 128 / GUIDE_BLOCK
const int GUIDE_W = 80;        // texels per table: 64 bins, 8 row sums, 1 total
const int GUIDE_PER_ROW = 51;  // tables per image row: 51 * 80 = 4080 wide
int g_guideTab = -1;           // table of the current vertex, -1 = not guided
bool g_guideKeep = true;       // does this path write to the tables (claude_guide_keep)
float g_guideT = 0.0;          // its total (read table)
vec3 g_guideN = vec3(0.0);
float g_guideT0 = -1.0, g_guided = 0.0, g_guideMis = 0.0, g_guideMis2 = 0.0; // claude_view 30
float g_guideTest = -1.0; // claude_view 31: a known integral through the first guided bounce
vec3 g_guidePair = vec3(0.0); // claude_view 32: guided bounces, ones whose re-evaluated pdf != the sampled one, worst |diff|

#ifdef CLAUDE_GUIDE_OK
ivec2 guideTexel(int t, int i)
{
	return ivec2((t % GUIDE_PER_ROW) * GUIDE_W + i, t / GUIDE_PER_ROW);
}
float guideLoad(int t, int i)
{
	return float(imageLoad(claudeGuideR, guideTexel(t, i)).r);
}
uint guideAlias(int t, int i)
{
	return imageLoad(claudeGuideA, guideTexel(t, i)).r;
}
#else
float guideLoad(int t, int i) { return 0.0; }
uint guideAlias(int t, int i) { return 0u; }
#endif

int guideFace(vec3 n)
{
	return abs(n.x) > 0.5 ? (n.x > 0.0 ? 0 : 1)
		: abs(n.y) > 0.5 ? (n.y > 0.0 ? 2 : 3) : (n.z > 0.0 ? 4 : 5);
}
int guideTable(vec3 cellW, vec3 n)
{
	ivec3 b = ivec3(floor(cellW / float(GUIDE_BLOCK))) & ivec3(GUIDE_GRID - 1);
	return ((b.z * GUIDE_GRID + b.y) * GUIDE_GRID + b.x) * 6 + guideFace(n);
}
// fixed tangents per axis: sampling and evaluation use the same frame
void guideFrame(vec3 n, out vec3 t1, out vec3 t2)
{
	if (abs(n.x) > 0.5) { t1 = vec3(0.0, 1.0, 0.0); t2 = vec3(0.0, 0.0, 1.0); }
	else if (abs(n.y) > 0.5) { t1 = vec3(1.0, 0.0, 0.0); t2 = vec3(0.0, 0.0, 1.0); }
	else { t1 = vec3(1.0, 0.0, 0.0); t2 = vec3(0.0, 1.0, 0.0); }
}
// Shirley-Chiu concentric map, square -> disk, and its inverse
// (round trip checked in Python to 4e-7)
vec2 concentric(vec2 sq)
{
	vec2 ab = 2.0 * sq - 1.0;
	if (ab.x == 0.0 && ab.y == 0.0)
		return vec2(0.0);
	float r, phi;
	if (abs(ab.x) > abs(ab.y)) { r = ab.x; phi = (PI / 4.0) * (ab.y / ab.x); }
	else { r = ab.y; phi = PI / 2.0 - (PI / 4.0) * (ab.x / ab.y); }
	return r * vec2(cos(phi), sin(phi));
}
vec2 concentricInv(vec2 d)
{
	float r = length(d);
	if (r < 1e-9)
		return vec2(0.5);
	float phi = atan(d.y, d.x);
	if (phi < -PI / 4.0)
		phi += PI2;
	vec2 ab;
	if (phi < PI / 4.0) ab = vec2(r, phi * 4.0 / PI * r);
	else if (phi < 3.0 * PI / 4.0) ab = vec2(-(phi - PI / 2.0) * 4.0 / PI * r, r);
	else if (phi < 5.0 * PI / 4.0) ab = vec2(-r, -(phi - PI) * 4.0 / PI * r);
	else ab = vec2((phi - 1.5 * PI) * 4.0 / PI * r, -r);
	return clamp(0.5 * ab + 0.5, 0.0, 0.999999);
}
int guideBinOf(vec2 sq)
{
	ivec2 c = ivec2(sq * 8.0);
	return c.y * 8 + c.x;
}
// the bounce pdf at this vertex divided by the plain cosine pdf
float guideFactorBin(int bin)
{
	if (g_guideTab < 0)
		return 1.0;
	float g = g_guideT > 0.0 ? guideLoad(g_guideTab, bin) / g_guideT : 1.0 / 64.0;
	return (1.0 - GUIDE_ALPHA) + 64.0 * GUIDE_ALPHA * g;
}
// THE CONTROL'S BOUNCES (claude_bounce_uniform 1, 2026-10-08, John: "a
// real dumb baseline ... random rays in random directions"): uniform over
// the hemisphere, pdf 1/(2 PI), instead of cosine-weighted. Written as a
// factor on the cosine pdf exactly as the guide is, so the throughput
// (tp /= factor) and every light sampler's MIS weight see the same density
// and the estimate stays unbiased with light sampling on or off.
uniform float claudeBounceUniform;
vec3 g_bounceN = vec3(0.0, 1.0, 0.0);   // the vertex normal the factor is about
vec3 uniformHemisphere(vec3 n, float u1, float u2)
{
	float z = u1;
	float r = sqrt(max(0.0, 1.0 - z * z));
	float phi = PI2 * u2;
	vec3 t = abs(n.y) > 0.5 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	vec3 tx = normalize(cross(t, n));
	vec3 ty = cross(n, tx);
	return normalize(tx * (r * cos(phi)) + ty * (r * sin(phi)) + n * z);
}
float guideFactorDir(vec3 wi)
{
	if (g_guideTab < 0 && claudeBounceUniform > 0.5)
		return 0.5 / max(abs(dot(wi, g_bounceN)), 1e-6);   // (1 / 2PI) / (cos / PI)
	if (g_guideTab < 0)
		return 1.0;
	vec3 t1, t2;
	guideFrame(g_guideN, t1, t2);
	return guideFactorBin(guideBinOf(concentricInv(vec2(dot(wi, t1), dot(wi, t2)))));
}
vec3 guideSample(vec3 n, float u1, float u2, float uc, out float factor, out int bin)
{
	vec2 sq = vec2(u1, u2);
	if (g_guideT > 0.0 && uc < GUIDE_ALPHA && claudeGuideImpl > 1.5) {
		// v2, ALIAS METHOD (2026-10-07): the table's alias, built once per epoch
		// by claude_guide_finalize from the same counts, picks bin j with
		// probability count_j / total in two reads; the within-bin point
		// takes u2 and one more draw
		float uj = u1 * 64.0;
		int j = min(int(uj), 63);
		uint a = guideAlias(g_guideTab, j);
		float prob = float(a & 0xFFFFFFu) * (1.0 / 16777216.0);
		int b = (fract(uj) < prob) ? j : int(a >> 24u);
		float u3 = rnd1();
		sq = (vec2(float(b % 8), float(b / 8)) + vec2(u2, min(u3, 0.999999))) / 8.0;
	} else
	// v1 (claude_guide_impl 1): walk the row sums, then the row (16 reads)
	if (g_guideT > 0.0 && uc < GUIDE_ALPHA) {
		float rs[8];
		float sum = 0.0;
		for (int r = 0; r < 8; r++) { rs[r] = guideLoad(g_guideTab, 64 + r); sum += rs[r]; }
		float t = u1 * sum, acc = 0.0;
		int row = 7;
		for (int r = 0; r < 8; r++) {
			if (rs[r] > 0.0 && t < acc + rs[r]) { row = r; break; }
			acc += rs[r];
		}
		float fy = rs[row] > 0.0 ? clamp((t - acc) / rs[row], 0.0, 0.999999) : 0.5;
		float cs[8];
		float csum = 0.0;
		for (int c = 0; c < 8; c++) { cs[c] = guideLoad(g_guideTab, row * 8 + c); csum += cs[c]; }
		t = u2 * csum;
		acc = 0.0;
		int col = 7;
		for (int c = 0; c < 8; c++) {
			if (cs[c] > 0.0 && t < acc + cs[c]) { col = c; break; }
			acc += cs[c];
		}
		float fx = cs[col] > 0.0 ? clamp((t - acc) / cs[col], 0.0, 0.999999) : 0.5;
		sq = (vec2(float(col), float(row)) + vec2(fx, fy)) / 8.0;
	}
	bin = guideBinOf(sq);
	factor = guideFactorBin(bin);
	vec2 d = concentric(sq);
	vec3 t1, t2;
	guideFrame(n, t1, t2);
	return normalize(t1 * d.x + t2 * d.y + n * sqrt(max(0.0, 1.0 - dot(d, d))));
}
float guideLum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
// one tally: the light that came back along the bounce, per unit
// throughput, importance-weighted. Fixed point 1/16 per unit radiance;
// one deposit is capped at 2^20 (the cap shapes the guide only).
void guideDeposit(int tab, int bin, float tpl, float l0, float f, vec3 lNow)
{
#ifdef CLAUDE_GUIDE_OK
	if (tab < 0 || tpl <= 0.0 || claudeGuideDeposit < 0.5
			|| (claudeGuideImpl > 1.5 && !g_guideKeep))
		return;
	float li = max(guideLum(lNow) - l0, 0.0) / (tpl * f);
	uint v = uint(min(li * 16.0, 1048576.0));
	if (v == 0u)
		return;
	imageAtomicAdd(claudeGuideW, guideTexel(tab, bin), v);
	// v1 also keeps row sums and the total live; v2 adds to the bin only
	// (rebuilt once per epoch by claude_guide_finalize, exact by
	// construction, and the shared total no longer serialises a block)
	if (claudeGuideImpl < 1.5) {
		imageAtomicAdd(claudeGuideW, guideTexel(tab, 64 + bin / 8), v);
		imageAtomicAdd(claudeGuideW, guideTexel(tab, 72), v);
	}
#endif
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

// LIGHT CHOICE BY PHYSICAL BOUND (claude_area_pick 1, 2026-10-07; light
// lists stage 1, DECISIONS 0w). The most light emitter i can send to x,
// with nothing in the way: its emitted luminance x the number of its
// exposed faces turned toward x (unit area each) / distance^2, the
// distance floored at one cell. A RULE (geometry and the emission law),
// not a tuned weight. Visibility is what it leaves out; stage 2 learns
// that. The sampler and neePdfSa() call this same function from the same
// x, so the pair stays exact.
float areaBound(vec4 e, vec3 x, out int k)
{
	vec3 c = e.xyz;
	int mask = int(e.w + 0.5);
	k = 0;
	for (int f = 0; f < 6; f++) {
		if (faceCandidate(c, mask, f, x))
			k++;
	}
	if (k == 0)
		return 0.0;
	vec4 sc = texture3D(claudeTraceGrid, (c + 0.5) / GRID_S);
	vec3 le = cellEmission(sc.a, cellAlbedo(sc.rgb));
	vec3 d = c + vec3(0.5) - x;
	return dot(le, vec3(0.2126, 0.7152, 0.0722)) * float(k) / max(dot(d, d), 1.0);
}
// LEARNED VISIBILITY (claude_area_pick 2, light lists stage 2). Per 4 m
// block and list slot, last epoch's shadow rays: tries and hits. The
// estimate is (hits + 1) / (tries + 2), Laplace's rule (the mean under a
// flat prior): a statistics rule, not a tuned number, and never zero, so
// every light with a bound keeps a probability.
int visBlock(vec3 x)
{
	ivec3 b = ivec3(floor((x + gridOrigin) / 4.0)) & ivec3(31);
	return (b.z * 32 + b.y) * 32 + b.x;
}
ivec2 visTexel(int blk, int slot, int k)
{
	int i = blk * 32 + slot * 2 + k;
	return ivec2(i & 1023, i >> 10);
}
// claudeVisP: per block, 17 floats (16 slot estimates, then p_call), laid
// out blk * 17 + i over a 1024-wide image; written by claude_vis_finalize
ivec2 visPTexel(int blk, int i)
{
	int k = blk * 17 + i;
	return ivec2(k & 1023, k >> 10);
}
float visEst(int blk, int slot)
{
#ifdef CLAUDE_GUIDE_OK
	return imageLoad(claudeVisP, visPTexel(blk, slot)).r;
#else
	return 1.0;
#endif
}
void visDeposit(int blk, int slot, int k)
{
#ifdef CLAUDE_GUIDE_OK
	imageAtomicAdd(claudeVisW, visTexel(blk, slot, k), 1u);
#endif
}
// slot j's weight at x: its bound, times its learned visibility at pick 2
float areaW(int j, vec3 x, int blk, out int k)
{
	float w = areaBound(areaEmitter(j), x, k);
	if (claudeAreaPick > 1.5 && w > 0.0)
		w *= visEst(blk, j);
	return w;
}
// HOW OFTEN TO SAMPLE AREA LIGHTS AT ALL (claude_area_skip). s = the
// expected share of this block's light samples that will reach a light;
// a block whose lights are all walled off stops paying a shadow ray per
// bounce. Both sides of the MIS pair fold it into p_l, so it costs noise,
// never light. TUNED: p_call = clamp(4 s, 1/16, 1) | learn by: equal-time
// error sweep, then the learned effort model (DECISIONS 0w)
float areaCall(vec3 x, int nLights)
{
	if (claudeAreaSkip < 0.5 || claudeAreaPick < 1.5)
		return 1.0;
#ifdef CLAUDE_GUIDE_OK
	return imageLoad(claudeVisP, visPTexel(visBlock(x), 16)).r;
#else
	return 1.0;
#endif
}
// the sum over the list, and emitter i's share of it
float areaPickProb(int i, vec3 x, int nLights)
{
	if (claudeAreaPick < 0.5)
		return 1.0 / float(nLights);
	float wsum = 0.0, wi = 0.0;
	int kk;
	for (int j = 0; j < AREA_CAP; j++) {
		if (j >= nLights)
			break;
		float w = areaW(j, x, visBlock(x), kk);
		wsum += w;
		if (j == i)
			wi = w;
	}
	return wsum > 0.0 ? wi / wsum : 0.0;
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
	float pList = 1.0 / float(nLights);
	if (claudeAreaPick > 0.5) {
		int iHit = 0;
		for (int i = 0; i < AREA_CAP; i++) {
			if (i >= nLights)
				break;
			if (all(lessThan(abs(areaEmitter(i).xyz - cellHit), vec3(CELL_MATCH_EPS)))) {
				iHit = i;
				break;
			}
		}
		pList = areaPickProb(iHit, x, nLights);
		if (pList <= 0.0)
			return 0.0;
	}
	int k = 0;
	for (int f = 0; f < 6; f++) {
		if (faceCandidate(cellHit, mask, f, x))
			k++;
	}
	if (k == 0)
		return 0.0; // unreachable: the hit face itself passed the test
	// p_A = pList/k over unit-square faces (pList = 1/N when uniform),
	// converted to solid angle
	return (dist * dist) * pList * areaCall(x, nLights) / (float(k) * cosY);
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

//
// curMed (2026-08-18) is the medium the RECEIVING vertex sits in, handed
// to the shadow march so that a surface at the bottom of a tank is not
// shadowed by the water it is standing in. It changes nothing about the
// pdf: the light sampler still cannot see through a refractive interface,
// because a straight shadow ray is not the path a refracted one takes.
// That is a variance cost and never an energy one — a face reachable only
// through glass is reached by the BSDF technique at MIS weight 1, since
// the interface disarms MIS (see the path loop's dielectric block).
// THE ONE WALK, declared here and defined with the far ladder below:
// every light sampler walks through it.
bool marchAll(vec3 ro, vec3 rd, float curMed, out vec3 hp, out vec3 n,
		out vec3 alb, out vec3 le, out float tHit, out vec3 cellOut,
		out vec4 palOut, out float idxOut, out vec3 hpFar);
float farExitT(vec3 p, vec3 d);

vec3 neeDirect(vec3 x, vec3 nx, vec3 rho, int nLights, float misOn,
		float curMed)
{
	g_neeDiag = vec4(0.0);
	g_neeDiagK = 0.0;
	float pCall = areaCall(x, nLights);
	if (pCall < 1.0 && rnd1() >= pCall)
		return vec3(0.0); // not this time (claude_area_skip); priced in p_l
	// uniform over the list (claude_area_pick 0), or by each light's
	// physical bound (1). Either way p_A is reproducible by neePdfSa()
	// from the hit alone: 1/(N*k), or bound_i/sum/k from the same x.
	float us = rnd1();
	int li = min(int(float(nLights) * us), nLights - 1);
	float pList = 1.0 / float(nLights);
	if (claudeAreaPick > 0.5) {
		// by physical bound: walk the CDF with the same draw
		float wsum = 0.0;
		int kk;
		for (int j = 0; j < AREA_CAP; j++) {
			if (j >= nLights)
				break;
			wsum += areaW(j, x, visBlock(x), kk);
		}
		if (wsum <= 0.0)
			return vec3(0.0);
		float t = us * wsum, acc = 0.0;
		li = -1;
		for (int j = 0; j < AREA_CAP; j++) {
			if (j >= nLights)
				break;
			float w = areaW(j, x, visBlock(x), kk);
			if (li < 0 && w > 0.0 && t < acc + w) {
				li = j;
				pList = w / wsum;
			}
			acc += w;
		}
		if (li < 0)
			return vec3(0.0);
	}
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
	vec4 shpal;
	float shidx;
	vec3 shfar;
	if (claudeAreaPick > 1.5)
		visDeposit(visBlock(x), li, 0);   // a try
	if (!marchAll(x, wi, curMed, shp, shn, shalb, shle, sht, shcell,
			shpal, shidx, shfar))
		return vec3(0.0);
	if (any(greaterThanEqual(abs(shcell - c), vec3(CELL_MATCH_EPS))))
		return vec3(0.0); // occluded
	if (claudeAreaPick > 1.5)
		visDeposit(visBlock(x), li, 1);   // a hit

	// THE LAW: one Le. shle came out of march(), which ran the same
	// cellEmission() the eye ray runs — no second formula here, and no
	// CPU-side copy of Le to drift from it. Deliberately NOT guarded
	// against shle == 0: a non-emissive cell in the list would have to
	// be a snapshot bug, and multiplying it through returns black on its
	// own. An early-out here would instead hide it, and would make
	// neeDirect() decline a direction neePdfSa() still prices — the one
	// asymmetry that actually loses energy.
	float pdfL = dist2 * pList * pCall / (float(k) * cosY); // p_l, sa (pList = 1/N when uniform)
	float pdfB = g_neeBScale * guideFactorDir(wi) * cosX / PI;             // p_b, sa
	float w = 1.0;
	if (misOn > 0.5)
		w = pdfL / (pdfL + pdfB);                        // balance
	// forensics: this call is about to contribute. Record WHAT it aimed
	// at and at what grazing angle, in world coords, for views 12/13.
	g_neeDiag = vec4(1.0, c.y + gridOrigin.y, cosX, float(li));
	g_neeDiagK = float(k);
	// f_r = rho/PI for a Lambertian; estimator = w * f_r * Le * cos_x/p_l
	// AIR: the light crosses `dist` metres of it (1 with no medium)
	// ...or the medium the vertex is in (water absorbs, 2026-10-05)
	vec3 tr = curMed < 0.5 ? vec3(airTr(dist)) : medTr(curMed, dist);
	return w * (rho / PI) * shle * (cosX / pdfL) * tr;
}

// neeDirect() for a point IN THE AIR (2026-10-04): the same emitter-list
// sample, with the phase function where a surface has rho/PI * cos and no
// facing test. `dir` is the path's travel direction into the point. This
// is what gives a torch or a glowing block its halo in haze.
vec3 neeDirectAir(vec3 x, vec3 dir, int nLights)
{
	float pCall = areaCall(x, nLights);
	if (pCall < 1.0 && rnd1() >= pCall)
		return vec3(0.0);
	float us = rnd1();
	int li = min(int(float(nLights) * us), nLights - 1);
	float pList = 1.0 / float(nLights);
	if (claudeAreaPick > 0.5) {
		// the same pick as neeDirect(), so neePdfSa() prices it right
		float wsum = 0.0;
		int kk;
		for (int j = 0; j < AREA_CAP; j++) {
			if (j >= nLights)
				break;
			wsum += areaW(j, x, visBlock(x), kk);
		}
		if (wsum <= 0.0)
			return vec3(0.0);
		float t = us * wsum, acc = 0.0;
		li = -1;
		for (int j = 0; j < AREA_CAP; j++) {
			if (j >= nLights)
				break;
			float w = areaW(j, x, visBlock(x), kk);
			if (li < 0 && w > 0.0 && t < acc + w) {
				li = j;
				pList = w / wsum;
			}
			acc += w;
		}
		if (li < 0)
			return vec3(0.0);
	}
	vec4 e = areaEmitter(li);
	vec3 c = e.xyz;
	int mask = int(e.w + 0.5);
	int k = 0;
	for (int f = 0; f < 6; f++) {
		if (faceCandidate(c, mask, f, x))
			k++;
	}
	if (k == 0)
		return vec3(0.0);
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
	vec3 nL = faceNormal(face);
	float u1 = rnd1();
	float u2 = rnd1();
	vec3 ta = abs(nL.y) > 0.5 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
	vec3 tx = cross(ta, nL);
	vec3 ty = cross(nL, tx);
	vec3 y = c + vec3(0.5) + nL * 0.5
			+ tx * (u1 - 0.5) + ty * (u2 - 0.5);
	vec3 d = y - x;
	float dist2 = dot(d, d);
	if (dist2 < 1e-8)
		return vec3(0.0);
	float dist = sqrt(dist2);
	vec3 wi = d / dist;
	float cosY = dot(nL, -wi);
	if (cosY <= NEE_COS_MIN)
		return vec3(0.0);
	vec3 shp, shn, shalb, shle, shcell;
	float sht;
	vec4 shpal;
	float shidx;
	vec3 shfar;
	if (!marchAll(x, wi, 0.0, shp, shn, shalb, shle, sht, shcell,
			shpal, shidx, shfar))
		return vec3(0.0);
	if (any(greaterThanEqual(abs(shcell - c), vec3(CELL_MATCH_EPS))))
		return vec3(0.0);
	float pdfL = dist2 * pList * pCall / (float(k) * cosY);
	float ph = hgPhase(dot(dir, wi), claudeAirG);
	float w = pdfL / (pdfL + ph);
	return w * ph * shle * airTr(dist) / pdfL;
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
//
// curMed, as in neeDirect(): the medium the receiving vertex is in, so a
// shadow ray leaving a surface inside a tank of water is not stopped by
// the water. It still cannot see the sky THROUGH the surface of that
// tank, and it should not — that path is refracted, and a straight ray is
// not it.
// =====================================================================
// AIMING AT FLAMES (claude_torch_nee, 2026-10-05)
// =====================================================================
// A torch's light leaves through a few 1/16 m flame voxels, and a path
// finds them only by chance: the sparkle. This aims at them. Pick one of
// the nearest flames uniformly, then a direction uniformly inside the cone
// that just contains the sphere around ALL its emitting voxels, and march
// it. Light counts ONLY if the ray really hits an emitting fine voxel, so
// the cone is a sampling region and never a light shape: nothing is
// invented where the flame is not (the 2026-08-13 jittered-target attempt
// aimed AT made-up points and was reverted for exactly that). Full-cube
// lamps are the area list's (neeDirect) and are not counted here.
//
// The density of a direction is the MIXTURE over every cone that contains
// it, ptPdfSa(), used identically by this estimator and by the BSDF ray
// that hits a flame (balance heuristic), so the pair sums to one for every
// direction and hit. A vertex inside a flame's sphere has no cone for it.
vec4 ptEmitter(int i)
{
	if (i == 0) return claudeEmitter0;
	if (i == 1) return claudeEmitter1;
	if (i == 2) return claudeEmitter2;
	if (i == 3) return claudeEmitter3;
	if (i == 4) return claudeEmitter4;
	if (i == 5) return claudeEmitter5;
	if (i == 6) return claudeEmitter6;
	return claudeEmitter7;
}

// cos of the cone's half angle from x, or 2.0 = no cone (x inside, or empty)
float ptConeCos(vec4 e, vec3 x)
{
	if (e.w <= 0.0)
		return 2.0;
	vec3 d = e.xyz - x;
	float d2 = dot(d, d);
	float r2 = e.w * e.w;
	if (d2 <= r2 * 1.0001)
		return 2.0;
	return sqrt(1.0 - r2 / d2);
}

// WHICH FLAME, AND WHETHER TO AIM AT ALL (2026-10-06). A flame is chosen
// in proportion to the solid angle its sphere covers from x, Omega_i =
// 2 pi (1 - cos theta_max), and the whole sampler runs only with
// probability q = min(1, sum Omega / PT_OMEGA_REF). Both are importance
// sampling, so both keep it unbiased: the density of a direction is
// q * (number of cones holding it) / sum Omega, and the BSDF side of the
// MIS pair reads that same number through ptPdfSa(). Why: uniform choice
// spent a shadow ray on every vertex even when every flame was tens of
// blocks away (forest: 1.80 ms of 13.25, measured.md 2026-10-06 ledger).
// TUNED: PT_OMEGA_REF = 0.01 sr, about a torch flame seen from 2-3 m |
// learn by: the speed ledger and the torch-room noise A/B.
const float PT_OMEGA_REF = 0.01;
float ptOmegaSum(vec3 x, int nPts)
{
	float sum = 0.0;
	for (int i = 0; i < 8; i++) {
		if (i >= nPts)
			break;
		float cm = ptConeCos(ptEmitter(i), x);
		if (cm < 1.5)
			sum += PI2 * (1.0 - cm);
	}
	return sum;
}

float ptPdfSa(vec3 x, vec3 wi, int nPts)
{
	float sum = ptOmegaSum(x, nPts);
	if (sum <= 0.0)
		return 0.0;
	float hits = 0.0;
	for (int i = 0; i < 8; i++) {
		if (i >= nPts)
			break;
		vec4 e = ptEmitter(i);
		float cm = ptConeCos(e, x);
		if (cm > 1.5)
			continue;
		if (dot(wi, normalize(e.xyz - x)) >= cm)
			hits += 1.0;
	}
	return min(1.0, sum / PT_OMEGA_REF) * hits / sum;
}

vec3 neePoint(vec3 x, vec3 nx, vec3 rho, int nPts, float curMed)
{
	float uq = rndPt();
	float us = rndPt();
	float u1 = rndPt();
	float u2 = rndPt();
	float sum = ptOmegaSum(x, nPts);
	if (sum <= 0.0 || uq >= min(1.0, sum / PT_OMEGA_REF))
		return vec3(0.0);
	// pick i with probability Omega_i / sum
	int li = -1;
	float acc = 0.0, cm = 2.0;
	vec4 e = vec4(0.0);
	for (int i = 0; i < 8; i++) {
		if (i >= nPts)
			break;
		vec4 ei = ptEmitter(i);
		float ci = ptConeCos(ei, x);
		if (ci > 1.5)
			continue;
		acc += PI2 * (1.0 - ci);
		if (li < 0 && us * sum < acc) {
			li = i;
			e = ei;
			cm = ci;
		}
	}
	if (li < 0)
		return vec3(0.0);
	vec3 w = normalize(e.xyz - x);
	float ct = 1.0 - u1 * (1.0 - cm);
	float st = sqrt(max(0.0, 1.0 - ct * ct));
	float ph = PI2 * u2;
	vec3 a = abs(w.x) > 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
	vec3 t = normalize(cross(a, w));
	vec3 b = cross(w, t);
	vec3 wi = normalize(t * (cos(ph) * st) + b * (sin(ph) * st) + w * ct);
	float cosX = dot(nx, wi);
	if (cosX <= 0.0)
		return vec3(0.0);
	vec3 shp, shn, shalb, shle, shcell;
	float sht;
	vec4 shpal;
	float shidx;
	vec3 shfar;
	if (!marchAll(x, wi, curMed, shp, shn, shalb, shle, sht, shcell,
			shpal, shidx, shfar))
		return vec3(0.0);
	if (!matFine(shpal) || !any(greaterThan(shle, vec3(0.0))))
		return vec3(0.0);         // not a flame: not this estimator's
	float pdfL = ptPdfSa(x, wi, nPts);
	float pdfB = g_neeBScale * guideFactorDir(wi) * cosX / PI;
	if (pdfL <= 0.0)
		return vec3(0.0);
	vec3 tr = curMed < 0.5 ? vec3(airTr(sht)) : medTr(curMed, sht);
	// f Le cos / p_L, times the balance weight p_L / (p_L + p_B)
	return (rho / PI) * shle * cosX / (pdfL + pdfB) * tr;
}

// =====================================================================
// THE FAR LADDER (claude_far_levels, 2026-10-06; John: "infinite draw
// distance should be possible right? By scaling lod?")
// =====================================================================
// Past the 128^3 1 m grid the world continues in coarser copies of
// itself: level k has 128^3 cells of 2^(k+1) m (2 m to +-128, 4 m to
// +-256, 8 m to +-512, ...), built by claude_lod's fold. Each cell's byte
// is a MATERIAL INDEX into the SAME palette as the near grid, read by the
// SAME arrival rule and lit by the SAME law -- physics-contract §7,
// "geometry may be laddered with distance; the light law may not". Every
// ray type uses this one walk: eye, bounce, and every shadow ray
// (marchAll() below), so no ray sees a different world from another.
//
// SEAMS ARE STRUCTURAL, NOT TUNED: the near grid's corner sits on even
// nodes and every level snaps to whole MapBlocks, so each boundary lies on
// cell faces of the next level out. A ray handed across a boundary starts
// on a clean face: no coarse cell straddles the seam (the half-in coarse
// cell was 2026-08-12's "rampart"). A ray always walks the FINEST level
// that holds its current point, stepping finer again if it heads inward.
//
// What a coarse cell is, today: a box, solid when at least half its
// volume was (claude_lod, majority fold), coloured by its top layer. Known
// to be approximate (opaque boxes thicken silhouettes and cannot carry the
// tilt of a hillside; spec/far-field-plan.md, research 2026-10-06); the
// 1 m vs 2 m energy referee measures how approximate before anything is
// added.
uniform sampler3D claudeCascades;   // unit 8: 128 x 128 x 640, level k = slab k
// unit 9, same layout: each solid cell's TIGHT BOX (min | max << 4 per
// axis, in units of max(1, cell/16) nodes). A solid far cell is hit where
// its box is, so one layer of ground in a 2 m cell is a 1 m slab and a 1 m
// trunk a 1 m column -- the box test is the "AABB math", and its entry
// face is the normal.
uniform sampler3D claudeCascadeBox;
uniform vec3 cascade0Origin;        // grid-local node coords of each
uniform vec3 cascade1Origin;        // level's cell (0,0,0)
uniform vec3 cascade2Origin;
uniform vec3 cascade3Origin;
uniform vec3 cascade4Origin;
uniform float claudeFarLevels;      // valid levels from 0 out (0 = none)
uniform float claudeFarOnly;        // INSTRUMENT: 1 = skip the 1 m grid
uniform float claudeFarLeafMedium;  // 1 = far leaf cells are a cloud of sheets
// A LEAF BLOCK'S OPTICAL DEPTH PER METRE, PER AXIS (2026-10-06). Geometry,
// not tuning: the leaf model is six perforated 1/16 sheets
// (claude_models.leaf_sheets), and the share of axis-parallel rays that
// miss every voxel of a block is T_x = T_z = 0.043, T_y = 0.184 (mean of
// the six leaf models; -ln T = 3.147 / 1.695 / 3.147). A ray of direction
// d meets x-sheets at a rate |d.x| per metre of x travelled, so a far cell
// of leaf density rho attenuates as rho * dot(|d|, LEAF_SIGMA) per metre.
// APPROXIMATION: sheet families treated as independently placed; the
// oblique transmission of one real block is not measured yet.
const vec3 LEAF_SIGMA = vec3(3.147, 1.695, 3.147);
// THE SAME FOR A TALL-GRASS NODE (claude_models crossed_cutout of
// mcl_flowers_tallgrass): straight-through T = 0.469 / 0.891 / 0.539, so
// the sun mostly reaches the ground through grass while a low view sees
// mostly blades. Ferns are within 15 %.
const vec3 GRASS_SIGMA = vec3(0.758, 0.116, 0.618);
uniform float claudeFarPlants;      // 1 = far plants as a layer of blades
// "NOTHING ABOVE HERE" (2026-10-06): grid-local y above which no far level
// holds anything. A ray outside the near grid that is above it and not
// moving down can hit nothing more -- the grid is a convex box it cannot
// re-enter -- so it has escaped to the sky. Exact. Found because the far
// view cost ~130 ms a frame in PLAY (direct light on): every sun shadow ray
// walked the far levels cell by cell (spec/measured.md 2026-10-06).
uniform float claudeFarTop;
uniform float claudePixelReset;     // 1 = a pixel whose surface changed drops its own history
bool g_farMedium = false;           // the last far hit was a cloud's
const int FAR_STEPS = 400;
// set by marchAll(): did the last hit land on a far level?
bool g_lastFar = false;
// a far hit's cellOut: no near-grid lookup may be made with it
const vec3 FAR_CELL = vec3(-10000.0);

vec3 farOrigin(int k)
{
	if (k == 0) return cascade0Origin;
	if (k == 1) return cascade1Origin;
	if (k == 2) return cascade2Origin;
	if (k == 3) return cascade3Origin;
	return cascade4Origin;
}
float farCellSize(int k) { return exp2(float(k + 1)); }
int farLevelCount() { return int(claudeFarLevels + 0.5); }
bool inBoxW(vec3 p, vec3 lo, float size)
{
	return all(greaterThanEqual(p, lo)) && all(lessThan(p, lo + vec3(size)));
}
bool nearOn() { return claudeFarOnly < 0.5; }
// the finest representation holding grid-local point p: -1 = the 1 m
// grid, 0.. = a far level, 99 = nothing (past the ladder)
int levelAt(vec3 p)
{
	if (nearOn() && inBoxW(p, vec3(0.0), GRID_S))
		return -1;
	int L = farLevelCount();
	for (int k = 0; k < 5; k++) {
		if (k >= L)
			break;
		if (inBoxW(p, farOrigin(k), 128.0 * farCellSize(k)))
			return k;
	}
	return 99;
}
// distance from p along d to the edge of everything that exists (the
// outermost level, or the 1 m grid when there is no ladder): where AIR
// ends for the free-flight sampler, and where sky light starts
float farExitT(vec3 p, vec3 d)
{
	int L = farLevelCount();
	vec3 lo = L > 0 ? farOrigin(L - 1) : vec3(0.0);
	float size = L > 0 ? 128.0 * farCellSize(L - 1) : GRID_S;
	vec3 room = mix(p - lo, lo + vec3(size) - p, step(0.0, d));
	vec3 tt = room / max(abs(d), vec3(1e-6));
	return max(min(tt.x, min(tt.y, tt.z)), 0.0);
}

// One level's walk. Starts AT p0 (tested, unlike the near walk's origin
// exclusion: a hand-off point lies on a cell face and its cell has not
// been seen; a surface restart lies in the air cell it came through, so
// testing it costs nothing). Returns 1 on a hit/interface, 0 when the ray
// leaves the level (pExit) -- outward, or inward into a finer region.
int marchFarLevel(int k, vec3 p0, vec3 rd, float tBase, float curMed,
		int entryAxis, out vec3 hp, out vec3 n, out vec3 alb, out vec3 le,
		out float tHit, out vec4 palOut, out float idxOut, out vec3 hpFar,
		out vec3 pExit)
{
	float h = farCellSize(k);
	vec3 org = farOrigin(k);
	vec3 q = (p0 - org) / h;
	vec3 stepDir = sign(rd);
	vec3 delta = 1.0 / max(abs(rd), vec3(DDA_MIN_ABS));
	vec3 c = floor(q + rd * 1e-4);
	vec3 sideDist = (stepDir * (c - q) + stepDir * 0.5 + 0.5) * delta;
	float t = 0.0;
	int axis = entryAxis;
	// the finer region inside this level, in this level's cell units
	vec3 fLo = k == 0 ? (vec3(0.0) - org) / h : (farOrigin(k - 1) - org) / h;
	float fSize = k == 0 ? (nearOn() ? GRID_S / h : 0.0)
			: 128.0 * farCellSize(k - 1) / h;
	pExit = p0;
	for (int i = 0; i < FAR_STEPS; i++) {
		g_steps += 1.0;
		if (rd.y >= 0.0 && p0.y + rd.y * (t * h) >= claudeFarTop) {
			pExit = p0 + rd * (t * h);
			return 2;   // above everything: escaped
		}
		if (any(lessThan(c, vec3(0.0))) || any(greaterThanEqual(c, vec3(128.0)))
				|| (fSize > 0.0 && all(greaterThanEqual(c, fLo))
					&& all(lessThan(c, fLo + vec3(fSize))))) {
			pExit = p0 + rd * (t * h);
			return 0;
		}
		vec4 sv = texture3D(claudeCascades,
				(vec3(c.x, c.y, c.z + 128.0 * float(k)) + 0.5)
				/ vec3(128.0, 128.0, 640.0));
		float idx = matIndex(sv.a);
		bool hitHere = idx != curMed;
		float tIn = t;              // where the ray is along the cell (level units)
		int ax = axis >= 0 ? axis : 1;
		if (hitHere) {
			vec4 pal0 = matPalIdx(idx);
			if (!matTransmits(idx, pal0)) {
				// the box test, inside this cell's stretch of the ray
				vec4 bx4 = texture3D(claudeCascadeBox,
						(vec3(c.x, c.y, c.z + 128.0 * float(k)) + 0.5)
						/ vec3(128.0, 128.0, 640.0));
				vec3 bb = floor(bx4.rgb * 255.0 + 0.5);
				// alpha: 255 solid box; 1..127 a leaf cell's leaf density;
				// 128..254 plant density (a plant-only leaf-material cell,
				// or the blade layer above a solid cell's box)
				float code = floor(bx4.a * 255.0 + 0.5);
				bool leafCell = abs(matHot(idx) - 3.0) < 0.5;
				float rhoL = (code >= 1.0 && code <= 127.0) ? code / 127.0 : 0.0;
				float rhoP = (code >= 128.0 && code <= 254.0) ? (code - 127.0) / 127.0 : 0.0;
				float mRho = 0.0;
				vec3 mSig = LEAF_SIGMA;
				bool mWhole = false, mAbove = false, skipCell = false;
				if (leafCell && rhoL > 0.0 && claudeFarLeafMedium > 0.5) {
					mRho = rhoL; mWhole = true;
				} else if (leafCell && rhoP > 0.0) {
					if (claudeFarPlants > 0.5) {
						mRho = rhoP; mSig = GRASS_SIGMA; mWhole = true;
					} else {
						skipCell = true;
					}
				} else if (!leafCell && rhoP > 0.0 && claudeFarPlants > 0.5) {
					mRho = rhoP; mSig = GRASS_SIGMA; mAbove = true;
				}
				float unit = max(1.0, h / 16.0);
				vec3 blo = org + c * h + mod(bb, 16.0) * unit;
				vec3 bhi = org + c * h + (floor(bb / 16.0) + 1.0) * unit;
				vec3 inv = 1.0 / (sign(rd) * max(abs(rd), vec3(1e-7))
						+ vec3(rd.x == 0.0 ? 1e-7 : 0.0, rd.y == 0.0 ? 1e-7 : 0.0,
							rd.z == 0.0 ? 1e-7 : 0.0));
				vec3 t0 = (blo - p0) * inv, t1 = (bhi - p0) * inv;
				vec3 tn = min(t0, t1), tf = max(t0, t1);
				float te = max(max(tn.x, tn.y), tn.z);
				float tx = min(min(tf.x, tf.y), tf.z);
				float c0 = t * h;
				float c1 = min(sideDist.x, min(sideDist.y, sideDist.z)) * h;
				bool boxSeg = te <= tx && tx >= c0 && te <= c1;
				if (skipCell) {
					hitHere = false;
				} else if (mWhole) {
					// a cloud of sheets filling the box: free flight inside
					// its stretch of this cell; the sheet family met is
					// the one whose exponential came first
					hitHere = false;
					if (boxSeg) {
						vec3 w3 = abs(rd) * mSig * mRho;
						float sig = w3.x + w3.y + w3.z;
						float a0 = max(te, c0), a1 = min(tx, c1);
						float s = -log(max(1e-7, 1.0 - rnd1())) / max(sig, 1e-6);
						if (a0 + s < a1) {
							hitHere = true;
							tIn = (a0 + s) / h;
							float u = rnd1() * sig;
							ax = u < w3.x ? 0 : (u < w3.x + w3.y ? 1 : 2);
							g_farMedium = true;
						}
					}
				} else {
					// the solid box, and (mAbove) a layer of blades between
					// the box top and the cell top; the nearer event wins
					float tBox = 1e30;
					int axBox = ax;
					if (boxSeg) {
						if (te > c0) {
							tBox = te;
							axBox = (tn.x >= tn.y && tn.x >= tn.z) ? 0
									: (tn.y >= tn.z ? 1 : 2);
						} else {
							tBox = c0;
						}
					}
					float tMed = 1e30;
					int axMed = 1;
					if (mAbove) {
						vec3 slo = vec3(org.x + c.x * h, bhi.y, org.z + c.z * h);
						vec3 shi = org + (c + 1.0) * h;
						if (shi.y > slo.y + 1e-4) {
							vec3 s0 = (slo - p0) * inv, s1 = (shi - p0) * inv;
							vec3 sn = min(s0, s1), sf = max(s0, s1);
							float a0 = max(max(max(sn.x, sn.y), sn.z), c0);
							float a1 = min(min(min(sf.x, sf.y), sf.z), c1);
							if (a0 < a1) {
								vec3 w3 = abs(rd) * mSig * mRho;
								float sig = w3.x + w3.y + w3.z;
								float s = -log(max(1e-7, 1.0 - rnd1())) / max(sig, 1e-6);
								if (a0 + s < a1) {
									tMed = a0 + s;
									float u = rnd1() * sig;
									axMed = u < w3.x ? 0 : (u < w3.x + w3.y ? 1 : 2);
								}
							}
						}
					}
					if (tMed < tBox) {
						tIn = tMed / h;
						ax = axMed;
						g_farMedium = true;
					} else if (tBox < 1e29) {
						tIn = tBox / h;
						ax = axBox;
					} else {
						hitHere = false;   // the empty part of the cell
					}
				}
			}
		}
		if (hitHere) {
			vec4 pal = matPalIdx(idx);
			n = vec3(0.0);
			if (ax == 0) n.x = -stepDir.x;
			else if (ax == 1) n.y = -stepDir.y;
			else n.z = -stepDir.z;
			vec3 ph = p0 + rd * (tIn * h);
			// off the face by a hair: the face may sit inside this cell (a
			// box), so the cell-clamping restart is not the right tool
			hp = ph + n * (1e-3 * h);
			hpFar = ph - n * (1e-3 * h);
			alb = cellAlbedo(sv.rgb);
			le = alb * pal.r;
			if (pal.r > 0.0)
				hotLaw(idx, FAR_CELL, false, vec3(0.0), alb, le);
			tHit = tBase + tIn * h;
			palOut = pal;
			idxOut = idx;
			return 1;
		}
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x; sideDist.x += delta.x; c.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y; sideDist.y += delta.y; c.y += stepDir.y; axis = 1;
		} else {
			t = sideDist.z; sideDist.z += delta.z; c.z += stepDir.z; axis = 2;
		}
	}
	pExit = p0 + rd * (t * h);
	return 0;
}

// which face of the box [lo, lo+size) a ray from inside leaves through
int boxExitAxis(vec3 p, vec3 d, vec3 lo, float size)
{
	vec3 room = mix(p - lo, lo + vec3(size) - p, step(0.0, d));
	vec3 tt = room / max(abs(d), vec3(1e-6));
	return tt.x < tt.y ? (tt.x < tt.z ? 0 : 2) : (tt.y < tt.z ? 1 : 2);
}

// THE ONE WALK every ray takes. The 1 m grid (with its 1/16 m rung) where
// it holds the point; the ladder everywhere else; the sky past both.
bool marchAll(vec3 ro, vec3 rd, float curMed, out vec3 hp, out vec3 n,
		out vec3 alb, out vec3 le, out float tHit, out vec3 cellOut,
		out vec4 palOut, out float idxOut, out vec3 hpFar)
{
	g_lastFar = false;
	g_farMedium = false;
	g_rays += 1.0;
	if (farLevelCount() == 0 && nearOn())
		return marchMed(ro, rd, curMed, hp, n, alb, le, tHit, cellOut,
				palOut, idxOut, hpFar);
	hp = ro; n = vec3(0.0, 1.0, 0.0); alb = vec3(0.0); le = vec3(0.0);
	tHit = DEPTH_MISS; cellOut = vec3(-1.0); palOut = vec4(0.0);
	idxOut = 0.0; hpFar = ro;
	vec3 p = ro;
	float tb = 0.0;
	int lv = levelAt(ro);
	int axis = -1;
	for (int pass = 0; pass < 12; pass++) {
		if (lv == 99)
			return false;
		if (lv == -1) {
			if (pass > 0) {
				// ENTERING the 1 m grid from outside: the near walk never
				// tests its first cell, so test it here (a solid boundary
				// cell is a hit on its outer face; its 1/16 m rung is not
				// walked on this one arrival)
				vec3 cs = floor(p + rd * 1e-4);
				vec4 s0 = texture3D(claudeTraceGrid, (cs + 0.5) / GRID_S);
				float i0 = matIndex(s0.a);
				if (i0 != curMed) {
					vec4 pal0 = matPalIdx(i0);
					n = vec3(0.0);
					vec3 sd = sign(rd);
					if (axis == 0) n.x = -sd.x;
					else if (axis == 1) n.y = -sd.y;
					else n.z = -sd.z;
					hp = restartPoint(p, n, cs + n, 1.0);
					hpFar = restartPoint(p, -n, cs, 1.0);
					alb = cellAlbedo(s0.rgb);
					le = alb * pal0.r;
					tHit = tb;
					cellOut = cs;
					palOut = pal0;
					idxOut = i0;
					return true;
				}
			}
			if (marchMed(p, rd, curMed, hp, n, alb, le, tHit, cellOut,
					palOut, idxOut, hpFar)) {
				tHit += tb;
				return true;
			}
			float te = airExitT(p, rd);
			axis = boxExitAxis(p, rd, vec3(0.0), GRID_S);
			p += rd * te;
			tb += te;
		} else {
			vec3 pe;
			int far = marchFarLevel(lv, p, rd, tb, curMed, axis, hp, n, alb, le,
					tHit, palOut, idxOut, hpFar, pe);
			if (far == 1) {
				cellOut = FAR_CELL;
				g_lastFar = true;
				return true;
			}
			if (far == 2)
				return false;   // above everything: the sky
			// which face it left by: the axis whose boundary pe sits on
			vec3 fr = abs(fract((pe - farOrigin(lv)) / farCellSize(lv) + 0.5) - 0.5);
			axis = fr.x < fr.y ? (fr.x < fr.z ? 0 : 2) : (fr.y < fr.z ? 1 : 2);
			tb += length(pe - p);
			p = pe;
		}
		lv = levelAt(p + rd * 1e-3);
	}
	return false;
}

// Is the point x inside anything a ray would stop at? (leaf transmission's
// "is there air behind this sheet")
bool pointSolid(vec3 x)
{
	vec3 c = floor(x);
	if (any(lessThan(c, vec3(0.0))) || any(greaterThanEqual(c, vec3(GRID_S))))
		return false;
	vec4 s = texture3D(claudeTraceGrid, (c + 0.5) / GRID_S);
	if (s.a <= MAT_AIR_MAX)
		return false;
	vec4 pal = matPal(s.a);
	if (!fineHereAt(pal, c))
		return true;
	return fineSolid(c, floor((x - c) * SUBV));
}

vec3 neeSky(vec3 x, vec3 nx, vec3 rho, float curMed)
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
	vec4 shpal;
	float shidx;
	vec3 shfar;
	if (marchAll(x, wi, curMed, shp, shn, shalb, shle, sht, shcell,
			shpal, shidx, shfar))
		return vec3(0.0); // occluded

	// THE LAW: one sky. skyBody() here is the same evaluation the camera
	// ray's escape runs — there is no second radiance for the shadow ray.
	float pdfL = 1.0 / (PI2 * (1.0 - bcos)); // p_sky, sa
	float pdfB = g_neeBScale * guideFactorDir(wi) * cosX / PI;  // p_b, sa
	float w = pdfL / (pdfL + pdfB);          // balance heuristic
	// AIR between here and the edge of the grid (1 with no medium)
	vec3 tr = curMed < 0.5 ? vec3(airTr(farExitT(x, wi)))
			: medTr(curMed, farExitT(x, wi));
	return w * (rho / PI) * skyBody(wi) * (cosX / pdfL) * tr;
}

// neeSky() for a point IN THE AIR: the body sample with the phase function
// and no facing test. These are the light shafts — a scattering point that
// can see the sun lights up, one in a building's shadow does not.
vec3 neeSkyAir(vec3 x, vec3 dir)
{
	vec3 bdir;
	float bcos;
	if (!skyNeeBody(bdir, bcos))
		return vec3(0.0);
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
	vec3 shp, shn, shalb, shle, shcell;
	float sht;
	vec4 shpal;
	float shidx;
	vec3 shfar;
	if (marchAll(x, wi, 0.0, shp, shn, shalb, shle, sht, shcell,
			shpal, shidx, shfar))
		return vec3(0.0);
	float pdfL = 1.0 / (PI2 * (1.0 - bcos));
	float ph = hgPhase(dot(dir, wi), claudeAirG);
	float w = pdfL / (pdfL + ph);
	return w * ph * skyBody(wi) * airTr(farExitT(x, wi)) / pdfL;
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
#ifdef CLAUDE_SPLIT_OUT
	// any early return (debug views, the no-grid passthrough) keeps the
	// direct history as it was rather than leaving attachment 1 undefined
	outDirect = texture2D(historyDirect, varTexCoord.st);
	outGbuf = texture2D(historyGbuf, varTexCoord.st);
	outMom = texture2D(historyMom, varTexCoord.st);
#endif
	vec2 uv = varTexCoord.st;
	if (gridDebug < 2.5) {
		// traced mode off: carry history through untouched
		gl_FragColor = texture2D(history, uv);
		return;
	}

	int view = int(claudeView + 0.5);
	int maxBounces = int(claudeBounces + 0.5);
#if CLAUDE_SUBPASS > 0
	// WHO GETS AN EXTRA PATH IN THIS PASS. nBefore = the samples the pixel
	// had before this frame (pass 0 and each earlier extra pass added one).
	// TUNED: fewer than 8 -> a second path, fewer than 4 -> a third |
	// learn by: the sampling-map network (DECISIONS 0w), trained on
	// reference-mode walks, judged by claude_playtest's dark-reveal number
	{
#ifdef CLAUDE_SPLIT_OUT
		float nBefore = texture2D(historyDirect, uv).a - float(CLAUDE_SUBPASS);
		bool go = claudeBoost > 0.5 && view == 0
				&& nBefore < (CLAUDE_SUBPASS == 1 ? 8.0 : 4.0);
#else
		bool go = false;
#endif
		if (!go) {
			gl_FragColor = texture2D(history, uv);
			return;
		}
	}
#endif
	// nLights is the ONE gate on the whole rung-2 block. 0 means the pure
	// path: with claudeNee = 0, or an empty/invalid list, not a single
	// line below behaves differently from rung 1 — including the RNG draw
	// order, which is what makes the A/B a real A/B and not two images
	// that merely look alike.
	int nLights = (claudeNee > 0.5 && claudeAreaNee > 0.5)
			? min(int(claudeAreaCount + 0.5), AREA_CAP) : 0;
	// The sky's own light-sampling technique rides the SAME dial and
	// nothing else. It is deliberately NOT gated on nLights: outdoors
	// there are frequently no listed emitters at all, and that is exactly
	// the scene where sampling the sun matters most.
	bool skyNee = claudeNee > 0.5;
	// the flames' own light sampler, under the same dial
	int nPts = (claudeNee > 0.5 && claudeTorchNee > 0.5)
			? min(int(claudePointCount + 0.5), 8) : 0;
	g_ptCtr = 0u;
	bool coneArmed = false;   // did the previous vertex aim at flames?
	bool areaArmed = false;   // ...at the area (lamp) list?

	g_rngState = hash13(vec3(gl_FragCoord.xy,
			// animationTimer is unbounded seconds; wrapped by an
			// irrational multiplier so consecutive frames land far
			// apart in the hash's domain and no frame rate aliases it
			fract(animationTimer * 91.7) * 1024.0));
	// the counter-based key, seeded from the same three facts
	g_rngKey = pcgHash(uint(gl_FragCoord.x)
			^ (uint(gl_FragCoord.y) << 11u)
			^ (uint(fract(animationTimer * 91.7) * 65536.0) << 22u));
	// claudeRng = 2 (2026-10-04): THE FRAME GETS 32 BITS, NOT 10. The key
	// above shifts the frame term left by 22, so only its low 10 bits
	// survive: 1,024 distinct streams per pixel, and frames past ~1,000
	// repeat earlier ones. MEASURED on cornell, two captures per depth:
	// per-image noise 7.00 / 3.41 / 1.89 / 2.13 at N = 256 / 1024 / 4096 /
	// 16384 — it should halve per 4x and instead STOPPED (luanti-docs
	// measured.md 2026-10-04). Mode 2 hashes an exact frame index, which
	// is still_frames while parked, so a parked capture is also the same
	// numbers every run.
	if (claudeRng > 1.5)
		g_rngKey = pcgHash((uint(gl_FragCoord.x)
				^ (uint(gl_FragCoord.y) << 11u))
				^ pcgHash(uint(claudeRngFrame) + 2654435769u));
#if CLAUDE_SUBPASS > 0
	// an independent stream for the extra pass
	g_rngKey = pcgHash(g_rngKey ^ (uint(CLAUDE_SUBPASS) * 2654435769u));
#endif
	g_rngCtr = 0u;
	g_skyCtr = 0u;
	// TUNED: claude_guide_keep 0.25 (a quarter of paths teach the guide) |
	// learn by: equal-time error sweep 1 / 0.5 / 0.25 / 0.125, doorway room
	g_guideKeep = float(pcgHash(g_rngKey ^ 0x5bd1e995u) & 0xFFFFu) < claudeGuideKeep * 65536.0;

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
	if ((view >= 1 && view <= 5) || view == 7 || view == 8 || (view >= 33 && view <= 39))
		jit = vec2(0.0);

	vec2 ndc = (uv + jit) * 2.0 - 1.0;
	vec3 rd = normalize(gridCamFwd
			+ ndc.x * gridCamRight + ndc.y * gridCamUp);
	// +0.5: gridCamPos is node-CENTRED (game.cpp: cam/BS - origin),
	// the DDA works in cell-index space where cell i spans [i, i+1)
	vec3 ro = gridCamPos + 0.5;

	// --- the path ----------------------------------------------------
	vec3 L = vec3(0.0);
	vec3 Ld = vec3(0.0);   // the direct part of L (see historyDirect)
	float nScat = 0.0;     // diffuse + air scatters so far (not interfaces)
	vec3 tp = vec3(1.0);
	vec3 p = ro;
	vec3 dir = rd;
	// THE MEDIUM THE PATH IS CURRENTLY INSIDE (2026-08-18), as a material
	// index. 0 is air, which is where every camera in this world starts.
	//
	// It is a single number and not a STACK, and that is exact rather than
	// a simplification: the walk hands back the index of the cell it
	// arrived at, so "what am I in now" is read off the geometry at every
	// interface instead of being remembered across them. A pane inside a
	// tank inside a pane needs no bookkeeping — each crossing names its
	// own new medium.
	float curMed = 0.0;

	float primaryT = DEPTH_MISS; // for the depth channel + view 4
	vec3 primaryHp = vec3(0.0);  // views 33/34: the walk's own hit point (not ro + rd * t)
	vec3 primaryN = vec3(0.0);   // view 1
	vec3 primaryAlb = vec3(0.0); // view 2
	bool primaryClear = false;   // the camera ray's first hit is glass/water
	// THE DENOISER'S GUIDE SURFACE (2026-10-06): the first non-transmissive
	// surface the camera ray reaches through any number of glass/water
	// crossings -- the primary hit itself for an opaque first hit, the
	// surface BEHIND the window for a window. Recorded only by samples that
	// got there by transmitting (a sample that reflects off the glass, or
	// scatters in the air first, records nothing this frame and the pixel
	// keeps its previous guide), so a window is not "two surfaces".
	bool guideOn = true;     // still on the camera's specular-transmit chain
	bool guideSet = false;
	bool guideSky = false;   // the chain escaped to the sky
	vec3 guideN = vec3(0.0);
	vec3 guideP = vec3(0.0);
	vec3 guideAlb = vec3(1.0);
	vec3 guideLe = vec3(0.0);
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

	// THE PRICE OF EACH WALK (2026-10-07): one camera ray per pixel through
	// ONE walk only, BEFORE the path loop (first try sat after it and
	// timed a whole frame), so the trace pass time is that walk's time. 36 = the
	// tree, 37 = today's walk, 38 = no walk (the pass's fixed cost).
	if (view == 36 || view == 37 || view == 38) {
		float tt = -1.0;
		if (claudeTreeDirs > 0.5) {
			// a fixed random direction per pixel, the same for both walks:
			// long rays (sky, sun, outdoor bounces), the tree's home ground
			uvec2 q = uvec2(gl_FragCoord.xy);
			uint h1 = (q.x * 1973u + q.y * 9277u + 26699u) | 1u;
			h1 ^= h1 >> 16; h1 *= 0x7feb352du; h1 ^= h1 >> 15; h1 *= 0x846ca68bu; h1 ^= h1 >> 16;
			uint h2 = h1 * 0x9e3779b9u + 0x632be5abu;
			h2 ^= h2 >> 16; h2 *= 0x7feb352du; h2 ^= h2 >> 15; h2 *= 0x846ca68bu; h2 ^= h2 >> 16;
			float zz = 1.0 - 2.0 * (float(h1 >> 8) / 16777216.0);
			float ph = 6.2831853 * (float(h2 >> 8) / 16777216.0);
			float rr = sqrt(max(0.0, 1.0 - zz * zz));
			rd = vec3(rr * cos(ph), zz, rr * sin(ph));
		}
		if (view == 37) {
			vec3 a1, a2, a3, a4, a5;
			float tM;
			if (march(ro, rd, a1, a2, a3, a4, tM, a5))
				tt = tM;
		}
#ifdef CLAUDE_TREE_OK
		if (view == 36) {
			ivec3 hb;
			int ha;
			float tb;
			if (claudeTreeVariant > 1.5 ? treeWalk2(ro * 16.0, rd, hb, ha, tb) : treeWalk(ro * 16.0, rd, hb, ha, tb))
				tt = tb / 16.0;
		}
#endif
#ifdef CLAUDE_TREE_OK
		// steps per ray, summed on the GPU (slots after the gate's records)
		if (view != 38) {
			atomicAdd(claudeTreeCount[2441], uint(view == 36 ? float(g_treeIters) : g_steps));
			atomicAdd(claudeTreeCount[2442], 1u);
			atomicAdd(claudeTreeCount[2443], uint(g_treeLoads));
		}
#endif
		gl_FragColor = vec4(vec3(tt < 0.0 ? 0.0 : 1.0 / (1.0 + tt)), 1.0);
		return;
	}
	// --- INSTRUMENT: claude_view 7 / 8, THE SUB-VOXEL SLIVER ----------
	// Bright specks inside the cabin's solid walls, 2026-08-17. There
	// were three ways that can happen and they need separating BEFORE
	// anything is fixed: the mask says air where the node is solid (the
	// BAKE), the walk steps over a solid sub-voxel (the TRAVERSAL), or
	// the walk reads the right mask at the wrong address (an OFFSET
	// READ, which shows as geometry drawn at a displaced position -- too
	// much of it in one place and too little in another). These two
	// views answer that, they touch nothing in march(), and they cost
	// nothing when they are not selected.
	//
	//   view 7  MASK DUMP. The whole 16^3 mask of ONE cell, as a 4x4
	//           sheet of 16x16 slices (tile = z, within a tile x is
	//           right and y is up), read through subvoxSolid() -- the
	//           same function the walk uses. Diff it against the model
	//           JSON the bake wrote and an offset read cannot survive:
	//           it displaces the whole pattern. The cell is the CAMERA's
	//           own cell + (3,-1,1), which at the leak-sliver vantage is
	//           the east wall the specks are in. VERDICT 2026-08-17:
	//           bit-identical to planks_spruce_baked.json, all 4096.
	//
	//   view 8  A SECOND OPINION on the traversal, by a DIFFERENT
	//           algorithm. march() is a DDA; this is a dumb point
	//           sampler at 1/32 m -- half a sub-voxel, so it cannot step
	//           over one -- asking the same texture the same question.
	//           R = it found a surface march() walked PAST
	//           G = how far past, in metres / 16
	//           B = march() hit where the sampler found nothing
	//           VERDICT 2026-08-17: R was 0 on every pixel of the frame,
	//           before the fix as well as after.
	//
	// With both of those negative the answer was the BAKE, and it was:
	// the masks themselves carried tunnels that cross a node seam. See
	// util/claude_models.py, "the bake floor".
	//
	// Both present LINEARLY (claude_present lists 7 and 8 beside 1-5):
	// these are encoded values, and ACES would bend them.
	// VIEW 19 — THE MATERIAL INDEX ROUND TRIP, drawn.
	//
	// The per-cell byte stopped being a set of bands on 2026-08-18 and
	// became an INDEX. A band tolerated a drift of +/-2 by construction
	// (its edges were placed at half-byte midpoints for exactly that
	// reason); an index tolerates none — one LSB is a different material.
	// So the exactness gets an instrument rather than an assumption.
	//
	// claudeMatProbe is 256x1x1 RGBA8 with the SAME internal format and
	// the same NEAREST/CLAMP parameters as the trace grid, and texel n
	// carries the byte n. This view screens it: the frame is split into
	// 256 vertical columns, column i reads probe texel i and runs the
	// walk's own matIndex() over it.
	//
	//   WHITE column = matIndex() returned exactly i
	//   BLACK column = it did not, and no frame should be believed
	//
	// Brightness, not hue: John is colourblind, and a red/green pass-fail
	// would be unreadable to him. The bottom eighth of the frame draws
	// the palette's emission column as a brightness ramp, so a palette
	// that uploaded as zeros is visible in the same picture.
	//
	// util/claude_matpal_roundtrip.py scores it threshold-free, and
	// game.cpp runs the same 256 values through GL at startup and reports
	// `matpal_roundtrip` in claude_stats.json, which CI asserts. Two
	// independent readings of one claim, because an instrument that only
	// exercises the values this world happens to contain is the blind
	// kind.
	if (view == 19) {
		float col = floor(uv.x * 256.0);
		float a = texture3D(claudeMatProbe,
				vec3((col + 0.5) / 256.0, 0.5, 0.5)).a;
		float ok = (abs(matIndex(a) - col) < 0.5) ? 1.0 : 0.0;
		if (uv.y < 0.125) {
			// the palette's own emission column, 0..2.4 mapped to 0..1
			gl_FragColor = vec4(vec3(matPal(a).r / 2.4), 1.0);
			return;
		}
		gl_FragColor = vec4(vec3(ok), 1.0);
		return;
	}
	// --- INSTRUMENT: claude_view 20, THE INTERFACE LADDER -------------
	//
	// WHAT IT IS ABOUT. Transparency in this renderer is one decision
	// taken at one surface: how much of the ray bounces off and which way
	// the rest of it bends. Both are closed-form functions of the
	// incidence angle and the two indices of refraction, and both are
	// invisible in a picture — a wrong Fresnel term or a wrong IOR still
	// draws plausible glass. So they get an instrument that does not need
	// a scene at all.
	//
	// This view draws the SHADER'S OWN fresnelDielectric() and the
	// SHADER'S OWN refract() across every incidence angle, as brightness,
	// in four horizontal bands. x is cos(theta_i), 0 at the left edge
	// (grazing) to 1 at the right (normal incidence).
	//
	//   band 0 (top)     R, air -> glass    eta = 1 / 1.52
	//   band 1           R, glass -> air    eta = 1.52  (TIR at the left)
	//   band 2           sin(theta_t), air -> glass
	//   band 3 (bottom)  sin(theta_t), glass -> air, 0 inside TIR
	//
	// Bands 0 and 1 are the ENERGY claim: R is what decides the split, and
	// T is 1 - R by construction in the path loop, so scoring R against
	// the closed form is scoring the split. Bands 2 and 3 are the
	// GEOMETRY claim, i.e. Snell's law and the IOR that went into it, and
	// they are the reason "state the IOR you chose" is answerable by
	// measurement rather than by reading the source.
	//
	// util/claude_fresnel_check.py screens it against the same four
	// functions computed off-GPU in double precision. It is a real
	// external reference and not a tautology: this file's Fresnel is the
	// exact unpolarised expression, so a Schlick approximation, a swapped
	// eta or a wrong IOR all show up as a curve that misses.
	//
	// Brightness, not hue (John is colourblind), and it presents LINEARLY
	// (claude_present) — the values ARE the message and ACES would bend
	// them.
	if (view == 20) {
		// THE BOTTOM EIGHTH IS THE PALETTE'S OWN IOR COLUMN, and it is
		// what stops the four bands above from being an instrument that
		// measures a constant. They are drawn at IOR_LADDER, a constant
		// in this file, so they say nothing about what game.cpp actually
		// uploaded -- and a palette of zeros would leave every real
		// interface at IOR 0 while the ladder still read perfect.
		//
		// 256 columns, column i painted (ior/2) where the material's
		// transmission column says it is a dielectric, and BLACK where it
		// does not. So the strip is dark almost everywhere and lights up
		// exactly on the glass and liquid rows: brightness 0.76 for
		// glass's 1.52 and 0.667 for water's 1.333, which are 24 display
		// steps apart and cannot be confused for one another.
		if (uv.y < 0.125) {
			float col = floor(uv.x * 256.0);
			vec4 mp = matPalIdx(col);
			float v = (mp.b > 0.5) ? (mp.a * 0.5) : 0.0;
			gl_FragColor = vec4(vec3(clamp(v, 0.0, 1.0)), 1.0);
			return;
		}
		float ci = clamp(uv.x, 0.001, 1.0);   // cos(theta_i)
		// the four bands live in the TOP seven eighths now
		float band = min(floor(((1.0 - uv.y) / 0.875) * 4.0), 3.0);
		float eta = (band == 0.0 || band == 2.0)
				? (1.0 / IOR_LADDER) : IOR_LADDER;
		float v;
		if (band < 1.5) {
			v = fresnelDielectric(ci, eta);
		} else {
			// the shader's own refract(), fed a canonical geometry: the
			// facing normal is +z and the incident ray comes down onto it
			// at the same cos. sin(theta_t) is read straight off the
			// returned vector, so a bend the path loop would take is the
			// bend this band draws.
			float si = sqrt(max(0.0, 1.0 - ci * ci));
			vec3 idir = vec3(si, 0.0, -ci);
			vec3 nrm = vec3(0.0, 0.0, 1.0);
			vec3 wt = refract(idir, nrm, eta);
			v = (dot(wt, wt) < 1e-8) ? 0.0 : length(wt.xy);
		}
		gl_FragColor = vec4(vec3(clamp(v, 0.0, 1.0)), 1.0);
		return;
	}
	if (view == 7) {
		vec3 cellD = floor(ro) + vec3(3.0, -1.0, 1.0);
		vec2 tile = floor(uv * 4.0);
		vec2 loc = floor(fract(uv * 4.0) * 16.0);
		float sz = (3.0 - tile.y) * 4.0 + tile.x;
		bool bit = inSubvoxRing(cellD)
				&& subvoxSolid(cellD - vec3(SUBV_R0),
						vec3(loc.x, loc.y, sz));
		gl_FragColor = vec4(vec3(bit ? 1.0 : 0.0), 1.0);
		return;
	}
	if (view == 8) {
		vec3 hp, n, alb, le, cell;
		float tHit;
		bool h = march(ro, rd, hp, n, alb, le, tHit, cell);
		float tEnd = h ? tHit : 8.0;
		float tSlow = -1.0;
		for (int k = 2; k < 256; k++) {
			float ts = float(k) / 32.0;
			if (ts >= tEnd) break;
			vec3 pw = ro + rd * ts;
			vec3 cw = floor(pw);
			if (any(lessThan(cw, vec3(0.0)))
					|| any(greaterThanEqual(cw, vec3(GRID_S))))
				break;
			vec4 sw = texture3D(claudeTraceGrid, (cw + 0.5) / GRID_S);
			if (sw.a <= MAT_AIR_MAX)
				continue;
			if (claudeDescend > 0.5 && matFine(matPal(sw.a))
					&& inSubvoxRing(cw)) {
				if (!subvoxSolid(cw - vec3(SUBV_R0),
						floor((pw - cw) * SUBV)))
					continue;
			}
			tSlow = ts;
			break;
		}
		float missed = (tSlow >= 0.0 && (!h || tSlow < tHit - 0.05))
				? 1.0 : 0.0;
		gl_FragColor = vec4(missed,
				missed * clamp((tEnd - tSlow) / 16.0, 0.0, 1.0),
				(h && tSlow < 0.0) ? 1.0 : 0.0,
				h ? min(tHit, DEPTH_MAX_HIT) / DEPTH_SCALE : 1.0);
		return;
	}

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
	// WHAT THIS INSTRUMENT IS BLIND TO: a FINE-MATERIAL cell is "solid" to
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
				if (s0.a > MAT_AIR_MAX)
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
					L = neeDirect(hp, n, alb, nInstr, 0.0, 0.0);
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

	float wB = 0.0, wGain = 0.0, wLprev = 0.0;
	// guide tallies waiting for their light: each bounce is credited with
	// what the next two vertices bring back (TUNED: 2 | learn by: 1/2/3
	// in the same error-at-equal-time sweep)
	int gpTab0 = -1, gpTab1 = -1, gpBin0 = 0, gpBin1 = 0;
	float gpTp0 = 0.0, gpTp1 = 0.0, gpL0 = 0.0, gpL1 = 0.0, gpF0 = 1.0, gpF1 = 1.0;
	for (int seg = 0; seg <= BOUNCE_CAP; seg++) {
		{
			float ln = L.r + L.g + L.b;
			if (ln > wLprev) { wGain = wB; wLprev = ln; }
		}
		if (seg > maxBounces)
			break;

		vec3 hp, n, alb, le, cell;
		float tHit;
		vec4 hitPal;
		float hitIdx;
		vec3 hpFar;
		bool hitS = marchAll(p, dir, curMed, hp, n, alb, le, tHit, cell,
				hitPal, hitIdx, hpFar);
		bool hitFar = g_lastFar;
		bool hitFarMed = hitFar && g_farMedium;
		// THE PIXEL'S SURFACE IS THE CAMERA RAY'S FIRST HIT, WHETHER OR NOT
		// THIS SAMPLE REACHES IT (2026-10-06). In air a camera ray may
		// scatter before the surface (the block below `continue`s), and the
		// primary record used to be written only further down, so in fog a
		// third of the samples recorded "no surface": every pixel's face
		// code flipped between frames, every pixel was MIXED, and the
		// denoiser had no neighbours anywhere (fog-dawn A/B: x1.00). The
		// geometry is the same for every sample of the pixel, so record it
		// here; the face-tile albedo is the one the surface block applies.
		if (seg == 0 && hitS) {
			primaryHit = true;
			primaryT = tHit;
			primaryN = n;
			primaryLe = le;
			primaryClear = hitIdx > 0.5 && matTransmits(hitIdx, hitPal);
			vec3 a0 = alb;
			if (claudeTexel > 0.5 && hitIdx > 0.5 && !matFine(hitPal)
					&& !matTransmits(hitIdx, hitPal))
				a0 = max(min(a0 * pow((cell.x > -5000.0 ? faceTileRatio(cell, hp, n) : vec3(1.0)), vec3(2.2)),
						vec3(1.0)), vec3(ALBEDO_FLOOR));
			primaryAlb = a0;
		}
		if (guideOn && !guideSet && hitS) {
			vec4 mp0 = curMed < 0.5 ? vec4(0.0) : matPalIdx(curMed);
			bool iface = matTransmits(curMed, mp0)
					&& matTransmits(hitIdx, hitPal) && hitIdx > 0.5;
			if (!iface) {
				vec3 g0 = alb;
				if (claudeTexel > 0.5 && hitIdx > 0.5 && !matFine(hitPal)
						&& !matTransmits(hitIdx, hitPal))
					g0 = max(min(g0 * pow((cell.x > -5000.0 ? faceTileRatio(cell, hp, n) : vec3(1.0)),
							vec3(2.2)), vec3(1.0)), vec3(ALBEDO_FLOOR));
				guideSet = true;
				guideN = n;
				guideP = p + dir * tHit;
				guideAlb = g0;
				guideLe = le;
			}
		}
		if (guideOn && !guideSet && !hitS)
			guideSky = true;
		// AIR (2026-10-04). On a segment travelled in air, sample where the
		// ray would next interact with the air: s ~ sigma_t exp(-sigma_t s).
		// Before the surface (or the grid's edge) it scatters or is
		// absorbed there, with weight sigma_s / sigma_t; otherwise the
		// segment is untouched (transmittance over its own probability is
		// exactly 1). That is the whole estimator, and it is why no energy
		// can be invented: a non-absorbing medium cannot move a sealed
		// furnace (the furnace-050-air arm). Debug views see geometry only.
		if (airSigT() > 0.0 && curMed < 0.5 && !(view >= 1 && view <= 5)) {
			float tSeg = hitS ? tHit : farExitT(p, dir);
			float ua = rnd1();
			float sAir = -log(max(1.0 - ua, 1e-12)) / airSigT();
			if (sAir < tSeg) {
				vec3 xa = p + dir * sAir;
				tp *= claudeAirScatter / airSigT();
				if (seg == maxBounces)
					break;
				// the lights, aimed at from the air point
				if (nLights > 0) {
					vec3 cA = tp * neeDirectAir(xa, dir, nLights);
					L += cA;
					if (nScat < 0.5)
						Ld += cA;
				}
				if (skyNee) {
					vec3 cB = tp * neeSkyAir(xa, dir);
					L += cB;
					if (nScat < 0.5)
						Ld += cB;
				}
				if (seg + 1 >= RR_START) {
					float q = clamp(max(tp.r, max(tp.g, tp.b)),
							RR_Q_MIN, RR_Q_MAX);
					if (rnd1() > q)
						break;
					tp /= q;
				}
				float v1 = rnd1();
				float v2 = rnd1();
				vec3 nd = hgSample(dir, claudeAirG, v1, v2);
				prevPdfB = hgPhase(dot(dir, nd), claudeAirG);
				prevX = xa;
				misArmed = nLights > 0 || skyNee;
				coneArmed = false;   // air vertices do not aim at flames
				areaArmed = nLights > 0;
				guideOn = false;     // off the camera's transmit chain
				dir = nd;
				p = xa;
				pathBounces += 1.0;
				nScat += 1.0;
				continue;
			}
		}
		// INSIDE A MEDIUM (water, 2026-10-05): Beer-Lambert over the
		// segment, applied before anything at its far end is added.
		// Deterministic, not sampled: with no scattering in the medium
		// the transmittance is the whole answer.
		if (curMed > 0.5 && !(view >= 1 && view <= 5))
			tp *= medTr(curMed, hitS ? tHit : airExitT(p, dir));
		if (!hitS) {
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
			vec3 cSky = tp * (skyDome(dir) + misW * skyBody(dir));
			L += cSky;
			if (nScat <= 1.0)
				Ld += cSky;
			break;
		}

		// THE FACE TILE (claudeTexel, 2026-10-04): a plain cube's albedo
		// takes the pattern of the face tile under the hit point. HERE and
		// not in marchMed(), which also serves every shadow ray — measured:
		// inside the walk it cost +0.2-0.3 ms of trace pass on rooms whose
		// picture it could not change. Le was taken from the cell colour
		// inside the walk and is not touched: texture is reflectance, never
		// emission (§4's domain is not this step's to move). Not a fine
		// cell (carved models keep one colour until their delit palettes
		// are wired — bf83f1108), not air, not glass. cellAlbedo is
		// pow(c, 2.2), so the sRGB ratio enters as ratio^2.2.
		if (claudeTexel > 0.5 && hitIdx > 0.5 && !matFine(hitPal)
				&& !matTransmits(hitIdx, hitPal))
			alb = max(min(alb * pow((cell.x > -5000.0 ? faceTileRatio(cell, hp, n) : vec3(1.0)), vec3(2.2)),
					vec3(1.0)), vec3(ALBEDO_FLOOR));

		// clay: march computed le from the TRUE albedo above; clamping
		// rho afterward changes reflectance only, never the lights
		if (view == 6)
			alb = CLAY_RHO;

		if (seg == 0) {
			primaryHit = true;
			primaryT = tHit;
			primaryHp = hp;
			primaryN = n;
			primaryAlb = alb;
			primaryLe = le;
			primaryClear = hitIdx > 0.5 && matTransmits(hitIdx, hitPal);
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
			float pdfL = areaArmed
					? neePdfSa(cell, n, prevX, tHit, cosY, nLights) : 0.0;
			// a flame hit: the flame sampler could have chosen this too
			if (coneArmed && matFine(hitPal))
				pdfL += ptPdfSa(prevX, dir, nPts);
			// A zero denominator means neither strategy claims a density
			// for this direction, which can only happen at a degenerate
			// cos; fall back to 1 rather than let a NaN into the history.
			float denom = prevPdfB + pdfL;
			misW = denom > 0.0 ? prevPdfB / denom : 1.0;
		}
		L += tp * misW * le;
		if (nScat <= 1.0)
			Ld += tp * misW * le;

		if (seg == maxBounces)
			break; // depth cap: no scatter from this vertex
		if (view >= 1 && view <= 4)
			break; // first-hit views need nothing past the primary

		// =============================================================
		// A REFRACTIVE INTERFACE (2026-08-18) — glass and water
		// =============================================================
		// This is the whole of transparency in this renderer, and the
		// shape of it is the reason the handoff said this step should be
		// EASIER than the rasterizer it replaces: the ray crosses the
		// surface and carries on. There is no depth sort, no
		// order-dependent blend, no OIT and no second pass, because
		// nothing here composites — the transport just continues.
		//
		// WHEN IT FIRES. Both sides of the boundary have to be things a
		// ray can travel in. `curMed` is what the ray is in now and
		// `hitIdx` is the material it just arrived at; air counts as
		// transmissive with IOR 1. Air -> glass and glass -> air both
		// fire; glass -> stone does not, and neither does air -> stone,
		// so an opaque scene never reaches this block at all.
		//
		// `curMed` IS UPDATED AT EVERY INTERFACE, and since PANES
		// (2026-08-23) "every interface" includes the ones that are not
		// here. The walk reports an interface wherever the material
		// CHANGES along the ray — at a 1 m cell wall, and now also at a
		// fine cell's mask boundary — and a ray leaving a medium always
		// crosses one of those before it can reach anything opaque. That
		// is what makes the `curMed = hitIdx` below sufficient: a
		// glass -> opaque hit reaches this vertex with the ray genuinely
		// still in glass (its restart `hp` is in the glass it came
		// through), and a ray that has already left the glass reaches it
		// with curMed = 0, because the glass -> air crossing was itself
		// an interface and fired this block.
		//
		// THAT IS THE FIX FOR measured.md "Defect 2 — glass X-rays the
		// opaque block behind it", and the line it lives on is in
		// marchMed()'s FINE RUNG, not here. Before PANES a ray refracted
		// into glass and then descending into an opaque fine cell — a
		// carved plank — walked that cell's air sub-voxels with
		// `if (!subvoxSolid(...)) continue;`, which does not compare
		// anything to curMed, so it crossed out of the glass without an
		// interface. curMed stayed at the glass index for the rest of the
		// path: the plank's shadow rays then read the first AIR cell as a
		// boundary and returned "occluded", and its bounce rays refracted
		// at that phantom interface, disarmed MIS, and added the sun disc
		// at weight 1 on top of the sky sample already taken — a bright
		// hole where a wall should be. The fine rung now compares the
		// sub-voxel's material to curMed like every other arrival, so the
		// medium ends where the glass ends.
		//
		// IT IS A SPECULAR VERTEX, and that single fact answers the
		// landmine this step was warned about. A delta BSDF has no light
		// sampler to be MIS-partnered with, so:
		//   - no next-event estimation is taken here (the two calls
		//     below are past the `continue`),
		//   - misArmed goes FALSE, so the next emitter this path lands
		//     on is added at weight 1.
		// That is what keeps an emissive face behind glass honest. The
		// area-emitter face mask still drops a face whose neighbour is
		// non-air, so the light sampler's density for such a face is 0;
		// the BSDF technique's weight for it is 1. Both halves of the
		// balance heuristic agree, which is exactly what §6 asks and what
		// the 1a failure was.
		//
		// THE THROUGHPUT IS NOT TOUCHED. Reflect with probability R
		// carrying weight R/R, refract with probability 1-R carrying
		// weight (1-R)/(1-R). Energy is conserved by CONSTRUCTION, which
		// is why the gate for it is transport (a lossless slab must not
		// move a sealed furnace) and arithmetic (claude_view 20 against
		// the closed-form Fresnel), never a check that R + T = 1 — that
		// one cannot fail here and would be a blind instrument.
		//
		// ONE PALETTE FETCH FOR THE MEDIUM, and it is skipped entirely
		// while the path is in air: matPalIdx(0) is never sampled, because
		// matTransmits()/matIor() answer for index 0 without a lookup. So
		// a path that never enters anything pays nothing for this block
		// beyond one compare per hit.
		vec4 medPal = curMed < 0.5 ? vec4(0.0) : matPalIdx(curMed);
		if (matTransmits(curMed, medPal) && matTransmits(hitIdx, hitPal)) {
			float eta = matIor(curMed, medPal) / matIor(hitIdx, hitPal);
			// march() guarantees n opposes the ray, so it is already the
			// facing normal and cosI is a plain dot.
			float cosI = clamp(dot(-dir, n), 0.0, 1.0);
			float R = fresnelDielectric(cosI, eta);
			vec3 wt = refract(dir, n, eta);
			// Total internal reflection: R is exactly 1 above the
			// critical angle and refract() returns the zero vector. The
			// length test is belt and braces on the same event, stated
			// rather than relied upon.
			if (dot(wt, wt) < 1e-8)
				R = 1.0;
			// THE ONE EXTRA RANDOM DRAW IN THIS FILE, and it is taken
			// only at an interface. An opaque scene contains none, so its
			// per-pixel random chain is the sequence it always was and
			// its goldens are comparable — which is gate 1.
			float ur = rnd1();
			if (ur < R) {
				dir = reflect(dir, n);
				guideOn = false;  // a reflection is not the window's view
				p = hp;           // the restart march() already clamped
				                  // into the cell the ray came THROUGH
			} else {
				dir = wt;
				// THE FAR SIDE. march()'s own restart point `hp` sits in
				// the region the ray came from, which is the wrong side
				// for a ray that is crossing: starting there would step
				// straight back into this same face and the path would
				// hammer the interface forever. `hpFar` is the same
				// construction the other way — clamped into the region
				// that was ENTERED, so the walk's "never test the cell
				// you start in" rule names the region the ray is
				// genuinely inside.
				//
				// IT IS COMPUTED BY THE WALK, NOT HERE (PANES,
				// 2026-08-23). This line used to read
				// `restartPoint(phit, -n, cell, 1.0)` and take `cell`,
				// the COARSE cell, on the grounds that a transmissive
				// material was never a fine one — game.cpp asserted that
				// pairing could not exist. A glass pane is exactly that
				// pairing, and its interfaces are 1/16 m apart, so a
				// restart clamped into the metre cell would drop a ray
				// entering the near face of a pane out the far side of
				// the whole cell. Only marchMed() knows which rung the
				// interface was found on, so marchMed() hands the point
				// back. At the coarse rung it is the identical
				// expression on identical inputs.
				p = hpFar;
				curMed = hitIdx;
			}
			misArmed = false;     // a delta lobe has no light-sampling
			coneArmed = false;
			areaArmed = false;
			                      // partner: the next Le arrives at
			                      // weight 1
			pathBounces += 1.0;
			continue;
		}

		// NEXT-EVENT ESTIMATION at this vertex, before the throughput
		// absorbs alb: neeDirect() carries its own rho/PI, and it must be
		// the rho this vertex actually reflects with — which in clay
		// (view 6) is CLAY_RHO, clamped above. The depth cap cuts this
		// term at the same vertex it cuts the BSDF half, so claudeBounces
		// means the same thing under either dial.
		// LEAVES TRANSMIT (see claudeLeafTransmit). A fine leaf voxel
		// scatters to BOTH sides: half the paths go back out the front,
		// half out through the sheet, each cosine-distributed, so f = rho/pi
		// either side and the density is 0.5 cos/pi. xb is the first air
		// behind the sheet: leaf blocks meet back to back, so a sheet may be
		// two voxels thick, and the second one passes light only through
		// its own colour once more (its inter-reflection is dropped: it can
		// lose light, never make it).
		bool leafT = false;
		vec3 xb = hp;
		vec3 leafBack = vec3(1.0);
		if (claudeLeafTransmit > 0.5 && hitIdx > 0.5 && view == 0
				&& abs(matHot(hitIdx) - 3.0) < 0.5 && matFine(hitPal)
				&& fineAt(cell)) {
			alb = min(alb, vec3(0.5));
			vec3 x1 = hp - n * (1.5 * RUNG_FINE);
			vec3 x2 = hp - n * (2.5 * RUNG_FINE);
			if (!pointSolid(x1)) {
				leafT = true;
				xb = x1;
			} else if (!pointSolid(x2)) {
				leafT = true;
				xb = x2;
				leafBack = alb;
			}
		}
		// A FAR CLOUD HIT is a leaf or a blade: the same two-sided law, the
		// back side being the far side of the virtual sheet (hpFar)
		if (!leafT && claudeLeafTransmit > 0.5 && hitFarMed && view == 0) {
			alb = min(alb, vec3(0.5));
			leafT = true;
			xb = hpFar;
		}
		g_neeBScale = leafT ? 0.5 : 1.0;
		g_guideTab = -1;
		g_guideT = 0.0;
#ifdef CLAUDE_GUIDE_OK
		// on in the picture (0), its linear readouts (25, 26) and its own
		// instrument (30); off in every other instrument view
		if (claudeGuide > 0.5 && !leafT && !hitFar && curMed < 0.5
				&& (view == 0 || view == 25 || view == 26 || view == 30 || view == 31 || view == 32)) {
			g_guideTab = guideTable(floor(cell + gridOrigin + vec3(0.5)), n);
			g_guideT = guideLoad(g_guideTab, 72);
			g_guideN = n;
			if (seg == 0 && view == 30) {
				g_guideT0 = g_guideT;
				// INSTRUMENT: do this table's row sums add up to its total?
				float rsum = 0.0, bsum = 0.0;
				for (int r = 0; r < 8; r++)
					rsum += guideLoad(g_guideTab, 64 + r);
				for (int i = 0; i < 64; i++)
					bsum += guideLoad(g_guideTab, i);
				// SIGNED, relative to the bins' own sum: total and row sums
				g_guideMis = (g_guideT - bsum) / max(bsum, 1.0);
				g_guideMis2 = (rsum - bsum) / max(bsum, 1.0);
			}
			// BISECTION (claude_guide 2): the same sampler and books, the
			// table ignored -- pdf factor exactly 1
			if (claudeGuide > 1.5)
				g_guideT = 0.0;
		}
#endif
		// a FAR vertex aims at the sun and sky only: the lamp and flame
		// lists are near emitters, and a hillside 300 m out gains nothing
		// from them. The MIS arms below say so (areaArmed / coneArmed),
		// so a BSDF ray from that vertex that does hit a lamp counts in
		// full: no light is lost, it is only found the slow way.
		g_bounceN = n;
		if (!hitFar && nLights > 0) {
			float r0cN = g_rays;
			vec3 cN = tp * neeDirect(hp, n, alb, nLights, 1.0, curMed);
			g_neeCalls += 1.0; g_neeZero += all(equal(cN, vec3(0.0))) ? 1.0 : 0.0; if (g_rays > r0cN) { g_neeCallsT[0] += 1.0; g_neeZeroT[0] += all(equal(cN, vec3(0.0))) ? 1.0 : 0.0; }
			L += cN;
			if (nScat < 0.5)
				Ld += cN;
		}
		// The sky's own light sample, at the same vertex and under the
		// same depth cap, drawing from the RESERVED counter range so the
		// path's own random sequence is untouched (see rndSky()).
		if (skyNee) {
			float r0cS = g_rays;
			vec3 cS = tp * neeSky(hp, n, alb, curMed);
			g_neeCalls += 1.0; g_neeZero += all(equal(cS, vec3(0.0))) ? 1.0 : 0.0; if (g_rays > r0cS) { g_neeCallsT[1] += 1.0; g_neeZeroT[1] += all(equal(cS, vec3(0.0))) ? 1.0 : 0.0; }
			L += cS;
			if (nScat < 0.5)
				Ld += cS;
		}
		if (!hitFar && nPts > 0) {
			float r0cP = g_rays;
			vec3 cP = tp * neePoint(hp, n, alb, nPts, curMed);
			g_neeCalls += 1.0; g_neeZero += all(equal(cP, vec3(0.0))) ? 1.0 : 0.0; if (g_rays > r0cP) { g_neeCallsT[2] += 1.0; g_neeZeroT[2] += all(equal(cP, vec3(0.0))) ? 1.0 : 0.0; }
			L += cP;
			if (nScat < 0.5)
				Ld += cP;
		}
		// the same three light samplers from BEHIND the sheet
		if (leafT) {
			vec3 tb = tp * leafBack;
			vec3 cB = vec3(0.0);
			if (nLights > 0)
				cB += neeDirect(xb, -n, alb, nLights, 1.0, curMed);
			if (skyNee)
				cB += neeSky(xb, -n, alb, curMed);
			if (nPts > 0)
				cB += neePoint(xb, -n, alb, nPts, curMed);
			L += tb * cB;
			if (nScat < 0.5)
				Ld += tb * cB;
		}
		g_neeBScale = 1.0;

		// f cos / pdf = 2 rho for either side of a leaf; rho otherwise
		tp *= leafT ? 2.0 * alb : alb;

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
		bool through = leafT && rnd1() < 0.5;
		float gF = 1.0;
		int gBin = 0;
		if (g_guideTab >= 0) {
			dir = guideSample(n, u1, u2, rnd1(), gF, gBin);
			tp /= gF;
			g_guided += 1.0;
			if (view == 32) {
				// INSTRUMENT: the light samplers re-evaluate the bounce pdf
				// for a direction; for the direction just sampled it must
				// be the factor the sampler used
				float fe = guideFactorDir(dir);
				g_guidePair.x += 1.0;
				if (abs(fe - gF) > 1e-4 * gF)
					g_guidePair.y += 1.0;
				g_guidePair.z = max(g_guidePair.z, abs(fe - gF) / gF);
			}
			if (view == 31 && g_guideTest < 0.0) {
				// INSTRUMENT: E[h(dir) / factor] over the guided pdf is the
				// cosine-weighted integral of h whatever the table holds; a
				// lobe at disk point (0.6, -0.3) so it is not flat
				vec3 t1, t2;
				guideFrame(n, t1, t2);
				vec2 dd = vec2(dot(dir, t1), dot(dir, t2)) - vec2(0.6, -0.3);
				g_guideTest = (1.0 + 50.0 * exp(-dot(dd, dd) / 0.02)) / gF;
			}
		} else if (claudeBounceUniform > 0.5) {
			dir = uniformHemisphere(through ? -n : n, u1, u2);
			gF = 0.5 / max(abs(dot(dir, n)), 1e-6);
			tp /= gF;
		} else
			dir = cosineHemisphere(through ? -n : n, u1, u2);
		if (through)
			tp *= leafBack;
		// the guide's books: settle the bounce two back, queue this one
		guideDeposit(gpTab1, gpBin1, gpTp1, gpL1, gpF1, L);
		gpTab1 = gpTab0; gpBin1 = gpBin0; gpTp1 = gpTp0; gpL1 = gpL0; gpF1 = gpF0;
		gpTab0 = g_guideTab; gpBin0 = gBin; gpTp0 = guideLum(tp); gpL0 = guideLum(L); gpF0 = gF;
		// Arm the BSDF half of the MIS pair. Russian roulette above does
		// not enter these pdfs: it scales the estimate by 1/q on the
		// survivors, which leaves the SAMPLING DENSITY of the direction
		// untouched, and the weights are densities.
		prevX = through ? xb : hp;
		prevPdfB = (leafT ? 0.5 : 1.0) * gF * abs(dot(n, dir)) / PI;
		g_guideTab = -1;
		// Armed when ANY light sampler ran at this vertex: the area list,
		// the sky, or both. Outdoors the list is often empty and the sky
		// is the only light there is — leaving this at `nLights > 0`
		// would have given the sun's BSDF half a weight of 1 while the
		// sky sampler was also paying it, i.e. the double count.
		misArmed = (!hitFar && (nLights > 0 || nPts > 0)) || skyNee;
		coneArmed = !hitFar && nPts > 0;
		areaArmed = !hitFar && nLights > 0;
		p = through ? xb : hp;
		pathBounces += 1.0;
		nScat += 1.0;
		wB += 1.0;
	}
	guideDeposit(gpTab0, gpBin0, gpTp0, gpL0, gpF0, L);
	guideDeposit(gpTab1, gpBin1, gpTp1, gpL1, gpF1, L);
	{
		float ln = L.r + L.g + L.b;
		if (ln > wLprev) wGain = wB;
		g_bounces += wB;
		g_bounceWasted += wB - wGain;
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

	// ML DATA EXPORT (2026-10-06, John: "AABB voxel stuff plus machine
	// learning can do something crazy"; the first test asks how much of the
	// BOUNCED light at a face its block neighbourhood predicts).
	//   view 22: which grid cell and face the camera sees, EXACT in 24 bits:
	//            R = x | fx<<7, G = y | fy<<7, B = z | fz<<7, face = fz fy fx
	//            (0..5 = -x +x -y +y -z +z); all 255 = no grid surface.
	//   view 23: the bounced light at that surface, demodulated:
	//            (L - Ld) / albedo, accumulated like a photo (Ld holds all the
	//            direct light at the first surface, its own glow included).
	// LADDER PLAN STAGE 1b (2026-10-07): the camera ray's FIRST hit through
	// the exact pixel centre (no jitter), to check a tree walk built offline
	// against this walk pixel by pixel.
	//   view 33: the 1 m cell and face, coded as view 22 (255 = none)
	//   view 34: the 1/16 piece inside that cell: R = sx*16 + sy, G = sz,
	//            B = 1 where a hit exists
	// LADDER PLAN STAGE 2 GATE (2026-10-07): today's walk and the tree walk
	// on the same camera ray, in the same shader, one colour per verdict:
	// green = same cell, face and distance; red = different 1 m cell;
	// yellow = same cell, different face; blue = same cell and face,
	// distance differs; magenta = only today's walk hit; cyan = only the
	// tree hit; black = neither (sky, far field).
	// LADDER B2 GATE (claude_view 39, run with claude_bricks 1): today's
	// walk on the same camera ray twice, reading its 1/16 m shapes from
	// the ring and atlas, then from the pool. Counted on the GPU like 35.
	if (view == 39) {
		vec3 hpA, nA, albA, leA, cellA, hpB, nB, albB, leB, cellB;
		float tA, tB;
		g_bricksOff = true;
		bool hA = march(ro, rd, hpA, nA, albA, leA, tA, cellA);
		g_bricksOff = false;
		bool hB = march(ro, rd, hpB, nB, albB, leB, tB, cellB);
		int vi;
		vec3 verdict;
		if (!hA && !hB) {
			vi = 0; verdict = vec3(0.0);
		} else if (hA != hB) {
			vi = hA ? 5 : 6; verdict = hA ? vec3(1.0, 0.0, 1.0) : vec3(0.0, 1.0, 1.0);
		} else if (any(notEqual(cellA, cellB))) {
			vi = 2; verdict = vec3(1.0, 0.0, 0.0);
		} else if (any(notEqual(nA, nB))) {
			vi = 3; verdict = vec3(1.0, 1.0, 0.0);
		} else if (tA != tB) {
			vi = 4; verdict = vec3(0.0, 0.0, 1.0);
		} else {
			vi = 1; verdict = vec3(0.0, 1.0, 0.0);
		}
#ifdef CLAUDE_TREE_OK
		atomicAdd(claudeTreeCount[vi], 1u);
#endif
		gl_FragColor = vec4(verdict, 1.0);
		return;
	}
	if (view == 35) {
		vec3 verdict = vec3(1.0, 1.0, 1.0);   // white: tree unavailable
#ifdef CLAUDE_TREE_OK
		vec3 hpM, nM, albM, leM, cellM;
		float tM;
		bool hm = march(ro, rd, hpM, nM, albM, leM, tM, cellM);
		hm = hm && all(greaterThanEqual(cellM, vec3(0.0))) && all(lessThan(cellM, vec3(GRID_S)));
		ivec3 hb;
		int ha;
		float tb;
		vec3 roT = ro * 16.0 + (claudeTreePlant > 0.5 ? vec3(1.0 / 64.0, 0.0, 0.0) : vec3(0.0));
		bool ht = claudeTreeVariant > 1.5 ? treeWalk2(roT, rd, hb, ha, tb) : treeWalk(roT, rd, hb, ha, tb);
		int vi;
		vec3 cellT = floor(vec3(hb) / 16.0);
		if (!hm && !ht) {
			verdict = vec3(0.0); vi = 0;
		} else if (hm && !ht) {
			verdict = vec3(1.0, 0.0, 1.0); vi = 5;
		} else if (!hm && ht) {
			verdict = vec3(0.0, 1.0, 1.0); vi = 6;
		} else {
			vec3 nT = vec3(0.0);
			if (ha >= 0)
				nT[ha] = -sign(rd[ha]);
			if (any(notEqual(cellT, floor(cellM)))) {
				verdict = vec3(1.0, 0.0, 0.0); vi = 2;
			} else if (any(greaterThan(abs(nT - nM), vec3(0.5)))) {
				verdict = vec3(1.0, 1.0, 0.0); vi = 3;
			} else if (abs(tb / 16.0 - tM) > 2e-3) {
				verdict = vec3(0.0, 0.0, 1.0); vi = 4;
			} else {
				verdict = vec3(0.0, 1.0, 0.0); vi = 1;
			}
		}
		atomicAdd(claudeTreeCount[vi], 1u);
		if (vi >= 2) {
			uint k = atomicAdd(claudeTreeCount[8], 1u);
			if (k < 128u) {
				uint b = 9u + 19u * k;
				int aM = abs(nM.x) > 0.5 ? 0 : abs(nM.y) > 0.5 ? 1 : 2;
				claudeTreeCount[b + 0u] = uint(gl_FragCoord.x);
				claudeTreeCount[b + 1u] = uint(gl_FragCoord.y);
				claudeTreeCount[b + 2u] = uint(vi);
				claudeTreeCount[b + 3u] = uint(hm ? cellM.x : -1.0);
				claudeTreeCount[b + 4u] = uint(hm ? cellM.y : -1.0);
				claudeTreeCount[b + 5u] = uint(hm ? cellM.z : -1.0);
				claudeTreeCount[b + 6u] = uint(hm ? aM : 9);
				claudeTreeCount[b + 7u] = floatBitsToUint(hm ? tM : -1.0);
				claudeTreeCount[b + 8u] = uint(hb.x);
				claudeTreeCount[b + 9u] = uint(hb.y);
				claudeTreeCount[b + 10u] = uint(hb.z);
				claudeTreeCount[b + 11u] = uint(ht ? ha : 9);
				claudeTreeCount[b + 12u] = floatBitsToUint(ht ? tb / 16.0 : -1.0);
				claudeTreeCount[b + 13u] = floatBitsToUint(rd.x);
				claudeTreeCount[b + 14u] = floatBitsToUint(rd.y);
				claudeTreeCount[b + 15u] = floatBitsToUint(rd.z);
				claudeTreeCount[b + 16u] = floatBitsToUint(ro.x);
				claudeTreeCount[b + 17u] = floatBitsToUint(ro.y);
				claudeTreeCount[b + 18u] = floatBitsToUint(ro.z);
			}
		}
#endif
		gl_FragColor = vec4(verdict, 1.0);
		return;
	}
	if (view == 33 || view == 34) {
		vec3 code = vec3(255.0);
		vec3 sub = vec3(0.0);
		if (primaryHit && primaryT < DEPTH_MISS * 0.5) {
			// the walk's OWN hit point: ro + rd * primaryT put floor hits
			// ~0.25 m inside the cell (2026-10-07), primaryT is not that
			vec3 q = primaryHp - primaryN * 1e-3;
			vec3 cc = floor(q);
			if (all(greaterThanEqual(cc, vec3(0.0))) && all(lessThan(cc, vec3(GRID_S)))) {
				vec3 an = abs(primaryN);
				float ax = an.x > 0.5 ? 0.0 : (an.y > 0.5 ? 1.0 : 2.0);
				float sg = (primaryN.x + primaryN.y + primaryN.z) > 0.0 ? 1.0 : 0.0;
				float face = ax * 2.0 + sg;
				code = cc + 128.0 * vec3(mod(face, 2.0), mod(floor(face / 2.0), 2.0), floor(face / 4.0));
				vec3 sv = floor((q - cc) * SUBV);
				sub = vec3(sv.x * 16.0 + sv.y, sv.z, 1.0);
			}
		}
		gl_FragColor = vec4((view == 33 ? code : sub) / 255.0, 1.0);
		return;
	}
	if (view == 22) {
		vec3 code = vec3(255.0);
		if (guideSet && !guideSky) {
			vec3 cc = floor(guideP - guideN * 1e-3);
			if (all(greaterThanEqual(cc, vec3(0.0))) && all(lessThan(cc, vec3(GRID_S)))) {
				vec3 an = abs(guideN);
				float ax = an.x > 0.5 ? 0.0 : (an.y > 0.5 ? 1.0 : 2.0);
				float sg = (guideN.x + guideN.y + guideN.z) > 0.0 ? 1.0 : 0.0;
				float face = ax * 2.0 + sg;
				code = cc + 128.0 * vec3(mod(face, 2.0), mod(floor(face / 2.0), 2.0),
						floor(face / 4.0));
			}
		}
		gl_FragColor = vec4(code / 255.0, 1.0);
		return;
	}
	// THE DARK-START INSTRUMENT (2026-10-06, John: "it starts off all
	// dark"): view 25 = this frame's raw linear radiance, written straight
	// out (no averaging, no denoiser); view 26 = the same radiance through
	// the normal averaging. Both presented as x/(1+x), exactly invertible.
	// The mean of many view-25 frames equals view 26's converged value iff
	// the per-frame estimate is unbiased.
	if (view == 25) {
		gl_FragColor = vec4(max(L, vec3(0.0)), 1.0);
		return;
	}
	if (view == 31)
		L = vec3(max(g_guideTest, 0.0) * 0.25);
	if (view == 23)
		L = (guideSet && !guideSky)
				? max(L - Ld, vec3(0.0)) / max(guideAlb, vec3(ALBEDO_FLOOR))
				: vec3(0.0);
	if (view == 32) {
		gl_FragColor = vec4(g_guidePair.x / 16.0, g_guidePair.y / 16.0, min(g_guidePair.z, 1.0), 1.0);
		return;
	}
	if (view == 30) {
		// R: first surface has a table with light in it; G: how far its
		// row sums are from its total, x1000 (0 = consistent); B: guided
		// bounces this path / 8
		// R: (total - sum of bins) / sum of bins, G: (sum of row sums - sum
		// of bins) / sum of bins, both as 0.5 + 50 x (so +-1% spans the
		// range); B: 1 where the first surface has a table with light
		gl_FragColor = vec4(clamp(0.5 + 50.0 * g_guideMis, 0.0, 1.0),
				clamp(0.5 + 50.0 * g_guideMis2, 0.0, 1.0), g_guideT0 > 0.0 ? 1.0 : 0.0, 1.0);
		return;
	}
	if (view == 27) {
		gl_FragColor = vec4(g_bounceWasted / 16.0, g_bounces / 16.0, g_neeZero / 32.0, 1.0);
		return;
	}
	if (view == 28) {
		gl_FragColor = vec4(g_neeCallsT / 16.0, 1.0);
		return;
	}
	if (view == 29) {
		gl_FragColor = vec4(g_neeZeroT / 16.0, 1.0);
		return;
	}
	if (view == 21) {
		// R = steps / 4096, G = rays / 64, B = steps per ray / 256 (linear)
		gl_FragColor = vec4(g_steps / 4096.0, g_rays / 64.0,
				(g_steps / max(g_rays, 1.0)) / 256.0, 1.0);
		return;
	}
	// --- accumulate ---------------------------------------------------
	// history is LINEAR radiance in a 16F target; averaging must happen
	// in linear light or jittered noise converges biased dark.
	// accumAlpha is the whole reset mechanism: 1.0 on teleport or a
	// grid rebase (history discarded), 0.5 while moving, 1/(2+N) at
	// rest — a true running average, so a parked camera converges by
	// 1/N rather than sitting at an EMA's perpetual noise floor.
	vec3 fresh = max(L, vec3(0.0));
#if CLAUDE_SUBPASS > 0
	// ONE MORE SAMPLE OF THIS FRAME: same camera as pass 0, so no
	// reprojection; averaged in by the pixel's own count. Depth, guide and
	// moments stay pass 0's (outGbuf / outMom were set from history above).
	{
		vec4 hS = texture2D(history, uv);
		vec4 hdS = texture2D(historyDirect, uv);
		float nS = hdS.a;
		float aS = 1.0 / (nS + 1.0);
		gl_FragColor = vec4(mix(max(hS.rgb, vec3(0.0)), fresh, aS), hS.a);
#ifdef CLAUDE_SPLIT_OUT
		outDirect = vec4(mix(max(hdS.rgb, vec3(0.0)), max(Ld, vec3(0.0)), aS),
				min(nS + 1.0, 4096.0));
#endif
		return;
	}
#endif
	vec3 prev = fresh;
	float a = 1.0;
	// THE PIXEL'S SURFACE, NOW (the same code the denoiser's guide stores;
	// glowing surfaces 0, the sky its own code). Computed here so a pixel
	// can tell, at rest, that it is looking at something else than its
	// history did.
	float faceNow = -1.0;   // -1: no fresh guide this frame
	if ((guideSet || guideSky) && view == 0) {
		faceNow = 0.0;
		if (guideSet && !any(greaterThan(guideLe, vec3(0.0)))) {
			vec3 an0 = abs(guideN);
			float ax0 = an0.x > 0.5 ? 0.0 : (an0.y > 0.5 ? 1.0 : 2.0);
			float sg0 = (guideN.x + guideN.y + guideN.z) > 0.0 ? 1.0 : 0.0;
			float co0 = ax0 < 0.5 ? guideP.x : (ax0 < 1.5 ? guideP.y : guideP.z);
			faceNow = 1.0 + (ax0 * 2.0 + sg0) * 65536.0
					+ clamp(floor((co0 + 1024.0) * 16.0 + 0.5), 0.0, 65535.0);
		}
		if (guideSky)
			faceNow = SKY_FACE_CODE;
	}
	vec2 huv = uv;          // where this pixel's history lives
	float nPix = 0.0;       // samples behind it (direct buffer alpha)
	if (accumAlpha < 0.999) {
		// MOVING (accumAlpha 0.5 is the C++'s "camera moved" value; a parked
		// camera runs 1/(2+N) <= 1/3, a teleport 1.0): reproject.
		bool moving = accumAlpha > 0.4;
		bool reproj = claudeReproject > 0.5 && moving;
		bool hValid = true;
		vec3 prevEye = claudePrevCamPos + 0.5;
		float tExp = -1.0;
		if (reproj) {
			// the point this pixel sees: its primary hit, or (sky) the
			// same direction infinitely far away
			vec3 dvec = primaryHit ? (ro + rd * primaryT) - prevEye : rd;
			float z = dot(dvec, claudePrevCamFwd);
			if (z > 1e-3) {
				vec2 ndcP = vec2(
						dot(dvec, claudePrevCamRight)
							/ (z * dot(claudePrevCamRight, claudePrevCamRight)),
						dot(dvec, claudePrevCamUp)
							/ (z * dot(claudePrevCamUp, claudePrevCamUp)));
				huv = ndcP * 0.5 + 0.5;
				hValid = all(greaterThanEqual(huv, vec2(0.0)))
						&& all(lessThanEqual(huv, vec2(1.0)));
				if (primaryHit)
					tExp = length(dvec);
			} else {
				hValid = false;
			}
		}
		vec4 h = texture2D(history, huv);
		if (hValid && all(lessThan(abs(h.rgb), vec3(1e6)))) {
			if (reproj) {
				// the history pixel must have seen the SAME surface: its
				// stored primary distance from the previous eye must match
				float tPrev = h.a * DEPTH_SCALE;
				bool same = primaryHit
						? abs(tPrev - tExp) < 0.04 * tExp + 0.08
						: tPrev > DEPTH_MAX_HIT - 1.0;   // sky saw sky
				if (same) {
#ifdef CLAUDE_SPLIT_OUT
					nPix = texture2D(historyDirect, huv).a;
#else
					nPix = 8.0;
#endif
					prev = max(h.rgb, vec3(0.0));
					a = max(1.0 / (nPix + 1.0), claudeMotionAlphaMin);
				}
			} else {
				prev = max(h.rgb, vec3(0.0));
				a = accumAlpha;
#ifdef CLAUDE_SPLIT_OUT
				nPix = texture2D(historyDirect, huv).a;
				// THE PIXEL'S OWN COUNT RULES WHEN IT IS YOUNGER than the
				// frame's: a pixel that started over (below) must average
				// its next samples by 1/(n+1), not by the frame's 1/500,
				// or it stays stuck near its first noisy sample (seen
				// 2026-10-06: dark patches lasting 60+ frames where a
				// light had been)
				if (claudePixelReset > 0.5)
					a = max(a, 1.0 / (nPix + 1.0));
#endif
				// A SURFACE CHANGE IS THIS PIXEL'S RESET, NOT THE FRAME'S
				// (claude_pixel_reset, 2026-10-06): at rest the whole image
				// shares one weight, so a moved light left glowing ghosts at
				// every place it had been (moving-light test: +30 %
				// brightness, -2.7 JOD). A pixel whose face code differs from
				// its history's starts over; a MIXED pixel (negative code: an
				// edge that sees two faces) is exempt, or edges would never
				// converge.
				// ...BUT ONLY FOR A SURFACE NO NEIGHBOUR HAD: the camera ray
				// is jittered inside its pixel, so an edge pixel flips
				// between two faces frame to frame, and resetting it on
				// every flip would never let an edge converge. A face
				// that a 3x3 neighbour's history already showed is that
				// jitter; a face none of them showed is new.
				float hFace = texture2D(historyGbuf, huv).a;
				if (claudePixelReset > 0.5 && faceNow >= 0.0 && hFace >= 0.0
						&& abs(hFace - faceNow) > 0.5) {
					bool seen = false;
					for (int dy = -1; dy <= 1; dy++)
					for (int dx = -1; dx <= 1; dx++) {
						float nf = texture2D(historyGbuf,
								huv + vec2(float(dx), float(dy)) * texelSize0).a;
						if (abs(abs(nf) - faceNow) < 0.5)
							seen = true;
					}
					if (!seen) {
						prev = fresh;
						a = 1.0;
						nPix = 0.0;
					}
				}
			}
		}
	}

	// THE RAW LAYER (claude_raw_frame, 2026-10-07; John: "only what we
	// actually collect"): this frame's rays and nothing remembered
	if (claudeRawFrame > 0.5) {
		prev = fresh;
		a = 1.0;
		nPix = 0.0;
	}
	gl_FragColor = vec4(mix(prev, fresh, a), tPack);
#ifdef CLAUDE_SPLIT_OUT
	vec3 freshD = max(Ld, vec3(0.0));
	vec3 prevD = freshD;
	if (a < 1.0) {
		vec4 hd = texture2D(historyDirect, huv);
		if (all(lessThan(abs(hd.rgb), vec3(1e6))))
			prevD = max(hd.rgb, vec3(0.0));
	}
	// alpha carries this pixel's sample count (reprojection; the present
	// pass's split ramp reads it too). Capped: it only ever feeds a weight.
	outDirect = vec4(mix(prevD, freshD, a), a < 1.0 ? min(nPix + 1.0, 4096.0) : 1.0);

	// the denoiser's guide and moments (see historyGbuf / historyMom)
	vec3 albNow = guideSet ? max(guideAlb, vec3(ALBEDO_FLOOR)) : vec3(1.0);
	float faceCode = 0.0;
	// no fresh guide this frame (a reflection, or fog before the window):
	// keep what the pixel had
	bool guideFresh = guideSet || guideSky;
	// glowing surfaces get code 0 too (2026-10-05): their light is their
	// own, so "radiance / albedo" there is not the light falling on them,
	// and the eye's white (claude_exposure) must not read them as such
	if (guideSet && view == 0
			&& !any(greaterThan(guideLe, vec3(0.0)))) {
		vec3 an = abs(guideN);
		float ax = an.x > 0.5 ? 0.0 : (an.y > 0.5 ? 1.0 : 2.0);
		float sgn = (guideN.x + guideN.y + guideN.z) > 0.0 ? 1.0 : 0.0;
		vec3 hpP = guideP;
		float coord = ax < 0.5 ? hpP.x : (ax < 1.5 ? hpP.y : hpP.z);
		// plane in 1/16 m from -1024 m (far levels reach past the grid)
		float q = clamp(floor((coord + 1024.0) * 16.0 + 0.5), 0.0, 65535.0);
		faceCode = 1.0 + (ax * 2.0 + sgn) * 65536.0 + q;
	}
	// THE SKY IS ONE SURFACE TOO (2026-10-06): a camera ray that escaped
	// gets a code of its own (above every face code), so the denoiser may
	// smooth the AIR in front of it -- fog over the sky stayed grainy,
	// because code 0 is never filtered. A clear sky is analytic and has no
	// variance, so the filter leaves it exactly alone. claude_exposure's
	// eye white skips this code (sky radiance is not light on a surface).
	if (guideSky && view == 0)
		faceCode = SKY_FACE_CODE;
	vec3 albAcc = albNow;
	float m2 = 0.0;
	float vfac = 1.0;
	float lD = dot(fresh / albNow, vec3(0.2126, 0.7152, 0.0722));
	m2 = lD * lD;
	float m1 = lD;
	if (a < 1.0) {
		vec4 hg = texture2D(historyGbuf, huv);
		vec4 hm = texture2D(historyMom, huv);
		if (all(lessThan(abs(hg.rgb), vec3(1e6))) && hm.g > 0.0
				&& hm.g <= 1.0 && hm.r >= 0.0 && hm.r < 1e12) {
			if (!guideFresh) {
				// no guide this frame: the history's guide stands
				faceCode = hg.a;
				albNow = max(hg.rgb, vec3(ALBEDO_FLOOR));
				lD = dot(fresh / albNow, vec3(0.2126, 0.7152, 0.0722));
				m2 = lD * lD;
				m1 = lD;
			}
			albAcc = mix(hg.rgb, albNow, a);
			m2 = mix(hm.r, m2, a);
			m1 = mix(hm.b, m1, a);
			vfac = (1.0 - a) * (1.0 - a) * hm.g + a * a;
			// A PIXEL THAT HAS SEEN TWO FACES IS NOT ON ONE SURFACE. The
			// sub-pixel jitter puts an edge pixel's samples on both sides
			// of the edge, so its average is a MIX of two surfaces. It is
			// marked MIXED (negative code, sticky until the history
			// resets): claude_denoise may still smooth it, by its own
			// noise, but it is never a neighbour and never feeds a
			// neighbour's noise estimate. Measured 2026-10-05 (measured.md,
			// "the denoiser"): letting its face-to-face contrast into the
			// clean pixels' estimate blurred their real light gradients,
			// and a 4000-frame image moved by up to 21/255 along edges.
			if (guideFresh && faceCode > 0.5
					&& (hg.a < -0.5 || abs(hg.a - faceCode) > 0.5))
				faceCode = -faceCode;
		}
	}
	outGbuf = vec4(albAcc, faceCode);
	outMom = vec4(m2, vfac, m1, 0.0);
#endif
}
