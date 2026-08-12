// claude_accum: one jittered path-traced sample per pixel per frame,
// blended into a persistent history buffer (EMA). Random ray directions
// turn the fixed-kernel artifacts (phantom kernel shadows, flat faces,
// razor shadow edges) into noise; the history average turns noise into
// converged soft lighting. accumAlpha comes from the CPU: 1.0 on
// teleport/volume-swap (hard reset), higher while moving, low while
// still (deep accumulation).
#define history texture0
#define faceCacheTex texture2
// world-space radiance cache (claude_radiance pass): 64^3 cells of
// 2 nodes, flattened to 512x512 as 8x8 tiles of 64x64 z-slices
#define radianceCache texture1

uniform sampler2D history;
uniform sampler2D radianceCache;
uniform sampler2D faceCacheTex;
uniform lowp float radianceStrength; // claude_radiance dial, 0 = off
uniform vec2 texelSize0;
uniform lowp float volumeDebug;
uniform sampler3D claudeVolume;
uniform sampler3D claudeCoarse; // 32^3 any-solid brick map (empty-leap)
uniform sampler3D claudeMaterials; // per-cell material id (x255)
uniform sampler2D claudeAtlas;     // 16x16 grid of 16px tiles; .a = height
uniform sampler2D claudeMatParams; // 256x1 per-material: R=spec G=gloss B=ore
#define MICRO_CARVE 2.0            // max sub-voxels a face may recede
uniform lowp float skyBounce;      // how much sky a bounced-off surface relays
uniform lowp float bounce2Strength; // claude_bounce2: 3rd bounce, 0 = off
// Bisection ladder (claude_bisect): 0 normal; 1 carve visible but
// cube-recipe inputs (face normal + face position) and shadow rays
// ignore carve; 2 +micro normal; 3 +micro position (= full shading);
// 5 +shadow-ray micro occlusion (= fully normal).
uniform lowp float claudeBisect;
// occupancy pyramid dial: 0 = classic 4-cell brick leap (exact old
// behavior), 1 = climb mips for 8/16/32-cell leaps through emptiness
uniform lowp float claudePyramid;
// contribution gate threshold for aimed emitter rays (0 = off)
uniform lowp float claudeNeeGate;
// accum-internal cost attribution (profiler night): 0 = full;
// 1 = skip emitter rays; 2 = skip sun rays; 3 = skip bounce rays;
// 4 = skip all three (primary visibility only)
uniform lowp float claudeCost;
// THE BOUNCE-RAY CLIFF (profiler: bounce = 33.5 of accum's 45 ms):
// 1 = replace the per-pixel bounce ray with a direct read of the
// face cache at the eye hit — the ADR-0006 endgame, previewed with
// v1's flat faces. All light still comes from real traced gathers.
uniform lowp float claudeFaceDirect;
uniform lowp float sunAngle;       // sun/moon angular DIAMETER, radians
uniform lowp float nightSkyGain;   // gain on the night dome
#define SKY_BOUNCE skyBounce
uniform vec3 volumeOrigin;         // volume cell (0,0,0) in world nodes
uniform lowp float textureAmount;  // 0 clay .. 1 full texture
uniform lowp float bevelStrength;  // analytic edge rounding (0..1)
uniform lowp float reliefStrength; // texture-derived micro relief (0..1)
uniform lowp float parallaxStrength; // march INTO the height field (0..1)
uniform lowp float jitterStrength;   // per-block colour variation (Teardown)
uniform lowp float microStrength;    // real micro-geometry depth (0..1)
#if __VERSION__ >= 130
#define texture3D texture
#endif
uniform vec3 volumeCamPos;   // camera in volume-local node units
uniform vec3 volumeCamFwd;   // unit look direction
uniform vec3 volumeCamRight; // camera right, pre-scaled by tan(fovX/2)
uniform vec3 volumeCamUp;    // camera up, pre-scaled by tan(fovY/2)
uniform vec3 volumeSunDir;   // unit direction toward the active light
uniform vec3 volumeLightCol; // active light color (sun/moon/none)
uniform lowp float dayNightRatio;
uniform float animationTimer;
uniform lowp float accumAlpha;
uniform vec3 prevCamPos;    // last frame's camera, volume-local
uniform vec3 prevCamFwd;
uniform vec3 prevCamRightU; // unit right/up (unscaled)
uniform vec3 prevCamUpU;
uniform vec2 prevCamTan;    // tan(fovX/2), tan(fovY/2)
uniform vec4 claudeEmitter0; // xyz = cell-space center, w = intensity
uniform vec4 claudeEmitter1;
uniform vec4 claudeEmitter2;
uniform vec4 claudeEmitter3;
uniform vec4 claudeEmitter4;
uniform vec4 claudeEmitter5;
uniform vec4 claudeEmitter6;
uniform vec4 claudeEmitter7;
uniform lowp float claudeEmitterCount;
uniform vec4 claudeHeldEmitter; // wielded light: own slot, never in emitters[]

CENTROID_ VARYING_ mediump vec2 varTexCoord;

// Interleaved gradient noise (Jimenez): spatially low-discrepancy, so
// 1-spp error reads as fine film grain instead of TV static; animated
// across frames by an R2-sequence offset.
float ign(vec2 p)
{
	return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

vec3 noise3(vec2 px, float seed)
{
	vec2 o = fract(vec2(seed * 0.7548776662, seed * 0.5698402909)) * 64.0;
	return vec3(ign(px + o),
			ign(px + o + vec2(17.0, 59.0)),
			ign(px + o + vec2(41.0, 23.0)));
}

// per-cell, per-frame coin flip for stochastic leaf transmission: rays
// pass through leaf cells ~45% of the time; temporal accumulation
// averages the flips into true dappled partial shadows
float cellHash(vec3 c)
{
	return fract(sin(dot(c + fract(animationTimer * 13.7),
			vec3(12.9898, 78.233, 37.719))) * 43758.5453);
}

// Deterministic transmittance per cell class: no coin flips, so no
// shimmer (the stochastic version was parked for exactly that). Cross
// three leaf cells and 0.55^3 of the light survives — real dappled shade.
float cellTransmit(float a)
{
	if (a > 0.50 && a < 0.53) return 0.55;  // leaves
	if (a > 0.55 && a < 0.60) return 0.92;  // glass
	if (a > 0.63 && a < 0.66) return 1.0;   // torch nub: too thin to shade
	if (a > 0.97 && a < 0.99) return 0.0;   // micro material: solid enough
	return 0.0;                              // opaque
}

bool leafPass(float a, vec3 cell)
{
	// PARKED (2026-08-11): light-ray dapple shimmers at current
	// accumulation depth; revisit with textures + deeper history.
	// Re-arm by restoring: a > 0.45 && a < 0.56 && cellHash(cell) < 0.45
	return false;
}

float pathDayLin()
{
	// engine dayNightRatio floors at ~0.175; remap to a true 0..1
	return clamp((dayNightRatio - 0.18) / 0.82, 0.0, 1.0);
}

vec3 pathAlbedo(vec3 raw)
{
	// NO luminance floor. Lifting dark albedo to 0.16 (and flooring it again
	// at 0.04) was propping up the bounceRay bug where shaded surfaces
	// returned black: everything was too dark, so materials were brightened
	// to compensate. With sky bounce and emitters working, the prop only
	// destroys contrast — dark stone could not be dark, and night became a
	// grey wash. The tiny floor that remains is numerical, not aesthetic.
	return max(pow(raw, vec3(2.2)), vec3(0.005));
}

// forward-declared: fog tint for far terrain = the sky WITHOUT the
// sun/moon disc. Mixing toward the full sky burned the 40x disc through
// distant mountains ("the moon and sun shine right through them").
vec3 pathSkyFog(vec3 rd);

vec3 pathSkyRadiance(vec3 rd)
{
	float up = clamp(rd.y, 0.0, 1.0);
	float day = pathDayLin();

	// The horizon must WARM as the sun drops. Near the horizon a ray takes a
	// long path through atmosphere, so blue scatters out and what survives is
	// orange — the old two-colour lerp had fixed endpoints and so had no dawn
	// and no dusk, only a blue sky dimmed toward black.
	float low = smoothstep(0.35, -0.05, volumeSunDir.y);   // 0 high sun .. 1 set
	vec3 zenith = mix(vec3(0.16, 0.34, 0.72), vec3(0.10, 0.15, 0.34), low);
	vec3 horizon = mix(vec3(0.62, 0.74, 0.92), vec3(0.95, 0.50, 0.22), low);
	// pow < 1 keeps the bright band tight to the horizon instead of a ramp
	vec3 sky = mix(horizon, zenith, pow(up, 0.42));

	float cosSun = max(dot(rd, volumeSunDir), 0.0);
	// Mie forward scattering — a broad warm halo that widens as the sun sets
	float mie = pow(cosSun, mix(28.0, 6.0, low)) * mix(0.35, 1.5, low);

	vec3 c = sky * day + volumeLightCol * mie;
	c += volumeLightCol * smoothstep(0.9993, 0.9997, cosSun) * 40.0;

	// NIGHT SKY. Previously the whole gradient was scaled by day, so at night
	// the dome went to black and the only light left was a dim directional
	// moon — night was dead outdoors while being washed out indoors. This is
	// not a fake ambient term: it is the sky's own faint radiance (airglow,
	// stars, scattered moonlight), and only rays that actually ESCAPE collect
	// it, so it cannot leak into a sealed cave.
	float night = 1.0 - day;
	// Moonlight SCATTERS through the atmosphere exactly as sunlight does —
	// it is sunlight bounced off a rock. That scattering is why a full-moon
	// sky is deep blue instead of black, and on a clear night it dominates
	// starlight completely. A constant night dome cannot tell a full moon
	// from a new one; this one is driven by the moon actually being up.
	float moonUp = clamp(volumeSunDir.y, 0.0, 1.0);
	// Kept deliberately WELL BELOW the moon's own directional light. At 12.0
	// the scattered dome outshone the beam several times over, and ambient
	// that strong erases shadows by definition — there was no moonshadow at
	// all. Moonlight behaves like sunlight: the direct beam dominates and the
	// sky fill is a fraction of it. Use claude_moon_gain for overall night
	// brightness, not this.
	float moonAmt = dot(volumeLightCol, vec3(0.33)) * 1.6 * moonUp;
	vec3 nightSky = mix(vec3(0.006, 0.010, 0.026),   // airglow, horizon
			vec3(0.010, 0.016, 0.040), up);          // airglow, zenith
	nightSky += mix(vec3(0.05, 0.08, 0.16), vec3(0.03, 0.06, 0.15), up)
			* moonAmt;
	// stars: sparse, only well above the horizon, and steady (no twinkle —
	// it would fight temporal accumulation)
	float st = fract(sin(dot(floor(rd * 220.0), vec3(12.9898, 78.233, 37.719)))
			* 43758.5453);
	nightSky += vec3(0.9, 0.92, 1.0) * step(0.9992, st) * smoothstep(0.05, 0.35, up) * 1.6;
	c += nightSky * night * nightSkyGain;
	return c;
}

// the sky's colour WITHOUT the sun/moon disc (see forward decl above):
// horizon/zenith gradient + a mild halo only, for fogging far terrain
vec3 pathSkyFog(vec3 rd)
{
	float up = clamp(rd.y, 0.0, 1.0);
	float day = pathDayLin();
	float low = smoothstep(0.35, -0.05, volumeSunDir.y);
	vec3 zenith = mix(vec3(0.16, 0.34, 0.72), vec3(0.10, 0.15, 0.34), low);
	vec3 horizon = mix(vec3(0.62, 0.74, 0.92), vec3(0.95, 0.50, 0.22), low);
	vec3 sky = mix(horizon, zenith, pow(up, 0.42));
	float cosSun = max(dot(rd, volumeSunDir), 0.0);
	float mie = pow(cosSun, mix(28.0, 6.0, low)) * mix(0.12, 0.5, low);
	float night = 1.0 - day;
	float moonUp = clamp(volumeSunDir.y, 0.0, 1.0);
	float moonAmt = dot(volumeLightCol, vec3(0.33)) * 1.6 * moonUp;
	vec3 nightSky = mix(vec3(0.006, 0.010, 0.026),
			vec3(0.010, 0.016, 0.040), up);
	return sky * day + volumeLightCol * mie
			+ (nightSky + vec3(0.04, 0.06, 0.12) * moonAmt)
				* night * nightSkyGain;
}

// Nested DDA: the SAME traversal as the world, one scale down. A ray
// entering a cell marches a 16^3 sub-grid in local coordinates. Misses fall
// through, so gaps are real, and the light rays use this too, so stones
// shadow each other honestly. The sub-grid is not stored — it is carved on
// demand, per face, by microSolid below.

// How deep this face recedes at one texel, in sub-voxels (0..C). Nearest
// sampling on purpose: 16x16 art maps 1:1 onto 16 sub-voxels, and the
// carve is integer anyway. Bilinear here is what made the stones lumpy.
float faceCarve(float slot, vec2 t, float C)
{
	vec2 mo = vec2(mod(slot, 16.0), floor(slot / 16.0)) * 16.0;
	vec2 p = clamp(floor(t), 0.0, 15.0);
	float h = texture2D(claudeAtlas, (mo + p + 0.5) / 256.0).a;
	return floor((1.0 - h) * C + 0.5);
}

// The face mapping is ALIGNED — deliberately identical in every cell.
//
// This looks like a missed opportunity for variation, and three attempts at
// variation are why it isn't. Per-cell rotation, mirroring, and a toroidal
// offset were each tried; all three produce the same fatal artifact. The
// carve depth at a shared edge is then whatever each block's own pattern
// says, so one side sits flush while the other is cut 12.5 cm deep, and the
// boundary becomes a CLIFF running the full metre. Grazing light turns every
// cliff into a shadow line whose length grows with distance from the lamp —
// a hard 1 m lattice, and worse than the repetition it was meant to hide.
//
// The rule: any per-cell change of mapping IS a discontinuity. Only a height
// field continuous in world space carves seamlessly. Minecraft tiles are
// drawn to tile at exactly 1 m, so the aligned mapping is already that — the
// pattern repeats, but nothing steps at the seam. Variation has to come from
// somewhere that isn't geometry (light, or a gentle albedo jitter).
vec2 faceUV(vec2 t, float h, float horiz)
{
	return t;
}

// Sub-voxel shape is carved AT TRACE TIME, per face, and ONLY on faces
// that are actually exposed. The old path baked one 16^3 grid carved by
// all six height maps at once, which failed twice over: a side face's
// carve punched holes through the front face, and a face pressed against
// a neighbour had to be re-filled with a 2-thick rind — a proud ridge at
// every 1 m boundary that cast its own shadow line. That ridge WAS the
// lattice John saw. Here an interface face is never carved at all, so
// neighbours meet flush, and every visible face shows its own projection.
// nbNeg/nbPos: 1 where that neighbour cell is solid.
bool microSolid(float slot, vec3 sc, float rot, vec3 nbNeg, vec3 nbPos)
{
	float C = max(1.0, floor(MICRO_CARVE * microStrength + 0.5));
	if (nbPos.y < 0.5 && sc.y > 15.0 - C
			&& 15.0 - sc.y < faceCarve(slot, faceUV(vec2(sc.x, sc.z), rot, 1.0), C))
		return false;
	if (nbNeg.y < 0.5 && sc.y < C
			&& sc.y < faceCarve(slot, faceUV(vec2(sc.x, 15.0 - sc.z), rot, 1.0), C))
		return false;
	if (nbPos.x < 0.5 && sc.x > 15.0 - C
			&& 15.0 - sc.x < faceCarve(slot, faceUV(vec2(sc.z, 15.0 - sc.y), rot, 0.0), C))
		return false;
	if (nbNeg.x < 0.5 && sc.x < C
			&& sc.x < faceCarve(slot, faceUV(vec2(15.0 - sc.z, 15.0 - sc.y), rot, 0.0), C))
		return false;
	if (nbPos.z < 0.5 && sc.z > 15.0 - C
			&& 15.0 - sc.z < faceCarve(slot, faceUV(vec2(15.0 - sc.x, 15.0 - sc.y), rot, 0.0), C))
		return false;
	if (nbNeg.z < 0.5 && sc.z < C
			&& sc.z < faceCarve(slot, faceUV(vec2(sc.x, 15.0 - sc.y), rot, 0.0), C))
		return false;
	return true;
}

bool microDDA(vec3 lo, vec3 rd, float slot, float rot,
		vec3 nbNeg, vec3 nbPos, out vec3 hitLocal, out vec3 hitNormal)
{
	vec3 p = clamp(lo, 0.0, 0.99999) * 16.0;
	vec3 cell = floor(p);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - p) + stepDir * 0.5 + 0.5) * invRd;
	int axis = -1;
	float t = 0.0;
	// 48, not 26: a diagonal march across a 16^3 grid needs up to 3*16 steps.
	// At 26 a grazing ray quit mid-block and reported "no hit", which reads as
	// a see-through eye ray or a missing shadow.
	for (int i = 0; i < 48; i++) {
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThan(cell, vec3(15.0))))
			return false;                      // left the cell: real gap
		if (microSolid(slot, cell, rot, nbNeg, nbPos)) {
			hitLocal = (p + rd * t) / 16.0;
			hitNormal = vec3(0.0);
			if (axis == 0) hitNormal.x = -stepDir.x;
			else if (axis == 1) hitNormal.y = -stepDir.y;
			else if (axis == 2) hitNormal.z = -stepDir.z;
			else hitNormal = -rd;
			return true;
		}
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x; sideDist.x += invRd.x; cell.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y; sideDist.y += invRd.y; cell.y += stepDir.y; axis = 1;
		} else {
			t = sideDist.z; sideDist.z += invRd.z; cell.z += stepDir.z; axis = 2;
		}
	}
	return false;
}

// visibility toward the (jittered) light direction: 1 lit, 0 blocked
void microNeighbours(vec3 cell, out vec3 nbNeg, out vec3 nbPos)
{
	const float S = 128.0;
	nbNeg = vec3(
		texture3D(claudeVolume, (cell + vec3(-0.5, 0.5, 0.5)) / S).a > 0.9 ? 1.0 : 0.0,
		texture3D(claudeVolume, (cell + vec3(0.5, -0.5, 0.5)) / S).a > 0.9 ? 1.0 : 0.0,
		texture3D(claudeVolume, (cell + vec3(0.5, 0.5, -0.5)) / S).a > 0.9 ? 1.0 : 0.0);
	nbPos = vec3(
		texture3D(claudeVolume, (cell + vec3(1.5, 0.5, 0.5)) / S).a > 0.9 ? 1.0 : 0.0,
		texture3D(claudeVolume, (cell + vec3(0.5, 1.5, 0.5)) / S).a > 0.9 ? 1.0 : 0.0,
		texture3D(claudeVolume, (cell + vec3(0.5, 0.5, 1.5)) / S).a > 0.9 ? 1.0 : 0.0);
}

// ---- far cascades (claude_lod Phase 2): 2 m / 4 m / 8 m octaves ----
// Detail halves per level so no seam ever jumps more than one octave:
// near volume 1 m to +/-64, then 2 m to +/-128, 4 m to +/-256, 8 m to
// +/-512. Each level is a full CONCENTRIC volume (covers the near field
// coarsely too), so promoted rays are always correct-but-coarse and
// never need demotion. Slabs 0/1/2 of a 5-slab texture. All positions
// are VOLUME-LOCAL node coords; level cells are (p - origin) / cell.
uniform sampler3D claudeCascades;      // 128x128x640 RGBA8
uniform sampler3D claudeCascadeCoarse; // 32x32x160 R8 any-solid bricks
uniform vec3 cascade0Origin;  // 2 m level origin, volume-local nodes
uniform vec3 cascade1Origin;  // 4 m
uniform vec3 cascade2Origin;  // 8 m
uniform vec3 cascade3Origin;  // 16 m
uniform vec3 cascade4Origin;  // 32 m
uniform vec3 cascadeValid;    // levels 0-2, 0/1 each
uniform vec3 cascadeValidB;   // levels 3-4 in .xy

vec4 cascadeSample(float slab, vec3 c)
{
	return texture3D(claudeCascades,
			vec3((c.xy + 0.5) / 128.0, (slab * 128.0 + c.z + 0.5) / 640.0));
}

// Binary sun occlusion marched in ONE cascade level from a volume-local
// point. Returns 1 lit, 0 blocked, -1 = left the box unblocked (caller
// continues in a coarser level). Start is biased 1.2 cells along the
// ray — the launch point sits on (or exits near) real terrain whose own
// coarse cell is >50% solid, and sampling it would self-shadow
// everything near any slope. The bias trades that for a slight light
// leak at terrain-scale silhouettes, invisible at this frequency.
float farShadowL(float slab, vec3 corigin, float csz, vec3 pvol, vec3 sd)
{
	vec3 pc = (pvol - corigin) / csz + sd * 1.2;
	if (any(lessThan(pc, vec3(0.0))) || any(greaterThanEqual(pc, vec3(128.0))))
		return -1.0;
	vec3 cell = floor(pc);
	vec3 stepDir = sign(sd);
	vec3 invRd = 1.0 / max(abs(sd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - pc) + stepDir * 0.5 + 0.5) * invRd;
	for (int i = 0; i < 160; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			sideDist.x += invRd.x; cell.x += stepDir.x;
		} else if (sideDist.y < sideDist.z) {
			sideDist.y += invRd.y; cell.y += stepDir.y;
		} else {
			sideDist.z += invRd.z; cell.z += stepDir.z;
		}
		if (any(lessThan(cell, vec3(0.0)))
				|| any(greaterThanEqual(cell, vec3(128.0))))
			return -1.0;
		vec3 cc = floor(cell / 4.0);
		if (texture3D(claudeCascadeCoarse, vec3((cc.xy + 0.5) / 32.0,
				(slab * 32.0 + cc.z + 0.5) / 160.0)).r < 0.5) {
			vec3 bb = cc * 4.0 + step(vec3(0.0), sd) * 4.0;
			vec3 rdg = (step(vec3(0.0), sd) * 2.0 - 1.0)
					* max(abs(sd), vec3(1e-6));
			vec3 tt = (bb - pc) / rdg;
			float tj = min(min(tt.x, tt.y), tt.z) + 1e-3;
			vec3 p2 = pc + sd * tj;
			cell = floor(p2);
			sideDist = tj + (stepDir * (cell - p2)
					+ stepDir * 0.5 + 0.5) * invRd;
			continue;
		}
		if (cascadeSample(slab, cell).a > 0.9)
			return 0.0;
	}
	return -1.0;
}

// Shadow chain: 2 m to +/-128, then 8 m to +/-512, then 32 m to
// +/-2048 — a mountain a mile out still blocks the sun. The chain MUST
// start at the finest far level: 8 m-only shadows let dawn light pour
// through every ridge thinner than half an 8 m cell, and the whole
// slope read backlit ("sun shines through it"). Grazing light is
// exactly when thin crests matter. 4 m and 16 m are skipped — each is
// close enough to its neighbour that the extra march buys nothing.
float farShadow(vec3 pvol, vec3 sd)
{
	if (cascadeValid.x > 0.5) {
		float v = farShadowL(0.0, cascade0Origin, 2.0, pvol, sd);
		if (v >= 0.0)
			return v;
	}
	if (cascadeValid.z > 0.5) {
		float v = farShadowL(2.0, cascade2Origin, 8.0, pvol, sd);
		if (v >= 0.0)
			return v;
	}
	if (cascadeValidB.y > 0.5) {
		float v = farShadowL(4.0, cascade4Origin, 32.0, pvol, sd);
		if (v >= 0.0)
			return v;
	}
	return 1.0;
}

// Ambient bounce for a far hit, marched in the SAME level's grid —
// John's principle: the pipeline is identical at every cascade level,
// only the block size changes. This is bounceRay's exact shape one
// octave up: a cosine ray that darkens in valleys (real AO), returns
// sky on escape (SKY_BOUNCE), and picks up sun-lit terrain color at its
// hit — which is what the hand-tuned skyAmb/ground-bounce approximation
// could never match ("the dynamic lighting range changes a lot").
vec3 farBounce(float slab, vec3 corigin, float csz, vec3 pvol, vec3 rd,
		vec3 sd)
{
	vec3 pc = (pvol - corigin) / csz;
	vec3 cell = floor(pc);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - pc) + stepDir * 0.5 + 0.5) * invRd;
	int axis = -1;
	for (int i = 0; i < 48; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			sideDist.x += invRd.x; cell.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			sideDist.y += invRd.y; cell.y += stepDir.y; axis = 1;
		} else {
			sideDist.z += invRd.z; cell.z += stepDir.z; axis = 2;
		}
		if (any(lessThan(cell, vec3(0.0)))
				|| any(greaterThanEqual(cell, vec3(128.0))))
			return pathSkyRadiance(rd) * SKY_BOUNCE; // escaped: sky bounce
		vec3 cc = floor(cell / 4.0);
		if (texture3D(claudeCascadeCoarse, vec3((cc.xy + 0.5) / 32.0,
				(slab * 32.0 + cc.z + 0.5) / 160.0)).r < 0.5) {
			vec3 bb = cc * 4.0 + step(vec3(0.0), rd) * 4.0;
			vec3 rdg = (step(vec3(0.0), rd) * 2.0 - 1.0)
					* max(abs(rd), vec3(1e-6));
			vec3 tt = (bb - pc) / rdg;
			float tj = min(min(tt.x, tt.y), tt.z);
			vec3 p2 = pc + rd * (tj + 1e-3);
			cell = floor(p2);
			sideDist = tj + 1e-3 + (stepDir * (cell - p2)
					+ stepDir * 0.5 + 0.5) * invRd;
			continue;
		}
		vec4 s = cascadeSample(slab, cell);
		if (s.a > 0.35 && axis >= 0) {
			// same shading bounceRay gives ITS hits: sky + sun, one deep
			vec3 n = vec3(0.0);
			if (axis == 0) n.x = -stepDir.x;
			else if (axis == 1) n.y = -stepDir.y;
			else n.z = -stepDir.z;
			vec3 hpv = corigin + (pc + rd * max(
					min(min(sideDist.x, sideDist.y), sideDist.z), 0.0))
					* csz; // approximate: cell-resolution is plenty here
			vec3 lit = pathSkyRadiance(n) * SKY_BOUNCE;
			float ndl = max(dot(n, sd), 0.0);
			if (ndl > 0.0)
				lit += volumeLightCol * ndl * 1.4
						* farShadow(corigin + (cell + n) * csz
							+ vec3(csz * 0.5), sd);
			return pathAlbedo(s.rgb) * lit;
		}
	}
	return pathSkyRadiance(rd) * SKY_BOUNCE * 0.5; // ran out: dim sky
}

// March ONE cascade level from world-node t0 along rd. Returns rgb +
// hit t in w on a hit; on box exit returns w = -(exitT + 1.0) so the
// caller resumes the NEXT level exactly where this one left off — one
// continuous t across every octave, no cracks, no double hits.
vec4 farTraceL(float slab, vec3 corigin, float csz, vec3 tint,
		vec3 ro, vec3 rd, vec3 sd, float t0)
{
	vec3 p0 = ro + rd * (t0 + 0.01 * csz);
	vec3 pc = (p0 - corigin) / csz;
	if (any(lessThan(pc, vec3(0.0))) || any(greaterThanEqual(pc, vec3(128.0))))
		return vec4(0.0, 0.0, 0.0, -(t0 + 1.0)); // outside: hand onward
	vec3 cell = floor(pc);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - pc) + stepDir * 0.5 + 0.5) * invRd;
	float tc = 0.0;
	int axis = -1;
	for (int i = 0; i < 192; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			tc = sideDist.x; sideDist.x += invRd.x; cell.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			tc = sideDist.y; sideDist.y += invRd.y; cell.y += stepDir.y; axis = 1;
		} else {
			tc = sideDist.z; sideDist.z += invRd.z; cell.z += stepDir.z; axis = 2;
		}
		if (any(lessThan(cell, vec3(0.0)))
				|| any(greaterThanEqual(cell, vec3(128.0))))
			return vec4(0.0, 0.0, 0.0, -(t0 + tc * csz + 1.0));
		vec3 cc = floor(cell / 4.0);
		if (texture3D(claudeCascadeCoarse, vec3((cc.xy + 0.5) / 32.0,
				(slab * 32.0 + cc.z + 0.5) / 160.0)).r < 0.5) {
			vec3 bb = cc * 4.0 + step(vec3(0.0), rd) * 4.0;
			vec3 rdg = (step(vec3(0.0), rd) * 2.0 - 1.0)
					* max(abs(rd), vec3(1e-6));
			vec3 tt = (bb - pc) / rdg;
			float tj = min(min(tt.x, tt.y), tt.z);
			if (tt.x <= tt.y && tt.x <= tt.z) axis = 0;
			else if (tt.y <= tt.z) axis = 1;
			else axis = 2;
			tc = tj;
			vec3 p2 = pc + rd * (tj + 1e-3);
			cell = floor(p2);
			sideDist = tj + 1e-3 + (stepDir * (cell - p2)
					+ stepDir * 0.5 + 0.5) * invRd;
			continue;
		}
		vec4 s = cascadeSample(slab, cell);
		if (s.a > 0.35 && axis >= 0) {
			vec3 n = vec3(0.0);
			if (axis == 0) n.x = -stepDir.x;
			else if (axis == 1) n.y = -stepDir.y;
			else n.z = -stepDir.z;
			float tw = t0 + tc * csz;
			vec3 hpv = ro + rd * tw;
			// reduced far shading: albedo x (sun + sky), per-cell jitter
			// so distant fields aren't flat. No micro, atlas, emitters,
			// or bounce — invisible at this angular size. Jitter SCALES
			// with cell size: +/-10% tuned for 8m read as loud noise on
			// 2m cells ("the 2m has a very different look").
			vec3 albedo = pathAlbedo(s.rgb);
			float jh = fract(sin(dot(cell + slab * 17.0,
					vec3(12.9898, 78.233, 37.719))) * 43758.5453);
			float jamp = min(0.10, 0.015 + 0.012 * csz);
			albedo *= 1.0 + (jh - 0.5) * 2.0 * jamp;
			// sd is the caller's DISC-JITTERED sun: accumulation averages
			// the binary shadow into penumbras. Unjittered, grazing dusk
			// light flipped adjacent cells fully lit/dark — the "picket
			// fence" checkerboard across far slopes.
			// direct: shadow origin QUANTIZED to the cell center — one
			// shadow value per face. Per-texel origins clipped the
			// uphill neighbour for texels near an edge, drawing a dark
			// frame around every far cell ("the frame stuff").
			float ndl = max(dot(n, sd), 0.0);
			vec3 direct = ndl > 0.0
					? volumeLightCol * (ndl * farShadow(
						corigin + (cell + 0.5 + n) * csz, sd))
					: vec3(0.0);
			// ambient: the near pipeline's own shape, one octave up — a
			// cosine bounce ray in THIS level's grid (farBounce). Frame-
			// varying direction hashed per cell; accumulation averages
			// it into a converged hemisphere exactly like the near field.
			float ftick = fract(animationTimer * 7.31);
			vec3 sp3 = vec3(
				fract(sin(dot(cell + ftick,
					vec3(12.9898, 78.233, 37.719))) * 43758.5453),
				fract(sin(dot(cell + ftick,
					vec3(93.989, 12.233, 57.719))) * 24634.6345),
				fract(sin(dot(cell + ftick,
					vec3(45.332, 88.443, 19.113))) * 31578.2846));
			vec3 ad = normalize(n + normalize(sp3 * 2.0 - 1.0 + vec3(1e-4)));
			if (dot(ad, n) < 0.0)
				ad = normalize(ad - 2.0 * dot(ad, n) * n);
			// true traced bounce for the rings you stare at (2m/4m,
			// where the seam lives); the 8m+ rings are fog-dominated and
			// keep the cheap sky/ground approximation — full bounce on
			// every ring cost ~5 fps for detail the haze erases anyway
			vec3 amb;
			if (slab < 1.5) {
				amb = farBounce(slab, corigin, csz,
						hpv + n * (0.5 * csz), ad, sd) * 1.15;
			} else {
				vec3 skyAmb = pathSkyRadiance(n);
				amb = mix(skyAmb, skyAmb * albedo * 2.5, 0.25) * SKY_BOUNCE
						+ volumeLightCol * albedo
							* (0.25 * clamp(volumeSunDir.y, 0.0, 1.0));
			}
			vec3 c = albedo * (direct + amb);
			if (s.a < 0.6) // far water: flat sky mirror
				c = mix(c, pathSkyFog(
						reflect(rd, vec3(0.0, 1.0, 0.0))), 0.6);
			// debug 5: per-level tint (2m red, 4m orange, 8m yellow...)
			if (volumeDebug > 4.5 && volumeDebug < 5.5)
				c = tint * (0.4 + 0.6 * ndl);
			// aerial perspective: the beauty term, and the concealer for
			// the data frontier (unseen terrain fades into atmosphere).
			// pathSkyFog, NOT pathSkyRadiance: the full sky contains the
			// 40x sun/moon disc, which burned straight through mountains.
			// Fog STARTS past the near seam (64 nodes) — fogging at the
			// seam itself drew an abrupt haze line exactly where the
			// resolution changes, doubling the visual discontinuity.
			c = mix(c, pathSkyFog(rd),
					1.0 - exp(-max(tw - 64.0, 0.0) / 2000.0));
			return vec4(c, tw);
		}
	}
	return vec4(0.0, 0.0, 0.0, -(t0 + tc * csz + 1.0));
}

// Chain the octaves: 2 m -> 4 m -> 8 m, each resuming at the previous
// level's exit t. An invalid level is skipped (the next one covers its
// box anyway, just coarser).
vec4 farTrace(vec3 ro, vec3 rd, vec3 sd, float t0)
{
	float tcur = t0;
	vec4 r;
	if (cascadeValid.x > 0.5) {
		r = farTraceL(0.0, cascade0Origin, 2.0,
				vec3(0.9, 0.15, 0.15), ro, rd, sd, tcur);
		if (r.w > 0.0)
			return r;
		tcur = -r.w - 1.0;
	}
	if (cascadeValid.y > 0.5) {
		r = farTraceL(1.0, cascade1Origin, 4.0,
				vec3(0.9, 0.55, 0.1), ro, rd, sd, tcur);
		if (r.w > 0.0)
			return r;
		tcur = -r.w - 1.0;
	}
	if (cascadeValid.z > 0.5) {
		r = farTraceL(2.0, cascade2Origin, 8.0,
				vec3(0.9, 0.9, 0.15), ro, rd, sd, tcur);
		if (r.w > 0.0)
			return r;
		tcur = -r.w - 1.0;
	}
	if (cascadeValidB.x > 0.5) {
		r = farTraceL(3.0, cascade3Origin, 16.0,
				vec3(0.2, 0.85, 0.25), ro, rd, sd, tcur);
		if (r.w > 0.0)
			return r;
		tcur = -r.w - 1.0;
	}
	if (cascadeValidB.y > 0.5) {
		r = farTraceL(4.0, cascade4Origin, 32.0,
				vec3(0.2, 0.8, 0.9), ro, rd, sd, tcur);
		if (r.w > 0.0)
			return r;
	}
	return vec4(0.0, 0.0, 0.0, -1.0);
}

float lightVis(vec3 ro, vec3 sd)
{
	const float S = 128.0;
	float vis = 1.0;
	float tcur = 0.0;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(sd);
	vec3 invRd = 1.0 / max(abs(sd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	for (int i = 0; i < 208; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			tcur = sideDist.x; sideDist.x += invRd.x; cell.x += stepDir.x;
		} else if (sideDist.y < sideDist.z) {
			tcur = sideDist.y; sideDist.y += invRd.y; cell.y += stepDir.y;
		} else {
			tcur = sideDist.z; sideDist.z += invRd.z; cell.z += stepDir.z;
		}
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(S))))
			// left the near volume unblocked: promote into the 8 m
			// cascade so FAR terrain still occludes — the mile-long
			// mountain shadow. No-op (returns 1) when cascades are off.
			return vis * farShadow(ro + sd * tcur, sd);
		vec3 cc = floor(cell / 4.0);
		if (textureLod(claudeCoarse, (cell + 0.5) / 128.0, 2.0).r < 0.5) {
			float lvl = 4.0;
			if (claudePyramid > 0.5 && textureLod(claudeCoarse,
					(cell + 0.5) / 128.0, 3.0).r < 0.5) {
				lvl = 8.0;
				if (textureLod(claudeCoarse,
						(cell + 0.5) / 128.0, 4.0).r < 0.5) {
					lvl = 16.0;
					if (textureLod(claudeCoarse,
							(cell + 0.5) / 128.0, 5.0).r < 0.5)
						lvl = 32.0;
				}
			}
			// empty brick: leap to its far side in one step
			vec3 bb = floor(cell / lvl) * lvl + step(vec3(0.0), sd) * lvl;
			vec3 rdg = (step(vec3(0.0), sd) * 2.0 - 1.0)
					* max(abs(sd), vec3(1e-6));
			vec3 tt = (bb - ro) / rdg;
			float tj = min(min(tt.x, tt.y), tt.z) + 1e-3;
			vec3 p2 = ro + sd * tj;
			cell = floor(p2);
			sideDist = tj + (stepDir * (cell - p2)
					+ stepDir * 0.5 + 0.5) * invRd;
			continue;
		}
		float a = texture3D(claudeVolume, (cell + 0.5) / S).a;
		if (a > 0.97 && a < 0.99 && microStrength > 0.0 && tcur < 20.0) {
			// micro material: shadow only if the sub-grid is actually hit
			float mslot = texture3D(claudeMaterials, (cell + 0.5) / S).r * 255.0;
			vec3 mh, mn;
			vec3 lentry = ro + sd * tcur - cell;
			float rot1 = fract(sin(dot(cell + volumeOrigin,
					vec3(41.3, 289.1, 77.7))) * 21311.7);
			vec3 nbN1, nbP1;
			microNeighbours(cell, nbN1, nbP1);
			if (mslot > 0.5 && microDDA(clamp(lentry, 0.0, 1.0), sd,
					floor(mslot + 0.5), rot1, nbN1, nbP1, mh, mn))
				return 0.0;
			continue;
		}
		if (a > 0.25) {
			float tr = cellTransmit(a);
			if (tr <= 0.0)
				return 0.0;
			vis *= tr;
			if (vis < 0.04)
				return 0.0;
		}
	}
	return 0.0;
}

// Cached multi-bounce radiance at a point (volume node coords). The
// cache stores light that has ALREADY bounced off at least one surface
// (sun/emitter light reflected around corners), updated incrementally by
// the claude_radiance pass — so adding it at a bounce hit turns the one
// explicit bounce into effectively N bounces over time. Addressing must
// match the update pass: cell = point/2, tile (z%8, z/8) of 64x64.
// Sampled one node off the face (hp + n) so the fetch lands in a cell
// whose air side the gather pass actually filled.
vec3 cacheRadiance(vec3 pnode)
{
	vec3 c = clamp(floor(pnode / 2.0), vec3(0.0), vec3(63.0));
	vec2 cuv = (c.xy + vec2(mod(c.z, 8.0), floor(c.z / 8.0)) * 64.0 + 0.5)
			/ 512.0;
	return texture2D(radianceCache, cuv).rgb;
}

// Per-FACE cached irradiance (claude_faces pass, ADR-0006 v1): the hit
// face's own value — oriented, leak-proof, corner-aware at block
// resolution. Addressing mirrors claude_faces: 4096x3072, one 128x128
// tile per (face, z-slice), tileIndex = face*128 + z, 32 tiles/row.
// Face order 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z.
vec3 faceCache(vec3 cell, vec3 n)
{
	float f;
	if (n.x > 0.5) f = 0.0;
	else if (n.x < -0.5) f = 1.0;
	else if (n.y > 0.5) f = 2.0;
	else if (n.y < -0.5) f = 3.0;
	else if (n.z > 0.5) f = 4.0;
	else f = 5.0;
	vec3 c = clamp(cell, vec3(0.0), vec3(127.0));
	float tileIndex = f * 128.0 + c.z;
	vec2 tile = vec2(mod(tileIndex, 32.0), floor(tileIndex / 32.0));
	vec2 uv = (tile * 128.0 + c.xy + 0.5) / vec2(4096.0, 3072.0);
	return texture2D(faceCacheTex, uv).rgb;
}

// The THIRD bounce (claude_bounce2): one more cosine hop fired from a
// bounce ray's hit. Escape returns sky — the same energy the SKY_BOUNCE
// approximation estimated, now properly sampled along one direction —
// and a solid hit returns that surface's sun-lit color, which is
// genuinely new light: a sunlit floor warming a ceiling. No deeper
// recursion; the radiance cache carries everything beyond.
// forward decl: defined after bounceRay, but bounce hits need torch NEE
vec3 emitterLight(vec3 hp, vec3 n);
vec3 emitterLightCheap(vec3 hp, vec3 n);

// Mode-11 instrument: emitterVis records WHY it returned what it did.
// 0 clear (never met a carved cell), 1 rim-passed, 2 blocked in OWN
// cell, 3 blocked crossing another carved cell, 4 blocked by full cube.
float g_evisCause = 0.0;

// Coarse-rung sun visibility for INDIRECT consumers (bounce hits):
// identical cell-exact A&W with pyramid leaps and far promotion, but no
// sub-voxel descent — carve detail in an indirect shadow is invisible,
// and this is where the old t<6/t<20 range gates' real insight lands
// cleanly: the CONSUMER picks the rung, not a hidden gate. Still 100%
// traced (John's no-cheats rule) — just a coarser rung of the ladder.
float lightVisCheap(vec3 ro, vec3 sd)
{
	const float S = 128.0;
	float vis = 1.0;
	float tcur = 0.0;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(sd);
	vec3 invRd = 1.0 / max(abs(sd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	for (int i = 0; i < 128; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			tcur = sideDist.x; sideDist.x += invRd.x; cell.x += stepDir.x;
		} else if (sideDist.y < sideDist.z) {
			tcur = sideDist.y; sideDist.y += invRd.y; cell.y += stepDir.y;
		} else {
			tcur = sideDist.z; sideDist.z += invRd.z; cell.z += stepDir.z;
		}
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(S))))
			return vis * farShadow(ro + sd * tcur, sd);
		if (textureLod(claudeCoarse, (cell + 0.5) / 128.0, 2.0).r < 0.5) {
			float lvl = 4.0;
			if (claudePyramid > 0.5 && textureLod(claudeCoarse,
					(cell + 0.5) / 128.0, 3.0).r < 0.5) {
				lvl = 8.0;
				if (textureLod(claudeCoarse,
						(cell + 0.5) / 128.0, 4.0).r < 0.5) {
					lvl = 16.0;
					if (textureLod(claudeCoarse,
							(cell + 0.5) / 128.0, 5.0).r < 0.5)
						lvl = 32.0;
				}
			}
			vec3 bb = floor(cell / lvl) * lvl + step(vec3(0.0), sd) * lvl;
			vec3 rdg = (step(vec3(0.0), sd) * 2.0 - 1.0)
					* max(abs(sd), vec3(1e-6));
			vec3 tt = (bb - ro) / rdg;
			float tj = min(min(tt.x, tt.y), tt.z) + 1e-3;
			vec3 p2 = ro + sd * tj;
			cell = floor(p2);
			sideDist = tj + (stepDir * (cell - p2)
					+ stepDir * 0.5 + 0.5) * invRd;
			tcur = tj;
			continue;
		}
		float a = texture3D(claudeVolume, (cell + 0.5) / S).a;
		if (a > 0.25) {
			// carved cells block as their 1m cube at this rung
			if (a > 0.6 && a < 0.97)
				continue; // emissive passes
			float tr = cellTransmit(a);
			if (a > 0.97 && a < 0.99)
				tr = 0.0;
			if (tr <= 0.0)
				return 0.0;
			vis *= tr;
			if (vis < 0.04)
				return 0.0;
		}
	}
	return vis;
}

vec3 skyProbe(vec3 ro, vec3 rd, vec3 sd)
{
	const float S = 128.0;
	float trans = 1.0;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	int axis = -1;
	for (int i = 0; i < 96; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			sideDist.x += invRd.x; cell.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			sideDist.y += invRd.y; cell.y += stepDir.y; axis = 1;
		} else {
			sideDist.z += invRd.z; cell.z += stepDir.z; axis = 2;
		}
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(S))))
			return pathSkyRadiance(rd) * SKY_BOUNCE * trans;
		vec3 cc = floor(cell / 4.0);
		if (textureLod(claudeCoarse, (cell + 0.5) / 128.0, 2.0).r < 0.5) {
			float lvl = 4.0;
			if (claudePyramid > 0.5 && textureLod(claudeCoarse,
					(cell + 0.5) / 128.0, 3.0).r < 0.5) {
				lvl = 8.0;
				if (textureLod(claudeCoarse,
						(cell + 0.5) / 128.0, 4.0).r < 0.5) {
					lvl = 16.0;
					if (textureLod(claudeCoarse,
							(cell + 0.5) / 128.0, 5.0).r < 0.5)
						lvl = 32.0;
				}
			}
			vec3 bb = floor(cell / lvl) * lvl + step(vec3(0.0), rd) * lvl;
			vec3 rdg = (step(vec3(0.0), rd) * 2.0 - 1.0)
					* max(abs(rd), vec3(1e-6));
			vec3 tt = (bb - ro) / rdg;
			float tj = min(min(tt.x, tt.y), tt.z);
			vec3 p2 = ro + rd * (tj + 1e-3);
			cell = floor(p2);
			sideDist = tj + 1e-3 + (stepDir * (cell - p2)
					+ stepDir * 0.5 + 0.5) * invRd;
			axis = -1;
			continue;
		}
		vec4 s = texture3D(claudeVolume, (cell + 0.5) / S);
		if (s.a > 0.25) {
			float tr = cellTransmit(s.a);
			if (tr > 0.0) {
				trans *= tr;
				if (trans < 0.05)
					return vec3(0.0);
				continue;
			}
			if (axis < 0)
				return vec3(0.0);
			vec3 n = vec3(0.0);
			if (axis == 0) n.x = -stepDir.x;
			else if (axis == 1) n.y = -stepDir.y;
			else n.z = -stepDir.z;
			vec3 hp = cell + 0.5 + n * 0.51;
			float ndl = max(dot(n, sd), 0.0);
			vec3 lit = pathSkyRadiance(n) * SKY_BOUNCE * 0.5;
			if (ndl > 0.0)
				lit += volumeLightCol * ndl * lightVisCheap(hp, sd) * 1.4;
			return pathAlbedo(s.rgb) * lit * trans;
		}
	}
	return vec3(0.0);
}

// radiance arriving from direction rd: sky on genuine exit, sun-lit
// one-bounce on hit, darkness otherwise (sd = jittered light direction)
vec3 bounceRay(vec3 ro, vec3 rd, vec3 sd)
{
	const float S = 128.0;
	float trans = 1.0;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	float t = 0.0;
	int axis = -1;
	for (int i = 0; i < 160; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x; sideDist.x += invRd.x; cell.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y; sideDist.y += invRd.y; cell.y += stepDir.y; axis = 1;
		} else {
			t = sideDist.z; sideDist.z += invRd.z; cell.z += stepDir.z; axis = 2;
		}
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(S)))) {
			if (cell.y < 0.0)
				return vec3(0.0);
			return pathSkyRadiance(rd) * trans;
		}
		vec3 cc = floor(cell / 4.0);
		if (textureLod(claudeCoarse, (cell + 0.5) / 128.0, 2.0).r < 0.5) {
			float lvl = 4.0;
			if (claudePyramid > 0.5 && textureLod(claudeCoarse,
					(cell + 0.5) / 128.0, 3.0).r < 0.5) {
				lvl = 8.0;
				if (textureLod(claudeCoarse,
						(cell + 0.5) / 128.0, 4.0).r < 0.5) {
					lvl = 16.0;
					if (textureLod(claudeCoarse,
							(cell + 0.5) / 128.0, 5.0).r < 0.5)
						lvl = 32.0;
				}
			}
			vec3 bb = floor(cell / lvl) * lvl + step(vec3(0.0), rd) * lvl;
			vec3 rdg = (step(vec3(0.0), rd) * 2.0 - 1.0)
					* max(abs(rd), vec3(1e-6));
			vec3 tt = (bb - ro) / rdg;
			float tj = min(min(tt.x, tt.y), tt.z);
			if (tt.x <= tt.y && tt.x <= tt.z) axis = 0;
			else if (tt.y <= tt.z) axis = 1;
			else axis = 2;
			t = tj;
			vec3 p2 = ro + rd * (tj + 1e-3);
			cell = floor(p2);
			sideDist = tj + 1e-3 + (stepDir * (cell - p2)
					+ stepDir * 0.5 + 0.5) * invRd;
			continue;
		}
		vec4 s = texture3D(claudeVolume, (cell + 0.5) / S);
		if (s.a > 0.25 && cellTransmit(s.a) > 0.0) {
			trans *= cellTransmit(s.a);
			if (trans < 0.04)
				return vec3(0.0);
			continue;
		}
		// Bounce rays must march the sub-grid too. Without this every GI and
		// ambient ray saw 1 m CUBES while the eye saw carved stone: indirect
		// light and occlusion were computed against the blocky world, so each
		// recess was darkened by its neighbour's full cube right at the 1 m
		// boundary. Ambient is most of the light here, so this dominated, and
		// unlike the emitter path it showed under sunlight too.
		// Range-gated hard at 6 m, unlike the eye and sun paths. A bounce ray's
		// sub-voxel detail only matters where it shapes CONTACT occlusion —
		// the near stone around a recess. Past a few metres the bounce is
		// low-frequency fill and cube-vs-carved is invisible, while the cost
		// is not: ungated at 20 m this cost 43 -> 17 fps on its own.
		if (s.a > 0.97 && s.a < 0.99 && microStrength > 0.0 && t < 6.0) {
			float ms = texture3D(claudeMaterials, (cell + 0.5) / S).r * 255.0;
			vec3 mh, mn;
			float rb = fract(sin(dot(cell + volumeOrigin,
					vec3(41.3, 289.1, 77.7))) * 21311.7);
			vec3 nbNb, nbPb;
			microNeighbours(cell, nbNb, nbPb);
			if (ms > 0.5 && microDDA(clamp(ro + rd * t - cell, 0.0, 1.0), rd,
					floor(ms + 0.5), rb, nbNb, nbPb, mh, mn)) {
				float fallm = 1.0 - t / 160.0;
				vec3 hpm = cell + mh + mn * 0.03125;
				vec3 litm = pathSkyRadiance(mn) * lightVisCheap(hpm, mn) * SKY_BOUNCE;
				float ndlm = max(dot(mn, sd), 0.0);
				if (ndlm > 0.0)
					litm += volumeLightCol * ndlm * lightVisCheap(hpm, sd) * 1.4;
				if (radianceStrength > 0.0)
					litm += faceCache(cell, mn) * radianceStrength;
				// torch NEE at the bounce vertex: a facet tilted away from
				// the torch gets filled by bounced torchlight from the lit
				// surfaces it faces (2026-08-12 — the missing second-bounce
				// term behind the "circular shadow" saga)
				litm += emitterLightCheap(hpm, mn);
				return pathAlbedo(s.rgb) * litm * fallm * trans;
			}
			continue;   // carved away here: the ray really does pass through
		}
		if (s.a > 0.25) {
			float fall = 1.0 - t / 160.0;
			// emissive hit: the surface IS a light — return its glow
			// directly (this is how torches light nearby walls)
			if (s.a > 0.6 && s.a < 0.97) {
				float e = clamp((s.a - 0.65) / 0.29, 0.0, 1.0);
				return pathAlbedo(s.rgb) * (0.4 + e * 2.0) * fall * trans;
			}
			vec3 n = vec3(0.0);
			if (axis == 0) n.x = -stepDir.x;
			else if (axis == 1) n.y = -stepDir.y;
			else n.z = -stepDir.z;
			// A surface lit only by SKY used to contribute nothing: this
			// returned black whenever the hit faced away from the sun or was
			// shadowed, so in shade, indoors, at dusk, or at night the whole
			// hemisphere term collapsed to zero. That is the "sometimes the
			// hemisphere light sucks" hole, and it is also a NOISE source —
			// an estimator that returns 0 or a large value has far more
			// variance than one that returns a smooth range.
			vec3 hp = ro + rd * t + n * 0.01;
			vec3 lit;
			if (bounce2Strength > 0.0) {
				// THIRD bounce (claude_bounce2): fire the next cosine
				// hop instead of the directional sky approximation.
				// skyProbe returns the same sky energy on escape (no
				// double counting) plus sun-lit surface color on a hit —
				// the genuinely new light (sunlit floor warms ceiling).
				// Direction hashed from position + frame so accumulation
				// integrates the hemisphere.
				vec3 h3 = vec3(
					fract(sin(dot(hp + fract(animationTimer * 9.17),
						vec3(12.9898, 78.233, 37.719))) * 43758.5453),
					fract(sin(dot(hp + fract(animationTimer * 9.17),
						vec3(93.989, 12.233, 57.719))) * 24634.6345),
					fract(sin(dot(hp + fract(animationTimer * 9.17),
						vec3(45.332, 88.443, 19.113))) * 31578.2846));
				vec3 ad2 = normalize(n + normalize(h3 * 2.0 - 1.0
						+ vec3(1e-4)));
				if (dot(ad2, n) < 0.0)
					ad2 = normalize(ad2 - 2.0 * dot(ad2, n) * n);
				vec3 hop = skyProbe(hp, ad2, sd);
				vec3 approx = pathSkyRadiance(n) * lightVis(hp, n)
						* SKY_BOUNCE;
				lit = mix(approx, hop, bounce2Strength);
			} else {
				lit = pathSkyRadiance(n) * lightVisCheap(hp, n) * SKY_BOUNCE;
			}
			float ndl = max(dot(n, sd), 0.0);
			if (ndl > 0.0)
				lit += volumeLightCol * ndl * lightVisCheap(hp, sd) * 1.4;
			// multi-bounce term: light already circulating in the cache
			// (this is what lets a torch fill a room instead of dying at
			// its first bounce). Read UNATTENUATED, deliberately (John,
			// 2026-08-11: "let's kill it, we'll learn how to get it back
			// for real"): the old contact ramp was fake occlusion
			// canceling fake light — the 2-node cache is too coarse to
			// know a crease is occluded and back-fills corners. The
			// honest fix is a cache that resolves occlusion itself
			// (per-face irradiance, ADR-0006); until then corners may
			// wash bright near creases rather than fake-darken.
			if (radianceStrength > 0.0)
				lit += faceCache(cell, n) * radianceStrength;
			// torch NEE at the bounce vertex (see micro branch above)
			lit += emitterLightCheap(hp, n);
			return pathAlbedo(s.rgb) * lit * fall * trans;
		}
	}
	return vec3(0.0);
}

// Bilinear height from the atlas alpha. The atlas is NEAREST-filtered
// (colour wants crisp pixels), so interpolate by hand: without this the
// height is constant inside each texel and relief reads as stairs.
float atlasHeight(vec2 uv)
{
	vec2 tc = uv * 256.0 - 0.5;
	vec2 f = fract(tc);
	vec2 b = (floor(tc) + 0.5) / 256.0;
	float st = 1.0 / 256.0;
	float h00 = texture2D(claudeAtlas, b).a;
	float h10 = texture2D(claudeAtlas, b + vec2(st, 0.0)).a;
	float h01 = texture2D(claudeAtlas, b + vec2(0.0, st)).a;
	float h11 = texture2D(claudeAtlas, b + vec2(st, st)).a;
	return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// Micro-geometry: march the tile's height field INSIDE a cell. Unlike
// parallax mapping this can miss entirely — the ray then continues on
// through the cell, so gaps are real, silhouettes are real, and shadow
// rays see the same shape. Height comes from the atlas alpha.
vec4 getEmitter(int i)
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

// visibility toward a nearby emitter: DDA capped just short of it, so
// the emitter itself doesn't occlude its own light
float emitterVis(vec3 ro, vec3 ld, float maxT)
{
	const float S = 128.0;
	vec3 cell = floor(ro);
	// UNIFORM SUB-VOXEL TRACING (2026-08-12, John: "16x16 subvoxels
	// with real tracing, same as if we had 16 one-metre voxels"). The
	// origin cell is tested like every other cell — the old exemption
	// let pit floors skip their own walls (falsely lit) while flush
	// tops got dinged by neighbours (falsely dark): inverted shadows.
	// Callers bias the origin ~1 sub-voxel off the surface, so features
	// must stand taller than a single sub-voxel to cast — that's the
	// quantization floor, not a hack.
	vec3 lo0 = ro - cell;
	// own-cell test ONLY when the origin is genuinely inside the cell:
	// clamping an above-the-cell origin onto the grid top started the
	// march INSIDE the flush top slab — descending rays self-blocked
	// instantly (elevated blocks black in mode 10, immune to rim rules)
	if (microStrength > 0.0
			&& (claudeBisect < 0.5 || claudeBisect > 4.5)
			&& all(greaterThanEqual(lo0, vec3(0.0)))
			&& all(lessThan(lo0, vec3(1.0)))) {
		float a0 = texture3D(claudeVolume, (cell + 0.5) / S).a;
		if (a0 > 0.97 && a0 < 0.99) {
			float m0 = texture3D(claudeMaterials, (cell + 0.5) / S).r * 255.0;
			vec3 mh0, mn0;
			float r0 = fract(sin(dot(cell + volumeOrigin,
					vec3(41.3, 289.1, 77.7))) * 21311.7);
			vec3 nbN0, nbP0;
			microNeighbours(cell, nbN0, nbP0);
			if (m0 > 0.5 && microDDA(clamp(ro - cell, 0.0, 1.0), ld,
					floor(m0 + 0.5), r0, nbN0, nbP0, mh0, mn0)) {
				// rim clip passes regardless of ray direction (see loop);
				// sample behind the hit face (boundary coin-flip fix)
				vec3 scA0 = floor((mh0 - mn0 * 0.03125) * 16.0)
						+ vec3(0.0, 1.0, 0.0);
				if (abs(mn0.y) < 0.5 && (scA0.y > 15.5
						|| !microSolid(floor(m0 + 0.5), scA0, r0, nbN0, nbP0))) {
					g_evisCause = max(g_evisCause, 1.0);
				} else {
					g_evisCause = 2.0;
					return 0.0;
				}
			}
		}
	}
	vec3 stepDir = sign(ld);
	vec3 invRd = 1.0 / max(abs(ld), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	float t = 0.0;
	for (int i = 0; i < 48; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x; sideDist.x += invRd.x; cell.x += stepDir.x;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y; sideDist.y += invRd.y; cell.y += stepDir.y;
		} else {
			t = sideDist.z; sideDist.z += invRd.z; cell.z += stepDir.z;
		}
		if (t >= maxT)
			return 1.0;
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(S))))
			return 1.0;
		// pyramid leap (NEW for emitter shadows — they previously had no
		// acceleration at all; dial-gated so dial-off is exact baseline)
		if (claudePyramid > 0.5
				&& textureLod(claudeCoarse, (cell + 0.5) / 128.0, 2.0).r < 0.5) {
			float lvl = 4.0;
			if (textureLod(claudeCoarse, (cell + 0.5) / 128.0, 3.0).r < 0.5) {
				lvl = 8.0;
				if (textureLod(claudeCoarse,
						(cell + 0.5) / 128.0, 4.0).r < 0.5) {
					lvl = 16.0;
					if (textureLod(claudeCoarse,
							(cell + 0.5) / 128.0, 5.0).r < 0.5)
						lvl = 32.0;
				}
			}
			vec3 bb = floor(cell / lvl) * lvl + step(vec3(0.0), ld) * lvl;
			vec3 rdg = (step(vec3(0.0), ld) * 2.0 - 1.0)
					* max(abs(ld), vec3(1e-6));
			vec3 tt = (bb - ro) / rdg;
			float tj = min(min(tt.x, tt.y), tt.z) + 1e-3;
			if (tj >= maxT)
				return 1.0; // empty all the way to the flame
			vec3 p2 = ro + ld * tj;
			cell = floor(p2);
			sideDist = tj + (stepDir * (cell - p2)
					+ stepDir * 0.5 + 0.5) * invRd;
			t = tj;
			continue;
		}
		float a = texture3D(claudeVolume, (cell + 0.5) / S).a;
		// carved cells: march the sub-grid, binary, same rule as the
		// origin cell above — sub-voxels ARE voxels, no special cases
		if (a > 0.97 && a < 0.99 && microStrength > 0.0 && t < 20.0
				&& (claudeBisect < 0.5 || claudeBisect > 4.5)) {
			float mslot = texture3D(claudeMaterials, (cell + 0.5) / S).r * 255.0;
			vec3 mh, mn;
			float rotE = fract(sin(dot(cell + volumeOrigin,
					vec3(41.3, 289.1, 77.7))) * 21311.7);
			vec3 nbNe, nbPe;
			microNeighbours(cell, nbNe, nbPe);
			if (mslot > 0.5 && microDDA(clamp(ro + ld * t - cell, 0.0, 1.0), ld,
					floor(mslot + 0.5), rotE, nbNe, nbPe, mh, mn)) {
				// RIM CLIP, shadow-march side (2026-08-12: after the
				// eye-normal fix the circle became HARD and block-
				// aligned; mode-10 heatmap showed whole ELEVATED blocks
				// black — the first version only pardoned ASCENDING
				// rays, damning every surface above flame height whose
				// shadow rays point slightly down). Direction doesn't
				// matter; the HIT FACE does: a clip on a column SIDE
				// whose top is exposed is a pebble rim — passes. A hit
				// on a column TOP, or a side continuing upward, is a
				// real surface in the way — blocks. Tunnel-safe: steep
				// piercing rays hit tops, tops block.
				// sample behind the hit face (boundary coin-flip fix)
				vec3 scAe = floor((mh - mn * 0.03125) * 16.0)
						+ vec3(0.0, 1.0, 0.0);
				if (abs(mn.y) < 0.5 && (scAe.y > 15.5
						|| !microSolid(floor(mslot + 0.5), scAe, rotE, nbNe, nbPe))) {
					g_evisCause = max(g_evisCause, 1.0);
				} else {
					g_evisCause = 3.0;
					return 0.0;
				}
			}
			continue;
		}
		if (a > 0.25 && !(a > 0.6 && a < 0.97)) {
			g_evisCause = 4.0;
			return 0.0;
		}
	}
	return 1.0;
}

// Next-event estimation: aimed contribution from the nearest emitters.
// The fix for John's lopsided torch pools — light no longer waits for a
// random ambient ray to stumble into the torch.
// Diffuse torch light; when gloss > 0 also accumulates a Blinn-Phong
// glint into specAcc, reusing the SAME visibility trace and falloff —
// specular costs no extra rays. v is the direction toward the eye.
vec3 emitterLightSpec(vec3 hp, vec3 n, vec3 v, float gloss, inout vec3 specAcc,
		int nmax)
{
	vec3 acc = vec3(0.0);
	// slots 0-7 = static torches; slot 8 = the held (wielded) light,
	// which lives in its own uniform so it can NEVER stomp a real torch
	for (int i = 0; i < 9; i++) {
		if (i < 8 && (i >= nmax || float(i) >= claudeEmitterCount))
			continue;
		vec4 em = i < 8 ? getEmitter(i) : claudeHeldEmitter;
		if (em.w <= 0.0)
			continue;
		vec3 L = em.xyz - hp;
		float d2 = dot(L, L);
		if (d2 > 625.0)
			continue; // beyond 25 cells: negligible
		float dist = max(sqrt(d2), 0.8);
		vec3 ld = L / dist;
		// WRAPPED cosine, emitters only: a real flame has size and rough
		// ground scatters, so torch light curls slightly past the 90 deg
		// cutoff. With the hard cutoff, grazing views of carved floors
		// showed the unlit BACKS of the bump flanks — a pitch-black
		// semicircle between viewer and lamp ("dark semi circles",
		// 2026-08-12). Wrap keeps direction and shadows, kills the void.
		float ndl = max((dot(n, ld) + 0.35) / 1.35, 0.0);
		if (ndl <= 0.0)
			continue;
		// CONTRIBUTION GATE (profiler night: accum = 67% of the frame;
		// aimed emitter rays a top cost inside it): if this light's
		// maximum possible contribution is sub-visible, skip the
		// visibility trace entirely. The bound uses already-known
		// distance and cosine — no rays spent deciding.
		if (claudeNeeGate > 0.0
				&& em.w * em.w * 10.0 * ndl / max(d2, 1.0) < claudeNeeGate)
			continue;
		// standoff 0.25 (was 0.9, a relic of torches-as-glowing-blocks:
		// bodiless point lights need no self-occlusion guard, and 0.9
		// left a shadowless bubble around every flame)
		float vis = emitterVis(hp, ld, dist - 0.25);
		float fall = em.w * em.w * 10.0 * vis / max(d2, 1.0);
		acc += vec3(1.0, 0.72, 0.42) * (fall * ndl);
		if (gloss > 0.0) {
			float nh = max(dot(n, normalize(ld + v)), 0.0);
			specAcc += vec3(1.0, 0.72, 0.42)
					* (pow(nh, mix(16.0, 96.0, gloss)) * fall);
		}
	}
	return acc;
}

vec3 emitterLight(vec3 hp, vec3 n)
{
	vec3 dummy = vec3(0.0);
	return emitterLightSpec(hp, n, vec3(0.0), 0.0, dummy, 8);
}

// BOUNCE-VERTEX LIGHT BUDGET (overnight 2026-08-12, Teardown-tier
// principle): indirect hits check only the 2 nearest torches (+ held).
// Direct eye-hit lighting keeps the full list; the indirect tail gets
// the cheap seat — same light, ~1/4 the aimed rays per bounce.
vec3 emitterLightCheap(vec3 hp, vec3 n)
{
	vec3 dummy = vec3(0.0);
	return emitterLightSpec(hp, n, vec3(0.0), 0.0, dummy, 2);
}

// Reproject a volume-local point into last frame's screen; returns
// history uv, or marks invalid via w<0.5. Depth check happens at the
// caller (needs the stored alpha).
vec3 reprojectUv(vec3 W)
{
	vec3 rel = W - (prevCamPos + 0.5);
	float z = dot(rel, prevCamFwd);
	if (z < 0.1)
		return vec3(0.0, 0.0, -1.0);
	vec2 puv = vec2(dot(rel, prevCamRightU) / (z * prevCamTan.x),
			dot(rel, prevCamUpU) / (z * prevCamTan.y)) * 0.5 + 0.5;
	if (any(lessThan(puv, vec2(0.002))) || any(greaterThan(puv, vec2(0.998))))
		return vec3(0.0, 0.0, -1.0);
	return vec3(puv, 1.0);
}

void main(void)
{
	vec2 uv = varTexCoord.st;
	if (volumeDebug < 2.5) {
		// traced mode off: carry history through untouched
		gl_FragColor = texture2D(history, uv);
		return;
	}

	vec3 rnd = noise3(gl_FragCoord.xy, animationTimer * 100.0);
	vec3 rnd2 = noise3(gl_FragCoord.xy + vec2(131.0, 71.0),
			animationTimer * 100.0 + 37.7);

	// subpixel jitter: free anti-aliasing through the average
	vec2 juv = uv + (rnd.xy - 0.5) * texelSize0;
	vec2 ndc = juv * 2.0 - 1.0;
	vec3 rd = normalize(volumeCamFwd + ndc.x * volumeCamRight + ndc.y * volumeCamUp);
	vec3 ro = volumeCamPos + 0.5;

	const float S = 128.0;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	float t = 0.0;
	int axis = -1;
	vec3 fresh = vec3(0.0);
	vec3 viewTint = vec3(1.0);
	bool done = false;

	for (int i = 0; i < 384; i++) {
		if (all(greaterThanEqual(cell, vec3(0.0))) && all(lessThan(cell, vec3(S)))) {
			vec3 cc = floor(cell / 4.0);
			if (textureLod(claudeCoarse, (cell + 0.5) / 128.0, 2.0).r < 0.5) {
			float lvl = 4.0;
			if (claudePyramid > 0.5 && textureLod(claudeCoarse,
					(cell + 0.5) / 128.0, 3.0).r < 0.5) {
				lvl = 8.0;
				if (textureLod(claudeCoarse,
						(cell + 0.5) / 128.0, 4.0).r < 0.5) {
					lvl = 16.0;
					if (textureLod(claudeCoarse,
							(cell + 0.5) / 128.0, 5.0).r < 0.5)
						lvl = 32.0;
				}
			}
				// empty brick: leap to its far side, keeping the crossing
				// axis so a hit right after the jump gets a true normal
				vec3 bb = floor(cell / lvl) * lvl + step(vec3(0.0), rd) * lvl;
				vec3 rdg = (step(vec3(0.0), rd) * 2.0 - 1.0)
						* max(abs(rd), vec3(1e-6));
				vec3 tt = (bb - ro) / rdg;
				float tj = min(min(tt.x, tt.y), tt.z);
				if (tt.x <= tt.y && tt.x <= tt.z) axis = 0;
				else if (tt.y <= tt.z) axis = 1;
				else axis = 2;
				t = tj;
				vec3 p2 = ro + rd * (tj + 1e-3);
				cell = floor(p2);
				sideDist = tj + 1e-3 + (stepDir * (cell - p2)
						+ stepDir * 0.5 + 0.5) * invRd;
				continue;
			}
			vec4 s = texture3D(claudeVolume, (cell + 0.5) / S);
			// see through glass (windows!): tint and keep going. Leaves
			// stay visible to eye rays — only LIGHT rays sieve through
			// them, since transmitting eye rays makes canopies seethe.
			if (s.a > 0.55 && s.a < 0.60) {
				viewTint *= vec3(0.86, 0.93, 0.90);
				continue;
			}
			// torch nub: analytic glowing sphere AT THE EMISSION POINT
			// (flame height 0.65, matching the emitter) — the visible
			// body of a pure point light, transparent to light rays
			if (s.a > 0.63 && s.a < 0.66) {
				vec3 ctr = cell + vec3(0.5, 0.65, 0.5);
				vec3 oc = ro - ctr;
				float bq = dot(oc, rd);
				float cq = dot(oc, oc) - 0.13 * 0.13;
				if (bq * bq - cq > 0.0) {
					fresh = pathAlbedo(s.rgb) * 4.0;
					done = true;
					break;
				}
				continue;
			}
			// micro-geometry cell: march the material's sub-voxel grid
			bool microMiss = false;
			if (s.a > 0.97 && s.a < 0.99 && axis >= 0 && microStrength > 0.0
					&& t < 32.0) {
				vec3 nn0 = vec3(0.0);
				if (axis == 0) nn0.x = -stepDir.x;
				else if (axis == 1) nn0.y = -stepDir.y;
				else nn0.z = -stepDir.z;
				float mid0 = texture3D(claudeMaterials, (cell + 0.5) / S).r * 255.0;
				vec3 hl, hn;
				float rot0 = fract(sin(dot(cell + volumeOrigin,
						vec3(41.3, 289.1, 77.7))) * 21311.7);
				vec3 nbN0, nbP0;
				microNeighbours(cell, nbN0, nbP0);
				if (mid0 > 0.5 && microDDA(clamp(ro + rd * t - cell, 0.0, 1.0),
						rd, floor(mid0 + 0.5), rot0, nbN0, nbP0, hl, hn)) {
					// Bias by HALF A SUB-VOXEL (1/32 node), not the 0.01 used
					// for 1 m faces. A sub-voxel is 6.25 cm, so a 1 cm bias is
					// 16% of one: a shadow ray leaving a carved stone at a
					// grazing angle re-entered the very sub-voxel it left, and
					// the surface shadowed itself — acne read as "shadows that
					// shouldn't be there".
					// RIM-CLIP NORMAL FIX (2026-08-12, John: "it doesn't
					// actually get carved" yet the circle appears): a
					// grazing eye ray entering a flush field clips the
					// SIDE of the first full-height column and inherits a
					// sideways normal on what is visually a flat floor —
					// wrong-facing shading with no visible geometry. If
					// the sub-voxel above the hit is open sky within the
					// grid, this is a TOP surface: shade it as one. Real
					// walls (columns continuing upward) keep side normals.
					// ENTRY-HIT GUARD (the circle's conviction, 2026-08-12
					// bisect: step 1 fake-normal clean, step 2 real-normal
					// circle — the normal is the whole artifact). A ray
					// entering a cell already inside a solid column gets a
					// hit with NO crossing axis: the normal comes back
					// zero. Zero normal -> cosine ~0 -> near-black pixel,
					// exactly in the grazing zone. Entry hits take the
					// entry face's normal.
					if (dot(hn, hn) < 0.5) {
						// mode-12 verdict (2026-08-12 02:29): flat floors
						// show a view-centered ARC of sideways normals —
						// grazing rays cross cell corners where top-vs-
						// side crossing is a float coin-toss, resolved
						// coherently per direction. The entry face is NOT
						// the surface; what's ABOVE the entry decides:
						// open above = top (shade up), buried = wall.
						vec3 scA2 = floor(hl * 16.0) + vec3(0.0, 1.0, 0.0);
						if (scA2.y > 15.5 || !microSolid(floor(mid0 + 0.5),
								scA2, rot0, nbN0, nbP0))
							hn = vec3(0.0, 1.0, 0.0);
						else
							hn = nn0;
					}
					if (abs(hn.y) < 0.5) {
						// sample the column BEHIND the hit face (bias
						// against the normal): the hit sits exactly ON
						// a face and floor() coin-flips the column —
						// whole blocks flipped dark on rounding luck
						vec3 scAbove = floor((hl - hn * 0.03125) * 16.0)
								+ vec3(0.0, 1.0, 0.0);
						if (scAbove.y > 15.5 || !microSolid(floor(mid0 + 0.5),
								scAbove, rot0, nbN0, nbP0))
							hn = vec3(0.0, 1.0, 0.0);
					}
					vec3 hp2 = cell + hl + hn * 0.03125;
					// bisect substitutions: step 1 = cube normal AND cube
					// position; step 2 = real normal, cube position;
					// step >= 3 (or 0) = real normal and position
					vec3 bnrm = hn;
					vec3 bpos = hp2;
					if (claudeBisect > 0.5 && claudeBisect < 1.5) {
						bnrm = nn0;
						bpos = ro + rd * t + nn0 * 0.01;
					} else if (claudeBisect > 1.5 && claudeBisect < 2.5) {
						bpos = ro + rd * t + nn0 * 0.01;
					}
					vec3 alb = pathAlbedo(s.rgb);
					// Texture the SUB-VOXEL, not just the block. Carved
					// surfaces previously took the cell's average colour and
					// never touched the atlas, so the atlas drove the carve
					// (via alpha) while contributing no detail to what the
					// carve exposed — stones were the right shape and a flat
					// colour. Pick the face from the hit normal and sample the
					// same detail encoding the uncarved path uses.
					// SHADING PARITY (John, 2026-08-12, caught by V-key A/B:
					// "the places that were specular highlights... get
					// darker" with carving on): this branch also samples the
					// per-material spec params the cube branch has — carving
					// must not switch surfaces to a poorer shading pipeline.
					float mSpecStr = 0.0, mSpecGloss = 0.0, mSpecMask = 1.0;
					if (textureAmount > 0.0) {
						vec2 st = abs(hn.y) > 0.5
								? vec2(hl.x, hn.y > 0.0 ? hl.z : 1.0 - hl.z)
								: (abs(hn.x) > 0.5
									? vec2(hn.x > 0.0 ? hl.z : 1.0 - hl.z, 1.0 - hl.y)
									: vec2(hn.z > 0.0 ? 1.0 - hl.x : hl.x, 1.0 - hl.y));
						vec2 mo0 = vec2(mod(mid0, 16.0), floor(mid0 / 16.0)) * 16.0;
						vec2 auv0 = (mo0 + clamp(floor(st * 16.0), 0.0, 15.0) + 0.5) / 256.0;
						vec3 det0 = texture2D(claudeAtlas, auv0).rgb * 2.0;
						alb *= mix(vec3(1.0), det0, textureAmount);
						vec3 mp0 = texture2D(claudeMatParams,
								vec2((floor(mid0 + 0.5) + 0.5) / 256.0, 0.5)).rgb;
						mSpecStr = mp0.r;
						mSpecGloss = mp0.g;
						if (mp0.b > 0.5)
							mSpecMask = smoothstep(0.70, 0.90, atlasHeight(auv0));
					}
					float jh = fract(sin(dot(cell, vec3(12.9898, 78.233, 37.719)))
							* 43758.5453);
					alb *= 1.0 + (jh - 0.5) * 2.0 * jitterStrength;
					// sub-rungs 2.1/2.2/2.3: the real normal feeds ONLY the
					// torch / sun / bounce term respectively; the other two
					// use the cube normal. Names which formula circles.
					vec3 bnT = bnrm, bnS = bnrm, bnB = bnrm;
					if (claudeBisect > 2.05 && claudeBisect < 2.15) {
						bnS = nn0; bnB = nn0;
					} else if (claudeBisect > 2.15 && claudeBisect < 2.25) {
						bnT = nn0; bnB = nn0;
					} else if (claudeBisect > 2.25 && claudeBisect < 2.35) {
						bnT = nn0; bnS = nn0;
					}
					vec3 sd2 = normalize(volumeSunDir + (rnd2 - 0.5) * sunAngle);
					float ndl2 = max(dot(bnS, sd2), 0.0);
					bool skipSun2 = claudeCost > 1.5 && claudeCost < 2.5
							|| claudeCost > 3.5;
					vec3 dir2 = (ndl2 > 0.0 && !skipSun2)
							? vec3(ndl2 * lightVis(bpos, sd2)) * volumeLightCol
							: vec3(0.0);
					vec3 sp2 = normalize(rnd * 2.0 - 1.0);
					vec3 ad2 = normalize(bnB + sp2);
					if (dot(ad2, bnB) < 0.0) ad2 = normalize(ad2 - 2.0 * dot(ad2, bnB) * bnB);
					vec3 amb2 = vec3(0.0);
					if (claudeCost < 2.5) {
						if (claudeFaceDirect > 0.5 && radianceStrength > 0.0)
							amb2 = faceCache(cell, bnB) * radianceStrength * 1.15;
						else
							amb2 = bounceRay(bpos, ad2, sd2) * 1.15;
					}
					// origin biased ~1.5 sub-voxels off the surface: with
					// uniform own-cell tracing, shadow features must stand
					// taller than a sub-voxel to cast (quantization floor).
					// Specular rides the same visibility as the cube branch.
					vec3 specAcc2 = vec3(0.0);
					float glossOn2 = mSpecStr * mSpecMask;
					bool skipE2 = claudeCost > 0.5 && claudeCost < 1.5
							|| claudeCost > 3.5;
					vec3 em2 = skipE2 ? vec3(0.0)
							: emitterLightSpec(bpos + bnT * 0.0625, bnT, -rd,
							glossOn2 > 0.005 ? mSpecGloss : 0.0, specAcc2, 8);
					fresh = alb * (dir2 + amb2 + em2);
					if (glossOn2 > 0.005) {
						if (ndl2 > 0.0 && dir2.r + dir2.g + dir2.b > 0.0) {
							float nh2 = max(dot(bnrm, normalize(sd2 - rd)), 0.0);
							fresh += volumeLightCol
									* pow(nh2, mix(16.0, 96.0, mSpecGloss))
									* glossOn2;
						}
						fresh += specAcc2 * glossOn2;
					}
					// term-isolation heatmaps (modes 7-10): render ONE
					// lighting term, no albedo, so artifacts name their
					// own source. 7 torch, 8 bounce/cache, 9 sun,
					// 10 raw visibility toward the nearest emitter.
					if (volumeDebug > 6.5) {
						if (volumeDebug < 7.5) fresh = em2;
						else if (volumeDebug < 8.5) fresh = amb2;
						else if (volumeDebug < 9.5) fresh = dir2;
						else if (volumeDebug < 10.5) {
							vec3 eL = claudeEmitter0.xyz - hp2;
							float eD = max(length(eL), 1e-3);
							fresh = vec3(emitterVis(hp2 + hn * 0.0625,
									eL / eD, eD - 0.25));
						} else if (volumeDebug < 11.5) {
							// mode 11: WHY-map. green clear, yellow
							// rim-passed, red own-cell block, blue
							// cross-cell block, white cube block.
							vec3 eL = claudeEmitter0.xyz - hp2;
							float eD = max(length(eL), 1e-3);
							g_evisCause = 0.0;
							float v11 = emitterVis(hp2 + hn * 0.0625,
									eL / eD, eD - 0.25);
							if (g_evisCause < 0.5)
								fresh = vec3(0.0, 0.8, 0.1) * max(v11, 0.2);
							else if (g_evisCause < 1.5)
								fresh = vec3(0.9, 0.8, 0.1);
							else if (g_evisCause < 2.5)
								fresh = vec3(0.9, 0.05, 0.05);
							else if (g_evisCause < 3.5)
								fresh = vec3(0.15, 0.3, 0.95);
							else
								fresh = vec3(0.95);
						} else {
							// mode 12: the shading normal as color —
							// lavender up, green/red sideways, dark
							// down/zero. The circle paints its own cause.
							fresh = hn * 0.5 + 0.5;
						}
					}
					done = true;
					break;
				}
				// no stone along this ray: the cell is a real gap here, so
				// it must NOT fall through to the solid-cube branch —
				// that fall-through was keeping silhouettes cubic.
				microMiss = true;
			}
			if (!microMiss && s.a > 0.25 && axis >= 0) {
				// emissive primary hit: self-lit, no rays needed
				if (s.a > 0.6 && s.a < 0.97) {
					float e = clamp((s.a - 0.65) / 0.29, 0.0, 1.0);
					fresh = pathAlbedo(s.rgb) * (0.5 + e * 5.0);
					done = true;
					break;
				}
				vec3 n = vec3(0.0);
				if (axis == 0) n.x = -stepDir.x;
				else if (axis == 1) n.y = -stepDir.y;
				else n.z = -stepDir.z;
				// origin jitter (Teardown: 'position jittering to hide
				// voxel artifacts'): launching secondary rays from exact
				// hit points quantizes occlusion into blocky staircases
				vec3 hp = ro + rd * t + n * 0.01;
				vec3 tj = (rnd2.zxy - 0.5) * 0.35;
				hp += tj - n * dot(tj, n); // jitter within the face plane
				vec3 albedo = volumeDebug > 3.5
						? vec3(0.55) : pathAlbedo(s.rgb);
				// Per-block colour jitter: Teardown's answer to flatness
				// without textures — a stone wall becomes a thousand
				// slightly different stones. Hashed from cell position so
				// it is stable in space and across frames.
				if (jitterStrength > 0.0) {
					float jh = fract(sin(dot(cell, vec3(12.9898, 78.233, 37.719)))
							* 43758.5453);
					float jh2 = fract(sin(dot(cell, vec3(93.9898, 12.233, 57.719)))
							* 24634.6345);
					albedo *= 1.0 + (jh - 0.5) * 2.0 * jitterStrength;
					albedo.rg *= 1.0 + (jh2 - 0.5) * 0.6 * jitterStrength;
				}
				// In-face hit position: the grid hands us exact UVs and,
				// unlike triangle meshes, an EXACT constant tangent frame
				// (the other two axes). Everything below rides on that.
				vec3 hploc = ro + rd * t - cell;
				vec2 fuv; vec3 udir, vdir;
				if (axis == 0) { fuv = vec2(hploc.z, 1.0 - hploc.y);
					udir = vec3(0.0, 0.0, 1.0); vdir = vec3(0.0, -1.0, 0.0); }
				else if (axis == 1) { fuv = vec2(hploc.x, hploc.z);
					udir = vec3(1.0, 0.0, 0.0); vdir = vec3(0.0, 0.0, 1.0); }
				else { fuv = vec2(hploc.x, 1.0 - hploc.y);
					udir = vec3(1.0, 0.0, 0.0); vdir = vec3(0.0, -1.0, 0.0); }
				fuv = clamp(fuv, 0.001, 0.999);
				// Analytic bevel: near a cell edge, tilt the normal toward
				// the neighbouring face so cubes read as chamfered blocks
				// (Teardown's rounded look) — no art, no height map.
				if (bevelStrength > 0.0) {
					vec2 e = (fuv - 0.5) * 2.0;             // -1..1
					vec2 k = sign(e) * smoothstep(0.55, 1.0, abs(e));
					n = normalize(n + (udir * k.x + vdir * k.y)
							* bevelStrength);
				}
				// textured albedo: the DDA hit's position on the face IS
				// its UV — the grid's free gift. Blend by the dial so
				// texture can never fully bury the lighting.
				// Surface relief is INDEPENDENT of color texture: John's
				// ask — carved depth on clay-colored blocks. Both read the
				// same atlas, one for height (alpha), one for color (rgb).
				float specStr = 0.0;   // per-material glint (ore bits)
				float specGloss = 0.0;
				float specMask = 1.0;
				if ((textureAmount > 0.0 || reliefStrength > 0.0)
						&& volumeDebug < 3.5) {
					float mid = texture3D(claudeMaterials,
							(cell + 0.5) / S).r * 255.0;
					if (mid > 0.5) {
						vec2 uv2 = clamp(fuv, 0.07, 0.93);
						float slot = floor(mid + 0.5);
						vec2 auv = (vec2(mod(slot, 16.0),
								floor(slot / 16.0)) + uv2) / 16.0;
						// Per-material response. For ores the spec is
						// MASKED to texels standing proud of the face —
						// the atlas height doubles as "is this an ore
						// bit": stone base recedes, ore stays flush, so
						// only the bits glint.
						vec3 mp = texture2D(claudeMatParams,
								vec2((slot + 0.5) / 256.0, 0.5)).rgb;
						specStr = mp.r;
						specGloss = mp.g;
						if (mp.b > 0.5)
							specMask = smoothstep(0.70, 0.90,
									atlasHeight(auv));
						// micro relief: slope of the detail map perturbs
						// the normal (texel-scale surface roughness)
						// Parallax: step along the view ray inside the
						// face until we drop below the height field. This
						// is the cue relief can't give — features slide
						// over each other as you move (real motion depth).
						if (parallaxStrength > 0.0) {
							vec3 V = -rd;
							vec2 vt = vec2(dot(V, udir), dot(V, vdir));
							float vz = max(abs(dot(V, n)), 0.15);
							vec2 maxShift = vt / vz * parallaxStrength * 0.06;
							float layer = 1.0;
							vec2 cur = auv;
							for (int pi = 0; pi < 8; pi++) {
								float h = atlasHeight(cur);
								if (h >= layer)
									break;
								layer -= 0.125;
								cur -= maxShift * 0.125;
							}
							vec2 lo = (vec2(mod(slot, 16.0),
									floor(slot / 16.0)) + 0.07) / 16.0;
							vec2 hi = (vec2(mod(slot, 16.0),
									floor(slot / 16.0)) + 0.93) / 16.0;
							auv = clamp(cur, lo, hi);
						}
						if (reliefStrength > 0.0) {
							// Height lives in the atlas alpha (texel
							// luminance): tilt the normal by its slope AND
							// darken the grooves. The groove shadow is what
							// actually reads as depth on bark and stone —
							// directional shading alone is the weak cue.
							// Symmetric central differences on the
							// bilinear height: consistent slopes, no
							// directional bias, smooth inside texels.
							float st = 1.0 / 256.0;
							float h0 = atlasHeight(auv);
							float hx = atlasHeight(auv + vec2(st, 0.0))
									- atlasHeight(auv - vec2(st, 0.0));
							float hy = atlasHeight(auv + vec2(0.0, st))
									- atlasHeight(auv - vec2(0.0, st));
							n = normalize(n - (udir * hx + vdir * hy)
									* reliefStrength * 9.0);
							albedo *= 1.0 - reliefStrength * 0.55
									* clamp(0.55 - h0, 0.0, 0.55) / 0.55;
						}
						// atlas stores DETAIL/2 (texel / tile average):
						// multiply the cell's own (palette-tinted) color,
						// so texture adds variation without changing a
						// face's average brightness
						if (textureAmount > 0.0) {
							vec3 det = texture2D(claudeAtlas, auv).rgb * 2.0;
							albedo *= mix(vec3(1.0), det, textureAmount);
						}
					}
				}

				// direct light: jittered within the solar/lunar disc
				// so the average converges to soft penumbras
				vec3 sd = normalize(volumeSunDir + (rnd2 - 0.5) * sunAngle);
				float ndl = max(dot(n, sd), 0.0);
				vec3 direct = vec3(0.0);
				float sunVis = 0.0;
				bool skipSun = claudeCost > 1.5 && claudeCost < 2.5
						|| claudeCost > 3.5;
				if (ndl > 0.0 && !skipSun) {
					sunVis = lightVis(hp, sd);
					direct = vec3(ndl * sunVis) * volumeLightCol;
				}

				// one cosine-weighted ambient ray: uniform sphere point
				// added to the normal
				vec3 sp = normalize(rnd * 2.0 - 1.0);
				vec3 ad = n + sp;
				if (dot(ad, ad) < 1e-4)
					ad = n;
				ad = normalize(ad);
				if (dot(ad, n) < 0.0)
					ad = normalize(ad - 2.0 * dot(ad, n) * n);
				bool skipB = claudeCost > 2.5;
				vec3 amb = vec3(0.0);
				if (!skipB) {
					if (claudeFaceDirect > 0.5 && radianceStrength > 0.0)
						amb = faceCache(cell, n) * radianceStrength * 1.15;
					else
						amb = bounceRay(hp, ad, sd) * 1.15;
				}

				// Specular rides the same visibility as diffuse — the
				// glint appears only where the light already lands.
				vec3 specAcc = vec3(0.0);
				float glossOn = specStr * specMask;
				bool skipE = claudeCost > 0.5 && claudeCost < 1.5
						|| claudeCost > 3.5;
				vec3 emDiff = skipE ? vec3(0.0) : emitterLightSpec(hp, n, -rd,
						glossOn > 0.005 ? specGloss : 0.0, specAcc, 8);
				fresh = albedo * (direct + amb + emDiff);
				if (glossOn > 0.005) {
					if (sunVis > 0.0) {
						float nh = max(dot(n, normalize(sd - rd)), 0.0);
						specAcc += volumeLightCol * (sunVis
								* pow(nh, mix(16.0, 96.0, specGloss)));
					}
					fresh += specAcc * glossOn;
				}

				// term-isolation heatmaps on UNCARVED surfaces too — the
				// first mode-7 seam read was garbage because grass
				// rendered NORMAL shading next to heat-mapped dirt
				// (instrument asymmetry, caught 2026-08-12 02:18)
				if (volumeDebug > 6.5 && volumeDebug < 10.5) {
					if (volumeDebug < 7.5) fresh = emDiff;
					else if (volumeDebug < 8.5) fresh = amb;
					else if (volumeDebug < 9.5) fresh = direct;
					else {
						vec3 eLc = claudeEmitter0.xyz - hp;
						float eDc = max(length(eLc), 1e-3);
						fresh = vec3(emitterVis(hp, eLc / eDc, eDc - 0.25));
					}
				}
				// mode 12 normal-map on UNCARVED surfaces (seam parity)
				if (volumeDebug > 11.5 && volumeDebug < 12.5)
					fresh = n * 0.5 + 0.5;
				// mode 11 WHY-map on UNCARVED surfaces too, so the seam
				// is instrumented on both sides (colors as micro branch)
				if (volumeDebug > 10.5 && volumeDebug < 11.5) {
					vec3 eL = claudeEmitter0.xyz - hp;
					float eD = max(length(eL), 1e-3);
					g_evisCause = 0.0;
					float v11c = emitterVis(hp, eL / eD, eD - 0.25);
					if (g_evisCause < 0.5)
						fresh = vec3(0.0, 0.8, 0.1) * max(v11c, 0.2);
					else if (g_evisCause < 1.5)
						fresh = vec3(0.9, 0.8, 0.1);
					else if (g_evisCause < 2.5)
						fresh = vec3(0.9, 0.05, 0.05);
					else if (g_evisCause < 3.5)
						fresh = vec3(0.15, 0.3, 0.95);
					else
						fresh = vec3(0.95);
				}
				// mirror water: one traced reflection + jittered glint
				// (water = alpha band around 100/255)
				if (s.a > 0.3 && s.a < 0.5 && n.y > 0.5) {
					vec3 rr = reflect(rd, vec3(0.0, 1.0, 0.0));
					vec3 refl = bounceRay(hp, rr, sd);
					refl += volumeLightCol
							* pow(max(dot(rr, sd), 0.0), 64.0) * 2.5;
					fresh = mix(fresh, refl, 0.65);
				}
				done = true;
				break;
			}
		} else if (i > 0) {
			// near volume exhausted: continue into the far cascades,
			// with a disc-jittered sun so far shadows average into
			// penumbras exactly like near ones
			vec3 fsd = normalize(volumeSunDir + (rnd2 - 0.5) * sunAngle);
			vec4 far = farTrace(ro, rd, fsd, t);
			if (far.w > 0.0) {
				fresh = far.rgb;
				t = far.w;
				axis = 0; // synthetic: far hits are real geometry
				done = true;
				break;
			}
			fresh = pathSkyRadiance(rd);
			done = true;
			break;
		}
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x; sideDist.x += invRd.x; cell.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y; sideDist.y += invRd.y; cell.y += stepDir.y; axis = 1;
		} else {
			t = sideDist.z; sideDist.z += invRd.z; cell.z += stepDir.z; axis = 2;
		}
	}
	if (!done)
		fresh = vec3(0.0);
	fresh *= viewTint;

	// Reprojection: find where THIS pixel's world point was on last
	// frame's screen, and only trust history whose stored hit distance
	// (alpha channel) agrees with the previous camera's view of it.
	// Depth scale is 4096 now, not 200: cascade hits land out to ~900
	// nodes and clamping them to 200 classified ALL far terrain as sky —
	// the present pass would then paint raster over it. Sky is the top
	// of the range; the history tolerance goes RELATIVE (2% of distance)
	// because at 800 nodes a 1.5-node absolute band rejects everything.
	float tHit = done && axis >= 0 ? min(t, 4090.0) : 4096.0;
	vec3 W = ro + rd * min(t, 4090.0);
	if (tHit >= 4095.0)
		W = ro + rd * 400.0; // sky: reproject by direction, far point
	vec3 fresh_g = pow(max(fresh, vec3(0.0)), vec3(1.0 / 2.2));
	vec3 rp = reprojectUv(W);
	float a = 1.0; // no valid history: fresh sample stands alone
	vec3 prev = vec3(0.0);
	if (rp.z > 0.5 && accumAlpha < 0.99) {
		vec4 h = texture2D(history, rp.xy);
		float tPrev = h.a * 4096.0;
		float tExp = min(length(W - (prevCamPos + 0.5)), 4096.0);
		bool skyMatch = tHit >= 4095.0 && tPrev >= 3900.0;
		if (skyMatch || abs(tPrev - tExp) < max(1.5, 0.02 * tExp)) {
			// clamp history's drift while moving (bounds resample mush) —
			// but NOT when deeply converged: yanking settled history
			// toward each frame's noise was itself a pulse source
			float band = accumAlpha < 0.1 ? 4.0 : 0.3;
			prev = fresh_g + clamp(h.rgb - fresh_g, vec3(-band), vec3(band));
			a = accumAlpha;
		} else {
			// depth mismatch = aliased edge flipping under subpixel
			// jitter. Rejecting outright makes edges shimmer forever;
			// tightly-clamped history lets them settle into stable AA.
			prev = fresh_g + clamp(h.rgb - fresh_g, vec3(-0.12), vec3(0.12));
			a = max(accumAlpha, 0.3);
		}
	}

	// accumulate in gamma space (RGBA8 history: better dark precision)
	gl_FragColor = vec4(mix(prev, fresh_g, a), tHit / 4096.0);
}
