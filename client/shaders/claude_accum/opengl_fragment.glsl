// claude_accum: one jittered path-traced sample per pixel per frame,
// blended into a persistent history buffer (EMA). Random ray directions
// turn the fixed-kernel artifacts (phantom kernel shadows, flat faces,
// razor shadow edges) into noise; the history average turns noise into
// converged soft lighting. accumAlpha comes from the CPU: 1.0 on
// teleport/volume-swap (hard reset), higher while moving, low while
// still (deep accumulation).
#define history texture0

uniform sampler2D history;
uniform vec2 texelSize0;
uniform lowp float volumeDebug;
uniform sampler3D claudeVolume;
uniform sampler3D claudeCoarse; // 32^3 any-solid brick map (empty-leap)
uniform sampler3D claudeMaterials; // per-cell material id (x255)
uniform sampler2D claudeAtlas;     // 16x16 grid of 16px tiles
uniform sampler3D claudeMicro;     // 256x256x16: per-material 16^3 grids
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
	vec3 a = pow(raw, vec3(2.2));
	float lum = dot(a, vec3(0.2126, 0.7152, 0.0722));
	if (lum < 0.16)
		a *= 0.16 / max(lum, 0.02);
	// additive floor: multiplicative lift can't rescue pure black
	return max(a, vec3(0.04));
}

vec3 pathSkyRadiance(vec3 rd)
{
	float up = clamp(rd.y, 0.0, 1.0);
	vec3 sky = mix(vec3(0.55, 0.66, 0.82), vec3(0.22, 0.42, 0.78), up);
	float cosSun = max(dot(rd, volumeSunDir), 0.0);
	vec3 c = sky * pathDayLin();
	float disc = smoothstep(0.9993, 0.9997, cosSun);
	c += volumeLightCol * (disc * 40.0
			+ pow(cosSun, 48.0) * 3.0 + pow(cosSun, 8.0) * 0.4);
	c += volumeLightCol * 0.18;
	return c;
}

// Nested DDA: the SAME traversal as the world, one scale down. Each
// material owns a 16^3 occupancy grid; a ray entering a cell marches it
// in local coordinates. Misses fall through, so gaps are real, and the
// light rays use this too, so stones shadow each other honestly.
bool microOcc(float slot, vec3 sc)
{
	vec2 mo = vec2(mod(slot, 16.0), floor(slot / 16.0)) * 16.0;
	vec3 uv3 = vec3((mo.x + sc.x + 0.5) / 256.0,
			(mo.y + sc.y + 0.5) / 256.0, (sc.z + 0.5) / 16.0);
	return texture3D(claudeMicro, uv3).r > 0.5;
}

// Per-cell rotation: four yaw orientations chosen by a position hash, so
// neighbouring blocks never share a pattern (the tell that made cells
// read as a repeating frame).
vec3 microRot(vec3 sc, float r)
{
	if (r < 1.0) return sc;
	if (r < 2.0) return vec3(sc.z, sc.y, 15.0 - sc.x);
	if (r < 3.0) return vec3(15.0 - sc.x, sc.y, 15.0 - sc.z);
	return vec3(15.0 - sc.z, sc.y, sc.x);
}

// A face pressed against another solid block must NOT stay carved, or
// every pair of neighbours leaves a trench between them. Fill the rind
// on interface faces so walls read as continuous stone and only exposed
// faces keep their relief. nbNeg/nbPos: 1 where that neighbour is solid.
bool microSolid(float slot, vec3 sc, float rot, vec3 nbNeg, vec3 nbPos)
{
	if (microOcc(slot, microRot(sc, rot)))
		return true;
	const float RIND = 2.0;
	// Fill ONLY where every face whose rind this sub-voxel lies in is an
	// interface. If it also lies in an EXPOSED face's rind, leave it
	// carved — otherwise each block grows an uncarved border strip on its
	// visible face and the wall reads as framed tiles.
	bool exposed = false, iface = false;
	if (sc.x < RIND) { if (nbNeg.x > 0.5) iface = true; else exposed = true; }
	if (sc.x > 15.0 - RIND) { if (nbPos.x > 0.5) iface = true; else exposed = true; }
	if (sc.y < RIND) { if (nbNeg.y > 0.5) iface = true; else exposed = true; }
	if (sc.y > 15.0 - RIND) { if (nbPos.y > 0.5) iface = true; else exposed = true; }
	if (sc.z < RIND) { if (nbNeg.z > 0.5) iface = true; else exposed = true; }
	if (sc.z > 15.0 - RIND) { if (nbPos.z > 0.5) iface = true; else exposed = true; }
	return iface && !exposed;
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
	for (int i = 0; i < 26; i++) {
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
			return vis;
		vec3 cc = floor(cell / 4.0);
		if (texture3D(claudeCoarse, (cc + 0.5) / 32.0).r < 0.5) {
			// empty brick: leap to its far side in one step
			vec3 bb = cc * 4.0 + step(vec3(0.0), sd) * 4.0;
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
			float rot1 = floor(fract(sin(dot(cell + volumeOrigin,
					vec3(41.3, 289.1, 77.7))) * 21311.7) * 4.0);
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
		if (texture3D(claudeCoarse, (cc + 0.5) / 32.0).r < 0.5) {
			vec3 bb = cc * 4.0 + step(vec3(0.0), rd) * 4.0;
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
			float ndl = max(dot(n, sd), 0.0);
			if (ndl <= 0.0)
				return vec3(0.0);
			float sv = lightVis(ro + rd * t + n * 0.01, sd);
			return pathAlbedo(s.rgb) * ndl * sv * fall
					* volumeLightCol * 1.4 * trans;
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
		float a = texture3D(claudeVolume, (cell + 0.5) / S).a;
		if (a > 0.25 && !(a > 0.6 && a < 0.97))
			return 0.0;
	}
	return 1.0;
}

// Next-event estimation: aimed contribution from the nearest emitters.
// The fix for John's lopsided torch pools — light no longer waits for a
// random ambient ray to stumble into the torch.
vec3 emitterLight(vec3 hp, vec3 n)
{
	vec3 acc = vec3(0.0);
	for (int i = 0; i < 8; i++) {
		if (float(i) >= claudeEmitterCount)
			break;
		vec4 em = getEmitter(i);
		vec3 L = em.xyz - hp;
		float d2 = dot(L, L);
		if (d2 > 625.0)
			continue; // beyond 25 cells: negligible
		float dist = max(sqrt(d2), 0.8);
		vec3 ld = L / dist;
		float ndl = max(dot(n, ld), 0.0);
		if (ndl <= 0.0)
			continue;
		float vis = emitterVis(hp, ld, dist - 0.9);
		acc += vec3(1.0, 0.72, 0.42)
				* (em.w * em.w * 10.0 * ndl * vis / max(d2, 1.0));
	}
	return acc;
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
			if (texture3D(claudeCoarse, (cc + 0.5) / 32.0).r < 0.5) {
				// empty brick: leap to its far side, keeping the crossing
				// axis so a hit right after the jump gets a true normal
				vec3 bb = cc * 4.0 + step(vec3(0.0), rd) * 4.0;
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
			// torch nub: analytic sphere inside the cell — sub-voxel
			// shape with no occupancy bitmask
			if (s.a > 0.63 && s.a < 0.66) {
				vec3 ctr = cell + 0.5;
				vec3 oc = ro - ctr;
				float bq = dot(oc, rd);
				float cq = dot(oc, oc) - 0.20 * 0.20;
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
				float rot0 = floor(fract(sin(dot(cell + volumeOrigin,
						vec3(41.3, 289.1, 77.7))) * 21311.7) * 4.0);
				vec3 nbN0, nbP0;
				microNeighbours(cell, nbN0, nbP0);
				if (mid0 > 0.5 && microDDA(clamp(ro + rd * t - cell, 0.0, 1.0),
						rd, floor(mid0 + 0.5), rot0, nbN0, nbP0, hl, hn)) {
					vec3 hp2 = cell + hl + hn * 0.01;
					vec3 alb = pathAlbedo(s.rgb);
					float jh = fract(sin(dot(cell, vec3(12.9898, 78.233, 37.719)))
							* 43758.5453);
					alb *= 1.0 + (jh - 0.5) * 2.0 * jitterStrength;
					vec3 sd2 = normalize(volumeSunDir + (rnd2 - 0.5) * 0.07);
					float ndl2 = max(dot(hn, sd2), 0.0);
					vec3 dir2 = ndl2 > 0.0
							? vec3(ndl2 * lightVis(hp2, sd2)) * volumeLightCol
							: vec3(0.0);
					vec3 sp2 = normalize(rnd * 2.0 - 1.0);
					vec3 ad2 = normalize(hn + sp2);
					if (dot(ad2, hn) < 0.0) ad2 = normalize(ad2 - 2.0 * dot(ad2, hn) * hn);
					vec3 amb2 = bounceRay(hp2, ad2, sd2) * 1.15;
					fresh = alb * (dir2 + amb2 + emitterLight(hp2, hn));
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
				if ((textureAmount > 0.0 || reliefStrength > 0.0)
						&& volumeDebug < 3.5) {
					float mid = texture3D(claudeMaterials,
							(cell + 0.5) / S).r * 255.0;
					if (mid > 0.5) {
						vec2 uv2 = clamp(fuv, 0.07, 0.93);
						float slot = floor(mid + 0.5);
						vec2 auv = (vec2(mod(slot, 16.0),
								floor(slot / 16.0)) + uv2) / 16.0;
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
				vec3 sd = normalize(volumeSunDir + (rnd2 - 0.5) * 0.07);
				float ndl = max(dot(n, sd), 0.0);
				vec3 direct = vec3(0.0);
				if (ndl > 0.0)
					direct = vec3(ndl * lightVis(hp, sd)) * volumeLightCol;

				// one cosine-weighted ambient ray: uniform sphere point
				// added to the normal
				vec3 sp = normalize(rnd * 2.0 - 1.0);
				vec3 ad = n + sp;
				if (dot(ad, ad) < 1e-4)
					ad = n;
				ad = normalize(ad);
				if (dot(ad, n) < 0.0)
					ad = normalize(ad - 2.0 * dot(ad, n) * n);
				vec3 amb = bounceRay(hp, ad, sd) * 1.15;

				fresh = albedo * (direct + amb + emitterLight(hp, n));

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
	float tHit = done && axis >= 0 ? min(t, 199.0) : 200.0;
	vec3 W = ro + rd * min(t, 400.0);
	if (tHit >= 199.5)
		W = ro + rd * 400.0; // sky: reproject by direction, far point
	vec3 fresh_g = pow(max(fresh, vec3(0.0)), vec3(1.0 / 2.2));
	vec3 rp = reprojectUv(W);
	float a = 1.0; // no valid history: fresh sample stands alone
	vec3 prev = vec3(0.0);
	if (rp.z > 0.5 && accumAlpha < 0.99) {
		vec4 h = texture2D(history, rp.xy);
		float tPrev = h.a * 200.0;
		float tExp = min(length(W - (prevCamPos + 0.5)), 200.0);
		bool skyMatch = tHit >= 199.5 && tPrev >= 190.0;
		if (skyMatch || abs(tPrev - tExp) < 1.5) {
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
	gl_FragColor = vec4(mix(prev, fresh_g, a), tHit / 200.0);
}
