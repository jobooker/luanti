// claude_radiance: world-space radiance cache UPDATE pass.
//
// A persistent 64^3 cache (2-node cells, same 128-node footprint as the
// volume) of surface-bounced light, stored FLATTENED into a 512x512 2D
// texture: an 8x8 grid of 64x64 tiles, one tile per z-slice. GL 4.1 has
// no image store and cannot bind all layers of a 3D texture as one FBO
// target, so the cache lives in 2D with manual addressing; ping-pong via
// texture0 (last frame's cache) -> this render target, swapped per frame.
//
// Each texel is one cell. Per updated cell: shoot a few gather rays at
// the world; a hit returns that surface's outgoing radiance computed as
// albedo * (sun direct + LAST FRAME'S CACHE at the hit). The cache
// feeding its own update is the whole trick — every full refresh cycle
// deepens the light by one more bounce, so torch light diffuses through
// rooms over ~a second instead of costing an exponential ray tree.
//
// Deliberate exclusions, so the cache stays strictly COMPLEMENTARY to
// the one-bounce paths claude_accum already has (no double counting):
// - sky escape contributes 0 (bounceRay already adds one sky bounce);
// - no emitter NEE at gather hits — instead emitter light is INJECTED
//   directly into cells (visibility-tested point term), which is what
//   makes the cache's content start at bounce 2 when bounceRay reads it
//   (cache * albedo_hit2 * ... -> eye). emitterLight() covers bounce 1.
//
// Amortization: 1/8 of cells refresh per frame (interleaved groups, not
// slabs, so refresh never sweeps as a visible wave); all other texels
// copy last frame's value through. Updated cells blend EMA 0.25 to
// smooth the few-ray noise. A volume-origin shift zero-resets the cache
// wholesale for two frames (both ping-pong targets), accepted for v1.
#define prevCache texture0

uniform sampler2D prevCache;
uniform lowp float volumeDebug;
uniform lowp float radianceStrength; // claude_radiance: 0 = pass disabled
uniform lowp float cacheSkyStrength; // claude_cache_sky: sky-escape seeding
uniform lowp float skyBounce;
uniform lowp float nightSkyGain;
uniform lowp float dayNightRatio;
uniform float claudeRadianceFrame;   // frame counter mod 8 (CPU-wrapped)
uniform lowp float claudeRadianceReset; // 1 = write zeros (origin shift)
uniform sampler3D claudeVolume;
uniform sampler3D claudeCoarse;
uniform vec3 volumeSunDir;
uniform vec3 volumeLightCol;
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
uniform lowp float claudePyramid; // occupancy-pyramid leap climb dial
uniform lowp float claudeFaceDirect; // faces feed bounces: lattice is dead
#if __VERSION__ >= 130
#define texture3D texture
#endif

CENTROID_ VARYING_ mediump vec2 varTexCoord;

// cache geometry — keep in sync with secondstage.cpp (512x512 target)
// and cacheRadiance() in claude_accum/opengl_fragment.glsl
#define CACHE_N 64.0     // cells per axis
#define CACHE_CELL 2.0   // nodes per cell
#define CACHE_TILES 8.0  // tile grid is 8x8 slices
#define CACHE_TEX 512.0

float cellTransmit(float a)
{
	if (a > 0.50 && a < 0.53) return 0.55;  // leaves
	if (a > 0.55 && a < 0.60) return 0.92;  // glass
	if (a > 0.63 && a < 0.66) return 1.0;   // torch nub: too thin to shade
	if (a > 0.97 && a < 0.99) return 0.0;   // micro material: solid enough
	return 0.0;                              // opaque
}

vec3 pathAlbedo(vec3 raw)
{
	return max(pow(raw, vec3(2.2)), vec3(0.005));
}

// Trimmed pathSkyRadiance (claude_accum) — gradient + night dome only, no
// sun disc / mie / stars: the cache is low-frequency fill and the disc
// would inject the sun twice (NEE at gather hits already counts it).
// Keep the palette in sync with claude_accum's pathSkyRadiance.
vec3 cacheSky(vec3 rd)
{
	float up = clamp(rd.y, 0.0, 1.0);
	float day = clamp((dayNightRatio - 0.18) / 0.82, 0.0, 1.0);
	float low = smoothstep(0.35, -0.05, volumeSunDir.y);
	vec3 zenith = mix(vec3(0.16, 0.34, 0.72), vec3(0.10, 0.15, 0.34), low);
	vec3 horizon = mix(vec3(0.62, 0.74, 0.92), vec3(0.95, 0.50, 0.22), low);
	vec3 c = mix(horizon, zenith, pow(up, 0.42)) * day;
	float night = 1.0 - day;
	float moonUp = clamp(volumeSunDir.y, 0.0, 1.0);
	float moonAmt = dot(volumeLightCol, vec3(0.33)) * 1.6 * moonUp;
	vec3 nightSky = mix(vec3(0.006, 0.010, 0.026),
			vec3(0.010, 0.016, 0.040), up);
	nightSky += mix(vec3(0.05, 0.08, 0.16), vec3(0.03, 0.06, 0.15), up)
			* moonAmt;
	return c + nightSky * night * nightSkyGain;
}

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

// last frame's cache, addressed by a point in volume node coords
vec3 cacheFetch(vec3 pnode)
{
	vec3 c = clamp(floor(pnode / CACHE_CELL), vec3(0.0), vec3(CACHE_N - 1.0));
	vec2 cuv = (c.xy + vec2(mod(c.z, CACHE_TILES),
			floor(c.z / CACHE_TILES)) * CACHE_N + 0.5) / CACHE_TEX;
	return texture2D(prevCache, cuv).rgb;
}

// occlusion toward the sun/moon: 1 lit, 0 blocked, partial through
// leaves/glass. lightVis minus the micro branch — the cache is 2-node
// resolution, sub-voxel shadow detail is invisible at this frequency.
float cacheShadow(vec3 ro, vec3 sd)
{
	const float S = 128.0;
	float vis = 1.0;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(sd);
	vec3 invRd = 1.0 / max(abs(sd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	for (int i = 0; i < 160; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			sideDist.x += invRd.x; cell.x += stepDir.x;
		} else if (sideDist.y < sideDist.z) {
			sideDist.y += invRd.y; cell.y += stepDir.y;
		} else {
			sideDist.z += invRd.z; cell.z += stepDir.z;
		}
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(S))))
			return vis;
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
		if (a > 0.25) {
			// emissive cells pass (a torch must not shade its own cell)
			if (a > 0.6 && a < 0.97)
				continue;
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

// visibility toward an emitter, capped just short of it
float cacheEmitterVis(vec3 ro, vec3 ld, float maxT)
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
		if (a > 0.25 && !(a > 0.6 && a < 0.97) && cellTransmit(a) <= 0.0)
			return 0.0;
	}
	return 1.0;
}

// radiance arriving at the cell from direction rd. Sky escape used to
// return 0 unconditionally (to avoid re-counting the SKY_BOUNCE term at
// bounce hits) — but that left the cache with NOTHING to propagate in
// caves, whose entire light supply is sky through the entrance. With
// claude_cache_sky > 0 an escaping ray returns the (disc-free) sky
// radiance: each cell measures its own sky visibility with real rays,
// so a cave seeds exactly as much as its opening admits and a sealed
// cave stays black. The overlap with SKY_BOUNCE outdoors is mild (cache
// is only read at solid bounce hits) and the dial owns the tradeoff.
vec3 gatherRay(vec3 ro, vec3 rd)
{
	const float S = 128.0;
	float trans = 1.0;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	float t = 0.0;
	int axis = -1;
	for (int i = 0; i < 96; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x; sideDist.x += invRd.x; cell.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y; sideDist.y += invRd.y; cell.y += stepDir.y; axis = 1;
		} else {
			t = sideDist.z; sideDist.z += invRd.z; cell.z += stepDir.z; axis = 2;
		}
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(S))))
			return cacheSky(rd) * skyBounce * cacheSkyStrength * trans;
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
			if (trans < 0.05)
				return vec3(0.0);
			continue;
		}
		if (s.a > 0.25) {
			// emissive surface (lava, glowstone): area lights enter the
			// cache here, same glow model as bounceRay
			if (s.a > 0.6 && s.a < 0.97) {
				float e = clamp((s.a - 0.65) / 0.29, 0.0, 1.0);
				return pathAlbedo(s.rgb) * (0.4 + e * 2.0) * trans;
			}
			vec3 n = vec3(0.0);
			if (axis == 0) n.x = -stepDir.x;
			else if (axis == 1) n.y = -stepDir.y;
			else if (axis == 2) n.z = -stepDir.z;
			else return vec3(0.0); // hit inside the starting cell: no face
			vec3 hp = ro + rd * t + n * 0.01;
			float ndl = max(dot(n, volumeSunDir), 0.0);
			vec3 direct = ndl > 0.0
					? volumeLightCol * ndl * cacheShadow(hp, volumeSunDir)
					: vec3(0.0);
			// LAST frame's cache at the hit: this term is what turns one
			// bounce into N over successive refresh cycles
			vec3 cached = cacheFetch(hp + n);
			return pathAlbedo(s.rgb) * (direct + cached) * trans;
		}
	}
	return vec3(0.0);
}

void main(void)
{
	vec2 uv = varTexCoord.st;
	// origin shift: the cache is volume-local, so a rebased volume makes
	// every cell's content wrong — zero it (2 frames clears both targets)
	if (claudeRadianceReset > 0.5) {
		gl_FragColor = vec4(0.0);
		return;
	}
	// off or not in traced mode: keep the cache empty, near-zero cost.
	// Also dead when face-direct is on (bounce rays read the FACE cache;
	// nothing consumes this lattice) — skip the whole update.
	if (volumeDebug < 2.5 || radianceStrength <= 0.0
			|| claudeFaceDirect > 0.5) {
		gl_FragColor = vec4(0.0);
		return;
	}

	vec2 px = floor(gl_FragCoord.xy);
	vec2 tile = floor(px / CACHE_N);
	vec3 cell = vec3(px - tile * CACHE_N, tile.y * CACHE_TILES + tile.x);

	vec4 old = texture2D(prevCache, uv);
	// amortize: only this frame's interleaved 1/8 group recomputes
	float group = mod(cell.x + cell.y * 2.0 + cell.z * 4.0, 8.0);
	if (abs(mod(claudeRadianceFrame, 8.0) - group) > 0.5) {
		gl_FragColor = old;
		return;
	}

	// gather origin: first AIR node among the cell's 2x2x2 nodes. Cells
	// straddling a surface (exactly the ones bounce rays read) get an
	// origin on the air side; fully solid cells store nothing.
	vec3 base = cell * CACHE_CELL;
	vec3 ro = vec3(-1.0);
	for (int k = 0; k < 8; k++) {
		vec3 o = vec3(mod(float(k), 2.0), mod(floor(float(k) / 2.0), 2.0),
				floor(float(k) / 4.0));
		vec3 node = base + o;
		if (texture3D(claudeVolume, (node + 0.5) / 128.0).a <= 0.25) {
			ro = node + 0.5;
			break;
		}
	}
	if (ro.x < 0.0) {
		gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
		return;
	}

	// emitter injection: direct torch light deposited INTO the cell.
	// Same falloff as emitterLight but no ndl (a cell has no normal);
	// the 0.5 stands in for the hemisphere-average cosine, so a wall lit
	// via the cache roughly matches one lit by emitterLight directly.
	vec3 inj = vec3(0.0);
	for (int i = 0; i < 9; i++) {
		if (i < 8 && float(i) >= claudeEmitterCount)
			continue;
		vec4 em = i < 8 ? getEmitter(i) : claudeHeldEmitter;
		if (em.w <= 0.0)
			continue;
		vec3 L = em.xyz - ro;
		float d2 = dot(L, L);
		if (d2 > 625.0)
			continue; // beyond 25 nodes: negligible
		float dist = max(sqrt(d2), 0.8);
		float vis = cacheEmitterVis(ro, L / dist, dist - 0.9);
		inj += vec3(1.0, 0.72, 0.42)
				* (em.w * em.w * 10.0 * 0.5 * vis / max(d2, 1.0));
	}

	// gather: 4 uniform-sphere rays, directions hashed from cell+frame so
	// successive refreshes rotate the set and the EMA integrates them
	vec3 acc = vec3(0.0);
	for (int k = 0; k < 4; k++) {
		float seed = claudeRadianceFrame * 4.0 + float(k);
		vec2 h = vec2(
			fract(sin(dot(cell, vec3(12.9898, 78.233, 37.719))
					+ seed * 17.13) * 43758.5453),
			fract(sin(dot(cell, vec3(93.9898, 12.233, 57.719))
					+ seed * 9.71) * 24634.6345));
		float z = 1.0 - 2.0 * h.x;
		float r = sqrt(max(1.0 - z * z, 0.0));
		float ph = 6.2831853 * h.y;
		vec3 dir = vec3(r * cos(ph), r * sin(ph), z);
		acc += gatherRay(ro, dir);
	}
	vec3 fresh = inj + acc * 0.25;

	// EMA against the previous value (alpha marks "has been written":
	// after a reset the first refresh takes the fresh sample whole)
	vec3 outc = old.a > 0.5 ? mix(old.rgb, fresh, 0.25) : fresh;
	gl_FragColor = vec4(outc, 1.0);
}
