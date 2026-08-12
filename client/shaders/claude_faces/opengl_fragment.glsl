// claude_faces: per-FACE irradiance cache UPDATE pass (ADR-0006 v1).
//
// The successor to claude_radiance's 2-node air-cell lattice: one cached
// irradiance value per exposed block FACE. Faces have what air cells
// never did — an orientation — so gathers are cosine-weighted around the
// true normal, a crease face genuinely collects less light than an open
// one (emergent AO at block resolution), and light cannot leak through a
// one-block wall because opposite faces are different texels.
//
// v1 scope: ONE value per face (sub-face texel patches come later).
// Self-feeding exactly like the lattice: gather hits read LAST frame's
// face cache at the hit face, so every full refresh deepens the light by
// one bounce. Sky escape seeds via cacheSkyStrength (claude_cache_sky).
//
// Layout: 128^3 nodes x 6 faces, flattened to 4096x3072. One 128x128
// tile per (face, z-slice): tileIndex = face*128 + z in [0,768), grid 32
// tiles wide x 24 rows. Texel (x,y) within tile = node x,y. Face index:
// 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z. Keep addressing in sync with
// faceCache() in claude_accum/opengl_fragment.glsl.
#define prevFaces texture0

uniform sampler2D prevFaces;
uniform lowp float volumeDebug;
uniform lowp float radianceStrength; // shared master dial with the lattice
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
uniform vec3 claudeOriginDelta;   // cells, non-zero only on the rebase frame
uniform lowp float claudeCacheRemap; // 1 = shift across rebase, 0 = zero (pulse)
#if __VERSION__ >= 130
#define texture3D texture
#endif

CENTROID_ VARYING_ mediump vec2 varTexCoord;

#define FTEX_W 4096.0
#define FTEX_H 3072.0
#define FTILE 128.0
#define FGRIDW 32.0

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

// face index from an axis-aligned normal
float faceIndex(vec3 n)
{
	if (n.x > 0.5) return 0.0;
	if (n.x < -0.5) return 1.0;
	if (n.y > 0.5) return 2.0;
	if (n.y < -0.5) return 3.0;
	if (n.z > 0.5) return 4.0;
	return 5.0;
}

// LAST frame's cached irradiance at a face (solid cell + outward normal)
vec3 faceFetch(vec3 cell, vec3 n)
{
	vec3 c = clamp(cell, vec3(0.0), vec3(127.0));
	float tileIndex = faceIndex(n) * 128.0 + c.z;
	vec2 tile = vec2(mod(tileIndex, FGRIDW), floor(tileIndex / FGRIDW));
	vec2 uv = (tile * FTILE + c.xy + 0.5) / vec2(FTEX_W, FTEX_H);
	return texture2D(prevFaces, uv).rgb;
}

// Trimmed pathSkyRadiance (claude_accum) — gradient + night dome only, no
// sun disc / mie / stars. Keep the palette in sync.
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

// occlusion toward the sun/moon: 1 lit, 0 blocked, partial through
// leaves/glass (same as claude_radiance's cacheShadow)
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

// radiance arriving at the face from direction rd. A solid hit returns
// that surface's outgoing light: albedo * (sun direct + ITS cached face
// value from last frame) — the self-feed that deepens one bounce per
// refresh. Sky escape seeds skylight via cacheSkyStrength; a sealed cave
// stays black because no ray escapes.
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
			// emissive surface (lava, glowstone): area lights enter here
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
			vec3 cached = faceFetch(cell, n);
			return pathAlbedo(s.rgb) * (direct + cached) * trans;
		}
	}
	return vec3(0.0);
}

void main(void)
{
	// origin shift: the cache is volume-local, but the LIGHT it stores is
	// world-anchored — so SHIFT addresses by the origin delta instead of
	// zeroing (zeroing showed as a pulse-to-black under face-direct;
	// John, 2026-08-12). Reset counter: 2 = remap frame, 1 = carry the
	// remapped data into the other ping-pong target.
	if (claudeRadianceReset > 0.5) {
		if (claudeCacheRemap < 0.5) {
			gl_FragColor = vec4(0.0);
			return;
		}
		if (claudeRadianceReset > 1.5) {
			vec2 rpx = floor(gl_FragCoord.xy);
			vec2 rtile = floor(rpx / FTILE);
			float rti = rtile.y * FGRIDW + rtile.x;
			float rf = floor(rti / 128.0);
			vec3 rnode = vec3(rpx - rtile * FTILE, rti - rf * 128.0);
			vec3 oldn = rnode + claudeOriginDelta;
			if (any(lessThan(oldn, vec3(0.0)))
					|| any(greaterThanEqual(oldn, vec3(128.0)))) {
				gl_FragColor = vec4(0.0);
				return;
			}
			float ti2 = rf * 128.0 + oldn.z;
			vec2 t2 = vec2(mod(ti2, FGRIDW), floor(ti2 / FGRIDW));
			gl_FragColor = texture2D(prevFaces,
					(t2 * FTILE + oldn.xy + 0.5) / vec2(FTEX_W, FTEX_H));
		} else {
			gl_FragColor = texture2D(prevFaces, varTexCoord.st);
		}
		return;
	}
	// off or not in traced mode: keep the cache empty
	if (volumeDebug < 2.5 || radianceStrength <= 0.0) {
		gl_FragColor = vec4(0.0);
		return;
	}

	vec2 px = floor(gl_FragCoord.xy);
	vec2 tile = floor(px / FTILE);
	float tileIndex = tile.y * FGRIDW + tile.x;
	float f = floor(tileIndex / 128.0);
	float z = tileIndex - f * 128.0;
	vec3 node = vec3(px - tile * FTILE, z);

	vec4 old = texture2D(prevFaces, varTexCoord.st);
	// amortize: only this frame's interleaved 1/8 group recomputes —
	// EXCEPT cold texels (new geometry, no history), which ride a
	// 2-frame wheel instead: an 8-frame dark wait on freshly-arrived
	// terrain read as "LOD pop pulses dark" (John, 2026-08-12)
	float group = mod(node.x + node.y * 2.0 + node.z * 4.0 + f * 3.0, 8.0);
	if (abs(mod(claudeRadianceFrame, 8.0) - group) > 0.5) {
		bool cold = old.a < 0.5;
		bool fastwheel = abs(mod(claudeRadianceFrame, 2.0)
				- mod(group, 2.0)) < 0.5;
		if (!(cold && fastwheel)) {
			gl_FragColor = old;
			return;
		}
	}

	// live-face test: my node solid (not liquid/leaves-thin), the node in
	// front of the face air. Everything else stores nothing.
	vec3 n = vec3(0.0);
	if (f < 0.5) n = vec3(1.0, 0.0, 0.0);
	else if (f < 1.5) n = vec3(-1.0, 0.0, 0.0);
	else if (f < 2.5) n = vec3(0.0, 1.0, 0.0);
	else if (f < 3.5) n = vec3(0.0, -1.0, 0.0);
	else if (f < 4.5) n = vec3(0.0, 0.0, 1.0);
	else n = vec3(0.0, 0.0, -1.0);

	float aSelf = texture3D(claudeVolume, (node + 0.5) / 128.0).a;
	vec3 nb = node + n;
	bool nbIn = all(greaterThanEqual(nb, vec3(0.0)))
			&& all(lessThan(nb, vec3(128.0)));
	float aNb = nbIn ? texture3D(claudeVolume, (nb + 0.5) / 128.0).a : 1.0;
	if (aSelf <= 0.25 || aNb > 0.25) {
		gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
		return;
	}

	// gather origin: just off the face center
	vec3 ro = node + 0.5 + n * 0.51;

	// emitter injection with the face's true cosine (the lattice used a
	// 0.5 hemisphere fudge because cells had no normal; faces do)
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
		vec3 ld = L / dist;
		float ndl = max(dot(n, ld), 0.0);
		if (ndl <= 0.0)
			continue;
		float vis = cacheEmitterVis(ro, ld, dist - 0.9);
		inj += vec3(1.0, 0.72, 0.42)
				* (em.w * em.w * 10.0 * ndl * vis / max(d2, 1.0));
	}

	// gather: 4 COSINE-weighted rays around the face normal (normal +
	// uniform sphere point), directions hashed from node+face+frame so
	// successive refreshes rotate the set and the EMA integrates them
	vec3 acc = vec3(0.0);
	for (int k = 0; k < 4; k++) {
		float seed = claudeRadianceFrame * 4.0 + float(k) + f * 31.7;
		vec2 h = vec2(
			fract(sin(dot(node, vec3(12.9898, 78.233, 37.719))
					+ seed * 17.13) * 43758.5453),
			fract(sin(dot(node, vec3(93.9898, 12.233, 57.719))
					+ seed * 9.71) * 24634.6345));
		float zr = 1.0 - 2.0 * h.x;
		float r = sqrt(max(1.0 - zr * zr, 0.0));
		float ph = 6.2831853 * h.y;
		vec3 sph = vec3(r * cos(ph), r * sin(ph), zr);
		vec3 dir = normalize(n + sph + vec3(1e-4));
		if (dot(dir, n) < 0.0)
			dir = normalize(dir - 2.0 * dot(dir, n) * n);
		acc += gatherRay(ro, dir);
	}
	vec3 fresh = inj + acc * 0.25;

	// EMA, ASYMMETRIC: darkening converges twice as fast as brightening —
	// this is the fix for "torch light takes forever to go away" (the
	// memory drains faster than it fills; brightening keeps the slow
	// blend that smooths few-ray noise)
	float aUp = 0.25;
	float aDown = 0.5;
	float goingDown = dot(fresh, vec3(1.0)) < dot(old.rgb, vec3(1.0)) ? 1.0 : 0.0;
	// cold texel (geometry new to the volume): seed from the tracer's
	// own sky term — the energy a gather returns for an unoccluded
	// face, assumed half-visible. Converges to the traced answer via
	// the EMA within a few refreshes; kills the dark flash on arrival.
	vec3 outc;
	if (old.a > 0.5) {
		outc = mix(old.rgb, fresh, mix(aUp, aDown, goingDown));
	} else {
		vec3 seed = cacheSky(n) * skyBounce * cacheSkyStrength * 0.5;
		outc = mix(seed, fresh, 0.4);
	}
	gl_FragColor = vec4(outc, 1.0);
}
