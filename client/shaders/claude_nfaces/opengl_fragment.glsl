// claude_nfaces: NEAR-RING sub-face irradiance atlas (ADR-0006 v2).
//
// The octave ladder applied to light itself: the coarse per-face cache
// (claude_faces, 1 texel per 1m face) is correct at distance — a far
// face subtends a pixel — but reads as a flat quilt up close (John's
// verdict 2026-08-12: "we basically lose ambient occlusion"). This
// pass maintains a 4x4-texel-per-face atlas for the 32^3 cells around
// the camera only: 0.25m ambient resolution where eyes can see it,
// ~50MB instead of the ~800MB a full-volume upgrade would cost.
//
// Layout: 6 faces x 32 ring z-slices = 192 tiles of 128px (32 cells x
// 4 texels), grid 16 tiles wide x 12 rows = 2048x1536. Ring corner in
// volume-local cells = claudeNearOrigin (clamped so the ring never
// leaves the 128^3 volume); claudeNearPrev is last frame's corner so
// a moving camera REMAPS surviving texels instead of dropping them.
// Reader: faceCacheHP() in claude_accum — keep addressing in sync.
#define prevNear texture0
#define coarseFaces texture1

uniform sampler2D prevNear;
uniform sampler2D coarseFaces;
uniform lowp float volumeDebug;
uniform lowp float radianceStrength;
uniform lowp float cacheSkyStrength;
uniform lowp float skyBounce;
uniform lowp float nightSkyGain;
uniform lowp float dayNightRatio;
uniform float claudeRadianceFrame;
uniform lowp float claudeRadianceReset;
uniform lowp float claudeFaceTexels; // <1.5 = pass idle (cheap fill)
uniform vec3 claudeNearOrigin;
uniform vec3 claudeNearPrev;
uniform sampler3D claudeVolume;
uniform sampler3D claudeCoarse;
uniform vec3 volumeSunDir;
uniform vec3 volumeLightCol;
uniform vec4 claudeEmitter0;
uniform vec4 claudeEmitter1;
uniform vec4 claudeEmitter2;
uniform vec4 claudeEmitter3;
uniform vec4 claudeEmitter4;
uniform vec4 claudeEmitter5;
uniform vec4 claudeEmitter6;
uniform vec4 claudeEmitter7;
uniform lowp float claudeEmitterCount;
uniform vec4 claudeHeldEmitter;
uniform lowp float claudePyramid;
#if __VERSION__ >= 130
#define texture3D texture
#endif

CENTROID_ VARYING_ mediump vec2 varTexCoord;

#define NTEX_W 2048.0
#define NTEX_H 1536.0
#define NTILE 128.0
#define NGRIDW 16.0
#define NSUB 4.0

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

float faceIndex(vec3 n)
{
	if (n.x > 0.5) return 0.0;
	if (n.x < -0.5) return 1.0;
	if (n.y > 0.5) return 2.0;
	if (n.y < -0.5) return 3.0;
	if (n.z > 0.5) return 4.0;
	return 5.0;
}

// gather hits feed from the COARSE cache (4096x3072 layout): the
// sub-face detail of a surface a bounce away is below perception
vec3 faceFetch(vec3 cell, vec3 n)
{
	vec3 c = clamp(cell, vec3(0.0), vec3(127.0));
	float tileIndex = faceIndex(n) * 128.0 + c.z;
	vec2 tile = vec2(mod(tileIndex, 32.0), floor(tileIndex / 32.0));
	vec2 uv = (tile * 128.0 + c.xy + 0.5) / vec2(4096.0, 3072.0);
	return texture2D(coarseFaces, uv).rgb;
}

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
			if (s.a > 0.6 && s.a < 0.97) {
				float e = clamp((s.a - 0.65) / 0.29, 0.0, 1.0);
				return pathAlbedo(s.rgb) * (0.4 + e * 2.0) * trans;
			}
			vec3 n = vec3(0.0);
			if (axis == 0) n.x = -stepDir.x;
			else if (axis == 1) n.y = -stepDir.y;
			else if (axis == 2) n.z = -stepDir.z;
			else return vec3(0.0);
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
	if (claudeRadianceReset > 0.5) {
		gl_FragColor = vec4(0.0);
		return;
	}
	if (volumeDebug < 2.5 || radianceStrength <= 0.0
			|| claudeFaceTexels < 1.5) {
		gl_FragColor = vec4(0.0);
		return;
	}

	vec2 px = floor(gl_FragCoord.xy);
	vec2 tile = floor(px / NTILE);
	float tileIndex = tile.y * NGRIDW + tile.x;
	float f = floor(tileIndex / 32.0);
	float zl = tileIndex - f * 32.0;
	vec2 sub = px - tile * NTILE;
	vec2 cellXY = floor(sub / NSUB);
	vec2 texel = sub - cellXY * NSUB;
	vec3 node = claudeNearOrigin + vec3(cellXY, zl);

	// last frame's value for THIS WORLD NODE: remap through the previous
	// ring origin so camera motion carries texels instead of dropping them
	vec4 old = vec4(0.0);
	vec3 lp = node - claudeNearPrev;
	if (all(greaterThanEqual(lp, vec3(0.0))) && all(lessThan(lp, vec3(32.0)))) {
		float tiP = f * 32.0 + lp.z;
		vec2 tileP = vec2(mod(tiP, NGRIDW), floor(tiP / NGRIDW));
		vec2 uvP = (tileP * NTILE + lp.xy * NSUB + texel + 0.5)
				/ vec2(NTEX_W, NTEX_H);
		old = texture2D(prevNear, uvP);
	}

	// amortize 1/4 per frame (twice the coarse cache's rate: near-field
	// AO is what eyes track)
	float group = mod(texel.x + texel.y * 2.0 + zl + f, 4.0);
	if (abs(mod(claudeRadianceFrame, 4.0) - group) > 0.5) {
		gl_FragColor = old;
		return;
	}

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

	// gather origin: the SUB-FACE texel center, just off the face — this
	// is the whole point: creases and corners get their own gathers
	vec3 tu, tv;
	if (f < 1.5) { tu = vec3(0.0, 1.0, 0.0); tv = vec3(0.0, 0.0, 1.0); }
	else if (f < 3.5) { tu = vec3(1.0, 0.0, 0.0); tv = vec3(0.0, 0.0, 1.0); }
	else { tu = vec3(1.0, 0.0, 0.0); tv = vec3(0.0, 1.0, 0.0); }
	float du = (texel.x + 0.5) / NSUB - 0.5;
	float dv = (texel.y + 0.5) / NSUB - 0.5;
	vec3 ro = node + 0.5 + n * 0.51 + tu * du + tv * dv;

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
			continue;
		float dist = max(sqrt(d2), 0.8);
		vec3 ld = L / dist;
		float ndl = max(dot(n, ld), 0.0);
		if (ndl <= 0.0)
			continue;
		float vis = cacheEmitterVis(ro, ld, dist - 0.9);
		inj += vec3(1.0, 0.72, 0.42)
				* (em.w * em.w * 10.0 * ndl * vis / max(d2, 1.0));
	}

	vec3 acc = vec3(0.0);
	for (int k = 0; k < 4; k++) {
		float seed = claudeRadianceFrame * 4.0 + float(k) + f * 31.7
				+ texel.x * 7.3 + texel.y * 3.1;
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

	float aUp = 0.25;
	float aDown = 0.5;
	float goingDown = dot(fresh, vec3(1.0)) < dot(old.rgb, vec3(1.0)) ? 1.0 : 0.0;
	vec3 outc = old.a > 0.5
			? mix(old.rgb, fresh, mix(aUp, aDown, goingDown))
			: fresh;
	gl_FragColor = vec4(outc, 1.0);
}
