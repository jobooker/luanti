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
	return a;
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

// visibility toward the (jittered) light direction: 1 lit, 0 blocked
float lightVis(vec3 ro, vec3 sd)
{
	const float S = 128.0;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(sd);
	vec3 invRd = 1.0 / max(abs(sd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	for (int i = 0; i < 208; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			sideDist.x += invRd.x; cell.x += stepDir.x;
		} else if (sideDist.y < sideDist.z) {
			sideDist.y += invRd.y; cell.y += stepDir.y;
		} else {
			sideDist.z += invRd.z; cell.z += stepDir.z;
		}
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(S))))
			return 1.0;
		if (texture3D(claudeVolume, (cell + 0.5) / S).a > 0.25)
			return 0.0;
	}
	return 0.0;
}

// radiance arriving from direction rd: sky on genuine exit, sun-lit
// one-bounce on hit, darkness otherwise (sd = jittered light direction)
vec3 bounceRay(vec3 ro, vec3 rd, vec3 sd)
{
	const float S = 128.0;
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
			return pathSkyRadiance(rd);
		}
		vec4 s = texture3D(claudeVolume, (cell + 0.5) / S);
		if (s.a > 0.25) {
			vec3 n = vec3(0.0);
			if (axis == 0) n.x = -stepDir.x;
			else if (axis == 1) n.y = -stepDir.y;
			else n.z = -stepDir.z;
			float ndl = max(dot(n, sd), 0.0);
			if (ndl <= 0.0)
				return vec3(0.0);
			float sv = lightVis(ro + rd * t + n * 0.01, sd);
			float fall = 1.0 - t / 160.0;
			return pathAlbedo(s.rgb) * ndl * sv * fall
					* volumeLightCol * 1.4;
		}
	}
	return vec3(0.0);
}

void main(void)
{
	vec2 uv = varTexCoord.st;
	vec3 prev = texture2D(history, uv).rgb;
	if (volumeDebug < 2.5) {
		// traced mode off: carry history through untouched
		gl_FragColor = vec4(prev, 1.0);
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
	bool done = false;

	for (int i = 0; i < 384; i++) {
		if (all(greaterThanEqual(cell, vec3(0.0))) && all(lessThan(cell, vec3(S)))) {
			vec4 s = texture3D(claudeVolume, (cell + 0.5) / S);
			if (s.a > 0.25 && axis >= 0) {
				vec3 n = vec3(0.0);
				if (axis == 0) n.x = -stepDir.x;
				else if (axis == 1) n.y = -stepDir.y;
				else n.z = -stepDir.z;
				vec3 hp = ro + rd * t + n * 0.01;
				vec3 albedo = volumeDebug > 3.5
						? vec3(0.55) : pathAlbedo(s.rgb);

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

				fresh = albedo * (direct + amb);

				// mirror water: one traced reflection + jittered glint
				if (s.a < 0.75 && n.y > 0.5) {
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

	// accumulate in gamma space (RGBA8 history: better dark precision)
	vec3 fresh_g = pow(max(fresh, vec3(0.0)), vec3(1.0 / 2.2));
	gl_FragColor = vec4(mix(prev, fresh_g, accumAlpha), 1.0);
}
