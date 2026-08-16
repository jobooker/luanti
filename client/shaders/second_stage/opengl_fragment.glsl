#define rendered texture0
#define bloom texture1
#define depthmap texture3

#ifdef GL_ES
// Dithering requires sufficient floating-point precision
#ifndef GL_FRAGMENT_PRECISION_HIGH
#undef ENABLE_DITHERING
#endif
#endif

struct ExposureParams {
	float compensationFactor;
};

uniform sampler2D rendered;
uniform sampler2D bloom;

uniform vec2 texelSize0;

uniform ExposureParams exposureParams;
uniform lowp float bloomIntensity;
uniform lowp float saturation;
uniform lowp float dayNightRatio;
uniform lowp float goldenHourStrength = 0.0;
uniform sampler2D depthmap;
uniform lowp float ssaoStrength = 0.0;

uniform lowp float gridDebug = 0.0;
uniform sampler3D claudeTraceGrid;
#if __VERSION__ >= 130
#define texture3D texture
#endif
uniform vec3 gridCamPos;   // camera in grid-local node units
uniform vec3 gridCamFwd;   // unit look direction
uniform vec3 gridCamRight; // camera right, pre-scaled by tan(fovX/2)
uniform vec3 gridCamUp;    // camera up, pre-scaled by tan(fovY/2)
uniform vec3 volumeSunDir;   // unit direction toward the sun
uniform vec2 volumeDepthRange; // camera near/far (world BS units)
uniform lowp float waterReflStrength = 0.0;
uniform lowp float giStrength = 0.0;
uniform lowp float giSplit = 0.0; // 1 = relight right half only (A/B seam)
uniform lowp float clayStrength = 0.0; // blend toward flat per-block color

// Shadow ray: second DDA march from a hit point toward the sun. Starts in
// the empty cell the primary ray hit from (caller nudges the origin out
// along the face normal) and tests only after the first step, so the
// surface never shadows itself. Leaving the grid = reached open sky.
float gridShadow(vec3 ro)
{
	const float S = 128.0;
	vec3 rd = volumeSunDir;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	for (int i = 0; i < 192; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			sideDist.x += invRd.x; cell.x += stepDir.x;
		} else if (sideDist.y < sideDist.z) {
			sideDist.y += invRd.y; cell.y += stepDir.y;
		} else {
			sideDist.z += invRd.z; cell.z += stepDir.z;
		}
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(S))))
			return 1.0;
		if (texture3D(claudeTraceGrid, (cell + 0.5) / S).a > 0.25)
			return 0.45;
	}
	return 1.0;
}

// Reflection march: DDA from just above a water surface, returning the
// shaded hit color (face + traced sun shadow) or a day-scaled sky
// gradient on miss. Advances before sampling so the origin cell is skipped.
vec3 gridReflect(vec3 ro, vec3 rd)
{
	const float S = 128.0;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	float t = 0.0;
	int axis = -1;
	for (int i = 0; i < 256; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x; sideDist.x += invRd.x; cell.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y; sideDist.y += invRd.y; cell.y += stepDir.y; axis = 1;
		} else {
			t = sideDist.z; sideDist.z += invRd.z; cell.z += stepDir.z; axis = 2;
		}
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(S))))
			break;
		vec4 s = texture3D(claudeTraceGrid, (cell + 0.5) / S);
		if (s.a > 0.25) {
			float face = axis == 1 ? (rd.y < 0.0 ? 1.0 : 0.45)
					: (axis == 0 ? 0.8 : 0.62);
			vec3 n = vec3(0.0);
			if (axis == 0) n.x = -stepDir.x;
			else if (axis == 1) n.y = -stepDir.y;
			else n.z = -stepDir.z;
			float shade = gridShadow(ro + rd * t + n * 0.01);
			// grid colors are full-bright; light the mirrored world
			// like the real one or night water reflects a daylit phantom
			return s.rgb * face * shade * clamp(dayNightRatio, 0.06, 1.0);
		}
	}
	float up = clamp(rd.y, 0.0, 1.0);
	vec3 sky = mix(vec3(0.70, 0.78, 0.86), vec3(0.28, 0.48, 0.80), up);
	return sky * clamp(dayNightRatio, 0.06, 1.0);
}

// Phase 2 v0 hemisphere GI. Light arriving from one direction: a short
// cone-ray march — sky light if it escapes, distance-weighted bounce
// color if it hits. Deliberately short range (20 cells): GI is about
// nearby geometry, and short rays keep the per-pixel cost bounded.
vec3 giSky(vec3 rd)
{
	float up = clamp(rd.y, 0.0, 1.0);
	vec3 sky = mix(vec3(0.70, 0.78, 0.86), vec3(0.28, 0.48, 0.80), up);
	// Directional sun glow: escaping rays aligned with the sun carry
	// direct warmth. Under canopy this makes block-level gaps into
	// dappled bright pools instead of one uniform dimness — the depth
	// cue dense tree cover was missing.
	float sunAmt = pow(max(dot(rd, volumeSunDir), 0.0), 6.0);
	sky += vec3(1.0, 0.9, 0.7) * sunAmt * 2.0;
	return sky * clamp(dayNightRatio, 0.06, 1.0);
}

// Short sun-visibility probe used from GI-ray hit points (64 cells is
// plenty for canopy scale; cheaper than the full shadow march).
float gridSunVis(vec3 ro)
{
	const float S = 128.0;
	vec3 rd = volumeSunDir;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
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
		if (texture3D(claudeTraceGrid, (cell + 0.5) / S).a > 0.25)
			return 0.0;
	}
	// unresolved after 144 cells: assume blocked, not lit — a timeout
	// credited as sunlight is how caves end up glowing
	return 0.0;
}

vec3 giTrace(vec3 ro, vec3 rd)
{
	const float S = 128.0;
	const int STEPS = 96;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	float t = 0.0;
	int axis = -1;
	for (int i = 0; i < STEPS; i++) {
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x; sideDist.x += invRd.x; cell.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y; sideDist.y += invRd.y; cell.y += stepDir.y; axis = 1;
		} else {
			t = sideDist.z; sideDist.z += invRd.z; cell.z += stepDir.z; axis = 2;
		}
		if (any(lessThan(cell, vec3(0.0))) || any(greaterThanEqual(cell, vec3(S)))) {
			// left through the bottom = deep underground, not sky
			if (cell.y < 0.0)
				return vec3(0.01);
			return giSky(rd);
		}
		vec4 s = texture3D(claudeTraceGrid, (cell + 0.5) / S);
		if (s.a > 0.25) {
			// Real forest bounce comes almost entirely from SUNLIT
			// surfaces — pools of sun on the floor re-radiating onto
			// trunks. Probe the hit point's sun visibility: lit hits
			// radiate warm and strong, shadowed hits barely leak.
			vec3 n = vec3(0.0);
			if (axis == 0) n.x = -stepDir.x;
			else if (axis == 1) n.y = -stepDir.y;
			else n.z = -stepDir.z;
			float sunlit = axis < 0 ? 0.0
					: gridSunVis(ro + rd * t + n * 0.01);
			float fall = 1.0 - t / float(STEPS);
			vec3 warm = mix(vec3(1.0), vec3(1.05, 0.93, 0.78), sunlit);
			return s.rgb * warm * (0.18 + 1.1 * sunlit) * fall
					* clamp(dayNightRatio, 0.06, 1.0);
		}
	}
	// ran out of steps while still enclosed: darkness, NOT sky — the
	// timeout-pays-skylight shortcut is why caves glowed
	return vec3(0.02);
}

// claude_grid ghost-depth view: one DDA ray per pixel (Amanatides & Woo)
// through the 128^3 occupancy snapshot. Voxel i spans [i, i+1) in cell
// space; node centers sit at integer grid-local coords, hence the +0.5
// shift on the ray origin. Shading: face brightness by hit axis (sun-from-
// above convention) times distance fog — geometry only, no lighting.
vec4 ghostView(vec2 uv)
{
	const float S = 128.0;
	vec2 ndc = uv * 2.0 - 1.0;
	vec3 rd = normalize(gridCamFwd + ndc.x * gridCamRight + ndc.y * gridCamUp);
	vec3 ro = gridCamPos + 0.5;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	float t = 0.0;
	int axis = -1; // axis of the last step = hit-face normal; -1 = ray origin cell
	for (int i = 0; i < 384; i++) {
		if (all(greaterThanEqual(cell, vec3(0.0))) && all(lessThan(cell, vec3(S)))) {
			// rgb = node average color; a: 0 air, ~0.5 water, 1 solid
			vec4 s = texture3D(claudeTraceGrid, (cell + 0.5) / S);
			if (s.a > 0.25) {
				float face = axis == 1 ? (rd.y < 0.0 ? 1.0 : 0.45)
						: (axis == 0 ? 0.8 : 0.62);
				float shade = 1.0;
				if (axis < 0) {
					face = 0.9;
				} else if (gridDebug < 1.5) {
					// nudge off the hit face, then trace toward the sun
					// (claude_grid_debug = 2 skips this: A/B compare)
					vec3 n = vec3(0.0);
					if (axis == 0) n.x = -stepDir.x;
					else if (axis == 1) n.y = -stepDir.y;
					else n.z = -stepDir.z;
					shade = gridShadow(ro + rd * t + n * 0.01);
				}
				// gentle: colors+shadows carry depth now; the old 0.015
				// clay-ghost fog crushed everything past ~70 cells
				float fog = exp(-t * 0.004);
				return vec4(s.rgb * face * shade * fog, 1.0);
			}
		} else if (i > 0) {
			break; // left the grid
		}
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x; sideDist.x += invRd.x; cell.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y; sideDist.y += invRd.y; cell.y += stepDir.y; axis = 1;
		} else {
			t = sideDist.z; sideDist.z += invRd.z; cell.z += stepDir.z; axis = 2;
		}
	}
	return vec4(0.0, 0.0, 0.04, 1.0); // miss: near-black, blue tint = "sky"
}

// ============ PURE PATH VIEW (claude_grid_debug = 3) ============
// 100% ray-traced illumination, zero ambient: every unit of brightness
// arrives via an explicit ray path — direct sun (N.L x traced visibility),
// sky dome (hemisphere rays paying ONLY on genuine sky exit), and one
// bounce (hit surfaces re-radiate only what the sun actually gives them).
// No face-brightness constants, no leak terms, no day-night floors.
// Blocks are exploited for geometry (exact DDA hits, exact face normals),
// never for lighting.

uniform vec3 volumeLightCol; // active light source color (sun/moon/none)

// The engine's dayNightRatio never drops below ~0.175 — a baked-in
// ambient floor. Remap it to a true 0..1 so traced midnight is dark.
float pathDayLin()
{
	return clamp((dayNightRatio - 0.18) / 0.82, 0.0, 1.0);
}

// Dark-material floor: MTG foliage textures average near-black, which in
// a pure multiply renderer eats all incident light ("low albedo is the
// work" — John's diagnosis, confirmed by the mode-4 A/B). Lift only very
// dark albedos, preserving hue.
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
	// gradient scales with daylight; disk/halo + scatter carry the
	// active light source's color (warm sun by day, cool moon by night)
	vec3 c = sky * pathDayLin();
	// crisp disc (sun face by day, full moon by night) + glow + halo
	float disc = smoothstep(0.9993, 0.9997, cosSun);
	c += volumeLightCol * (disc * 40.0
			+ pow(cosSun, 48.0) * 3.0 + pow(cosSun, 8.0) * 0.4);
	c += volumeLightCol * 0.18; // light scattered across the sky dome
	return c;
}

// Radiance arriving at ro from direction rd: sky if the ray truly exits,
// sun-lit bounce if it hits a surface the sun reaches, darkness otherwise.
vec3 pathRay(vec3 ro, vec3 rd)
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
				return vec3(0.0); // exited underground: no light there
			return pathSkyRadiance(rd);
		}
		vec4 s = texture3D(claudeTraceGrid, (cell + 0.5) / S);
		if (s.a > 0.25) {
			vec3 n = vec3(0.0);
			if (axis == 0) n.x = -stepDir.x;
			else if (axis == 1) n.y = -stepDir.y;
			else n.z = -stepDir.z;
			float ndl = max(dot(n, volumeSunDir), 0.0);
			if (ndl <= 0.0)
				return vec3(0.0); // faces away from sun: radiates nothing
			float sv = gridSunVis(ro + rd * t + n * 0.01);
			// albedo x incident light, attenuated by distance falloff
			float fall = 1.0 - t / 160.0;
			return pathAlbedo(s.rgb) * ndl * sv * fall
					* volumeLightCol * 1.4;
		}
	}
	return vec3(0.0); // unresolved while enclosed: darkness
}

vec4 pathView(vec2 uv)
{
	const float S = 128.0;
	vec2 ndc = uv * 2.0 - 1.0;
	vec3 rd = normalize(gridCamFwd + ndc.x * gridCamRight + ndc.y * gridCamUp);
	vec3 ro = gridCamPos + 0.5;
	vec3 cell = floor(ro);
	vec3 stepDir = sign(rd);
	vec3 invRd = 1.0 / max(abs(rd), vec3(1e-6));
	vec3 sideDist = (stepDir * (cell - ro) + stepDir * 0.5 + 0.5) * invRd;
	float t = 0.0;
	int axis = -1;
	for (int i = 0; i < 384; i++) {
		if (all(greaterThanEqual(cell, vec3(0.0))) && all(lessThan(cell, vec3(S)))) {
			vec4 s = texture3D(claudeTraceGrid, (cell + 0.5) / S);
			if (s.a > 0.25 && axis >= 0) {
				vec3 n = vec3(0.0);
				if (axis == 0) n.x = -stepDir.x;
				else if (axis == 1) n.y = -stepDir.y;
				else n.z = -stepDir.z;
				vec3 hp = ro + rd * t + n * 0.01;
				// mode 4: lighting-only — neutral albedo shows pure
				// transport, isolating material darkness from light bugs
				vec3 albedo = gridDebug > 3.5
						? vec3(0.55) : pathAlbedo(s.rgb);
				// direct light: sun by day, moon by night (color and
				// intensity arrive via volumeLightCol)
				float ndl = max(dot(n, volumeSunDir), 0.0);
				vec3 direct = vec3(0.0);
				if (ndl > 0.0)
					direct = vec3(ndl * gridSunVis(hp))
							* volumeLightCol;
				// sky + bounce: five hemisphere rays, cosine-biased
				vec3 t1 = normalize(cross(n,
						abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0)
								: vec3(1.0, 0.0, 0.0)));
				vec3 t2 = cross(n, t1);
				vec3 amb = pathRay(hp, n);
				amb += pathRay(hp, normalize(n * 0.65 + t1 * 0.75));
				amb += pathRay(hp, normalize(n * 0.65 - t1 * 0.75));
				amb += pathRay(hp, normalize(n * 0.65 + t2 * 0.75));
				amb += pathRay(hp, normalize(n * 0.65 - t2 * 0.75));
				amb *= 0.2;
				vec3 c = albedo * (direct + amb);
				// water surfaces mirror: one traced reflection ray plus
				// a specular sun glint — still 100% ray-earned light
				if (s.a < 0.75 && n.y > 0.5) {
					vec3 rr = reflect(rd, vec3(0.0, 1.0, 0.0));
					vec3 refl = pathRay(hp, rr);
					refl += volumeLightCol
							* pow(max(dot(rr, volumeSunDir), 0.0), 64.0)
							* 2.5;
					c = mix(c, refl, 0.65);
				}
				return vec4(pow(max(c, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
			}
		} else if (i > 0) {
			// primary ray left the grid: show the sky itself
			return vec4(pow(pathSkyRadiance(rd), vec3(1.0 / 2.2)), 1.0);
		}
		if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
			t = sideDist.x; sideDist.x += invRd.x; cell.x += stepDir.x; axis = 0;
		} else if (sideDist.y < sideDist.z) {
			t = sideDist.y; sideDist.y += invRd.y; cell.y += stepDir.y; axis = 1;
		} else {
			t = sideDist.z; sideDist.z += invRd.z; cell.z += stepDir.z; axis = 2;
		}
	}
	return vec4(0.0, 0.0, 0.0, 1.0);
}

// Cheap single-pass SSAO from the depth buffer alone: spiral taps around
// each pixel; nearer samples within a depth window count as occluders.
// Thresholds scale with (1 - depth) to roughly compensate for the
// nonlinear depth distribution. Sky (depth ~1) is excluded.
float sampleAO(vec2 uv, float d0)
{
	if (d0 >= 0.9999)
		return 0.0;
	float zscale = max(1.0 - d0, 1e-4);
	float ao = 0.0;
	const int N = 12;
	for (int i = 0; i < N; i++) {
		float a = float(i) * 2.39996; // golden angle spiral
		float r = 12.0 * (float(i) + 0.5) / float(N);
		vec2 offs = vec2(cos(a), sin(a)) * r * texelSize0;
		float ds = texture2D(depthmap, uv + offs).r;
		float diff = d0 - ds; // > 0: sample is closer -> potential occluder
		float t1 = 0.0004 * zscale;
		float t2 = 0.02 * zscale;
		ao += smoothstep(t1 * 0.25, t1, diff) * (1.0 - smoothstep(t2 * 0.5, t2, diff));
	}
	return ao / float(N);
}

CENTROID_ VARYING_ mediump vec2 varTexCoord;

#ifdef ENABLE_AUTO_EXPOSURE
VARYING_ float exposure; // linear exposure factor, see vertex shader
#endif

#ifdef ENABLE_BLOOM

vec4 applyBloom(vec4 color, vec2 uv)
{
	vec3 light = texture2D(bloom, uv).rgb;
#ifdef ENABLE_BLOOM_DEBUG
	if (uv.x > 0.5 && uv.y < 0.5)
		return vec4(light, color.a);
	if (uv.x < 0.5)
		return color;
#endif
	color.rgb = mix(color.rgb, light, bloomIntensity);
	return color;
}

#endif

#if ENABLE_TONE_MAPPING

/* Hable's UC2 Tone mapping parameters
	A = 0.22;
	B = 0.30;
	C = 0.10;
	D = 0.20;
	E = 0.01;
	F = 0.30;
	W = 11.2;
	equation used:  ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
*/

// highp for GLES, see <https://github.com/luanti-org/luanti/pull/14688>
highp vec3 uncharted2Tonemap(highp vec3 x)
{
	return ((x * (0.22 * x + 0.03) + 0.002) / (x * (0.22 * x + 0.3) + 0.06)) - 0.03333;
}

vec4 applyToneMapping(vec4 color)
{
	color = vec4(pow(color.rgb, vec3(2.2)), color.a);
	const float gamma = 1.6;
	const float exposureBias = 5.5;
	color.rgb = uncharted2Tonemap(exposureBias * color.rgb);
	// Precalculated white_scale from
	//vec3 whiteScale = 1.0 / uncharted2Tonemap(vec3(W));
	vec3 whiteScale = vec3(1.036015346);
	color.rgb *= whiteScale;
	return vec4(pow(color.rgb, vec3(1.0 / gamma)), color.a);
}
#endif

vec3 applySaturation(vec3 color, float factor)
{
	// Calculate the perceived luminosity from the RGB color.
	// See also: https://www.w3.org/WAI/GL/wiki/Relative_luminance
	float brightness = dot(color, vec3(0.2125, 0.7154, 0.0721));
	return mix(vec3(brightness), color, factor);
}

#ifdef ENABLE_DITHERING
// From http://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
// and https://www.shadertoy.com/view/MslGR8 (5th one starting from the bottom)
// NOTE: `frag_coord` is in pixels (i.e. not normalized UV).
vec3 screen_space_dither(highp vec2 frag_coord) {
	// Iestyn's RGB dither (7 asm instructions) from Portal 2 X360, slightly modified for VR.
	highp vec3 dither = vec3(dot(vec2(171.0, 231.0), frag_coord));
	dither.rgb = fract(dither.rgb / vec3(103.0, 71.0, 97.0));

	// Subtract 0.5 to avoid slightly brightening the whole viewport.
	return (dither.rgb - 0.5) / 255.0;
}
#endif

void main(void)
{
	vec2 uv = varTexCoord.st;

	// claude_grid_debug: 1/2 = ghost view here; 3/4 (path-traced) are
	// produced by the claude_accum/claude_present steps downstream
	// The ghost view is a DEBUG-only path (claude_grid_debug 1/2). Its guard
	// reads the gridDebug uniform, which is not reliably delivered to this
	// shader on a core profile — an unset uniform reads 0 on GL 2.1 (working by
	// luck) but not necessarily on core, so the branch fired every frame and
	// returned a flat image. That single line is what made post-processing look
	// like it broke terrain rendering on 4.1. Compile it out unless explicitly
	// asked for; the traced modes (3/4) are produced downstream regardless.
#ifdef ENABLE_GHOST_VIEW
	if (gridDebug > 0.5 && gridDebug < 2.5) {
		gl_FragColor = ghostView(uv);
		return;
	}
#endif

#ifdef ENABLE_SSAA
	vec4 color = vec4(0.);
	for (float dx = 1.; dx < SSAA_SCALE; dx += 2.)
	for (float dy = 1.; dy < SSAA_SCALE; dy += 2.)
		color += texture2D(rendered, uv + texelSize0 * vec2(dx, dy)).rgba;
	color /= SSAA_SCALE * SSAA_SCALE / 4.;
#else
	vec4 color = texture2D(rendered, uv).rgba;
#endif

	// translate to linear colorspace (approximate)
	color.rgb = pow(color.rgb, vec3(2.2));

	// Traced lighting in the real render (claude_water_reflections +
	// claude_gi): reconstruct this pixel's position from the depth buffer
	// once, then let each effect consult the grid.
	if ((waterReflStrength > 0.0 || giStrength > 0.0) && gridDebug < 0.5) {
		float dw = texture2D(depthmap, uv).r;
		if (dw < 0.9999) {
			vec2 ndcw = uv * 2.0 - 1.0;
			vec3 vdir = gridCamFwd + ndcw.x * gridCamRight + ndcw.y * gridCamUp;
			float zn = volumeDepthRange.x;
			float zf = volumeDepthRange.y;
			float ez = 2.0 * zn * zf / (zf + zn - (2.0 * dw - 1.0) * (zf - zn));
			// BS = 10: eye depth is in world units, the grid in nodes
			vec3 p = gridCamPos + 0.5 + vdir * (ez / 10.0);
			vec3 wcell = floor(p - vec3(0.0, 0.05, 0.0));
			bool inVol = all(greaterThanEqual(wcell, vec3(0.0)))
					&& all(lessThan(wcell, vec3(128.0)));
			bool isWater = false;

			// Clay mode (mid strengths only): blend the textured surface
			// toward its flat per-block color. Pure mode (>= 0.95) paints
			// albedo inside the lit branch instead — painting it here
			// without guaranteed lighting made distant pixels (where
			// depth-reconstruction noise misses the surface cell) render
			// full-bright unlit: the 'lit cave' bug.
			if (inVol && clayStrength > 0.0
					&& (giSplit < 0.5 || uv.x > 0.5)) {
				vec3 ccell = floor(p + normalize(vdir) * 0.1
						- vec3(0.0, 0.02, 0.0));
				if (all(greaterThanEqual(ccell, vec3(0.0)))
						&& all(lessThan(ccell, vec3(128.0)))) {
					vec4 cv = texture3D(claudeTraceGrid, (ccell + 0.5) / 128.0);
					if (cv.a > 0.25) {
						if (clayStrength > 0.95) {
							// stock-lighting comparison mode: keep the
							// raster's light, swap texture for flat block
							// color — light estimated as pixel luminance
							// over known material luminance
							vec3 alb = pathAlbedo(cv.rgb);
							float pl = dot(color.rgb,
									vec3(0.2126, 0.7152, 0.0722));
							float al = dot(alb,
									vec3(0.2126, 0.7152, 0.0722));
							float lightEst = clamp(pl / max(al, 0.03),
									0.0, 1.5);
							color.rgb = alb * lightEst;
						} else {
							color.rgb = mix(color.rgb,
									pow(cv.rgb, vec3(2.2)), clayStrength);
						}
					}
				}
			}

			// Water reflections: if this pixel is a tagged water cell,
			// reflect the view ray about +Y and march it.
			if (inVol && waterReflStrength > 0.0) {
				vec4 wv = texture3D(claudeTraceGrid, (wcell + 0.5) / 128.0);
				if (wv.a > 0.25 && wv.a < 0.75) {
					isWater = true;
					vec3 vn = normalize(vdir);
					vec3 rdir = reflect(vn, vec3(0.0, 1.0, 0.0));
					vec3 ro2 = vec3(p.x, wcell.y + 1.001, p.z);
					vec3 refl = gridReflect(ro2, rdir);
					float fres = pow(1.0 - clamp(-vn.y, 0.0, 1.0), 2.0);
					float k = waterReflStrength * (0.25 + 0.55 * fres);
					color.rgb = mix(color.rgb, pow(refl, vec3(2.2)), k);
				}
			}

			// Hemisphere GI: normal from screen-space depth gradients,
			// 4 cone rays; modulate against the open-sky baseline so
			// unoccluded ground is unchanged, overhangs darken, and lit
			// colored surfaces bleed onto neighbors.
			// Voxel-exact geometry: the surface cell is just inside the
			// point along the view ray, and the normal is whichever face
			// of that cell the point sits on. No screen-space derivatives
			// (their 2px garbage fringe at every silhouette was the
			// 'flittering artifacts'), and entity pixels skip naturally
			// because their containing cell is air.
			vec3 vn2 = normalize(vdir);
			// 0.3-node inward probe: depth reconstruction wobbles a few
			// tenths of a node at range; a shallow probe misses the
			// surface cell and drops pixels out of the lit path
			vec3 scell = floor(p + vn2 * 0.3);
			bool onSurface = false;
			vec3 nrm = vec3(0.0, 1.0, 0.0);
			if (inVol && giStrength > 0.0 && !isWater) {
				vec4 sv = texture3D(claudeTraceGrid, (scell + 0.5) / 128.0);
				if (sv.a > 0.25) {
					onSurface = true;
					vec3 q = p - (scell + 0.5);
					vec3 aq = abs(q);
					if (aq.x > aq.y && aq.x > aq.z)
						nrm = vec3(sign(q.x), 0.0, 0.0);
					else if (aq.y > aq.z)
						nrm = vec3(0.0, sign(q.y), 0.0);
					else
						nrm = vec3(0.0, 0.0, sign(q.z));
				}
			}
			if (onSurface && (giSplit < 0.5 || uv.x > 0.5)) {
				vec3 t1 = normalize(cross(nrm,
						abs(nrm.y) < 0.9 ? vec3(0.0, 1.0, 0.0)
								: vec3(1.0, 0.0, 0.0)));
				vec3 t2 = cross(nrm, t1);
				// 0.6-cell bias: depth-reconstructed positions carry
				// view-dependent noise; a small offset leaves ray origins
				// flickering in/out of the surface cell (shadow acne
				// crawling as the camera moves). Stay above the noise.
				vec3 ro3 = p + nrm * 0.6;
				vec3 d1 = normalize(nrm * 0.8 + t1 * 0.6);
				vec3 d2 = normalize(nrm * 0.8 - t1 * 0.6);
				vec3 d3 = normalize(nrm * 0.8 + t2 * 0.6);
				vec3 d4 = normalize(nrm * 0.8 - t2 * 0.6);
				// 5th ray biased toward the sky: vertical walls otherwise
				// fire only sideways and miss open sky overhead, making
				// roofless trenches/courtyards as dark as caves
				vec3 d5 = normalize(nrm * 0.35 + vec3(0.0, 1.0, 0.0));
				vec3 gi = giTrace(ro3, d1) + giTrace(ro3, d2)
						+ giTrace(ro3, d3) + giTrace(ro3, d4)
						+ giTrace(ro3, d5);
				vec3 base = giSky(d1) + giSky(d2) + giSky(d3) + giSky(d4)
						+ giSky(d5);
				vec3 m = clamp(gi / max(base, vec3(1e-3)), 0.0, 1.5);
				// Directional sun: N.L diffuse + a traced shadow ray.
				// Luanti's face lighting has no sun-angle term at all
				// (every south face is as bright as every north face),
				// which is why forests read flat. Sun-side trunks
				// brighten, back sides fall dark, trunks cast traced
				// shadows on the ground. Fades out toward night so the
				// relight collapses to pure ambient after dusk.
				float dayL = clamp((dayNightRatio - 0.3) / 0.4, 0.0, 1.0);
				float ndl = max(dot(nrm, volumeSunDir), 0.0);
				// gridShadow's 0.45 floor suits the ghost view, but
				// for relighting it makes shadows read ~18% dimmer than
				// sun — imperceptible. Remap: shadowed = 12% of direct.
				float shraw = ndl > 0.02 ? gridShadow(ro3) : 0.0;
				float direct = ndl * (shraw > 0.9 ? 1.0 : 0.12);
				vec3 relight = m * mix(1.0, 0.45, dayL)
						+ vec3(direct * 0.8 * dayL);
				// Mild compression: the raster already encodes occlusion,
				// so a raw multiply crushes interiors — but too much
				// compression (0.55 tried) flattens the whole effect.
				relight = pow(clamp(relight, 0.0, 1.6), vec3(0.7));
				color.rgb *= mix(vec3(1.0), relight, giStrength);

				// PURE TRACED MODE (clay >= 0.95): discard the raster
				// lighting entirely — pixel = block albedo x traced
				// light (hemisphere ambient + traced sun), the ghost's
				// math in the live game. No blend, no compression: what
				// the rays actually see. Unlit interiors go black until
				// Phase 3 emissives — that's real tracing being honest.
				if (clayStrength > 0.95) {
					// albedo from the SAME cell the lighting used —
					// keeps 'painted' and 'lit' inseparable
					vec3 albedo = pow(texture3D(claudeTraceGrid,
							(scell + 0.5) / 128.0).rgb, vec3(2.2));
					vec3 sunCol = vec3(1.06, 0.96, 0.82);
					color.rgb = albedo * (gi * 0.26
							+ vec3(direct) * sunCol * dayL);
				}
			}
		}
	}

	// SSAO: darken creases before exposure/bloom so glow stays clean
	if (ssaoStrength > 0.0) {
		float ao = sampleAO(uv, texture2D(depthmap, uv).r);
		color.rgb *= 1.0 - ssaoStrength * 0.7 * ao;
	}

#ifdef ENABLE_BLOOM_DEBUG
	if (uv.x > 0.5 || uv.y > 0.5)
#endif
	{
		color.rgb *= exposureParams.compensationFactor;
#ifdef ENABLE_AUTO_EXPOSURE
		color.rgb *= exposure;
#endif
	}

#ifdef ENABLE_BLOOM
	color = applyBloom(color, uv);
#endif


	color.rgb = clamp(color.rgb, vec3(0.), vec3(1.));

	// return to sRGB colorspace (approximate)
	color.rgb = pow(color.rgb, vec3(1.0 / 2.2));

#ifdef ENABLE_BLOOM_DEBUG
	if (uv.x > 0.5 || uv.y > 0.5)
#endif
	{
#if ENABLE_TONE_MAPPING
		color = applyToneMapping(color);
#endif

		color.rgb = applySaturation(color.rgb, saturation);

		// Golden hour: warm the whole frame through dawn/dusk transitions,
		// keyed to the engine's day-night ratio. Zero at full day and full
		// night; golden_hour_strength setting scales it (0 disables).
		if (goldenHourStrength > 0.0) {
			float g = smoothstep(0.30, 0.60, dayNightRatio)
				* (1.0 - smoothstep(0.85, 0.97, dayNightRatio));
			vec3 warm = vec3(1.12, 1.03, 0.88);
			color.rgb *= mix(vec3(1.0), warm, g * goldenHourStrength);
		}
	}

#ifdef ENABLE_DITHERING
	// Apply dithering just before quantisation
	color.rgb += screen_space_dither(gl_FragCoord.xy);
#endif

	gl_FragColor = vec4(color.rgb, 1.0); // force full alpha to avoid holes in the image.
}
