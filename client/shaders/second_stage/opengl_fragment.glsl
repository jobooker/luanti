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
uniform lowp float goldenHourStrength;
uniform sampler2D depthmap;
uniform lowp float ssaoStrength;

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
