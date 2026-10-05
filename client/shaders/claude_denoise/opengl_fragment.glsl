// claude_denoise (2026-10-05): an edge-aware a-trous filter over the
// traced image, for DISPLAY ONLY. SVGF-family (Schied et al. 2017,
// "Spatiotemporal Variance-Guided Filtering"): estimate how noisy each
// pixel still is, then blur it only toward neighbours on the SAME
// surface whose brightness differs by less than that noise.
//
// WHAT MAKES IT SAFE TO HAVE ON. It reads the running average and never
// writes it: the tracer's history, and so every referee and every
// converged photo, is untouched. And it switches itself off as the image
// converges: the noise estimate is the variance of the AVERAGE (one
// sample's variance x historyMom.g, which falls as 1/N while parked), so
// a parked pixel's neighbours soon differ by many times its noise and
// stop being averaged in. What it never does is touch a pixel whose
// noise it cannot estimate as small — it only removes the part that is
// noise. claude_ci pins claude_denoise 0 regardless.
//
// SAME SURFACE IS EXACT HERE. Every surface in this world is an
// axis-aligned voxel face on a 1/16 m plane, and the tracer writes which
// one (gbuf.a); a tap on any other face has weight 0. No depth or normal
// tolerance to tune. 0 = sky, glass, water: never filtered.
//
// TEXTURE IS DIVIDED OUT FIRST (demodulation): the filter works on light
// arriving at the surface, and the accumulated albedo is multiplied back
// in at the end, so block textures stay sharp.
//
// Passes, by CLAUDE_DN_ITER (one shader, six compiles, secondstage.cpp):
//   0      read radiance + moments, write (irradiance, variance)
//   1..5   a-trous at step 1, 2, 4, 8, 16, write (irradiance, variance)
//   5      ...then albedo back in, primary distance in alpha (for
//          claude_present's upsample, which reads it as `accum`)
// With claude_denoise 0 (or a debug view, or traced mode off) every pass
// copies and pass 5 writes the raw running average itself: bit-exact.

#define src texture0
#define gbuf texture1
#define aux texture2      // pass 0: moments. Later: the raw running average
#define direct texture3
uniform sampler2D src;
uniform sampler2D gbuf;
uniform sampler2D aux;
uniform sampler2D direct;
uniform float claudeDenoise;
uniform float claudeView;
uniform lowp float gridDebug;
uniform vec2 texelSize0;      // this pass's input texel
CENTROID_ VARYING_ mediump vec2 varTexCoord;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
const float ALB_MIN = 0.005;  // claude_trace ALBEDO_FLOOR
// SVGF's luminance edge-stopping constant, as published (sigma_l = 4).
// LITERATURE VALUE, not fitted here | learn by: the RMS-vs-converged
// pricing in util/claude_denoise_price.py.
const float SIGMA_L = 4.0;
// Fewer samples than this behind a pixel (vfac > 1/4) and its own
// moments are too young to trust: estimate the variance from the
// same-face neighbourhood instead (SVGF's own fallback, 7x7).
const float VFAC_YOUNG = 0.25;

bool off()
{
	return claudeDenoise < 0.5 || claudeView > 0.5 || gridDebug < 2.5;
}

// May q stand in for p? Same face as p, and q itself a CLEAN pixel (not
// mixed: a mixed pixel's value is part another surface). p may be mixed:
// its face is the one it saw last, and its code carries a minus sign.
bool tapOK(float codeP, float codeQ)
{
	return codeQ > 0.5 && abs(codeQ - abs(codeP)) < 0.5;
}

bool inside(vec2 uv)
{
	return all(greaterThanEqual(uv, vec2(0.0)))
			&& all(lessThanEqual(uv, vec2(1.0)));
}

void main(void)
{
	vec2 uv = varTexCoord.st;
#if CLAUDE_DN_ITER == 0
	// ---- noise estimate -------------------------------------------------
	vec4 c = texture2D(src, uv);
	vec4 g = texture2D(gbuf, uv);
	if (off() || abs(g.a) < 0.5) {
		gl_FragColor = vec4(c.rgb, 0.0);
		return;
	}
	vec3 irr = c.rgb / max(g.rgb, vec3(ALB_MIN));
	float l = dot(irr, LUMA);
	vec4 m = texture2D(aux, uv);
	float var;
	if (m.g <= VFAC_YOUNG) {
		// one sample's variance (moments) x the share left in the average
		var = max(m.r - m.b * m.b, 0.0) * m.g;
	} else {
		// young history: the spread of the AVERAGES around it on the same
		// face, which is already the variance of what is displayed
		float s1 = 0.0, s2 = 0.0, n = 0.0;
		for (int dy = -3; dy <= 3; dy++)
		for (int dx = -3; dx <= 3; dx++) {
			vec2 q = uv + vec2(float(dx), float(dy)) * texelSize0;
			if (!inside(q))
				continue;
			vec4 gq = texture2D(gbuf, q);
			if (!(dx == 0 && dy == 0) && !tapOK(g.a, gq.a))
				continue;
			float lq = dot(texture2D(src, q).rgb / max(gq.rgb, vec3(ALB_MIN)),
					LUMA);
			s1 += lq;
			s2 += lq * lq;
			n += 1.0;
		}
		s1 /= n;
		var = max(s2 / n - s1 * s1, 0.0);
	}
	gl_FragColor = vec4(irr, var);
#else
	// ---- one a-trous step -----------------------------------------------
	const float STEP = float(1 << (CLAUDE_DN_ITER - 1));
	vec4 cp = texture2D(src, uv);
	vec4 gp = texture2D(gbuf, uv);
	vec4 raw = texture2D(aux, uv);
	bool skip = off() || abs(gp.a) < 0.5;
	vec4 outv = cp;
	if (!skip) {
		// the noise estimate, smoothed 3x3 on the same face first
		float vs = 0.0, vw = 0.0;
		for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++) {
			vec2 q = uv + vec2(float(dx), float(dy)) * texelSize0;
			if (!inside(q))
				continue;
			if (!(dx == 0 && dy == 0) && !tapOK(gp.a, texture2D(gbuf, q).a))
				continue;
			float k = (dx == 0 ? 0.5 : 0.25) * (dy == 0 ? 0.5 : 0.25);
			vs += texture2D(src, q).a * k;
			vw += k;
		}
		float sl = SIGMA_L * sqrt(max(vs / vw, 0.0)) + 1e-8;
		float lp = dot(cp.rgb, LUMA);
		float h[5];
		h[0] = 1.0 / 16.0; h[1] = 1.0 / 4.0; h[2] = 3.0 / 8.0;
		h[3] = 1.0 / 4.0; h[4] = 1.0 / 16.0;
		vec3 sum = vec3(0.0);
		float wsum = 0.0, vsum = 0.0;
		for (int dy = -2; dy <= 2; dy++)
		for (int dx = -2; dx <= 2; dx++) {
			vec2 q = uv + vec2(float(dx), float(dy)) * STEP * texelSize0;
			if (!inside(q))
				continue;
			if (!(dx == 0 && dy == 0) && !tapOK(gp.a, texture2D(gbuf, q).a))
				continue;
			vec4 cq = texture2D(src, q);
			float w = h[dx + 2] * h[dy + 2]
					* exp(-abs(dot(cq.rgb, LUMA) - lp) / sl);
			sum += cq.rgb * w;
			wsum += w;
			vsum += w * w * cq.a;
		}
		outv = vec4(sum / wsum, vsum / (wsum * wsum));
	}
#if CLAUDE_DN_ITER == 5
	// albedo back in; the primary distance where claude_present wants it
	if (skip)
		gl_FragColor = raw;
	else
		gl_FragColor = vec4(outv.rgb * max(gp.rgb, vec3(ALB_MIN)), raw.a);
#else
	gl_FragColor = outv;
#endif
#endif
}
