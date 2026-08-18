// claude_present: the pipeline's last word. Traced modes upsample the
// half-res accumulated buffer with a joint-bilateral filter: the 4
// nearest lighting texels are weighted by how well their traced hit
// distance agrees with the FULL-RES raster depth at this pixel, so
// silhouettes stay crisp while surfaces stay smooth. Raster modes pass
// the merged frame through.
#define merged texture0
#define accum texture1
#define depthmap texture2

uniform sampler2D merged;
uniform sampler2D accum;
uniform sampler2D depthmap;
uniform lowp float gridDebug;
// claude_view: 0 = photo, 1-5 = claude_trace's diagnostic views. Only
// used to choose the display transform below; the photo path is
// untouched.
uniform float claudeView;
uniform vec2 texelSize0;       // full-res texel (from merged)
uniform vec2 volumeDepthRange; // camera near/far, world BS units

CENTROID_ VARYING_ mediump vec2 varTexCoord;

void main(void)
{
	vec2 uv = varTexCoord.st;
	if (gridDebug < 2.5) {
		gl_FragColor = vec4(texture2D(merged, uv).rgb, 1.0);
		return;
	}

	// CLAUDE-DEBUG (temporary): pure-green 12px block in the bottom-left
	// proves the TRACED present path executed. A raster frame passed
	// through (gridDebug lost/reverted) scores scene-level stddev too,
	// which fooled the harness once (afb4168) — the marker cannot appear
	// on that path, so verdict = marker AND stddev.
	if (gl_FragCoord.x < 12.0 && gl_FragCoord.y < 12.0) {
		gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);
		return;
	}

	// full-res raster depth -> linear distance in node units.
	// Depth scale is 4096 (cascade hits reach ~900 nodes; the old 200
	// clamp classified all far terrain as sky and painted raster over it)
	// THE SKY BRANCH IS GONE (2026-08-17), and its absence is the proof
	// that the trace owns the sky.
	//
	// Until today this shader pasted the RASTER frame wherever raster
	// depth was empty and the traced ray had also escaped: the game's own
	// skybox — sun, clouds, stars, and Mineclonia's phase-correct moon —
	// composited over a renderer that returned black in those directions.
	// That was honest while claude_trace had no sky at all, and it was
	// written in the blood of the blue-sun incident (an analytic disc
	// drew the MOON as a blue SUN, because at night volumeSunDir and
	// volumeLightCol literally are the moon's and the disc reused the
	// sun's 40x multiplier).
	//
	// claude_trace now has skyRadiance(direction): one function, feeding
	// both the background a camera ray sees and the light a shadow ray
	// samples, and reading Luanti's real Sky — including that same moon
	// sprite. A composite here would be a SECOND sky, and the image would
	// be lit by one and painted with the other, which is exactly the
	// inconsistency physics-contract §4 forbids.
	//
	// What went with it: clouds and stars are no longer drawn. They were
	// never lights, and the punt list in claude_trace says so.
	//
	// depthmap and volumeDepthRange stay: they are the joint-bilateral
	// upsample's GUIDE, which is a different job. An empty depth leaves
	// the guide at 4096, which is exactly where a traced miss packs its
	// own distance, so a sky pixel's four taps agree perfectly.
	float d = texture2D(depthmap, uv).r;
	float zn = volumeDepthRange.x;
	float zf = volumeDepthRange.y;
	float guide = 4096.0;
	if (d < 0.9999) {
		float ez = 2.0 * zn * zf / (zf + zn - (2.0 * d - 1.0) * (zf - zn));
		guide = min(ez / 10.0, 4090.0); // BS = 10
	}

	// 4 nearest half-res texels, bilinear x depth-agreement weights
	vec2 ht = texelSize0 * 2.0;
	vec2 base = (floor(uv / ht - 0.5) + 0.5) * ht;
	vec2 f = clamp((uv - base) / ht, 0.0, 1.0);
	vec3 sum = vec3(0.0);
	float wsum = 0.0;
	for (int i = 0; i < 4; i++) {
		vec2 o = vec2(i == 1 || i == 3 ? 1.0 : 0.0,
				i == 2 || i == 3 ? 1.0 : 0.0);
		vec4 s = texture2D(accum, base + o * ht);
		float bw = (o.x > 0.5 ? f.x : 1.0 - f.x)
				* (o.y > 0.5 ? f.y : 1.0 - f.y);
		// depth-agreement weight, RELATIVE at range: at 800 nodes a
		// per-node penalty would zero every tap. 0.8%, not 2% — the
		// looser band blended across far cell edges and BLURRED the
		// whole cascade field ("softwarey")
		float dw = exp(-abs(s.a * 4096.0 - guide)
				/ max(1.7, 0.008 * guide));
		float w = bw * dw + 1e-5;
		sum += s.rgb * w;
		wsum += w;
	}
	vec3 c = sum / wsum;

	// DIAGNOSTIC VIEWS (claude_view 1-5) present LINEARLY. Their values
	// are the message: a 6-step gray normal ladder, a linear Le, a
	// log-scaled distance. ACES would compress the top of that ladder
	// into indistinguishable near-whites and the gamma would bend the
	// steps, so a wrong normal would stop reading as a wrong brightness.
	// View 6 (clay) is lit radiance and falls through to the photo
	// transform. The claude_view == 0 path below is unchanged, byte for
	// byte — the furnace and Cornell referees invert exactly that
	// transform.
	//
	// SO DO VIEWS 9/10/11 (roadmap 1a's direct-light referee), and that
	// is deliberate rather than an oversight: what they carry is linear
	// radiance, not a gray ladder, and the thing that judges them is
	// claude_cornell_check.py, which inverts EXACTLY the transform below
	// (aces_inverse then gamma) on the same five region boxes it uses for
	// every other Cornell frame. Presenting them linearly would need a
	// second decode path in the referee, i.e. a second thing to keep
	// honest, for no gain — the direct term in this room peaks around
	// 0.1 linear, nowhere near the ACES shoulder.
	// 7 and 8 are the DESCENT AUDIT (claude_trace, 2026-08-17): a rung
	// flag, two counters and a cell index. Encoded values, so linear.
	// 19 is the MATERIAL INDEX ROUND TRIP (2026-08-18): a pass/fail
	// screen of 256 columns plus the palette's emission ramp. Encoded
	// values, so linear — ACES would bend the ramp and squash the top of
	// it into indistinguishable near-whites, and the ramp is there to be
	// read as brightness.
	// 20 is THE INTERFACE LADDER (claude_trace, 2026-08-18): Fresnel
	// reflectance and sin(theta_t) drawn as brightness against incidence
	// angle. The values are the message and a referee inverts them
	// directly, so ACES must not touch them.
	if ((claudeView > 0.5 && claudeView < 5.5)
			|| (claudeView > 6.5 && claudeView < 8.5)
			|| (claudeView > 18.5 && claudeView < 20.5)) {
		gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
		return;
	}

	// accum is LINEAR radiance now; the display transform is the ONE
	// art knob (energy audit): ACES filmic fit (Narkowicz), then gamma
	vec3 lin = max(c, vec3(0.0));
	vec3 aces = clamp(lin * (2.51 * lin + 0.03)
			/ (lin * (2.43 * lin + 0.59) + 0.14), 0.0, 1.0);
	gl_FragColor = vec4(pow(aces, vec3(1.0 / 2.2)), 1.0);
}
