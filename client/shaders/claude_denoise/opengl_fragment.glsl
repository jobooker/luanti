// claude_denoise: edge-aware spatial pass over the accumulated traced
// buffer (Teardown-style 12-tap Poisson bilateral). Runs on the DISPLAY
// path only — the accumulation history stays raw, so filtering never
// compounds frame over frame. Depth lives in the alpha channel (t/200);
// taps from a different depth get near-zero weight, so lighting smooths
// within surfaces but never bleeds across silhouettes.
#define src texture0

uniform sampler2D src;
uniform lowp float claudeDenoise;
uniform vec2 texelSize0;
uniform lowp float volumeDebug;

CENTROID_ VARYING_ mediump vec2 varTexCoord;

void main(void)
{
	if (claudeDenoise < 0.5) { // raw-vs-denoised A/B (John's bisect)
		gl_FragColor = texture2D(texture0, varTexCoord.st);
		return;
	}
	vec2 uv = varTexCoord.st;
	vec4 c = texture2D(src, uv);
	if (volumeDebug < 2.5) {
		gl_FragColor = c;
		return;
	}
	float t0 = c.a;
	vec3 sum = c.rgb;
	float wsum = 1.0;
	vec2 P0 = vec2(-0.326, -0.406);
	vec2 P1 = vec2(-0.840, -0.074);
	vec2 P2 = vec2(-0.696, 0.457);
	vec2 P3 = vec2(-0.203, 0.621);
	vec2 P4 = vec2(0.962, -0.195);
	vec2 P5 = vec2(0.473, -0.480);
	vec2 P6 = vec2(0.519, 0.767);
	vec2 P7 = vec2(0.185, -0.893);
	vec2 P8 = vec2(0.507, 0.064);
	vec2 P9 = vec2(0.896, 0.412);
	vec2 P10 = vec2(-0.322, -0.933);
	vec2 P11 = vec2(-0.792, -0.598);
	float R = 2.0;
	for (int i = 0; i < 12; i++) {
		vec2 o;
		if (i == 0) o = P0; else if (i == 1) o = P1;
		else if (i == 2) o = P2; else if (i == 3) o = P3;
		else if (i == 4) o = P4; else if (i == 5) o = P5;
		else if (i == 6) o = P6; else if (i == 7) o = P7;
		else if (i == 8) o = P8; else if (i == 9) o = P9;
		else if (i == 10) o = P10; else o = P11;
		vec4 s = texture2D(src, uv + o * R * texelSize0);
		// gaussian spatial falloff (box blur reads as fuzz) x depth edge
		float w = exp(-2.0 * dot(o, o)) * exp(-abs(s.a - t0) * 80.0);
		sum += s.rgb * w;
		wsum += w;
	}
	gl_FragColor = vec4(sum / wsum, t0);
}
