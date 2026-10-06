// claude_exposure (2026-10-05): the camera adapts to the scene like an
// eye. One output pixel. Display only: nothing traced reads it.
//
// MEASURE. The log-average luminance of what is about to be shown (the
// denoised image, before exposure), sampled on a 48x27 grid. Log-average
// (geometric mean) because brightness is perceived in ratios: one bright
// window must not set the exposure for a dark room.
//
// TARGET. The scene's average maps to a KEY value, and the key itself
// falls in the dark: a dim scene is shown dimmer than a bright one,
// which is how vision works (you see that it is night). Krawczyk,
// Myszkowski & Seidel 2005, "Lightness perception in tone reproduction
// for HDR images":  key = 1.03 - 2 / (2 + log10(L + 1)),  L in cd/m2.
// LITERATURE VALUE, not fitted here.
//
// ADAPT. In log brightness, toward the target with a time constant:
// claudeAdaptBrighter when the scene got brighter (light adaptation, fast
// in life), claudeAdaptDarker when it got darker (dark adaptation, slow).
// Those two are John's call (game.cpp m_adapt_*).
//
// OUT: .r = the exposure factor (ACES input for the key, over adapted L),
// .g = 1 (written), .b = log2 of the adapted luminance, .a = log2 of
// this frame's measured luminance.

#define shown texture0
#define previous texture1
#define guide texture2
uniform sampler2D shown;
uniform sampler2D previous;
uniform sampler2D guide;      // claude_trace's guide: albedo rgb, face code
uniform float claudeAdaptColour;
uniform float claudeAdaptBrighter;
uniform float claudeAdaptDarker;
uniform float claudeFrameDt;
uniform float claudeUnitCdm2;
CENTROID_ VARYING_ mediump vec2 varTexCoord;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
const float L_FLOOR = 1e-9;   // renderer units: far below starlight

// TEXEL 1: THE EYE'S WHITE (2026-10-05). The colour of the LIGHT, not of
// the scene: the average over visible surfaces of the light arriving at
// them (shown radiance / that pixel's albedo, i.e. the denoiser's
// demodulated irradiance), so orange planks do not read as orange light.
// Sky, glass and water (face code 0) carry no albedo and are skipped.
// Unit luminance; adapted with claudeAdaptColour. claude_present turns
// it into a CAT16 white balance at the CIECAM02 degree of adaptation.
void whitePixel()
{
	vec3 acc = vec3(0.0);
	float n = 0.0;
	for (int j = 0; j < 27; j++)
	for (int i = 0; i < 48; i++) {
		vec2 q = (vec2(float(i), float(j)) + 0.5) / vec2(48.0, 27.0);
		vec4 g = texture2D(guide, q);
		if (abs(g.a) < 0.5)
			continue;
		vec3 e = max(texture2D(shown, q).rgb, vec3(0.0))
				/ max(g.rgb, vec3(0.005));
		if (all(lessThan(e, vec3(1e6)))) {
			acc += e;
			n += 1.0;
		}
	}
	vec3 now = vec3(1.0);
	float y = dot(acc, LUMA);
	if (n > 0.0 && y > 0.0)
		now = acc / y;
	vec4 p = texture2D(previous, vec2(0.75, 0.5));
	vec3 w = now;
	if (p.a > 0.5 && all(lessThan(abs(p.rgb), vec3(100.0)))) {
		float k = 1.0 - exp(-max(claudeFrameDt, 0.0)
				/ max(claudeAdaptColour, 1e-3));
		w = p.rgb + (now - p.rgb) * k;
	}
	gl_FragColor = vec4(w, 1.0);
}

void main(void)
{
	if (gl_FragCoord.x > 1.0) {
		whitePixel();
		return;
	}
	float acc = 0.0;
	for (int j = 0; j < 27; j++)
	for (int i = 0; i < 48; i++) {
		vec2 q = (vec2(float(i), float(j)) + 0.5) / vec2(48.0, 27.0);
		float l = dot(max(texture2D(shown, q).rgb, vec3(0.0)), LUMA);
		acc += log2(max(l, L_FLOOR));
	}
	float now = acc / (48.0 * 27.0);       // log2 of the geometric mean
	vec4 p = texture2D(previous, vec2(0.25, 0.5));
	float adapted = now;
	if (p.g > 0.5 && abs(p.b) < 200.0) {
		float tau = now > p.b ? claudeAdaptBrighter : claudeAdaptDarker;
		float k = 1.0 - exp(-max(claudeFrameDt, 0.0) / max(tau, 1e-3));
		adapted = p.b + (now - p.b) * k;
	}
	float L = exp2(adapted);                 // renderer units
	float Lcd = L * claudeUnitCdm2;          // cd/m2
	float key = max(1.03 - 2.0 / (2.0 + log(Lcd + 1.0) / log(10.0)), 0.02);
	// THE KEY BELONGS TO REINHARD'S CURVE, display = x / (1 + x), which is
	// what Krawczyk et al. used it with. This renderer shows through ACES
	// (claude_present), which lifts the same input higher (0.18 -> 0.27
	// against Reinhard's 0.15). So aim for the same DISPLAY value the
	// published method gives the average: y = key / (1 + key), and find
	// the ACES input that produces it (the Narkowicz fit, inverted).
	float y = key / (1.0 + key);
	float qa = 2.51 - 2.43 * y, qb = 0.03 - 0.59 * y, qc = -0.14 * y;
	float x = (-qb + sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa);
	gl_FragColor = vec4(x / L, 1.0, adapted, now);
}
