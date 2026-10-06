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
uniform sampler2D shown;
uniform sampler2D previous;
uniform float claudeAdaptBrighter;
uniform float claudeAdaptDarker;
uniform float claudeFrameDt;
uniform float claudeUnitCdm2;
CENTROID_ VARYING_ mediump vec2 varTexCoord;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
const float L_FLOOR = 1e-9;   // renderer units: far below starlight

void main(void)
{
	float acc = 0.0;
	for (int j = 0; j < 27; j++)
	for (int i = 0; i < 48; i++) {
		vec2 q = (vec2(float(i), float(j)) + 0.5) / vec2(48.0, 27.0);
		float l = dot(max(texture2D(shown, q).rgb, vec3(0.0)), LUMA);
		acc += log2(max(l, L_FLOOR));
	}
	float now = acc / (48.0 * 27.0);       // log2 of the geometric mean
	vec4 p = texture2D(previous, vec2(0.5));
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
