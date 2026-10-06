// claude_exposure (2026-10-05): the camera adapts to the scene like an
// eye. One output pixel. Display only: nothing traced reads it.
//
// MEASURE. The log-average luminance of what is about to be shown (the
// denoised image, before exposure), sampled on a 48x27 grid. Log-average
// (geometric mean) because brightness is perceived in ratios: one bright
// window must not set the exposure for a dark room.
//
// TARGET. A camera's: ISO 2720 metering (see main()). Dimmed in the dark
// by Krawczyk et al. 2005's key, key = 1.03 - 2 / (2 + log10(L + 1)), L in
// cd/m2, relative to daylight -- a rule of thumb by its authors' account.
//
// ADAPT. In log brightness, toward the target with a time constant:
// claudeAdaptBrighter when the scene got brighter (light adaptation, fast
// in life), claudeAdaptDarker when it got darker (dark adaptation, slow).
// Those two are John's call (game.cpp m_adapt_*).
//
// OUT: .r = the exposure factor (ISO metering x the dark's dimming, / L),
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
		// code 0 (glass, water, emitters) and the sky's own code
		// (1 + 6 * 4096) carry no light-on-a-surface
		if (abs(g.a) < 0.5 || abs(abs(g.a) - 24577.0) < 0.5)
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
	// THE CAMERA: ISO 2720 reflected-light metering, calibration K = 12.5
	// at ISO 100 with the saturation-based constant q = 0.65 -> 1.2, the
	// form Lagarde & de Rousiers (Frostbite, "Moving Frostbite to PBR")
	// and Luanti's own update_exposure use: exposure = 1 / (9.6 L_avg).
	// Scale-free, so it holds in renderer units too. (Until 2026-10-06 the
	// target was Krawczyk's key mapped through Reinhard: a rule of thumb
	// that made daylight about one stop too bright -- the sunlit forest
	// floor clipped; at 0.5x it read like the raster renderer's colours.)
	// THE DARK: Krawczyk 2005's key still falls in dim scenes, so night
	// is shown darker than day. It is used RELATIVE to its value at
	// 1000 cd/m2 (ordinary daylight), capped at 1, so it only ever dims.
	// TUNED: the 1000 cd/m2 reference | learn by: John's eye at dusk.
	float key = max(1.03 - 2.0 / (2.0 + log(Lcd + 1.0) / log(10.0)), 0.02);
	float keyDay = 1.03 - 2.0 / (2.0 + log(1001.0) / log(10.0));
	float dim = min(key / keyDay, 1.0);
	float x = dim / 9.6;
	gl_FragColor = vec4(x / L, 1.0, adapted, now);
}
