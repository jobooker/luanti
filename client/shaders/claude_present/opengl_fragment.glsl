// claude_present: the pipeline's last word. Traced modes upsample the
// half-res accumulated buffer with a joint-bilateral filter: the 4
// nearest lighting texels are weighted by how well their traced hit
// distance agrees with the FULL-RES raster depth at this pixel, so
// silhouettes stay crisp while surfaces stay smooth. Raster modes pass
// the merged frame through.
#define merged texture0
#define accum texture1
#define depthmap texture2
#define autoExposure texture3

uniform sampler2D merged;
uniform sampler2D accum;
uniform sampler2D depthmap;
// the direct part of accum (claude_trace historyDirect) and the ramp
// the adapted exposure, one pixel (client/shaders/claude_exposure):
// .r = the factor to multiply by, .g = 1 once written
uniform sampler2D autoExposure;
uniform float claudeAutoExposure;
// THE EYE (2026-10-05), both display only, both need real units:
// claude_white_balance: CAT16 (Li et al. 2017) chromatic adaptation from
//   the adapted white (claude_exposure texel 1) to the display white, at
//   the EFFECTIVE degree of adaptation measured by Zhai & Luo 2017 (CIC25,
//   "A study of neutral white and degree of chromatic adaptation", Eq. 1,
//   display-colour fit): D = 0.709 (1 - 814 K / CCT). The warmer the
//   light, the less the eye adapts to it, so torchlight stays warm. Their
//   data span 3000-16000 K: firelight (1300-1900 K) is an EXTRAPOLATION.
//   CCT by McCamy 1992. (The first version used CIECAM02's D, which is
//   >= 0.82 at any colour: lava light turned the room pink.) The dial
//   scales D; 1 = the published fit.
// claude_night_vision: rods. Per pixel, sigma = 0.04 / (0.04 + Y), Y in
//   cd/m2 (Hunt 1995 via Krawczyk 2005 eq. 4.7), and the displayed colour
//   moves toward a grey of the same displayed luminance tinted
//   (1.05, 0.97, 1.27), the blue shift (Krawczyk eq. 4.15). Per pixel, so
//   a torch flame stays coloured while the moonlit field goes grey-blue.
//   Not modelled: the night's loss of sharpness.
uniform float claudeUnits;
uniform float claudeUnitCdm2;
uniform float claudeWhiteBalance;
uniform float claudeNightVision;
const vec3 LUMA_P = vec3(0.2126, 0.7152, 0.0722);
// CAT16 x (linear sRGB -> XYZ), rows; and its inverse (colour-science 0.4.7)
const vec3 CAT_A0 = vec3(0.3027248, 0.6023702, 0.0704613);
const vec3 CAT_A1 = vec3(0.1537329, 0.7773669, 0.0853695);
const vec3 CAT_A2 = vec3(0.0279452, 0.1478798, 0.9091063);
const vec3 CAT_I0 = vec3(5.4464547, -4.2153768, -0.0262889);
const vec3 CAT_I1 = vec3(-1.0779672, 2.1441028, -0.1177927);
const vec3 CAT_I2 = vec3(0.0079281, -0.2191933, 1.1199502);
vec3 toLms(vec3 c) { return vec3(dot(CAT_A0, c), dot(CAT_A1, c), dot(CAT_A2, c)); }
vec3 fromLms(vec3 l) { return vec3(dot(CAT_I0, l), dot(CAT_I1, l), dot(CAT_I2, l)); }
// MEASUREMENT EXPOSURE (claude_exposure, 2026-10-05). Linear radiance is
// scaled by this before the ACES curve. 1 = the look. The furnace referees
// shoot at 0.25 so a rho = 0.73 room (L = 6.49) lands mid-curve instead of
// at byte 255 — 98.7 % of furnace-073's patch WAS byte 255, which reads
// back as 7.22 whatever the truth. The referee divides it back out.
uniform float claudeExposure;
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
	// THE SPLIT now happens in claude_denoise's last pass, per trace pixel.

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
			|| (claudeView > 18.5 && claudeView < 21.5)) {
		gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
		return;
	}

	// accum is LINEAR radiance now; the display transform is the ONE
	// art knob (energy audit): ACES filmic fit (Narkowicz), then gamma
	// THE EYE: white balance on radiance, then the rods' share per pixel
	float sigma = 0.0;
	if (claudeUnits > 0.5 && claudeView < 0.5) {
		vec4 aw = texture2D(autoExposure, vec2(0.75, 0.5));
		vec4 a0 = texture2D(autoExposure, vec2(0.25, 0.5));
		if (claudeWhiteBalance > 0.0 && aw.a > 0.5 && a0.g > 0.5) {
			// the white's CCT (McCamy 1992), from its CIE xy
			vec3 wx = vec3(dot(vec3(0.4124, 0.3576, 0.1805), aw.rgb),
					dot(vec3(0.2126, 0.7152, 0.0722), aw.rgb),
					dot(vec3(0.0193, 0.1192, 0.9505), aw.rgb));
			float sxyz = max(wx.x + wx.y + wx.z, 1e-6);
			float cx = wx.x / sxyz, cy = wx.y / sxyz;
			float mn = (cx - 0.3320) / (0.1858 - cy);
			float cct = 449.0 * mn * mn * mn + 3525.0 * mn * mn
					+ 6823.3 * mn + 5520.33;
			float d = clamp(0.709 * (1.0 - 814.0 / max(cct, 814.0))
					* claudeWhiteBalance, 0.0, 1.0);
			vec3 lw = max(toLms(aw.rgb), vec3(1e-6));
			vec3 lr = toLms(vec3(1.0));
			c = max(fromLms((d * lr / lw + (1.0 - d)) * toLms(c)), vec3(0.0));
		}
		if (claudeNightVision > 0.5) {
			float ycd = dot(max(c, vec3(0.0)), LUMA_P) * claudeUnitCdm2;
			sigma = 0.04 / (0.04 + ycd);
		}
	}

	// exposure: the manual dial, times the eye's adaptation when on
	float ex = claudeExposure > 0.0 ? claudeExposure : 1.0;
	if (claudeAutoExposure > 0.5) {
		vec4 ae = texture2D(autoExposure, vec2(0.25, 0.5));
		if (ae.g > 0.5 && ae.r > 0.0 && ae.r < 1e9)
			ex *= ae.r;
	}
	vec3 lin = max(c, vec3(0.0)) * ex;
	vec3 aces = clamp(lin * (2.51 * lin + 0.03)
			/ (lin * (2.43 * lin + 0.59) + 0.14), 0.0, 1.0);
	if (sigma > 0.0) {
		float lt = dot(aces, LUMA_P);
		aces = clamp(aces * (1.0 - sigma)
				+ vec3(1.05, 0.97, 1.27) * lt * sigma, 0.0, 1.0);
	}
	gl_FragColor = vec4(pow(aces, vec3(1.0 / 2.2)), 1.0);
}
