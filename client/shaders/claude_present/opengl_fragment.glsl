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
uniform lowp float volumeDebug;
uniform vec2 texelSize0;       // full-res texel (from merged)
uniform vec2 volumeDepthRange; // camera near/far, world BS units

CENTROID_ VARYING_ mediump vec2 varTexCoord;

void main(void)
{
	vec2 uv = varTexCoord.st;
	if (volumeDebug < 2.5) {
		gl_FragColor = vec4(texture2D(merged, uv).rgb, 1.0);
		return;
	}

	// full-res raster depth -> linear distance in node units
	float d = texture2D(depthmap, uv).r;
	float zn = volumeDepthRange.x;
	float zf = volumeDepthRange.y;
	float guide = 200.0;
	if (d < 0.9999) {
		float ez = 2.0 * zn * zf / (zf + zn - (2.0 * d - 1.0) * (zf - zn));
		guide = min(ez / 10.0, 200.0); // BS = 10
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
		float dw = exp(-abs(s.a * 200.0 - guide) * 0.6);
		float w = bw * dw + 1e-5;
		sum += s.rgb * w;
		wsum += w;
	}
	vec3 c = sum / wsum;

	// Teardown's filmic rolloff: highlights compress instead of clip
	vec3 lin = pow(max(c, vec3(0.0)), vec3(2.2));
	lin = vec3(1.0) - exp(-lin * 1.6);
	gl_FragColor = vec4(pow(lin, vec3(1.0 / 2.2)), 1.0);
}
