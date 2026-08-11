// claude_present: the pipeline's last word. In traced modes (>= 3) show
// the temporally-accumulated path-traced buffer (half-res, upscaled
// bilinearly via the sampler); otherwise pass the normal merged frame.
#define merged texture0
#define accum texture1

uniform sampler2D merged;
uniform sampler2D accum;
uniform lowp float volumeDebug;

CENTROID_ VARYING_ mediump vec2 varTexCoord;

void main(void)
{
	vec2 uv = varTexCoord.st;
	if (volumeDebug > 2.5) {
		// Teardown's filmic rolloff: highlights compress instead of clip
		vec3 lin = pow(texture2D(accum, uv).rgb, vec3(2.2));
		lin = vec3(1.0) - exp(-lin * 1.6);
		gl_FragColor = vec4(pow(lin, vec3(1.0 / 2.2)), 1.0);
	} else {
		gl_FragColor = vec4(texture2D(merged, uv).rgb, 1.0);
	}
}
