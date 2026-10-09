// claude_denoise_learned (2026-10-09): today's a-trous filter with its
// weights supplied per pixel by a small network, as compute passes.
// The model and its derivation: docs-draft/mldenoise.md (engine worktree
// mldenoise), util/claude_mldenoise_atrous.py (the PyTorch twin this must
// match). claude_learned.cpp compiles this file once per kernel, with
// "#version 460" and the K_* / BASE / weight-offset defines prepended.
//
// COORDINATES. The model was trained on dumps that read the textures with
// Irrlicht's lock(), which flips render targets so row 0 is the TOP. The
// network's convolutions and its 2x pooling are not symmetric under a
// vertical flip, so every kernel here works in that top-down "logical"
// row order: the engine's own textures (accum, gbuf, moments, direct,
// the output) are read and written through LD/ST, which flip; the
// network's own images are stored top-down.
//
// THE NETWORK MAY ONLY RE-WEIGH REAL SAMPLES: per pass a factor on the
// brightness test's width, per tap an affinity exp(-|f_p - f_q|^2), and a
// convex blend over the six stages. Same-face test, kernel, noise
// estimate and young-pixel fade are today's, unchanged.

layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 9) readonly buffer ClaudeMLDW { float wt[]; };

uniform ivec2 uFull;     // trace resolution
uniform ivec2 uQ;        // 1/4 (floor, as avg_pool2d)
uniform ivec2 uE;        // 1/8
uniform float uEx;       // exposure the features are taken at
uniform int uStep;       // a-trous step, 1 2 4 8 16
uniform int uPass;       // 1..5
uniform float uYoung;    // claude_denoise_young

const float ALB_MIN = 0.005;
const float SKY = 1.0 + 6.0 * 65536.0;
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

#define LD(im, p) imageLoad(im, ivec2((p).x, uFull.y - 1 - (p).y))
#define ST(im, p, v) imageStore(im, ivec2((p).x, uFull.y - 1 - (p).y), v)

bool inFull(ivec2 q) { return q.x >= 0 && q.y >= 0 && q.x < uFull.x && q.y < uFull.y; }
bool tapOK(float codeP, float codeQ) { return codeQ > 0.5 && abs(codeQ - abs(codeP)) < 0.5; }
float leaky(float x) { return x > 0.0 ? x : 0.1 * x; }

// ---------------------------------------------------------------- pass 0
#ifdef K_PREP
layout(rgba32f, binding = BASE + 0) uniform readonly image2D iAcc;
layout(rgba32f, binding = BASE + 1) uniform readonly image2D iGbuf;
layout(rgba32f, binding = BASE + 2) uniform readonly image2D iMom;
layout(rgba32f, binding = BASE + 3) uniform writeonly image2D oS;
void main()
{
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (!inFull(p))
		return;
	vec4 c = LD(iAcc, p);
	vec4 g = LD(iGbuf, p);
	if (abs(g.a) < 0.5) {
		imageStore(oS, p, vec4(c.rgb, 0.0));
		return;
	}
	vec3 irr = c.rgb / max(g.rgb, vec3(ALB_MIN));
	vec4 m = LD(iMom, p);
	float var;
	if (m.g <= 0.25) {
		var = max(m.r - m.b * m.b, 0.0) * m.g;
	} else {
		float s1 = 0.0, s2 = 0.0, n = 0.0;
		for (int dy = -3; dy <= 3; dy++)
		for (int dx = -3; dx <= 3; dx++) {
			ivec2 q = p + ivec2(dx, dy);
			if (!inFull(q))
				continue;
			vec4 gq = LD(iGbuf, q);
			if (!(dx == 0 && dy == 0) && !tapOK(g.a, gq.a))
				continue;
			float lq = dot(LD(iAcc, q).rgb / max(gq.rgb, vec3(ALB_MIN)), LUMA);
			s1 += lq;
			s2 += lq * lq;
			n += 1.0;
		}
		s1 /= n;
		var = max(s2 / n - s1 * s1, 0.0);
	}
	imageStore(oS, p, vec4(irr, var));
}
#endif

// ---------------------------------------------------------------- features, pooled 4x4
#ifdef K_POOL
layout(rgba32f, binding = BASE + 0) uniform readonly image2D iAcc;
layout(rgba32f, binding = BASE + 1) uniform readonly image2D iDir;
layout(rgba32f, binding = BASE + 2) uniform readonly image2D iGbuf;
layout(rgba32f, binding = BASE + 3) uniform readonly image2D iMom;
layout(rgba32f, binding = BASE + 4) uniform writeonly image2D oF[5];
void main()
{
	ivec2 t = ivec2(gl_GlobalInvocationID.xy);
	if (t.x >= uQ.x || t.y >= uQ.y)
		return;
	float f[20];
	for (int i = 0; i < 20; i++)
		f[i] = 0.0;
	for (int dy = 0; dy < 4; dy++)
	for (int dx = 0; dx < 4; dx++) {
		ivec2 p = t * 4 + ivec2(dx, dy);
		vec4 a = LD(iAcc, p);
		vec4 d = LD(iDir, p);
		vec4 g = LD(iGbuf, p);
		vec4 m = LD(iMom, p);
		// as util/claude_mldenoise.py load_set(clip=True) and features()
		vec3 acc = clamp(a.rgb, 0.0, 6e4);
		vec3 dr = clamp(d.rgb, 0.0, 6e4);
		float vfac = clamp(m.g, 0.0, 1.0);
		float sd = clamp(sqrt(max(m.r - m.b * m.b, 0.0) * vfac), 0.0, 6e4);
		vec3 alb = max(g.rgb, vec3(ALB_MIN));
		vec3 fi = log(vec3(1.0) + acc / alb * uEx);
		vec3 fd = log(vec3(1.0) + dr / alb * uEx);
		vec3 sa = sqrt(alb);
		float code = g.a;
		float c = abs(code);
		float sky = abs(c - SKY) < 0.5 ? 1.0 : 0.0;
		float none = c < 0.5 ? 1.0 : 0.0;
		float face = (sky < 0.5 && none < 0.5) ? 1.0 : 0.0;
		float k = clamp(floor((c - 1.0) / 65536.0), 0.0, 5.0);
		float ax = floor(k / 2.0);
		float sg = (k - 2.0 * ax) * 2.0 - 1.0;
		vec3 nrm = vec3(ax == 0.0 ? sg : 0.0, ax == 1.0 ? sg : 0.0, ax == 2.0 ? sg : 0.0) * face;
		float mixed = code < -0.5 ? 1.0 : 0.0;
		float fdist = log2(1.0 + max(a.a, 0.0) * 4096.0) / 12.0;
		float sr = 0.0, sdn = 0.0;
		if (p.x + 1 < uFull.x)
			sr = abs(LD(iGbuf, p + ivec2(1, 0)).a - code) < 0.5 ? 1.0 : 0.0;
		if (p.y + 1 < uFull.y)
			sdn = abs(LD(iGbuf, p + ivec2(0, 1)).a - code) < 0.5 ? 1.0 : 0.0;
		float fsd = log(1.0 + sd * uEx);
		float fn = -log2(max(vfac, 1e-6)) / 11.0;
		float v[20] = float[20](fi.r, fi.g, fi.b, fd.r, fd.g, fd.b, sa.r, sa.g, sa.b,
				nrm.x, nrm.y, nrm.z, sky, none, mixed, fdist, sr, sdn, fsd, fn);
		for (int i = 0; i < 20; i++)
			f[i] += v[i];
	}
	for (int j = 0; j < 5; j++)
		imageStore(oF[j], t, vec4(f[4 * j], f[4 * j + 1], f[4 * j + 2], f[4 * j + 3]) / 16.0);
}
#endif

// ---------------------------------------------------------------- 3x3 conv + leaky, at 1/4
#ifdef K_CONV
layout(rgba32f, binding = BASE + 0) uniform readonly image2D iIn[CIN / 4];
layout(rgba32f, binding = BASE + 8) uniform writeonly image2D oOut[COUT / 4];
void main()
{
	ivec2 t = ivec2(gl_GlobalInvocationID.xy);
	if (t.x >= uQ.x || t.y >= uQ.y)
		return;
	float o[COUT];
	for (int j = 0; j < COUT; j++)
		o[j] = wt[BOFF + j];
	for (int ky = 0; ky < 3; ky++)
	for (int kx = 0; kx < 3; kx++) {
		ivec2 q = t + ivec2(kx - 1, ky - 1);
		if (q.x < 0 || q.y < 0 || q.x >= uQ.x || q.y >= uQ.y)
			continue;
		for (int g = 0; g < CIN / 4; g++) {
			vec4 v = imageLoad(iIn[g], q);
			for (int cc = 0; cc < 4; cc++) {
				int ci = 4 * g + cc;
				for (int j = 0; j < COUT; j++)
					o[j] += wt[WOFF + ((j * CIN + ci) * 3 + ky) * 3 + kx] * v[cc];
			}
		}
	}
	for (int g = 0; g < COUT / 4; g++)
		imageStore(oOut[g], t, vec4(leaky(o[4 * g]), leaky(o[4 * g + 1]), leaky(o[4 * g + 2]), leaky(o[4 * g + 3])));
}
#endif

// ---------------------------------------------------------------- q + nearest(conv_b(pool2(q)))
#ifdef K_B
layout(rgba32f, binding = BASE + 0) uniform readonly image2D iQ[C / 4];
layout(rgba32f, binding = BASE + 8) uniform writeonly image2D oS[C / 4];
void main()
{
	ivec2 t = ivec2(gl_GlobalInvocationID.xy);
	if (t.x >= uQ.x || t.y >= uQ.y)
		return;
	// torch nearest: src = min(floor(dst * (in / out)), in - 1), in float
	ivec2 e = ivec2(min(int(floor(float(t.x) * (float(uE.x) / float(uQ.x)))), uE.x - 1),
			min(int(floor(float(t.y) * (float(uE.y) / float(uQ.y)))), uE.y - 1));
	float o[C];
	for (int j = 0; j < C; j++)
		o[j] = wt[BOFF + j];
	for (int ky = 0; ky < 3; ky++)
	for (int kx = 0; kx < 3; kx++) {
		ivec2 r = e + ivec2(kx - 1, ky - 1);
		if (r.x < 0 || r.y < 0 || r.x >= uE.x || r.y >= uE.y)
			continue;
		for (int g = 0; g < C / 4; g++) {
			vec4 v = (imageLoad(iQ[g], 2 * r) + imageLoad(iQ[g], 2 * r + ivec2(1, 0))
					+ imageLoad(iQ[g], 2 * r + ivec2(0, 1)) + imageLoad(iQ[g], 2 * r + ivec2(1, 1))) * 0.25;
			for (int cc = 0; cc < 4; cc++) {
				int ci = 4 * g + cc;
				for (int j = 0; j < C; j++)
					o[j] += wt[WOFF + ((j * C + ci) * 3 + ky) * 3 + kx] * v[cc];
			}
		}
	}
	for (int g = 0; g < C / 4; g++) {
		vec4 q = imageLoad(iQ[g], t);
		imageStore(oS[g], t, q + vec4(leaky(o[4 * g]), leaky(o[4 * g + 1]), leaky(o[4 * g + 2]), leaky(o[4 * g + 3])));
	}
}
#endif

// ---------------------------------------------------------------- conv c + head + activations
#ifdef K_C
layout(rgba32f, binding = BASE + 0) uniform readonly image2D iS[C / 4];
layout(rgba32f, binding = BASE + 8) uniform writeonly image2D oM[4];
void main()
{
	ivec2 t = ivec2(gl_GlobalInvocationID.xy);
	if (t.x >= uQ.x || t.y >= uQ.y)
		return;
	float o[C];
	for (int j = 0; j < C; j++)
		o[j] = wt[BOFF + j];
	for (int ky = 0; ky < 3; ky++)
	for (int kx = 0; kx < 3; kx++) {
		ivec2 q = t + ivec2(kx - 1, ky - 1);
		if (q.x < 0 || q.y < 0 || q.x >= uQ.x || q.y >= uQ.y)
			continue;
		for (int g = 0; g < C / 4; g++) {
			vec4 v = imageLoad(iS[g], q);
			for (int cc = 0; cc < 4; cc++) {
				int ci = 4 * g + cc;
				for (int j = 0; j < C; j++)
					o[j] += wt[WOFF + ((j * C + ci) * 3 + ky) * 3 + kx] * v[cc];
			}
		}
	}
	for (int j = 0; j < C; j++)
		o[j] = leaky(o[j]);
	float h[15];
	for (int k = 0; k < 15; k++) {
		float s = wt[HBOFF + k];
		for (int j = 0; j < C; j++)
			s += wt[HWOFF + k * C + j] * o[j];
		h[k] = s;
	}
	float sig[5];
	for (int k = 0; k < 5; k++)
		sig[k] = exp(clamp(h[k], -4.0, 4.0));
	float lg[6], mx = -1e30;
	for (int k = 0; k < 6; k++) {
		lg[k] = h[9 + k] + wt[MBOFF + k];
		mx = max(mx, lg[k]);
	}
	float se = 0.0;
	for (int k = 0; k < 6; k++) {
		lg[k] = exp(lg[k] - mx);
		se += lg[k];
	}
	for (int k = 0; k < 6; k++)
		lg[k] /= se;
	imageStore(oM[0], t, vec4(sig[0], sig[1], sig[2], sig[3]));
	imageStore(oM[1], t, vec4(sig[4], h[5], h[6], h[7]));
	imageStore(oM[2], t, vec4(h[8], lg[0], lg[1], lg[2]));
	imageStore(oM[3], t, vec4(lg[3], lg[4], lg[5], 0.0));
}
#endif

// ---------------------------------------------------------------- the maps, bilinear to full resolution
#ifdef K_UP
layout(rgba32f, binding = BASE + 0) uniform readonly image2D iM[4];
layout(rgba32f, binding = BASE + 4) uniform writeonly image2D oP[4];
// torch bilinear, align_corners=False: src = max(scale * (dst + 0.5) - 0.5, 0)
void taps(int dst, int insz, int outsz, out int i0, out int i1, out float l1)
{
	float scale = float(insz) / float(outsz);
	float src = max(scale * (float(dst) + 0.5) - 0.5, 0.0);
	i0 = int(src);
	i1 = i0 + (i0 < insz - 1 ? 1 : 0);
	l1 = src - float(i0);
}
void main()
{
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (!inFull(p))
		return;
	int x0, x1, y0, y1;
	float lx, ly;
	taps(p.x, uQ.x, uFull.x, x0, x1, lx);
	taps(p.y, uQ.y, uFull.y, y0, y1, ly);
	vec4 m[4];
	for (int j = 0; j < 4; j++) {
		vec4 a = imageLoad(iM[j], ivec2(x0, y0)), b = imageLoad(iM[j], ivec2(x1, y0));
		vec4 c = imageLoad(iM[j], ivec2(x0, y1)), d = imageLoad(iM[j], ivec2(x1, y1));
		m[j] = (1.0 - ly) * ((1.0 - lx) * a + lx * b) + ly * ((1.0 - lx) * c + lx * d);
	}
	// repacked for the passes: P0 = the 4 affinity features (read at every
	// tap), P1 = sig 1-4, P2 = (sig 5, mix 0-2), P3 = (mix 3-5)
	imageStore(oP[0], p, vec4(m[1].y, m[1].z, m[1].w, m[2].x));
	imageStore(oP[1], p, m[0]);
	imageStore(oP[2], p, vec4(m[1].x, m[2].y, m[2].z, m[2].w));
	imageStore(oP[3], p, vec4(m[3].x, m[3].y, m[3].z, 0.0));
}
#endif

// ---------------------------------------------------------------- one weighted a-trous pass
#ifdef K_PASS
layout(rgba32f, binding = BASE + 0) uniform readonly image2D iS;
layout(rgba32f, binding = BASE + 1) uniform readonly image2D iGbuf;
layout(rgba32f, binding = BASE + 2) uniform readonly image2D iMom;
layout(rgba32f, binding = BASE + 3) uniform readonly image2D iDir;
layout(rgba32f, binding = BASE + 4) uniform readonly image2D iAcc;
layout(rgba32f, binding = BASE + 5) uniform readonly image2D iP[4];
layout(rgba32f, binding = BASE + 9) uniform readonly image2D iB;
layout(rgba32f, binding = BASE + 10) uniform writeonly image2D oS;
layout(rgba32f, binding = BASE + 11) uniform writeonly image2D oB;
layout(rgba32f, binding = BASE + 12) uniform writeonly image2D oDen;
const float H5[5] = float[5](1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0);
void main()
{
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (!inFull(p))
		return;
	vec4 g = LD(iGbuf, p);
	float code = g.a;
	bool valid = abs(code) >= 0.5;
	vec4 cp = imageLoad(iS, p);
	vec4 p1 = imageLoad(iP[1], p), p2 = imageLoad(iP[2], p), p3 = imageLoad(iP[3], p);
	float sigm = uPass == 1 ? p1.x : uPass == 2 ? p1.y : uPass == 3 ? p1.z : uPass == 4 ? p1.w : p2.x;
	float mixk = uPass == 1 ? p2.z : uPass == 2 ? p2.w : uPass == 3 ? p3.x : uPass == 4 ? p3.y : p3.z;
	vec3 bin = uPass == 1 ? p2.y * cp.rgb : imageLoad(iB, p).rgb;
	vec4 outv = cp;
	if (valid) {
		float vs = 0.0, vw = 0.0;
		for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++) {
			ivec2 q = p + ivec2(dx, dy);
			if (!inFull(q))
				continue;
			if (!(dx == 0 && dy == 0) && !tapOK(code, LD(iGbuf, q).a))
				continue;
			float k = (dx == 0 ? 0.5 : 0.25) * (dy == 0 ? 0.5 : 0.25);
			vs += imageLoad(iS, q).a * k;
			vw += k;
		}
		float sl = (4.0 * sqrt(max(vs / vw, 0.0)) + 1e-8) * sigm;
		float lp = dot(cp.rgb, LUMA);
		float neff = uPass == 5 ? LD(iDir, p).a : 1.0 / max(LD(iMom, p).g, 1e-4);
		float kappa = uYoung > 0.5 ? clamp((neff - 1.0) / uYoung, 0.0, 1.0) : 1.0;
		vec4 fp = imageLoad(iP[0], p);
		vec3 sum = vec3(0.0);
		float wsum = 0.0, vsum = 0.0;
		for (int dy = -2; dy <= 2; dy++)
		for (int dx = -2; dx <= 2; dx++) {
			ivec2 q = p + ivec2(dx, dy) * uStep;
			if (!inFull(q))
				continue;
			bool ctr = dx == 0 && dy == 0;
			if (!ctr && !tapOK(code, LD(iGbuf, q).a))
				continue;
			vec4 cq = imageLoad(iS, q);
			float w = H5[dx + 2] * H5[dy + 2] * exp(-kappa * abs(dot(cq.rgb, LUMA) - lp) / sl);
			if (!ctr) {
				vec4 df = imageLoad(iP[0], q) - fp;
				w *= exp(-dot(df, df));
			}
			sum += cq.rgb * w;
			wsum += w;
			vsum += w * w * cq.a;
		}
		outv = vec4(sum / wsum, vsum / (wsum * wsum));
	}
	vec3 bout = bin + mixk * outv.rgb;
	if (uPass < 5) {
		imageStore(oS, p, outv);
		imageStore(oB, p, vec4(bout, 0.0));
	} else {
		vec4 a = LD(iAcc, p);
		vec3 c = valid ? bout * max(g.rgb, vec3(ALB_MIN)) : a.rgb;
		ST(oDen, p, vec4(c, a.a));
	}
}
#endif
