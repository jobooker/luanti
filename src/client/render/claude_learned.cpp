// claude_denoise_learned (2026-10-09): see claude_learned.h.
#include "claude_learned.h"
#include "porting.h"
#include "settings.h"
#include "log.h"
#include "filesys.h"
#include <mt_opengl.h>
using GLint = int;
using GLuint = unsigned int;
#include <ITexture.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <iomanip>
#include <iterator>

bool g_claude_dn_learned_wanted = false;
float g_claude_dn_learned_ex = 1.0f;
float g_claude_dn_learned_young = 64.0f;

namespace {

// THE WEIGHTS FILE (util/claude_mldenoise_atrous.py export): "MLDNW001",
// then int32 cin, c, nfeat, n_floats, then the floats in this order:
// a0.w [c][cin][3][3], a0.b [c], a1.w [c][c][3][3], a1.b, b.w, b.b, c.w, c.b,
// h.w [15][c], h.b [15], mix_bias [6]
struct Model {
	bool tried = false, ok = false;
	int cin = 0, c = 0, nfeat = 0;
	std::vector<float> w;
	int a0w, a0b, a1w, a1b, bw, bb, cw, cb, hw, hb, mb;
	GLuint ssbo = 0;
	GLuint prog[7] = {};   // prep, pool, conv a0, conv a1, b, c, up, (pass uses [6+1])
	GLuint pass = 0;
	GLuint tex_full[4 + 4] = {};   // S ping-pong (2), B ping-pong (2), P (4)
	GLuint tex_q[5 + 6 + 6 + 6 + 4] = {};   // F (5), H (6), Q (6), S (6), M (4)
	int W = 0, H = 0;
	int base = 8;
};
Model g_m;

std::string fnv64(const std::string &s)
{
	unsigned long long h = 1469598103934665603ULL;
	for (unsigned char ch : s) {
		h ^= ch;
		h *= 1099511628211ULL;
	}
	std::ostringstream o;
	o << std::hex << std::setw(16) << std::setfill('0') << h;
	return o.str();
}

GLuint compile(const std::string &src, const char *what)
{
	GLuint sh = GL.CreateShader(GL.COMPUTE_SHADER);
	const char *p = src.c_str();
	GL.ShaderSource(sh, 1, &p, nullptr);
	GL.CompileShader(sh);
	GLint ok = 0;
	GL.GetShaderiv(sh, GL.COMPILE_STATUS, &ok);
	if (!ok) {
		char log[4096] = {};
		GL.GetShaderInfoLog(sh, sizeof(log), nullptr, log);
		errorstream << "[claude_denoise_learned] " << what << " failed to compile: " << log << std::endl;
		return 0;
	}
	GLuint pr = GL.CreateProgram();
	GL.AttachShader(pr, sh);
	GL.LinkProgram(pr);
	GL.GetProgramiv(pr, GL.LINK_STATUS, &ok);
	if (!ok) {
		char log[4096] = {};
		GL.GetProgramInfoLog(pr, sizeof(log), nullptr, log);
		errorstream << "[claude_denoise_learned] " << what << " failed to link: " << log << std::endl;
		return 0;
	}
	return pr;
}

bool init()
{
	Model &M = g_m;
	if (M.tried)
		return M.ok;
	M.tried = true;
	std::string path = g_settings->exists("claude_denoise_learned_weights")
			? g_settings->get("claude_denoise_learned_weights")
			: porting::path_share + DIR_DELIM + "util" + DIR_DELIM + "claude_mldenoise_weights.bin";
	std::ifstream f(path, std::ios::binary);
	std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (blob.size() < 24 || blob.compare(0, 8, "MLDNW001") != 0) {
		errorstream << "[claude_denoise_learned] no weights at " << path << std::endl;
		return false;
	}
	int hdr[4];
	memcpy(hdr, blob.data() + 8, 16);
	M.cin = hdr[0];
	M.c = hdr[1];
	M.nfeat = hdr[2];
	int n = hdr[3];
	if (M.cin != 20 || M.c % 4 || M.nfeat != 4 || (size_t)(24 + 4 * n) != blob.size()) {
		errorstream << "[claude_denoise_learned] weights at " << path << " do not match this build ("
				<< M.cin << " in, " << M.c << " wide, " << M.nfeat << " features, " << n << " floats)" << std::endl;
		return false;
	}
	M.w.resize(n);
	memcpy(M.w.data(), blob.data() + 24, 4 * n);
	int o = 0, c = M.c;
	M.a0w = o; o += c * M.cin * 9; M.a0b = o; o += c;
	M.a1w = o; o += c * c * 9; M.a1b = o; o += c;
	M.bw = o; o += c * c * 9; M.bb = o; o += c;
	M.cw = o; o += c * c * 9; M.cb = o; o += c;
	M.hw = o; o += 15 * c; M.hb = o; o += 15;
	M.mb = o; o += 6;
	if (o != n) {
		errorstream << "[claude_denoise_learned] weights hold " << n << " floats, the model needs " << o << std::endl;
		return false;
	}
	actionstream << "[claude_denoise_learned] weights " << path << " fnv64 " << fnv64(blob)
			<< " (" << n << " floats, width " << c << ")" << std::endl;

	GLint units = 0, cunits = 0;
	GL.GetIntegerv(GL.MAX_IMAGE_UNITS, &units);
	GL.GetIntegerv(GL.MAX_COMPUTE_IMAGE_UNIFORMS, &cunits);
	if (units < M.base + 14 || cunits < 14) {
		errorstream << "[claude_denoise_learned] needs image units " << M.base + 14 << " and 14 per pass; driver has "
				<< units << " / " << cunits << std::endl;
		return false;
	}
	std::string src;
	{
		std::ifstream sf(porting::path_share + DIR_DELIM + "client" + DIR_DELIM + "shaders" + DIR_DELIM
				+ "claude_denoise" + DIR_DELIM + "learned.comp.glsl");
		src.assign((std::istreambuf_iterator<char>(sf)), std::istreambuf_iterator<char>());
	}
	if (src.empty()) {
		errorstream << "[claude_denoise_learned] learned.comp.glsl not found" << std::endl;
		return false;
	}
	auto hdr_of = [&](const std::string &k) {
		std::ostringstream s;
		s << "#version 460\n#define " << k << " 1\n#define BASE " << M.base << "\n";
		return s.str();
	};
	auto conv_hdr = [&](int cin, int cout, int woff, int boff) {
		std::ostringstream s;
		s << hdr_of("K_CONV") << "#define CIN " << cin << "\n#define COUT " << cout << "\n#define WOFF " << woff
				<< "\n#define BOFF " << boff << "\n";
		return s.str();
	};
	std::ostringstream bh, ch;
	bh << hdr_of("K_B") << "#define C " << c << "\n#define WOFF " << M.bw << "\n#define BOFF " << M.bb << "\n";
	ch << hdr_of("K_C") << "#define C " << c << "\n#define WOFF " << M.cw << "\n#define BOFF " << M.cb
			<< "\n#define HWOFF " << M.hw << "\n#define HBOFF " << M.hb << "\n#define MBOFF " << M.mb << "\n";
	const std::string srcs[8] = {hdr_of("K_PREP"), hdr_of("K_POOL"), conv_hdr(M.cin, c, M.a0w, M.a0b),
			conv_hdr(c, c, M.a1w, M.a1b), bh.str(), ch.str(), hdr_of("K_UP"), hdr_of("K_PASS")};
	const char *names[8] = {"prep", "pool", "conv a0", "conv a1", "b", "c+head", "up", "pass"};
	for (int i = 0; i < 8; i++) {
		GLuint p = compile(srcs[i] + src, names[i]);
		if (!p)
			return false;
		if (i < 7)
			M.prog[i] = p;
		else
			M.pass = p;
	}
	GL.GenBuffers(1, &M.ssbo);
	GL.BindBuffer(GL.SHADER_STORAGE_BUFFER, M.ssbo);
	GL.BufferData(GL.SHADER_STORAGE_BUFFER, 4 * n, M.w.data(), GL.STATIC_DRAW);
	GL.BindBuffer(GL.SHADER_STORAGE_BUFFER, 0);
	actionstream << "[claude_denoise_learned] 8 compute programs ready" << std::endl;
	M.ok = true;
	return true;
}

void ensureTextures(int W, int H)
{
	Model &M = g_m;
	if (M.W == W && M.H == H && M.tex_full[0])
		return;
	if (M.tex_full[0]) {
		GL.DeleteTextures(8, M.tex_full);
		GL.DeleteTextures(27, M.tex_q);
	}
	M.W = W;
	M.H = H;
	auto make = [](GLuint *t, int n, int w, int h) {
		GL.GenTextures(n, t);
		for (int i = 0; i < n; i++) {
			GL.BindTexture(GL.TEXTURE_2D, t[i]);
			GL.TexStorage2D(GL.TEXTURE_2D, 1, GL.RGBA32F, w, h);
		}
		GL.BindTexture(GL.TEXTURE_2D, 0);
	};
	make(M.tex_full, 8, W, H);
	make(M.tex_q, 27, W / 4, H / 4);
	actionstream << "[claude_denoise_learned] buffers " << W << "x" << H << " and " << W / 4 << "x" << H / 4 << std::endl;
}

void img(int unit, GLuint tex, bool write)
{
	GL.BindImageTexture(g_m.base + unit, tex, 0, 0, 0, write ? GL.WRITE_ONLY : GL.READ_ONLY, GL.RGBA32F);
}

void uniforms(GLuint p, int W, int H, int step, int pass)
{
	auto loc = [&](const char *n) { return GL.GetUniformLocation(p, n); };
	GL.Uniform2i(loc("uFull"), W, H);
	GL.Uniform2i(loc("uQ"), W / 4, H / 4);
	GL.Uniform2i(loc("uE"), W / 8, (H / 4) / 2);
	GL.Uniform1f(loc("uEx"), g_claude_dn_learned_ex);
	GL.Uniform1i(loc("uStep"), step);
	GL.Uniform1i(loc("uPass"), pass);
	GL.Uniform1f(loc("uYoung"), g_claude_dn_learned_young);
}

} // namespace

// INSTRUMENT (2026-10-09, the forest gap): the network's own intermediate
// images on a dump frame, to compare stage by stage with the PyTorch twin.
// <path>.lF.f32 = the pooled features (1/4 res, 5 x RGBA, top-down rows),
// <path>.lM.f32 = the activated maps (1/4 res, 4 x RGBA), channels
// interleaved per texel: [y][x][4 * image + component].
void claudeLearnedDump(const std::string &path)
{
	Model &M = g_m;
	if (!M.ok || !M.tex_q[0] || !g_claude_dn_learned_wanted)
		return;
	const int w = M.W / 4, h = M.H / 4;
	auto grab = [&](const GLuint *t, int n, const std::string &suffix) {
		std::vector<float> one((size_t)w * h * 4), all((size_t)w * h * 4 * n);
		for (int i = 0; i < n; i++) {
			GL.BindTexture(GL.TEXTURE_2D, t[i]);
			GL.GetTexImage(GL.TEXTURE_2D, 0, GL.RGBA, GL.FLOAT, one.data());
			for (size_t k = 0; k < (size_t)w * h; k++)
				for (int c = 0; c < 4; c++)
					all[k * 4 * n + 4 * i + c] = one[k * 4 + c];
		}
		GL.BindTexture(GL.TEXTURE_2D, 0);
		std::ofstream o(path + suffix, std::ios::binary);
		o.write((const char *)all.data(), all.size() * 4);
	};
	grab(M.tex_q, 5, ".lF.f32");
	grab(M.tex_q + 23, 4, ".lM.f32");
}

bool claudeLearnedOn()
{
	return g_claude_dn_learned_wanted && init();
}

ClaudeLearnedDenoise::ClaudeLearnedDenoise(TextureBuffer *_buffer, u8 _acc, u8 _dir, u8 _gbuf, u8 _mom, u8 _den) :
		buffer(_buffer), acc(_acc), dir(_dir), gbuf(_gbuf), mom(_mom), den(_den)
{
	// read the weights and build the passes at startup, so the hash is in
	// the log whether or not the dial is ever turned on
	init();
}

void ClaudeLearnedDenoise::run(PipelineContext &context)
{
	if (!claudeLearnedOn())
		return;
	Model &M = g_m;
	video::ITexture *tA = buffer->getTexture(acc), *tD = buffer->getTexture(dir),
			*tG = buffer->getTexture(gbuf), *tM = buffer->getTexture(mom), *tO = buffer->getTexture(den);
	if (!tA || !tD || !tG || !tM || !tO || !tA->getNativeHandle())
		return;
	const int W = tA->getSize().Width, H = tA->getSize().Height;
	ensureTextures(W, H);
	GLuint A = tA->getNativeHandle(), D = tD->getNativeHandle(), G = tG->getNativeHandle(),
			Mo = tM->getNativeHandle(), O = tO->getNativeHandle();
	GLuint *F = M.tex_q, *Hh = M.tex_q + 5, *Q = M.tex_q + 11, *S = M.tex_q + 17, *Mm = M.tex_q + 23;
	GLuint *Sp = M.tex_full, *Bp = M.tex_full + 2, *P = M.tex_full + 4;
	const int nc = M.c / 4;

	GLint prev = 0;
	GL.GetIntegerv(GL.CURRENT_PROGRAM, &prev);
	GL.BindBufferBase(GL.SHADER_STORAGE_BUFFER, 9, M.ssbo);
	auto go = [&](GLuint p, int w, int h) {
		GL.DispatchCompute((w + 7) / 8, (h + 7) / 8, 1);
		GL.MemoryBarrier(GL.SHADER_IMAGE_ACCESS_BARRIER_BIT);
		(void)p;
	};
	auto use = [&](GLuint p, int step = 1, int pass = 0) {
		GL.UseProgram(p);
		uniforms(p, W, H, step, pass);
	};
	const int Wq = W / 4, Hq = H / 4;
	// pass 0: the noise estimate, as today's
	use(M.prog[0]);
	img(0, A, false); img(1, G, false); img(2, Mo, false); img(3, Sp[0], true);
	go(M.prog[0], W, H);
	// features, pooled
	use(M.prog[1]);
	img(0, A, false); img(1, D, false); img(2, G, false); img(3, Mo, false);
	for (int i = 0; i < 5; i++) img(4 + i, F[i], true);
	go(M.prog[1], Wq, Hq);
	// conv a0: F -> H; conv a1: H -> Q
	use(M.prog[2]);
	for (int i = 0; i < 5; i++) img(i, F[i], false);
	for (int i = 0; i < nc; i++) img(8 + i, Hh[i], true);
	go(M.prog[2], Wq, Hq);
	use(M.prog[3]);
	for (int i = 0; i < nc; i++) img(i, Hh[i], false);
	for (int i = 0; i < nc; i++) img(8 + i, Q[i], true);
	go(M.prog[3], Wq, Hq);
	// q + nearest(conv b(pool2 q)) -> S
	use(M.prog[4]);
	for (int i = 0; i < nc; i++) img(i, Q[i], false);
	for (int i = 0; i < nc; i++) img(8 + i, S[i], true);
	go(M.prog[4], Wq, Hq);
	// conv c + head -> maps
	use(M.prog[5]);
	for (int i = 0; i < nc; i++) img(i, S[i], false);
	for (int i = 0; i < 4; i++) img(8 + i, Mm[i], true);
	go(M.prog[5], Wq, Hq);
	// maps to full resolution
	use(M.prog[6]);
	for (int i = 0; i < 4; i++) img(i, Mm[i], false);
	for (int i = 0; i < 4; i++) img(4 + i, P[i], true);
	go(M.prog[6], W, H);
	// the five weighted passes
	for (int it = 1; it <= 5; it++) {
		use(M.pass, 1 << (it - 1), it);
		img(0, Sp[(it - 1) % 2], false);
		img(1, G, false); img(2, Mo, false); img(3, D, false); img(4, A, false);
		for (int i = 0; i < 4; i++) img(5 + i, P[i], false);
		img(9, Bp[(it - 1) % 2], false);
		img(10, Sp[it % 2], true);
		img(11, Bp[it % 2], true);
		img(12, O, true);
		go(M.pass, W, H);
	}
	GL.MemoryBarrier(GL.ALL_BARRIER_BITS);
	GL.UseProgram(prev);
}
