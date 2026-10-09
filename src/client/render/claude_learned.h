// claude_denoise_learned (2026-10-09): today's a-trous denoiser with its
// weights supplied per pixel by a small trained network, as GL compute
// passes. Dial claude_denoise_learned (default 0). With it off the six
// claude_denoise passes run exactly as before and nothing here runs.
// Model, training and the head-to-head: docs-draft/mldenoise.md; the
// PyTorch twin these passes must match: util/claude_mldenoise_atrous.py.
#pragma once

#include "pipeline.h"
#include <string>

// set by game.cpp's uniform setter every frame: the dial, and whether the
// filter would run at all (claude_denoise on, photo view, traced mode)
extern bool g_claude_dn_learned_wanted;
// the exposure the network's features are taken at (claude_exposure x the
// eye's adapted factor when auto exposure is on)
extern float g_claude_dn_learned_ex;
// claude_denoise_young, as the filter passes get it
extern float g_claude_dn_learned_young;
// 1 = as first built; 2 = lean: no affinity features, fp16 maps, the upsample fused
extern int g_claude_dn_learned_mode;

// true when the learned passes will replace the six filter passes this frame
bool claudeLearnedOn();
// instrument: the network's pooled features and maps next to a set dump
void claudeLearnedDump(const std::string &path);

// Runs the whole learned denoiser and writes the shown picture into `den`
// (what claude_present and claude_exposure read). A no-op unless
// claudeLearnedOn().
class ClaudeLearnedDenoise : public TrivialRenderStep
{
public:
	ClaudeLearnedDenoise(TextureBuffer *_buffer, u8 _acc, u8 _dir, u8 _gbuf, u8 _mom, u8 _den);
	void run(PipelineContext &context) override;

private:
	TextureBuffer *buffer;
	u8 acc, dir, gbuf, mom, den;
};

// A pipeline step that runs only while the learned denoiser is off: wraps
// each of today's six filter passes, so dial 0 is today's pipeline exactly.
class ClaudeUnlessLearned : public RenderStep
{
public:
	ClaudeUnlessLearned(RenderStep *_step) : step(_step) {}
	void setRenderSource(RenderSource *source) override { step->setRenderSource(source); }
	void setRenderTarget(RenderTarget *target) override { step->setRenderTarget(target); }
	void reset(PipelineContext &context) override { step->reset(context); }
	void run(PipelineContext &context) override
	{
		if (!claudeLearnedOn())
			step->run(context);
	}

private:
	RenderStep *step;
};
