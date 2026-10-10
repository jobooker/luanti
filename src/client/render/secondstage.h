// Luanti
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2010-2013 celeron55, Perttu Ahola <celeron55@gmail.com>
// Copyright (C) 2017 numzero, Lobachevskiy Vitaliy <numzer0@yandex.ru>

#pragma once
#include "pipeline.h"

/**
 *  Step to apply post-processing filter to the rendered image
 */
class PostProcessingStep : public RenderStep
{
public:
	/**
	 * Construct a new PostProcessingStep object
	 *
	 * @param shader_id ID of the shader in IShaderSource
	 * @param texture_map Map of textures to be chosen from the render source
	 */
	PostProcessingStep(u32 shader_id, const std::vector<u8> &texture_map);


	void setRenderSource(RenderSource *source) override;
	void setRenderTarget(RenderTarget *target) override;
	void reset(PipelineContext &context) override;
	void run(PipelineContext &context) override;

	/**
	 * Configure bilinear filtering for a specific texture layer
	 *
	 * @param index Index of the texture layer
	 * @param value true to enable the bilinear filter, false to disable
	 */
	void setBilinearFilter(u8 index, bool value);
private:
	u32 shader_id;
	std::vector<u8> texture_map;
	RenderSource *source { nullptr };
	RenderTarget *target { nullptr };
	video::SMaterial material;

	void configureMaterial();
};


class ResolveMSAAStep : public TrivialRenderStep
{
public:
	ResolveMSAAStep(TextureBufferOutput *_msaa_fbo, TextureBufferOutput *_target_fbo) :
			msaa_fbo(_msaa_fbo), target_fbo(_target_fbo) {};
	void run(PipelineContext &context) override;

private:
	TextureBufferOutput *msaa_fbo;
	TextureBufferOutput *target_fbo;
};


// claude_exposure's one pixel, read back to the CPU every 30th frame for
// claude_stats.json (auto_exposure): [factor, log2 adapted luminance,
// log2 measured luminance, written]. Display data, never fed back.
extern float g_claude_auto_exposure[4];
extern float g_claude_white[4];   // the eye's adapted white (claude_exposure texel 1)

class ClaudeExposureReadback : public TrivialRenderStep
{
public:
	ClaudeExposureReadback(TextureBuffer *_buffer, u8 _index) :
			buffer(_buffer), index(_index) {};
	void run(PipelineContext &context) override;

private:
	TextureBuffer *buffer;
	u8 index;
	u32 frames = 0;
};

// THE LINEAR READBACK (claude_accum_dump, 2026-10-08): on request, the
// latest accumulated radiance (float, before exposure and the display
// curve) to <path>.f32 with <path>.json. The referees read the final 8-bit
// picture and undo an assumed display curve; this reads what the tracer
// actually averaged, for checks that must not depend on that inversion.
extern std::string g_claude_accum_dump;
// THE DENOISER'S WHOLE INPUT SET (claude_dump_at, 2026-10-08, the ML
// denoiser prototype): on request, this frame's accumulated radiance, its
// direct part, the guide (accumulated albedo + face code), the moments and
// the denoiser's output (what claude_present shows), each to
// <path>.<name>.f32 plus one <path>.json. game.cpp sets the path on the
// frames a schedule asks for (still_frames = N), so one accumulation gives
// the inputs at several depths.
extern std::string g_claude_set_dump;
extern float g_claude_set_dump_frames;
extern std::string g_claude_set_dump_extra;
extern unsigned g_claude_set_dumps_written;
class ClaudeSetReadback : public TrivialRenderStep
{
public:
	ClaudeSetReadback(TextureBuffer *_buffer, std::vector<u8> _idx, std::vector<std::string> _names) :
			buffer(_buffer), idx(_idx), names(_names) {};
	void run(PipelineContext &context) override;

private:
	TextureBuffer *buffer;
	std::vector<u8> idx;
	std::vector<std::string> names;
};

// the per-pixel buffer next to the accumulation (rgb = direct light, a =
// the pixel's own sample count): claude_direct_dump, same protocol
extern std::string g_claude_direct_dump;
class ClaudeAccumReadback : public TrivialRenderStep
{
public:
	ClaudeAccumReadback(TextureBuffer *_buffer, u8 _index, std::string *_req = &g_claude_accum_dump) :
			buffer(_buffer), index(_index), req(_req) {};
	void run(PipelineContext &context) override;

private:
	TextureBuffer *buffer;
	u8 index;
	std::string *req;
};

RenderStep *addPostProcessing(RenderPipeline *pipeline, RenderStep *previousStep, v2f scale, Client *client);
