// Luanti
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2010-2013 celeron55, Perttu Ahola <celeron55@gmail.com>
// Copyright (C) 2017 numzero, Lobachevskiy Vitaliy <numzer0@yandex.ru>
// Copyright (C) 2020 appgurueu, Lars Mueller <appgurulars@gmx.de>

#include "secondstage.h"
#include <fstream>
#include <cmath>
#include "client/client.h"
#include "client/shader.h"
#include "settings.h"
#include "plain.h"
#include <ISceneManager.h>

PostProcessingStep::PostProcessingStep(u32 _shader_id, const std::vector<u8> &_texture_map) :
	shader_id(_shader_id), texture_map(_texture_map)
{
	assert(texture_map.size() <= video::MATERIAL_MAX_TEXTURES);
	configureMaterial();
}

void PostProcessingStep::configureMaterial()
{
	material.UseMipMaps = false;
	material.ZBuffer = video::ECFN_LESSEQUAL;
	material.ZWriteEnable = video::EZW_ON;
	for (u32 k = 0; k < texture_map.size(); ++k) {
		material.TextureLayers[k].AnisotropicFilter = 0;
		material.TextureLayers[k].MinFilter = video::ETMINF_NEAREST_MIPMAP_NEAREST;
		material.TextureLayers[k].MagFilter = video::ETMAGF_NEAREST;
		material.TextureLayers[k].TextureWrapU = video::ETC_CLAMP_TO_EDGE;
		material.TextureLayers[k].TextureWrapV = video::ETC_CLAMP_TO_EDGE;
	}
}

void PostProcessingStep::setRenderSource(RenderSource *_source)
{
	source = _source;
}

void PostProcessingStep::setRenderTarget(RenderTarget *_target)
{
	target = _target;
}

void PostProcessingStep::reset(PipelineContext &context)
{
}

void PostProcessingStep::run(PipelineContext &context)
{
	if (target)
		target->activate(context);

	// attach the shader
	material.MaterialType = context.client->getShaderSource()->getShaderInfo(shader_id).material;

	auto driver = context.device->getVideoDriver();

	for (u32 i = 0; i < texture_map.size(); i++)
		material.TextureLayers[i].Texture = source->getTexture(texture_map[i]);

	static const video::SColor color = video::SColor(0, 0, 0, 255);
	static const video::S3DVertex vertices[4] = {
			video::S3DVertex(1.0, -1.0, 0.0, 0.0, 0.0, -1.0,
					color, 1.0, 0.0),
			video::S3DVertex(-1.0, -1.0, 0.0, 0.0, 0.0, -1.0,
					color, 0.0, 0.0),
			video::S3DVertex(-1.0, 1.0, 0.0, 0.0, 0.0, -1.0,
					color, 0.0, 1.0),
			video::S3DVertex(1.0, 1.0, 0.0, 0.0, 0.0, -1.0,
					color, 1.0, 1.0),
	};
	static const u16 indices[6] = {0, 1, 2, 2, 3, 0};
	driver->setMaterial(material);
	driver->drawVertexPrimitiveList(&vertices, 4, &indices, 2);
}

void PostProcessingStep::setBilinearFilter(u8 index, bool value)
{
	assert(index < video::MATERIAL_MAX_TEXTURES);
	material.TextureLayers[index].MinFilter = value ? video::ETMINF_LINEAR_MIPMAP_NEAREST : video::ETMINF_NEAREST_MIPMAP_NEAREST;
	material.TextureLayers[index].MagFilter = value ? video::ETMAGF_LINEAR : video::ETMAGF_NEAREST;
}

RenderStep *addPostProcessing(RenderPipeline *pipeline, RenderStep *previousStep, v2f scale, Client *client)
{
	auto buffer = pipeline->createOwned<TextureBuffer>();
	auto driver = client->getSceneManager()->getVideoDriver();

	// configure texture formats
	video::ECOLOR_FORMAT color_format = selectColorFormat(driver);
	video::ECOLOR_FORMAT depth_format = selectDepthFormat(driver);

	verbosestream << "addPostProcessing(): color = "
		<< video::ColorFormatName(color_format) << ", depth = "
		<< video::ColorFormatName(depth_format) << std::endl;

	// init post-processing buffer
	static const u8 TEXTURE_COLOR = 0;
	static const u8 TEXTURE_DEPTH = 1;
	static const u8 TEXTURE_BLOOM = 2;
	static const u8 TEXTURE_EXPOSURE_1 = 3;
	static const u8 TEXTURE_EXPOSURE_2 = 4;
	static const u8 TEXTURE_FXAA = 5;
	static const u8 TEXTURE_VOLUME = 6;

	static const u8 TEXTURE_MSAA_COLOR = 7;
	static const u8 TEXTURE_MSAA_DEPTH = 8;

	static const u8 TEXTURE_SCALE_DOWN = 10;
	static const u8 TEXTURE_SCALE_UP = 20;

	// because bloom_format is floating point
	const bool bloom_available = driver->queryFeature(video::EVDF_RENDER_TO_FLOAT_TEXTURE);
	const bool enable_bloom = g_settings->getBool("enable_bloom") && bloom_available;
	const bool enable_volumetric_light = g_settings->getBool("enable_volumetric_lighting") && enable_bloom;
	const bool enable_auto_exposure = g_settings->getBool("enable_auto_exposure") && bloom_available;
	if (g_settings->getBool("enable_bloom") && !bloom_available) {
		warningstream << "Ignoring configured bloom since it's not supported by "
			"the current video driver." << std::endl;
	}
	if (g_settings->getBool("enable_auto_exposure") && !bloom_available) {
		warningstream << "Ignoring configured auto exposure since it's not supported by "
			"the current video driver." << std::endl;
	}

	verbosestream << "addPostProcessing(): bloom = "
		<< enable_bloom << (enable_volumetric_light ? " + volumetric" : "")
		<< ", exposure = " << enable_auto_exposure << std::endl;

	const std::string antialiasing = g_settings->get("antialiasing");
	const u16 antialiasing_scale = MYMAX(2, g_settings->getU16("fsaa"));

	// This code only deals with MSAA in combination with post-processing. MSAA without
	// post-processing works via a flag at OpenGL context creation instead.
	// To make MSAA work with post-processing, we need multisample texture support,
	// which has higher OpenGL (ES) version requirements.
	// Note: This is not about renderbuffer objects, but about textures,
	// since that's what we use and what Irrlicht allows us to use.

	const bool msaa_available = driver->queryFeature(video::EVDF_TEXTURE_MULTISAMPLE);
	const bool enable_msaa = antialiasing == "fsaa" && msaa_available;
	if (antialiasing == "fsaa" && !msaa_available) {
		warningstream << "Ignoring configured FSAA since it's not supported in "
			"combination with post-processing by the current video driver." << std::endl;
	}

	const bool enable_ssaa = antialiasing == "ssaa";
	const bool enable_fxaa = g_settings->getBool("fxaa");

	verbosestream << "addPostProcessing(): AA = "
		<< (enable_msaa ? "msaa" : enable_ssaa ? "ssaa" : "none")
		<< " " << antialiasing_scale << "x" << (enable_fxaa ? " + fxaa" : "") << std::endl;

	// Super-sampling is simply rendering into a larger texture.
	// Downscaling is done by the final step when rendering to the screen.
	if (enable_ssaa) {
		scale *= antialiasing_scale;
	}

	if (enable_msaa) {
		buffer->setTexture(TEXTURE_MSAA_COLOR, scale, "3d_render_msaa", color_format, false, antialiasing_scale);
		buffer->setTexture(TEXTURE_MSAA_DEPTH, scale, "3d_depthmap_msaa", depth_format, false, antialiasing_scale);
	}

	buffer->setTexture(TEXTURE_COLOR, scale, "3d_render", color_format);
	buffer->setTexture(TEXTURE_EXPOSURE_1, core::dimension2du(1,1), "exposure_1", color_format, /*clear:*/ true);
	buffer->setTexture(TEXTURE_EXPOSURE_2, core::dimension2du(1,1), "exposure_2", color_format, /*clear:*/ true);
	buffer->setTexture(TEXTURE_DEPTH, scale, "3d_depthmap", depth_format);

	// attach buffer to the previous step
	if (enable_msaa) {
		TextureBufferOutput *msaa = pipeline->createOwned<TextureBufferOutput>(buffer, std::vector<u8> { TEXTURE_MSAA_COLOR }, TEXTURE_MSAA_DEPTH);
		previousStep->setRenderTarget(msaa);
		TextureBufferOutput *normal = pipeline->createOwned<TextureBufferOutput>(buffer, std::vector<u8> { TEXTURE_COLOR }, TEXTURE_DEPTH);
		pipeline->addStep<ResolveMSAAStep>(msaa, normal);
	} else {
		previousStep->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, std::vector<u8> { TEXTURE_COLOR }, TEXTURE_DEPTH));
	}

	// shared variables
	u32 shader_id;

	// Number of mipmap levels of the bloom downsampling texture
	// (this affects the bloom strength, so don't blindly change it)
	const u8 MIPMAP_LEVELS = 4;

	// color_format can be a normalized integer format, but bloom requires
	// values outside of [0,1] so this needs to be a different one.
	const auto bloom_format = video::ECF_A16B16G16R16F;

	// post-processing stage

	u8 source = TEXTURE_COLOR;

	// common downsampling step for bloom or autoexposure
	if (enable_bloom || enable_auto_exposure) {

		v2f downscale = scale * 0.5f;
		for (u8 i = 0; i < MIPMAP_LEVELS; i++) {
			buffer->setTexture(TEXTURE_SCALE_DOWN + i, downscale, std::string("downsample") + std::to_string(i), bloom_format);
			if (enable_bloom)
				buffer->setTexture(TEXTURE_SCALE_UP + i, downscale, std::string("upsample") + std::to_string(i), bloom_format);
			downscale *= 0.5f;
		}

		if (enable_bloom) {
			buffer->setTexture(TEXTURE_BLOOM, scale, "bloom", bloom_format);

			// get bright spots
			u32 shader_id = client->getShaderSource()->getShaderRaw("extract_bloom");
			RenderStep *extract_bloom = pipeline->addStep<PostProcessingStep>(shader_id, std::vector<u8> { source, TEXTURE_EXPOSURE_1 });
			extract_bloom->setRenderSource(buffer);
			extract_bloom->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, TEXTURE_BLOOM));
			source = TEXTURE_BLOOM;
		}

		if (enable_volumetric_light) {
			buffer->setTexture(TEXTURE_VOLUME, scale, "volume", bloom_format);

			shader_id = client->getShaderSource()->getShaderRaw("volumetric_light");
			auto volume = pipeline->addStep<PostProcessingStep>(shader_id, std::vector<u8> { source, TEXTURE_DEPTH });
			volume->setRenderSource(buffer);
			volume->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, TEXTURE_VOLUME));
			source = TEXTURE_VOLUME;
		}

		// downsample
		shader_id = client->getShaderSource()->getShaderRaw("bloom_downsample");
		for (u8 i = 0; i < MIPMAP_LEVELS; i++) {
			auto step = pipeline->addStep<PostProcessingStep>(shader_id, std::vector<u8> { source });
			step->setRenderSource(buffer);
			step->setBilinearFilter(0, true);
			step->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, TEXTURE_SCALE_DOWN + i));
			source = TEXTURE_SCALE_DOWN + i;
		}
	}

	// Bloom pt 2
	if (enable_bloom) {
		// upsample
		shader_id = client->getShaderSource()->getShaderRaw("bloom_upsample");
		for (u8 i = MIPMAP_LEVELS - 1; i > 0; i--) {
			auto step = pipeline->addStep<PostProcessingStep>(shader_id, std::vector<u8> { u8(TEXTURE_SCALE_DOWN + i - 1), source });
			step->setRenderSource(buffer);
			step->setBilinearFilter(0, true);
			step->setBilinearFilter(1, true);
			step->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, u8(TEXTURE_SCALE_UP + i - 1)));
			source = TEXTURE_SCALE_UP + i - 1;
		}
	}

	// Dynamic Exposure pt2
	if (enable_auto_exposure) {
		shader_id = client->getShaderSource()->getShaderRaw("update_exposure");
		auto update_exposure = pipeline->addStep<PostProcessingStep>(shader_id, std::vector<u8> { TEXTURE_EXPOSURE_1, u8(TEXTURE_SCALE_DOWN + MIPMAP_LEVELS - 1) });
		update_exposure->setBilinearFilter(1, true);
		update_exposure->setRenderSource(buffer);
		update_exposure->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, TEXTURE_EXPOSURE_2));
	}

	// FXAA
	u8 final_stage_source = TEXTURE_COLOR;

	if (enable_fxaa) {
		final_stage_source = TEXTURE_FXAA;

		buffer->setTexture(TEXTURE_FXAA, scale, "fxaa", color_format);
		shader_id = client->getShaderSource()->getShaderRaw("fxaa");
		PostProcessingStep *effect = pipeline->createOwned<PostProcessingStep>(shader_id, std::vector<u8> { TEXTURE_COLOR });
		pipeline->addStep(effect);
		effect->setBilinearFilter(0, true);
		effect->setRenderSource(buffer);
		effect->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, TEXTURE_FXAA));
	}

	// final merge
	shader_id = client->getShaderSource()->getShaderRaw("second_stage");
	PostProcessingStep *effect = pipeline->createOwned<PostProcessingStep>(shader_id, std::vector<u8> { final_stage_source, TEXTURE_SCALE_UP, TEXTURE_EXPOSURE_2, TEXTURE_DEPTH });
	pipeline->addStep(effect);
	if (enable_ssaa)
		effect->setBilinearFilter(0, true);
	effect->setBilinearFilter(1, true);
	effect->setRenderSource(buffer);

	if (enable_auto_exposure) {
		pipeline->addStep<SwapTexturesStep>(buffer, TEXTURE_EXPOSURE_1, TEXTURE_EXPOSURE_2);
	}

	// BINARY SEARCH SWITCH (claude_bypass=1): return vanilla's tail and add
	// none of our steps, so `effect` renders straight to the screen exactly as
	// upstream does. This separates "our chain broke core-profile
	// post-processing" from "Luanti's own post-processing does not work on a
	// core profile at all" — which nothing else can distinguish, since with
	// enable_post_processing=false the whole chain is absent.
	if (g_settings->exists("claude_bypass")
			&& g_settings->getFloat("claude_bypass", 0.0f, 1.0f) > 0.5f)
		return effect;

	// claude traced-mode chain — RUNG 1, ONE PASS.
	// The final merge lands in a texture; claude_trace path-traces one
	// sample per pixel per frame into a persistent ping-pong history; a
	// present step picks the trace output (traced modes) or the merged
	// raster frame, and becomes the pipeline's returned tail.
	//
	// The five-pass chain that used to live here (claude_radiance ->
	// claude_faces -> claude_nfaces -> claude_accum -> claude_denoise)
	// and its four cache textures (RCACHE/FCACHE/NCACHE ping-pongs,
	// DENOISED) are GONE. physics-contract.md §6: photo mode is pure
	// path tracing with no next-event estimation, no caches and no
	// temporal tricks. Everything deleted here was an estimator; the
	// estimators come back one at a time, each measured against this.
	static const u8 TEXTURE_ACCUM_1 = 30;
	static const u8 TEXTURE_ACCUM_2 = 31;
	static const u8 TEXTURE_MERGED = 32;
	// the direct/bounced split's own ping-pong (claude_split, 2026-10-05)
	static const u8 TEXTURE_DIRECT_1 = 33;
	static const u8 TEXTURE_DIRECT_2 = 34;
	// THE DENOISER (claude_denoise, 2026-10-05). The tracer writes a guide
	// (accumulated albedo + which voxel face the pixel sees) and the
	// luminance moments the noise estimate needs, both ping-ponged like the
	// history; six filter passes then read them and write DEN_A/DEN_B.
	static const u8 TEXTURE_GBUF_1 = 35;
	static const u8 TEXTURE_GBUF_2 = 36;
	static const u8 TEXTURE_MOM_1 = 37;
	static const u8 TEXTURE_MOM_2 = 38;
	static const u8 TEXTURE_DEN_A = 39;
	static const u8 TEXTURE_DEN_B = 40;
	// AUTO EXPOSURE (claude_auto_exposure, 2026-10-05): a 1x1 ping-pong
	// holding the adapted scene brightness, written by claude_exposure.
	static const u8 TEXTURE_EXP_1 = 41;
	static const u8 TEXTURE_EXP_2 = 42;
	// Trace resolution, relative to the render target. 0.5 was chosen on a
	// retina laptop, where a 2x backing store downsampled the result and gave
	// free supersampling; on a plain 1080p external monitor the same 0.5 is
	// simply half the pixels you look at, and the scene reads soft while the
	// full-res UI over it stays crisp. Read at pipeline construction, so a
	// change needs a client restart.
	float trace_scale = 0.5f;
	if (g_settings->exists("claude_trace_scale"))
		trace_scale = rangelim(g_settings->getFloat("claude_trace_scale"), 0.25f, 1.0f);
	// The accumulation buffers hold HDR radiance (the sun disc alone is x40)
	// and are written directly, never alpha-blended into — so they can always
	// be float, whatever the raster path chose. Tying them to
	// post_processing_texture_bits forced a choice between two bad options:
	// leave it at 10 and clip every value above 1.0 into a NORMALISED buffer,
	// or raise it to 16 and drag the whole raster pipeline onto a float target
	// — which broke alpha blending for the sky's sun and moon quads on this
	// legacy GL driver and drew them as opaque SQUARES.
	// The history MUST be 32F, not 16F. With the true-1/N average the
	// blend weight reaches ~3e-4 after a minute parked; at fp16 the
	// per-frame correction then rounds to nothing and the "average"
	// silently sheds exactly the rare-large-sample tail that carries
	// deep-bounce energy. MEASURED 2026-08-15, furnace-073: 6.09 at
	// ~150 frames -> 4.755 at ~4500 frames on a 16F target (ratio to
	// analytic 0.94 -> 0.73). The old 0.02 alpha floor had been hiding
	// this by never letting accumulation get deep. referee: furnace-073
	// at deep settle.
	video::ECOLOR_FORMAT accum_format = color_format;
	if (driver->queryTextureFormat(video::ECF_A32B32G32R32F)
			&& driver->queryFeature(video::EVDF_RENDER_TO_FLOAT_TEXTURE))
		accum_format = video::ECF_A32B32G32R32F;
	else if (driver->queryTextureFormat(video::ECF_A16B16G16R16F)
			&& driver->queryFeature(video::EVDF_RENDER_TO_FLOAT_TEXTURE))
		accum_format = video::ECF_A16B16G16R16F;
	buffer->setTexture(TEXTURE_ACCUM_1, scale * trace_scale, "claude_accum_1", accum_format);
	buffer->setTexture(TEXTURE_ACCUM_2, scale * trace_scale, "claude_accum_2", accum_format);
	buffer->setTexture(TEXTURE_MERGED, scale, "claude_merged", color_format);
	buffer->setTexture(TEXTURE_DIRECT_1, scale * trace_scale, "claude_direct_1", accum_format);
	buffer->setTexture(TEXTURE_DIRECT_2, scale * trace_scale, "claude_direct_2", accum_format);
	buffer->setTexture(TEXTURE_GBUF_1, scale * trace_scale, "claude_gbuf_1", accum_format);
	buffer->setTexture(TEXTURE_GBUF_2, scale * trace_scale, "claude_gbuf_2", accum_format);
	buffer->setTexture(TEXTURE_MOM_1, scale * trace_scale, "claude_mom_1", accum_format);
	buffer->setTexture(TEXTURE_MOM_2, scale * trace_scale, "claude_mom_2", accum_format);
	buffer->setTexture(TEXTURE_DEN_A, scale * trace_scale, "claude_den_a", accum_format);
	buffer->setTexture(TEXTURE_DEN_B, scale * trace_scale, "claude_den_b", accum_format);
	// 2x1 (2026-10-05): texel 0 = exposure, texel 1 = the adapted white
	buffer->setTexture(TEXTURE_EXP_1, core::dimension2du(2, 1), "claude_exp_1", accum_format, /*clear:*/ true);
	buffer->setTexture(TEXTURE_EXP_2, core::dimension2du(2, 1), "claude_exp_2", accum_format, /*clear:*/ true);

	effect->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, TEXTURE_MERGED));

	// claude_trace: THE truth renderer (client/shaders/claude_trace).
	// Reads the previous frame's history (ACCUM_1) plus the volume
	// samplers game.cpp binds outside the material system, writes the
	// fresh running average to ACCUM_2. The tail swap renames ACCUM_2 to
	// ACCUM_1 for next frame, exactly as before.
	shader_id = client->getShaderSource()->getShaderRaw("claude_trace");
	PostProcessingStep *trace = pipeline->addStep<PostProcessingStep>(shader_id,
			std::vector<u8> { TEXTURE_ACCUM_1, TEXTURE_DIRECT_1,
					TEXTURE_GBUF_1, TEXTURE_MOM_1 });
	trace->setRenderSource(buffer);
	trace->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer,
			std::vector<u8> { TEXTURE_ACCUM_2, TEXTURE_DIRECT_2,
					TEXTURE_GBUF_2, TEXTURE_MOM_2 }));

	// EXTRA SAMPLES FOR THE PIXELS THAT NEED THEM (claude_subpasses,
	// 2026-10-07; DECISIONS 0w, "rays per pixel"). Two more passes of the
	// same tracer, compiled with CLAUDE_SUBPASS 1 and 2. Pass k re-traces
	// only the pixels its rule picks (later: a learned sampling map) and
	// copies every other pixel through. Ping-pong _2 -> _1 -> _2, so the
	// denoiser, the present step and the swap below still find the result
	// in the _2 textures. Read at pipeline construction (restart to change);
	// claude_boost switches the rule on and off live.
	int subpasses = 0;
	if (g_settings->exists("claude_subpasses")
			&& g_settings->getFloat("claude_subpasses", 0.0f, 2.0f) > 0.5f)
		subpasses = 2;
	for (int sp = 1; sp <= subpasses; sp++) {
		ShaderConstants sc;
		sc["CLAUDE_SUBPASS"] = sp;
		u32 sid = client->getShaderSource()->getShader("claude_trace", sc,
				video::EMT_TRANSPARENT_ALPHA_CHANNEL_REF);
		std::vector<u8> from = (sp % 2 == 1)
				? std::vector<u8> { TEXTURE_ACCUM_2, TEXTURE_DIRECT_2, TEXTURE_GBUF_2, TEXTURE_MOM_2 }
				: std::vector<u8> { TEXTURE_ACCUM_1, TEXTURE_DIRECT_1, TEXTURE_GBUF_1, TEXTURE_MOM_1 };
		std::vector<u8> to = (sp % 2 == 1)
				? std::vector<u8> { TEXTURE_ACCUM_1, TEXTURE_DIRECT_1, TEXTURE_GBUF_1, TEXTURE_MOM_1 }
				: std::vector<u8> { TEXTURE_ACCUM_2, TEXTURE_DIRECT_2, TEXTURE_GBUF_2, TEXTURE_MOM_2 };
		PostProcessingStep *extra = pipeline->addStep<PostProcessingStep>(sid, from);
		extra->setRenderSource(buffer);
		extra->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, to));
	}

	// claude_denoise: DISPLAY ONLY. It reads the running average and never
	// writes it, so the history the tracer accumulates — and every referee
	// — is exactly what it was; only what is SHOWN is filtered. Pass 0
	// estimates each pixel's remaining noise, passes 1-5 are the a-trous
	// steps 1, 2, 4, 8, 16, and pass 5 also puts the albedo back and the
	// primary distance in alpha, which is what claude_present's upsample
	// reads from its texture 1. One shader, compiled six times with the
	// pass number as a constant. Ping-pong: 0->A 1->B 2->A 3->B 4->A 5->B.
	for (int it = 0; it <= 5; it++) {
		ShaderConstants dn_consts;
		dn_consts["CLAUDE_DN_ITER"] = it;
		u32 dn_id = client->getShaderSource()->getShader("claude_denoise",
				dn_consts, video::EMT_TRANSPARENT_ALPHA_CHANNEL_REF);
		u8 out = (it % 2 == 0) ? TEXTURE_DEN_A : TEXTURE_DEN_B;
		u8 in = (it % 2 == 0) ? TEXTURE_DEN_B : TEXTURE_DEN_A;
		std::vector<u8> inputs = (it == 0)
				? std::vector<u8> { TEXTURE_ACCUM_2, TEXTURE_GBUF_2,
						TEXTURE_MOM_2, TEXTURE_DIRECT_2 }
				: std::vector<u8> { in, TEXTURE_GBUF_2, TEXTURE_ACCUM_2,
						it == 5 ? TEXTURE_DIRECT_2 : TEXTURE_MOM_2 };
		PostProcessingStep *dn = pipeline->addStep<PostProcessingStep>(dn_id,
				inputs);
		dn->setRenderSource(buffer);
		dn->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer,
				out));
	}

	// claude_exposure: one pixel. Reads what is about to be shown (DEN_B)
	// and last frame's adapted brightness, writes this frame's. claude_
	// present reads it as texture 3 (the split, which used to need that
	// slot, now happens in claude_denoise's last pass).
	shader_id = client->getShaderSource()->getShaderRaw("claude_exposure");
	PostProcessingStep *expo = pipeline->addStep<PostProcessingStep>(shader_id,
			std::vector<u8> { TEXTURE_DEN_B, TEXTURE_EXP_1, TEXTURE_GBUF_2 });
	expo->setRenderSource(buffer);
	expo->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer,
			TEXTURE_EXP_2));
	pipeline->addStep<ClaudeExposureReadback>(buffer, TEXTURE_EXP_2);

	// claude_present is UNCHANGED in its slots: texture 1 is the traced
	// lighting it upsamples and tonemaps. It used to be the denoiser's
	// output; it is now the tracer's own, straight out of ACCUM_2.
	shader_id = client->getShaderSource()->getShaderRaw("claude_present");
	PostProcessingStep *present = pipeline->createOwned<PostProcessingStep>(shader_id,
			std::vector<u8> { TEXTURE_MERGED, TEXTURE_DEN_B, TEXTURE_DEPTH,
					TEXTURE_EXP_2 });
	pipeline->addStep(present);
	// joint-bilateral upsample does its own tap weighting: keep NEAREST
	present->setRenderSource(buffer);

	pipeline->addStep<SwapTexturesStep>(buffer, TEXTURE_ACCUM_1, TEXTURE_ACCUM_2);
	// after the swap, ACCUM_1 holds this frame's accumulated radiance
	pipeline->addStep<ClaudeAccumReadback>(buffer, TEXTURE_ACCUM_1);
	pipeline->addStep<SwapTexturesStep>(buffer, TEXTURE_DIRECT_1, TEXTURE_DIRECT_2);
	pipeline->addStep<SwapTexturesStep>(buffer, TEXTURE_GBUF_1, TEXTURE_GBUF_2);
	pipeline->addStep<SwapTexturesStep>(buffer, TEXTURE_MOM_1, TEXTURE_MOM_2);
	pipeline->addStep<SwapTexturesStep>(buffer, TEXTURE_EXP_1, TEXTURE_EXP_2);
	// after every swap: the _1 textures hold this frame's, DEN_B what was shown
	pipeline->addStep<ClaudeSetReadback>(buffer,
			std::vector<u8> { TEXTURE_ACCUM_1, TEXTURE_DIRECT_1, TEXTURE_GBUF_1,
					TEXTURE_MOM_1, TEXTURE_DEN_B },
			std::vector<std::string> { "accum", "direct", "gbuf", "mom", "den" });

	return present;
}

std::string g_claude_accum_dump;
void ClaudeAccumReadback::run(PipelineContext &context)
{
	if (g_claude_accum_dump.empty())
		return;
	const std::string path = g_claude_accum_dump;
	g_claude_accum_dump.clear();
	video::ITexture *tex = buffer->getTexture(index);
	if (!tex) {
		warningstream << "[claude_accum_dump] no accumulation texture" << std::endl;
		return;
	}
	const auto fmt = tex->getColorFormat();
	const core::dimension2du sz = tex->getSize();
	const void *px = tex->lock(video::ETLM_READ_ONLY);
	if (!px)
		return;
	const size_t n = (size_t)sz.Width * sz.Height * 4;
	std::vector<float> out(n);
	if (fmt == video::ECF_A32B32G32R32F) {
		memcpy(out.data(), px, n * 4);
	} else if (fmt == video::ECF_A16B16G16R16F) {
		const u16 *h = (const u16 *)px;
		for (size_t i = 0; i < n; i++) {
			u32 v = h[i], sgn = (v >> 15) & 1, e = (v >> 10) & 31, m = v & 1023;
			float f = e == 0 ? std::ldexp((float)m, -24)
					: e == 31 ? INFINITY : std::ldexp((float)(m | 1024), (int)e - 25);
			out[i] = sgn ? -f : f;
		}
	}
	tex->unlock();
	std::ofstream o(path + ".f32", std::ios::binary);
	o.write((const char *)out.data(), out.size() * 4);
	std::ofstream j(path + ".json");
	j << "{\"w\": " << sz.Width << ", \"h\": " << sz.Height << ", \"format\": \""
			<< (fmt == video::ECF_A32B32G32R32F ? "rgba32f" : "rgba16f") << "\"}\n";
	actionstream << "[claude_accum_dump] " << sz.Width << "x" << sz.Height << " -> " << path
			<< ".f32" << std::endl;
}

std::string g_claude_set_dump;
float g_claude_set_dump_frames = 0.0f;
std::string g_claude_set_dump_extra;
unsigned g_claude_set_dumps_written = 0;
void ClaudeSetReadback::run(PipelineContext &context)
{
	if (g_claude_set_dump.empty())
		return;
	const std::string path = g_claude_set_dump;
	g_claude_set_dump.clear();
	core::dimension2du sz(0, 0);
	std::string fmts;
	for (size_t t = 0; t < idx.size(); t++) {
		video::ITexture *tex = buffer->getTexture(idx[t]);
		if (!tex)
			continue;
		const auto fmt = tex->getColorFormat();
		sz = tex->getSize();
		const void *px = tex->lock(video::ETLM_READ_ONLY);
		if (!px)
			continue;
		const size_t n = (size_t)sz.Width * sz.Height * 4;
		std::vector<float> out(n);
		if (fmt == video::ECF_A32B32G32R32F) {
			memcpy(out.data(), px, n * 4);
		} else if (fmt == video::ECF_A16B16G16R16F) {
			const u16 *h = (const u16 *)px;
			for (size_t i = 0; i < n; i++) {
				u32 v = h[i], sgn = (v >> 15) & 1, e = (v >> 10) & 31, m = v & 1023;
				float f = e == 0 ? std::ldexp((float)m, -24)
						: e == 31 ? INFINITY : std::ldexp((float)(m | 1024), (int)e - 25);
				out[i] = sgn ? -f : f;
			}
		}
		tex->unlock();
		std::ofstream o(path + "." + names[t] + ".f32", std::ios::binary);
		o.write((const char *)out.data(), out.size() * 4);
		fmts += (fmts.empty() ? "\"" : ", \"") + names[t] + "\"";
	}
	std::ofstream j(path + ".json");
	j << "{\"w\": " << sz.Width << ", \"h\": " << sz.Height << ", \"still_frames\": "
			<< g_claude_set_dump_frames << g_claude_set_dump_extra << ", \"textures\": [" << fmts << "]}\n";
	g_claude_set_dumps_written++;
	actionstream << "[claude_dump_at] " << sz.Width << "x" << sz.Height << " at "
			<< g_claude_set_dump_frames << " -> " << path << std::endl;
}

float g_claude_auto_exposure[4] = {0.0f, 0.0f, 0.0f, 0.0f};
float g_claude_white[4] = {0.0f, 0.0f, 0.0f, 0.0f};

void ClaudeExposureReadback::run(PipelineContext &context)
{
	if (frames++ % 30 != 0)
		return;
	video::ITexture *tex = buffer->getTexture(index);
	if (!tex || tex->getColorFormat() != video::ECF_A32B32G32R32F)
		return;
	const float *px = (const float *)tex->lock(video::ETLM_READ_ONLY);
	if (!px)
		return;
	// RGBA order in a 32F texel: .r factor, .g written, .b adapted, .a now
	g_claude_auto_exposure[0] = px[0];
	g_claude_auto_exposure[1] = px[2];
	g_claude_auto_exposure[2] = px[3];
	g_claude_auto_exposure[3] = px[1];
	// texel 1: the adapted white's chroma (unit luminance), written flag
	g_claude_white[0] = px[4];
	g_claude_white[1] = px[5];
	g_claude_white[2] = px[6];
	g_claude_white[3] = px[7];
	tex->unlock();
}

void ResolveMSAAStep::run(PipelineContext &context)
{
	context.device->getVideoDriver()->blitRenderTarget(msaa_fbo->getIrrRenderTarget(context),
			target_fbo->getIrrRenderTarget(context));
}
