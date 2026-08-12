// Luanti
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2010-2013 celeron55, Perttu Ahola <celeron55@gmail.com>
// Copyright (C) 2017 numzero, Lobachevskiy Vitaliy <numzer0@yandex.ru>
// Copyright (C) 2020 appgurueu, Lars Mueller <appgurulars@gmx.de>

#include "secondstage.h"
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

	// claude traced-mode chain: the final merge now lands in a texture;
	// a half-res path-traced sample accumulates into a persistent
	// ping-pong history; a present step picks accum (traced modes) or
	// the merged raster frame, and becomes the pipeline's returned tail.
	static const u8 TEXTURE_ACCUM_1 = 30;
	static const u8 TEXTURE_ACCUM_2 = 31;
	static const u8 TEXTURE_MERGED = 32;
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
	video::ECOLOR_FORMAT accum_format = color_format;
	if (driver->queryTextureFormat(video::ECF_A16B16G16R16F)
			&& driver->queryFeature(video::EVDF_RENDER_TO_FLOAT_TEXTURE))
		accum_format = video::ECF_A16B16G16R16F;
	buffer->setTexture(TEXTURE_ACCUM_1, scale * trace_scale, "claude_accum_1", accum_format);
	buffer->setTexture(TEXTURE_ACCUM_2, scale * trace_scale, "claude_accum_2", accum_format);
	buffer->setTexture(TEXTURE_MERGED, scale, "claude_merged", color_format);

	effect->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, TEXTURE_MERGED));

	// claude_radiance: world-space radiance cache, updated on the GPU as a
	// fragment pass. GL 4.1 has no image store and cannot FBO-attach every
	// layer of a 3D texture at once, so the 64^3 cache (2-node cells, same
	// 128-node footprint as the volume) is FLATTENED: 512x512 = an 8x8 grid
	// of 64x64 tiles, one per z-slice, addressed manually in the shaders.
	// Ping-pong pair, fixed size (not screen-scaled), float so torch-level
	// HDR values survive; clear:true so the first frames read zeros, not
	// uninitialized memory. Update order per frame: radiance pass reads
	// RCACHE_1 -> writes RCACHE_2; accum reads the fresh RCACHE_2; the swap
	// at the pipeline tail renames it to RCACHE_1 for next frame.
	// The pass early-outs (writes zeros) while claude_radiance is 0, so its
	// standing cost when off is a 512x512 fill — negligible.
	static const u8 TEXTURE_RCACHE_1 = 34;
	static const u8 TEXTURE_RCACHE_2 = 35;
	buffer->setTexture(TEXTURE_RCACHE_1, core::dimension2du(512, 512),
			"claude_rcache_1", accum_format, /*clear:*/ true);
	buffer->setTexture(TEXTURE_RCACHE_2, core::dimension2du(512, 512),
			"claude_rcache_2", accum_format, /*clear:*/ true);

	shader_id = client->getShaderSource()->getShaderRaw("claude_radiance");
	PostProcessingStep *radiance = pipeline->addStep<PostProcessingStep>(shader_id,
			std::vector<u8> { TEXTURE_RCACHE_1 });
	radiance->setRenderSource(buffer);
	radiance->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, TEXTURE_RCACHE_2));

	shader_id = client->getShaderSource()->getShaderRaw("claude_accum");
	PostProcessingStep *accum = pipeline->addStep<PostProcessingStep>(shader_id,
			std::vector<u8> { TEXTURE_ACCUM_1, TEXTURE_RCACHE_2 });
	accum->setRenderSource(buffer);
	accum->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, TEXTURE_ACCUM_2));

	// edge-aware spatial denoise on the display path only — history
	// accumulates raw, so the filter never compounds
	static const u8 TEXTURE_DENOISED = 33;
	buffer->setTexture(TEXTURE_DENOISED, scale * trace_scale, "claude_denoised", color_format);
	shader_id = client->getShaderSource()->getShaderRaw("claude_denoise");
	PostProcessingStep *denoise = pipeline->addStep<PostProcessingStep>(shader_id,
			std::vector<u8> { TEXTURE_ACCUM_2 });
	denoise->setRenderSource(buffer);
	denoise->setRenderTarget(pipeline->createOwned<TextureBufferOutput>(buffer, TEXTURE_DENOISED));

	shader_id = client->getShaderSource()->getShaderRaw("claude_present");
	PostProcessingStep *present = pipeline->createOwned<PostProcessingStep>(shader_id,
			std::vector<u8> { TEXTURE_MERGED, TEXTURE_DENOISED, TEXTURE_DEPTH });
	pipeline->addStep(present);
	// joint-bilateral upsample does its own tap weighting: keep NEAREST
	present->setRenderSource(buffer);

	pipeline->addStep<SwapTexturesStep>(buffer, TEXTURE_ACCUM_1, TEXTURE_ACCUM_2);
	pipeline->addStep<SwapTexturesStep>(buffer, TEXTURE_RCACHE_1, TEXTURE_RCACHE_2);

	return present;
}

void ResolveMSAAStep::run(PipelineContext &context)
{
	context.device->getVideoDriver()->blitRenderTarget(msaa_fbo->getIrrRenderTarget(context),
			target_fbo->getIrrRenderTarget(context));
}
