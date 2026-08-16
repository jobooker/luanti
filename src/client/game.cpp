// Luanti
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2010-2013 celeron55, Perttu Ahola <celeron55@gmail.com>

#include "game_internal.h"

#include <cmath>
#include <csignal>
#include "client/gameui.h"
#include "client/inputhandler.h"
#include "client/texturepaths.h"
#include "client/keys.h"
#include "client/joystick_controller.h"
#include "client/mapblock_mesh.h"
#include "client/sound.h"
#include "clientmap.h"
#include "clientmedia.h" // For clientMediaUpdateCacheCopy
#include "config.h"
#include "content_cao.h"
#include "filesys.h" // claude_dial_file: resolve relative paths against path_user
#include "content/subgames.h"
#include "client/event_manager.h"
#include "fontengine.h"
#include "itemdef.h"
#include "gameparams.h"
#include "gettext.h"
#include "gui/guiChatConsole.h"
#include "texturesource.h"
#include "gui/mainmenumanager.h"
#include "gui/profilergraph.h"
#include "localplayer.h"
#include "minimap.h"
#include "network/networkexceptions.h"
#include "nodedef.h"         // Needed for determining pointing to nodes
#include "node_visuals.h"    // claude_grid: per-nodetype minimap_color
#include "client/claude_lod.h" // far cascade (Phase 1)
#include "client/claude_nodebox.h" // real geometry from ContentFeatures::node_box
#include "nodemetadata.h"
#include "particles.h"
#include "porting.h"
#include <fstream>
#include <json/json.h> // claude_models manifest (phase 4.5)
#include <sstream>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <set> // claude_grid: legacy setting-name warning, once per name
#include <cstring>
#include <iomanip> // claude_stats: hex grid_hash
// claude_grid uploads its 3D textures through raw GL directly.
// Cross-platform GL via the engine's own loader (irr/include/mt_opengl.h):
// it supplies BOTH entry points and enums as members of the global `GL`,
// and LoadAllProcedures() runs on both driver paths. The old
// <OpenGL/gl.h> was Apple-only, and on Windows opengl32.dll exports just
// GL 1.1 — ActiveTexture/TexImage3D/TexSubImage3D would have been null.
#include <mt_opengl.h>
// Legacy non-core fallback only (claudeUseR8() == false); mt_opengl omits
// these because core profile removed them.
#ifndef GL_LUMINANCE
#define GL_LUMINANCE 0x1909
#endif
#ifndef GL_LUMINANCE8
#define GL_LUMINANCE8 0x8040
#endif
// mt_opengl keeps its scalar typedefs private, so declare the three we
// use with their standard Khronos definitions.
using GLint = int;
using GLenum = unsigned int;
using GLuint = unsigned int;
#include "client/render/pipeline.h"
// ClaudeGpuProf (per-pass GPU times)
#include "profiler.h"
#include "raycast.h"
#include "server.h"
#include "settings.h"
#include "shader.h"
#include "sound_maker.h"
#include "threading/lambda.h"
#include "translation.h"
#include "util/basic_macros.h"
#include "util/directiontables.h"
#include "util/quicktune_shortcutter.h"
#include "version.h"
#include "script/scripting_client.h"
#include "hud.h"
#include <AnimatedMeshSceneNode.h>
#include <ICameraSceneNode.h>
#include "util/tracy_wrapper.h"
#include "item_visuals_manager.h"

#if USE_SOUND
	#include "client/sound/sound_openal.h"
#endif

typedef s32 SamplerLayer_t;

// claude_grid: one-shot 128^3 occupancy snapshot of the map around the
// camera, held as a raw GL.R8 3D texture bound to texture unit 4 — outside
// Irrlicht's material system, which only manages units 0-3, so nothing else
// touches the binding. Written by claudeTraceGridSnapshot() (triggered through
// claude_settings_patch.conf), read each frame by the uniform setter below
// and marched in the second_stage shader when claude_grid_debug is set.
struct ClaudeTraceGrid
{
	static constexpr int SIZE = 128;
	u32 tex = 0; // GL texture name (GLuint)
	u32 coarse_tex = 0; // 32^3 any-solid brick map (unit 5): rays leap empty bricks
	v3s16 origin; // node coords of voxel (0,0,0)
	bool valid = false;
	u64 last_snap_ms = 0;
	u64 content_hash = 0;
	// nearest emissive cells (grid cell coords + intensity), for NEE
	float emitters[8][4] = {};
	// Measured flame size (cell units): RMS spread of the model's
	// emissive voxels about the glow centroid, x1.6, computed in the
	// loader. NOT consumed by any shader — the jittered-target NEE that
	// briefly used it was REVERTED (John, 2026-08-13: sub-voxel mutual
	// lighting gets weird with randomized targets; proper emissive-voxel
	// transport first). Kept as model data for the future NEE+MIS block.
	float emitter_rad[8] = {};
	int emitter_count = 0;      // static (snapshot) emitters
	int emitter_runtime = 0;    // static + held light this frame
	// AREA EMITTERS (rung 2, next-event estimation). A SECOND list, and
	// deliberately not a reuse of emitters[] above, which cannot serve:
	// that one holds POINT lights (torch flames, model glow centroids)
	// with a scalar intensity, and it explicitly EXCLUDES the emissive
	// cells (ADR-0009 #4) that are the only thing claude_trace's
	// cellEmission() answers for. §4 forbids the reuse anyway — "a light
	// is not a point", and a point has no area to sample.
	//
	// One entry per emissive CELL, class 170..240:
	//   xyz = integer grid-cell coords (the DDA's own cell space)
	//   w   = air-exposed face mask; bit 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z,
	//         5 -Z. A face whose neighbour is non-air is dropped: in
	//         rung 1 every non-air class is opaque, so that is a face no
	//         ray can reach, and dropping it keeps the light-sampling pdf
	//         identical on both sides of the MIS weight.
	// NO RADIANCE IS STORED HERE, on purpose. THE LAW is one Le (§4): the
	// shader re-reads the cell out of claudeTraceGrid and runs the same
	// cellAlbedo()/cellEmission() the eye ray runs. A CPU-side copy of Le
	// would be exactly the second emission formula the contract forbids.
	static constexpr int AREA_CAP = 16;
	float area[AREA_CAP][4] = {};
	int area_count = 0;         // live slots, <= AREA_CAP
	// AREA_TOTAL IS NOT AN OCCUPANCY COUNT (found 2026-08-15, roadmap
	// 1b): this is the count of emissive CELLS the snapshot found for
	// next-event estimation, class 170..240 only. It is legitimately 0
	// for a fully solid, fully lightless sealed room -- every referee
	// room built before 1b happened to have a lamp, so `area_total > 0`
	// silently worked as an "is there a grid" proxy for months and was
	// never actually testing that. See solid_count below for the real one.
	int area_total = 0;         // emissive cells the snapshot actually found
	// solid_count / snap_seq (roadmap 1b, gate hardening 2026-08-16):
	// `valid` is set true once and never cleared (see below), so
	// `grid_valid == 1` alone is a TAUTOLOGY after the first snapshot
	// ever taken on a seat -- an all-air bubble at a brand-new vantage
	// still reads valid=1. solid_count is the number of non-air cells
	// the WALK JUST COMPLETED actually found (computed unconditionally,
	// every call, before the unchanged-content early return), so it is
	// zero exactly when the bubble really is empty. snap_seq increments
	// once per call to claudeTraceGridSnapshot(), so a caller can prove a
	// NEW walk happened between two reads rather than reading a stale
	// solid_count left over from a walk at a completely different
	// vantage.
	u32 solid_count = 0;
	u32 snap_seq = 0;
	// accum_resets: how many times the accumulator has been ZEROED (not
	// clamped) since the client started. Exists so a harness can PROVE a
	// reset happened rather than infer it from a sampled value.
	//
	// Why it had to exist (2026-08-16, settle calibration). CI resets
	// between arms by teleporting away and back, then proves it by
	// polling still_frames for a value <= 3 or below what it read
	// before. Both clauses look through a 1 Hz keyhole:
	// claude_stats.json is rewritten once per second and the counter
	// climbs at 70-90 per second, so the window in which a reset frame
	// is visible as "<= 3" is ~0.03 s wide, and the "below before"
	// clause is unusable whenever `before` is ALREADY small -- which is
	// exactly the cozy-day-dark-ci case, where the previous arm's lamp
	// swap had clamped it to 11. Measured: the check timed out reporting
	// "still_frames never dropped after reset (was 11, last seen 1353)"
	// for a reset the engine guarantees and the aim guard independently
	// confirms. That false RED had already been recorded once, in
	// spec/measured.md "Gate 4, second half", as part of a flake blamed
	// wholly on the grass ABM.
	//
	// A counter cannot be missed by sampling: it only goes up, so any
	// two reads bracket every reset between them. Instruments terminate;
	// thresholds loop.
	u32 accum_resets = 0;
	// ---- INCREMENTAL RE-SNAP (2026-08-16) ------------------------------
	// The CPU mirrors of the uploaded grids, kept alive BETWEEN
	// snapshots so a changed 16^3 block can be re-walked in place and
	// uploaded as a sub-box. They used to be locals of the snapshot
	// function, which is why the only way to reflect a dug node was to
	// re-read all 2.1 M cells (spec/measured.md "Volume re-snap fix").
	static constexpr int BLK = 16;                // grid block edge, cells
	static constexpr int NB = SIZE / BLK;         // 8 blocks per axis
	static constexpr int NBLOCKS = NB * NB * NB;  // 512
	std::vector<u8> occ;      // SIZE^3 RGBA: rgb = colour, a = class
	std::vector<u8> mids;     // SIZE^3 material ids
	std::vector<u8> pyr[6];   // occupancy mip pyramid, 128..4 cubed
	// PER-BLOCK HASHES, combined order-independently into content_hash.
	// The old content_hash was an FNV chain over every cell in index
	// order, which cannot be updated for a sub-box at all — so the whole
	// "unchanged -> skip the upload and don't disturb accumulation"
	// early return depended on re-walking everything.
	u64 block_hash[NBLOCKS] = {};
	u32 block_solid[NBLOCKS] = {};
	// Point emitters carry the cell that produced them, so a block
	// re-walk can drop exactly the ones that block owned (a dug lamp that
	// stays in this list keeps getting sampled by NEE, and the room stays
	// lit with nothing in the log) and so the list can be restored to
	// whole-grid cell order before the nearest-8 distance sort.
	struct PointEmit { u32 cell; float v[5]; };
	std::vector<PointEmit> emit_all;
	// Held (wielded) light lives in its OWN slot, never in emitters[]:
	// writing it into emitters[7] STOMPED the 8th-nearest real torch in
	// place, and the content-hash snapshot gate preserved the corruption
	// indefinitely — a dark pool around the stomped torch plus a phantom
	// light at a stale camera position ("circular shadow" bug,
	// 2026-08-12). w = 0 means no held light this frame.
	float held_emitter[4] = {};
	// textured-albedo path: per-cell material id grid (unit 6) + a
	// 256x256 atlas of 16px top-tile images (unit 7), palette grown lazily
	u32 material_tex = 0;
	u32 atlas_tex = 0;
	u32 matparams_tex = 0;      // 256x1 per-material: R=spec G=gloss B=ore
	// REAL SUB-VOXEL BITS: 16^3 bits per node for the 32^3 ring at
	// grid-local [48,80)^3 — model masks or full-solid, baked per
	// snapshot (ADR-0011: the ONLY sub-voxel occupancy); the shader's
	// microSolid is one fetch. R8 64x512x512 (one byte = 8 x-subvoxels):
	// integer samplers silently kill the Irrlicht material (the
	// flat-blue outage), so bits ride a float sampler with floor/mod
	// extraction.
	u32 subvox_tex = 0;
	std::vector<u8> subvox;
	// AUTHORED MODELS (phase 4.5, ADR-0005/0010): 16^3 occupancy masks
	// loaded from <path_user>/util/claude_models/*.json per the
	// manifest, pre-rotated to the four facedir yaws (512 bytes each,
	// bit = (z*16+y)*16+x). modelids tags snapshot cells: idx<<2 | rot,
	// 0 = no model. The bake substitutes these bits for the carve law.
	std::vector<std::array<std::vector<u8>, 4>> models;
	std::unordered_map<content_t, u8> model_of; // content -> 1-based idx
	bool models_loaded = false;
	std::vector<u8> modelids;
	// v2: per-voxel COLOR + EMISSION. Palette-indexed (the plan's "not
	// really a map" answer): each model ships 4 pre-rotated 16^3 index
	// grids into a 16x16x1024 R8 atlas (layer = (idx0*4+rot)*16+sz) and
	// a 256x64 RGBA palette row (rgb + emit/15 in alpha). Emissive
	// models also carry a per-rot glow centroid so the snapshot can
	// stand a point light at the mouth (the small-emitter law) while
	// the voxels render the visible fire.
	std::vector<std::array<std::vector<u8>, 4>> model_vox;
	std::vector<std::vector<u8>> model_pal;      // 256*4 RGBA each
	std::vector<std::array<std::array<float, 5>, 4>> model_glow; // xyz,int,radius
	u32 model_ids_tex = 0, model_atlas_tex = 0, model_pal_tex = 0;
	bool model_tex_dirty = false;
	// REAL GEOMETRY FROM THE NODE DEFINITION (2026-08-16, coverage 2).
	// Stairs, slabs and beds already carry their true shape in
	// ContentFeatures::node_box, and vanilla's own raycast reads it. The
	// converter turns that box list into the SAME 16^3 mask the authored
	// models produce, so there is one sub-voxel format and one consumer.
	//
	// Cached by (content << 8) | param2, NOT by content alone:
	// transformNodeBox rotates a NODEBOX_FIXED box list by facedir, so one
	// stair node at two param2 values is two different shapes. That is
	// also why the block hash mixes the mask id below — see the walk.
	//
	// Ring-local (32^3), because the sub-voxel ring is [48,80)^3 and a
	// cell outside it has nowhere to put bits: outside the ring a nodebox
	// node stays an honest 1 m cube, exactly as it is today.
	static constexpr int NBOX_R0 = 48, NBOX_R1 = 80;
	static constexpr int NBOX_RING = NBOX_R1 - NBOX_R0;
	static constexpr size_t NBOX_CAP = 4096; // distinct shapes; 2 MB
	bool nodebox_on = true;
	std::vector<u16> nbox_ids;                   // RING^3, 0 = none
	std::vector<std::array<u8, 512>> nbox_masks; // 1-based via nbox_ids
	std::unordered_map<u32, u16> nbox_of;        // shape cache
	std::unordered_map<content_t, u8> palette;
	std::vector<u8> atlas; // BGRA
	std::vector<u8> matparams;  // 256 RGBA rows, indexed by material id
	bool atlas_dirty = false;
	// temporal accumulation state (updated once per frame)
	v3f prev_cam_pos;
	v3f prev_cam_dir;
	v3s16 prev_origin;
	float accum_alpha = 1.0f;
	v3f prev_light_dir;          // last frame's sun/moon direction
	v3f prev_light_col;          // last frame's sun/moon colour
	int light_body = 0;          // 0 none, 1 sun, 2 moon (for stats)
	float still_frames = 0.0f;
	// last frame's ray-camera basis (grid-local), for reprojection:
	// shader_* is what the shader sees (frame N-1); cur_* staged this frame
	v3f shader_prev_pos, shader_prev_fwd, shader_prev_rightu, shader_prev_upu;
	float shader_prev_tanx = 1.0f, shader_prev_tany = 1.0f;
	v3f cur_pos, cur_fwd, cur_rightu, cur_upu;
	float cur_tanx = 1.0f, cur_tany = 1.0f;
	// near-ring sub-face atlas (ADR-0006 v2): ring corner in grid-local
	// cell coords, plus last frame's for cross-shift address remapping
	v3f near_origin = v3f(48.0f, 48.0f, 48.0f);
	v3f near_prev = v3f(48.0f, 48.0f, 48.0f);
	// grid-rebase delta in cells (origin_new - origin_old) on the shift
	// frame, zero otherwise: lets cache passes REMAP instead of zeroing
	// (the pulse-to-black John caught 2026-08-12)
	v3f origin_delta = v3f(0.0f, 0.0f, 0.0f);
	// far cascades (claude_lod Phase 2+3): detail degrades in OCTAVES —
	// 2 m to +/-128, 4 m to +/-256, 8 m to +/-512, 16 m to +/-1024,
	// 32 m to +/-2048 — so the eye never jumps more than one resolution
	// doubling at a seam. Slabs 0..4 of the stacked textures.
	u32 cascades_tex = 0;        // 128x128x640 RGBA8, unit 8
	u32 cascades_coarse_tex = 0; // 32x32x160 R8, unit 9
	struct CascLevel {
		v3s16 origin;            // WORLD node coords of cell (0,0,0)
		bool valid = false;
		u64 version = 0;         // summary contentVersion at last build
		u64 build_time = 0;
		u32 solid = 0;
		float ms = 0.0f;         // last build+upload cost (stats)
	} casc[5];                   // [0]=2m [1]=4m [2]=8m [3]=16m [4]=32m
};
static ClaudeTraceGrid g_claude_grid;


// Throw away the running average and start it again, exactly the way a
// camera move does (see the accum_alpha block in claudeUpdateGrid).
//
// A DEBUG KEY THAT CHANGES WHAT IS RENDERED MUST CALL THIS. At a parked
// camera accum_alpha is ~1/N, so a frame rendered under the old dial
// state survives in the running average for thousands of frames and the
// A/B the key exists to perform compares one blend against another. The
// shader header records the same trap for the debug views (views 1-5
// write their deterministic image straight into the ping-pong history)
// and says the way out is to reset the way the CPU already knows how.
//
// accum_resets is a COUNTER, not a flag, and it is bumped here for the
// reason spec/environment-laws.md gives: claude_stats.json is rewritten
// once per second, so any check that samples still_frames can miss a
// reset entirely, while two reads of a counter bracket every reset
// between them.
static void claudeResetAccumulation()
{
	g_claude_grid.accum_alpha = 1.0f;
	g_claude_grid.still_frames = 0.0f;
	g_claude_grid.accum_resets++;
}

class GameGlobalShaderUniformSetter : public IShaderUniformSetter
{
	Sky *m_sky;
	Client *m_client;

	CachedVertexShaderSetting<float> m_animation_timer_vertex{"animationTimer"};
	CachedPixelShaderSetting<float> m_animation_timer_pixel{"animationTimer"};
	CachedVertexShaderSetting<float>
		m_animation_timer_delta_vertex{"animationTimerDelta"};
	CachedPixelShaderSetting<float>
		m_animation_timer_delta_pixel{"animationTimerDelta"};
	int m_crack_animation_length_i;
	CachedPixelShaderSetting<float> m_crack_animation_length{"crackAnimationLength"};
	int m_crack_level_i = -1;
	CachedPixelShaderSetting<float> m_crack_level{"crackLevel"};
	int m_crack_texture_scale_i = 0;
	CachedPixelShaderSetting<float> m_crack_texture_scale{"crackTextureScale"};
	CachedPixelShaderSetting<float, 3> m_day_light{"dayLight"};
	CachedPixelShaderSetting<float, 3> m_minimap_yaw{"yawVec"};
	CachedPixelShaderSetting<float, 3> m_camera_offset_pixel{"cameraOffset"};
	CachedVertexShaderSetting<float, 3> m_camera_offset_vertex{"cameraOffset"};
	CachedPixelShaderSetting<float, 3> m_camera_position_pixel{"cameraPosition"};
	CachedVertexShaderSetting<float, 3> m_camera_position_vertex{"cameraPosition"};
	CachedVertexShaderSetting<float, 2> m_texel_size0_vertex{"texelSize0"};
	CachedPixelShaderSetting<float, 2> m_texel_size0_pixel{"texelSize0"};
	v2f m_texel_size0;

	CachedStructPixelShaderSetting<float, 7> m_exposure_params_pixel{
		"exposureParams",
		std::array<const char*, 7> {
			"luminanceMin", "luminanceMax", "exposureCorrection",
			"speedDarkBright", "speedBrightDark", "centerWeightPower",
			"compensationFactor"
		}};
	float m_user_exposure_compensation;
	bool m_bloom_enabled;
	CachedPixelShaderSetting<float> m_bloom_intensity_pixel{"bloomIntensity"};
	CachedPixelShaderSetting<float> m_bloom_strength_pixel{"bloomStrength"};
	CachedPixelShaderSetting<float> m_bloom_radius_pixel{"bloomRadius"};
	CachedPixelShaderSetting<float> m_saturation_pixel{"saturation"};
	CachedPixelShaderSetting<float> m_day_night_ratio_pixel{"dayNightRatio"};
	CachedPixelShaderSetting<float> m_golden_hour_pixel{"goldenHourStrength"};
	float m_golden_hour_strength;
	CachedPixelShaderSetting<float> m_ssao_strength_pixel{"ssaoStrength"};
	float m_ssao_strength;
	CachedPixelShaderSetting<float> m_bump_strength_pixel{"bumpStrength"};
	float m_bump_strength;
	CachedPixelShaderSetting<SamplerLayer_t, 1, false> m_grid_sampler_pixel{"claudeTraceGrid"};
	CachedPixelShaderSetting<SamplerLayer_t> m_coarse_sampler_pixel{"claudeCoarse"};
	CachedPixelShaderSetting<SamplerLayer_t> m_materials_sampler_pixel{"claudeMaterials"};
	CachedPixelShaderSetting<SamplerLayer_t> m_atlas_sampler_pixel{"claudeAtlas"};
	CachedPixelShaderSetting<SamplerLayer_t, 1, false> m_matparams_sampler_pixel{"claudeMatParams"};
	CachedPixelShaderSetting<SamplerLayer_t, 1, false> m_cascades_sampler_pixel{"claudeCascades"};
	CachedPixelShaderSetting<SamplerLayer_t, 1, false> m_cascades_coarse_sampler_pixel{"claudeCascadeCoarse"};
	CachedPixelShaderSetting<float, 3, false> m_cascade0_origin_pixel{"cascade0Origin"};
	CachedPixelShaderSetting<float, 3, false> m_cascade1_origin_pixel{"cascade1Origin"};
	CachedPixelShaderSetting<float, 3, false> m_cascade2_origin_pixel{"cascade2Origin"};
	CachedPixelShaderSetting<float, 3, false> m_cascade3_origin_pixel{"cascade3Origin"};
	CachedPixelShaderSetting<float, 3, false> m_cascade4_origin_pixel{"cascade4Origin"};
	CachedPixelShaderSetting<float, 3, false> m_cascade_valid_pixel{"cascadeValid"};
	CachedPixelShaderSetting<float, 3, false> m_cascade_valid2_pixel{"cascadeValidB"};
	CachedPixelShaderSetting<float, 3, false> m_grid_origin_pixel{"gridOrigin"};
	CachedPixelShaderSetting<float, 1, false> m_texture_amount_pixel{"textureAmount"};
	CachedPixelShaderSetting<float, 1, false> m_gray_pixel{"grayWorld"};
	CachedPixelShaderSetting<float, 1, false> m_pure_pixel{"purePhoto"};
	CachedPixelShaderSetting<float, 1, false> m_bevel_pixel{"bevelStrength"};
	CachedPixelShaderSetting<float, 1, false> m_relief_pixel{"reliefStrength"};
	CachedPixelShaderSetting<float, 1, false> m_parallax_pixel{"parallaxStrength"};
	CachedPixelShaderSetting<float, 1, false> m_jitter_pixel{"jitterStrength"};
	CachedPixelShaderSetting<float, 1, false> m_skybounce_pixel{"skyBounce"};
	CachedPixelShaderSetting<float, 1, false> m_sunangle_pixel{"sunAngle"};
	CachedPixelShaderSetting<float, 1, false> m_nightsky_pixel{"nightSkyGain"};
	float m_texture_amount, m_gray, m_bevel, m_relief, m_parallax, m_jitter;
	float m_pure = 0.0f;
	float m_skybounce, m_sunangle, m_nightsky, m_moongain;
	float m_bounce2 = 0.0f;
	CachedPixelShaderSetting<float, 1, false> m_bounce2_pixel{"bounce2Strength"};
	float m_cache_sky = 0.0f;
	CachedPixelShaderSetting<float, 1, false> m_cache_sky_pixel{"cacheSkyStrength"};
	float m_bisect = 0.0f;
	CachedPixelShaderSetting<float, 1, false> m_bisect_pixel{"claudeBisect"};
	float m_pyramid = 0.0f;
	CachedPixelShaderSetting<float, 1, false> m_pyramid_pixel{"claudePyramid"};
	float m_nee_gate = 0.0f;
	CachedPixelShaderSetting<float, 1, false> m_nee_gate_pixel{"claudeNeeGate"};
	float m_cost = 0.0f;
	CachedPixelShaderSetting<float, 1, false> m_cost_pixel{"claudeCost"};
	float m_face_direct = 0.0f;
	CachedPixelShaderSetting<float, 1, false> m_face_direct_pixel{"claudeFaceDirect"};
	float m_tiers = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_tiers_pixel{"claudeTiers"};
	float m_bounce_stride = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_bounce_stride_pixel{"claudeBounceStride"};
	float m_face_texels = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_face_texels_pixel{"claudeFaceTexels"};
	float m_cache_remap = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_cache_remap_pixel{"claudeCacheRemap"};
	float m_far_hist = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_far_hist_pixel{"claudeFarHist"};
	float m_light_ladder = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_light_ladder_pixel{"claudeLightLadder"};
	float m_lod_dither = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_lod_dither_pixel{"claudeLodDither"};
	float m_far_grain = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_far_grain_pixel{"claudeFarGrain"};
	float m_far_fog = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_far_fog_pixel{"claudeFarFog"};
	float m_sky_az = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_sky_az_pixel{"claudeSkyAz"};
	float m_subvox = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_subvox_pixel{"claudeSubvox"};
	// SUB-VOXEL DESCENT (2026-08-16). 1 (default) = march() steps into a
	// class-250 cell's 16^3 mask; 0 = the pre-descend behaviour, a 250
	// cell is an opaque 1 m cube. THE A/B PARTNER for the energy and
	// cost gates: one variable, no rebuild between arms.
	//
	// A NEW NAME rather than reviving claudeSubvox, deliberately. That
	// dial is in the dead-uniform list AND the seat conf pins it to 0
	// (util/claude_seat_conf.ref), so wiring descent onto it would have
	// shipped the whole step switched off, silently, on the one seat
	// that measures it — the hidden-default class, self-inflicted.
	float m_descend = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_descend_pixel{"claudeDescend"};
	float m_refine = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_refine_pixel{"claudeRefine"};
	float m_denoise = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_denoise_pixel{"claudeDenoise"};
	// claude_trace diagnostic view: 0 = photo (the truth renderer, and
	// the only mode a referee may be run against), 1 = first-hit normal
	// as a six-step GRAY ladder, 2 = albedo, 3 = Le, 4 = distance,
	// 5 = bounce count. Gray, not RGB: there are exactly six cardinal
	// normals, so a wrong one reads as a wrong brightness patch.
	float m_view = 0.0f;
	CachedPixelShaderSetting<float, 1, false> m_view_pixel{"claudeView"};
	// claude_trace path-depth cap: 0 = primary emission only (you see
	// only what emits), 1 = direct light only, 24 = full transport with
	// the Russian-roulette schedule. The direct/indirect separation
	// switch — no extra view modes needed for it.
	float m_bounces = 24.0f;
	CachedPixelShaderSetting<float, 1, false> m_bounces_pixel{"claudeBounces"};
	// claude_trace transport mode: 0 = the pure photo path (§6 truth
	// mode — no next-event estimation at all, byte-identical to rung 1),
	// 1 = next-event estimation with MIS. THE A/B SWITCH: the two must
	// agree in expectation at a parked camera, and a disagreement is a
	// bug in the estimator, never a reason to retune the truth (§6).
	float m_nee = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_nee_pixel{"claudeNee"};
	// claude_trace RNG source. 1 (DEFAULT since roadmap 1a) = a
	// counter-based PCG keyed on (pixel, frame, draw index). 0 = the
	// rung-1 sine-free hash CHAIN (g_rngState = hash11(g_rngState +
	// phi)), kept reachable for one release as the A/B partner and
	// documented as THE OLD, BIASED CHAIN: an iterated float hash is a
	// trajectory, not a generator, and its successive-draw structure
	// biased the cosine-hemisphere sampler by ~9% against a compact
	// overhead light — deterministically, which is why it reproduced to
	// four decimals across seats. It made PHOTO MODE 1-8% dark in
	// Cornell (spec/measured.md "1a"), so every Cornell number taken
	// before that line was measured against a dark truth renderer.
	float m_rng = 1.0f;
	CachedPixelShaderSetting<float, 1, false> m_rng_pixel{"claudeRng"};
	CachedPixelShaderSetting<SamplerLayer_t, 1, false> m_subvox_sampler_pixel{"claudeSubvoxTex"};
	CachedPixelShaderSetting<SamplerLayer_t, 1, false> m_modelids_sampler_pixel{"claudeModelIds"};
	CachedPixelShaderSetting<SamplerLayer_t, 1, false> m_modelatlas_sampler_pixel{"claudeModelAtlas"};
	CachedPixelShaderSetting<SamplerLayer_t, 1, false> m_modelpal_sampler_pixel{"claudeModelPal"};
	CachedPixelShaderSetting<float, 3, false> m_origin_delta_pixel{"claudeOriginDelta"};
	CachedPixelShaderSetting<float, 3, false> m_near_origin_pixel{"claudeNearOrigin"};
	CachedPixelShaderSetting<float, 3, false> m_near_prev_pixel{"claudeNearPrev"};
	CachedPixelShaderSetting<float, 1, false> m_grid_debug_pixel{"gridDebug"};
	CachedPixelShaderSetting<float, 3, false> m_grid_cam_pos_pixel{"gridCamPos"};
	CachedPixelShaderSetting<float, 3, false> m_grid_cam_fwd_pixel{"gridCamFwd"};
	CachedPixelShaderSetting<float, 3, false> m_grid_cam_right_pixel{"gridCamRight"};
	CachedPixelShaderSetting<float, 3, false> m_grid_cam_up_pixel{"gridCamUp"};
	CachedPixelShaderSetting<float, 3, false> m_volume_sun_dir_pixel{"volumeSunDir"};
	CachedPixelShaderSetting<float, 3, false> m_volume_light_col_pixel{"volumeLightCol"};
	CachedPixelShaderSetting<float, 2, false> m_volume_depth_range_pixel{"volumeDepthRange"};
	CachedPixelShaderSetting<float> m_water_refl_pixel{"waterReflStrength"};
	CachedPixelShaderSetting<float> m_gi_strength_pixel{"giStrength"};
	CachedPixelShaderSetting<float> m_gi_split_pixel{"giSplit"};
	CachedPixelShaderSetting<float> m_clay_pixel{"clayStrength"};
	CachedPixelShaderSetting<float> m_accum_alpha_pixel{"accumAlpha"};
	CachedPixelShaderSetting<float, 3, false> m_prev_pos_pixel{"prevCamPos"};
	CachedPixelShaderSetting<float, 3, false> m_prev_fwd_pixel{"prevCamFwd"};
	CachedPixelShaderSetting<float, 3, false> m_prev_rightu_pixel{"prevCamRightU"};
	CachedPixelShaderSetting<float, 3, false> m_prev_upu_pixel{"prevCamUpU"};
	CachedPixelShaderSetting<float, 2, false> m_prev_tan_pixel{"prevCamTan"};
	CachedPixelShaderSetting<float, 4, false> m_emitter_pixel[8] = {
		{"claudeEmitter0"}, {"claudeEmitter1"}, {"claudeEmitter2"},
		{"claudeEmitter3"}, {"claudeEmitter4"}, {"claudeEmitter5"},
		{"claudeEmitter6"}, {"claudeEmitter7"}};
	CachedPixelShaderSetting<float> m_emitter_count_pixel{"claudeEmitterCount"};
	CachedPixelShaderSetting<float, 4, false> m_held_emitter_pixel{"claudeHeldEmitter"};
	// AREA emitters for NEE — ONE UNIFORM PER SLOT, exactly the shape
	// claudeEmitter0..7 has always had. Not a `uniform vec4 a[16]`: GL
	// reports an array uniform's name as "a[0]" on some drivers and "a"
	// on others, and COpenGLSLMaterialRenderer::getPixelShaderConstantID
	// does a literal string compare against whatever glGetActiveUniform
	// handed back — so an array would resolve on one driver and silently
	// no-op on the next. Sixteen scalars resolve everywhere.
	CachedPixelShaderSetting<float, 4, false> m_area_pixel[ClaudeTraceGrid::AREA_CAP] = {
		{"claudeArea0"}, {"claudeArea1"}, {"claudeArea2"}, {"claudeArea3"},
		{"claudeArea4"}, {"claudeArea5"}, {"claudeArea6"}, {"claudeArea7"},
		{"claudeArea8"}, {"claudeArea9"}, {"claudeArea10"}, {"claudeArea11"},
		{"claudeArea12"}, {"claudeArea13"}, {"claudeArea14"}, {"claudeArea15"}};
	CachedPixelShaderSetting<float> m_area_count_pixel{"claudeAreaCount"};
	float m_grid_debug;
	float m_water_reflections;
	float m_gi_strength;
	float m_gi_split;
	float m_clay;
	bool m_volumetric_light_enabled;
	CachedPixelShaderSetting<float, 3>
		m_sun_position_pixel{"sunPositionScreen"};
	CachedPixelShaderSetting<float> m_sun_brightness_pixel{"sunBrightness"};
	CachedPixelShaderSetting<float, 3>
		m_moon_position_pixel{"moonPositionScreen"};
	CachedPixelShaderSetting<float> m_moon_brightness_pixel{"moonBrightness"};
	CachedPixelShaderSetting<float, 1, false>
		m_volumetric_light_strength_pixel{"volumetricLightStrength"};

	static constexpr std::array<const char*, 46> SETTING_CALLBACKS = {
		"exposure_compensation",
		"golden_hour_strength",
		"ssao_strength",
		"bump_strength",
		"claude_grid_debug",
		"claude_water_reflections",
		"claude_gi",
		"claude_gi_split",
		"claude_clay",
		"claude_texture",
		"claude_gray",
		"claude_pure",
		"claude_bevel",
		"claude_relief",
		"claude_parallax",
		"claude_jitter",
		"claude_skybounce",
		"claude_sun_angle",
		"claude_night_sky",
		"claude_moon_gain",
		"claude_bounce2",
		"claude_cache_sky",
		"claude_bisect",
		"claude_pyramid",
		"claude_nee_gate",
		"claude_cost",
		"claude_face_direct",
		"claude_tiers",
		"claude_bounce_stride",
		"claude_face_texels",
		"claude_cache_remap",
		"claude_far_hist",
		"claude_light_ladder",
		"claude_lod_dither",
		"claude_far_grain",
		"claude_far_fog",
		"claude_sky_azimuth",
		"claude_subvox",
		"claude_descend",
		"claude_refine",
		"claude_denoise",
		"claude_view",
		"claude_bounces",
		"claude_nee",
		"claude_rng",
	};

	static float readGoldenHourStrength()
	{
		if (!g_settings->exists("golden_hour_strength"))
			return 1.0f;
		return g_settings->getFloat("golden_hour_strength", 0.0f, 2.0f);
	}

	static float readSsaoStrength()
	{
		if (!g_settings->exists("ssao_strength"))
			return 0.7f;
		return g_settings->getFloat("ssao_strength", 0.0f, 2.0f);
	}

	static float readBumpStrength()
	{
		if (!g_settings->exists("bump_strength"))
			return 0.8f;
		return g_settings->getFloat("bump_strength", 0.0f, 2.0f);
	}

	static float readGridDebug()
	{
		if (!g_settings->exists("claude_grid_debug"))
			return 0.0f;
		// 1 = ghost view with shadow rays, 2 = ghost without (A/B),
		// 3 = pure path-traced view (zero ambient, all light via rays),
		// 4 = mode 3 with neutral albedo (lighting-only diagnostic),
		// 5 = cascade-level tint, 6 = sub-voxel light quantization,
		// 7-10 = term-isolation heatmaps (torch/bounce/sun/visibility),
		// 11 = shadow-ray WHY-map (cause-coded colors).
		// The old 4.0 clamp silently rewrote every mode-5/6 request to 4
		// — a whole evening of "nothing changed" (John caught it).
		return g_settings->getFloat("claude_grid_debug", 0.0f, 12.0f);
	}

	static float readWaterReflections()
	{
		if (!g_settings->exists("claude_water_reflections"))
			return 0.0f;
		return g_settings->getFloat("claude_water_reflections", 0.0f, 1.0f);
	}

	static float readGiStrength()
	{
		if (!g_settings->exists("claude_gi"))
			return 0.0f;
		return g_settings->getFloat("claude_gi", 0.0f, 1.0f);
	}

	static float readGiSplit()
	{
		if (!g_settings->exists("claude_gi_split"))
			return 0.0f;
		// 1 = relight only the right half of the screen (A/B seam)
		return g_settings->getFloat("claude_gi_split", 0.0f, 1.0f);
	}

	// clay render: 1 = neutral-gray albedo everywhere in the traced
	// pipeline - pure light transport, no material color (John,
	// 2026-08-13: "clay style, no color"). Grays emissive Le too (the
	// NEE warm tint survives); a diagnostic look, not a ship look.
	static float readGray()
	{
		if (!g_settings->exists("claude_gray"))
			return 0.0f;
		return g_settings->getFloat("claude_gray", 0.0f, 1.0f);
	}

	// claude_pure: photo mode runs ONE lighting system (emissive voxels
	// only, no point-light NEE) — see purePhoto in claude_accum.
	static float readPure()
	{
		if (!g_settings->exists("claude_pure"))
			return 0.0f;
		return g_settings->getFloat("claude_pure", 0.0f, 1.0f);
	}

	static float readTextureAmount()
	{
		if (!g_settings->exists("claude_texture"))
			return 0.0f;
		// 0 = pure clay, 1 = full tile texture (the flatness-guard dial)
		return g_settings->getFloat("claude_texture", 0.0f, 1.0f);
	}

	static float readBevel()
	{
		if (!g_settings->exists("claude_bevel"))
			return 0.35f;
		return g_settings->getFloat("claude_bevel", 0.0f, 1.0f);
	}

	static float readRelief()
	{
		if (!g_settings->exists("claude_relief"))
			return 0.0f;
		return g_settings->getFloat("claude_relief", 0.0f, 1.0f);
	}

	static float readParallax()
	{
		if (!g_settings->exists("claude_parallax"))
			return 0.0f;
		return g_settings->getFloat("claude_parallax", 0.0f, 1.0f);
	}

	static float readJitter()
	{
		if (!g_settings->exists("claude_jitter"))
			return 0.10f;
		return g_settings->getFloat("claude_jitter", 0.0f, 1.0f);
	}

	// Sun/moon angular DIAMETER in degrees. Both are ~0.53 deg in reality,
	// and that half-degree is the whole reason shadow edges soften with
	// distance from the caster. The old hardcoded jitter was ~4 deg — about
	// eight times too wide, which smeared every shadow edge.
	static float readSunAngle()
	{
		if (!g_settings->exists("claude_sun_angle"))
			return 0.53f;
		return g_settings->getFloat("claude_sun_angle", 0.0f, 20.0f);
	}

	static float readNightSky()
	{
		if (!g_settings->exists("claude_night_sky"))
			return 1.0f;
		return g_settings->getFloat("claude_night_sky", 0.0f, 20.0f);
	}

	static float readMoonGain()
	{
		if (!g_settings->exists("claude_moon_gain"))
			return 1.0f;
		return g_settings->getFloat("claude_moon_gain", 0.0f, 20.0f);
	}

	static float readSkyBounce()
	{
		if (!g_settings->exists("claude_skybounce"))
			return 0.6f;
		return g_settings->getFloat("claude_skybounce", 0.0f, 2.0f);
	}

	// Third bounce: the ambient ray's hit fires ONE more hop at full
	// sub-voxel fidelity (bounce 3). 0 = off (default).
	static float readBounce2()
	{
		if (!g_settings->exists("claude_bounce2"))
			return 0.0f;
		return g_settings->getFloat("claude_bounce2", 0.0f, 1.0f);
	}

	// Cache sky seeding: gather rays that escape to sky deposit the sky's
	// radiance into the cell instead of 0, so skylight propagates into
	// caves cell-to-cell. 0 = off (the old surfaces-only cache).
	static float readCacheSky()
	{
		if (!g_settings->exists("claude_cache_sky"))
			return 0.0f;
		return g_settings->getFloat("claude_cache_sky", 0.0f, 2.0f);
	}

	// Carve-pipeline bisection ladder (2026-08-12 debugging instrument):
	// 0 normal; 1 carve geometry only, cube-recipe shading, shadow rays
	// ignore carve; 2 +micro normal; 3 +micro position; 4 +micro term
	// recipe; 5 +shadow-ray micro occlusion (= full carving).
	static float readBisect()
	{
		if (!g_settings->exists("claude_bisect"))
			return 0.0f;
		return g_settings->getFloat("claude_bisect", 0.0f, 5.0f);
	}

	// Occupancy-pyramid leap climb (overnight 2026-08-12): 0 = classic
	// 4-cell brick leap only, 1 = climb mips for 8/16/32-cell leaps.
	static float readPyramid()
	{
		if (!g_settings->exists("claude_pyramid"))
			return 0.0f;
		return g_settings->getFloat("claude_pyramid", 0.0f, 1.0f);
	}

	static float readNeeGate()
	{
		if (!g_settings->exists("claude_nee_gate"))
			return 0.0f;
		return g_settings->getFloat("claude_nee_gate", 0.0f, 0.1f);
	}

	static float readCost()
	{
		if (!g_settings->exists("claude_cost"))
			return 0.0f;
		return g_settings->getFloat("claude_cost", 0.0f, 4.0f);
	}

	static float readFaceDirect()
	{
		if (!g_settings->exists("claude_face_direct"))
			return 0.0f;
		return g_settings->getFloat("claude_face_direct", 0.0f, 1.0f);
	}

	// Default 1: the bounce-vertex budget tiers ARE the ship state; 0
	// restores full-quality indirect-hit shading for A/B.
	static float readTiers()
	{
		if (!g_settings->exists("claude_tiers"))
			return 1.0f;
		return g_settings->getFloat("claude_tiers", 0.0f, 1.0f);
	}

	// 1 = bounce ray every pixel every frame (default); N = Russian-
	// roulette to 1/N of pixels, weighted N, history averages the rest.
	static float readBounceStride()
	{
		if (!g_settings->exists("claude_bounce_stride"))
			return 1.0f;
		return g_settings->getFloat("claude_bounce_stride", 1.0f, 8.0f);
	}

	// 1 = coarse per-face cache only; 2 = near-ring 4x4 sub-face atlas
	static float readFaceTexels()
	{
		if (!g_settings->exists("claude_face_texels"))
			return 1.0f;
		return g_settings->getFloat("claude_face_texels", 1.0f, 2.0f);
	}

	// 1 (default) = shift face caches across a grid rebase; 0 = the
	// old zero-and-rebuild (the pulse), kept for A/B
	static float readCacheRemap()
	{
		if (!g_settings->exists("claude_cache_remap"))
			return 1.0f;
		return g_settings->getFloat("claude_cache_remap", 0.0f, 1.0f);
	}

	// 1 (default) = deep temporal history for the far field in motion
	static float readFarHist()
	{
		if (!g_settings->exists("claude_far_hist"))
			return 1.0f;
		return g_settings->getFloat("claude_far_hist", 0.0f, 1.0f);
	}

	// 1 (default) = far cells read the cached light ladder (ADR-0008)
	static float readLightLadder()
	{
		if (!g_settings->exists("claude_light_ladder"))
			return 1.0f;
		return g_settings->getFloat("claude_light_ladder", 0.0f, 1.0f);
	}

	// 1 (default) = dithered LOD hand-off bands (one transition rule)
	static float readLodDither()
	{
		if (!g_settings->exists("claude_lod_dither"))
			return 1.0f;
		return g_settings->getFloat("claude_lod_dither", 0.0f, 1.0f);
	}

	// 1 (default) = multi-scale grain inside far cell faces
	static float readFarGrain()
	{
		if (!g_settings->exists("claude_far_grain"))
			return 1.0f;
		return g_settings->getFloat("claude_far_grain", 0.0f, 1.0f);
	}

	// 1 (default) = aerial-perspective haze on far terrain
	static float readFarFog()
	{
		if (!g_settings->exists("claude_far_fog"))
			return 1.0f;
		return g_settings->getFloat("claude_far_fog", 0.0f, 1.0f);
	}

	// 1 (default) = dawn/dusk glow confined to the sun's side of the sky
	static float readSkyAz()
	{
		if (!g_settings->exists("claude_sky_azimuth"))
			return 1.0f;
		return g_settings->getFloat("claude_sky_azimuth", 0.0f, 1.0f);
	}

	// 1 (default) = real per-node sub-voxel bits; 0 = trace-time carve
	static float readSubvox()
	{
		if (!g_settings->exists("claude_subvox"))
			return 1.0f;
		return g_settings->getFloat("claude_subvox", 0.0f, 1.0f);
	}

	// 1 (default) = march() descends into class-250 cells; 0 = they are
	// opaque 1 m cubes, the behaviour of every build before 2026-08-16
	static float readDescend()
	{
		if (!g_settings->exists("claude_descend"))
			return 1.0f;
		return g_settings->getFloat("claude_descend", 0.0f, 1.0f);
	}

	// 1 (default) = extra rays per pixel while the camera is at rest
	static float readRefine()
	{
		if (!g_settings->exists("claude_refine"))
			return 1.0f;
		return g_settings->getFloat("claude_refine", 0.0f, 1.0f);
	}

	// 1 (default) = edge-aware spatial denoise; 0 = raw samples
	static float readDenoise()
	{
		if (!g_settings->exists("claude_denoise"))
			return 1.0f;
		return g_settings->getFloat("claude_denoise", 0.0f, 1.0f);
	}

	// claude_trace diagnostic view selector, 0..16. 0 (default) = photo:
	// the truth renderer, untouched by any debug branch. 6 = clay:
	// photo transport with reflectance clamped to CLAY_RHO. 9/10/11 are
	// roadmap 1a's direct-light referee (the two MIS halves and the
	// analytic answer) and 12/13/14/15 its forensics; 7 and 8 are unused and
	// render as photo. The
	// UPPER BOUND IS A CLAMP, not a validator: getFloat silently pins an
	// out-of-range value, so a view added in the shader and not widened
	// here renders as the nearest legal view and nothing says so.
	static float readView()
	{
		if (!g_settings->exists("claude_view"))
			return 0.0f;
		return g_settings->getFloat("claude_view", 0.0f, 16.0f);
	}

	// claude_trace path-depth cap, 0..24. 24 (default) = full transport.
	static float readBounces()
	{
		if (!g_settings->exists("claude_bounces"))
			return 24.0f;
		return g_settings->getFloat("claude_bounces", 0.0f, 24.0f);
	}

	// claude_trace transport mode, 0/1. 1 (default) = next-event
	// estimation + MIS. 0 = the pure photo path of rung 1, unchanged
	// down to the RNG draw order — the truth mode a referee is run
	// against, and the other half of the A/B (§6).
	static float readNee()
	{
		if (!g_settings->exists("claude_nee"))
			return 1.0f;
		return g_settings->getFloat("claude_nee", 0.0f, 1.0f);
	}

	// claude_trace RNG source, 0/1. 1 (default) = the counter-based PCG.
	// 0 = the old, biased hash chain, which is what every Cornell number
	// in measured.md before the "1a" section was taken with. See m_rng.
	static float readRng()
	{
		if (!g_settings->exists("claude_rng"))
			return 1.0f;
		return g_settings->getFloat("claude_rng", 0.0f, 1.0f);
	}


	static float readClay()
	{
		if (!g_settings->exists("claude_clay"))
			return 0.0f;
		// blend surfaces toward flat per-block color (ghost/Teardown look)
		return g_settings->getFloat("claude_clay", 0.0f, 1.0f);
	}

public:
	void onSettingsChange(const std::string &name)
	{
		if (name == "exposure_compensation")
			m_user_exposure_compensation = g_settings->getFloat("exposure_compensation", -1.0f, 1.0f);
		if (name == "golden_hour_strength")
			m_golden_hour_strength = readGoldenHourStrength();
		if (name == "ssao_strength")
			m_ssao_strength = readSsaoStrength();
		if (name == "bump_strength")
			m_bump_strength = readBumpStrength();
		if (name == "claude_grid_debug")
			m_grid_debug = readGridDebug();
		if (name == "claude_water_reflections")
			m_water_reflections = readWaterReflections();
		if (name == "claude_gi")
			m_gi_strength = readGiStrength();
		if (name == "claude_gi_split")
			m_gi_split = readGiSplit();
		if (name == "claude_clay")
			m_clay = readClay();
		if (name == "claude_texture")
			m_texture_amount = readTextureAmount();
		if (name == "claude_gray")
			m_gray = readGray();
		if (name == "claude_pure")
			m_pure = readPure();
		if (name == "claude_bevel")
			m_bevel = readBevel();
		if (name == "claude_relief")
			m_relief = readRelief();
		if (name == "claude_parallax")
			m_parallax = readParallax();
		if (name == "claude_jitter")
			m_jitter = readJitter();
		if (name == "claude_skybounce")
			m_skybounce = readSkyBounce();
		if (name == "claude_sun_angle")
			m_sunangle = readSunAngle();
		if (name == "claude_night_sky")
			m_nightsky = readNightSky();
		if (name == "claude_moon_gain")
			m_moongain = readMoonGain();
		if (name == "claude_bounce2")
			m_bounce2 = readBounce2();
		if (name == "claude_cache_sky")
			m_cache_sky = readCacheSky();
		if (name == "claude_bisect")
			m_bisect = readBisect();
		if (name == "claude_pyramid")
			m_pyramid = readPyramid();
		if (name == "claude_nee_gate")
			m_nee_gate = readNeeGate();
		if (name == "claude_cost")
			m_cost = readCost();
		if (name == "claude_face_direct")
			m_face_direct = readFaceDirect();
		if (name == "claude_tiers")
			m_tiers = readTiers();
		if (name == "claude_bounce_stride")
			m_bounce_stride = readBounceStride();
		if (name == "claude_face_texels")
			m_face_texels = readFaceTexels();
		if (name == "claude_cache_remap")
			m_cache_remap = readCacheRemap();
		if (name == "claude_far_hist")
			m_far_hist = readFarHist();
		if (name == "claude_light_ladder")
			m_light_ladder = readLightLadder();
		if (name == "claude_lod_dither")
			m_lod_dither = readLodDither();
		if (name == "claude_far_grain")
			m_far_grain = readFarGrain();
		if (name == "claude_far_fog")
			m_far_fog = readFarFog();
		if (name == "claude_sky_azimuth")
			m_sky_az = readSkyAz();
		if (name == "claude_subvox")
			m_subvox = readSubvox();
		if (name == "claude_descend")
			m_descend = readDescend();
		if (name == "claude_refine")
			m_refine = readRefine();
		if (name == "claude_denoise")
			m_denoise = readDenoise();
		if (name == "claude_view")
			m_view = readView();
		if (name == "claude_bounces")
			m_bounces = readBounces();
		if (name == "claude_nee")
			m_nee = readNee();
		if (name == "claude_rng")
			m_rng = readRng();
	}

	static void settingsCallback(const std::string &name, void *userdata)
	{
		reinterpret_cast<GameGlobalShaderUniformSetter*>(userdata)->onSettingsChange(name);
	}

	void setSky(Sky *sky) { m_sky = sky; }

	GameGlobalShaderUniformSetter(Sky *sky, Game *game) :
		m_sky(sky),
		m_client(game->getClient())
	{
		for (auto &name : SETTING_CALLBACKS)
			g_settings->registerChangedCallback(name, settingsCallback, this);

		m_user_exposure_compensation = g_settings->getFloat("exposure_compensation", -1.0f, 1.0f);
		m_golden_hour_strength = readGoldenHourStrength();
		m_ssao_strength = readSsaoStrength();
		m_bump_strength = readBumpStrength();
		m_grid_debug = readGridDebug();
		m_water_reflections = readWaterReflections();
		m_gi_strength = readGiStrength();
		m_gi_split = readGiSplit();
		m_clay = readClay();
		m_texture_amount = readTextureAmount();
		m_gray = readGray();
		m_pure = readPure();
		m_bevel = readBevel();
		m_relief = readRelief();
		m_parallax = readParallax();
		m_jitter = readJitter();
		m_skybounce = readSkyBounce();
		m_sunangle = readSunAngle();
		m_nightsky = readNightSky();
		m_moongain = readMoonGain();
		m_bounce2 = readBounce2();
		m_cache_sky = readCacheSky();
		m_bisect = readBisect();
		m_pyramid = readPyramid();
		m_nee_gate = readNeeGate();
		m_cost = readCost();
		m_face_direct = readFaceDirect();
		m_tiers = readTiers();
		m_bounce_stride = readBounceStride();
		m_face_texels = readFaceTexels();
		m_cache_remap = readCacheRemap();
		m_far_hist = readFarHist();
		m_light_ladder = readLightLadder();
		m_lod_dither = readLodDither();
		m_far_grain = readFarGrain();
		m_far_fog = readFarFog();
		m_sky_az = readSkyAz();
		m_subvox = readSubvox();
		m_descend = readDescend();
		m_refine = readRefine();
		m_denoise = readDenoise();
		m_view = readView();
		m_bounces = readBounces();
		m_nee = readNee();
		m_rng = readRng();
		m_bloom_enabled = g_settings->getBool("enable_bloom");
		m_volumetric_light_enabled = g_settings->getBool("enable_volumetric_lighting") && m_bloom_enabled;
		m_crack_animation_length_i = game->crack_animation_length;
	}

	~GameGlobalShaderUniformSetter()
	{
		g_settings->deregisterAllChangedCallbacks(this);
	}

	void onSetUniforms(video::IMaterialRendererServices *services) override
	{
		u32 daynight_ratio = (float)m_client->getEnv().getDayNightRatio();
		video::SColorf sunlight;
		get_sunlight_color(&sunlight, daynight_ratio);
		m_day_light.set(sunlight, services);

		u32 animation_timer = m_client->getEnv().getFrameTime() % 1000000;
		float animation_timer_f = (float)animation_timer / 100000.f;
		m_animation_timer_vertex.set(&animation_timer_f, services);
		m_animation_timer_pixel.set(&animation_timer_f, services);

		float animation_timer_delta_f = (float)m_client->getEnv().getFrameTimeDelta() / 100000.f;
		m_animation_timer_delta_vertex.set(&animation_timer_delta_f, services);
		m_animation_timer_delta_pixel.set(&animation_timer_delta_f, services);

		if (m_client->getMinimap()) {
			v3f minimap_yaw = m_client->getMinimap()->getYawVec();
			m_minimap_yaw.set(minimap_yaw, services);
		}

		v3f offset = intToFloat(m_client->getCamera()->getOffset(), BS);
		m_camera_offset_pixel.set(offset, services);
		m_camera_offset_vertex.set(offset, services);

		v3f camera_position = m_client->getCamera()->getPosition();
		m_camera_position_pixel.set(camera_position, services);
		m_camera_position_pixel.set(camera_position, services);

		m_texel_size0_vertex.set(m_texel_size0, services);
		m_texel_size0_pixel.set(m_texel_size0, services);

		{
			float tmp = m_crack_animation_length_i;
			m_crack_animation_length.set(&tmp, services);
			tmp = m_crack_level_i;
			m_crack_level.set(&tmp, services);
			tmp = m_crack_texture_scale_i;
			m_crack_texture_scale.set(&tmp, services);
		}

		const auto &lighting = m_client->getEnv().getLocalPlayer()->getLighting();

		const AutoExposure &exposure_params = lighting.exposure;
		std::array<float, 7> exposure_buffer = {
			std::pow(2.0f, exposure_params.luminance_min),
			std::pow(2.0f, exposure_params.luminance_max),
			exposure_params.exposure_correction,
			exposure_params.speed_dark_bright,
			exposure_params.speed_bright_dark,
			exposure_params.center_weight_power,
			powf(2.f, m_user_exposure_compensation)
		};
		m_exposure_params_pixel.set(exposure_buffer.data(), services);

		if (m_bloom_enabled) {
			float intensity = std::max(lighting.bloom_intensity, 0.0f);
			m_bloom_intensity_pixel.set(&intensity, services);
			float strength_factor = std::max(lighting.bloom_strength_factor, 0.0f);
			m_bloom_strength_pixel.set(&strength_factor, services);
			float radius = std::max(lighting.bloom_radius, 0.0f);
			m_bloom_radius_pixel.set(&radius, services);
		}

		float saturation = lighting.saturation;
		m_saturation_pixel.set(&saturation, services);

		float dnr = daynight_ratio / 1000.0f;
		m_day_night_ratio_pixel.set(&dnr, services);
		m_golden_hour_pixel.set(&m_golden_hour_strength, services);
		m_ssao_strength_pixel.set(&m_ssao_strength, services);
		m_bump_strength_pixel.set(&m_bump_strength, services);

		// claude_grid ghost view: hand the shader a ray-generation basis
		// in grid-local node units. Camera position is absolute world
		// coords / BS minus the grid origin — computed CPU-side so the
		// trace never involves camera-offset space. Right/up are pre-scaled
		// by tan(fov/2) so the shader builds rays with two multiply-adds.
		{
			float dbg = g_claude_grid.valid ? m_grid_debug : 0.0f;
			m_grid_debug_pixel.set(&dbg, services);
			float refl = g_claude_grid.valid ? m_water_reflections : 0.0f;
			m_water_refl_pixel.set(&refl, services);
			float gi = g_claude_grid.valid ? m_gi_strength : 0.0f;
			m_gi_strength_pixel.set(&gi, services);
			m_gi_split_pixel.set(&m_gi_split, services);
			float clay = g_claude_grid.valid ? m_clay : 0.0f;
			m_clay_pixel.set(&clay, services);
			m_accum_alpha_pixel.set(&g_claude_grid.accum_alpha, services);
			m_prev_pos_pixel.set(g_claude_grid.shader_prev_pos, services);
			m_prev_fwd_pixel.set(g_claude_grid.shader_prev_fwd, services);
			m_prev_rightu_pixel.set(g_claude_grid.shader_prev_rightu, services);
			m_prev_upu_pixel.set(g_claude_grid.shader_prev_upu, services);
			float ptan[2] = {g_claude_grid.shader_prev_tanx,
					g_claude_grid.shader_prev_tany};
			m_prev_tan_pixel.set(ptan, services);
			for (int e = 0; e < 8; e++)
				m_emitter_pixel[e].set(g_claude_grid.emitters[e], services);
			float ecount = (float)g_claude_grid.emitter_runtime;
			m_emitter_count_pixel.set(&ecount, services);
			m_held_emitter_pixel.set(g_claude_grid.held_emitter, services);
			// AREA emitters (claude_trace NEE). Sent whether or not the
			// grid is valid: an invalid grid leaves area_count at 0,
			// and a count of 0 is exactly "no light sampling", which the
			// estimator handles by handing every BSDF-found emitter the
			// full balance weight — i.e. it degrades to the photo path
			// rather than to a wrong image.
			for (int e = 0; e < ClaudeTraceGrid::AREA_CAP; e++)
				m_area_pixel[e].set(g_claude_grid.area[e], services);
			float acount = g_claude_grid.valid
					? (float)g_claude_grid.area_count : 0.0f;
			m_area_count_pixel.set(&acount, services);
			float b2 = g_claude_grid.valid ? m_bounce2 : 0.0f;
			m_bounce2_pixel.set(&b2, services);
			m_cache_sky_pixel.set(&m_cache_sky, services);
			m_bisect_pixel.set(&m_bisect, services);
			m_pyramid_pixel.set(&m_pyramid, services);
			m_nee_gate_pixel.set(&m_nee_gate, services);
			m_cost_pixel.set(&m_cost, services);
			m_face_direct_pixel.set(&m_face_direct, services);
			m_tiers_pixel.set(&m_tiers, services);
			m_bounce_stride_pixel.set(&m_bounce_stride, services);
			m_face_texels_pixel.set(&m_face_texels, services);
			m_near_origin_pixel.set(g_claude_grid.near_origin, services);
			m_near_prev_pixel.set(g_claude_grid.near_prev, services);
			m_origin_delta_pixel.set(g_claude_grid.origin_delta, services);
			m_cache_remap_pixel.set(&m_cache_remap, services);
			m_far_hist_pixel.set(&m_far_hist, services);
			m_light_ladder_pixel.set(&m_light_ladder, services);
			m_lod_dither_pixel.set(&m_lod_dither, services);
			m_far_grain_pixel.set(&m_far_grain, services);
			m_far_fog_pixel.set(&m_far_fog, services);
			m_sky_az_pixel.set(&m_sky_az, services);
			// REBIND EVERY FRAME, UNCONDITIONALLY. These 3D textures are
			// bound with raw GL outside Irrlicht's material system, and
			// they were only bound inside claudeTraceGridSnapshot() — which
			// runs every ~2 s, not per frame. Units 4-8 fall inside
			// Irrlicht's managed range (MATERIAL_MAX_TEXTURES), and the
			// GL3 cache handler resets those units between snapshots, so
			// on a core profile every sampler read empty and every ray
			// escaped to sky. The legacy 2.1 driver evidently left them
			// alone, which is why this only ever showed on 4.1.
			//
			// Unconditional because a sampler3D uniform left at its
			// default 0 aliases unit 0 with sampler2D texture0 — one unit,
			// two sampler types — which makes the PROGRAM invalid on core
			// and every draw it makes undefined. So the sampler uniforms
			// must be delivered whenever the program runs, not only once
			// a traced consumer is switched on. (Before the first grid
			// snapshot the textures are 0 and the binds no-op; the
			// uniforms still point the samplers at distinct units, which
			// is what validity requires.)
			//
			// Also: restore the active-texture unit the DRIVER'S CACHE
			// believes is current, not a hard-coded GL.TEXTURE0 — a raw
			// restore to 0 desyncs COpenGLCoreCacheHandler's ActiveTexture
			// mirror, after which cached setActiveTexture(X) calls are
			// skipped as "already X" while GL really sits at 0, and
			// subsequent material binds land on the wrong unit. That kind
			// of frame-order-dependent corruption is exactly the
			// works-once-never-again symptom.
			{
				GLint prev_active = GL.TEXTURE0;
				GL.GetIntegerv(GL.ACTIVE_TEXTURE, &prev_active);
				if (g_claude_grid.tex) {
					GL.ActiveTexture(GL.TEXTURE10);
					GL.BindTexture(GL.TEXTURE_3D, g_claude_grid.tex);
				}
				if (g_claude_grid.coarse_tex) {
					GL.ActiveTexture(GL.TEXTURE11);
					GL.BindTexture(GL.TEXTURE_3D, g_claude_grid.coarse_tex);
				}
				if (g_claude_grid.material_tex) {
					GL.ActiveTexture(GL.TEXTURE12);
					GL.BindTexture(GL.TEXTURE_3D, g_claude_grid.material_tex);
				}
				if (g_claude_grid.atlas_tex) {
					GL.ActiveTexture(GL.TEXTURE13);
					GL.BindTexture(GL.TEXTURE_2D, g_claude_grid.atlas_tex);
				}
				if (g_claude_grid.matparams_tex) {
					GL.ActiveTexture(GL.TEXTURE15);
					GL.BindTexture(GL.TEXTURE_2D, g_claude_grid.matparams_tex);
				}
				if (g_claude_grid.subvox_tex) {
					GL.ActiveTexture(GL.TEXTURE7);
					GL.BindTexture(GL.TEXTURE_3D, g_claude_grid.subvox_tex);
				}
				if (g_claude_grid.model_ids_tex) {
					GL.ActiveTexture(GL.TEXTURE0 + 16);
					GL.BindTexture(GL.TEXTURE_3D,
							g_claude_grid.model_ids_tex);
					GL.ActiveTexture(GL.TEXTURE0 + 17);
					GL.BindTexture(GL.TEXTURE_3D,
							g_claude_grid.model_atlas_tex);
					GL.ActiveTexture(GL.TEXTURE0 + 18);
					GL.BindTexture(GL.TEXTURE_2D,
							g_claude_grid.model_pal_tex);
				}
				if (g_claude_grid.cascades_tex) {
					GL.ActiveTexture(GL.TEXTURE8);
					GL.BindTexture(GL.TEXTURE_3D, g_claude_grid.cascades_tex);
					GL.ActiveTexture(GL.TEXTURE9);
					GL.BindTexture(GL.TEXTURE_3D,
							g_claude_grid.cascades_coarse_tex);
				}
				GL.ActiveTexture(prev_active);

				SamplerLayer_t layer = 10;
				m_grid_sampler_pixel.set(&layer, services);
				SamplerLayer_t clayer = 11;
				m_coarse_sampler_pixel.set(&clayer, services);
				SamplerLayer_t mlayer = 12;
				m_materials_sampler_pixel.set(&mlayer, services);
				SamplerLayer_t alayer = 13;
				m_atlas_sampler_pixel.set(&alayer, services);
				SamplerLayer_t player = 15;
				m_matparams_sampler_pixel.set(&player, services);
				SamplerLayer_t svlayer = 7;
				m_subvox_sampler_pixel.set(&svlayer, services);
				SamplerLayer_t midl = 16, matl = 17, mpal = 18;
				m_modelids_sampler_pixel.set(&midl, services);
				m_modelatlas_sampler_pixel.set(&matl, services);
				m_modelpal_sampler_pixel.set(&mpal, services);
				m_subvox_pixel.set(&m_subvox, services);
				m_descend_pixel.set(&m_descend, services);
				m_refine_pixel.set(&m_refine, services);
				m_denoise_pixel.set(&m_denoise, services);
				// claude_trace's three dials. Delivered here, next to the
				// samplers, because claude_present consumes claudeView
				// too and both programs run every frame regardless of
				// mode — a value only set when a consumer is on would
				// leave one of them reading stale state after a toggle.
				m_view_pixel.set(&m_view, services);
				m_bounces_pixel.set(&m_bounces, services);
				m_nee_pixel.set(&m_nee, services);
				m_rng_pixel.set(&m_rng, services);
				SamplerLayer_t cascl = 8;
				m_cascades_sampler_pixel.set(&cascl, services);
				SamplerLayer_t casccl = 9;
				m_cascades_coarse_sampler_pixel.set(&casccl, services);
				// cascade origins handed to the shader VOLUME-LOCAL
				// (cascade world origin minus grid world origin), so the
				// shader converts a grid-local point to cascade cells
				// with one subtract and a divide
				CachedPixelShaderSetting<float, 3, false> *corg_pixels[5] = {
					&m_cascade0_origin_pixel, &m_cascade1_origin_pixel,
					&m_cascade2_origin_pixel, &m_cascade3_origin_pixel,
					&m_cascade4_origin_pixel };
				float cvalid[5];
				for (int lv = 0; lv < 5; lv++) {
					v3f corg = v3f((float)(g_claude_grid.casc[lv].origin.X
								- g_claude_grid.origin.X),
							(float)(g_claude_grid.casc[lv].origin.Y
								- g_claude_grid.origin.Y),
							(float)(g_claude_grid.casc[lv].origin.Z
								- g_claude_grid.origin.Z));
					corg_pixels[lv]->set(corg, services);
					cvalid[lv] = g_claude_grid.casc[lv].valid ? 1.0f : 0.0f;
				}
				m_cascade_valid_pixel.set(cvalid, services);
				float cvalid2[3] = {cvalid[3], cvalid[4], 0.0f};
				m_cascade_valid2_pixel.set(cvalid2, services);
			}
			if (dbg > 0.0f || refl > 0.0f || gi > 0.0f || clay > 0.0f) {
				v3f vorg((float)g_claude_grid.origin.X,
						(float)g_claude_grid.origin.Y,
						(float)g_claude_grid.origin.Z);
				m_grid_origin_pixel.set(vorg, services);
				m_texture_amount_pixel.set(&m_texture_amount, services);
				m_gray_pixel.set(&m_gray, services);
				m_pure_pixel.set(&m_pure, services);
				m_bevel_pixel.set(&m_bevel, services);
				m_relief_pixel.set(&m_relief, services);
				m_parallax_pixel.set(&m_parallax, services);
				m_jitter_pixel.set(&m_jitter, services);
				m_skybounce_pixel.set(&m_skybounce, services);
				float sun_rad = m_sunangle * 0.0174532925f;
				m_sunangle_pixel.set(&sun_rad, services);
				m_nightsky_pixel.set(&m_nightsky, services);
				Camera *camera = m_client->getCamera();
				v3f local = camera->getPosition() / BS
						- v3f(g_claude_grid.origin.X,
							g_claude_grid.origin.Y,
							g_claude_grid.origin.Z);
				m_grid_cam_pos_pixel.set(local, services);
				v3f fwd = camera->getDirection();
				fwd.normalize();
				v3f right = v3f(0.f, 1.f, 0.f).crossProduct(fwd);
				right.normalize();
				v3f up = fwd.crossProduct(right);
				right *= std::tan(camera->getFovX() * 0.5f);
				up *= std::tan(camera->getFovY() * 0.5f);
				m_grid_cam_fwd_pixel.set(fwd, services);
				m_grid_cam_right_pixel.set(right, services);
				m_grid_cam_up_pixel.set(up, services);
				// Traced light source: the sun when it's up (warm, ramped
				// by day-night ratio), else the moon (cool, dim, a real
				// light source so traced nights aren't pitch black), else
				// no light at all. Directions are offset-independent.
				v3f sun(0.55f, 0.40f, 0.35f);
				v3f lcol(0.0f, 0.0f, 0.0f);
				// Choose by which body is actually ABOVE THE HORIZON, never by
				// the visible flags: Mineclonia leaves getSunVisible() true all
				// night, so the moon branch never ran once. Nights were lit by
				// a dim warm SUN pointing ~60 deg underground — every upward
				// face had dot(n,l) <= 0 and got no direct light at all, and
				// the moon-scattered sky term keyed off a negative Y so it was
				// identically zero. The moon is exactly antipodal to the sun
				// (sky.cpp differs only 90 vs 270), so whenever the sun is
				// down, the moon is up.
				if (m_sky) {
					v3f sdir = m_sky->getSunDirection();
					v3f mdir = m_sky->getMoonDirection();
					if (m_sky->getSunVisible() && sdir.Y > 0.0f) {
						sun = sdir;
						float ramp = std::min(dnr, 1.0f);
						lcol = v3f(1.0f * ramp, 0.95f * ramp, 0.82f * ramp);
						g_claude_grid.light_body = 1;
					} else if (m_sky->getMoonVisible() && mdir.Y > 0.0f) {
						sun = mdir;
						lcol = v3f(0.10f, 0.13f, 0.22f) * m_moongain;
						g_claude_grid.light_body = 2;
					} else {
						g_claude_grid.light_body = 0;
					}
				} else {
					lcol = v3f(1.0f, 0.95f, 0.82f);
					g_claude_grid.light_body = 1;
				}
				sun.normalize();
				// Temporal accumulation invalidated on CAMERA motion only, so
				// standing still averaged dozens of frames while the SUN swept
				// across the sky — it smeared into a streak. The sun disc is
				// 0.53 deg wide and moves ~6 deg/s at time_speed 1440, so it
				// smears eleven disc-widths a second; at the default speed it
				// reads as haze rather than a streak, which is why it went
				// unnoticed. Treat a moved sky as a moved camera.
				if ((sun - g_claude_grid.prev_light_dir).getLength() > 1e-4f
						|| (lcol - g_claude_grid.prev_light_col).getLength()
								> 1e-4f) {
					g_claude_grid.still_frames = 0.0f;
					g_claude_grid.accum_resets++;
					g_claude_grid.accum_alpha =
							std::max(g_claude_grid.accum_alpha, 0.5f);
				}
				g_claude_grid.prev_light_dir = sun;
				g_claude_grid.prev_light_col = lcol;
				m_volume_sun_dir_pixel.set(sun, services);
				m_volume_light_col_pixel.set(lcol, services);
				// near/far for reconstructing eye depth from the depth
				// buffer (world BS units; shader divides by BS for nodes)
				auto cn = camera->getCameraNode();
				float range[2] = {cn->getNearValue(), cn->getFarValue()};
				m_volume_depth_range_pixel.set(range, services);
			}
		}

		if (m_volumetric_light_enabled) {
			// Map directional light to screen space
			auto camera_node = m_client->getCamera()->getCameraNode();
			core::matrix4 transform = camera_node->getProjectionMatrix();
			transform *= camera_node->getViewMatrix();

			if (m_sky->getSunVisible()) {
				v3f sun_position = camera_node->getAbsolutePosition() +
						10000.f * m_sky->getSunDirection();
				transform.transformVect(sun_position);
				sun_position.normalize();

				m_sun_position_pixel.set(sun_position, services);

				float sun_brightness = core::clamp(107.143f * m_sky->getSunDirection().Y, 0.f, 1.f);
				m_sun_brightness_pixel.set(&sun_brightness, services);
			} else {
				m_sun_position_pixel.set(v3f(0.f, 0.f, -1.f), services);

				float sun_brightness = 0.f;
				m_sun_brightness_pixel.set(&sun_brightness, services);
			}

			if (m_sky->getMoonVisible()) {
				v3f moon_position = camera_node->getAbsolutePosition() +
						10000.f * m_sky->getMoonDirection();
				transform.transformVect(moon_position);
				moon_position.normalize();

				m_moon_position_pixel.set(moon_position, services);

				float moon_brightness = core::clamp(107.143f * m_sky->getMoonDirection().Y, 0.f, 1.f);
				m_moon_brightness_pixel.set(&moon_brightness, services);
			} else {
				m_moon_position_pixel.set(v3f(0.f, 0.f, -1.f), services);

				float moon_brightness = 0.f;
				m_moon_brightness_pixel.set(&moon_brightness, services);
			}

			float volumetric_light_strength = lighting.volumetric_light_strength;
			m_volumetric_light_strength_pixel.set(&volumetric_light_strength, services);
		}

		// STEP 0 OF THE DESCEND HANDOFF, and it protects every step
		// after it: ask GL which of the uniforms we just set the linked
		// program actually declares, and say the dead ones out loud
		// once. See shader.h. Free after it has fired.
		claudeUniformCensusReport();
	}

	void onSetMaterial(const video::SMaterial &material) override
	{
		// This is set only for node materials which have a crack, see mapblock_mesh.cpp.
		auto pair = MapBlockMesh::unpackCrackMaterialParam(material.MaterialTypeParam);
		m_crack_level_i = pair.first;
		m_crack_texture_scale_i = pair.second;

		video::ITexture *texture = material.getTexture(0);
		if (texture) {
			core::dimension2du size = texture->getSize();
			m_texel_size0 = v2f(1.f / size.Width, 1.f / size.Height);
		} else {
			m_texel_size0 = v2f();
		}
	}
};


class GameGlobalShaderUniformSetterFactory : public IShaderUniformSetterFactory
{
	Sky *m_sky = nullptr;
	Game *m_game;
	std::vector<GameGlobalShaderUniformSetter*> created_nosky;
public:
	GameGlobalShaderUniformSetterFactory(Game *game) :
		m_game(game)
	{}

	void setSky(Sky *sky)
	{
		m_sky = sky;
		for (GameGlobalShaderUniformSetter *ggscs : created_nosky) {
			ggscs->setSky(m_sky);
		}
		created_nosky.clear();
	}

	virtual IShaderUniformSetter* create(const std::string &name)
	{
		if (str_starts_with(name, "shadow/"))
			return nullptr;
		auto *scs = new GameGlobalShaderUniformSetter(m_sky, m_game);
		if (!m_sky)
			created_nosky.push_back(scs);
		return scs;
	}
};

class NodeShaderConstantSetter : public IShaderConstantSetter
{
public:
	NodeShaderConstantSetter() = default;
	~NodeShaderConstantSetter() = default;

	void onGenerate(const std::string &name, ShaderConstants &constants) override
	{
		if (constants.find("MATERIAL_TYPE") == constants.end())
			return; // not a node shader
		[[maybe_unused]] const auto material_type =
			static_cast<MaterialType>(std::get<int>(constants["MATERIAL_TYPE"]));

#define PROVIDE(constant) constants[ #constant ] = (int)constant

		PROVIDE(TILE_MATERIAL_BASIC);
		PROVIDE(TILE_MATERIAL_ALPHA);
		PROVIDE(TILE_MATERIAL_LIQUID_TRANSPARENT);
		PROVIDE(TILE_MATERIAL_LIQUID_OPAQUE);
		PROVIDE(TILE_MATERIAL_WAVING_LEAVES);
		PROVIDE(TILE_MATERIAL_WAVING_PLANTS);
		PROVIDE(TILE_MATERIAL_OPAQUE);
		PROVIDE(TILE_MATERIAL_WAVING_LIQUID_BASIC);
		PROVIDE(TILE_MATERIAL_WAVING_LIQUID_TRANSPARENT);
		PROVIDE(TILE_MATERIAL_WAVING_LIQUID_OPAQUE);
		PROVIDE(TILE_MATERIAL_PLAIN);
		PROVIDE(TILE_MATERIAL_PLAIN_ALPHA);

#undef PROVIDE

		bool enable_waving_water = g_settings->getBool("enable_waving_water");
		constants["ENABLE_WAVING_WATER"] = enable_waving_water ? 1 : 0;
		if (enable_waving_water) {
			constants["WATER_WAVE_HEIGHT"] = g_settings->getFloat("water_wave_height");
			constants["WATER_WAVE_LENGTH"] = g_settings->getFloat("water_wave_length");
			constants["WATER_WAVE_SPEED"] = g_settings->getFloat("water_wave_speed");
		}
		switch (material_type) {
			case TILE_MATERIAL_WAVING_LIQUID_TRANSPARENT:
			case TILE_MATERIAL_WAVING_LIQUID_OPAQUE:
			case TILE_MATERIAL_WAVING_LIQUID_BASIC:
				constants["MATERIAL_WAVING_LIQUID"] = 1;
				break;
			default:
				constants["MATERIAL_WAVING_LIQUID"] = 0;
				break;
		}
		switch (material_type) {
			case TILE_MATERIAL_WAVING_LIQUID_TRANSPARENT:
			case TILE_MATERIAL_WAVING_LIQUID_OPAQUE:
			case TILE_MATERIAL_WAVING_LIQUID_BASIC:
			case TILE_MATERIAL_LIQUID_TRANSPARENT:
				constants["MATERIAL_WATER_REFLECTIONS"] = 1;
				break;
			default:
				constants["MATERIAL_WATER_REFLECTIONS"] = 0;
				break;
		}

		constants["ENABLE_WAVING_LEAVES"] = g_settings->getBool("enable_waving_leaves") ? 1 : 0;
		constants["ENABLE_WAVING_PLANTS"] = g_settings->getBool("enable_waving_plants") ? 1 : 0;
	}
};

/****************************************************************************
 ****************************************************************************/

Game::Game() :
	m_chat_log_buf(g_logger),
	m_game_ui(new GameUI())
{
	clearTextureNameCache();

	const char *settings[] = {
		"chat_log_level", "doubletap_jump", "toggle_sneak_key", "toggle_aux1_key",
		"enable_joysticks", "enable_fog", "mouse_sensitivity", "joystick_frustum_sensitivity",
		"repeat_place_time", "repeat_dig_time", "noclip", "free_move", "fog_start",
		"cinematic", "cinematic_camera_smoothing", "camera_smoothing", "invert_mouse",
		"enable_hotbar_mouse_wheel", "invert_hotbar_mouse_wheel", "pause_on_lost_focus",
		"keyboard_camera_speed",
	};
	for (auto s : settings)
		g_settings->registerChangedCallback(s, &settingChangedCallback, this);

	readSettings();

	// The conf on disk is the other way a dead claude_volume_* name gets
	// in, and it is the one nobody re-reads. Say so once, at startup.
	claudeWarnRenamedSettings(g_settings, "minetest.conf");
}


Game::~Game()
{
	delete client;
	soundmaker.reset();
	sound_manager.reset();

	delete server;

	delete hud;
	delete camera;
	delete quicktune;
	delete eventmgr;
	delete texture_src;
	delete shader_src;
	delete nodedef_manager;
	delete itemdef_manager;
	delete draw_control;

	clearTextureNameCache();

	g_settings->deregisterAllChangedCallbacks(this);

	if (m_rendering_engine)
		m_rendering_engine->finalize();
}

bool Game::startup(volatile std::sig_atomic_t *kill,
		InputHandler *input,
		RenderingEngine *rendering_engine,
		const GameStartData &start_data,
		std::string &error_message,
		bool *reconnect,
		ChatBackend *chat_backend)
{

	// "cache"
	m_rendering_engine        = rendering_engine;
	device                    = m_rendering_engine->get_raw_device();
	this->kill                = kill;
	this->error_message       = &error_message;
	reconnect_requested       = reconnect;
	this->input               = input;
	this->chat_backend        = chat_backend;
	simple_singleplayer_mode  = start_data.isSinglePlayer();

	input->reloadKeybindings();

	driver = device->getVideoDriver();
	smgr = m_rendering_engine->get_scene_manager();

	driver->setTextureCreationFlag(video::ETCF_CREATE_MIP_MAPS, g_settings->getBool("mip_map"));

	// Reinit runData
	runData = GameRunData();
	runData.time_from_last_punch = 10.0;

	m_game_ui->initFlags();
	if (g_settings->getBool("show_debug")) {
		m_flags.debug_state = 1;
		m_game_ui->m_flags.show_minimal_debug = true;
	}

	m_first_loop_after_window_activation = true;

	g_client_translations->clear();

	if (!init(start_data.world_spec.path, start_data.address,
			start_data.socket_port, start_data.game_spec))
		return false;

	if (!createClient(start_data))
		return false;

	m_rendering_engine->initialize(client, hud);

	m_game_formspec.init(client, m_rendering_engine, input);

	return true;
}


// claude_grid: LEGACY SETTING NAMES ARE LOUD, NEVER SILENT.
//
// On 2026-08-16 the tracer's 128^3 map mirror was renamed "volume" ->
// "grid" (spec/environment-laws.md, "The volume -> grid rename"). Three
// SETTINGS moved with it, and renaming a setting is the dangerous half of
// a rename: an old name in a conf, a patch file or a dial file is not an
// error, it is simply a key nothing reads. It configures nothing and says
// nothing.
//
// That exact failure class already cost this project a session, before the
// rename even existed: a leftover claude_settings_patch.conf pushed
// `claude_volume_follow = 0` onto a seat whose minetest.conf said 1, the
// client ran with follow OFF, and grepping the conf -- the instrument
// everyone reached for -- could not see it (util/claude_ci.py PATCH_HEADER;
// environment-laws "a runtime /set shadows the conf").
//
// So every old name found anywhere gets one loud line naming its
// replacement, on actionstream (the stream CI captures into client.log) and
// on warningstream. Once per name per process: the patch file is re-read at
// ~1 Hz and a per-poll warning would bury the log it is trying to be seen
// in. Covered by src/unittest/test_claude_grid_settings.cpp.
static const struct {
	const char *old_name;
	const char *new_name;
} g_claude_renamed_settings[] = {
	{"claude_volume_follow",   "claude_grid_follow"},
	{"claude_volume_debug",    "claude_grid_debug"},
	{"claude_volume_snapshot", "claude_grid_snapshot"},
};

int claudeWarnRenamedSettings(const Settings *src, const char *source)
{
	static std::set<std::string> warned;
	int found = 0;
	for (const auto &r : g_claude_renamed_settings) {
		if (!src->existsLocal(r.old_name))
			continue;
		found++;
		if (!warned.insert(r.old_name).second)
			continue; // already said once; still counted for the caller
		actionstream << "[claude_grid] IGNORED SETTING '" << r.old_name
				<< "' (" << (source ? source : "?") << "): renamed to '"
				<< r.new_name << "' on 2026-08-16 — the old name does "
				<< "NOTHING. See spec/environment-laws.md." << std::endl;
		warningstream << "[claude_grid] IGNORED SETTING '" << r.old_name
				<< "' -> use '" << r.new_name << "'" << std::endl;
	}
	return found;
}


// claude_settings_patch: poll <path_user>/claude_settings_patch.conf (~1 Hz)
// and apply its key = value lines through g_settings — the same route the
// in-game settings GUI uses, so live-appliable settings (view range, shadows,
// bloom, undersampling, ...) take effect without a client restart. External
// tooling overwrites the file; identical content is not re-applied.
// Blit one node type's top tile (16px, scaled) into the material atlas.
static void claudeAtlasAdd(Client *client, u8 mid, const ContentFeatures &f,
		video::SColor fallback)
{
	if (g_claude_grid.atlas.empty())
		g_claude_grid.atlas.assign(256 * 256 * 4, 0);
	u32 *dst = (u32 *)g_claude_grid.atlas.data();
	int ax = (mid % 16) * 16, ay = (mid / 16) * 16;
	bool ok = false;
	const std::string &tname = f.tiledef[0].name;
	// Ore materials (Mineclonia "stone_with_<kind>" / deepslate variants):
	// the little bits should POKE OUT of the stone and glint. Luminance
	// height gets this exactly backwards for coal — dark specks read as
	// holes — so ores switch to a color-distance rule below, and their
	// per-material spec/gloss row drives the shader's glint (masked to the
	// proud texels, so the stone base stays matte).
	bool is_ore = false;
	float ore_spec = 0.0f, ore_gloss = 0.0f;
	{
		size_t w = f.name.find("_with_");
		if (w != std::string::npos) {
			is_ore = true;
			std::string kind = f.name.substr(w + 6);
			if (kind == "coal") { ore_spec = 0.75f; ore_gloss = 0.65f; }
			else if (kind == "iron") { ore_spec = 0.60f; ore_gloss = 0.45f; }
			else if (kind == "copper") { ore_spec = 0.80f; ore_gloss = 0.55f; }
			else if (kind == "gold") { ore_spec = 0.95f; ore_gloss = 0.70f; }
			else if (kind == "diamond") { ore_spec = 1.00f; ore_gloss = 0.85f; }
			else if (kind == "emerald") { ore_spec = 0.90f; ore_gloss = 0.75f; }
			else if (kind == "redstone") { ore_spec = 0.30f; ore_gloss = 0.30f; }
			else if (kind == "lapis") { ore_spec = 0.35f; ore_gloss = 0.40f; }
			else { ore_spec = 0.50f; ore_gloss = 0.50f; }
		}
	}
	if (g_claude_grid.matparams.empty())
		g_claude_grid.matparams.assign(256 * 4, 0);
	{
		u8 *row = &g_claude_grid.matparams[(size_t)mid * 4];
		row[0] = (u8)(ore_spec * 255.0f + 0.5f);
		row[1] = (u8)(ore_gloss * 255.0f + 0.5f);
		row[2] = is_ore ? 255 : 0;
		row[3] = 255;
	}
	if (!tname.empty()) {
		video::IImage *img = client->tsrc()->claudeGetImage(tname);
		if (img) {
			u32 buf[16 * 16];
			img->copyToScaling(buf, 16, 16, video::ECF_A8R8G8B8);
			// Store DETAIL, not color: each texel divided by the tile's
			// own average, encoded /2 in 0..255. The per-cell color (which
			// already carries the biome palette tint) supplies the hue, so
			// grayscale-plus-palette tiles (Mineclonia grass, leaves) work,
			// and texture can add variation without shifting a face's
			// average brightness — it cannot flatten the lighting.
			double sum[3] = {0, 0, 0};
			for (int k = 0; k < 256; k++) {
				sum[0] += (buf[k] >> 16) & 0xFF;
				sum[1] += (buf[k] >> 8) & 0xFF;
				sum[2] += buf[k] & 0xFF;
			}
			double avg[3] = {std::max(sum[0] / 256.0, 1.0),
					std::max(sum[1] / 256.0, 1.0),
					std::max(sum[2] / 256.0, 1.0)};
			// Height variance decides whether this material has real
			// relief. Snow/smooth tiles vary only by dithering; treating
			// that grain as geometry produced speckle and dark blotches.
			double lmean = 0.0, lvar = 0.0;
			double lum[256];
			for (int k = 0; k < 256; k++) {
				lum[k] = (((buf[k] >> 16) & 0xFF) * 0.30
						+ ((buf[k] >> 8) & 0xFF) * 0.59
						+ (buf[k] & 0xFF) * 0.11);
				lmean += lum[k];
			}
			lmean /= 256.0;
			for (int k = 0; k < 256; k++)
				lvar += (lum[k] - lmean) * (lum[k] - lmean);
			bool carved = std::sqrt(lvar / 256.0) > 14.0;
			// HEIGHT field in alpha, normalised: the tile's BRIGHTEST texels
			// sit flush with the cell boundary and only darker ones recede.
			// Un-normalised, a mid-tone tile carved every face inward and the
			// block read as a small stone floating inside an invisible shell.
			float hgt[256];
			if (carved) {
				double hmin = 255.0, hmax = 0.0;
				for (int k = 0; k < 256; k++) {
					hmin = std::min(hmin, lum[k]);
					hmax = std::max(hmax, lum[k]);
				}
				double span = std::max(hmax - hmin, 12.0);
				for (int k = 0; k < 256; k++)
					hgt[k] = (float)std::clamp((lum[k] - hmin) / span, 0.0, 1.0);
				// NOTE: no border lift. Flattening the outer ring so blocks
				// met flush AT the seam just traded a dark lattice for a
				// bright one — the seam still knew where it was. The tile is
				// left alone and the shader slides each cell's pattern by its
				// own offset instead (faceUV), which moves the mortar off the
				// boundary rather than papering over it.
			} else {
				// flat material: fully flush, no carve at all
				for (int k = 0; k < 256; k++)
					hgt[k] = 1.0f;
			}
			if (is_ore) {
				// Height from COLOR DISTANCE to the tile mean, not
				// luminance: stone base (near the mean) recedes one
				// sub-voxel, ore bits (far from it) stay flush — coal
				// pokes out instead of reading as holes. The ^1.5 pushes
				// stone-noise mids down so only real bits stand proud
				// (the spec mask keys off h > ~0.78).
				double dist[256], dmax = 1.0;
				for (int k = 0; k < 256; k++) {
					double dr = (double)((buf[k] >> 16) & 0xFF) - avg[0];
					double dg = (double)((buf[k] >> 8) & 0xFF) - avg[1];
					double db = (double)(buf[k] & 0xFF) - avg[2];
					dist[k] = std::sqrt(dr * dr + dg * dg + db * db);
					dmax = std::max(dmax, dist[k]);
				}
				for (int k = 0; k < 256; k++) {
					double nd = std::pow(std::clamp(dist[k] / dmax, 0.0, 1.0), 1.5);
					hgt[k] = (float)(0.65 + 0.35 * nd);
				}
			}
			for (int py = 0; py < 16; py++) {
				for (int px = 0; px < 16; px++) {
					u32 t = buf[py * 16 + px];
					u32 c[3] = {(t >> 16) & 0xFFu, (t >> 8) & 0xFFu, t & 0xFFu};
					u32 lv = (u32)std::clamp(hgt[py * 16 + px] * 255.0f + 0.5f,
							0.0f, 255.0f);
					u32 o = (lv << 24);
					for (int ch = 0; ch < 3; ch++) {
						double d = (double)c[ch] / avg[ch] * 0.5 * 255.0;
						u32 v = (u32)std::clamp(d, 0.0, 255.0);
						o |= v << (16 - ch * 8);
					}
					dst[(ay + py) * 256 + ax + px] = o;
				}
			}
			img->drop();
			ok = true;
		}
	}
	if (!ok) {
		// neutral detail (0.5 encodes 1.0x): texture-free material
		for (int py = 0; py < 16; py++)
			for (int px = 0; px < 16; px++)
				dst[(ay + py) * 256 + ax + px] = 0xFF808080u;
	}
	// (The per-material 16^3 stamp bake that lived here is gone with the
	// runtime carve, ADR-0011 — sub-voxel shape comes from the model
	// shop's authored 16^3 models, nowhere else.)
	(void)fallback;
	g_claude_grid.atlas_dirty = true;
}

// GL_LUMINANCE was REMOVED in the OpenGL core profile, so these uploads
// silently produced nothing there: the coarse brick map read zero everywhere,
// every ray leapt past all geometry, and the traced world came out as pure
// sky. GL.R8/GL.RED is the core replacement and exists from 3.0; the legacy
// 2.1 driver has only LUMINANCE. Shaders read .r either way.
static bool claudeUseR8()
{
	static int cached = -1;
	if (cached < 0) {
		const char *v = (const char *)GL.GetString(GL.VERSION);
		int major = (v && v[0] >= '0' && v[0] <= '9') ? (v[0] - '0') : 2;
		cached = (major >= 3) ? 1 : 0;
	}
	return cached == 1;
}

// claude_grid_snapshot: walk the client's loaded map ±SIZE/2 nodes around
// the camera into a solid/air occupancy grid and upload it as a GL.R8 3D
// texture on unit 4. One-shot: the grid does not follow the camera
// afterwards (streaming updates are a later patch). Unloaded map (IGNORE)
// reads as air, so rays pass through it and miss. Runs on the main thread
// Authored-model loader (phase 4.5 v1: occupancy shapes only). Reads
// <path_user>/util/claude_models/manifest.json + model JSONs once per
// run; masks are pre-rotated to the four facedir yaws. Emission and
// per-voxel color are v2 (the furnace mouth still renders as an
// uncarved glowing cube until then).
static void claudeLoadModels(const NodeDefManager *ndef)
{
	auto &V = g_claude_grid;
	if (V.models_loaded)
		return;
	V.models_loaded = true; // one attempt; missing files = no models
	std::string dir = porting::path_user + "/util/claude_models";
	Json::Value manifest;
	{
		std::ifstream f(dir + "/manifest.json");
		if (!f.good())
			return;
		try { f >> manifest; } catch (...) { return; }
	}
	const Json::Value &nodes = manifest["nodes"];
	for (const auto &mname : nodes.getMemberNames()) {
		std::ifstream mf(dir + "/" + mname + ".json");
		if (!mf.good())
			continue;
		Json::Value md;
		try { mf >> md; } catch (...) { continue; }
		const Json::Value &vox = md["voxels"];
		if (vox.size() != 16)
			continue;
		const Json::Value &pal = md["palette"];
		std::vector<u8> palrgba(256 * 4, 0);
		int npal = std::min((int)pal.size(), 256);
		for (int p = 1; p < npal; p++) {
			if (pal[p].isNull())
				continue;
			const Json::Value &rgb = pal[p]["rgb"];
			palrgba[p * 4 + 0] = (u8)rgb[0].asInt();
			palrgba[p * 4 + 1] = (u8)rgb[1].asInt();
			palrgba[p * 4 + 2] = (u8)rgb[2].asInt();
			int emit = pal[p]["emit"].asInt();
			palrgba[p * 4 + 3] = (u8)(std::min(emit, 15) * 255 / 15);
		}
		std::array<std::vector<u8>, 4> rots;
		std::array<std::vector<u8>, 4> vrots;
		std::array<std::array<float, 5>, 4> glow;
		for (int r = 0; r < 4; r++) {
			rots[r].assign(512, 0);
			vrots[r].assign(4096, 0);
			glow[r] = {0, 0, 0, 0, 0};
		}
		double glow_sq = 0.0; // rotation-invariant spread accumulator
		for (int z = 0; z < 16; z++)
		for (int y = 0; y < 16; y++)
		for (int x = 0; x < 16; x++) {
			int pi = vox[z][y][x].asInt();
			if (pi == 0)
				continue;
			int emit = (pi < npal && !pal[pi].isNull())
					? pal[pi]["emit"].asInt() : 0;
			int rx = x, rz = z;
			for (int r = 0; r < 4; r++) {
				rots[r][(size_t)(rz * 16 + y) * 2 + (rx >> 3)]
						|= (u8)(1 << (rx & 7));
				vrots[r][(size_t)(rz * 16 + y) * 16 + rx] =
						(u8)std::min(pi, 255);
				if (emit > 0) {
					glow[r][0] += rx + 0.5f;
					glow[r][1] += y + 0.5f;
					glow[r][2] += rz + 0.5f;
					glow[r][3] += 1.0f;
					if (r == 0)
						glow_sq += (rx + 0.5) * (rx + 0.5)
								+ (y + 0.5) * (y + 0.5)
								+ (rz + 0.5) * (rz + 0.5);
				}
				int nx = 15 - rz, nz = rx; // 90 deg about +y
				rx = nx; rz = nz;
			}
		}
		float glowmax = 0.0f;
		for (int z = 0; z < 16 && glow[0][3] > 0; z++)
		for (int y = 0; y < 16; y++)
		for (int x = 0; x < 16; x++) {
			int pi = vox[z][y][x].asInt();
			if (pi > 0 && pi < npal && !pal[pi].isNull())
				glowmax = std::max(glowmax,
						(float)pal[pi]["emit"].asInt());
		}
		for (int r = 0; r < 4; r++) {
			if (glow[r][3] > 0) {
				float n = glow[r][3];
				glow[r][0] /= n * 16.0f; // cell-local 0..1
				glow[r][1] /= n * 16.0f;
				glow[r][2] /= n * 16.0f;
				if (r == 0) {
					// RMS spread of glow voxels about the centroid, in
					// cell units — the flame's physical size. x1.6 turns
					// RMS into an effective extent (a uniform blob's rim).
					double mx = glow[r][0] * 16.0, my = glow[r][1] * 16.0,
							mz = glow[r][2] * 16.0;
					double var = glow_sq / n
							- (mx * mx + my * my + mz * mz);
					float rad = (float)(1.6 * std::sqrt(std::max(var, 0.0))
							/ 16.0);
					glow[0][4] = glow[1][4] = glow[2][4] = glow[3][4] =
							std::clamp(rad, 0.03f, 0.45f);
				}
				glow[r][3] = glowmax / 14.0f;     // NEE intensity
			}
		}
		V.models.push_back(rots);
		V.model_vox.push_back(vrots);
		V.model_pal.push_back(palrgba);
		V.model_glow.push_back(glow);
		V.model_tex_dirty = true;
		u8 idx = (u8)V.models.size(); // 1-based
		for (const auto &nn : nodes[mname]) {
			content_t cid = ndef->getId(nn.asString());
			if (cid != CONTENT_IGNORE)
				V.model_of[cid] = idx;
		}
		if (V.models.size() >= 63)
			break; // idx<<2 | rot must fit one byte
	}
	infostream << "[claude_models] " << V.models.size() << " models, "
			<< V.model_of.size() << " node bindings" << std::endl;
}

// ---- INCREMENTAL VOLUME RE-SNAP -------------------------------------
// (2026-08-16; spec/measured.md "Volume re-snap fix".) The grid used to
// be rebuilt from scratch on a 2 s timer that the 1 Hz poll quantized to
// every 3 s. MEASURED at the cozy-ci vantage, Release: 12.0 ms of CPU
// walk on EVERY tick even when nothing had changed — the unchanged-hash
// early return skips the upload, never the walk — and 28.6 ms when
// something had (walk 12.5, bake 4.4, GL 10.9, emitters 0.7). Worst
// frame with follow on 39.7 ms vs 15.8 ms with it off, same seat, same
// scene, one variable. That is a hitch you can see, and it ran on a
// clock rather than on the world changing.
//
// Now the CPU mirrors live in ClaudeTraceGrid, the client pushes changed
// block positions into a dirty set (Client::claudeMarkBlockDirty, fed
// from addNode / removeNode / handleCommand_BlockData), and the poll
// re-walks only the 16^3 grid blocks those touch and uploads only that
// sub-box. The full walk survives for exactly two callers: bootstrap,
// and a >24-node re-centre — which moves the origin, so nothing can be
// reused (the deadband makes that rare, and shift-and-fill is not worth
// the class of bug it invites).

// Pack a sub-box out of one of the linear 3D CPU mirrors so that a
// glTexSubImage3D has a contiguous source. The alternative,
// GL_UNPACK_ROW_LENGTH / IMAGE_HEIGHT, leaves global pixel-store state
// behind for Irrlicht to trip over; a 16^3 box is 16 KB, so packing is
// not worth being clever about.
static void claudePackBox(const u8 *src, int dx, int dy, int comp,
		int x0, int y0, int z0, int w, int h, int d, std::vector<u8> &out)
{
	out.resize((size_t)w * h * d * comp);
	const size_t rowbytes = (size_t)w * comp;
	size_t o = 0;
	for (int z = z0; z < z0 + d; z++)
	for (int y = y0; y < y0 + h; y++) {
		memcpy(&out[o], src + (((size_t)z * dy + y) * dx + x0) * comp,
				rowbytes);
		o += rowbytes;
	}
}

// Walk ONE 16^3 grid block into the persistent CPU mirrors.
//
// This is the only place the map is read and cells are classified. The
// full walk is 512 calls to this, so the incremental path CANNOT drift
// from the full path: gate 4 asks the two to produce the identical
// grid, and the only honest way to promise that is to have one
// implementation rather than two that agree today.
// 1 (default) = read stairs/slabs/beds from ContentFeatures::node_box;
// 0 = every solid stays an honest 1 m cube, i.e. the behaviour before
// 2026-08-16. An explicit default rather than an absent-key fallback,
// because an unset dial silently taking a hidden default is a whole bug
// class in this tree (environment-laws.md, "the hidden-default class").
//
// This is also the A/B toggle the debugging discipline requires: the
// cost recorded in spec/measured.md "Nodebox geometry" is one dial apart.
static bool claudeNodeBoxEnabled()
{
	if (!g_settings->exists("claude_nodebox"))
		return true;
	return g_settings->getFloat("claude_nodebox", 0.0f, 1.0f) >= 0.5f;
}

// Ring-local index for the sub-voxel ring, or -1 outside it. The ring is
// the ONLY place a cell can carry 16^3 bits (game.cpp's subvox texture is
// 32 cells wide), so it gates the nodebox path exactly as it gates the
// authored-model path for point lights.
static inline int claudeNBoxRing(int x, int y, int z)
{
	constexpr int R0 = ClaudeTraceGrid::NBOX_R0, R1 = ClaudeTraceGrid::NBOX_R1;
	constexpr int W = ClaudeTraceGrid::NBOX_RING;
	if (x < R0 || x >= R1 || y < R0 || y >= R1 || z < R0 || z >= R1)
		return -1;
	return ((z - R0) * W + (y - R0)) * W + (x - R0);
}

// One node's real box list -> a 16^3 mask id, memoised per shape.
//
// The rasterizer runs once per DISTINCT (content, param2), not once per
// cell: a cabin floor of 200 identical slabs converts one box list. The
// cache is never invalidated because a node definition does not change
// inside a session; param2 is in the key, so a rotated stair is a
// different entry rather than a stale one.
//
// Returns 0 for "leave this cell a 1 m cube", which is the answer for
// every shape the grid must not express: an empty box list (a
// NODEBOX_REGULAR cube), a box list that fills the whole cell anyway
// (a mask that costs 512 bytes to say nothing), and a shape found after
// the budget is spent. 0 is CACHED too, so a fallback costs one
// rasterization, not one per cell per walk.
static u16 claudeNodeBoxMaskId(const NodeDefManager *ndef, const MapNode &n,
		content_t c, const ContentFeatures &f)
{
	ClaudeTraceGrid &V = g_claude_grid;
	u32 key = ((u32)c << 8) | (u32)n.getParam2();
	auto it = V.nbox_of.find(key);
	if (it != V.nbox_of.end())
		return it->second;

	u16 id = 0;
	if (V.nbox_masks.size() < ClaudeTraceGrid::NBOX_CAP) {
		std::vector<aabb3f> boxes;
		// Luanti's OWN reader, not a re-derivation of it: this is the
		// function the mesh generator and the raycast use, so the traced
		// shape cannot drift from the shape the player collides with.
		// neighbours = 0 is safe because claudeNodeBoxConvertible()
		// refuses NODEBOX_CONNECTED, the only type that reads them.
		n.getNodeBoxes(ndef, &boxes, 0);
		std::array<u8, 512> mask{};
		int bits = claudeRasterizeBoxes(boxes, mask.data());
		if (bits > 0 && bits < 4096) {
			V.nbox_masks.push_back(mask);
			id = (u16)V.nbox_masks.size();
		}
	} else if (V.nbox_of.size() == ClaudeTraceGrid::NBOX_CAP) {
		warningstream << "[claude_grid] nodebox shape cache full at "
				<< ClaudeTraceGrid::NBOX_CAP
				<< "; further shapes render as 1 m cubes" << std::endl;
	}
	V.nbox_of[key] = id;
	return id;
}

static void claudeTraceGridWalkBlock(Client *client, const NodeDefManager *ndef,
		Map &map, v3s16 origin, int b)
{
	ClaudeTraceGrid &V = g_claude_grid;
	constexpr int S = ClaudeTraceGrid::SIZE;
	constexpr int B = ClaudeTraceGrid::BLK;
	constexpr int NB = ClaudeTraceGrid::NB;
	const int bx = b % NB, by = (b / NB) % NB, bz = b / (NB * NB);
	// The point emitters this block owned die with the re-walk and are
	// re-pushed below. Miss this and next-event estimation keeps aiming
	// at a lamp that was dug: the room stays lit and nothing says why.
	V.emit_all.erase(std::remove_if(V.emit_all.begin(), V.emit_all.end(),
			[&](const ClaudeTraceGrid::PointEmit &e) {
				return (int)(e.cell % S) / B == bx
						&& (int)((e.cell / S) % S) / B == by
						&& (int)(e.cell / ((size_t)S * S)) / B == bz;
			}), V.emit_all.end());
	u8 *occ = V.occ.data();
	u8 *mids = V.mids.data();
	size_t i = 0; // the cell the loop below is on; captured by emit_push
	auto emit_push = [&](const std::array<float, 5> &e) {
		V.emit_all.push_back({(u32)i, {e[0], e[1], e[2], e[3], e[4]}});
	};
	// Seeded by block index, so 512 empty blocks do not all hash alike
	// and a solid that MOVES between blocks changes both their hashes.
	u64 hash = 14695981039346656037ULL ^ (u64)b;
	u32 solid = 0;
	for (s16 z = bz * B; z < (bz + 1) * B; z++)
	for (s16 y = by * B; y < (by + 1) * B; y++)
	for (s16 x = bx * B; x < (bx + 1) * B; x++) {
		i = ((size_t)z * S + y) * S + x;
		// Air until this walk says otherwise. The old full walk got this
		// for free from a freshly zeroed vector; an in-place re-walk has
		// to do it explicitly, and a dug node is exactly the case that
		// would otherwise leave the old class sitting there.
		occ[i * 4 + 0] = 0; occ[i * 4 + 1] = 0;
		occ[i * 4 + 2] = 0; occ[i * 4 + 3] = 0;
		mids[i] = 0;
		V.modelids[i] = 0;
		// Ring-local shape id, cleared for the same reason modelids is:
		// an in-place re-walk that leaves the old id sitting there is a
		// dug stair whose shadow stays behind.
		int nbring = claudeNBoxRing(x, y, z);
		if (nbring >= 0 && !V.nbox_ids.empty())
			V.nbox_ids[nbring] = 0;
		u16 nbid = 0;
		MapNode n = map.getNode(origin + v3s16(x, y, z));
		content_t c = n.getContent();
		if (c == CONTENT_AIR || c == CONTENT_IGNORE)
			continue;
		const ContentFeatures &f = ndef->get(c);
		video::SColor col(255, 180, 180, 180);
		if (f.visuals && f.visuals->minimap_color.getAlpha() > 0)
			col = f.visuals->minimap_color;
		// biome/palette tint (Mineclonia grass and leaves are grayscale
		// tiles colored per-node through param2) — modulate the cell color
		if (f.visuals) {
			video::SColor tint(255, 255, 255, 255);
			f.visuals->getColor(n.getParam2(), &tint);
			if (tint.getRed() != 255 || tint.getGreen() != 255
					|| tint.getBlue() != 255) {
				col.setRed(col.getRed() * tint.getRed() / 255);
				col.setGreen(col.getGreen() * tint.getGreen() / 255);
				col.setBlue(col.getBlue() * tint.getBlue() / 255);
			}
		}
		// ADR-0009 #3: authored color SURVIVES for full-cube emitters —
		// rho and Le come from the node's real average color, or albedo
		// variants cannot exist (the warm-force pinned every emissive
		// cell to r=255,g>=200,b>=120 and the furnace rho carried only
		// in blue). The transparent-texture problem the force was built
		// for (torches averaging near-black) lives in the POINT-light
		// branch below, which sets its own warm color explicitly.
		occ[i * 4 + 0] = col.getRed();
		occ[i * 4 + 1] = col.getGreen();
		occ[i * 4 + 2] = col.getBlue();
		// Non-occluding decorations: grass tufts, flowers, rails, signs are
		// quads inside a cell, not cubes — leaving them solid makes a
		// flower cast a full block shadow. Emissive ones (torches) stay,
		// since they are light sources.
		if (f.light_source == 0
				&& (f.drawtype == NDT_PLANTLIKE
					|| f.drawtype == NDT_PLANTLIKE_ROOTED
					|| f.drawtype == NDT_FIRELIKE
					|| f.drawtype == NDT_SIGNLIKE
					|| f.drawtype == NDT_RAILLIKE
					|| f.drawtype == NDT_TORCHLIKE))
			continue;

		// alpha = occupancy class: 0 air, 100 water, 130 leaves (partial
		// transmission), 145 glass (see-through), 170..240 emissive
		// (170 + light_source*5, so shaders recover brightness), 255 solid
		// Invisible light nodes (wielded_light's airlike emitters) should
		// light the world without rendering as a glowing cube.
		// Invisible light nodes are wielded_light's raster-era workaround
		// (a light node dropped in the world a beat behind the player).
		// We now light the held item client-side every frame, so these
		// only produce a duplicate, laggy second light — skip entirely.
		if (f.light_source > 0 && f.drawtype == NDT_AIRLIKE)
			continue;

		u8 acls = 255;
		// THE VOXEL LAW (John, 2026-08-13): "we render voxels. some
		// voxels are 1/16m, some are 1m. period." A 1m voxel is never
		// carved at runtime. Class 250 (sub-voxel geometry) is granted
		// ONLY by the authored-model branch below; every other solid is
		// an honest 1m cube. The per-tile heightfield carve that used to
		// run here survives solely as a bake-time generator in the model
		// shop (util/claude_models.py).
		// Small emitters (torches) are thin sticks inside their cell.
		// Class 165 renders them as a sub-voxel nub instead of a full
		// glowing cube that looks like it replaced a block.
		if (f.light_source > 0 && !f.isLiquid()
				&& f.drawtype != NDT_NORMAL) {
			// AUTHORED MODEL for a point-light node (torch, lantern,
			// campfire — John: "where are my real torches!"): the
			// model renders as real sub-voxel geometry with its own
			// palette (flame voxels glow via the atlas) and the LIGHT
			// stays a point at the model's glow centroid — same law
			// as the modeled furnace.
			// Ring-gated: outside the subvox ring [48,80)^3 a class-250
			// cell has no bits to express and would render as a 1m cube
			// in the node's near-black average color (John's "black
			// cube"). Distant point lights keep the 165 nub until the
			// ladder rework gives models real far rungs.
			if (x >= 48 && x < 80 && y >= 48 && y < 80
					&& z >= 48 && z < 80) {
				auto mit = g_claude_grid.model_of.find(c);
				if (mit != g_claude_grid.model_of.end()) {
					u8 rot = n.getParam2() & 3;
					g_claude_grid.modelids[i] =
							(u8)((mit->second << 2) | rot);
					const auto &gl = g_claude_grid
							.model_glow[mit->second - 1][rot];
					if (gl[3] > 0.0f)
						emit_push({x + gl[0], y + gl[1],
								z + gl[2],
								std::min<int>(f.light_source, 14)
									/ 14.0f, gl[4]});
					occ[i * 4 + 0] = col.getRed();
					occ[i * 4 + 1] = col.getGreen();
					occ[i * 4 + 2] = col.getBlue();
					occ[i * 4 + 3] = 250;
					// ROTATION IS SHAPE (2026-08-16). See the general
					// path below for why param2 has to reach the hash.
					hash = hash * 1099511628211ULL
							+ (u64)i * 7919 + 250
							+ (u64)g_claude_grid.modelids[i] * 31;
					solid++;
					continue;
				}
			}
			// Any sub-block light model (Mineclonia torches are MESH,
			// not torchlike; also lanterns, plants, fire): pure point.
			// Only full-cube glowing blocks (glowstone, lamps,
			// NDT_NORMAL) keep their geometry.
			// PURE POINT LIGHT (John, 2026-08-12): emitter at flame
			// height, and the cell carries class 165 ONLY so the eye
			// ray draws a small glowing nub at the emission point
			// ("give the torch a single glowing thing right where it
			// emits") — light rays pass through it (cellTransmit 1.0),
			// so it still cannot occlude or re-radiate its own light.
			emit_push({(float)x + 0.5f, (float)y + 0.65f,
					(float)z + 0.5f,
					std::min<int>(f.light_source, 14) / 14.0f,
					0.08f}); // nub: assume a small flame
			occ[i * 4 + 0] = 255; occ[i * 4 + 1] = 220; occ[i * 4 + 2] = 150;
			occ[i * 4 + 3] = 165;
			hash = hash * 1099511628211ULL + (u64)i * 7919 + 165;
			solid++;
			continue;
		}
		if (f.light_source > 0) {
			// emissive beats liquid: lava must GLOW, not mirror
			acls = 170 + (u8)std::min<int>(f.light_source, 14) * 5;
			// NEE list is for POINT lights only (ADR-0009 #4). Full-cube
			// area emitters were ALSO pushed here, double-counting them:
			// once as geometry the path hits, once as an aimed point
			// light — in an all-emissive furnace room the 8 nearest wall
			// cells became extra lights. Area emitters are barn doors
			// the ambient ray can't miss; they get no aimed slot.
		} else if (f.isLiquid())
			acls = 100;
		else if (f.drawtype == NDT_ALLFACES
				|| f.drawtype == NDT_ALLFACES_OPTIONAL)
			acls = 130;
		else if (f.drawtype == NDT_GLASSLIKE
				|| f.drawtype == NDT_GLASSLIKE_FRAMED
				|| f.drawtype == NDT_GLASSLIKE_FRAMED_OPTIONAL)
			acls = 145;
		// authored model: tag the cell and join the micro class so
		// every path carves it. v2: emissive full-cube nodes (the lit
		// furnace) join too — their fire voxels render via the palette
		// and CAST via a point light at the glow centroid (the
		// small-emitter law; honest until the NEE+MIS area block).
		// Sub-cube point lights (torch/lantern/campfire, class 165)
		// keep their existing nub+NEE treatment.
		if (!g_claude_grid.model_of.empty()
				&& (acls >= 250 || (f.light_source > 0
					&& f.drawtype == NDT_NORMAL))) {
			auto mit = g_claude_grid.model_of.find(c);
			if (mit != g_claude_grid.model_of.end()) {
				acls = 250;
				u8 rot = n.getParam2() & 3;
				g_claude_grid.modelids[i] =
						(u8)((mit->second << 2) | rot);
				const auto &gl = g_claude_grid
						.model_glow[mit->second - 1][rot];
				if (gl[3] > 0.0f)
					emit_push({x + gl[0], y + gl[1],
							z + gl[2], gl[3], gl[4]});
			}
		}
		// REAL GEOMETRY FROM THE NODE DEFINITION. After the authored-model
		// branch on purpose: a hand-made model beats an extracted box
		// list, so model_of stays the override it has always been.
		//
		// Gated to acls == 255 (a plain opaque solid): water, glass,
		// leaves and every emissive class keep the classification they
		// have today, and NODEBOX_REGULAR cannot reach here at all
		// because claudeNodeBoxConvertible() refuses it (handoff gate 2).
		if (V.nodebox_on && nbring >= 0 && acls == 255
				&& V.modelids[i] == 0
				&& f.drawtype == NDT_NODEBOX
				&& claudeNodeBoxConvertible(f.node_box)) {
			nbid = claudeNodeBoxMaskId(ndef, n, c, f);
			if (nbid) {
				acls = 250;              // "this cell has sub-voxel bits"
				V.nbox_ids[nbring] = nbid;
			}
		}
		occ[i * 4 + 3] = acls;
		hash = hash * 1099511628211ULL + (u64)i * 7919 + acls + col.getRed();
		// SHAPE IS PART OF THE CONTENT, and until 2026-08-16 it was not.
		//
		// The hash decides whether a re-walked block gets baked and
		// uploaded (claudeTraceGridIncremental: `if (block_hash[b] ==
		// before) continue;`). It mixed the class byte and the red
		// channel and NOTHING derived from param2 -- so rotating a node
		// in place re-walked the block, updated the CPU mirror, produced
		// an identical hash, and skipped the upload. The GPU kept the old
		// shape and the log said "none changed". That was already true of
		// the authored models' `rot`; a nodebox makes it true of every
		// stair, and the handoff names it as the landmine that
		// invalidates the whole step.
		//
		// The mask ID is the right thing to mix, not param2 itself: it is
		// the geometry that will actually be baked, so two param2 values
		// that mean the same shape correctly hash the same, and a param2
		// change that means nothing here (a biome tint, a liquid level)
		// does not churn the upload path.
		if (nbid)
			hash = hash * 1099511628211ULL + (u64)nbid * 7919;
		else if (V.modelids[i])
			hash = hash * 1099511628211ULL + (u64)V.modelids[i] * 31;
		// material id (0 = untextured); palette + atlas grow on first sight
		auto pit = g_claude_grid.palette.find(c);
		if (pit != g_claude_grid.palette.end()) {
			mids[i] = pit->second;
		} else if (g_claude_grid.palette.size() < 254) {
			u8 mid = (u8)(g_claude_grid.palette.size() + 1);
			g_claude_grid.palette[c] = mid;
			claudeAtlasAdd(client, mid, f, col);
			mids[i] = mid;
		}
		solid++;
	}
	V.block_hash[b] = hash;
	V.block_solid[b] = solid;
}

// Combine the 512 per-block hashes ORDER-INDEPENDENTLY, which is the
// whole reason they exist: the old hash was an FNV chain over all 2.1 M
// cells in index order, and a chain cannot be updated for a sub-box.
// XOR is the order-independent part; the splitmix64 finalizer is what
// stops two blocks whose walks happened to land on related values from
// cancelling each other out. Deterministic, so the incremental path and
// a full walk of the same world agree bit for bit (gate 4).
static u64 claudeTraceGridContentHash(v3s16 origin)
{
	u64 h = 14695981039346656037ULL ^ (u64)origin.X
			^ ((u64)origin.Y << 20) ^ ((u64)origin.Z << 40);
	for (int b = 0; b < ClaudeTraceGrid::NBLOCKS; b++) {
		u64 z = g_claude_grid.block_hash[b] + 0x9E3779B97F4A7C15ULL;
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
		h ^= z ^ (z >> 31);
	}
	return h;
}

// Point emitters -> nearest 8; area emitters -> nearest AREA_CAP.
//
// Neither list is patched, both are rebuilt from state that is already
// correct, and that is deliberate:
//
//  - emit_all is re-sorted into whole-grid CELL order first, so the
//    distance sort below sees the identical input sequence a full z,y,x
//    walk would have handed it. std::sort is not stable; a tie decided
//    differently is a different torch in slot 7, an image that moves,
//    and nothing in the diff to explain it.
//  - the area list is regenerated wholesale from the finished occupancy.
//    It costs 0.7 ms MEASURED, and it makes "the incremental list equals
//    the full-walk list" true by construction rather than by assertion.
//    A cell's air-exposed face mask depends on its six neighbours, so an
//    incremental area update would have to re-own a one-cell halo around
//    every dirty block anyway; 0.7 ms buys the entire problem away.
static void claudeTraceGridFinishEmitters()
{
	ClaudeTraceGrid &V = g_claude_grid;
	constexpr int S = ClaudeTraceGrid::SIZE;
	std::sort(V.emit_all.begin(), V.emit_all.end(),
			[](const ClaudeTraceGrid::PointEmit &a,
					const ClaudeTraceGrid::PointEmit &b) {
				return a.cell < b.cell;
			});
	std::vector<std::array<float, 5>> emitters;
	emitters.reserve(V.emit_all.size());
	for (const auto &e : V.emit_all)
		emitters.push_back({e.v[0], e.v[1], e.v[2], e.v[3], e.v[4]});
	// nearest-8 emitters to the camera (= grid center) for NEE
	std::sort(emitters.begin(), emitters.end(),
			[](const std::array<float, 5> &a, const std::array<float, 5> &b) {
				auto d2 = [](const std::array<float, 5> &e) {
					float dx = e[0] - 64.f, dy = e[1] - 64.f, dz = e[2] - 64.f;
					return dx * dx + dy * dy + dz * dz;
				};
				return d2(a) < d2(b);
			});
	V.emitter_count = std::min<size_t>(emitters.size(), 8);
	for (int e = 0; e < 8; e++) {
		for (int k = 0; k < 4; k++)
			V.emitters[e][k] = e < V.emitter_count ? emitters[e][k] : 0.0f;
		V.emitter_rad[e] = e < V.emitter_count ? emitters[e][4] : 0.0f;
	}

	// ---- AREA EMITTERS: the list claude_trace's NEE samples -------------
	// A second pass over the finished occupancy, not a push during the
	// fill loop, for two reasons: the face mask needs neighbours the fill
	// loop has not written yet, and the class byte is not final until the
	// authored-model branch has had its say (an emissive full cube that
	// gets a model becomes class 250, which cellEmission() does NOT
	// answer for — it must not appear here, or the light sampler would
	// aim at a cell with no Le and the MIS weights would disagree with
	// the transport).
	//
	// The band is exactly the shader's: CLASS_EMIT_LO/HI bracket 170..240
	// at the half-byte midpoints, so 170..240 inclusive here is the same
	// set of cells, with no 8-bit round-trip on either side.
	{
		const u8 *occ = V.occ.data();
		std::vector<std::array<float, 4>> areas;
		auto is_air = [&](s16 ax, s16 ay, s16 az) {
			// Out of the grid counts as NOT air: the outward faces of a
			// boundary cell can only be seen from outside the 128^3, and
			// no ray ever starts there. Calling them exposed would put
			// area in the pdf that no BSDF sample can ever reach.
			if (ax < 0 || ay < 0 || az < 0 || ax >= S || ay >= S || az >= S)
				return false;
			size_t j = (((size_t)az * S + ay) * S + ax) * 4 + 3;
			return occ[j] == 0;
		};
		for (s16 z = 0; z < S; z++)
		for (s16 y = 0; y < S; y++)
		for (s16 x = 0; x < S; x++) {
			u8 cls = occ[((((size_t)z * S + y) * S + x)) * 4 + 3];
			if (cls < 170 || cls > 240)
				continue;
			int mask = 0;
			if (is_air(x + 1, y, z)) mask |= 1;
			if (is_air(x - 1, y, z)) mask |= 2;
			if (is_air(x, y + 1, z)) mask |= 4;
			if (is_air(x, y - 1, z)) mask |= 8;
			if (is_air(x, y, z + 1)) mask |= 16;
			if (is_air(x, y, z - 1)) mask |= 32;
			// Sealed inside solid: every face is unreachable, so it can
			// contribute nothing and would only inflate the 1/N selection
			// probability of the emitters that CAN be reached.
			if (mask == 0)
				continue;
			areas.push_back({(float)x, (float)y, (float)z, (float)mask});
		}
		// nearest to the camera (= grid centre) first, the same key the
		// point-emitter list uses
		std::sort(areas.begin(), areas.end(),
				[](const std::array<float, 4> &a, const std::array<float, 4> &b) {
					auto d2 = [](const std::array<float, 4> &e) {
						float dx = e[0] - 64.f, dy = e[1] - 64.f, dz = e[2] - 64.f;
						return dx * dx + dy * dy + dz * dz;
					};
					return d2(a) < d2(b);
				});
		V.area_total = (int)areas.size();
		V.area_count = (int)std::min<size_t>(areas.size(),
				ClaudeTraceGrid::AREA_CAP);
		for (int e = 0; e < ClaudeTraceGrid::AREA_CAP; e++)
			for (int k = 0; k < 4; k++)
				V.area[e][k] = e < V.area_count ? areas[e][k] : 0.0f;
		// NO SILENT TRUNCATION. Overflow is not a correctness failure —
		// the shader's light-sampling pdf is zero for an unlisted cell, so
		// the balance heuristic hands those emitters' full radiance to the
		// BSDF sample and the image still converges to photo mode — but it
		// IS a variance failure, and it must be visible in the log when a
		// room looks noisier than it should.
		if (V.area_total > V.area_count) {
			const auto &first_drop = areas[V.area_count];
			warningstream << "[claude_grid] area emitters "
					<< V.area_total << " > cap "
					<< ClaudeTraceGrid::AREA_CAP << ": dropping "
					<< (V.area_total - V.area_count)
					<< " from next-event sampling, farthest first; nearest"
					   " dropped cell is (" << (int)first_drop[0] << ","
					<< (int)first_drop[1] << "," << (int)first_drop[2]
					<< ") at " << std::sqrt(
							(first_drop[0] - 64.f) * (first_drop[0] - 64.f)
							+ (first_drop[1] - 64.f) * (first_drop[1] - 64.f)
							+ (first_drop[2] - 64.f) * (first_drop[2] - 64.f))
					<< " cells. Dropped emitters still light the scene"
					   " through BSDF sampling (MIS weight 1), so the image"
					   " stays correct and gets noisier." << std::endl;
		}
	}
}

// REAL SUB-VOXEL BITS: the ring [48,80)^3 (grid-local; the grid
// follows the camera) holds one bit per 1/16 m voxel — the ONLY
// sub-voxel occupancy the renderer ever sees. Modeled cells take their
// authored 16^3 mask; every other solid is 4096 ones, a plain 1m voxel.
// No runtime carving (John, 2026-08-13: "true 1m voxels no longer get
// carved unless they have a 1/16 higher res model").
//
// Re-baked over the given cell box only. Each ring cell zeroes its own
// 512 bytes before deciding, so a box refill leaves exactly what a full
// re-bake would (the old code std::fill'd the whole 16 MB first, which
// is not something a sub-box may do).
static void claudeTraceGridBakeSubvox(int x0, int y0, int z0, int w, int h, int d)
{
	ClaudeTraceGrid &V = g_claude_grid;
	constexpr int S = ClaudeTraceGrid::SIZE;
	// ONE ring definition. The nodebox path indexes nbox_ids with the
	// same constants (claudeNBoxRing), and two copies of 48/80 is how a
	// mask ends up written one cell away from the cell that claims it.
	const int R0 = ClaudeTraceGrid::NBOX_R0, R1 = ClaudeTraceGrid::NBOX_R1;
	auto &sv = V.subvox;
	if (sv.empty())
		sv.assign((size_t)64 * 512 * 512, 0);
	int lx = std::max(x0, R0), hx = std::min(x0 + w, R1);
	int ly = std::max(y0, R0), hy = std::min(y0 + h, R1);
	int lz = std::max(z0, R0), hz = std::min(z0 + d, R1);
	for (int vz = lz; vz < hz; vz++)
	for (int vy = ly; vy < hy; vy++)
	for (int vx = lx; vx < hx; vx++) {
		int rx = vx - R0, ry = vy - R0, rz = vz - R0;
		size_t vi = (size_t)(vz * S + vy) * S + vx;
		u8 a = V.occ[vi * 4 + 3];
		u8 mtag = a > 230 ? V.modelids[vi] : 0;
		const u8 *mm = nullptr;
		if (mtag >> 2) {
			// authored model: its pre-rotated mask IS the cell
			mm = V.models[(mtag >> 2) - 1][mtag & 3].data();
		} else if (a > 230 && !V.nbox_ids.empty()) {
			// NODE-DEFINITION GEOMETRY (stairs, slabs, beds). Same
			// 512-byte layout, same consumer, one rung below an authored
			// model in priority. rx/ry/rz are already ring-local.
			u16 nb = V.nbox_ids[((size_t)rz * ClaudeTraceGrid::NBOX_RING
					+ ry) * ClaudeTraceGrid::NBOX_RING + rx];
			if (nb)
				mm = V.nbox_masks[nb - 1].data();
		}
		// a <= 230 is air / water / glass / nub: no bits
		u8 fill = (a > 230 && !mm) ? 0xFF : 0x00;
		for (int sz2 = 0; sz2 < 16; sz2++)
		for (int sy2 = 0; sy2 < 16; sy2++) {
			size_t row = ((size_t)(rz * 16 + sz2) * 512
					+ (ry * 16 + sy2)) * 64 + (size_t)rx * 2;
			if (mm) {
				sv[row] = mm[(size_t)(sz2 * 16 + sy2) * 2];
				sv[row + 1] = mm[(size_t)(sz2 * 16 + sy2) * 2 + 1];
			} else {
				sv[row] = fill;
				sv[row + 1] = fill;
			}
		}
	}
}

// OCCUPANCY MIP PYRAMID on unit 11 (Teardown's accelerator, ADR-0007
// overnight 2026-08-12): the old 32^3 brick map is level 2 of a 128^3 R8
// texture with real GL mips 0..5 (any-content, max-reduced). Shaders
// sample with textureLod: level 2 reproduces the old leap exactly; the
// claude_pyramid dial lets marchers CLIMB to 8/16/32-cell leaps through
// deep emptiness.
//
// THE BRICK MAP IS RECOMPUTED FROM CELLS, NEVER TOGGLED (handoff
// landmine): a dug node inside a brick that still holds other solids
// must not clear the brick, and the only way to be sure of that is to
// re-reduce from the occupancy. Levels 0..2 are redone over the dirty
// box; 3..5 are 4.6 KB in total and are redone wholesale, which also
// sidesteps a 1-wide row hitting GL's 4-byte unpack alignment.
static void claudeTraceGridBakePyramid(int x0, int y0, int z0, int w, int h, int d)
{
	ClaudeTraceGrid &V = g_claude_grid;
	constexpr int S = ClaudeTraceGrid::SIZE;
	for (int z = z0; z < z0 + d; z++)
	for (int y = y0; y < y0 + h; y++)
	for (int x = x0; x < x0 + w; x++) {
		size_t i = ((size_t)z * S + y) * S + x;
		V.pyr[0][i] = V.occ[i * 4 + 3] ? 255 : 0;
	}
	auto reduce = [&](int lvl, int n, int rx0, int ry0, int rz0,
			int rw, int rh, int rd) {
		const std::vector<u8> &src = V.pyr[lvl - 1];
		std::vector<u8> &dst = V.pyr[lvl];
		for (int z = rz0; z < rz0 + rd; z++)
		for (int y = ry0; y < ry0 + rh; y++)
		for (int x = rx0; x < rx0 + rw; x++) {
			u8 v = 0;
			for (int k = 0; k < 8 && !v; k++) {
				int sx = x * 2 + (k & 1), sy = y * 2 + ((k >> 1) & 1),
					sz = z * 2 + (k >> 2);
				v |= src[((size_t)sz * n * 2 + sy) * n * 2 + sx];
			}
			dst[((size_t)z * n + y) * n + x] = v;
		}
	};
	// levels 1 and 2 over the box, shrinking (lo rounds down, hi up)
	for (int lvl = 1; lvl <= 2; lvl++) {
		int n = S >> lvl;
		int lx = x0 >> lvl, ly = y0 >> lvl, lz = z0 >> lvl;
		int hx = (x0 + w + (1 << lvl) - 1) >> lvl;
		int hy = (y0 + h + (1 << lvl) - 1) >> lvl;
		int hz = (z0 + d + (1 << lvl) - 1) >> lvl;
		reduce(lvl, n, lx, ly, lz, hx - lx, hy - ly, hz - lz);
	}
	// levels 3..5 wholesale: 4096 + 512 + 64 bytes
	for (int lvl = 3; lvl <= 5; lvl++) {
		int n = S >> lvl;
		reduce(lvl, n, 0, 0, 0, n, n, n);
	}
}

static void claudeTraceGridTexParams3D()
{
	GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
	GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
	GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_WRAP_S, GL.CLAMP_TO_EDGE);
	GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_WRAP_T, GL.CLAMP_TO_EDGE);
	GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_WRAP_R, GL.CLAMP_TO_EDGE);
}

// v2 model textures: the voxel-palette atlas + palettes, uploaded once
// on load. Unchanged; lifted out so both upload paths can call it.
static void claudeTraceGridUploadModelTex()
{
	ClaudeTraceGrid &V = g_claude_grid;
	if (!V.model_tex_dirty)
		return;
	V.model_tex_dirty = false;
	// atlas 16x16x1024: layer = (idx0*4+rot)*16+sz, 16 models max
	std::vector<u8> atlas((size_t)16 * 16 * 1024, 0);
	std::vector<u8> pals((size_t)256 * 64 * 4, 0);
	size_t nm = std::min<size_t>(V.model_vox.size(), 16);
	for (size_t m = 0; m < nm; m++) {
		for (int r = 0; r < 4; r++) {
			const auto &vr = V.model_vox[m][r];
			for (int sz2 = 0; sz2 < 16; sz2++) {
				size_t layer = ((m * 4 + r) * 16 + sz2);
				for (int sy2 = 0; sy2 < 16; sy2++)
				for (int sx2 = 0; sx2 < 16; sx2++)
					atlas[(layer * 16 + sy2) * 16 + sx2] =
							vr[(size_t)(sz2 * 16 + sy2) * 16 + sx2];
			}
		}
		memcpy(&pals[m * 256 * 4], V.model_pal[m].data(), 256 * 4);
	}
	bool fresh_ma = !V.model_atlas_tex;
	if (fresh_ma)
		GL.GenTextures(1, &V.model_atlas_tex);
	GL.ActiveTexture(GL.TEXTURE0 + 17);
	GL.BindTexture(GL.TEXTURE_3D, V.model_atlas_tex);
	if (fresh_ma) {
		claudeTraceGridTexParams3D();
		GL.TexImage3D(GL.TEXTURE_3D, 0,
				claudeUseR8() ? GL.R8 : GL_LUMINANCE8, 16, 16, 1024, 0,
				claudeUseR8() ? GL.RED : GL_LUMINANCE,
				GL.UNSIGNED_BYTE, nullptr);
	}
	GL.TexSubImage3D(GL.TEXTURE_3D, 0, 0, 0, 0, 16, 16, 1024,
			claudeUseR8() ? GL.RED : GL_LUMINANCE,
			GL.UNSIGNED_BYTE, atlas.data());
	bool fresh_mp = !V.model_pal_tex;
	if (fresh_mp)
		GL.GenTextures(1, &V.model_pal_tex);
	GL.ActiveTexture(GL.TEXTURE0 + 18);
	GL.BindTexture(GL.TEXTURE_2D, V.model_pal_tex);
	if (fresh_mp) {
		GL.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
		GL.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
		GL.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_S, GL.CLAMP_TO_EDGE);
		GL.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_T, GL.CLAMP_TO_EDGE);
		GL.TexImage2D(GL.TEXTURE_2D, 0, GL.RGBA8, 256, 64, 0,
				GL.RGBA, GL.UNSIGNED_BYTE, nullptr);
	}
	GL.TexSubImage2D(GL.TEXTURE_2D, 0, 0, 0, 256, 64,
			GL.RGBA, GL.UNSIGNED_BYTE, pals.data());
}

// tile atlas on unit 13 + per-material response params on unit 15,
// uploaded only when the palette grew. Append-only, so it is the same
// call on both paths.
static void claudeTraceGridUploadAtlas()
{
	ClaudeTraceGrid &V = g_claude_grid;
	if (!V.atlas_dirty)
		return;
	if (!V.atlas_tex)
		GL.GenTextures(1, &V.atlas_tex);
	GL.ActiveTexture(GL.TEXTURE13);
	GL.BindTexture(GL.TEXTURE_2D, V.atlas_tex);
	GL.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
	GL.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
	GL.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_S, GL.CLAMP_TO_EDGE);
	GL.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_T, GL.CLAMP_TO_EDGE);
	GL.TexImage2D(GL.TEXTURE_2D, 0, GL.RGBA8, 256, 256, 0, GL.BGRA,
			GL.UNSIGNED_BYTE, V.atlas.data());
	// per-material response params (256x1 RGBA), unit 15
	if (!V.matparams_tex)
		GL.GenTextures(1, &V.matparams_tex);
	if (V.matparams.empty())
		V.matparams.assign(256 * 4, 0);
	GL.ActiveTexture(GL.TEXTURE15);
	GL.BindTexture(GL.TEXTURE_2D, V.matparams_tex);
	GL.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
	GL.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
	GL.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_S, GL.CLAMP_TO_EDGE);
	GL.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_T, GL.CLAMP_TO_EDGE);
	GL.TexImage2D(GL.TEXTURE_2D, 0, GL.RGBA8, 256, 1, 0, GL.RGBA,
			GL.UNSIGNED_BYTE, V.matparams.data());
	V.atlas_dirty = false;
}

// Full upload of every grid texture: bootstrap and re-centre only.
static void claudeTraceGridUploadFull()
{
	ClaudeTraceGrid &V = g_claude_grid;
	constexpr int S = ClaudeTraceGrid::SIZE;
	// Save the unit the driver's cache believes is active and restore it
	// at the end — restoring a hard-coded GL.TEXTURE0 desyncs the
	// COpenGLCoreCacheHandler mirror (see the per-frame rebind block).
	GLint prev_active_unit = GL.TEXTURE0;
	GL.GetIntegerv(GL.ACTIVE_TEXTURE, &prev_active_unit);
	if (!V.tex)
		GL.GenTextures(1, &V.tex);
	GL.ActiveTexture(GL.TEXTURE10);
	GL.BindTexture(GL.TEXTURE_3D, V.tex);
	claudeTraceGridTexParams3D();
	GL.TexImage3D(GL.TEXTURE_3D, 0, GL.RGBA8, S, S, S, 0, GL.RGBA,
			GL.UNSIGNED_BYTE, V.occ.data());
	{
		bool fresh_sv = !V.subvox_tex;
		if (fresh_sv)
			GL.GenTextures(1, &V.subvox_tex);
		GL.ActiveTexture(GL.TEXTURE7);
		GL.BindTexture(GL.TEXTURE_3D, V.subvox_tex);
		if (fresh_sv) {
			claudeTraceGridTexParams3D();
			GL.TexImage3D(GL.TEXTURE_3D, 0,
					claudeUseR8() ? GL.R8 : GL_LUMINANCE8, 64, 512, 512,
					0, claudeUseR8() ? GL.RED : GL_LUMINANCE,
					GL.UNSIGNED_BYTE, nullptr);
		}
		GL.TexSubImage3D(GL.TEXTURE_3D, 0, 0, 0, 0, 64, 512, 512,
				claudeUseR8() ? GL.RED : GL_LUMINANCE,
				GL.UNSIGNED_BYTE, V.subvox.data());
		bool fresh_ids = !V.model_ids_tex;
		if (fresh_ids)
			GL.GenTextures(1, &V.model_ids_tex);
		GL.ActiveTexture(GL.TEXTURE0 + 16);
		GL.BindTexture(GL.TEXTURE_3D, V.model_ids_tex);
		if (fresh_ids) {
			claudeTraceGridTexParams3D();
			GL.TexImage3D(GL.TEXTURE_3D, 0,
					claudeUseR8() ? GL.R8 : GL_LUMINANCE8, S, S, S, 0,
					claudeUseR8() ? GL.RED : GL_LUMINANCE,
					GL.UNSIGNED_BYTE, nullptr);
		}
		GL.TexSubImage3D(GL.TEXTURE_3D, 0, 0, 0, 0, S, S, S,
				claudeUseR8() ? GL.RED : GL_LUMINANCE,
				GL.UNSIGNED_BYTE, V.modelids.data());
		claudeTraceGridUploadModelTex();
	}
	{
		if (!V.coarse_tex)
			GL.GenTextures(1, &V.coarse_tex);
		GL.ActiveTexture(GL.TEXTURE11);
		GL.BindTexture(GL.TEXTURE_3D, V.coarse_tex);
		GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_MIN_FILTER,
				GL.NEAREST_MIPMAP_NEAREST);
		GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
		GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_WRAP_S, GL.CLAMP_TO_EDGE);
		GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_WRAP_T, GL.CLAMP_TO_EDGE);
		GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_WRAP_R, GL.CLAMP_TO_EDGE);
		GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_BASE_LEVEL, 0);
		GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_MAX_LEVEL, 5);
		GLenum ifmt = claudeUseR8() ? GL.R8 : GL_LUMINANCE8;
		GLenum fmt = claudeUseR8() ? GL.RED : GL_LUMINANCE;
		for (int lvl = 0; lvl <= 5; lvl++) {
			int n = S >> lvl;
			GL.TexImage3D(GL.TEXTURE_3D, lvl, ifmt, n, n, n, 0, fmt,
					GL.UNSIGNED_BYTE, V.pyr[lvl].data());
		}
	}
	// material-id grid on unit 12
	if (!V.material_tex)
		GL.GenTextures(1, &V.material_tex);
	GL.ActiveTexture(GL.TEXTURE12);
	GL.BindTexture(GL.TEXTURE_3D, V.material_tex);
	claudeTraceGridTexParams3D();
	GL.TexImage3D(GL.TEXTURE_3D, 0, claudeUseR8() ? GL.R8 : GL_LUMINANCE8,
			S, S, S, 0, claudeUseR8() ? GL.RED : GL_LUMINANCE,
			GL.UNSIGNED_BYTE, V.mids.data());
	claudeTraceGridUploadAtlas();
	GL.ActiveTexture(prev_active_unit);
}

// Sub-box upload: the incremental half. Same texture-unit save/restore
// wrapper as the full path — restoring a hard-coded GL.TEXTURE0 desyncs
// Irrlicht's cache mirror, and that is just as true for a 16 KB upload.
//
// Every box handed here is a whole-number of 16-cell grid blocks, so
// every R8 row width is a multiple of 4 and GL's default 4-byte unpack
// alignment cannot bite. The guard below refuses the box rather than
// upload skewed rows if that ever stops being true.
static bool claudeTraceGridUploadBox(int x0, int y0, int z0, int w, int h, int d)
{
	ClaudeTraceGrid &V = g_claude_grid;
	constexpr int S = ClaudeTraceGrid::SIZE;
	if ((w & 3) || (x0 & 3))
		return false;
	static std::vector<u8> staging;
	GLint prev_active_unit = GL.TEXTURE0;
	GL.GetIntegerv(GL.ACTIVE_TEXTURE, &prev_active_unit);
	GLenum fmt = claudeUseR8() ? GL.RED : GL_LUMINANCE;

	claudePackBox(V.occ.data(), S, S, 4, x0, y0, z0, w, h, d, staging);
	GL.ActiveTexture(GL.TEXTURE10);
	GL.BindTexture(GL.TEXTURE_3D, V.tex);
	GL.TexSubImage3D(GL.TEXTURE_3D, 0, x0, y0, z0, w, h, d, GL.RGBA,
			GL.UNSIGNED_BYTE, staging.data());

	claudePackBox(V.modelids.data(), S, S, 1, x0, y0, z0, w, h, d, staging);
	GL.ActiveTexture(GL.TEXTURE0 + 16);
	GL.BindTexture(GL.TEXTURE_3D, V.model_ids_tex);
	GL.TexSubImage3D(GL.TEXTURE_3D, 0, x0, y0, z0, w, h, d, fmt,
			GL.UNSIGNED_BYTE, staging.data());

	claudePackBox(V.mids.data(), S, S, 1, x0, y0, z0, w, h, d, staging);
	GL.ActiveTexture(GL.TEXTURE12);
	GL.BindTexture(GL.TEXTURE_3D, V.material_tex);
	GL.TexSubImage3D(GL.TEXTURE_3D, 0, x0, y0, z0, w, h, d, fmt,
			GL.UNSIGNED_BYTE, staging.data());

	GL.ActiveTexture(GL.TEXTURE11);
	GL.BindTexture(GL.TEXTURE_3D, V.coarse_tex);
	for (int lvl = 0; lvl <= 2; lvl++) {
		int n = S >> lvl;
		int lx = x0 >> lvl, ly = y0 >> lvl, lz = z0 >> lvl;
		int bw = ((x0 + w + (1 << lvl) - 1) >> lvl) - lx;
		int bh = ((y0 + h + (1 << lvl) - 1) >> lvl) - ly;
		int bd = ((z0 + d + (1 << lvl) - 1) >> lvl) - lz;
		claudePackBox(V.pyr[lvl].data(), n, n, 1, lx, ly, lz, bw, bh, bd,
				staging);
		GL.TexSubImage3D(GL.TEXTURE_3D, lvl, lx, ly, lz, bw, bh, bd, fmt,
				GL.UNSIGNED_BYTE, staging.data());
	}
	for (int lvl = 3; lvl <= 5; lvl++) {
		int n = S >> lvl;
		GL.TexSubImage3D(GL.TEXTURE_3D, lvl, 0, 0, 0, n, n, n, fmt,
				GL.UNSIGNED_BYTE, V.pyr[lvl].data());
	}

	// sub-voxel ring [48,80)^3 = grid blocks 3 and 4 on each axis, so
	// an intersection is always a whole number of 16-cell spans and the
	// 2-bytes-per-cell x extent is always a multiple of 4
	{
		const int R0 = ClaudeTraceGrid::NBOX_R0, R1 = ClaudeTraceGrid::NBOX_R1;
		int lx = std::max(x0, R0), hx = std::min(x0 + w, R1);
		int ly = std::max(y0, R0), hy = std::min(y0 + h, R1);
		int lz = std::max(z0, R0), hz = std::min(z0 + d, R1);
		if (lx < hx && ly < hy && lz < hz && !V.subvox.empty()) {
			int tx = (lx - R0) * 2, tw = (hx - lx) * 2;
			int ty = (ly - R0) * 16, th = (hy - ly) * 16;
			int tz = (lz - R0) * 16, td = (hz - lz) * 16;
			claudePackBox(V.subvox.data(), 64, 512, 1, tx, ty, tz,
					tw, th, td, staging);
			GL.ActiveTexture(GL.TEXTURE7);
			GL.BindTexture(GL.TEXTURE_3D, V.subvox_tex);
			GL.TexSubImage3D(GL.TEXTURE_3D, 0, tx, ty, tz, tw, th, td,
					fmt, GL.UNSIGNED_BYTE, staging.data());
		}
	}
	claudeTraceGridUploadModelTex();
	claudeTraceGridUploadAtlas();
	GL.ActiveTexture(prev_active_unit);
	return true;
}

// with the GL context current; the full 2.1 M-cell walk causes a brief
// hitch (MEASURED below), which is acceptable for the two callers that
// still need it: bootstrap and a >24-node re-centre.
static void claudeTraceGridSnapshot(Client *client)
{
	constexpr int S = ClaudeTraceGrid::SIZE;
	ClaudeTraceGrid &V = g_claude_grid;
	u64 t0 = porting::getTimeMs();
	// PHASE TIMERS (gate 5 of the grid re-snap handoff): the cost of a
	// re-snap had been attributed to "the walk" by a comment nobody had
	// measured, and it was wrong by 20x in one direction and 25x in the
	// other. Every phase is timed in us and printed.
	u64 tu0 = porting::getTimeUs();
	v3s16 center = floatToInt(client->getCamera()->getPosition(), BS);
	v3s16 origin = center - v3s16(S / 2, S / 2, S / 2);
	// ORIGIN DEADBAND (John, 2026-08-13: "things shift when I move
	// around"): the detail ring rides grid-local coords, so an
	// origin that re-quantizes with every camera step sweeps the
	// carve/cube boundary through the world at walking pace. Keep the
	// previous origin while the camera stays within +/-6 m of the
	// grid center — walking around a room then shifts NOTHING, and
	// a genuine relocation costs one rebase instead of a pop per step.
	{
		v3s16 prev_center = V.prev_origin + v3s16(S / 2, S / 2, S / 2);
		v3s16 d = center - prev_center;
		if (V.valid && std::abs(d.X) <= 6 && std::abs(d.Y) <= 6
				&& std::abs(d.Z) <= 6)
			origin = V.prev_origin;
	}
	Map &map = client->getEnv().getMap();
	const NodeDefManager *ndef = client->getNodeDefManager();
	claudeLoadModels(ndef);
	V.nodebox_on = claudeNodeBoxEnabled();
	// RGBA per cell: rgb = the node type's average color (same one the
	// minimap uses), alpha = occupancy class: 0 air, 128 water (so later
	// reflection rays can recognize it), 255 solid.
	if (V.occ.size() != (size_t)S * S * S * 4) {
		V.occ.assign((size_t)S * S * S * 4, 0);
		V.mids.assign((size_t)S * S * S, 0);
		V.modelids.assign((size_t)S * S * S, 0);
		V.nbox_ids.assign((size_t)ClaudeTraceGrid::NBOX_RING
				* ClaudeTraceGrid::NBOX_RING
				* ClaudeTraceGrid::NBOX_RING, 0);
		for (int lvl = 0; lvl <= 5; lvl++) {
			int n = S >> lvl;
			V.pyr[lvl].assign((size_t)n * n * n, 0);
		}
	}
	// A full walk owns every cell, so nothing survives it: drop the whole
	// point-emitter list rather than let 512 per-block erases do it.
	V.emit_all.clear();
	for (int b = 0; b < ClaudeTraceGrid::NBLOCKS; b++)
		claudeTraceGridWalkBlock(client, ndef, map, origin, b);
	u32 solid = 0;
	for (int b = 0; b < ClaudeTraceGrid::NBLOCKS; b++)
		solid += V.block_solid[b];
	u64 hash = claudeTraceGridContentHash(origin);
	// Updated unconditionally, BEFORE the unchanged-content early
	// return below: this walk really did just run, whether or not its
	// result differs from last time, so solid_count/snap_seq must
	// reflect it either way -- otherwise the early-return path would
	// leave a stale solid_count sitting there looking like a fresh read.
	V.solid_count = solid;
	V.snap_seq++;
	u64 walk_us = porting::getTimeUs() - tu0;
	// world unchanged since the last snapshot: skip the upload and — key
	// for image stability — do NOT disturb the converged accumulation
	if (V.valid && hash == V.content_hash && origin == V.origin) {
		V.last_snap_ms = porting::getTimeMs();
		actionstream << "[claude_grid] snapshot UNCHANGED (upload"
				" skipped) walk=" << walk_us << "us" << std::endl;
		return;
	}
	V.content_hash = hash;
	u64 tbake = porting::getTimeUs();
	claudeTraceGridBakeSubvox(0, 0, 0, S, S, S);
	claudeTraceGridBakePyramid(0, 0, 0, S, S, S);
	u64 bake_us = porting::getTimeUs() - tbake;
	u64 tgl = porting::getTimeUs();
	claudeTraceGridUploadFull();
	u64 gl_us = porting::getTimeUs() - tgl;
	u64 temit = porting::getTimeUs();
	claudeTraceGridFinishEmitters();
	u64 emit_us = porting::getTimeUs() - temit;

	V.origin = origin;
	V.valid = true;
	V.last_snap_ms = porting::getTimeMs();
	// let the accumulator adapt to new world content within ~1s even
	// when deeply converged (placed torches shouldn't fade in slowly)
	if (V.still_frames > 10.0f)
		V.still_frames = 10.0f;
	actionstream << "[claude_grid] snapshot FULL origin=(" << origin.X << ","
			<< origin.Y << "," << origin.Z << ") solid=" << solid << "/"
			<< (S * S * S) << " area_emitters="
			<< V.area_count << "/" << V.area_total
			// palette caps at 254 and never evicts: past the cap new
			// materials render UNTEXTURED and nothing said so until now
			// (spec/measured.md "Owed to the harness")
			<< " palette=" << V.palette.size() << "/254"
			<< " in " << (porting::getTimeMs() - t0) << " ms"
			<< " [walk=" << walk_us << "us bake=" << bake_us
			<< "us gl=" << gl_us << "us emit=" << emit_us
			<< "us total=" << (porting::getTimeUs() - tu0) << "us]"
			// DIAL STATE NEXT TO THE MEASUREMENT (environment-laws.md:
			// "record the dial state with every capture, not in your
			// head"). nbox_shapes is the converter's whole working set:
			// distinct (content, param2) box lists actually seen.
			<< " nodebox=" << (V.nodebox_on ? 1 : 0)
			<< " nbox_shapes=" << V.nbox_masks.size()
			<< "/" << V.nbox_of.size()
			<< std::endl;
}

// THE INCREMENTAL PATH. Drain the client's dirty-block set and fold just
// those blocks into the live grid. Returns true if the grid changed.
//
// Everything here is gated on a per-block hash actually differing, not
// on a packet having arrived: BLOCKDATA lands for reasons that do not
// change a single node (a re-send, a block leaving and re-entering
// range), and resetting still_frames on those would restart convergence
// in a static scene — the exact opposite of the fix.
static bool claudeTraceGridIncremental(Client *client)
{
	constexpr int S = ClaudeTraceGrid::SIZE;
	constexpr int B = ClaudeTraceGrid::BLK;
	constexpr int NB = ClaudeTraceGrid::NB;
	ClaudeTraceGrid &V = g_claude_grid;
	std::vector<v3s16> dirty;
	bool overflow = client->takeClaudeDirtyBlocks(dirty);
	if (!V.valid)
		return false;
	if (overflow) {
		// A mass edit (a mod filling a box, a world deploy) blew the
		// dirty-set cap, so the list is not the whole truth and folding
		// it in would leave a grid that disagrees with the world in
		// places nothing would ever look at again.
		warningstream << "[claude_grid] dirty-block set overflowed;"
				" falling back to a full walk" << std::endl;
		claudeTraceGridSnapshot(client);
		return true;
	}
	if (dirty.empty())
		return false;
	u64 tu0 = porting::getTimeUs();
	// map block (16^3 world nodes, world-aligned) -> grid cell box ->
	// the grid blocks it touches. The grid origin is arbitrary, so
	// one map block straddles up to 2 grid blocks per axis; re-walking
	// whole grid blocks costs at most 8 x 4096 cells and keeps one
	// alignment story for the hashes, the pyramid and the uploads.
	static std::vector<u8> mark;
	mark.assign(ClaudeTraceGrid::NBLOCKS, 0);
	int marked = 0;
	for (const v3s16 &bp : dirty) {
		v3s16 lo = bp * MAP_BLOCKSIZE - V.origin;
		int x0 = std::max(0, (int)lo.X), x1 = std::min(S, (int)lo.X + MAP_BLOCKSIZE);
		int y0 = std::max(0, (int)lo.Y), y1 = std::min(S, (int)lo.Y + MAP_BLOCKSIZE);
		int z0 = std::max(0, (int)lo.Z), z1 = std::min(S, (int)lo.Z + MAP_BLOCKSIZE);
		if (x0 >= x1 || y0 >= y1 || z0 >= z1)
			continue; // outside the bubble entirely
		for (int bz = z0 / B; bz <= (z1 - 1) / B; bz++)
		for (int by = y0 / B; by <= (y1 - 1) / B; by++)
		for (int bx = x0 / B; bx <= (x1 - 1) / B; bx++) {
			int b = (bz * NB + by) * NB + bx;
			if (!mark[b]) { mark[b] = 1; marked++; }
		}
	}
	if (!marked)
		return false;
	Map &map = client->getEnv().getMap();
	const NodeDefManager *ndef = client->getNodeDefManager();
	claudeLoadModels(ndef);
	V.nodebox_on = claudeNodeBoxEnabled();
	int cx0 = S, cy0 = S, cz0 = S, cx1 = 0, cy1 = 0, cz1 = 0;
	int changed = 0;
	for (int b = 0; b < ClaudeTraceGrid::NBLOCKS; b++) {
		if (!mark[b])
			continue;
		u64 before = V.block_hash[b];
		claudeTraceGridWalkBlock(client, ndef, map, V.origin, b);
		if (V.block_hash[b] == before)
			continue;
		changed++;
		int bx = b % NB, by = (b / NB) % NB, bz = b / (NB * NB);
		cx0 = std::min(cx0, bx * B); cx1 = std::max(cx1, (bx + 1) * B);
		cy0 = std::min(cy0, by * B); cy1 = std::max(cy1, (by + 1) * B);
		cz0 = std::min(cz0, bz * B); cz1 = std::max(cz1, (bz + 1) * B);
	}
	u64 walk_us = porting::getTimeUs() - tu0;
	V.snap_seq++;
	u32 solid = 0;
	for (int b = 0; b < ClaudeTraceGrid::NBLOCKS; b++)
		solid += V.block_solid[b];
	V.solid_count = solid;
	if (!changed) {
		// A packet arrived and changed nothing. Say so and leave the
		// accumulator alone; this is the case the old timer could not
		// distinguish from a real edit.
		V.last_snap_ms = porting::getTimeMs();
		infostream << "[claude_grid] incremental " << marked
				<< " blocks, none changed, walk=" << walk_us << "us"
				<< std::endl;
		return false;
	}
	u64 hash = claudeTraceGridContentHash(V.origin);
	V.content_hash = hash;
	u64 tbake = porting::getTimeUs();
	claudeTraceGridBakeSubvox(cx0, cy0, cz0, cx1 - cx0, cy1 - cy0, cz1 - cz0);
	claudeTraceGridBakePyramid(cx0, cy0, cz0, cx1 - cx0, cy1 - cy0, cz1 - cz0);
	u64 bake_us = porting::getTimeUs() - tbake;
	u64 tgl = porting::getTimeUs();
	bool boxed = claudeTraceGridUploadBox(cx0, cy0, cz0,
			cx1 - cx0, cy1 - cy0, cz1 - cz0);
	if (!boxed)
		claudeTraceGridUploadFull();
	u64 gl_us = porting::getTimeUs() - tgl;
	u64 temit = porting::getTimeUs();
	claudeTraceGridFinishEmitters();
	u64 emit_us = porting::getTimeUs() - temit;
	V.last_snap_ms = porting::getTimeMs();
	// let the accumulator adapt to new world content within ~1s even
	// when deeply converged (placed torches shouldn't fade in slowly)
	if (V.still_frames > 10.0f)
		V.still_frames = 10.0f;
	actionstream << "[claude_grid] incremental " << changed << "/"
			<< marked << " blocks box=(" << cx0 << "," << cy0 << ","
			<< cz0 << ")+(" << (cx1 - cx0) << "," << (cy1 - cy0) << ","
			<< (cz1 - cz0) << ") solid=" << solid
			<< " area_emitters=" << V.area_count << "/" << V.area_total
			<< " [walk=" << walk_us << "us bake=" << bake_us
			<< "us gl=" << gl_us << "us emit=" << emit_us
			<< "us total=" << (porting::getTimeUs() - tu0) << "us]"
			<< (boxed ? "" : " (full upload fallback)") << std::endl;
	return true;
}

// Once per frame: decide how strongly this frame's traced sample should
// overwrite the accumulation history. Teleports and grid swaps trash
// the history entirely; gentle drift blends fast; stillness accumulates
// deep (the converged, soft-lit image).
static void claudeUpdateAccum(Client *client)
{
	Camera *cam = client->getCamera();
	v3f p = cam->getPosition();
	v3f d = cam->getDirection();
	float moved = p.getDistanceFrom(g_claude_grid.prev_cam_pos);
	float turned = (d - g_claude_grid.prev_cam_dir).getLength();
	bool origin_changed = g_claude_grid.origin != g_claude_grid.prev_origin;
	v3s16 odelta = g_claude_grid.origin - g_claude_grid.prev_origin;
	g_claude_grid.origin_delta = origin_changed
			? v3f(odelta.X, odelta.Y, odelta.Z) : v3f(0.0f, 0.0f, 0.0f);
	g_claude_grid.prev_cam_pos = p;
	g_claude_grid.prev_cam_dir = d;
	g_claude_grid.prev_origin = g_claude_grid.origin;
	// near-ring sub-face atlas follows the camera, corner clamped so the
	// 32^3 ring never leaves the grid; prev kept one frame for remap.
	// Across a grid rebase, express last frame's corner in the NEW
	// grid space so the remap keeps pointing at the same world cells.
	g_claude_grid.near_prev = origin_changed
			? g_claude_grid.near_origin - g_claude_grid.origin_delta
			: g_claude_grid.near_origin;
	v3f lp = p / BS - v3f(g_claude_grid.origin.X,
			g_claude_grid.origin.Y, g_claude_grid.origin.Z);
	g_claude_grid.near_origin = v3f(
			core::clamp(std::floor(lp.X) - 16.0f, 0.0f, 96.0f),
			core::clamp(std::floor(lp.Y) - 16.0f, 0.0f, 96.0f),
			core::clamp(std::floor(lp.Z) - 16.0f, 0.0f, 96.0f));
	// With reprojection the shader revalidates history per pixel; the CPU
	// only forces a full reset when the coordinate space itself changes.
	// Teardown-style shallow history in motion (spatial denoise carries
	// the smoothing); when still, a TRUE running average (weight 1/N)
	// so the image converges to actual stillness instead of the EMA's
	// perpetual 5%-new-sample pulse.
	if (origin_changed || moved > 20.0f) {
		g_claude_grid.accum_alpha = 1.0f;
		g_claude_grid.still_frames = 0.0f;
		g_claude_grid.accum_resets++;
	} else if (moved > 0.05f || turned > 1e-4f) {
		g_claude_grid.accum_alpha = 0.5f;
		g_claude_grid.still_frames = 0.0f;
		g_claude_grid.accum_resets++;
	} else {
		g_claude_grid.still_frames += 1.0f;
		// True 1/N running average, NO floor. The old renderer floored
		// this at 0.02, which silently turns the average into an EMA
		// whose noise never drops below ~10% of per-sample sigma — the
		// "never stops bubbling" defect. A parked camera must actually
		// converge; motion and teleports reset above.
		g_claude_grid.accum_alpha =
				1.0f / (2.0f + g_claude_grid.still_frames);
	}

	// Held light: if the wielded item is a light-emitting node, place an
	// emitter at the camera every frame. Real-time by construction — no
	// server round trip, no snapshot lag, no light node in the world.
	g_claude_grid.emitter_runtime = g_claude_grid.emitter_count;
	g_claude_grid.held_emitter[3] = 0.0f; // cleared unless wielding a light
	if (g_claude_grid.valid) {
		LocalPlayer *lp = client->getEnv().getLocalPlayer();
		const NodeDefManager *ndef = client->getNodeDefManager();
		if (lp && ndef) {
			ItemStack sel, hand;
			ItemStack wield = lp->getWieldedItem(&sel, &hand);
			content_t cid = CONTENT_IGNORE;
			if (!wield.name.empty() && ndef->getId(wield.name, cid)
					&& cid != CONTENT_IGNORE) {
				u8 ls = ndef->get(cid).light_source;
				if (ls > 0) {
					v3f lpos = p / BS - v3f(g_claude_grid.origin.X,
							g_claude_grid.origin.Y, g_claude_grid.origin.Z);
					g_claude_grid.held_emitter[0] = lpos.X;
					g_claude_grid.held_emitter[1] = lpos.Y + 0.2f;
					g_claude_grid.held_emitter[2] = lpos.Z;
					g_claude_grid.held_emitter[3] =
							std::min<int>(ls, 14) / 14.0f;
				}
			}
		}
	}

	// stage the ray-camera basis: what was current becomes the shader's
	// previous frame
	g_claude_grid.shader_prev_pos = g_claude_grid.cur_pos;
	g_claude_grid.shader_prev_fwd = g_claude_grid.cur_fwd;
	g_claude_grid.shader_prev_rightu = g_claude_grid.cur_rightu;
	g_claude_grid.shader_prev_upu = g_claude_grid.cur_upu;
	g_claude_grid.shader_prev_tanx = g_claude_grid.cur_tanx;
	g_claude_grid.shader_prev_tany = g_claude_grid.cur_tany;

	v3f fwd = d;
	fwd.normalize();
	v3f right = v3f(0.f, 1.f, 0.f).crossProduct(fwd);
	right.normalize();
	v3f up = fwd.crossProduct(right);
	g_claude_grid.cur_pos = p / BS
			- v3f(g_claude_grid.origin.X, g_claude_grid.origin.Y,
					g_claude_grid.origin.Z);
	g_claude_grid.cur_fwd = fwd;
	g_claude_grid.cur_rightu = right;
	g_claude_grid.cur_upu = up;
	g_claude_grid.cur_tanx = std::tan(cam->getFovX() * 0.5f);
	g_claude_grid.cur_tany = std::tan(cam->getFovY() * 0.5f);
}

// claude_stats: when enabled, write rolling frame statistics to
// <path_user>/claude_stats.json once per second so external tooling can
// measure performance without reading the debug overlay off a screenshot.
static void claudeWriteStats(f32 dtime, f32 busy_us, f32 draw_us)
{
	if (!g_settings->exists("claude_stats")
			|| g_settings->getFloat("claude_stats", 0.0f, 1.0f) < 0.5f)
		return;
	static f32 window = 0.0f;
	static u32 frames = 0;
	static f32 worst = 0.0f;
	static f32 best = 1e9f;
	static f32 total = 0.0f;
	static f32 busy_total = 0.0f;
	static f32 draw_total = 0.0f;
	window += dtime;
	frames++;
	total += dtime;
	busy_total += busy_us;
	draw_total += draw_us;
	worst = std::max(worst, dtime);
	best = std::min(best, dtime);
	if (window < 1.0f)
		return;
	std::ostringstream os;
	os << "{\"fps\": " << (frames / std::max(window, 1e-3f))
			<< ", \"frame_ms_avg\": " << (total / frames * 1000.0f)
			<< ", \"frame_ms_worst\": " << (worst * 1000.0f)
			<< ", \"frame_ms_best\": " << (best * 1000.0f)
			<< ", \"frames\": " << frames
			<< ", \"grid_valid\": " << (g_claude_grid.valid ? 1 : 0)
			// grid_valid is a tautology after the first snapshot ever
			// taken on a seat (set true once, never cleared) -- these
			// two are the actual "is there something here" proof.
			// solid_count: non-air cells the LAST WALK found (roadmap 1b).
			<< ", \"grid_solid\": " << g_claude_grid.solid_count
			<< ", \"grid_snap_seq\": " << g_claude_grid.snap_seq
			// The content hash itself, so a harness can prove the
			// INCREMENTAL path converges to the same state a full walk
			// would: edit a node, put it back, and this must return to
			// the value it had before. It is the per-block hashes
			// combined order-independently, so it is stable across the
			// two paths by construction rather than by luck.
			<< ", \"grid_hash\": \"" << std::hex << std::setw(16)
			<< std::setfill('0') << g_claude_grid.content_hash
			<< std::dec << std::setfill(' ') << "\""
			<< ", \"emitters\": " << g_claude_grid.emitter_count
			// the two numbers that explain a noisy room: how many area
			// emitters NEE can aim at, and how many exist
			<< ", \"area_emitters\": " << g_claude_grid.area_count
			<< ", \"area_total\": " << g_claude_grid.area_total
			<< ", \"accum_alpha\": " << g_claude_grid.accum_alpha
			<< ", \"light_body\": " << g_claude_grid.light_body
			<< ", \"light_y\": " << g_claude_grid.prev_light_dir.Y
			<< ", \"light_lum\": " << (g_claude_grid.prev_light_col.X
					+ g_claude_grid.prev_light_col.Y
					+ g_claude_grid.prev_light_col.Z) / 3.0f
			<< ", \"still_frames\": " << g_claude_grid.still_frames
			// zeroings, not clamps -- see ClaudeTraceGrid::accum_resets
			<< ", \"accum_resets\": " << g_claude_grid.accum_resets
			<< ", \"casc_valid\": [" << (g_claude_grid.casc[0].valid ? 1 : 0)
			<< "," << (g_claude_grid.casc[1].valid ? 1 : 0)
			<< "," << (g_claude_grid.casc[2].valid ? 1 : 0)
			<< "," << (g_claude_grid.casc[3].valid ? 1 : 0)
			<< "," << (g_claude_grid.casc[4].valid ? 1 : 0)
			<< "], \"casc_solid\": [" << g_claude_grid.casc[0].solid
			<< "," << g_claude_grid.casc[1].solid
			<< "," << g_claude_grid.casc[2].solid
			<< "," << g_claude_grid.casc[3].solid
			<< "," << g_claude_grid.casc[4].solid
			<< "], \"casc_ms\": [" << g_claude_grid.casc[0].ms
			<< "," << g_claude_grid.casc[1].ms
			<< "," << g_claude_grid.casc[2].ms
			<< "," << g_claude_grid.casc[3].ms
			<< "," << g_claude_grid.casc[4].ms
			<< "], \"summary_blocks\": " << claude_lod::summaryCount();
	os << ", \"draw_ms\": " << (draw_total / frames / 1000.0f)
			<< ", \"busy_ms\": " << (busy_total / frames / 1000.0f);
	os << ", \"pass_ms\": [";
	for (int i = 0; i < g_claude_gpuprof.n; i++)
		os << (i ? "," : "") << g_claude_gpuprof.ms[i];
	os << "]}\n";
	std::ofstream f(porting::path_user + "/claude_stats.json",
			std::ios::trunc);
	f << os.str();
	window = 0.0f; frames = 0; worst = 0.0f; best = 1e9f; total = 0.0f;
	busy_total = 0.0f; draw_total = 0.0f;
}

// Exposes trace accumulation state to the F5 debug overlay (gameui.cpp),
// which lives outside this translation unit and so can't reach the
// file-local g_claude_grid directly.
void claudeGetTraceStats(float *still_frames, float *accum_alpha)
{
	*still_frames = g_claude_grid.still_frames;
	*accum_alpha = g_claude_grid.accum_alpha;
}

// claude_lod Phase 2: (re)build and upload cascade levels when stale or
// strayed. Runs from the 1 Hz settings poll and builds AT MOST ONE level
// per invocation, so the worst frame eats one build (2 m is the big one,
// tens of ms walking 16M nodes) per second — never more.
static void claudeCascadeUpdate(Client *client)
{
	// PURE 1 m MODE (claude_cascades = 0): the LOD ladder is not merely
	// frozen, it is RETIRED — every level's validity is cleared so the
	// shader's farTrace finds no valid rung and eye rays end at the 128^3
	// grid edge (sky beyond). Without the invalidation, flipping the
	// dial off at runtime only stopped REBUILDS and the stale ladder kept
	// being marched, which would read as "the dial does nothing".
	if (!g_settings->exists("claude_cascades")
			|| g_settings->getFloat("claude_cascades", 0.0f, 1.0f) < 0.5f) {
		for (int lv = 0; lv < 5; lv++)
			g_claude_grid.casc[lv].valid = false;
		return;
	}
	static const int CELL[5] = {2, 4, 8, 16, 32};
	static const u64 CADENCE[5] = {8000, 6000, 4000, 12000, 20000};
	v3s16 center = floatToInt(client->getCamera()->getPosition(), BS);
	// far-data feed: sweep <path_user>/claude_far/ for new server-sampled
	// terrain every ~5 s (each file ingested once; version bump triggers
	// the normal cascade rebuild below)
	static u64 far_last = 0;
	u64 now_ms = porting::getTimeMs();
	if (now_ms - far_last > 5000) {
		far_last = now_ms;
		size_t n = claude_lod::ingestFarDir(client);
		if (n)
			infostream << "claude_far: ingested " << n << " blocks"
					<< std::endl;
	}
	u64 ver = claude_lod::contentVersion();

	for (int lv = 0; lv < 5; lv++) {
		auto &L = g_claude_grid.casc[lv];
		const int half = 128 * CELL[lv] / 2;
		const int stray = 16 * CELL[lv]; // 32 / 64 / 128 nodes
		bool need = !L.valid;
		if (!need) {
			v3s16 c0 = L.origin + v3s16(half, half, half);
			v3s16 d = center - c0;
			if (std::abs(d.X) > stray || std::abs(d.Y) > stray
					|| std::abs(d.Z) > stray)
				need = true;
			else if (ver != L.version
					&& porting::getTimeMs() - L.build_time > CADENCE[lv])
				need = true;
		}
		if (!need)
			continue;
		u64 t0 = porting::getTimeMs();
		v3s16 origin = center - v3s16(half, half, half);
		// snap to one coarse brick (4 cells) so brick boundaries and
		// world-space cell identity are stable across recenters — AND to
		// whole MapBlocks (16): buildCascade2 tiles blocks from the
		// origin, so an 8-aligned-but-not-16-aligned origin displaced
		// the ENTIRE 2m level by 8 nodes on half of all rebuilds. That
		// parity roulette was John's "fucked up wall" + comb teeth: a
		// seam mismatch can never exceed one coarse cell (his invariant)
		// unless a level is bodily shifted. Caught 2026-08-12 ~09:50.
		const s16 snapv = (s16)std::max(CELL[lv] * 4, 16);
		const s16 snap_mask = (s16)~(snapv - 1);
		origin.X &= snap_mask; origin.Y &= snap_mask; origin.Z &= snap_mask;
		static std::vector<u8> rgba, coarse;
		// ONE fold for every rung (John's consolidation, 2026-08-12):
		// the 2m level now builds from summary fine-bits through the
		// same scatter as every other level — one alignment contract,
		// the origin-parity bug class is unrepresentable.
		u32 solid = claude_lod::buildCascadeSummary(origin, CELL[lv],
				rgba, coarse);
		if (solid == 0 && !L.valid)
			return; // no data yet; retry next poll (and skip coarser too)

		GLint prev_active_unit = GL.TEXTURE0;
		GL.GetIntegerv(GL.ACTIVE_TEXTURE, &prev_active_unit);
		bool fresh_alloc = !g_claude_grid.cascades_tex;
		if (fresh_alloc) {
			GL.GenTextures(1, &g_claude_grid.cascades_tex);
			GL.GenTextures(1, &g_claude_grid.cascades_coarse_tex);
		}
		GL.ActiveTexture(GL.TEXTURE8);
		GL.BindTexture(GL.TEXTURE_3D, g_claude_grid.cascades_tex);
		if (fresh_alloc) {
			GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
			GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
			GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_WRAP_S, GL.CLAMP_TO_EDGE);
			GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_WRAP_T, GL.CLAMP_TO_EDGE);
			GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_WRAP_R, GL.CLAMP_TO_EDGE);
			GL.TexImage3D(GL.TEXTURE_3D, 0, GL.RGBA8, 128, 128, 640, 0,
					GL.RGBA, GL.UNSIGNED_BYTE, nullptr);
		}
		GL.TexSubImage3D(GL.TEXTURE_3D, 0, 0, 0, lv * 128, 128, 128, 128,
				GL.RGBA, GL.UNSIGNED_BYTE, rgba.data());
		GL.ActiveTexture(GL.TEXTURE9);
		GL.BindTexture(GL.TEXTURE_3D, g_claude_grid.cascades_coarse_tex);
		if (fresh_alloc) {
			GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
			GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
			GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_WRAP_S, GL.CLAMP_TO_EDGE);
			GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_WRAP_T, GL.CLAMP_TO_EDGE);
			GL.TexParameteri(GL.TEXTURE_3D, GL.TEXTURE_WRAP_R, GL.CLAMP_TO_EDGE);
			GL.TexImage3D(GL.TEXTURE_3D, 0,
					claudeUseR8() ? GL.R8 : GL_LUMINANCE8, 32, 32, 160, 0,
					claudeUseR8() ? GL.RED : GL_LUMINANCE,
					GL.UNSIGNED_BYTE, nullptr);
		}
		GL.TexSubImage3D(GL.TEXTURE_3D, 0, 0, 0, lv * 32, 32, 32, 32,
				claudeUseR8() ? GL.RED : GL_LUMINANCE,
				GL.UNSIGNED_BYTE, coarse.data());
		GL.ActiveTexture(prev_active_unit);

		L.origin = origin;
		L.valid = true;
		L.version = ver;
		L.build_time = porting::getTimeMs();
		L.solid = solid;
		L.ms = (float)(porting::getTimeMs() - t0);
		break; // one level per poll: bounded hitch
	}
}

// Shared parse/diff/apply for a "key = value" patch file: no-op if the
// file is unreadable, empty, or byte-identical to the caller's cache
// (last_applied — one per source file, so the settings patch and the
// /dial console channel never re-trigger each other off a shared
// timer). Otherwise parses it as a Settings block and applies every
// line, honoring the same screenshot/re-snapshot pseudo-keys either
// source file may use.
// Returns true when this call actually applied a NEW patch, so the
// caller can do the once-per-change work (see the GameUI flags below)
// without fighting a keypress on every 1 Hz tick.
static bool claudeApplyPatchFile(const std::string &path,
		std::string &last_applied, Client *client)
{
	std::ifstream f(path);
	if (!f.good())
		return false;
	std::string content((std::istreambuf_iterator<char>(f)),
			std::istreambuf_iterator<char>());
	if (content.empty() || content == last_applied)
		return false;
	last_applied = content;
	Settings patch;
	std::istringstream is(content);
	if (!patch.parseConfigLines(is))
		return false;
	// A patch file is the channel that once shadowed the conf in silence;
	// an old claude_volume_* name arriving here must not be a no-op.
	claudeWarnRenamedSettings(&patch, path.c_str());
	for (const std::string &name : patch.getNames()) {
		// Pseudo-key: any value change triggers a screenshot (same call as
		// the F12 keybind), saved to the usual screenshots directory.
		if (name == "claude_screenshot") {
			client->makeScreenshot();
			actionstream << "[claude_settings_patch] screenshot taken"
					<< std::endl;
			continue;
		}
		// Pseudo-key: any value change re-snapshots the grid around the
		// current camera position.
		if (name == "claude_grid_snapshot") {
			claudeTraceGridSnapshot(client);
			continue;
		}
		g_settings->set(name, patch.get(name));
		actionstream << "[claude_settings_patch] " << name << " = "
				<< patch.get(name) << std::endl;
	}
	return true;
}

// claude_input_lock: a measurement seat must not be drivable by hand.
// Turning the camera does NOT reset the accumulator, so one stray
// mouse-look inside a 60 s settle blends two views into a frame that
// looks converged and carries a perfectly clean dial state — it
// happened on 2026-08-15 and nothing in the capture record could have
// shown it. With this set, mouse/keyboard look and movement are
// ignored; the harness still drives the player through the server
// (teleports arrive as player_force_move and are applied), and F1/F2
// and the menu still work, so a human can watch without steering.
static bool claudeInputLocked()
{
	return g_settings->exists("claude_input_lock")
			&& g_settings->getFloat("claude_input_lock", 0.0f, 1.0f) > 0.5f;
}

static void pollSettingsPatch(f32 dtime, Client *client, GameUI *game_ui)
{
	static f32 timer = 0.0f;
	static std::string last_applied;
	static std::string last_applied_dial;
	timer += dtime;
	if (timer < 1.0f)
		return;
	timer = 0.0f;

	// claude_grid follow (Phase 0-lite streaming): whenever any grid
	// consumer (ghost view or water reflections) is enabled, keep a grid
	// alive around the camera — bootstrap one if none exists (e.g. right
	// after a restart), re-snapshot when the camera strays >24 nodes from
	// the current center, and fold in whatever blocks the server told us
	// changed. claude_grid_follow = 0 restores the frozen-bubble
	// behavior (the CI seat used to need that; it does not any more).
	//
	// WAS A KNOWN DEFECT, now measured and fixed (spec/measured.md
	// "Volume re-snap fix"). This block used to re-snapshot the whole
	// 128^3 grid on a "every 2 s" timer that the 1 Hz poll quantized to
	// every 3 s. MEASURED at cozy-ci, Release, before: 12.0 ms of CPU
	// walk on EVERY tick (the unchanged-hash early return skips the
	// upload, never the walk), 28.6 ms when content had changed, worst
	// frame 39.7 ms against a 15.8 ms follow-off baseline. An earlier
	// comment here claimed ~15 ms and a later one ~295 ms; both were
	// written without a timer in the function, which is why there is one
	// now and why it prints on every exit.
	//
	// AFTER: no timer at all. Snaps are event-driven (Client's
	// dirty-block set, fed from addNode / removeNode / BLOCKDATA) and
	// incremental (re-walk the changed 16^3 grid blocks, upload that
	// sub-box). Costs are in the [claude_grid] log lines and in
	// measured.md; the full walk survives only for the two cases where
	// nothing can be reused.
	{
		auto setting_on = [](const char *name) {
			return g_settings->exists(name)
					&& g_settings->getFloat(name, 0.0f, 2.0f) > 0.0f;
		};
		bool follow = !g_settings->exists("claude_grid_follow")
				|| g_settings->getFloat("claude_grid_follow", 0.0f, 1.0f) > 0.0f;
		bool consumer_on = setting_on("claude_grid_debug")
				|| setting_on("claude_water_reflections")
				|| setting_on("claude_gi")
				|| setting_on("claude_clay");
		if (follow && !g_claude_grid.valid && consumer_on) {
			claudeTraceGridSnapshot(client);
		} else if (follow && g_claude_grid.valid) {
			constexpr s16 H = ClaudeTraceGrid::SIZE / 2;
			v3s16 center = g_claude_grid.origin + v3s16(H, H, H);
			v3s16 d = floatToInt(client->getCamera()->getPosition(), BS) - center;
			if (std::abs(d.X) > 24 || std::abs(d.Y) > 24
					|| std::abs(d.Z) > 24) {
				// re-centre: the origin moves, so no cell of the old
				// grid is at the address it used to be. Full walk.
				claudeTraceGridSnapshot(client);
			} else if (consumer_on) {
				claudeTraceGridIncremental(client);
			}
		}
		if (!follow || !consumer_on) {
			// Nothing is going to consume the dirty list; drain it so it
			// cannot grow without bound while the bubble is frozen. The
			// overflow flag is cleared with it, and the next real
			// snapshot is a full walk anyway.
			std::vector<v3s16> discard;
			client->takeClaudeDirtyBlocks(discard);
		}
		if (consumer_on)
			claudeCascadeUpdate(client);
	}

	bool applied = claudeApplyPatchFile(
			porting::path_user + "/claude_settings_patch.conf",
			last_applied, client);

	// claude_dial_file: a SECOND, optional patch file, named by this
	// client setting, polled on the same 1 Hz tick through the identical
	// apply path above. It exists so the server-side /dial chatcommand
	// (mods/claude_bridge) — which can only read/write inside the running
	// world's directory, not path_user — has a channel to reach the
	// client: /dial writes claude_dial.conf into the world dir, and the
	// dial file setting just needs to point there. Empty path = off.
	std::string dial_file = g_settings->exists("claude_dial_file")
			? g_settings->get("claude_dial_file") : "";
	if (!dial_file.empty()) {
		std::string dial_path = fs::IsPathAbsolute(dial_file)
				? dial_file
				: porting::path_user + DIR_DELIM + dial_file;
		applied |= claudeApplyPatchFile(dial_path, last_applied_dial,
				client);
	}

	// claude_show_hud / claude_show_chat: the HUD and the chat backlog
	// are the only two client dials that lived exclusively on the F1/F2
	// keybinds, which a headless capture seat cannot press. They are
	// PIXELS inside measured crops (the Cornell floor box is drawn over
	// by the hotbar), so a referee frame taken with them on is a
	// contaminated referee. Applied only when a patch actually changed,
	// so a human's F1 still works between writes.
	if (applied && game_ui)
		game_ui->applyClaudeFlagSettings();
}

void Game::run()
{
	ZoneScoped;

	ProfilerGraph graph;
	RunStats stats = {};
	CameraOrientation cam_view_target = {};
	CameraOrientation cam_view = {};
	FpsControl draw_times;
	f32 dtime; // in seconds

	// Clear the profiler
	{
		Profiler::GraphValues dummyvalues;
		g_profiler->graphPop(dummyvalues);
	}

	draw_times.reset();

	set_light_curve(g_settings->getFloat("display_gamma"));

	m_touch_simulate_aux1 = g_settings->getBool("fast_move")
			&& client->checkPrivilege("fast");

	const core::dimension2du initial_screen_size(
			g_settings->getU16("screen_w"),
			g_settings->getU16("screen_h")
		);
	const bool initial_window_maximized = !g_settings->getBool("fullscreen") &&
			g_settings->getBool("window_maximized");

#ifdef __ANDROID__
	porting::setPlayingNowNotification(true);
#endif

	auto framemarker = FrameMarker("Game::run()-frame").started();

	while (m_rendering_engine->run()
			&& !(*kill || g_gamecallback->shutdown_requested
			|| (server && server->isShutdownRequested()))) {

		framemarker.end();

		// Calculate dtime =
		//    m_rendering_engine->run() from this iteration
		//  + Sleep time until the wanted FPS are reached
		draw_times.limit(device, &dtime);

		framemarker.start();

		g_fontengine->handleReload();

		pollSettingsPatch(dtime, client, m_game_ui.get());
		claudeUpdateAccum(client);
		claudeWriteStats(dtime, draw_times.busy_time, stats.drawtime);

		const auto current_dynamic_info = ClientDynamicInfo::getCurrent();
		if (!current_dynamic_info.equal(client_display_info)) {
			client_display_info = current_dynamic_info;
			dynamic_info_send_timer = 0.2f;
		}

		if (dynamic_info_send_timer > 0.0f) {
			dynamic_info_send_timer -= dtime;
			if (dynamic_info_send_timer <= 0.0f) {
				client->sendUpdateClientInfo(current_dynamic_info);
			}
		}

		// Prepare render data for next iteration

		updateStats(&stats, draw_times, dtime);
		updateInteractTimers(dtime);

		if (!checkConnection())
			break;
		if (!m_game_formspec.handleCallbacks())
			break;

		processQueues();

		m_game_ui->clearInfoText();

		updateProfilers(stats, draw_times, dtime);

		// Update camera offset once before doing anything.
		// In contrast to other updates the latency of this doesn't matter,
		// since it's invisible to the user. But it needs to be consistent.
		updateCameraOffset();

		processUserInput(dtime);
		// Update camera before player movement to avoid camera lag of one frame
		updateCameraDirection(&cam_view_target, dtime);
		if (m_cache_cam_smoothing <= 0.0f) {
			cam_view.camera_yaw = cam_view_target.camera_yaw;
			cam_view.camera_pitch = cam_view_target.camera_pitch;
		} else {
			f32 cam_damp_lambda = 1.0f / m_cache_cam_smoothing * dtime;
			cam_view.camera_yaw = damp(
					cam_view.camera_yaw,
					cam_view_target.camera_yaw,
					cam_damp_lambda
			);
			cam_view.camera_pitch = damp(
					cam_view.camera_pitch,
					cam_view_target.camera_pitch,
					cam_damp_lambda
			);
		}
		updatePlayerControl(cam_view);

		updatePauseState();
		if (m_is_paused)
			dtime = 0.0f;

		step(dtime);

		processClientEvents(&cam_view_target);
		updateDebugState();
		// Update camera here so it is in-sync with CAO position
		updateCamera(dtime);
		updateSound(dtime);
		processPlayerInteraction(dtime, m_game_ui->m_flags.show_hud);
		updateFrame(&graph, &stats, dtime, cam_view);
		updateProfilerGraphs(&graph);

		if (m_does_lost_focus_pause_game && !device->isWindowFocused() && !isMenuActive()) {
			m_game_formspec.showPauseMenu();
		}
	}

	framemarker.end();

#ifdef __ANDROID__
	porting::setPlayingNowNotification(false);
#endif

	RenderingEngine::autosaveScreensizeAndCo(initial_screen_size, initial_window_maximized);
}


void Game::shutdown()
{
	// Delete text and menus first
	m_game_ui->clearText();
	m_game_formspec.reset();
	while (g_menumgr.menuCount() > 0) {
		g_menumgr.deleteFront();
	}

	if (g_touchcontrols)
		g_touchcontrols->hide();

	// Restore normal mouse cursor
	auto *cur_control = device->getCursorControl();
	if (cur_control) {
		cur_control->setVisible(true);
		cur_control->setRelativeMode(false);
	}

	clouds.reset();

	gui_chat_console.reset();

	sky.reset();

	// only if the shutdown progress bar isn't shown yet
	if (m_shutdown_progress == 0.0f)
		showOverlayMessage(N_("Shutting down..."), 0, 0);

	chat_backend->addMessage(L"", L"# Disconnected.");
	chat_backend->addMessage(L"", L"");

	if (client) {
		client->Stop();
		while (!client->isShutdown()) {
			assert(texture_src != NULL);
			assert(shader_src != NULL);
			texture_src->processQueue();
			shader_src->processQueue();
			sleep_ms(100);
		}
	}

	delete client;
	client = nullptr;
	soundmaker.reset();
	sound_manager.reset();

	auto stop_thread = runInThread([=] {
		delete server;
		server = nullptr;
	}, "ServerStop");

	FpsControl fps_control;
	fps_control.reset();

	while (stop_thread->isRunning()) {
		m_rendering_engine->run();
		f32 dtime;
		fps_control.limit(device, &dtime);
		showOverlayMessage(N_("Shutting down..."), dtime, 0, &m_shutdown_progress);
	}

	stop_thread->rethrow();

	// to be continued in Game::~Game
}


/****************************************************************************/
/****************************************************************************
 Startup
 ****************************************************************************/
/****************************************************************************/

bool Game::init(
		const std::string &map_dir,
		const std::string &address,
		u16 port,
		const SubgameSpec &gamespec)
{
	texture_src = createTextureSource();

	showOverlayMessage(N_("Loading..."), 0, 0);

	shader_src = createShaderSource();

	itemdef_manager = createItemDefManager();
	nodedef_manager = createNodeDefManager();

	m_item_visuals_manager = std::make_unique<ItemVisualsManager>();

	eventmgr = new EventManager();
	quicktune = new QuicktuneShortcutter();

	if (!(texture_src && shader_src && itemdef_manager && nodedef_manager
			&& eventmgr && quicktune))
		return false;

	if (!initSound())
		return false;

	// Create a server if not connecting to an existing one
	if (address.empty()) {
		if (!createServer(map_dir, gamespec, port))
			return false;
	}

	return true;
}

bool Game::initSound()
{
#if USE_SOUND
	if (g_sound_manager_singleton.get()) {
		infostream << "Attempting to use OpenAL audio" << std::endl;
		sound_manager = createOpenALSoundManager(g_sound_manager_singleton.get(),
				std::make_unique<SoundFallbackPathProvider>());
		if (!sound_manager)
			infostream << "Failed to initialize OpenAL audio" << std::endl;
	} else {
		infostream << "Sound disabled." << std::endl;
	}
#endif

	if (!sound_manager) {
		infostream << "Using dummy audio." << std::endl;
		sound_manager = std::make_unique<DummySoundManager>();
	}

	soundmaker = std::make_unique<SoundMaker>(sound_manager.get(), nodedef_manager);
	soundmaker->registerReceiver(eventmgr);

	return true;
}

bool Game::createServer(const std::string &map_dir,
		const SubgameSpec &gamespec, u16 port)
{
	showOverlayMessage(N_("Creating server..."), 0, 5);

	std::string bind_str;
	if (simple_singleplayer_mode) {
		// Make the simple singleplayer server only accept connections from localhost,
		// which also makes Windows Defender not show a warning.
		bind_str = "127.0.0.1";
	} else {
		bind_str = g_settings->get("bind_address");
	}

	Address bind_addr(0, 0, 0, 0, port);

	if (g_settings->getBool("ipv6_server"))
		bind_addr.setAddress(static_cast<IPv6AddressBytes*>(nullptr));
	try {
		bind_addr.Resolve(bind_str.c_str());
	} catch (const ResolveError &e) {
		warningstream << "Resolving bind address \"" << bind_str
			<< "\" failed: " << e.what()
			<< " -- Listening on all addresses." << std::endl;
	}
	if (bind_addr.isIPv6() && !g_settings->getBool("enable_ipv6")) {
		*error_message = fmtgettext("Unable to listen on %s because IPv6 is disabled",
			bind_addr.serializeString().c_str());
		errorstream << *error_message << std::endl;
		return false;
	}

	server = new Server(map_dir, gamespec, simple_singleplayer_mode, bind_addr,
			false, nullptr, error_message);

	auto start_thread = runInThread([=] {
		server->start();
		copyServerClientCache();
	}, "ServerStart");

	input->clear();
	bool success = true;

	FpsControl fps_control;
	fps_control.reset();

	while (start_thread->isRunning()) {
		if (!m_rendering_engine->run() || input->cancelPressed())
			success = false;
		f32 dtime;
		fps_control.limit(device, &dtime);

		if (success)
			showOverlayMessage(N_("Creating server..."), dtime, 5);
		else
			showOverlayMessage(N_("Shutting down..."), dtime, 0, &m_shutdown_progress);
	}

	start_thread->rethrow();

	return success;
}

void Game::copyServerClientCache()
{
	// It would be possible to let the client directly read the media files
	// from where the server knows they are. But aside from being more complicated
	// it would also *not* fill the media cache and cause slower joining of
	// remote servers.
	// (Imagine that you launch a game once locally and then connect to a server.)

	assert(server);
	auto map = server->getMediaList();
	u32 n = 0;
	for (auto &it : map) {
		assert(it.first.size() == 20); // SHA1
		if (clientMediaUpdateCacheCopy(it.first, it.second))
			n++;
	}
	infostream << "Copied " << n << " files directly from server to client cache"
		<< std::endl;
}

bool Game::createClient(const GameStartData &start_data)
{
	showOverlayMessage(N_("Creating client..."), 0, 10);

	draw_control = new MapDrawControl();
	if (!draw_control)
		return false;

	bool could_connect, connect_aborted;
	if (!connectToServer(start_data, &could_connect, &connect_aborted))
		return false;

	if (!could_connect) {
		if (error_message->empty() && !connect_aborted) {
			// Should not happen if error messages are set properly
			*error_message = gettext("Connection failed for unknown reason");
			errorstream << *error_message << std::endl;
		}
		return false;
	}

	if (!getServerContent(&connect_aborted)) {
		if (error_message->empty() && !connect_aborted) {
			// Should not happen if error messages are set properly
			*error_message = gettext("Connection failed for unknown reason");
			errorstream << *error_message << std::endl;
		}
		return false;
	}

	// Pre-calculate crack length
	{
		auto size = texture_src->getTextureDimensions("crack_anylength.png");
		if (size.Width && size.Height)
			crack_animation_length = size.Height / size.Width;
		else
			crack_animation_length = 5;
	}

	shader_src->addShaderConstantSetter(
		std::make_unique<NodeShaderConstantSetter>());

	auto scsf_up = std::make_unique<GameGlobalShaderUniformSetterFactory>(this);
	auto* scsf = scsf_up.get();
	shader_src->addShaderUniformSetterFactory(std::move(scsf_up));

	shader_src->addShaderUniformSetterFactory(
		std::make_unique<FogShaderUniformSetterFactory>());

	ShadowRenderer::preInit(shader_src);

	// Update cached textures, meshes and materials
	client->afterContentReceived();

	/* Camera
	 */
	camera = new Camera(*draw_control, client, m_rendering_engine);
	if (client->modsLoaded())
		client->getScript()->on_camera_ready(camera);
	client->setCamera(camera);

	/* Clouds
	 */
	clouds = make_irr<Clouds>(smgr, shader_src, -1, myrand());

	/* Skybox
	 */
	sky = make_irr<Sky>(-1, m_rendering_engine, texture_src, shader_src);
	scsf->setSky(sky.get());

	if (!initGui())
		return false;

	/* Set window caption
	 */
	auto driver_name = driver->getName();
	std::string str = std::string(PROJECT_NAME_C) +
			" " + g_version_hash + " [";
	str += simple_singleplayer_mode ? gettext("Singleplayer")
			: gettext("Multiplayer");
	str += "] [";
	str += driver_name;
	str += "]";

	device->setWindowCaption(utf8_to_wide(str).c_str());

	LocalPlayer *player = client->getEnv().getLocalPlayer();
	player->hurt_tilt_timer = 0;
	player->hurt_tilt_strength = 0;

	hud = new Hud(client, player, &player->inventory);

	mapper = client->getMinimap();

	if (mapper && client->modsLoaded())
		client->getScript()->on_minimap_ready(mapper);

	return true;
}

bool Game::shouldShowTouchControls()
{
	if (!device->supportsTouchEvents())
		return false;

	const std::string &touch_controls = g_settings->get("touch_controls");
	if (touch_controls == "auto")
		return RenderingEngine::getLastPointerType() == PointerType::Touch;
	return is_yes(touch_controls);
}

bool Game::initGui()
{
	m_game_ui->init();

	// Remove stale "recent" chat messages from previous connections
	chat_backend->clearRecentChat();

	// Make sure the size of the recent messages buffer is right
	chat_backend->applySettings();

	// Chat backend and console
	gui_chat_console = make_irr<GUIChatConsole>(guienv, guienv->getRootGUIElement(),
			-1, chat_backend, client, &g_menumgr);

	if (shouldShowTouchControls())
		g_touchcontrols = new TouchControls(device, texture_src);

	return true;
}

bool Game::connectToServer(const GameStartData &start_data,
		bool *connect_ok, bool *connection_aborted)
{
	*connect_ok = false;	// Let's not be overly optimistic
	*connection_aborted = false;
	const auto &address_name = start_data.address;

	showOverlayMessage(N_("Resolving address..."), 0, 15);

	Address connect_address(0, 0, 0, 0, start_data.socket_port);
	Address fallback_address;

	try {
		connect_address.Resolve(address_name.c_str(), &fallback_address);

		if (connect_address.isAny()) {
			// replace with localhost IP
			if (connect_address.isIPv6()) {
				IPv6AddressBytes addr_bytes;
				addr_bytes.bytes[15] = 1;
				connect_address.setAddress(&addr_bytes);
			} else {
				connect_address.setAddress(127, 0, 0, 1);
			}
		}
	} catch (ResolveError &e) {
		*error_message = fmtgettext("Couldn't resolve address: %s", e.what());

		errorstream << *error_message << std::endl;
		return false;
	}

	// this shouldn't normally happen since Address::Resolve() checks for enable_ipv6
	if (g_settings->getBool("enable_ipv6")) {
		// empty
	} else if (connect_address.isIPv6()) {
		*error_message = fmtgettext("Unable to connect to %s because IPv6 is disabled", connect_address.serializeString().c_str());
		errorstream << *error_message << std::endl;
		return false;
	} else if (fallback_address.isIPv6()) {
		fallback_address = Address();
	}

	fallback_address.setPort(connect_address.getPort());
	if (fallback_address.isValid()) {
		infostream << "Resolved two addresses for \"" << address_name
			<< "\" isIPv6[0]=" << connect_address.isIPv6()
			<< " isIPv6[1]=" << fallback_address.isIPv6() << std::endl;
	} else {
		infostream << "Resolved one address for \"" << address_name
			<< "\" isIPv6=" << connect_address.isIPv6() << std::endl;
	}


	try {
		client = new Client(start_data.name.c_str(),
				start_data.password,
				*draw_control, texture_src, shader_src,
				itemdef_manager, nodedef_manager, sound_manager.get(), eventmgr,
				m_rendering_engine,
				m_item_visuals_manager.get(),
				start_data.allow_login_or_register);
	} catch (const BaseException &e) {
		*error_message = fmtgettext("Error creating client: %s", e.what());
		errorstream << *error_message << std::endl;
		return false;
	}

	client->migrateModStorage();
	client->m_simple_singleplayer_mode = simple_singleplayer_mode;
	client->m_internal_server = !!server;

	/*
		Wait for server to accept connection
	*/

	client->connect(connect_address, address_name);

	try {
		input->clear();

		FpsControl fps_control;
		f32 dtime;
		f32 wait_time = 0; // in seconds
		bool did_fallback = false;

		fps_control.reset();

		auto framemarker = FrameMarker("Game::connectToServer()-frame").started();

		while (m_rendering_engine->run()) {

			framemarker.end();
			fps_control.limit(device, &dtime);
			framemarker.start();

			// Update client and server
			step(dtime);

			// End condition
			if (client->getState() == LC_Init) {
				*connect_ok = true;
				break;
			}

			// Break conditions
			if (*connection_aborted)
				break;

			if (!checkConnection())
				break;

			if (input->cancelPressed()) {
				*connection_aborted = true;
				infostream << "Connect aborted [Escape]" << std::endl;
				break;
			}

			wait_time += dtime;
			if (server) {
				// never time out
			} else if (wait_time > GAME_FALLBACK_TIMEOUT && !did_fallback) {
				if (!client->hasServerReplied() && fallback_address.isValid()) {
					client->connect(fallback_address, address_name);
				}
				did_fallback = true;
			} else if (wait_time > GAME_CONNECTION_TIMEOUT) {
				*error_message = gettext("Connection timed out.");
				errorstream << *error_message << std::endl;
				break;
			}

			// Update status
			showOverlayMessage(N_("Connecting to server..."), dtime, 20);
		}
		framemarker.end();
	} catch (con::PeerNotFoundException &e) {
		warningstream << "This should not happen. Please report a bug." << std::endl;
		return false;
	}

	return true;
}

bool Game::getServerContent(bool *aborted)
{
	input->clear();

	FpsControl fps_control;
	f32 dtime; // in seconds

	fps_control.reset();

	auto framemarker = FrameMarker("Game::getServerContent()-frame").started();
	while (m_rendering_engine->run()) {
		framemarker.end();
		fps_control.limit(device, &dtime);
		framemarker.start();

		// Update client and server
		step(dtime);

		// End condition
		if (client->mediaReceived() && client->itemdefReceived() &&
				client->nodedefReceived()) {
			return true;
		}

		// Error conditions
		if (!checkConnection())
			return false;

		if (client->getState() < LC_Init) {
			*error_message = gettext("Client disconnected");
			errorstream << *error_message << std::endl;
			return false;
		}

		if (input->cancelPressed()) {
			*aborted = true;
			infostream << "Connect aborted [Escape]" << std::endl;
			return false;
		}

		// Display status
		int progress = 25;

		if (!client->itemdefReceived()) {
			progress = 25;
			m_rendering_engine->draw_load_screen(wstrgettext("Item definitions..."),
					guienv, texture_src, dtime, progress);
		} else if (!client->nodedefReceived()) {
			progress = 30;
			m_rendering_engine->draw_load_screen(wstrgettext("Node definitions..."),
					guienv, texture_src, dtime, progress);
		} else {
			std::ostringstream message;
			std::fixed(message);
			message.precision(0);
			float receive = client->mediaReceiveProgress() * 100;
			message << gettext("Media...");
			if (receive > 0)
				message << " " << receive << "%";
			message.precision(2);

			if ((USE_CURL == 0) ||
					(!g_settings->getBool("enable_remote_media_server"))) {
				float cur = client->getCurRate();
				std::string cur_unit = gettext("KiB/s");

				if (cur > 900) {
					cur /= 1024.0;
					cur_unit = gettext("MiB/s");
				}

				message << " (" << cur << ' ' << cur_unit << ")";
			}

			// 30% -> 65%
			progress = 30 + std::ceil(client->mediaReceiveProgress() * 35 + 0.5f);
			m_rendering_engine->draw_load_screen(utf8_to_wide(message.str()), guienv,
				texture_src, dtime, progress);
		}
	}
	framemarker.end();

	*aborted = true;
	infostream << "Connect aborted [device]" << std::endl;
	return false;
}


/****************************************************************************/
/****************************************************************************
 Run
 ****************************************************************************/
/****************************************************************************/

inline void Game::updateInteractTimers(f32 dtime)
{
	if (runData.nodig_delay_timer >= 0)
		runData.nodig_delay_timer -= dtime;

	if (runData.object_hit_delay_timer >= 0)
		runData.object_hit_delay_timer -= dtime;

	runData.time_from_last_punch += dtime;
}


/* returns false if game should exit, otherwise true
 */
bool Game::checkConnection()
{
	if (client->accessDenied()) {
		// May be mod-provided, thus may contain color and translation
		const std::string reason = wide_to_utf8(
			unescape_translate(utf8_to_wide(client->accessDeniedReason())));

		*error_message = fmtgettext("Access denied. Reason: %s", reason.c_str());
		*reconnect_requested = client->reconnectRequested();
		errorstream << *error_message << std::endl;
		return false;
	}

	return true;
}

void Game::processQueues()
{
	texture_src->processQueue();
	shader_src->processQueue();
}

void Game::updateDebugState()
{
	LocalPlayer *player = client->getEnv().getLocalPlayer();

	// debug UI and wireframe
	bool has_debug = client->checkPrivilege("debug");
	bool has_basic_debug = has_debug || (player->hud_flags & HUD_FLAG_BASIC_DEBUG);

	if (m_game_ui->m_flags.show_basic_debug) {
		if (!has_basic_debug)
			m_game_ui->m_flags.show_basic_debug = false;
	} else if (m_game_ui->m_flags.show_minimal_debug) {
		if (has_basic_debug)
			m_game_ui->m_flags.show_basic_debug = true;
	}
	if (!has_basic_debug)
		hud->disableBlockBounds();
	if (!has_debug) {
		draw_control->show_wireframe = false;
		smgr->setGlobalDebugData(0, bbox_debug_flag);
		m_flags.disable_camera_update = false;
		m_game_formspec.disableDebugView();
	}

	// noclip
	draw_control->allow_noclip = m_cache_enable_noclip && client->checkPrivilege("noclip");
}

void Game::updateProfilers(const RunStats &stats, const FpsControl &draw_times,
		f32 dtime)
{
	float profiler_print_interval =
			g_settings->getFloat("profiler_print_interval");
	bool print_to_log = true;

	// Update game UI anyway but don't log
	if (profiler_print_interval <= 0) {
		print_to_log = false;
		profiler_print_interval = 3;
	}

	// Update graphs
	g_profiler->graphAdd("Time non-rendering [us]",
		draw_times.busy_time - stats.drawtime);
	g_profiler->graphAdd("Sleep [us]", draw_times.sleep_time);

	g_profiler->graphSet("FPS", 1.0f / dtime);

	auto stats2 = driver->getFrameStats();
	g_profiler->avg("Irr: drawcalls", stats2.Drawcalls);
	if (stats2.Drawcalls > 0)
		g_profiler->avg("Irr: primitives per drawcall",
			stats2.PrimitivesDrawn / float(stats2.Drawcalls));
	g_profiler->avg("Irr: HW buffers uploaded", stats2.HWBuffersUploaded);
	g_profiler->avg("Irr: HW buffers active", stats2.HWBuffersActive);
	u32 skinned_meshes = stats2.SWSkinnedMeshes + stats2.HWSkinnedMeshes;
	if (skinned_meshes > 0) {
		f32 use_pct = std::floor(100.0f * stats2.HWSkinnedMeshes / skinned_meshes);
		g_profiler->avg("Irr: HW skinning use [%]", use_pct);
	}

	if (profiler_interval.step(dtime, profiler_print_interval)) {
		if (print_to_log) {
			infostream << "Profiler:" << std::endl;
			g_profiler->print(infostream);
		}

		m_game_ui->updateProfiler();
		g_profiler->clear();
	}
}

void Game::updateStats(RunStats *stats, const FpsControl &draw_times,
		f32 dtime)
{

	f32 jitter;
	Jitter *jp;

	/* Time average and jitter calculation
	 */
	jp = &stats->dtime_jitter;
	jp->avg = jp->avg * 0.96 + dtime * 0.04;

	jitter = dtime - jp->avg;

	if (jitter > jp->max)
		jp->max = jitter;

	jp->counter += dtime;

	if (jp->counter > 0.0) {
		jp->counter -= 3.0;
		jp->max_sample = jp->max;
		jp->max_fraction = jp->max_sample / (jp->avg + 0.001);
		jp->max = 0.0;
	}

	/* Busytime average and jitter calculation
	 */
	jp = &stats->busy_time_jitter;
	jp->avg = jp->avg + draw_times.getBusyMs() * 0.02;

	jitter = draw_times.getBusyMs() - jp->avg;

	if (jitter > jp->max)
		jp->max = jitter;
	if (jitter < jp->min)
		jp->min = jitter;

	jp->counter += dtime;

	if (jp->counter > 0.0) {
		jp->counter -= 3.0;
		jp->max_sample = jp->max;
		jp->min_sample = jp->min;
		jp->max = 0.0;
		jp->min = 0.0;
	}
}



/****************************************************************************
 Input handling
 ****************************************************************************/

void Game::processUserInput(f32 dtime)
{
	bool desired = shouldShowTouchControls();
	if (desired && !g_touchcontrols) {
		g_touchcontrols = new TouchControls(device, texture_src);

	} else if (!desired && g_touchcontrols) {
		delete g_touchcontrols;
		g_touchcontrols = nullptr;
	}

	// Reset input if window not active or some menu is active
	if (!device->isWindowActive() || isMenuActive() || guienv->hasFocus(gui_chat_console.get())) {
		if (m_game_focused) {
			m_game_focused = false;
			infostream << "Game lost focus" << std::endl;
			input->releaseAllKeys();
		} else {
			input->clear();
		}

		if (g_touchcontrols)
			g_touchcontrols->hide();

	} else {
		if (g_touchcontrols) {
			/* on touchcontrols step may generate own input events which ain't
			 * what we want in case we just did clear them */
			g_touchcontrols->show();
			g_touchcontrols->step(dtime);
		}

		m_game_focused = true;
	}

	if (!guienv->hasFocus(gui_chat_console.get()) && gui_chat_console->isOpen()
		&& !gui_chat_console->isMyDescendant(guienv->getFocus()))
	{
		gui_chat_console->closeConsoleAtOnce();
	}

	// Input handler step() (used by the random input generator)
	input->step(dtime);

#ifdef __ANDROID__
	if (!m_game_formspec.handleAndroidUIInput())
		handleAndroidChatInput();
#endif

	// Increase timer for double tap of "keymap_jump"
	if (m_cache_doubletap_jump && runData.jump_timer_up <= 0.2f)
		runData.jump_timer_up += dtime;
	if (m_cache_doubletap_jump && runData.jump_timer_down <= 0.4f)
		runData.jump_timer_down += dtime;

	processKeyInput();
	// P: tap start/stop time, hold to fast-forward (needs dtime, so it
	// lives outside the pressed-key chain)
	claudeTimeKey(isKeyDown(KeyType::CLAUDE_TIME_TOGGLE), dtime);
	processItemSelection(&runData.new_playeritem);
}


void Game::processKeyInput()
{
	if (wasKeyDown(KeyType::DROP)) {
		dropSelectedItem(isKeyDown(KeyType::SNEAK));
	} else if (wasKeyDown(KeyType::AUTOFORWARD)) {
		toggleAutoforward();
	} else if (wasKeyDown(KeyType::BACKWARD)) {
		if (g_settings->getBool("continuous_forward"))
			toggleAutoforward();
	} else if (wasKeyDown(KeyType::INVENTORY)) {
		m_game_formspec.showPlayerInventory(nullptr);
	} else if (input->cancelPressed()) {
#ifdef __ANDROID__
		m_android_chat_open = false;
#endif
		if (!gui_chat_console->isOpenInhibited()) {
			m_game_formspec.showPauseMenu();
		}
	} else if (wasKeyDown(KeyType::CHAT)) {
		openConsole(0.2, L"");
	} else if (wasKeyDown(KeyType::CMD)) {
		openConsole(0.2, L"/");
	} else if (wasKeyDown(KeyType::CMD_LOCAL)) {
		if (client->modsLoaded())
			openConsole(0.2, L".");
		else
			m_game_ui->showTranslatedStatusText("Client side scripting is disabled");
	} else if (wasKeyDown(KeyType::CONSOLE)) {
		openConsole(core::clamp(g_settings->getFloat("console_height"), 0.1f, 1.0f));
	} else if (wasKeyDown(KeyType::FREEMOVE)) {
		toggleFreeMove();
	} else if (wasKeyDown(KeyType::JUMP)) {
		toggleFreeMoveAlt();
	} else if (wasKeyDown(KeyType::PITCHMOVE)) {
		togglePitchMove();
	} else if (wasKeyDown(KeyType::FASTMOVE)) {
		toggleFast();
	} else if (wasKeyDown(KeyType::NOCLIP)) {
		toggleNoClip();
#if USE_SOUND
	} else if (wasKeyDown(KeyType::MUTE)) {
		bool new_mute_sound = !g_settings->getBool("mute_sound");
		g_settings->setBool("mute_sound", new_mute_sound);
		if (new_mute_sound)
			m_game_ui->showTranslatedStatusText("Sound muted");
		else
			m_game_ui->showTranslatedStatusText("Sound unmuted");
	} else if (wasKeyDown(KeyType::INC_VOLUME)) {
		float new_volume = g_settings->getFloat("sound_volume", 0.0f, 0.9f) + 0.1f;
		g_settings->setFloat("sound_volume", new_volume);
		std::wstring msg = fwgettext("Volume changed to %d%%", myround(new_volume * 100));
		m_game_ui->showStatusText(msg);
	} else if (wasKeyDown(KeyType::DEC_VOLUME)) {
		float new_volume = g_settings->getFloat("sound_volume", 0.1f, 1.0f) - 0.1f;
		g_settings->setFloat("sound_volume", new_volume);
		std::wstring msg = fwgettext("Volume changed to %d%%", myround(new_volume * 100));
		m_game_ui->showStatusText(msg);
#else
	} else if (wasKeyDown(KeyType::MUTE) || wasKeyDown(KeyType::INC_VOLUME)
			|| wasKeyDown(KeyType::DEC_VOLUME)) {
		m_game_ui->showTranslatedStatusText("Sound system is not supported on this build");
#endif
	} else if (wasKeyDown(KeyType::CINEMATIC)) {
		toggleCinematic();
	} else if (wasKeyPressed(KeyType::SCREENSHOT)) {
		client->makeScreenshot();
	} else if (wasKeyPressed(KeyType::TOGGLE_BLOCK_BOUNDS)) {
		toggleBlockBounds();
	} else if (wasKeyPressed(KeyType::TOGGLE_HUD)) {
		m_game_ui->toggleHud();
	} else if (wasKeyPressed(KeyType::MINIMAP)) {
		toggleMinimap(isKeyDown(KeyType::SNEAK));
	} else if (wasKeyPressed(KeyType::TOGGLE_CHAT)) {
		m_game_ui->toggleChat(client);
	} else if (wasKeyPressed(KeyType::TOGGLE_FOG)) {
		toggleFog();
	} else if (wasKeyPressed(KeyType::TOGGLE_CLAUDE_TRACE)) {
		toggleClaudeTrace();
	} else if (wasKeyPressed(KeyType::TOGGLE_CLAUDE_BOUNCE)) {
		toggleClaudeBounce();
	} else if (wasKeyPressed(KeyType::CLAUDE_TIME_BACK)) {
		claudeTimeNudge(-1);
	} else if (wasKeyPressed(KeyType::CLAUDE_TIME_FWD)) {
		claudeTimeNudge(1);
	} else if (wasKeyDown(KeyType::TOGGLE_UPDATE_CAMERA)) {
		toggleUpdateCamera();
	} else if (wasKeyPressed(KeyType::CAMERA_MODE)) {
		camera->toggleCameraMode();
		updateCameraMode();
	} else if (wasKeyPressed(KeyType::TOGGLE_DEBUG)) {
		toggleDebug();
	} else if (wasKeyPressed(KeyType::TOGGLE_PROFILER)) {
		m_game_ui->toggleProfiler();
	} else if (wasKeyDown(KeyType::INCREASE_VIEWING_RANGE)) {
		increaseViewRange();
	} else if (wasKeyDown(KeyType::DECREASE_VIEWING_RANGE)) {
		decreaseViewRange();
	} else if (wasKeyPressed(KeyType::RANGESELECT)) {
		toggleFullViewRange();
	} else if (wasKeyDown(KeyType::ZOOM)) {
		checkZoomEnabled();
	} else if (wasKeyDown(KeyType::QUICKTUNE_NEXT)) {
		quicktune->next();
	} else if (wasKeyDown(KeyType::QUICKTUNE_PREV)) {
		quicktune->prev();
	} else if (wasKeyDown(KeyType::QUICKTUNE_INC)) {
		quicktune->inc();
	} else if (wasKeyDown(KeyType::QUICKTUNE_DEC)) {
		quicktune->dec();
	}

	if (!isKeyDown(KeyType::JUMP) && runData.reset_jump_timer) {
		runData.reset_jump_timer = false;
		runData.jump_timer_up = 0.0f;
	}

	if (quicktune->hasMessage()) {
		m_game_ui->showStatusText(utf8_to_wide(quicktune->getMessage()));
	}
}

void Game::processItemSelection(u16 *new_playeritem)
{
	LocalPlayer *player = client->getEnv().getLocalPlayer();

	*new_playeritem = player->getWieldIndex();
	u16 max_item = player->getMaxHotbarItemcount();
	if (max_item == 0)
		return;
	max_item -= 1;

	/* Item selection using mouse wheel
	 */
	s32 wheel = input->getMouseWheel();
	if (!m_enable_hotbar_mouse_wheel)
		wheel = 0;
	if (m_invert_hotbar_mouse_wheel)
		wheel *= -1;

	s32 dir = wheel;

	if (wasKeyDown(KeyType::HOTBAR_NEXT))
		dir = -1;

	if (wasKeyDown(KeyType::HOTBAR_PREV))
		dir = 1;

	if (dir < 0)
		*new_playeritem = *new_playeritem < max_item ? *new_playeritem + 1 : 0;
	else if (dir > 0)
		*new_playeritem = *new_playeritem > 0 ? *new_playeritem - 1 : max_item;
	// else dir == 0

	/* Item selection using hotbar slot keys
	 */
	for (u16 i = 0; i <= max_item; i++) {
		if (wasKeyDown((GameKeyType) (KeyType::SLOT_1 + i))) {
			*new_playeritem = i;
			break;
		}
	}

	if (g_touchcontrols) {
		std::optional<u16> selection = g_touchcontrols->getHotbarSelection();
		if (selection)
			*new_playeritem = *selection;
	}

	// Clamp selection again in case it wasn't changed but max_item was
	*new_playeritem = MYMIN(*new_playeritem, max_item);
}


void Game::dropSelectedItem(bool single_item)
{
	IDropAction *a = new IDropAction();
	a->count = single_item ? 1 : 0;
	a->from_inv.setCurrentPlayer();
	a->from_list = "main";
	a->from_i = client->getEnv().getLocalPlayer()->getWieldIndex();
	client->inventoryAction(a);
}

void Game::openConsole(float scale, const wchar_t *line)
{
	assert(scale > 0.0f && scale <= 1.0f);

#ifdef __ANDROID__
	if (!porting::hasPhysicalKeyboardAndroid()) {
		porting::showTextInputDialog("", "", 2);
		m_android_chat_open = true;
	} else {
#endif
	if (gui_chat_console->isOpenInhibited())
		return;
	gui_chat_console->openConsole(scale);
	if (line) {
		gui_chat_console->setCloseOnEnter(true);
		gui_chat_console->replaceAndAddToHistory(line);
	}
#ifdef __ANDROID__
	} // else
#endif
}

#ifdef __ANDROID__
void Game::handleAndroidChatInput()
{
	// It has to be a text input
	if (m_android_chat_open && porting::getLastInputDialogType() == porting::TEXT_INPUT) {
		porting::AndroidDialogState dialogState = porting::getInputDialogState();
		if (dialogState == porting::DIALOG_INPUTTED) {
			std::string text = porting::getInputDialogMessage();
			client->typeChatMessage(utf8_to_wide(text));
		}
		if (dialogState != porting::DIALOG_SHOWN)
			m_android_chat_open = false;
	}
}
#endif

void Game::toggleFreeMove()
{
	bool free_move = !g_settings->getBool("free_move");
	g_settings->set("free_move", bool_to_cstr(free_move));

	if (free_move) {
		if (client->checkPrivilege("fly")) {
			m_game_ui->showTranslatedStatusText("Fly mode enabled");
		} else {
			m_game_ui->showTranslatedStatusText("Fly mode enabled (note: no 'fly' privilege)");
		}
	} else {
		m_game_ui->showTranslatedStatusText("Fly mode disabled");
	}
}

void Game::toggleFreeMoveAlt()
{
	if (!runData.reset_jump_timer) {
		runData.jump_timer_down_before = runData.jump_timer_down;
		runData.jump_timer_down = 0.0f;
	}

	// key down (0.2 s max.), then key up (0.2 s max.), then key down
	if (m_cache_doubletap_jump && runData.jump_timer_up < 0.2f &&
			runData.jump_timer_down_before < 0.4f) // 0.2 + 0.2
		toggleFreeMove();

	runData.reset_jump_timer = true;
}


void Game::togglePitchMove()
{
	bool pitch_move = !g_settings->getBool("pitch_move");
	g_settings->set("pitch_move", bool_to_cstr(pitch_move));

	if (pitch_move) {
		m_game_ui->showTranslatedStatusText("Pitch move mode enabled");
	} else {
		m_game_ui->showTranslatedStatusText("Pitch move mode disabled");
	}
}


void Game::toggleFast()
{
	bool fast_move = !g_settings->getBool("fast_move");
	bool has_fast_privs = client->checkPrivilege("fast");
	g_settings->set("fast_move", bool_to_cstr(fast_move));

	if (fast_move) {
		if (has_fast_privs) {
			m_game_ui->showTranslatedStatusText("Fast mode enabled");
		} else {
			m_game_ui->showTranslatedStatusText("Fast mode enabled (note: no 'fast' privilege)");
		}
	} else {
		m_game_ui->showTranslatedStatusText("Fast mode disabled");
	}

	m_touch_simulate_aux1 = fast_move && has_fast_privs;
}


void Game::toggleNoClip()
{
	bool noclip = !g_settings->getBool("noclip");
	g_settings->set("noclip", bool_to_cstr(noclip));

	if (noclip) {
		if (client->checkPrivilege("noclip")) {
			m_game_ui->showTranslatedStatusText("Noclip mode enabled");
		} else {
			m_game_ui->showTranslatedStatusText("Noclip mode enabled (note: no 'noclip' privilege)");
		}
	} else {
		m_game_ui->showTranslatedStatusText("Noclip mode disabled");
	}
}

void Game::toggleCinematic()
{
	bool cinematic = !g_settings->getBool("cinematic");
	g_settings->set("cinematic", bool_to_cstr(cinematic));

	if (cinematic)
		m_game_ui->showTranslatedStatusText("Cinematic mode enabled");
	else
		m_game_ui->showTranslatedStatusText("Cinematic mode disabled");
}

void Game::toggleBlockBounds()
{
	LocalPlayer *player = client->getEnv().getLocalPlayer();
	if (!(client->checkPrivilege("debug") || (player->hud_flags & HUD_FLAG_BASIC_DEBUG))) {
		m_game_ui->showTranslatedStatusText("Can't show block bounds (disabled by game or mod)");
		return;
	}
	enum Hud::BlockBoundsMode newmode = hud->toggleBlockBounds();
	switch (newmode) {
		case Hud::BLOCK_BOUNDS_OFF:
			m_game_ui->showTranslatedStatusText("Block bounds hidden");
			break;
		case Hud::BLOCK_BOUNDS_CURRENT:
			m_game_ui->showTranslatedStatusText("Block bounds shown for current block");
			break;
		case Hud::BLOCK_BOUNDS_NEAR:
			m_game_ui->showTranslatedStatusText("Block bounds shown for nearby blocks");
			break;
		default:
			break;
	}
}

// Autoforward by toggling continuous forward.
void Game::toggleAutoforward()
{
	bool autorun_enabled = !g_settings->getBool("continuous_forward");
	g_settings->set("continuous_forward", bool_to_cstr(autorun_enabled));

	if (autorun_enabled)
		m_game_ui->showTranslatedStatusText("Automatic forward enabled");
	else
		m_game_ui->showTranslatedStatusText("Automatic forward disabled");
}

void Game::toggleMinimap(bool shift_pressed)
{
	if (!mapper || !m_game_ui->m_flags.show_hud || !g_settings->getBool("enable_minimap"))
		return;

	if (shift_pressed)
		mapper->toggleMinimapShape();
	else
		mapper->nextMode();

	// TODO: When legacy minimap is deprecated, keep only HUD minimap stuff here

	// Not so satisying code to keep compatibility with old fixed mode system
	// -->
	u32 hud_flags = client->getEnv().getLocalPlayer()->hud_flags;

	// If radar is disabled, try to find a non radar mode or fall back to 0
	if (!(hud_flags & HUD_FLAG_MINIMAP_RADAR_VISIBLE))
		while (mapper->getModeIndex() &&
				mapper->getModeDef().type == MINIMAP_TYPE_RADAR)
			mapper->nextMode();
	// <--
	// End of 'not so satifying code'
	if (hud && hud->hasElementOfType(HUD_ELEM_MINIMAP))
		m_game_ui->showStatusText(utf8_to_wide(mapper->getModeDef().label));
	else
		m_game_ui->showTranslatedStatusText("Minimap currently disabled by game or mod");
}

// One key, one client: vanilla raster vs the traced renderer. The whole
// traced pipeline hangs off claude_grid_debug (0 = pass-through, 3 =
// path traced), read live by the uniform setter, so flipping the setting
// IS the toggle — no restart, no second build.
void Game::toggleClaudeTrace()
{
	float cur = g_settings->getFloat("claude_grid_debug", 0.0f, 10.0f);
	bool to_traced = cur < 2.5f;
	g_settings->set("claude_grid_debug", to_traced ? "3" : "0");
	if (to_traced)
		m_game_ui->showTranslatedStatusText("Ray tracing ON");
	else
		m_game_ui->showTranslatedStatusText("Ray tracing OFF (vanilla)");
}

// G: direct light only vs full transport, for instant A/B — the tell is
// the wall a torch cannot directly see. (B was taken: hotbar_previous.)
void Game::toggleClaudeBounce()
{
	float cur = g_settings->getFloat("claude_bounces", 0.0f, 64.0f);
	bool to_full = cur < 12.0f;
	g_settings->set("claude_bounces", to_full ? "24" : "1");
	claudeResetAccumulation();
	if (to_full)
		m_game_ui->showTranslatedStatusText(
				"Bounces 24 (full transport)");
	else
		m_game_ui->showTranslatedStatusText("Multi-bounce OFF (one bounce)");
}

// P: tap = start/stop time, hold (>0.4s) = fast-forward while held,
// releasing restores the tap state. The client tracks running/stopped
// itself (it can't read the server setting), so a manual /set
// time_speed can drift the toggle by one tap — self-corrects on use.
void Game::claudeTimeKey(bool down, f32 dtime)
{
	if (down) {
		m_claude_time_held += dtime;
		if (m_claude_time_held > 0.4f && !m_claude_time_fast) {
			m_claude_time_fast = true;
			client->sendChatMessage(utf8_to_wide("/set time_speed 5000"));
			m_game_ui->showTranslatedStatusText("Time: fast-forward");
		}
		return;
	}
	if (m_claude_time_held <= 0.0f)
		return;
	if (m_claude_time_fast)
		m_claude_time_fast = false; // release restores the tap state
	else
		m_claude_time_running = !m_claude_time_running;
	client->sendChatMessage(utf8_to_wide(m_claude_time_running
			? "/set time_speed 72" : "/set time_speed 0"));
	if (m_claude_time_running)
		m_game_ui->showTranslatedStatusText("Time: running");
	else
		m_game_ui->showTranslatedStatusText("Time: stopped");
	m_claude_time_held = 0.0f;
}

// [ / ]: nudge server time an hour back/forward (sends /time; the
// player account carries settime). Golden-hour hunting without typing.
void Game::claudeTimeNudge(int dir)
{
	u32 tod = client->getEnv().getTimeOfDay(); // 0..23999
	int t = ((int)tod + dir * 1000 + 24000) % 24000;
	client->sendChatMessage(utf8_to_wide("/time " + std::to_string(t)));
	if (dir > 0)
		m_game_ui->showTranslatedStatusText("Time +1 hour");
	else
		m_game_ui->showTranslatedStatusText("Time -1 hour");
}

void Game::toggleFog()
{
	bool flag = !g_settings->getBool("enable_fog");
	g_settings->setBool("enable_fog", flag);
	bool allowed = sky->getFogDistance() < 0 || client->checkPrivilege("debug");
	if (!allowed)
		m_game_ui->showTranslatedStatusText("Fog enabled by game or mod");
	else if (flag)
		m_game_ui->showTranslatedStatusText("Fog enabled");
	else
		m_game_ui->showTranslatedStatusText("Fog disabled");
}


void Game::toggleDebug()
{
	LocalPlayer *player = client->getEnv().getLocalPlayer();
	bool has_debug = client->checkPrivilege("debug");
	bool has_basic_debug = has_debug || (player->hud_flags & HUD_FLAG_BASIC_DEBUG);

	// Initial: No debug info
	// 1x toggle: Debug text
	// 2x toggle: Debug text with profiler graph
	// 3x toggle: Debug text and wireframe (needs "debug" priv)
	// 4x toggle: Debug text and bbox (needs "debug" priv)
	//
	// The debug text can be in 2 modes: minimal and basic.
	// * Minimal: Only technical client info that not gameplay-relevant
	// * Basic: Info that might give gameplay advantage, e.g. pos, angle
	// Basic mode is used when player has the debug HUD flag set,
	// otherwise the Minimal mode is used.

	auto &state = m_flags.debug_state;
	state = (state + 1) % 5;
	if (state >= 3 && !has_debug)
		state = 0;

//
// IT USED TO FLIP claude_radiance, WHICH DID NOTHING AT ALL. That dial
// fed `radianceStrength`, a uniform no shader in this tree declares, so
// the key printed "Multi-bounce ON" and changed not one pixel — the
// silent-uniform bug class in spec/environment-laws.md, found by John
// pressing the key and seeing nothing (2026-08-16). The dial that
// actually decides how far light bounces is claude_bounces (the shader
// declares `claudeBounces` and the path loop reads it), so the key now
// flips that: 1 = direct light only, 24 = the full transport the goldens
// are taken with.
//
// The accumulator is reset here for the same reason the view key resets
// it: at a parked camera accumAlpha is ~1/N, so the frames taken under
// the OLD bounce count would survive in the running average for a long
// time and the A/B would compare a blend against a blend.
	m_game_ui->m_flags.show_minimal_debug = state > 0;
	m_game_ui->m_flags.show_basic_debug = state > 0 && has_basic_debug;
	m_game_ui->m_flags.show_profiler_graph = state == 2;
	draw_control->show_wireframe = state == 3;
	smgr->setGlobalDebugData(state == 4 ? bbox_debug_flag : 0,
			state == 4 ? 0 : bbox_debug_flag);

	if (state == 1) {
		m_game_ui->showTranslatedStatusText("Debug info shown");
	} else if (state == 2) {
		m_game_ui->showTranslatedStatusText("Profiler graph shown");
	} else if (state == 3) {
		if (driver->getDriverType() == video::EDT_OGLES2)
			m_game_ui->showTranslatedStatusText("Wireframe not supported by video driver");
		else
			m_game_ui->showTranslatedStatusText("Wireframe shown");
	} else if (state == 4) {
		m_game_ui->showTranslatedStatusText("Bounding boxes shown");
	} else {
		m_game_ui->showTranslatedStatusText("All debug info hidden");
	}
}


void Game::toggleUpdateCamera()
{
	auto &flag = m_flags.disable_camera_update;
	flag = client->checkPrivilege("debug") ? !flag : false;
	if (flag)
		m_game_ui->showTranslatedStatusText("Camera update disabled");
	else
		m_game_ui->showTranslatedStatusText("Camera update enabled");
}


void Game::increaseViewRange()
{
	s16 range = g_settings->getS16("viewing_range");
	s16 range_new = range + 10;
	s16 server_limit = sky->getFogDistance();

	if (range_new >= 4000) {
		range_new = 4000;
		std::wstring msg = server_limit >= 0 && range_new > server_limit ?
				fwgettext("Viewing range changed to %d (the maximum), but limited to %d by game or mod", range_new, server_limit) :
				fwgettext("Viewing range changed to %d (the maximum)", range_new);
		m_game_ui->showStatusText(msg);
	} else {
		std::wstring msg = server_limit >= 0 && range_new > server_limit ?
				fwgettext("Viewing range changed to %d, but limited to %d by game or mod", range_new, server_limit) :
				fwgettext("Viewing range changed to %d", range_new);
		m_game_ui->showStatusText(msg);
	}
	g_settings->set("viewing_range", itos(range_new));
}


void Game::decreaseViewRange()
{
	s16 range = g_settings->getS16("viewing_range");
	s16 range_new = range - 10;
	s16 server_limit = sky->getFogDistance();

	if (range_new <= 20) {
		range_new = 20;
		std::wstring msg = server_limit >= 0 && range_new > server_limit ?
				fwgettext("Viewing changed to %d (the minimum), but limited to %d by game or mod", range_new, server_limit) :
				fwgettext("Viewing changed to %d (the minimum)", range_new);
		m_game_ui->showStatusText(msg);
	} else {
		std::wstring msg = server_limit >= 0 && range_new > server_limit ?
				fwgettext("Viewing range changed to %d, but limited to %d by game or mod", range_new, server_limit) :
				fwgettext("Viewing range changed to %d", range_new);
		m_game_ui->showStatusText(msg);
	}
	g_settings->set("viewing_range", itos(range_new));
}


void Game::toggleFullViewRange()
{
	draw_control->range_all = !draw_control->range_all;
	if (draw_control->range_all) {
		if (sky->getFogDistance() >= 0) {
			m_game_ui->showTranslatedStatusText("Unlimited viewing range enabled, but forbidden by game or mod");
		} else {
			m_game_ui->showTranslatedStatusText("Unlimited viewing range enabled");
		}
	} else {
		m_game_ui->showTranslatedStatusText("Unlimited viewing range disabled");
	}
}


void Game::checkZoomEnabled()
{
	LocalPlayer *player = client->getEnv().getLocalPlayer();
	// Only an ABSOLUTE server FOV override actually blocks zoom; a multiplier
	// scales it (see Camera::update). Testing spec.fov > 0.0f for both is why
	// this nagged forever after a single sprint under Mineclonia.
	PlayerFovSpec spec = player->getFov();
	if (player->getZoomFOV() < 0.001f || (spec.fov > 0.0f && !spec.is_multiplier))
		m_game_ui->showTranslatedStatusText("Zoom currently disabled by game or mod");
}

void Game::updateCameraDirection(CameraOrientation *cam, float dtime)
{
	auto *cur_control = device->getCursorControl();

	/* On Linux and Windows, enabling relative mouse mode somehow results
	in simulated mouse events being generated from touch events, even though
	SDL_HINT_MOUSE_TOUCH_EVENTS and SDL_HINT_TOUCH_MOUSE_EVENTS are set to 0.
	Since we have our own code to synthesize mouse events from touch events,
	this results in duplicated input. To avoid that, we don't enable relative
	mouse mode if we're in touchscreen mode. */
	if (cur_control)
		cur_control->setRelativeMode(!g_touchcontrols && !isMenuActive());

	if ((device->isWindowActive() && device->isWindowFocused()
			&& !isMenuActive()) || input->isRandom()) {

		if (cur_control && !input->isRandom()) {
			// Mac OSX gets upset if this is set every frame
			if (cur_control->isVisible())
				cur_control->setVisible(false);
		}

		if (m_first_loop_after_window_activation && !g_touchcontrols) {
			m_first_loop_after_window_activation = false;

			input->setMousePos(driver->getScreenSize().Width / 2,
				driver->getScreenSize().Height / 2);
		} else if (!claudeInputLocked()) {
			updateCameraOrientation(cam, dtime);
		}

	} else {
		// Mac OSX gets upset if this is set every frame
		if (cur_control && !cur_control->isVisible())
			cur_control->setVisible(true);

		m_first_loop_after_window_activation = true;
	}
	if (g_touchcontrols)
		m_first_loop_after_window_activation = true;
}

// Get the factor to multiply with sensitivity to get the same mouse/joystick
// responsiveness independently of FOV.
f32 Game::getSensitivityScaleFactor() const
{
	f32 fov_y = client->getCamera()->getFovY();

	// Multiply by a constant such that it becomes 1.0 at 72 degree FOV and
	// 16:9 aspect ratio to minimize disruption of existing sensitivity
	// settings.
	return std::tan(fov_y / 2.0f) * 1.3763819f;
}

bool Game::isTouchShootlineUsed() const
{
	return g_touchcontrols && g_touchcontrols->isShootlineAvailable() &&
			camera->getCameraMode() == CAMERA_MODE_FIRST;
}

void Game::updateCameraOrientation(CameraOrientation *cam, float dtime)
{
	f32 sens_scale = getSensitivityScaleFactor();

	if (g_touchcontrols) {
		// User setting is already applied by TouchControls.
		cam->camera_yaw   += g_touchcontrols->getYawChange()   * sens_scale;
		cam->camera_pitch += g_touchcontrols->getPitchChange() * sens_scale;
	} else {
		v2s32 center(driver->getScreenSize().Width / 2, driver->getScreenSize().Height / 2);
		v2s32 dist = input->getMousePos() - center;

		if (m_invert_mouse || camera->getCameraMode() == CAMERA_MODE_THIRD_FRONT) {
			dist.Y = -dist.Y;
		}

		cam->camera_yaw   -= dist.X * m_cache_mouse_sensitivity * sens_scale;
		cam->camera_pitch += dist.Y * m_cache_mouse_sensitivity * sens_scale;

		if (dist.X != 0 || dist.Y != 0)
			input->setMousePos(center.X, center.Y);
	}

	if (m_cache_enable_joysticks) {
		f32 c = m_cache_joystick_frustum_sensitivity * dtime * sens_scale;
		cam->camera_yaw -= input->joystick.getAxisWithoutDead(JA_FRUSTUM_HORIZONTAL) * c;
		cam->camera_pitch += input->joystick.getAxisWithoutDead(JA_FRUSTUM_VERTICAL) * c;
	}

	// Keyboard look
	const f32 rate = m_cache_keyboard_camera_speed * dtime * sens_scale;

	if (input->isKeyDown(KeyType::CAMERA_YAW_LEFT))
		cam->camera_yaw += rate;
	if (input->isKeyDown(KeyType::CAMERA_YAW_RIGHT))
		cam->camera_yaw -= rate;
	if (input->isKeyDown(KeyType::CAMERA_PITCH_UP))
		cam->camera_pitch -= rate;
	if (input->isKeyDown(KeyType::CAMERA_PITCH_DOWN))
		cam->camera_pitch += rate;

	cam->camera_pitch = rangelim(cam->camera_pitch, -90, 90);
}


// Get the state of an optionally togglable key
bool Game::getTogglableKeyState(GameKeyType key, bool toggling_enabled, bool prev_key_state)
{
	if (!toggling_enabled)
		return isKeyDown(key);
	else
		return prev_key_state ^ wasKeyPressed(key);
}


void Game::updatePlayerControl(const CameraOrientation &cam)
{
	LocalPlayer *player = client->getEnv().getLocalPlayer();

	// In free move (fly), the "toggle_sneak_key" setting would prevent precise
	// up/down movements. Hence, enable the feature only during 'normal' movement.
	const bool allow_sneak_toggle = m_cache_toggle_sneak_key &&
		!(player->getPlayerSettings().free_move && client->checkPrivilege("fly"));

	//TimeTaker tt("update player control", NULL, PRECISION_NANO);

	PlayerControl control(
		isKeyDown(KeyType::FORWARD),
		isKeyDown(KeyType::BACKWARD),
		isKeyDown(KeyType::LEFT),
		isKeyDown(KeyType::RIGHT),
		isKeyDown(KeyType::JUMP) || player->getAutojump(),
		getTogglableKeyState(KeyType::AUX1,  m_cache_toggle_aux1_key, player->control.aux1),
		getTogglableKeyState(KeyType::SNEAK, allow_sneak_toggle,      player->control.sneak),
		isKeyDown(KeyType::ZOOM),
		isKeyDown(KeyType::DIG),
		isKeyDown(KeyType::PLACE),
		cam.camera_pitch,
		cam.camera_yaw,
		input->getJoystickSpeed(),
		input->getJoystickDirection()
	);
	control.setMovementFromKeys();

	// claude_input_lock: drop every key/joystick input, keep the look
	// angles the camera already has (a locked seat still turns when the
	// SERVER moves it, e.g. a harness teleport).
	if (claudeInputLocked()) {
		control = PlayerControl();
		control.pitch = cam.camera_pitch;
		control.yaw = cam.camera_yaw;
	}

	// autoforward if set: move at maximum speed
	if (player->getPlayerSettings().continuous_forward &&
			client->activeObjectsReceived() && !player->isDead()) {
		control.movement_speed = 1.0f;
		// sideways movement only
		float dx = std::sin(control.movement_direction);
		control.movement_direction = std::atan2(dx, 1.0f);
	}

	/* For touch, simulate holding down AUX1 (fast move) if the user has
	 * the fast_move setting toggled on. If there is an aux1 key defined for
	 * touch then its meaning is inverted (i.e. holding aux1 means walk and
	 * not fast)
	 */
	if (g_touchcontrols && m_touch_simulate_aux1) {
		control.aux1 = control.aux1 ^ true;
	}

	client->setPlayerControl(control);

	//tt.stop();
}

void Game::updatePauseState()
{
	bool was_paused = this->m_is_paused;
	this->m_is_paused = this->simple_singleplayer_mode && g_menumgr.pausesGame();

	if (!was_paused && this->m_is_paused) {
		this->pauseAnimation();
		this->sound_manager->pauseAll();
	} else if (was_paused && !this->m_is_paused) {
		this->resumeAnimation();
		this->sound_manager->resumeAll();
	}
}


inline void Game::step(f32 dtime)
{
	ZoneScoped;

	if (server) {
		float fps_max = !device->isWindowFocused() && simple_singleplayer_mode ?
				g_settings->getFloat("fps_max_unfocused") :
				g_settings->getFloat("fps_max");
		fps_max = std::max(fps_max, 1.0f);
		/*
		 * Unless you have a barebones game, running the server at more than 60Hz
		 * is hardly realistic and you're at the point of diminishing returns.
		 * fps_max is also not necessarily anywhere near the FPS actually achieved
		 * (also due to vsync).
		 */
		fps_max = std::min(fps_max, 60.0f);

		server->setStepSettings(Server::StepSettings{
				1.0f / fps_max,
				m_is_paused
			});

		server->step();
	}

	if (!m_is_paused)
		client->step(dtime);
}

static void pauseNodeAnimation(PausedNodesList &paused, scene::ISceneNode *node) {
	if (!node)
		return;
	for (auto &&child: node->getChildren())
		pauseNodeAnimation(paused, child);
	if (node->getType() != scene::ESNT_ANIMATED_MESH)
		return;
	auto animated_node = static_cast<scene::AnimatedMeshSceneNode *>(node);
	float speed = animated_node->getAnimationSpeed();
	if (!speed)
		return;
	paused.emplace_back(grab(animated_node), speed);
	animated_node->setAnimationSpeed(0.0f);
}

void Game::pauseAnimation()
{
	pauseNodeAnimation(paused_animated_nodes, smgr->getRootSceneNode());
}

void Game::resumeAnimation()
{
	for (auto &&pair: paused_animated_nodes)
		pair.first->setAnimationSpeed(pair.second);
	paused_animated_nodes.clear();
}

const ClientEventHandler Game::clientEventHandler[CLIENTEVENT_MAX] = {
	{&Game::handleClientEvent_None},
	{&Game::handleClientEvent_PlayerDamage},
	{&Game::handleClientEvent_PlayerForceMove},
	{&Game::handleClientEvent_DeathscreenLegacy},
	{&Game::handleClientEvent_ShowFormSpec},
	{&Game::handleClientEvent_ShowCSMFormSpec},
	{&Game::handleClientEvent_ShowPauseMenuFormSpec},
	{&Game::handleClientEvent_HandleParticleEvent},
	{&Game::handleClientEvent_HandleParticleEvent},
	{&Game::handleClientEvent_HandleParticleEvent},
	{&Game::handleClientEvent_HudAdd},
	{&Game::handleClientEvent_HudRemove},
	{&Game::handleClientEvent_HudChange},
	{&Game::handleClientEvent_SetSky},
	{&Game::handleClientEvent_SetSun},
	{&Game::handleClientEvent_SetMoon},
	{&Game::handleClientEvent_SetStars},
	{&Game::handleClientEvent_OverrideDayNightRatio},
	{&Game::handleClientEvent_CloudParams},
	{&Game::handleClientEvent_UpdateCamera},
};

void Game::handleClientEvent_None(ClientEvent *event, CameraOrientation *cam)
{
	FATAL_ERROR("ClientEvent type None received");
}

void Game::handleClientEvent_PlayerDamage(ClientEvent *event, CameraOrientation *cam)
{
	if (client->modsLoaded())
		client->getScript()->on_damage_taken(event->player_damage.amount);

	if (!event->player_damage.effect)
		return;

	// Damage flash and hurt tilt are not used at death
	if (client->getHP() > 0) {
		LocalPlayer *player = client->getEnv().getLocalPlayer();

		f32 hp_max = player->getCAO() ?
			player->getCAO()->getProperties().hp_max : PLAYER_MAX_HP_DEFAULT;
		f32 damage_ratio = event->player_damage.amount / hp_max;

		if (g_settings->getBool("hurt_flash_enabled")) {
			runData.damage_flash += 95.0f + 64.f * damage_ratio;
			runData.damage_flash = MYMIN(runData.damage_flash, 127.0f);
		}

		player->hurt_tilt_timer = 1.5f;
		player->hurt_tilt_strength =
			rangelim(damage_ratio * 5.0f, 1.0f, 4.0f);
	}

	// Play damage sound
	client->getEventManager()->put(new SimpleTriggerEvent(MtEvent::PLAYER_DAMAGE));
}

void Game::handleClientEvent_PlayerForceMove(ClientEvent *event, CameraOrientation *cam)
{
	cam->camera_yaw = event->player_force_move.yaw;
	cam->camera_pitch = event->player_force_move.pitch;
}

void Game::handleClientEvent_DeathscreenLegacy(ClientEvent *event, CameraOrientation *cam)
{
	m_game_formspec.showDeathFormspecLegacy();
}

void Game::handleClientEvent_ShowFormSpec(ClientEvent *event, CameraOrientation *cam)
{
	auto &fs = event->show_formspec;

	if (fs.formname->empty() && !fs.formspec->empty()) {
		m_game_formspec.showPlayerInventory(fs.formspec);
	} else {
		m_game_formspec.showFormSpec(*fs.formspec, *fs.formname);
	}

	delete fs.formspec;
	delete fs.formname;
}

void Game::handleClientEvent_ShowCSMFormSpec(ClientEvent *event, CameraOrientation *cam)
{
	m_game_formspec.showCSMFormSpec(*event->show_formspec.formspec,
		*event->show_formspec.formname);

	delete event->show_formspec.formspec;
	delete event->show_formspec.formname;
}

void Game::handleClientEvent_ShowPauseMenuFormSpec(ClientEvent *event, CameraOrientation *cam)
{
	m_game_formspec.showPauseMenuFormSpec(*event->show_formspec.formspec,
		*event->show_formspec.formname);

	delete event->show_formspec.formspec;
	delete event->show_formspec.formname;
}

void Game::handleClientEvent_HandleParticleEvent(ClientEvent *event,
		CameraOrientation *cam)
{
	LocalPlayer *player = client->getEnv().getLocalPlayer();
	client->getParticleManager()->handleParticleEvent(event, client, player);
}

void Game::handleClientEvent_HudAdd(ClientEvent *event, CameraOrientation *cam)
{
	LocalPlayer *player = client->getEnv().getLocalPlayer();

	u32 server_id = event->hudadd->server_id;
	// ignore if we already have a HUD with that ID
	auto i = m_hud_server_to_client.find(server_id);
	if (i != m_hud_server_to_client.end()) {
		delete event->hudadd;
		return;
	}

	HudElement *e = new HudElement;
	e->type   = static_cast<HudElementType>(event->hudadd->type);
	e->pos    = event->hudadd->pos;
	e->name   = event->hudadd->name;
	e->scale  = event->hudadd->scale;
	e->text   = event->hudadd->text;
	e->number = event->hudadd->number;
	e->item   = event->hudadd->item;
	e->dir    = event->hudadd->dir;
	e->align  = event->hudadd->align;
	e->offset = event->hudadd->offset;
	e->world_pos = event->hudadd->world_pos;
	e->size      = v2f::from(event->hudadd->size);
	e->z_index   = event->hudadd->z_index;
	e->text2     = event->hudadd->text2;
	e->style     = event->hudadd->style;
	m_hud_server_to_client[server_id] = player->addHud(e);

	delete event->hudadd;
}

void Game::handleClientEvent_HudRemove(ClientEvent *event, CameraOrientation *cam)
{
	LocalPlayer *player = client->getEnv().getLocalPlayer();

	auto i = m_hud_server_to_client.find(event->hudrm.id);
	if (i != m_hud_server_to_client.end()) {
		HudElement *e = player->removeHud(i->second);
		delete e;
		m_hud_server_to_client.erase(i);
	}

}

void Game::handleClientEvent_HudChange(ClientEvent *event, CameraOrientation *cam)
{
	LocalPlayer *player = client->getEnv().getLocalPlayer();

	HudElement *e = nullptr;

	auto i = m_hud_server_to_client.find(event->hudchange->id);
	if (i != m_hud_server_to_client.end()) {
		e = player->getHud(i->second);
	}

	if (e == nullptr) {
		delete event->hudchange;
		return;
	}

#define CASE_SET(statval, prop, dataprop) \
	case statval: \
		e->prop = event->hudchange->dataprop; \
		break

	switch (event->hudchange->stat) {
		CASE_SET(HUD_STAT_POS, pos, v2fdata);

		CASE_SET(HUD_STAT_NAME, name, sdata);

		CASE_SET(HUD_STAT_SCALE, scale, v2fdata);

		CASE_SET(HUD_STAT_TEXT, text, sdata);

		CASE_SET(HUD_STAT_NUMBER, number, data);

		CASE_SET(HUD_STAT_ITEM, item, data);

		CASE_SET(HUD_STAT_DIR, dir, data);

		CASE_SET(HUD_STAT_ALIGN, align, v2fdata);

		CASE_SET(HUD_STAT_OFFSET, offset, v2fdata);

		CASE_SET(HUD_STAT_WORLD_POS, world_pos, v3fdata);

		CASE_SET(HUD_STAT_SIZE, size, v2fdata);

		CASE_SET(HUD_STAT_Z_INDEX, z_index, data);

		CASE_SET(HUD_STAT_TEXT2, text2, sdata);

		CASE_SET(HUD_STAT_STYLE, style, data);

		case HudElementStat_END:
			break;
	}

#undef CASE_SET

	delete event->hudchange;
}

void Game::handleClientEvent_SetSky(ClientEvent *event, CameraOrientation *cam)
{
	sky->setVisible(false);
	// Whether clouds are visible in front of a custom skybox.
	sky->setCloudsEnabled(event->set_sky->clouds);

	// Clear the old textures out in case we switch rendering type.
	sky->clearSkyboxTextures();
	// Handle according to type
	if (event->set_sky->type == "regular") {
		// Shows the mesh skybox
		sky->setVisible(true);
		// Update mesh based skybox colours if applicable.
		sky->setSkyColors(event->set_sky->sky_color);
		sky->setHorizonTint(
			event->set_sky->fog_sun_tint,
			event->set_sky->fog_moon_tint,
			event->set_sky->fog_tint_type
		);
	} else if (event->set_sky->type == "skybox" &&
			event->set_sky->textures.size() == 6) {
		// Disable the dynamic mesh skybox:
		sky->setVisible(false);
		// Set fog colors:
		sky->setFallbackBgColor(event->set_sky->bgcolor);
		// Set sunrise and sunset fog tinting:
		sky->setHorizonTint(
			event->set_sky->fog_sun_tint,
			event->set_sky->fog_moon_tint,
			event->set_sky->fog_tint_type
		);
		// Add textures to skybox.
		for (int i = 0; i < 6; i++)
			sky->addTextureToSkybox(event->set_sky->textures[i], i, texture_src);
	} else {
		// Handle everything else as plain color.
		if (event->set_sky->type != "plain")
			infostream << "Unknown sky type: "
				<< (event->set_sky->type) << std::endl;
		sky->setVisible(false);
		sky->setFallbackBgColor(event->set_sky->bgcolor);
		// Disable directional sun/moon tinting on plain or invalid skyboxes.
		sky->setHorizonTint(
			event->set_sky->bgcolor,
			event->set_sky->bgcolor,
			"custom"
		);
	}

	// Orbit Tilt:
	sky->setBodyOrbitTilt(event->set_sky->body_orbit_tilt);

	// fog
	// do not override a potentially smaller client setting.
	sky->setFogDistance(event->set_sky->fog_distance);

	// if the fog distance is reset, switch back to the client's viewing_range
	if (event->set_sky->fog_distance < 0)
		draw_control->wanted_range = g_settings->getS16("viewing_range");

	if (event->set_sky->fog_start >= 0)
		sky->setFogStart(rangelim(event->set_sky->fog_start, 0.0f, 0.99f));
	else
		sky->setFogStart(rangelim(g_settings->getFloat("fog_start"), 0.0f, 0.99f));

	sky->setFogColor(event->set_sky->fog_color);

	sky->setAutoCaveBrightness(event->set_sky->auto_dim_skybox);

	delete event->set_sky;
}

void Game::handleClientEvent_SetSun(ClientEvent *event, CameraOrientation *cam)
{
	sky->setSunVisible(event->sun_params->visible);
	sky->setSunTexture(event->sun_params->texture,
		event->sun_params->tonemap, texture_src);
	sky->setSunScale(event->sun_params->scale);
	sky->setSunriseVisible(event->sun_params->sunrise_visible);
	sky->setSunriseTexture(event->sun_params->sunrise, texture_src);
	delete event->sun_params;
}

void Game::handleClientEvent_SetMoon(ClientEvent *event, CameraOrientation *cam)
{
	sky->setMoonVisible(event->moon_params->visible);
	sky->setMoonTexture(event->moon_params->texture,
		event->moon_params->tonemap, texture_src);
	sky->setMoonScale(event->moon_params->scale);
	delete event->moon_params;
}

void Game::handleClientEvent_SetStars(ClientEvent *event, CameraOrientation *cam)
{
	sky->setStarsVisible(event->star_params->visible);
	sky->setStarCount(event->star_params->count);
	sky->setStarColor(event->star_params->starcolor);
	sky->setStarScale(event->star_params->scale);
	sky->setStarDayOpacity(event->star_params->day_opacity);
	sky->setStarSeed(event->star_params->star_seed);
	delete event->star_params;
}

void Game::handleClientEvent_OverrideDayNightRatio(ClientEvent *event,
		CameraOrientation *cam)
{
	client->getEnv().setDayNightRatioOverride(
		event->override_day_night_ratio.do_override,
		event->override_day_night_ratio.ratio_f * 1000.0f);
}

void Game::handleClientEvent_CloudParams(ClientEvent *event, CameraOrientation *cam)
{
	clouds->setDensity(event->cloud_params.density);
	clouds->setColorBright(video::SColor(event->cloud_params.color_bright));
	clouds->setColorAmbient(video::SColor(event->cloud_params.color_ambient));
	clouds->setColorShadow(video::SColor(event->cloud_params.color_shadow));
	clouds->setHeight(event->cloud_params.height);
	clouds->setThickness(event->cloud_params.thickness);
	clouds->setSpeed(v2f(event->cloud_params.speed_x, event->cloud_params.speed_y));
}

void Game::handleClientEvent_UpdateCamera(ClientEvent *event, CameraOrientation *cam)
{
	// no parameters to update here, this just makes sure the camera is in the
	// state it should be after something was changed.
	updateCameraMode();
}

void Game::processClientEvents(CameraOrientation *cam)
{
	while (client->hasClientEvents()) {
		std::unique_ptr<ClientEvent> event(client->getClientEvent());
		FATAL_ERROR_IF(event->type >= CLIENTEVENT_MAX, "Invalid clientevent type");
		const ClientEventHandler& evHandler = clientEventHandler[event->type];
		(this->*evHandler.handler)(event.get(), cam);
	}
}

void Game::updateChat(f32 dtime)
{
	auto color_for = [](LogLevel level) -> const char* {
		switch (level) {
		case LL_ERROR  : return "\x1b(c@#F00)"; // red
		case LL_WARNING: return "\x1b(c@#EE0)"; // yellow
		case LL_INFO   : return "\x1b(c@#BBB)"; // grey
		case LL_VERBOSE: return "\x1b(c@#888)"; // dark grey
		case LL_TRACE  : return "\x1b(c@#888)"; // dark grey
		default        : return "";
		}
	};

	// Get new messages from error log buffer
	std::vector<LogEntry> entries = m_chat_log_buf.take();
	for (const auto& entry : entries) {
		std::string line;
		line.append(color_for(entry.level)).append(entry.combined);
		chat_backend->addMessage(L"", utf8_to_wide(line));
	}

	// Get new messages from client
	std::wstring message;
	while (client->getChatMessage(message)) {
		chat_backend->addUnparsedMessage(message);
	}

	// Remove old messages
	chat_backend->step(dtime);

	// Display all messages in a static text element
	auto &buf = chat_backend->getRecentBuffer();
	if (buf.getLinesModified()) {
		buf.resetLinesModified();
		m_game_ui->setChatText(chat_backend->getRecentChat(), buf.getLineCount());
	}

	// Make sure that the size is still correct
	m_game_ui->updateChatSize();
}

void Game::updateCamera(f32 dtime)
{
	ClientEnvironment &env = client->getEnv();
	LocalPlayer *player = env.getLocalPlayer();

	// For interaction purposes, get info about the held item
	ItemStack playeritem, hand;
	{
		ItemStack selected;
		playeritem = player->getWieldedItem(&selected, &hand);
	}

	ToolCapabilities playeritem_toolcap =
		playeritem.getToolCapabilities(itemdef_manager, &hand);

	float full_punch_interval = playeritem_toolcap.full_punch_interval;
	float tool_reload_ratio = runData.time_from_last_punch / full_punch_interval;

	tool_reload_ratio = std::min(tool_reload_ratio, 1.0f);
	camera->update(player, dtime, tool_reload_ratio);
	camera->step(dtime);

	if (!m_flags.disable_camera_update) {
		client->getEnv().getClientMap().updateCamera(camera->getPosition(),
			camera->getDirection(), camera->getFovMax(), camera->getOffset(),
			player->light_color);
	}
}

void Game::updateCameraMode()
{
	LocalPlayer *player = client->getEnv().getLocalPlayer();

	// Obey server choice
	if (player->allowed_camera_mode != CAMERA_MODE_ANY)
		camera->setCameraMode(player->allowed_camera_mode);

	GenericCAO *playercao = player->getCAO();
	if (playercao) {
		// Make the player visible depending on camera mode.
		playercao->updateMeshCulling();
		playercao->setChildrenVisible(camera->getCameraMode() > CAMERA_MODE_FIRST);
	}
}

void Game::updateCameraOffset()
{
	ClientEnvironment &env = client->getEnv();

	v3s16 old_camera_offset = camera->getOffset();

	camera->updateOffset();

	v3s16 camera_offset = camera->getOffset();

	m_camera_offset_changed = camera_offset != old_camera_offset;
	if (!m_camera_offset_changed)
		return;

	if (!m_flags.disable_camera_update) {
		auto *shadow = RenderingEngine::get_shadow_renderer();
		if (shadow) {
			shadow->getDirectionalLight().updateCameraOffset(camera);
			// FIXME: I bet we can be smarter about this and don't need to redraw
			// the shadow map at all, but this is for someone else to figure out.
			if (!g_settings->getFlag("performance_tradeoffs"))
				shadow->setForceUpdateShadowMap();
		}

		env.getClientMap().updateCamera(camera->getPosition(),
			camera->getDirection(), camera->getFovMax(), camera_offset,
			env.getLocalPlayer()->light_color);

		env.updateCameraOffset(camera_offset);
		clouds->updateCameraOffset(camera_offset);
	}
}

void Game::updateSound(f32 dtime)
{
	// Update sound listener
	LocalPlayer *player = client->getEnv().getLocalPlayer();
	ClientActiveObject *parent = player->getParent();
	v3s16 camera_offset = camera->getOffset();
	sound_manager->updateListener(
			(1.0f/BS) * camera->getCameraNode()->getPosition()
					+ intToFloat(camera_offset, 1.0f),
			(1.0f/BS) * (parent ? parent->getVelocity() : player->getSpeed()),
			camera->getDirection(),
			camera->getCameraNode()->getUpVector());

	sound_volume_control(sound_manager.get(), device->isWindowActive());

	// Update sound maker
	ClientMap &map = client->getEnv().getClientMap();
	MapNode n = map.getNode(player->getFootstepNodePos());
	soundmaker->update(dtime, player->makes_footstep_sound,
			nodedef_manager->get(n).sound_footstep);
}


void Game::processPlayerInteraction(f32 dtime, bool show_hud)
{
	LocalPlayer *player = client->getEnv().getLocalPlayer();

	const v3f camera_direction = camera->getDirection();
	const v3s16 camera_offset  = camera->getOffset();

	/*
		Calculate what block is the crosshair pointing to
	*/

	ItemStack selected_item, hand_item;
	const ItemStack &tool_item = player->getWieldedItem(&selected_item, &hand_item);

	const ItemDefinition &selected_def = tool_item.getDefinition(itemdef_manager);
	f32 d = getToolRange(tool_item, hand_item, itemdef_manager);

	core::line3d<f32> shootline;

	switch (camera->getCameraMode()) {
	case CAMERA_MODE_ANY:
	case CameraMode_END:
		assert(false);
		break;
	case CAMERA_MODE_FIRST:
		// Shoot from camera position, with bobbing
		shootline.start = camera->getPosition();
		break;
	case CAMERA_MODE_THIRD:
		// Shoot from player head, no bobbing
		shootline.start = camera->getHeadPosition();
		break;
	case CAMERA_MODE_THIRD_FRONT:
		shootline.start = camera->getHeadPosition();
		// prevent player pointing anything in front-view
		d = 0;
		break;
	}
	shootline.end = shootline.start + camera_direction * BS * d;

	if (isTouchShootlineUsed()) {
		shootline = g_touchcontrols->getShootline();
		// Scale shootline to the acual distance the player can reach
		shootline.end = shootline.start +
				shootline.getVector().normalize() * BS * d;
		shootline.start += intToFloat(camera_offset, BS);
		shootline.end += intToFloat(camera_offset, BS);
	}

	PointedThing pointed = updatePointedThing(shootline,
			selected_def.liquids_pointable,
			selected_def.pointabilities,
			!runData.btn_down_for_dig,
			camera_offset);

	if (pointed != runData.pointed_old)
		infostream << "Pointing at " << pointed.dump() << std::endl;

	if (g_touchcontrols) {
		auto mode = selected_def.touch_interaction.getMode(selected_def, pointed.type);
		g_touchcontrols->applyContextControls(mode);
		// applyContextControls may change dig/place input.
		// Update again so that TOSERVER_INTERACT packets have the correct controls set.
		player->control.dig = isKeyDown(KeyType::DIG);
		player->control.place = isKeyDown(KeyType::PLACE);
	}

	// Note that updating the selection mesh every frame is not particularly efficient,
	// but the halo rendering code is already inefficient so there's no point in optimizing it here
	hud->updateSelectionMesh(camera_offset);

	// Allow digging again if button is not pressed
	if (runData.digging_blocked && !isKeyDown(KeyType::DIG))
		runData.digging_blocked = false;

	/*
		Stop digging when
		- releasing dig button
		- pointing away from node
	*/
	if (runData.digging) {
		if (wasKeyReleased(KeyType::DIG)) {
			infostream << "Dig button released (stopped digging)" << std::endl;
			runData.digging = false;
		} else if (pointed != runData.pointed_old) {
			if (pointed.type == POINTEDTHING_NODE
					&& runData.pointed_old.type == POINTEDTHING_NODE
					&& pointed.node_undersurface
							== runData.pointed_old.node_undersurface) {
				// Still pointing to the same node, but a different face.
				// Don't reset.
			} else {
				infostream << "Pointing away from node (stopped digging)" << std::endl;
				runData.digging = false;
				hud->updateSelectionMesh(camera_offset);
			}
		}

		if (!runData.digging) {
			client->interact(INTERACT_STOP_DIGGING, runData.pointed_old);
			client->setCrack(-1, v3s16(0, 0, 0));
			runData.dig_time = 0.0;
		}
	} else if (runData.dig_instantly && wasKeyReleased(KeyType::DIG)) {
		// Remove e.g. torches faster when clicking instead of holding dig button
		runData.nodig_delay_timer = 0;
		runData.dig_instantly = false;
	}

	if (!runData.digging && runData.btn_down_for_dig && !isKeyDown(KeyType::DIG))
		runData.btn_down_for_dig = false;

	runData.punching = false;

	soundmaker->m_player_leftpunch_sound = SoundSpec();
	soundmaker->m_player_leftpunch_sound2 = pointed.type != POINTEDTHING_NOTHING ?
		selected_def.sound_use : selected_def.sound_use_air;

	// Prepare for repeating, unless we're not supposed to
	if (isKeyDown(KeyType::PLACE) && !g_settings->getBool("safe_dig_and_place"))
		runData.repeat_place_timer += dtime;
	else
		runData.repeat_place_timer = 0;

	if (selected_def.usable && isKeyDown(KeyType::DIG)) {
		if (wasKeyPressed(KeyType::DIG) && (!client->modsLoaded() ||
				!client->getScript()->on_item_use(selected_item, pointed)))
			client->interact(INTERACT_USE, pointed);
	} else if (pointed.type == POINTEDTHING_NODE) {
		handlePointingAtNode(pointed, selected_item, hand_item, dtime);
	} else if (pointed.type == POINTEDTHING_OBJECT) {
		v3f player_position  = player->getPosition();
		bool basic_debug_allowed = client->checkPrivilege("debug") || (player->hud_flags & HUD_FLAG_BASIC_DEBUG);
		handlePointingAtObject(pointed, tool_item, hand_item, player_position,
				m_game_ui->m_flags.show_basic_debug && basic_debug_allowed);
	} else if (isKeyDown(KeyType::DIG)) {
		// When button is held down in air, show continuous animation
		runData.punching = true;
		// Run callback even though item is not usable
		if (wasKeyPressed(KeyType::DIG) && client->modsLoaded())
			client->getScript()->on_item_use(selected_item, pointed);
	} else if (wasKeyPressed(KeyType::PLACE)) {
		handlePointingAtNothing(selected_item);
	}

	runData.pointed_old = pointed;

	if (runData.punching || wasKeyPressed(KeyType::DIG))
		camera->setDigging(0); // dig animation

	input->clearWasKeyPressed();
	input->clearWasKeyReleased();
	// Ensure DIG & PLACE are marked as handled
	wasKeyDown(KeyType::DIG);
	wasKeyDown(KeyType::PLACE);

	input->joystick.clearWasKeyPressed(KeyType::DIG);
	input->joystick.clearWasKeyPressed(KeyType::PLACE);

	input->joystick.clearWasKeyReleased(KeyType::DIG);
	input->joystick.clearWasKeyReleased(KeyType::PLACE);
}


PointedThing Game::updatePointedThing(
	const core::line3d<f32> &shootline,
	bool liquids_pointable,
	const std::optional<Pointabilities> &pointabilities,
	bool look_for_object,
	const v3s16 &camera_offset)
{
	std::vector<aabb3f> *selectionboxes = hud->getSelectionBoxes();
	selectionboxes->clear();
	hud->setSelectedFaceNormal(v3f());
	static thread_local const bool show_entity_selectionbox = g_settings->getBool(
		"show_entity_selectionbox");

	ClientEnvironment &env = client->getEnv();
	ClientMap &map = env.getClientMap();
	const NodeDefManager *nodedef = map.getNodeDefManager();

	runData.selected_object = NULL;
	hud->pointing_at_object = false;

	RaycastState s(shootline, look_for_object, liquids_pointable, pointabilities);
	PointedThing result;
	env.continueRaycast(&s, &result);
	if (result.type == POINTEDTHING_OBJECT) {
		hud->pointing_at_object = true;

		runData.selected_object = client->getEnv().getActiveObject(result.object_id);
		aabb3f selection_box{{0.0f, 0.0f, 0.0f}};
		if (show_entity_selectionbox && runData.selected_object->doShowSelectionBox() &&
				runData.selected_object->getSelectionBox(&selection_box)) {
			v3f pos = runData.selected_object->getPosition();
			selectionboxes->push_back(selection_box);
			hud->setSelectionPos(pos, camera_offset);
			GenericCAO* gcao = dynamic_cast<GenericCAO*>(runData.selected_object);
			if (gcao != nullptr && gcao->getProperties().rotate_selectionbox)
				hud->setSelectionRotationRadians(gcao->getSceneNode()
						->getAbsoluteTransformation().getRotationRadians());
			else
				hud->setSelectionRotationRadians(v3f());
		}
		hud->setSelectedFaceNormal(result.raw_intersection_normal);
	} else if (result.type == POINTEDTHING_NODE) {
		// Update selection boxes
		MapNode n = map.getNode(result.node_undersurface);
		std::vector<aabb3f> boxes;
		n.getSelectionBoxes(nodedef, &boxes,
			n.getNeighbors(result.node_undersurface, &map));

		f32 d = 0.002f * BS;
		for (aabb3f box : boxes) {
			box.MinEdge -= v3f(d, d, d);
			box.MaxEdge += v3f(d, d, d);
			selectionboxes->push_back(box);
		}
		hud->setSelectionPos(intToFloat(result.node_undersurface, BS),
			camera_offset);
		hud->setSelectionRotationRadians(v3f());
		hud->setSelectedFaceNormal(result.intersection_normal);
	}

	// Update selection mesh light level and vertex colors
	if (!selectionboxes->empty()) {
		v3f pf = hud->getSelectionPos();
		v3s16 p = floatToInt(pf, BS);

		// Get selection mesh light level
		MapNode n = map.getNode(p);
		u16 node_light = getInteriorLight(n, -1, nodedef);
		u16 light_level = node_light;

		for (const v3s16 &dir : g_6dirs) {
			n = map.getNode(p + dir);
			node_light = getInteriorLight(n, -1, nodedef);
			if (node_light > light_level)
				light_level = node_light;
		}

		u32 daynight_ratio = client->getEnv().getDayNightRatio();
		video::SColor c;
		final_color_blend(&c, light_level, daynight_ratio);

		// Modify final color a bit with time
		u32 timer = client->getEnv().getFrameTime() % 5000;
		float timerf = (float) (core::PI * ((timer / 2500.0) - 0.5));
		float sin_r = 0.08f * std::sin(timerf);
		float sin_g = 0.08f * std::sin(timerf + core::PI * 0.5f);
		float sin_b = 0.08f * std::sin(timerf + core::PI);
		c.setRed(core::clamp(core::round32(c.getRed() * (0.8 + sin_r)), 0, 255));
		c.setGreen(core::clamp(core::round32(c.getGreen() * (0.8 + sin_g)), 0, 255));
		c.setBlue(core::clamp(core::round32(c.getBlue() * (0.8 + sin_b)), 0, 255));

		// Set mesh final color
		hud->setSelectionMeshColor(c);
	}
	return result;
}


void Game::handlePointingAtNothing(const ItemStack &playerItem)
{
	infostream << "Attempted to place item while pointing at nothing" << std::endl;
	PointedThing fauxPointed;
	fauxPointed.type = POINTEDTHING_NOTHING;
	client->interact(INTERACT_ACTIVATE, fauxPointed);
}


void Game::handlePointingAtNode(const PointedThing &pointed,
	const ItemStack &selected_item, const ItemStack &hand_item, f32 dtime)
{
	v3s16 nodepos = pointed.node_undersurface;
	v3s16 neighborpos = pointed.node_abovesurface;

	/*
		Check information text of node
	*/

	ClientMap &map = client->getEnv().getClientMap();

	if (runData.nodig_delay_timer <= 0.0 && isKeyDown(KeyType::DIG)
			&& !runData.digging_blocked
			&& client->checkPrivilege("interact")) {
		handleDigging(pointed, nodepos, selected_item, hand_item, dtime);
	}

	// This should be done after digging handling
	NodeMetadata *meta = map.getNodeMetadata(nodepos);

	if (meta) {
		m_game_ui->setInfoText(unescape_translate(utf8_to_wide(
			meta->getString("infotext"))));
	} else {
		MapNode n = map.getNode(nodepos);

		if (nodedef_manager->get(n).name == "unknown") {
			m_game_ui->setInfoText(L"Unknown node");
		}
	}

	if ((wasKeyPressed(KeyType::PLACE) ||
			runData.repeat_place_timer >= m_repeat_place_time) &&
			client->checkPrivilege("interact")) {
		runData.repeat_place_timer = 0;
		infostream << "Place button pressed while looking at ground" << std::endl;

		// Placing animation (always shown for feedback)
		camera->setDigging(1);

		soundmaker->m_player_rightpunch_sound = SoundSpec();

		// If the wielded item has node placement prediction,
		// make that happen
		// And also set the sound and send the interact
		// But first check for meta formspec and rightclickable
		auto &def = selected_item.getDefinition(itemdef_manager);
		bool placed = nodePlacement(def, selected_item, nodepos, neighborpos,
			pointed, meta);

		if (placed && client->modsLoaded())
			client->getScript()->on_placenode(pointed, def);
	}
}

bool Game::nodePlacement(const ItemDefinition &selected_def,
	const ItemStack &selected_item, const v3s16 &nodepos, const v3s16 &neighborpos,
	const PointedThing &pointed, const NodeMetadata *meta)
{
	const auto &prediction = selected_def.node_placement_prediction;

	const NodeDefManager *nodedef = client->ndef();
	ClientMap &map = client->getEnv().getClientMap();
	MapNode node;
	bool is_valid_position;

	node = map.getNode(nodepos, &is_valid_position);
	if (!is_valid_position) {
		soundmaker->m_player_rightpunch_sound = selected_def.sound_place_failed;
		return false;
	}

	// formspec in meta
	if (meta && !meta->getString("formspec").empty() && !input->isRandom()
			&& !isKeyDown(KeyType::SNEAK)) {
		// on_rightclick callbacks are called anyway
		if (nodedef_manager->get(map.getNode(nodepos)).rightclickable)
			client->interact(INTERACT_PLACE, pointed);

		m_game_formspec.showNodeFormspec(meta->getString("formspec"), nodepos);
		return false;
	}

	// on_rightclick callback
	if (prediction.empty() || (nodedef->get(node).rightclickable &&
			!isKeyDown(KeyType::SNEAK))) {
		// Report to server
		client->interact(INTERACT_PLACE, pointed);
		return false;
	}

	verbosestream << "Node placement prediction for "
		<< selected_def.name << " is " << prediction << std::endl;
	v3s16 p = neighborpos;

	// Place inside node itself if buildable_to
	MapNode n_under = map.getNode(nodepos, &is_valid_position);
	if (is_valid_position) {
		if (nodedef->get(n_under).buildable_to) {
			p = nodepos;
		} else {
			node = map.getNode(p, &is_valid_position);
			if (is_valid_position && !nodedef->get(node).buildable_to) {
				soundmaker->m_player_rightpunch_sound = selected_def.sound_place_failed;
				// Report to server
				client->interact(INTERACT_PLACE, pointed);
				return false;
			}
		}
	}

	// Find id of predicted node
	content_t id;
	bool found = nodedef->getId(prediction, id);

	if (!found) {
		errorstream << "Node placement prediction failed for "
			<< selected_def.name << " (places " << prediction
			<< ") - Name not known" << std::endl;
		// Handle this as if prediction was empty
		// Report to server
		client->interact(INTERACT_PLACE, pointed);
		return false;
	}

	const ContentFeatures &predicted_f = nodedef->get(id);

	// Compare core.item_place_node() for what the server does with param2
	MapNode predicted_node(id, 0, 0);

	const auto place_param2 = selected_def.place_param2;

	if (place_param2) {
		predicted_node.setParam2(*place_param2);
	} else if (predicted_f.param_type_2 == CPT2_WALLMOUNTED ||
			predicted_f.param_type_2 == CPT2_COLORED_WALLMOUNTED) {
		v3s16 dir = nodepos - neighborpos;

		if (abs(dir.Y) > MYMAX(abs(dir.X), abs(dir.Z))) {
			// If you change this code, also change builtin/game/item.lua
			u8 predicted_param2 = dir.Y < 0 ? 1 : 0;
			if (selected_def.wallmounted_rotate_vertical) {
				bool rotate90 = false;
				v3f ppos = client->getEnv().getLocalPlayer()->getPosition() / BS;
				v3f pdir = v3f::from(neighborpos) - ppos;
				switch (predicted_f.drawtype) {
					case NDT_TORCHLIKE: {
						rotate90 = !((pdir.X < 0 && pdir.Z > 0) ||
								(pdir.X > 0 && pdir.Z < 0));
						if (dir.Y > 0) {
							rotate90 = !rotate90;
						}
						break;
					};
					case NDT_SIGNLIKE: {
						rotate90 = std::abs(pdir.X) < std::abs(pdir.Z);
						break;
					}
					default: {
						rotate90 = std::abs(pdir.X) > std::abs(pdir.Z);
						break;
					}
				}
				if (rotate90) {
					predicted_param2 += 6;
				}
			}
			predicted_node.setParam2(predicted_param2);
		} else if (abs(dir.X) > abs(dir.Z)) {
			predicted_node.setParam2(dir.X < 0 ? 3 : 2);
		} else {
			predicted_node.setParam2(dir.Z < 0 ? 5 : 4);
		}
	} else if (predicted_f.param_type_2 == CPT2_FACEDIR ||
			predicted_f.param_type_2 == CPT2_COLORED_FACEDIR ||
			predicted_f.param_type_2 == CPT2_4DIR ||
			predicted_f.param_type_2 == CPT2_COLORED_4DIR) {
		v3s16 dir = nodepos - floatToInt(client->getEnv().getLocalPlayer()->getPosition(), BS);

		if (abs(dir.X) > abs(dir.Z)) {
			predicted_node.setParam2(dir.X < 0 ? 3 : 1);
		} else {
			predicted_node.setParam2(dir.Z < 0 ? 2 : 0);
		}
	}

	// Check attachment if node is in group attached_node
	int an = itemgroup_get(predicted_f.groups, "attached_node");
	if (an != 0) {
		v3s16 pp;

		if (an == 3) {
			pp = p + v3s16(0, -1, 0);
		} else if (an == 4) {
			pp = p + v3s16(0, 1, 0);
		} else if (an == 2) {
			if (predicted_f.param_type_2 == CPT2_FACEDIR ||
					predicted_f.param_type_2 == CPT2_COLORED_FACEDIR ||
					predicted_f.param_type_2 == CPT2_4DIR ||
					predicted_f.param_type_2 == CPT2_COLORED_4DIR) {
				pp = p + facedir_dirs[predicted_node.getFaceDir(nodedef)];
			} else {
				pp = p;
			}
		} else if (predicted_f.param_type_2 == CPT2_WALLMOUNTED ||
				predicted_f.param_type_2 == CPT2_COLORED_WALLMOUNTED) {
			pp = p + predicted_node.getWallMountedDir(nodedef);
		} else {
			pp = p + v3s16(0, -1, 0);
		}

		if (!nodedef->get(map.getNode(pp)).walkable) {
			soundmaker->m_player_rightpunch_sound = selected_def.sound_place_failed;
			// Report to server
			client->interact(INTERACT_PLACE, pointed);
			return false;
		}
	}

	// Apply color
	if (!place_param2 && (predicted_f.param_type_2 == CPT2_COLOR
			|| predicted_f.param_type_2 == CPT2_COLORED_FACEDIR
			|| predicted_f.param_type_2 == CPT2_COLORED_4DIR
			|| predicted_f.param_type_2 == CPT2_COLORED_WALLMOUNTED)) {
		const auto &indexstr = selected_item.metadata.
			getString("palette_index", 0);
		if (!indexstr.empty()) {
			s32 index = mystoi(indexstr);
			if (predicted_f.param_type_2 == CPT2_COLOR) {
				predicted_node.setParam2(index);
			} else if (predicted_f.param_type_2 == CPT2_COLORED_WALLMOUNTED) {
				// param2 = pure palette index + other
				predicted_node.setParam2((index & 0xf8) | (predicted_node.getParam2() & 0x07));
			} else if (predicted_f.param_type_2 == CPT2_COLORED_FACEDIR) {
				// param2 = pure palette index + other
				predicted_node.setParam2((index & 0xe0) | (predicted_node.getParam2() & 0x1f));
			} else if (predicted_f.param_type_2 == CPT2_COLORED_4DIR) {
				// param2 = pure palette index + other
				predicted_node.setParam2((index & 0xfc) | (predicted_node.getParam2() & 0x03));
			}
		}
	}

	// Add node to client map
	try {
		LocalPlayer *player = client->getEnv().getLocalPlayer();

		// Don't place node when player would be inside new node
		// NOTE: This is to be eventually implemented by a mod as client-side Lua
		if (!predicted_f.walkable ||
				g_settings->getBool("enable_build_where_you_stand") ||
				(client->checkPrivilege("noclip") && g_settings->getBool("noclip")) ||
				(predicted_f.walkable &&
					neighborpos != player->getStandingNodePos() + v3s16(0, 1, 0) &&
					neighborpos != player->getStandingNodePos() + v3s16(0, 2, 0))) {
			// This triggers the required mesh update too
			client->addNode(p, predicted_node);
			// Report to server
			client->interact(INTERACT_PLACE, pointed);
			// A node is predicted, also play a sound
			soundmaker->m_player_rightpunch_sound = selected_def.sound_place;
			return true;
		} else {
			soundmaker->m_player_rightpunch_sound = selected_def.sound_place_failed;
			return false;
		}
	} catch (const InvalidPositionException &e) {
		errorstream << "Node placement prediction failed for "
			<< selected_def.name << " (places "
			<< prediction << ") - Position not loaded" << std::endl;
		soundmaker->m_player_rightpunch_sound = selected_def.sound_place_failed;
		return false;
	}
}

void Game::handlePointingAtObject(const PointedThing &pointed, const ItemStack &tool_item,
		const ItemStack &hand_item, const v3f &player_position, bool show_debug)
{
	std::wstring infotext = unescape_translate(
		utf8_to_wide(runData.selected_object->infoText()));

	if (show_debug) {
		if (!infotext.empty()) {
			infotext += L"\n";
		}
		infotext += utf8_to_wide(runData.selected_object->debugInfoText());
	}

	m_game_ui->setInfoText(infotext);

	if (isKeyDown(KeyType::DIG)) {
		bool do_punch = false;
		bool do_punch_damage = false;

		if (runData.object_hit_delay_timer <= 0.0) {
			do_punch = true;
			do_punch_damage = true;
			runData.object_hit_delay_timer = object_hit_delay;
		}

		if (wasKeyPressed(KeyType::DIG))
			do_punch = true;

		if (do_punch) {
			infostream << "Punched object" << std::endl;
			runData.punching = true;
			runData.nodig_delay_timer = std::max(0.15f, m_repeat_dig_time);
		}

		if (do_punch_damage) {
			// Report direct punch
			v3f objpos = runData.selected_object->getPosition();
			v3f dir = (objpos - player_position).normalize();

			bool disable_send = runData.selected_object->directReportPunch(
					dir, &tool_item, &hand_item, runData.time_from_last_punch);
			runData.time_from_last_punch = 0;

			if (!disable_send)
				client->interact(INTERACT_START_DIGGING, pointed);
		}
	} else if (wasKeyDown(KeyType::PLACE)) {
		infostream << "Pressed place button while pointing at object" << std::endl;
		client->interact(INTERACT_PLACE, pointed);  // place
	}
}


void Game::handleDigging(const PointedThing &pointed, const v3s16 &nodepos,
		const ItemStack &selected_item, const ItemStack &hand_item, f32 dtime)
{
	// See also: serverpackethandle.cpp, action == 2
	LocalPlayer *player = client->getEnv().getLocalPlayer();
	ClientMap &map = client->getEnv().getClientMap();
	MapNode n = map.getNode(nodepos);
	const auto &features = nodedef_manager->get(n);
	const ItemStack &tool_item = selected_item.name.empty() ? hand_item : selected_item;

	// NOTE: Similar piece of code exists on the server side for
	// cheat detection.
	// Get digging parameters
	DigParams params = getDigParams(features.groups,
			&tool_item.getToolCapabilities(itemdef_manager, &hand_item),
			tool_item.wear);

	// If can't dig, try hand
	if (!params.diggable) {
		params = getDigParams(features.groups,
				&hand_item.getToolCapabilities(itemdef_manager));
	}

	if (!params.diggable) {
		// I guess nobody will wait for this long
		runData.dig_time_complete = 10000000.0;
	} else {
		runData.dig_time_complete = params.time;

		client->getParticleManager()->addNodeParticle(player, nodepos, n);
	}

	if (!runData.digging) {
		infostream << "Started digging" << std::endl;
		runData.dig_instantly = runData.dig_time_complete == 0;
		if (client->modsLoaded() && client->getScript()->on_punchnode(nodepos, n))
			return;

		client->interact(INTERACT_START_DIGGING, pointed);
		runData.digging = true;
		runData.btn_down_for_dig = true;
	}

	if (!runData.dig_instantly) {
		runData.dig_index = (float)crack_animation_length
				* runData.dig_time
				/ runData.dig_time_complete;
	} else {
		// This is for e.g. torches
		runData.dig_index = crack_animation_length;
	}

	const auto &sound_dig = features.sound_dig;

	if (sound_dig.exists() && params.diggable) {
		if (sound_dig.name == "__group") {
			if (!params.main_group.empty()) {
				soundmaker->m_player_leftpunch_sound.gain = 0.5;
				soundmaker->m_player_leftpunch_sound.name =
						std::string("default_dig_") +
						params.main_group;
			}
		} else {
			soundmaker->m_player_leftpunch_sound = sound_dig;
		}
	}

	// Don't show cracks if not diggable
	if (runData.dig_time_complete >= 100000.0) {
	} else if (runData.dig_index < crack_animation_length) {
		client->setCrack(runData.dig_index, nodepos);
	} else {
		infostream << "Digging completed" << std::endl;
		client->setCrack(-1, v3s16(0, 0, 0));

		runData.dig_time = 0;
		runData.digging = false;
		// we successfully dug, now block it from repeating if we want to be safe
		if (g_settings->getBool("safe_dig_and_place"))
			runData.digging_blocked = true;

		runData.nodig_delay_timer =
				runData.dig_time_complete / (float)crack_animation_length;

		// We don't want a corresponding delay to very time consuming nodes
		// and nodes without digging time (e.g. torches) get a fixed delay.
		if (runData.nodig_delay_timer > 0.3f)
			runData.nodig_delay_timer = 0.3f;
		else if (runData.dig_instantly)
			runData.nodig_delay_timer = 0.15f;

		// Ensure that the delay between breaking nodes
		// (dig_time_complete + nodig_delay_timer) is at least the
		// value of the repeat_dig_time setting.
		runData.nodig_delay_timer = std::max(runData.nodig_delay_timer,
				m_repeat_dig_time - runData.dig_time_complete);

		if (client->modsLoaded() &&
				client->getScript()->on_dignode(nodepos, n)) {
			return;
		}

		if (features.node_dig_prediction == "air") {
			client->removeNode(nodepos);
		} else if (!features.node_dig_prediction.empty()) {
			content_t id;
			bool found = nodedef_manager->getId(features.node_dig_prediction, id);
			if (found)
				client->addNode(nodepos, id, true);
		}
		// implicit else: no prediction

		client->interact(INTERACT_DIGGING_COMPLETED, pointed);

		client->getParticleManager()->addDiggingParticles(player, nodepos, n);

		// Send event to trigger sound
		client->getEventManager()->put(new NodeDugEvent(nodepos, n));
	}

	if (runData.dig_time_complete < 100000.0) {
		runData.dig_time += dtime;
	} else {
		runData.dig_time = 0;
		client->setCrack(-1, nodepos);
	}

	camera->setDigging(0);  // Dig animation
}

void Game::updateFrame(ProfilerGraph *graph, RunStats *stats, f32 dtime,
		const CameraOrientation &cam)
{
	ZoneScoped;
	TimeTaker tt_update("Game::updateFrame()");
	LocalPlayer *player = client->getEnv().getLocalPlayer();

	/*
		Frame time
	*/

	client->getEnv().updateFrameTime(m_is_paused);

	/*
		Fog range
	*/

	if (sky->getFogDistance() >= 0) {
		draw_control->wanted_range = MYMIN(draw_control->wanted_range, sky->getFogDistance());
	}
	if (draw_control->range_all && sky->getFogDistance() < 0) {
		runData.fog_range = FOG_RANGE_ALL;
	} else {
		runData.fog_range = draw_control->wanted_range * BS;
	}

	/*
		Calculate general brightness
	*/
	u32 daynight_ratio = client->getEnv().getDayNightRatio();
	float time_brightness = decode_light_f((float)daynight_ratio / 1000.0f);
	float direct_brightness;
	bool sunlight_seen;

	// When in noclip mode force same sky brightness as above ground so you
	// can see properly
	bool noclip_fly = draw_control->allow_noclip &&
			m_cache_enable_free_move &&
			client->checkPrivilege("fly");
	if (!sky->getAutoCaveBrightness() || noclip_fly) {
		direct_brightness = time_brightness;
		sunlight_seen = true;
	} else {
		float old_brightness = sky->getBrightness();
		direct_brightness = client->getEnv().getClientMap()
				.getBackgroundBrightness(MYMIN(runData.fog_range * 1.2, 60 * BS),
						daynight_ratio, (int)(old_brightness * 255.5), &sunlight_seen)
				/ 255.0;
	}

	float time_of_day_smooth = runData.time_of_day_smooth;
	float time_of_day = client->getEnv().getTimeOfDayF();

	static const float maxsm = 0.05f;
	static const float todsm = 0.05f;

	if (std::fabs(time_of_day - time_of_day_smooth) > maxsm &&
			std::fabs(time_of_day - time_of_day_smooth + 1.0) > maxsm &&
			std::fabs(time_of_day - time_of_day_smooth - 1.0) > maxsm)
		time_of_day_smooth = time_of_day;

	if (time_of_day_smooth > 0.8 && time_of_day < 0.2)
		time_of_day_smooth = time_of_day_smooth * (1.0 - todsm)
				+ (time_of_day + 1.0) * todsm;
	else
		time_of_day_smooth = time_of_day_smooth * (1.0 - todsm)
				+ time_of_day * todsm;

	runData.time_of_day_smooth = time_of_day_smooth;

	sky->update(time_of_day_smooth, time_brightness, direct_brightness,
			sunlight_seen, camera->getCameraMode(), player->getYaw(),
			player->getPitch());

	/*
		Update clouds
	*/
	updateClouds(dtime);

	/*
		Update particles
	*/
	client->getParticleManager()->step(dtime);

	/*
		Damage camera tilt
	*/
	if (player->hurt_tilt_timer > 0.0f) {
		player->hurt_tilt_timer -= dtime * 6.0f;

		if (player->hurt_tilt_timer < 0.0f)
			player->hurt_tilt_strength = 0.0f;
	}

	/*
		Update minimap pos and rotation
	*/
	if (mapper && m_game_ui->m_flags.show_hud) {
		mapper->setPos(floatToInt(player->getPosition(), BS));
		mapper->setAngle(player->getYaw());
	}

	/*
		Get chat messages from client
	*/

	updateChat(dtime);

	/*
		Inventory
	*/

	if (player->getWieldIndex() != runData.new_playeritem)
		client->setPlayerItem(runData.new_playeritem);

	if (client->updateWieldedItem()) {
		// Update wielded tool
		ItemStack selected_item, hand_item;
		ItemStack &tool_item = player->getWieldedItem(&selected_item, &hand_item);

		bool skip_anim = client->consumeSkipNextWieldAnimation();
		camera->wield(tool_item, !skip_anim);
	}

	/*
		Update block draw list every 200ms or when camera direction has
		changed much
	*/
	runData.update_draw_list_timer += dtime;
	runData.touch_blocks_timer += dtime;

	constexpr float update_draw_list_delta = 0.2f;
	constexpr float touch_mapblock_delta = 4.0f;

	v3f camera_direction = camera->getDirection();

	// call only one of updateDrawList, touchMapBlocks, or updateShadow per frame
	// (the else-ifs below are intentional)
	if (runData.update_draw_list_timer >= update_draw_list_delta
			|| runData.update_draw_list_last_cam_dir.getDistanceFrom(camera_direction) > 0.2
			|| m_camera_offset_changed
			|| client->getEnv().getClientMap().needsUpdateDrawList()) {
		runData.update_draw_list_timer = 0;
		client->getEnv().getClientMap().updateDrawList();
		runData.update_draw_list_last_cam_dir = camera_direction;
	} else if (runData.touch_blocks_timer > touch_mapblock_delta) {
		client->getEnv().getClientMap().touchMapBlocks();
		runData.touch_blocks_timer = 0;
	} else if (RenderingEngine::get_shadow_renderer()) {
		updateShadows();
	}

	m_game_ui->update(*stats, client, draw_control, cam, runData.pointed_old,
			gui_chat_console.get(), dtime);

	m_game_formspec.update();

	/*
		==================== Drawing begins ====================
	*/
	if (device->isWindowVisible())
		drawScene(graph, stats);
	/*
		==================== End scene ====================
	*/

	// Damage flash is drawn in drawScene, but the timing update is done here to
	// keep dtime out of the drawing code.
	if (runData.damage_flash > 0.0f) {
		runData.damage_flash -= 384.0f * dtime;
	}

	g_profiler->avg("Game::updateFrame(): update frame [ms]", tt_update.stop(true));
}

void Game::updateClouds(float dtime)
{
	if (this->sky->getCloudsVisible()) {
		this->clouds->setVisible(true);
		this->clouds->step(dtime);
		// this->camera->getPosition is not enough for third-person camera.
		v3f camera_node_position = this->camera->getCameraNode()->getPosition();
		v3s16 camera_offset      = this->camera->getOffset();
		camera_node_position.X   = camera_node_position.X + camera_offset.X * BS;
		camera_node_position.Y   = camera_node_position.Y + camera_offset.Y * BS;
		camera_node_position.Z   = camera_node_position.Z + camera_offset.Z * BS;
		this->clouds->update(camera_node_position, this->sky->getCloudColor());
		if (this->clouds->isCameraInsideCloud() && this->fogEnabled()) {
			// If camera is inside cloud and fog is enabled, use cloud's colors as sky colors.
			video::SColor clouds_dark = this->clouds->getColor().getInterpolated(
					video::SColor(255, 0, 0, 0), 0.9);
			this->sky->overrideColors(clouds_dark, this->clouds->getColor());
			this->sky->setInClouds(true);
			this->runData.fog_range = std::fmin(this->runData.fog_range * 0.5f, 32.0f * BS);
			// Clouds are not drawn in this case.
			this->clouds->setVisible(false);
		}
	} else {
		this->clouds->setVisible(false);
	}
}

/* Log times and stuff for visualization */
inline void Game::updateProfilerGraphs(ProfilerGraph *graph)
{
	Profiler::GraphValues values;
	g_profiler->graphPop(values);
	graph->put(values);
}

/****************************************************************************
 * Shadows
 *****************************************************************************/
void Game::updateShadows()
{
	ShadowRenderer *shadow = RenderingEngine::get_shadow_renderer();
	if (!shadow)
		return;

	float in_timeofday = std::fmod(runData.time_of_day_smooth, 1.0f);

	const auto &lighting = client->getEnv().getLocalPlayer()->getLighting();
	shadow->setShadowTint(lighting.shadow_tint);

	const float offset_constant = 10000.0f;

	v3f light;
	if (lighting.shadow_direction.getLengthSQ() > 0.0f) {
		// Custom shadow direction: bypass sun/moon visibility check
		shadow->setShadowIntensity(lighting.shadow_intensity);
		light = lighting.shadow_direction;
	} else {
		float timeoftheday = getWickedTimeOfDay(in_timeofday);
		bool is_day = timeoftheday > 0.25f && timeoftheday < 0.75f;
		bool is_shadow_visible = is_day ? sky->getSunVisible() : sky->getMoonVisible();
		shadow->setShadowIntensity(is_shadow_visible ? lighting.shadow_intensity : 0.0f);
		light = is_day ? sky->getSunDirection() : sky->getMoonDirection();
	}

	v3f sun_pos = light * offset_constant;
	shadow->getDirectionalLight().setDirection(sun_pos);
	shadow->setTimeOfDay(in_timeofday);

	shadow->getDirectionalLight().updateFrustum(camera, client);
}

void Game::drawScene(ProfilerGraph *graph, RunStats *stats)
{
	ZoneScoped;

	const video::SColor fog_color = this->sky->getFogColor();
	const video::SColor sky_color = this->sky->getSkyColor();

	/*
		Fog
	*/
	if (this->fogEnabled()) {
		this->driver->setFog(
				fog_color,
				video::EFT_FOG_LINEAR,
				this->runData.fog_range * this->sky->getFogStart(),
				this->runData.fog_range * 1.0f,
				0.f, // unused
				false, // pixel fog
				true // range fog
		);
	} else {
		this->driver->setFog(
				fog_color,
				video::EFT_FOG_LINEAR,
				FOG_RANGE_ALL,
				FOG_RANGE_ALL + 100 * BS,
				0.f, // unused
				false, // pixel fog
				false // range fog
		);
	}

	/*
		Drawing
	*/
	TimeTaker tt_draw("Draw scene", nullptr, PRECISION_MICRO);
	this->driver->beginScene(true, true, sky_color);

	const LocalPlayer *player = this->client->getEnv().getLocalPlayer();
	bool draw_wield_tool = (this->m_game_ui->m_flags.show_hud &&
			(player->hud_flags & HUD_FLAG_WIELDITEM_VISIBLE) &&
			(this->camera->getCameraMode() == CAMERA_MODE_FIRST));
	bool draw_crosshair = (
			(player->hud_flags & HUD_FLAG_CROSSHAIR_VISIBLE) &&
			(this->camera->getCameraMode() != CAMERA_MODE_THIRD_FRONT));

	if (isTouchShootlineUsed())
		draw_crosshair = false;

	this->m_rendering_engine->draw_scene(sky_color, this->m_game_ui->m_flags.show_hud,
			draw_wield_tool, draw_crosshair);

	/*
		Profiler graph
	*/
	v2u32 screensize = this->driver->getScreenSize();

	if (this->m_game_ui->m_flags.show_profiler_graph) {
		auto font = g_fontengine->getFont(
			g_fontengine->getDefaultFontSize() * 0.9f, FM_Mono);
		graph->draw(10, screensize.Y - 10, driver, font);
	}

	/*
		Damage flash
	*/
	if (this->runData.damage_flash > 0.0f) {
		video::SColor color(this->runData.damage_flash, 180, 0, 0);
		this->driver->draw2DRectangle(color,
					core::rect<s32>(0, 0, screensize.X, screensize.Y),
					NULL);
	}

	this->driver->endScene();

	stats->drawtime = tt_draw.stop(true);
	g_profiler->graphAdd("Draw scene [us]", stats->drawtime);

}

/****************************************************************************
 Misc
 ****************************************************************************/

void Game::showOverlayMessage(const char *msg, float dtime, int percent, float *indef_pos)
{
	m_rendering_engine->draw_load_screen(wstrgettext(msg), guienv, texture_src,
			dtime, percent, indef_pos);
}

void Game::settingChangedCallback(const std::string &setting_name, void *data)
{
	((Game *)data)->readSettings();
}

void Game::readSettings()
{
	LogLevel chat_log_level = Logger::stringToLevel(g_settings->get("chat_log_level"));
	if (chat_log_level == LL_MAX) {
		warningstream << "Supplied unrecognized chat_log_level; showing none." << std::endl;
		chat_log_level = LL_NONE;
	}
	m_chat_log_buf.setLogLevel(chat_log_level);

	m_cache_doubletap_jump               = g_settings->getBool("doubletap_jump");
	m_cache_toggle_sneak_key             = g_settings->getBool("toggle_sneak_key");
	m_cache_toggle_aux1_key              = g_settings->getBool("toggle_aux1_key");
	m_cache_enable_joysticks             = g_settings->getBool("enable_joysticks");
	m_cache_enable_fog                   = g_settings->getBool("enable_fog");
	m_cache_mouse_sensitivity            = g_settings->getFloat("mouse_sensitivity", 0.001f, 10.0f);
	m_cache_keyboard_camera_speed        = g_settings->getFloat("keyboard_camera_speed", 0.001f, 720.0f);
	m_cache_joystick_frustum_sensitivity = std::max(g_settings->getFloat("joystick_frustum_sensitivity"), 0.001f);
	m_repeat_place_time                  = g_settings->getFloat("repeat_place_time", 0.16f, 2.0f);
	m_repeat_dig_time                    = g_settings->getFloat("repeat_dig_time", 0.0f, 2.0f);

	m_cache_enable_noclip                = g_settings->getBool("noclip");
	m_cache_enable_free_move             = g_settings->getBool("free_move");

	m_cache_cam_smoothing = 0;
	if (g_settings->getBool("cinematic"))
		m_cache_cam_smoothing = g_settings->getFloat("cinematic_camera_smoothing");
	else
		m_cache_cam_smoothing = g_settings->getFloat("camera_smoothing");

	m_cache_cam_smoothing = std::max(0.0f, m_cache_cam_smoothing);
	m_cache_mouse_sensitivity = rangelim(m_cache_mouse_sensitivity, 0.001, 100.0);

	m_invert_mouse = g_settings->getBool("invert_mouse");
	m_enable_hotbar_mouse_wheel = g_settings->getBool("enable_hotbar_mouse_wheel");
	m_invert_hotbar_mouse_wheel = g_settings->getBool("invert_hotbar_mouse_wheel");

	m_does_lost_focus_pause_game = g_settings->getBool("pause_on_lost_focus");
}

/****************************************************************************/
/****************************************************************************
 extern function for launching the game
 ****************************************************************************/
/****************************************************************************/

void the_game(volatile std::sig_atomic_t *kill,
		InputHandler *input,
		RenderingEngine *rendering_engine,
		const GameStartData &start_data,
		std::string &error_message,
		ChatBackend &chat_backend,
		bool *reconnect_requested) // Used for local game
{
	Game game;

	try {

		if (game.startup(kill, input, rendering_engine, start_data,
				error_message, reconnect_requested, &chat_backend)) {
			game.run();
		}

	} catch (SerializationError &e) {
		const std::string ver_err = fmtgettext("The server is probably running a different version of %s.", PROJECT_NAME_C);
		error_message = strgettext("A serialization error occurred:") +"\n"
				+ e.what() + "\n\n" + ver_err;
		errorstream << error_message << std::endl;
	} catch (ServerError &e) {
		error_message = e.what();
		errorstream << "ServerError: " << error_message << std::endl;
	} catch (ModError &e) {
		// DO NOT TRANSLATE the `ModError`, it's used by `ui.lua`
		error_message = std::string("ModError: ") + e.what() +
				strgettext("\nCheck debug.txt for details.");
		errorstream << error_message << std::endl;
	} catch (con::PeerNotFoundException &e) {
		error_message = gettext("Connection error (timed out?)");
		errorstream << error_message << std::endl;
	} catch (ShaderException &e) {
		error_message = e.what();
		errorstream << error_message << std::endl;
	}

	game.shutdown();
}
