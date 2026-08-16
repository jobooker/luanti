// Luanti
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2013 celeron55, Perttu Ahola <celeron55@gmail.com>

#pragma once

#include "irrlichttypes.h"
#include "config.h"
#include <csignal>
#include <string>

#if !IS_CLIENT_BUILD
#error Do not include in server builds
#endif

class InputHandler;
class ChatBackend;
class Settings;
class RenderingEngine;
struct GameStartData;

struct Jitter {
	f32 max, min, avg, counter, max_sample, min_sample, max_fraction;
};

struct RunStats {
	u64 drawtime; // (us)

	Jitter dtime_jitter, busy_time_jitter;
};

struct CameraOrientation {
	f32 camera_yaw;    // "right/left"
	f32 camera_pitch;  // "up/down"
};

#define GAME_FALLBACK_TIMEOUT 1.8f
#define GAME_CONNECTION_TIMEOUT 10.0f

void the_game(volatile std::sig_atomic_t *kill,
		InputHandler *input,
		RenderingEngine *rendering_engine,
		const GameStartData &start_data,
		std::string &error_message,
		ChatBackend &chat_backend,
		bool *reconnect_requested);

// Trace accumulation stats for the F5 debug overlay (gameui.cpp); the
// backing state (g_claude_grid) is file-local to game.cpp.
void claudeGetTraceStats(float *still_frames, float *accum_alpha);

// claude_grid: report and warn (once per name, per process) about legacy
// claude_volume_* setting names left over from the 2026-08-16 "volume" ->
// "grid" rename. Returns how many legacy names `src` carries, whatever was
// already logged. A renamed setting whose old name is silently ignored is a
// bug, not a rename — see spec/environment-laws.md.
// Exposed for src/unittest/test_claude_grid_settings.cpp.
int claudeWarnRenamedSettings(const Settings *src, const char *source);
