// claude_lod: far-terrain cascade data for the traced renderer.
// Phase 1 of the cascaded clipmap (see vault: luanti-lod-clipmap-plan.md,
// ADR-0004): per-MapBlock summaries folded at receive time, and an 8 m
// cascade (128^3 cells, +/-512 m) built from them on demand.
#pragma once

#include "irrlichttypes_bloated.h"
#include <string>
#include <vector>

class Client;
class MapBlock;

namespace claude_lod
{

// Fold one received MapBlock into the summary cache: 4^3 subcells of
// 4^3 nodes each, storing occupancy / water counts and summed minimap
// color. ~256 B per block, computed once per receive (microseconds —
// direct array reads, no map lookups). Called from the main thread
// (packet handling); the cache is mutex-guarded anyway so a future
// warm-scan thread can feed it too.
// fill_only: keep an existing summary (a block the server sent is fresher
// than the copy in the world file)
void summarizeBlock(Client *client, MapBlock *block, bool fill_only = false);

// Build a summary-fed cascade level: 128^3 cells of cell_nodes (4 or 8)
// covering origin_nodes + 128*cell_nodes. rgba: 128^3 * 4 (rgb =
// occupancy-weighted average color, a = class: 0 air, 100 water, 255
// solid). coarse: 32^3 any-solid. Returns the number of solid cells.
u32 buildCascadeSummary(v3s16 origin_nodes, int cell_nodes,
		std::vector<u8> &rgba, std::vector<u8> &boxes, const u8 matidx[4]);


// Bumped whenever a block summary changes; cheap staleness gate for
// the cascade rebuild schedule.
u64 contentVersion();

// Blocks currently summarized (stats).
size_t summaryCount();
// bytes held by stored (non-empty) summaries (stats)
size_t summaryBytes();
// drop summaries farther than TWICE radius_nodes from center (Chebyshev,
// in blocks): a camera that walks away does not keep the old world.
// Twice, so a camera wandering at a window's edge does not thrash.
size_t evictFar(v3s16 center_nodes, int radius_nodes);

// FAR TERRAIN FROM THE WORLD FILE (2026-10-06). The server sends only the
// blocks inside the camera's view cone and not occluded (clientiface.cpp
// GetNextBlocks), out to max_block_send_distance; the light law needs the
// world around the camera whatever it faces (a hill behind you shadows the
// ground in front). When the world's map.sqlite is on this machine, a
// worker thread reads it directly and folds each block through
// summarizeBlock -- the SAME fold as a received block, fill-only. Off
// unless claude_far_world names a world directory.
void startFarDb(Client *client, const std::string &world_dir);
// the window to fill: blocks within radius_nodes of center, nearest first
void setFarWindow(v3s16 center_nodes, int radius_nodes);
// joins the worker; must run before the Client is destroyed
void stopFarDb();
// blocks the worker has folded so far (stats)
size_t farDbLoaded();

}
