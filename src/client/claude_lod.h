// claude_lod: far-terrain cascade data for the traced renderer.
// Phase 1 of the cascaded clipmap (see vault: luanti-lod-clipmap-plan.md,
// ADR-0004): per-MapBlock summaries folded at receive time, and an 8 m
// cascade (128^3 cells, +/-512 m) built from them on demand.
#pragma once

#include "irrlichttypes_bloated.h"
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
void summarizeBlock(Client *client, MapBlock *block);

// Build a summary-fed cascade level: 128^3 cells of cell_nodes (4 or 8)
// covering origin_nodes + 128*cell_nodes. rgba: 128^3 * 4 (rgb =
// occupancy-weighted average color, a = class: 0 air, 100 water, 255
// solid). coarse: 32^3 any-solid. Returns the number of solid cells.
u32 buildCascadeSummary(v3s16 origin_nodes, int cell_nodes,
		std::vector<u8> &rgba, std::vector<u8> &coarse);

// Build the 2 m level: FINER than the summaries, so it walks loaded
// MapBlocks directly (getNodeNoCheck + a per-content classification LUT
// — no per-node NodeDefManager lookups). Unloaded blocks read as air.
// ~16M node reads; tens of ms, so the caller schedules it sparsely.
u32 buildCascade2(Client *client, v3s16 origin_nodes,
		std::vector<u8> &rgba, std::vector<u8> &coarse);

// Bumped whenever a block summary changes; cheap staleness gate for
// the cascade rebuild schedule.
u64 contentVersion();

// Blocks currently summarized (stats).
size_t summaryCount();

}
