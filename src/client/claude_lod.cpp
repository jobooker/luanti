// claude_lod: far-terrain cascade data. See claude_lod.h.
#include "claude_lod.h"

#include "client/client.h"
#include "mapblock.h"
#include "mapnode.h"
#include "nodedef.h"
#include "client/node_visuals.h" // minimap_color
#include "util/numeric.h"

#include <mutex>
#include <unordered_map>

namespace claude_lod
{

// One MapBlock folded to 4^3 subcells of 4^3 nodes: enough resolution
// to build any cascade level >= 4 m without ever touching nodes again.
// Sums fit comfortably: occupancy max 64/subcell, color sum max 64*255.
struct BlockSummary
{
	u8 occ[64];    // solid-ish nodes per subcell (0..64)
	u8 water[64];  // liquid nodes per subcell
	u16 rsum[64], gsum[64], bsum[64]; // summed minimap color of counted nodes
};

static std::mutex g_mutex;
static std::unordered_map<v3s16, BlockSummary> g_summaries;
static u64 g_version = 0;

void summarizeBlock(Client *client, MapBlock *block)
{
	const NodeDefManager *ndef = client->getNodeDefManager();
	BlockSummary s = {};
	for (s16 z = 0; z < MAP_BLOCKSIZE; z++)
	for (s16 y = 0; y < MAP_BLOCKSIZE; y++)
	for (s16 x = 0; x < MAP_BLOCKSIZE; x++) {
		MapNode n = block->getNodeNoCheck(x, y, z);
		content_t c = n.getContent();
		if (c == CONTENT_AIR || c == CONTENT_IGNORE)
			continue;
		const ContentFeatures &f = ndef->get(c);
		// Same skip rules as claudeVolumeSnapshot: decorations are quads,
		// not cubes, and airlike light nodes are invisible.
		if (f.light_source == 0
				&& (f.drawtype == NDT_PLANTLIKE
					|| f.drawtype == NDT_PLANTLIKE_ROOTED
					|| f.drawtype == NDT_FIRELIKE
					|| f.drawtype == NDT_SIGNLIKE
					|| f.drawtype == NDT_RAILLIKE
					|| f.drawtype == NDT_TORCHLIKE))
			continue;
		if (f.light_source > 0 && f.drawtype == NDT_AIRLIKE)
			continue;
		int sub = (z / 4) * 16 + (y / 4) * 4 + (x / 4);
		video::SColor col(255, 180, 180, 180);
		if (f.visuals && f.visuals->minimap_color.getAlpha() > 0)
			col = f.visuals->minimap_color;
		if (f.isLiquid()) {
			s.water[sub]++;
		} else {
			s.occ[sub]++;
		}
		s.rsum[sub] += col.getRed();
		s.gsum[sub] += col.getGreen();
		s.bsum[sub] += col.getBlue();
	}
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		g_summaries[block->getPos()] = s;
		g_version++;
	}
}

u64 contentVersion()
{
	std::lock_guard<std::mutex> lock(g_mutex);
	return g_version;
}

size_t summaryCount()
{
	std::lock_guard<std::mutex> lock(g_mutex);
	return g_summaries.size();
}

u32 buildCascade8(v3s16 origin_nodes, std::vector<u8> &rgba,
		std::vector<u8> &coarse)
{
	constexpr int N = 128;    // cells per axis
	constexpr int CELL = 8;   // nodes per cell
	rgba.assign((size_t)N * N * N * 4, 0);
	coarse.assign(32 * 32 * 32, 0);
	u32 solid_cells = 0;

	std::lock_guard<std::mutex> lock(g_mutex);
	if (g_summaries.empty())
		return 0;

	// An 8-node cell at 8-node alignment always lies inside ONE MapBlock
	// (16 nodes), spanning exactly 2^3 of its 4-node subcells — so each
	// cell is one map lookup plus eight array reads.
	size_t i = 0;
	const BlockSummary *cached = nullptr;
	v3s16 cached_pos(32767, 32767, 32767);
	for (int cz = 0; cz < N; cz++)
	for (int cy = 0; cy < N; cy++)
	for (int cx = 0; cx < N; cx++, i++) {
		v3s16 base = origin_nodes + v3s16(cx * CELL, cy * CELL, cz * CELL);
		v3s16 bp(base.X >> 4, base.Y >> 4, base.Z >> 4);
		if (bp != cached_pos) {
			auto it = g_summaries.find(bp);
			cached = it == g_summaries.end() ? nullptr : &it->second;
			cached_pos = bp;
		}
		if (!cached)
			continue; // never seen: air (the fog owns the data frontier)
		int sx = (base.X & 15) / 4, sy = (base.Y & 15) / 4,
			sz = (base.Z & 15) / 4;
		u32 occ = 0, water = 0, r = 0, g = 0, b = 0;
		for (int oz = 0; oz < 2; oz++)
		for (int oy = 0; oy < 2; oy++)
		for (int ox = 0; ox < 2; ox++) {
			int sub = (sz + oz) * 16 + (sy + oy) * 4 + (sx + ox);
			occ += cached->occ[sub];
			water += cached->water[sub];
			r += cached->rsum[sub];
			g += cached->gsum[sub];
			b += cached->bsum[sub];
		}
		u32 counted = occ + water;
		if (counted == 0)
			continue;
		// >= 50% of the cell's 512 nodes occupied => solid; water wins
		// only when it outnumbers solid matter (a lake surface cell)
		u8 cls = 0;
		if (occ >= 256)
			cls = 255;
		else if (water >= 256 && water > occ)
			cls = 100;
		else
			continue; // sparse: air at this resolution
		rgba[i * 4 + 0] = (u8)(r / counted);
		rgba[i * 4 + 1] = (u8)(g / counted);
		rgba[i * 4 + 2] = (u8)(b / counted);
		rgba[i * 4 + 3] = cls;
		solid_cells++;
		coarse[((cz / 4) * 32 + (cy / 4)) * 32 + (cx / 4)] = 255;
	}
	return solid_cells;
}

}
