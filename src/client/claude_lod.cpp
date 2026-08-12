// claude_lod: far-terrain cascade data. See claude_lod.h.
#include "claude_lod.h"

#include "client/client.h"
#include "client/clientenvironment.h"
#include "map.h"
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

u32 buildCascadeSummary(v3s16 origin_nodes, int cell_nodes,
		std::vector<u8> &rgba, std::vector<u8> &coarse)
{
	constexpr int N = 128; // cells per axis
	const int CELL = cell_nodes;
	const int SUBS = CELL / 4; // subcells (4 nodes) per cell per axis: 1 or 2
	rgba.assign((size_t)N * N * N * 4, 0);
	coarse.assign(32 * 32 * 32, 0);
	u32 solid_cells = 0;

	std::lock_guard<std::mutex> lock(g_mutex);
	if (g_summaries.empty())
		return 0;

	// A CELL-node cell at CELL-node alignment always lies inside ONE
	// MapBlock (16 nodes) for CELL in {4, 8} — one map lookup plus a few
	// array reads per cell.
	const u32 half = (u32)(CELL * CELL * CELL) / 2; // 50% occupancy
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
		for (int oz = 0; oz < SUBS; oz++)
		for (int oy = 0; oy < SUBS; oy++)
		for (int ox = 0; ox < SUBS; ox++) {
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
		// >= 50% of the cell's nodes occupied => solid; water wins only
		// when it outnumbers solid matter (a lake surface cell)
		u8 cls = 0;
		if (occ >= half)
			cls = 255;
		else if (water >= half && water > occ)
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

// per-content classification LUT so the 2 m walk never touches the
// NodeDefManager in its inner loop. 0 = unknown (resolve), 1 = air/skip,
// 2 = solid, 3 = water. Color packed 0xRRGGBB alongside.
static std::vector<u8> g_cls_lut;
static std::vector<u32> g_col_lut;

static inline u8 classify(const NodeDefManager *ndef, content_t c)
{
	if (c >= g_cls_lut.size()) {
		g_cls_lut.resize(c + 256, 0);
		g_col_lut.resize(c + 256, 0xB4B4B4);
	}
	u8 cls = g_cls_lut[c];
	if (cls)
		return cls;
	const ContentFeatures &f = ndef->get(c);
	cls = 2;
	if (c == CONTENT_AIR || c == CONTENT_IGNORE)
		cls = 1;
	else if (f.light_source == 0
			&& (f.drawtype == NDT_PLANTLIKE
				|| f.drawtype == NDT_PLANTLIKE_ROOTED
				|| f.drawtype == NDT_FIRELIKE
				|| f.drawtype == NDT_SIGNLIKE
				|| f.drawtype == NDT_RAILLIKE
				|| f.drawtype == NDT_TORCHLIKE))
		cls = 1;
	else if (f.light_source > 0 && f.drawtype == NDT_AIRLIKE)
		cls = 1;
	else if (f.isLiquid())
		cls = 3;
	if (cls != 1 && f.visuals && f.visuals->minimap_color.getAlpha() > 0) {
		video::SColor col = f.visuals->minimap_color;
		g_col_lut[c] = (col.getRed() << 16) | (col.getGreen() << 8)
				| col.getBlue();
	}
	g_cls_lut[c] = cls;
	return cls;
}

u32 buildCascade2(Client *client, v3s16 origin_nodes,
		std::vector<u8> &rgba, std::vector<u8> &coarse)
{
	constexpr int N = 128, CELL = 2;
	rgba.assign((size_t)N * N * N * 4, 0);
	coarse.assign(32 * 32 * 32, 0);
	u32 solid_cells = 0;
	Map &map = client->getEnv().getMap();
	const NodeDefManager *ndef = client->getNodeDefManager();

	// walk whole loaded MapBlocks (16^3 = 8^3 cells each); origin is
	// 32-node snapped so blocks tile the box exactly
	constexpr int BLOCKS = N * CELL / 16; // 16 across
	for (int bz = 0; bz < BLOCKS; bz++)
	for (int by = 0; by < BLOCKS; by++)
	for (int bx = 0; bx < BLOCKS; bx++) {
		v3s16 bpos((origin_nodes.X >> 4) + bx, (origin_nodes.Y >> 4) + by,
				(origin_nodes.Z >> 4) + bz);
		MapBlock *block = map.getBlockNoCreateNoEx(bpos);
		if (!block)
			continue; // not loaded: air
		// cell base within the level: 8 cells per axis per block
		int cbx = bx * 8, cby = by * 8, cbz = bz * 8;
		for (int cz = 0; cz < 8; cz++)
		for (int cy = 0; cy < 8; cy++)
		for (int cx = 0; cx < 8; cx++) {
			u32 occ = 0, water = 0, r = 0, g = 0, b = 0;
			for (int oz = 0; oz < 2; oz++)
			for (int oy = 0; oy < 2; oy++)
			for (int ox = 0; ox < 2; ox++) {
				MapNode n = block->getNodeNoCheck(cx * 2 + ox,
						cy * 2 + oy, cz * 2 + oz);
				u8 cls = classify(ndef, n.getContent());
				if (cls == 1)
					continue;
				u32 col = g_col_lut[n.getContent()];
				if (cls == 3) water++; else occ++;
				r += (col >> 16) & 0xFF;
				g += (col >> 8) & 0xFF;
				b += col & 0xFF;
			}
			u32 counted = occ + water;
			if (counted == 0)
				continue;
			u8 cls = 0;
			if (occ >= 4)
				cls = 255;
			else if (water >= 4 && water > occ)
				cls = 100;
			else
				continue;
			size_t i = ((size_t)(cbz + cz) * N + (cby + cy)) * N
					+ (cbx + cx);
			rgba[i * 4 + 0] = (u8)(r / counted);
			rgba[i * 4 + 1] = (u8)(g / counted);
			rgba[i * 4 + 2] = (u8)(b / counted);
			rgba[i * 4 + 3] = cls;
			solid_cells++;
			coarse[(((cbz + cz) / 4) * 32 + ((cby + cy) / 4)) * 32
					+ ((cbx + cx) / 4)] = 255;
		}
	}
	return solid_cells;
}

}
