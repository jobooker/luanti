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

#include "filesys.h"
#include "porting.h"

#include <json/json.h>
#include <cstring>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace claude_lod
{

// One MapBlock folded to 4^3 subcells of 4^3 nodes: enough resolution
// to build any cascade level >= 4 m without ever touching nodes again.
// Sums fit comfortably: occupancy max 64/subcell, color sum max 64*255.
struct BlockSummary
{
	u8 occ[64];    // solid-ish nodes per subcell (0..64)
	u8 water[64];  // liquid nodes per subcell
	u8 leaf[64];   // of occ: foliage nodes — leaf-dominant cells fold to
	               // class 180 so forests read as canopy, not cliff wall
	// 2m-grain occupancy: bit o of fine[sub] = octant o (2^3 nodes) of
	// the subcell is >=5/8 solid; finew = same for liquids. This is what
	// lets ONE fold build every rung — the separate 2m block-walker
	// (whose alignment contract caused the origin-parity wall) is gone.
	u8 fine[64], finew[64];
	// per-octant mean color, RGB565 (0 = unset -> fall back to the
	// subcell mean): real 2m-scale variation folded from real nodes —
	// John's law: "variation comes from more voxels, not texture"
	u16 fineCol[64][8];
	u16 rsum[64], gsum[64], bsum[64]; // summed minimap color of counted nodes
};

static std::mutex g_mutex;
static std::unordered_map<v3s16, BlockSummary> g_summaries;
static u64 g_version = 0;

void summarizeBlock(Client *client, MapBlock *block)
{
	const NodeDefManager *ndef = client->getNodeDefManager();
	BlockSummary s = {};
	static thread_local u8 so[64][8], wo[64][8];
	static thread_local u32 rc[64][8], gc[64][8], bc[64][8];
	memset(so, 0, sizeof(so));
	memset(wo, 0, sizeof(wo));
	memset(rc, 0, sizeof(rc));
	memset(gc, 0, sizeof(gc));
	memset(bc, 0, sizeof(bc));
	for (s16 z = 0; z < MAP_BLOCKSIZE; z++)
	for (s16 y = 0; y < MAP_BLOCKSIZE; y++)
	for (s16 x = 0; x < MAP_BLOCKSIZE; x++) {
		MapNode n = block->getNodeNoCheck(x, y, z);
		content_t c = n.getContent();
		if (c == CONTENT_AIR || c == CONTENT_IGNORE)
			continue;
		const ContentFeatures &f = ndef->get(c);
		// Same skip rules as claudeTraceGridSnapshot: decorations are quads,
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
		// thin leveled layers (snow): counting them as full nodes raised
		// coarse ground +1m over snowfields — the LOD "wall" John caught
		// 2026-08-12. The snow LOOK survives: MCL ground under layers is
		// the white snowy-grass variant, full snow blocks aren't leveled.
		if (f.param_type_2 == CPT2_LEVELED)
			continue;
		int sub = (z / 4) * 16 + (y / 4) * 4 + (x / 4);
		int oct = ((z % 4) / 2) * 4 + ((y % 4) / 2) * 2 + (x % 4) / 2;
		video::SColor col(255, 180, 180, 180);
		if (f.visuals && f.visuals->minimap_color.getAlpha() > 0)
			col = f.visuals->minimap_color;
		// per-node biome tint (param2 palette): Mineclonia grass/leaves/
		// water are GRAYSCALE textures tinted at render time. The near
		// snapshot applies this; dropping it here turned every tinted
		// block gray in the cascades ("the quality of the color changes
		// dramatically" at the 1m/2m seam).
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
		if (f.isLiquid()) {
			s.water[sub]++;
			wo[sub][oct]++;
		} else {
			s.occ[sub]++;
			so[sub][oct]++;
			if (f.drawtype == NDT_ALLFACES_OPTIONAL)
				s.leaf[sub]++;
		}
		rc[sub][oct] += col.getRed();
		gc[sub][oct] += col.getGreen();
		bc[sub][oct] += col.getBlue();
		s.rsum[sub] += col.getRed();
		s.gsum[sub] += col.getGreen();
		s.bsum[sub] += col.getBlue();
	}
	for (int sub = 0; sub < 64; sub++)
		for (int o = 0; o < 8; o++) {
			if (so[sub][o] >= 5)
				s.fine[sub] |= (u8)(1 << o);
			if (wo[sub][o] >= 5)
				s.finew[sub] |= (u8)(1 << o);
			u32 n = (u32)so[sub][o] + wo[sub][o];
			if (n > 0)
				s.fineCol[sub][o] = (u16)(
					(((rc[sub][o] / n) >> 3) << 11)
					| (((gc[sub][o] / n) >> 2) << 5)
					| ((bc[sub][o] / n) >> 3));
		}
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		g_summaries[block->getPos()] = s;
		g_version++;
	}
}

// ---- far-data feed: JSON dropped by tooling (server bridge sample ->
// scp) becomes synthetic summaries. File format, one object per file:
// {"blocks":[{"p":[bx,by,bz],"sub":[[idx,occ,water,"node:name",p2],..]},..]}
static std::unordered_set<std::string> g_far_loaded;

size_t ingestFarDir(Client *client)
{
	std::string dir = porting::path_user + DIR_DELIM + "claude_far";
	if (!fs::PathExists(dir))
		return 0;
	const NodeDefManager *ndef = client->getNodeDefManager();
	size_t added = 0;
	for (const auto &e : fs::GetDirListing(dir)) {
		if (e.dir || e.name.size() < 6
				|| e.name.substr(e.name.size() - 5) != ".json")
			continue;
		if (g_far_loaded.count(e.name))
			continue;
		g_far_loaded.insert(e.name);
		std::ifstream f(dir + DIR_DELIM + e.name);
		Json::Value root;
		try {
			f >> root;
		} catch (...) {
			continue;
		}
		for (const Json::Value &b : root["blocks"]) {
			if (!b["p"].isArray() || b["p"].size() != 3)
				continue;
			v3s16 bp(b["p"][0].asInt(), b["p"][1].asInt(),
					b["p"][2].asInt());
			BlockSummary s = {};
			for (const Json::Value &sc : b["sub"]) {
				if (!sc.isArray() || sc.size() < 5)
					continue;
				int idx = sc[0].asInt();
				if (idx < 0 || idx >= 64)
					continue;
				int occ = std::min(sc[1].asInt(), 64);
				int water = std::min(sc[2].asInt(), 64);
				content_t c = ndef->getId(sc[3].asString());
				video::SColor col(255, 180, 180, 180);
				if (c != CONTENT_IGNORE) {
					const ContentFeatures &cf = ndef->get(c);
					if (cf.visuals
							&& cf.visuals->minimap_color.getAlpha() > 0)
						col = cf.visuals->minimap_color;
					if (cf.visuals) {
						video::SColor tint(255, 255, 255, 255);
						cf.visuals->getColor((u8)sc[4].asInt(), &tint);
						if (tint.getRed() != 255 || tint.getGreen() != 255
								|| tint.getBlue() != 255) {
							col.setRed(col.getRed() * tint.getRed() / 255);
							col.setGreen(col.getGreen()
									* tint.getGreen() / 255);
							col.setBlue(col.getBlue()
									* tint.getBlue() / 255);
						}
					}
				}
				s.occ[idx] = (u8)occ;
				s.water[idx] = (u8)water;
				int cnt = occ + water;
				s.rsum[idx] = (u16)(col.getRed() * cnt);
				s.gsum[idx] = (u16)(col.getGreen() * cnt);
				s.bsum[idx] = (u16)(col.getBlue() * cnt);
			}
			{
				std::lock_guard<std::mutex> lock(g_mutex);
				// real received blocks win: only fill holes
				if (!g_summaries.count(bp)) {
					g_summaries[bp] = s;
					added++;
				}
			}
		}
	}
	if (added) {
		std::lock_guard<std::mutex> lock(g_mutex);
		g_version++;
	}
	return added;
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
	rgba.assign((size_t)N * N * N * 4, 0);
	coarse.assign(32 * 32 * 32, 0);

	// SCATTER, not gather: iterate the blocks we actually HAVE (a few
	// thousand) and deposit their 4-node subcells into cells. Cost is
	// independent of cell size, and a 32 m cell spanning 2x2x2 blocks —
	// which breaks the one-block-per-cell gather — is handled for free.
	// Accumulators are static and reused (u16 counts: max 32^3 = 32768
	// nodes per cell fits; u32 color sums).
	static std::vector<u16> occ_acc, water_acc, leaf_acc;
	static std::vector<u32> r_acc, g_acc, b_acc;
	static std::vector<s16> top_acc;   // highest occupied subcell layer
	static std::vector<u16> top_n;     // counted nodes in that layer
	constexpr size_t NC = (size_t)N * N * N;
	occ_acc.assign(NC, 0);
	water_acc.assign(NC, 0);
	leaf_acc.assign(NC, 0);
	r_acc.assign(NC, 0);
	g_acc.assign(NC, 0);
	b_acc.assign(NC, 0);
	top_acc.assign(NC, -32768);
	top_n.assign(NC, 0);

	{
		std::lock_guard<std::mutex> lock(g_mutex);
		if (g_summaries.empty())
			return 0;
		const int span = N * CELL;
		for (const auto &kv : g_summaries) {
			v3s16 bbase(kv.first.X * 16, kv.first.Y * 16, kv.first.Z * 16);
			v3s16 rel = bbase - origin_nodes;
			if (rel.X < 0 || rel.Y < 0 || rel.Z < 0
					|| rel.X + 16 > span || rel.Y + 16 > span
					|| rel.Z + 16 > span)
				continue;
			const BlockSummary &s = kv.second;
			for (int sz = 0; sz < 4; sz++)
			for (int sy = 0; sy < 4; sy++)
			for (int sx = 0; sx < 4; sx++) {
				int sub = sz * 16 + sy * 4 + sx;
				u32 cnt = (u32)s.occ[sub] + s.water[sub];
				if (cnt == 0)
					continue;
				if (CELL == 2) {
					// octant expansion: one 4m subcell = 8 2m cells,
					// occupancy from the fine bit-sets, color/class
					// shared from the subcell (color grain at 2m is
					// invisible past the 64-node promotion distance)
					bool leafdom = s.leaf[sub] * 2 > s.occ[sub];
					u32 smr = s.rsum[sub] / cnt, smg = s.gsum[sub] / cnt,
						smb = s.bsum[sub] / cnt;
					for (int o = 0; o < 8; o++) {
						// real octant color; 0 = unset (far-fed data)
						u32 mr = smr, mg = smg, mb = smb;
						u16 fc = s.fineCol[sub][o];
						if (fc != 0) {
							mr = ((fc >> 11) & 31) << 3;
							mg = ((fc >> 5) & 63) << 2;
							mb = (fc & 31) << 3;
						}
						bool fs = (s.fine[sub] >> o) & 1;
						bool fw = (s.finew[sub] >> o) & 1;
						if (!fs && !fw)
							continue;
						int cxo = (rel.X + sx * 4 + (o & 1) * 2) / 2;
						int cyo = (rel.Y + sy * 4 + ((o >> 1) & 1) * 2) / 2;
						int czo = (rel.Z + sz * 4 + ((o >> 2) & 1) * 2) / 2;
						size_t ci2 = ((size_t)czo * N + cyo) * N + cxo;
						if (fs) {
							occ_acc[ci2] += 8;
							if (leafdom)
								leaf_acc[ci2] += 8;
						} else {
							water_acc[ci2] += 8;
						}
						s16 subY2 = (s16)(rel.Y + sy * 4);
						if (subY2 > top_acc[ci2]) {
							top_acc[ci2] = subY2;
							top_n[ci2] = 8;
							r_acc[ci2] = mr * 8;
							g_acc[ci2] = mg * 8;
							b_acc[ci2] = mb * 8;
						} else if (subY2 == top_acc[ci2]) {
							top_n[ci2] = (u16)(top_n[ci2] + 8);
							r_acc[ci2] += mr * 8;
							g_acc[ci2] += mg * 8;
							b_acc[ci2] += mb * 8;
						}
					}
					continue;
				}
				size_t ci = (((size_t)(rel.Z + sz * 4) / CELL) * N
						+ ((rel.Y + sy * 4) / CELL)) * N
						+ ((rel.X + sx * 4) / CELL);
				occ_acc[ci] += s.occ[sub];
				water_acc[ci] += s.water[sub];
				leaf_acc[ci] += s.leaf[sub];
				// COLOR = the cell's TOP occupied layer only. The grid
				// average mixed one white snow cap with seven dirt nodes
				// into green-brown ("snow at 1m rendered as maybe green,
				// before we go to it") — the face you SEE is the top.
				s16 subY = (s16)(rel.Y + sy * 4);
				if (subY > top_acc[ci]) {
					top_acc[ci] = subY;
					top_n[ci] = (u16)cnt;
					r_acc[ci] = s.rsum[sub];
					g_acc[ci] = s.gsum[sub];
					b_acc[ci] = s.bsum[sub];
				} else if (subY == top_acc[ci]) {
					top_n[ci] = (u16)(top_n[ci] + cnt);
					r_acc[ci] += s.rsum[sub];
					g_acc[ci] += s.gsum[sub];
					b_acc[ci] += s.bsum[sub];
				}
			}
		}
	}

	// threshold pass: >= 62% occupancy => solid. Biased ABOVE half so the
	// coarse surface ERODES rather than dilates: at 50% a cell just over
	// half full rounded the surface UP a full metre, and at the ring
	// boundary that rounding stood next to exact 1m terrain as a raised
	// rampart (John: "a tall border wall" at the 1m/2m seam). A slight
	// dip at the seam reads as terrain; a wall reads as a wall.
	const u32 half = (u32)((u32)CELL * CELL * CELL * 62 / 100);
	u32 solid_cells = 0;
	size_t i = 0;
	for (int cz = 0; cz < N; cz++)
	for (int cy = 0; cy < N; cy++)
	for (int cx = 0; cx < N; cx++, i++) {
		u32 counted = (u32)occ_acc[i] + water_acc[i];
		if (counted == 0)
			continue;
		u8 cls = 0;
		if (occ_acc[i] >= half)
			// leaf-dominant solids are FOLIAGE (180): the far shader
			// lights them as sky-bathed canopy and lets sun through —
			// opaque-black forest ramparts were "the LOD 2 wall"
			cls = leaf_acc[i] * 2 > occ_acc[i] ? 180 : 255;
		else if (water_acc[i] >= half && water_acc[i] > occ_acc[i])
			cls = 100;
		else
			continue;
		u32 tn = std::max(top_n[i], (u16)1);
		rgba[i * 4 + 0] = (u8)std::min(r_acc[i] / tn, 255u);
		rgba[i * 4 + 1] = (u8)std::min(g_acc[i] / tn, 255u);
		rgba[i * 4 + 2] = (u8)std::min(b_acc[i] / tn, 255u);
		rgba[i * 4 + 3] = cls;
		solid_cells++;
		coarse[((cz / 4) * 32 + (cy / 4)) * 32 + (cx / 4)] = 255;
	}
	return solid_cells;
}

}
