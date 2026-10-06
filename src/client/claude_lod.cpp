// claude_lod: far-terrain cascade data. See claude_lod.h.
#include "claude_lod.h"

#include "client/client.h"
#include "client/clientenvironment.h"
#include "map.h"
#include "mapblock.h"
#include "mapnode.h"
#include "nodedef.h"
#include "client/node_visuals.h" // minimap_color
#include <cmath>
#include "util/numeric.h"

#include "filesys.h"
#include "porting.h"

#include "gamedef.h"
#include "content/mods.h"
#include "database/database-sqlite3.h"
#include "util/serialize.h"
#include "log.h"
#include <atomic>
#include <algorithm>
#include <sstream>
#include <thread>
#include <cstring>
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
	// LAVA, apart from water (2026-10-06): a liquid that emits. Folded
	// to the lava material so far lava glows by the same law as near lava
	// instead of reading as a lake.
	u8 lava[64];
	u8 finel[64];
	// TIGHT BOX of the solid nodes in each octant (2026-10-06, far view):
	// bits 0-2 = min x/y/z node (0 or 1), bits 3-5 = max x/y/z, bit 7 =
	// the octant holds a solid at all. Folded into every level's per-cell
	// box, so a coarse cell carries WHERE its solid sits, not just that it
	// exists: one layer of ground in a 2 m cell is a 1 m slab, a 1 m trunk
	// is a 1 m column, and the far ground meets the near ground flush.
	u8 sbox[64][8];
	// per-octant mean color, RGB565 (0 = unset -> fall back to the
	// subcell mean): real 2m-scale variation folded from real nodes —
	// John's law: "variation comes from more voxels, not texture"
	u8 fineCol[64][8][3]; // sRGB r, g, b (set wherever the octant holds nodes)
	u32 rsum[64], gsum[64], bsum[64]; // summed LINEAR colour (x LIN) of the colour nodes
	// how many nodes the colour sums hold: the TOP-EXPOSED ones (a node with
	// open air above it) when the subcell has any, else all of them
	u8 ccount[64];
	// PLANTS (2026-10-06): plantlike nodes (tall grass, flowers, ferns) per
	// subcell, which octants hold any, and their mean colour (sRGB of the
	// linear mean, biome tint applied). The far levels fold them into a
	// cloud of blades over the ground (claude_far_plants); before this they
	// were skipped, and the judge measured the missing grass as the far
	// field's largest error.
	u8 plant[64];
	u8 plantFine[64];
	u8 plantCol[64][3];
};

// sRGB byte <-> linear * LIN, for summing albedo in LINEAR light.
// PRECISION (2026-10-06): linear at 255 steps and RGB565 octant colours
// that TRUNCATED their low bits cost the far ground half a quantum per
// channel -- the LOD energy referee read the 2 m plains 6 % darker than the
// 1 m, and the albedo-only view showed the whole deficit in the colours
// (spec/measured.md). Linear sums now carry 16 bits; octant colours are
// rounded 8-bit sRGB.
static constexpr u32 LIN = 65535;
static inline u32 toLin(u32 v)
{
	return (u32)(LIN * std::pow(v / 255.0, 2.2) + 0.5);
}
static inline u32 toSrgb(u32 v)
{
	return (u32)std::clamp(255.0 * std::pow(v / (double)LIN, 1.0 / 2.2) + 0.5,
			0.0, 255.0);
}

static std::mutex g_mutex;
static std::unordered_map<v3s16, BlockSummary> g_summaries;
// ALL-AIR BLOCKS (2026-10-06): 68 % of the 67k blocks the world-file
// reader folded held nothing, at ~3.8 KB each. They are remembered as
// known-empty (so neither the reader nor the builder looks again) instead
// of stored.
static std::unordered_set<v3s16> g_empty;
static u64 g_version = 0;

void summarizeBlock(Client *client, MapBlock *block, bool fill_only)
{
	const NodeDefManager *ndef = client->getNodeDefManager();
	BlockSummary s = {};
	static thread_local u8 so[64][8], wo[64][8], lo[64][8];
	static thread_local u8 bmn[64][8][3], bmx[64][8][3];
	static thread_local u32 rc[64][8], gc[64][8], bc[64][8];
	// the same sums over TOP-EXPOSED nodes only, and their counts
	static thread_local u32 erc[64][8], egc[64][8], ebc[64][8];
	static thread_local u8 en[64][8], an[64][8];
	static thread_local u32 pr[64], pg[64], pb[64];
	memset(so, 0, sizeof(so));
	memset(wo, 0, sizeof(wo));
	memset(lo, 0, sizeof(lo));
	memset(bmn, 1, sizeof(bmn));
	memset(bmx, 0, sizeof(bmx));
	memset(rc, 0, sizeof(rc));
	memset(gc, 0, sizeof(gc));
	memset(bc, 0, sizeof(bc));
	memset(erc, 0, sizeof(erc));
	memset(egc, 0, sizeof(egc));
	memset(ebc, 0, sizeof(ebc));
	memset(en, 0, sizeof(en));
	memset(an, 0, sizeof(an));
	memset(pr, 0, sizeof(pr));
	memset(pg, 0, sizeof(pg));
	memset(pb, 0, sizeof(pb));
	// does this node count as matter for the fold (the same skips as the
	// walk below)? Used to ask "is the node above open air"
	auto counts = [&](MapNode m) -> bool {
		content_t cc = m.getContent();
		if (cc == CONTENT_AIR || cc == CONTENT_IGNORE)
			return false;
		const ContentFeatures &ff = ndef->get(cc);
		if (ff.light_source == 0
				&& (ff.drawtype == NDT_PLANTLIKE
					|| ff.drawtype == NDT_PLANTLIKE_ROOTED
					|| ff.drawtype == NDT_FIRELIKE
					|| ff.drawtype == NDT_SIGNLIKE
					|| ff.drawtype == NDT_RAILLIKE
					|| ff.drawtype == NDT_TORCHLIKE))
			return false;
		if (ff.light_source > 0 && ff.drawtype == NDT_AIRLIKE)
			return false;
		if (ff.param_type_2 == CPT2_LEVELED)
			return false;
		return true;
	};
	for (s16 z = 0; z < MAP_BLOCKSIZE; z++)
	for (s16 y = 0; y < MAP_BLOCKSIZE; y++)
	for (s16 x = 0; x < MAP_BLOCKSIZE; x++) {
		MapNode n = block->getNodeNoCheck(x, y, z);
		content_t c = n.getContent();
		if (c == CONTENT_AIR || c == CONTENT_IGNORE)
			continue;
		const ContentFeatures &f = ndef->get(c);
		if (f.light_source == 0 && f.drawtype == NDT_PLANTLIKE) {
			int psub = (z / 4) * 16 + (y / 4) * 4 + (x / 4);
			int poct = ((z % 4) / 2) * 4 + ((y % 4) / 2) * 2 + (x % 4) / 2;
			video::SColor pc(255, 120, 160, 80);
			if (f.visuals && f.visuals->minimap_color.getAlpha() > 0)
				pc = f.visuals->minimap_color;
			if (f.visuals) {
				video::SColor tint(255, 255, 255, 255);
				f.visuals->getColor(n.getParam2(), &tint);
				pc.setRed(pc.getRed() * tint.getRed() / 255);
				pc.setGreen(pc.getGreen() * tint.getGreen() / 255);
				pc.setBlue(pc.getBlue() * tint.getBlue() / 255);
			}
			pr[psub] += toLin(pc.getRed());
			pg[psub] += toLin(pc.getGreen());
			pb[psub] += toLin(pc.getBlue());
			if (s.plant[psub] < 255)
				s.plant[psub]++;
			s.plantFine[psub] |= (u8)(1 << poct);
			continue;
		}
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
		if (f.isLiquid() && f.light_source > 0) {
			s.lava[sub]++;
			lo[sub][oct]++;
		} else if (f.isLiquid()) {
			s.water[sub]++;
			wo[sub][oct]++;
		} else {
			s.occ[sub]++;
			so[sub][oct]++;
			u8 lx = (u8)(x & 1), ly = (u8)(y & 1), lz = (u8)(z & 1);
			bmn[sub][oct][0] = std::min(bmn[sub][oct][0], lx);
			bmn[sub][oct][1] = std::min(bmn[sub][oct][1], ly);
			bmn[sub][oct][2] = std::min(bmn[sub][oct][2], lz);
			bmx[sub][oct][0] = std::max(bmx[sub][oct][0], lx);
			bmx[sub][oct][1] = std::max(bmx[sub][oct][1], ly);
			bmx[sub][oct][2] = std::max(bmx[sub][oct][2], lz);
			// leaves: ALLFACES_OPTIONAL as registered, ALLFACES once
			// leaves_style = fancy has resolved it at load (until
			// 2026-10-06 only the first was counted, so no far cell was
			// ever a leaf cell)
			if (f.drawtype == NDT_ALLFACES_OPTIONAL || f.drawtype == NDT_ALLFACES)
				s.leaf[sub]++;
		}
		// COLOUR IS WHAT YOU SEE: a node's colour counts toward its cell
		// when its TOP is open air (ADR-0010: albedo weighted by exposed
		// face area; from above, the area is the tops). The LOD energy
		// referee read the 2 m plains 5 % darker than the 1 m: each octant
		// averaged the grass with the dirt under it. Summed in LINEAR light.
		// At the block's top layer the node above is in another block:
		// counted as open.
		bool topOpen = (y == MAP_BLOCKSIZE - 1)
				|| !counts(block->getNodeNoCheck(x, y + 1, z));
		u32 lr = toLin(col.getRed()), lg = toLin(col.getGreen()),
				lb = toLin(col.getBlue());
		rc[sub][oct] += lr;
		gc[sub][oct] += lg;
		bc[sub][oct] += lb;
		an[sub][oct]++;
		if (topOpen) {
			erc[sub][oct] += lr;
			egc[sub][oct] += lg;
			ebc[sub][oct] += lb;
			en[sub][oct]++;
		}
	}
	for (int sub = 0; sub < 64; sub++)
		for (int o = 0; o < 8; o++) {
			// MAJORITY (2026-10-06): an octant is solid when at least half
			// of its 8 nodes are (ADR-0010's fold). The >= 5/8 rule it
			// replaces biased every coarse surface DOWN on purpose (the
			// 2026-08-12 "rampart" fix), which shifts mean height; the
			// rampart's real cause, a misaligned seam, is fixed by aligned
			// origins instead (the near grid now snaps to even nodes).
			if (so[sub][o] >= 4)
				s.fine[sub] |= (u8)(1 << o);
			else if (lo[sub][o] >= 4)
				s.finel[sub] |= (u8)(1 << o);
			else if (wo[sub][o] >= 4)
				s.finew[sub] |= (u8)(1 << o);
			if (so[sub][o] > 0)
				s.sbox[sub][o] = (u8)(0x80 | bmn[sub][o][0]
						| (bmn[sub][o][1] << 1) | (bmn[sub][o][2] << 2)
						| (bmx[sub][o][0] << 3) | (bmx[sub][o][1] << 4)
						| (bmx[sub][o][2] << 5));
			u32 n = (u32)so[sub][o] + wo[sub][o] + lo[sub][o];
			// stored as sRGB (RGB565) of the octant's MEAN LINEAR colour,
			// over its top-exposed nodes when it has any
			(void)n;
			bool ex = en[sub][o] > 0;
			u32 cn = ex ? en[sub][o] : an[sub][o];
			if (cn > 0) {
				s.fineCol[sub][o][0] = (u8)toSrgb(((ex ? erc : rc)[sub][o] + cn / 2) / cn);
				s.fineCol[sub][o][1] = (u8)toSrgb(((ex ? egc : gc)[sub][o] + cn / 2) / cn);
				s.fineCol[sub][o][2] = (u8)toSrgb(((ex ? ebc : bc)[sub][o] + cn / 2) / cn);
			}
		}
	for (int sub = 0; sub < 64; sub++) {
		u32 er = 0, eg = 0, eb = 0, ar = 0, ag = 0, ab = 0, ne = 0, na = 0;
		for (int o = 0; o < 8; o++) {
			er += erc[sub][o]; eg += egc[sub][o]; eb += ebc[sub][o];
			ar += rc[sub][o]; ag += gc[sub][o]; ab += bc[sub][o];
			ne += en[sub][o]; na += an[sub][o];
		}
		bool ex = ne > 0;
		s.rsum[sub] = ex ? er : ar;
		s.gsum[sub] = ex ? eg : ag;
		s.bsum[sub] = ex ? eb : ab;
		s.ccount[sub] = (u8)(ex ? ne : na);
		if (s.plant[sub] > 0) {
			u32 pn = s.plant[sub];
			s.plantCol[sub][0] = (u8)toSrgb((pr[sub] + pn / 2) / pn);
			s.plantCol[sub][1] = (u8)toSrgb((pg[sub] + pn / 2) / pn);
			s.plantCol[sub][2] = (u8)toSrgb((pb[sub] + pn / 2) / pn);
		}
	}
	{
		const v3s16 bp = block->getPos();
		bool empty = true;
		for (int i = 0; i < 64 && empty; i++)
			empty = s.occ[i] == 0 && s.water[i] == 0 && s.lava[i] == 0;
		std::lock_guard<std::mutex> lock(g_mutex);
		if (fill_only && (g_summaries.count(bp) || g_empty.count(bp)))
			return;
		if (empty) {
			// a block that held something and was emptied changes the view
			if (g_summaries.erase(bp))
				g_version++;
			g_empty.insert(bp);
		} else {
			g_empty.erase(bp);
			g_summaries[bp] = s;
			g_version++;
		}
	}
}

// ---- far terrain from the world file (see claude_lod.h) ----

// MapBlock::deSerialize maps node names through a gamedef, and the
// client's FATAL-ERRORs on a name it does not know. This one forwards to
// the client and maps an unknown name to CONTENT_UNKNOWN instead.
class FarGameDef : public IGameDef
{
public:
	FarGameDef(Client *c) : m_c(c) {}
	IItemDefManager *getItemDefManager() override { return m_c->getItemDefManager(); }
	const NodeDefManager *getNodeDefManager() override { return m_c->getNodeDefManager(); }
	ICraftDefManager *getCraftDefManager() override { return nullptr; }
	u16 allocateUnknownNodeId(const std::string &name) override { return CONTENT_UNKNOWN; }
	const std::vector<ModSpec> &getMods() const override { return m_mods; }
	const ModSpec *getModSpec(const std::string &modname) const override { return nullptr; }
	ModStorageDatabase *getModStorageDatabase() override { return nullptr; }
	bool joinModChannel(const std::string &channel) override { return false; }
	bool leaveModChannel(const std::string &channel) override { return false; }
	bool sendModChannelMessage(const std::string &channel,
			const std::string &message) override { return false; }
	ModChannel *getModChannel(const std::string &channel) override { return nullptr; }
	bool isClient() override { return true; }
private:
	Client *m_c;
	std::vector<ModSpec> m_mods;
};

static std::thread g_fd_thread;
static std::atomic<bool> g_fd_run{false};
static std::atomic<int> g_fd_cx{0}, g_fd_cy{0}, g_fd_cz{0}, g_fd_r{0};
static std::atomic<size_t> g_fd_loaded{0};

static void farDbLoop(Client *client, std::string dir)
{
	FarGameDef gd(client);
	std::unique_ptr<MapDatabaseSQLite3> db;
	try {
		db = std::make_unique<MapDatabaseSQLite3>(dir);
	} catch (std::exception &e) {
		errorstream << "claude_far_world: cannot open " << dir << ": "
				<< e.what() << std::endl;
		return;
	}
	std::vector<v3s16> all;
	std::unordered_set<v3s16> tried;
	u64 last_list = 0;
	std::vector<std::pair<int, v3s16>> todo;
	while (g_fd_run) {
		u64 now = porting::getTimeMs();
		// re-list now and then: the server keeps generating and saving
		if (all.empty() || now - last_list > 30000) {
			all.clear();
			try {
				db->listAllLoadableBlocks(all);
			} catch (std::exception &e) {
				errorstream << "claude_far_world: list: " << e.what() << std::endl;
			}
			last_list = now;
		}
		v3s16 c(g_fd_cx / MAP_BLOCKSIZE, g_fd_cy / MAP_BLOCKSIZE,
				g_fd_cz / MAP_BLOCKSIZE);
		int rb = g_fd_r / MAP_BLOCKSIZE;
		// what eviction drops, the reader must be free to load again
		for (auto it = tried.begin(); it != tried.end();) {
			v3s16 d = *it - c;
			if (std::max({std::abs(d.X), std::abs(d.Y), std::abs(d.Z)}) > 2 * rb)
				it = tried.erase(it);
			else
				++it;
		}
		todo.clear();
		{
			std::lock_guard<std::mutex> lock(g_mutex);
			for (const v3s16 &p : all) {
				v3s16 d = p - c;
				int m = std::max({std::abs(d.X), std::abs(d.Y), std::abs(d.Z)});
				if (m > rb || tried.count(p) || g_summaries.count(p)
						|| g_empty.count(p))
					continue;
				todo.emplace_back(d.X * d.X + d.Y * d.Y + d.Z * d.Z, p);
			}
		}
		if (todo.empty()) {
			for (int i = 0; i < 10 && g_fd_run; i++)
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}
		// nearest first, a batch at a time so a moving camera re-sorts
		std::sort(todo.begin(), todo.end(),
				[](const auto &a, const auto &b) { return a.first < b.first; });
		size_t batch = std::min<size_t>(todo.size(), 512);
		std::string blob;
		for (size_t i = 0; i < batch && g_fd_run; i++) {
			v3s16 p = todo[i].second;
			tried.insert(p);
			blob.clear();
			try {
				db->loadBlock(p, &blob);
				if (blob.empty())
					continue;
				std::istringstream is(blob, std::ios_base::binary);
				u8 version = readU8(is);
				MapBlock block(p, &gd);
				block.deSerialize(is, version, true);
				if (!block.isGenerated())
					continue;
				summarizeBlock(client, &block, true);
				g_fd_loaded++;
			} catch (std::exception &e) {
				// a block mid-write or of a newer format: skip it
			}
		}
	}
}

void startFarDb(Client *client, const std::string &world_dir)
{
	if (g_fd_run || world_dir.empty())
		return;
	// never CREATE a database: the sqlite backend opens read-write/create
	if (!fs::PathExists(world_dir + DIR_DELIM + "map.sqlite")) {
		errorstream << "claude_far_world: no map.sqlite in " << world_dir
				<< std::endl;
		return;
	}
	g_fd_run = true;
	g_fd_thread = std::thread(farDbLoop, client, world_dir);
	actionstream << "claude_far_world: reading " << world_dir << std::endl;
}

void setFarWindow(v3s16 center_nodes, int radius_nodes)
{
	g_fd_cx = center_nodes.X;
	g_fd_cy = center_nodes.Y;
	g_fd_cz = center_nodes.Z;
	g_fd_r = radius_nodes;
}

void stopFarDb()
{
	g_fd_run = false;
	if (g_fd_thread.joinable())
		g_fd_thread.join();
}

size_t farDbLoaded()
{
	return g_fd_loaded;
}

u64 contentVersion()
{
	std::lock_guard<std::mutex> lock(g_mutex);
	return g_version;
}

size_t evictFar(v3s16 center_nodes, int radius_nodes)
{
	v3s16 c(center_nodes.X / MAP_BLOCKSIZE, center_nodes.Y / MAP_BLOCKSIZE,
			center_nodes.Z / MAP_BLOCKSIZE);
	const int lim = 2 * radius_nodes / MAP_BLOCKSIZE;
	auto far = [&](const v3s16 &p) {
		v3s16 d = p - c;
		return std::max({std::abs(d.X), std::abs(d.Y), std::abs(d.Z)}) > lim;
	};
	size_t n = 0;
	std::lock_guard<std::mutex> lock(g_mutex);
	for (auto it = g_summaries.begin(); it != g_summaries.end();) {
		if (far(it->first)) { it = g_summaries.erase(it); n++; }
		else ++it;
	}
	for (auto it = g_empty.begin(); it != g_empty.end();) {
		if (far(*it)) { it = g_empty.erase(it); n++; }
		else ++it;
	}
	// no version bump: nothing evicted lies inside any level
	return n;
}

size_t summaryBytes()
{
	std::lock_guard<std::mutex> lock(g_mutex);
	return g_summaries.size() * sizeof(BlockSummary);
}

size_t summaryCount()
{
	std::lock_guard<std::mutex> lock(g_mutex);
	return g_summaries.size() + g_empty.size();
}

u32 buildCascadeSummary(v3s16 origin_nodes, int cell_nodes,
		std::vector<u8> &rgba, std::vector<u8> &boxes, const u8 matidx[4])
{
	constexpr int N = 128; // cells per axis
	const int CELL = cell_nodes;
	rgba.assign((size_t)N * N * N * 4, 0);
	// per cell: the tight box of its solid nodes, packed min | max << 4
	// per axis in units of max(1, CELL/16) nodes (see the shader's
	// marchFarLevel); a=255 marks a box
	boxes.assign((size_t)N * N * N * 4, 0);
	static std::vector<u8> bxmin, bxmax;
	bxmin.assign((size_t)N * N * N * 3, 255);
	bxmax.assign((size_t)N * N * N * 3, 0);
	auto boxAdd = [&](size_t ci, int nx, int ny, int nz) {
		// node offsets inside the cell, 0 .. CELL-1
		u8 v[3] = {(u8)nx, (u8)ny, (u8)nz};
		for (int a = 0; a < 3; a++) {
			bxmin[ci * 3 + a] = std::min(bxmin[ci * 3 + a], v[a]);
			bxmax[ci * 3 + a] = std::max(bxmax[ci * 3 + a], v[a]);
		}
	};
	// every solid octant's box, deposited into the cell that holds it
	auto octBoxes = [&](const BlockSummary &s, int sub, int sx, int sy,
			int sz, v3s16 rel) {
		for (int o = 0; o < 8; o++) {
			u8 b = s.sbox[sub][o];
			if (!(b & 0x80))
				continue;
			int ox = rel.X + sx * 4 + (o & 1) * 2;
			int oy = rel.Y + sy * 4 + ((o >> 1) & 1) * 2;
			int oz = rel.Z + sz * 4 + ((o >> 2) & 1) * 2;
			int cx = ox / CELL, cy = oy / CELL, cz = oz / CELL;
			size_t ci = ((size_t)cz * N + cy) * N + cx;
			int bx = ox - cx * CELL, by = oy - cy * CELL, bz = oz - cz * CELL;
			boxAdd(ci, bx + (b & 1), by + ((b >> 1) & 1), bz + ((b >> 2) & 1));
			boxAdd(ci, bx + ((b >> 3) & 1), by + ((b >> 4) & 1),
					bz + ((b >> 5) & 1));
		}
	};

	// SCATTER, not gather: iterate the blocks we actually HAVE (a few
	// thousand) and deposit their 4-node subcells into cells. Cost is
	// independent of cell size, and a 32 m cell spanning 2x2x2 blocks —
	// which breaks the one-block-per-cell gather — is handled for free.
	// Accumulators are static and reused (u16 counts: max 32^3 = 32768
	// nodes per cell fits; u32 color sums).
	static std::vector<u16> occ_acc, water_acc, leaf_acc, lava_acc, plant_acc;
	static std::vector<u32> pr_acc, pg_acc, pb_acc;
	static std::vector<u32> r_acc, g_acc, b_acc;
	static std::vector<s16> top_acc;   // highest occupied subcell layer
	static std::vector<u16> top_n;     // counted nodes in that layer
	constexpr size_t NC = (size_t)N * N * N;
	occ_acc.assign(NC, 0);
	water_acc.assign(NC, 0);
	leaf_acc.assign(NC, 0);
	plant_acc.assign(NC, 0);
	pr_acc.assign(NC, 0);
	pg_acc.assign(NC, 0);
	pb_acc.assign(NC, 0);
	lava_acc.assign(NC, 0);
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
				u32 cnt = (u32)s.occ[sub] + s.water[sub] + s.lava[sub];
				if (cnt == 0)
					continue;
				if (s.occ[sub] > 0)
					octBoxes(s, sub, sx, sy, sz, rel);
				if (CELL == 2) {
					// octant expansion: one 4m subcell = 8 2m cells,
					// occupancy from the fine bit-sets, color/class
					// shared from the subcell (color grain at 2m is
					// invisible past the 64-node promotion distance)
					bool leafdom = s.leaf[sub] * 2 > s.occ[sub];
					for (int o = 0; o < 8; o++) {
						// the octant's own colour (an octant that folds to
						// anything holds >= 4 nodes, so it is always set)
						u32 mr = toLin(s.fineCol[sub][o][0]),
							mg = toLin(s.fineCol[sub][o][1]),
							mb = toLin(s.fineCol[sub][o][2]);
						bool fs = (s.fine[sub] >> o) & 1;
						bool fw = (s.finew[sub] >> o) & 1;
						bool fl = (s.finel[sub] >> o) & 1;
						int cxo = (rel.X + sx * 4 + (o & 1) * 2) / 2;
						int cyo = (rel.Y + sy * 4 + ((o >> 1) & 1) * 2) / 2;
						int czo = (rel.Z + sz * 4 + ((o >> 2) & 1) * 2) / 2;
						size_t ci2 = ((size_t)czo * N + cyo) * N + cxo;
						if ((s.plantFine[sub] >> o) & 1) {
							// the subcell's plants, shared among the octants
							// that hold any
							u32 share = s.plant[sub]
									/ std::max(1, __builtin_popcount(s.plantFine[sub]));
							plant_acc[ci2] += (u16)share;
							pr_acc[ci2] += toLin(s.plantCol[sub][0]) * share;
							pg_acc[ci2] += toLin(s.plantCol[sub][1]) * share;
							pb_acc[ci2] += toLin(s.plantCol[sub][2]) * share;
						}
						if (!fs && !fw && !fl)
							continue;
						if (fs) {
							occ_acc[ci2] += 8;
							if (leafdom)
								leaf_acc[ci2] += 8;
						} else if (fl) {
							lava_acc[ci2] += 8;
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
				lava_acc[ci] += s.lava[sub];
				leaf_acc[ci] += s.leaf[sub];
				if (s.plant[sub] > 0) {
					plant_acc[ci] += s.plant[sub];
					pr_acc[ci] += toLin(s.plantCol[sub][0]) * s.plant[sub];
					pg_acc[ci] += toLin(s.plantCol[sub][1]) * s.plant[sub];
					pb_acc[ci] += toLin(s.plantCol[sub][2]) * s.plant[sub];
				}
				// COLOR = the cell's TOP occupied layer only. The grid
				// average mixed one white snow cap with seven dirt nodes
				// into green-brown ("snow at 1m rendered as maybe green,
				// before we go to it") — the face you SEE is the top.
				s16 subY = (s16)(rel.Y + sy * 4);
				if (subY > top_acc[ci]) {
					top_acc[ci] = subY;
					top_n[ci] = (u16)s.ccount[sub];
					r_acc[ci] = s.rsum[sub];
					g_acc[ci] = s.gsum[sub];
					b_acc[ci] = s.bsum[sub];
				} else if (subY == top_acc[ci]) {
					top_n[ci] = (u16)(top_n[ci] + s.ccount[sub]);
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
	// MAJORITY, at least half the cell (ADR-0010; was >= 62 %, see the
	// octant rule in summarizeBlock for why)
	const u32 half = (u32)((u32)CELL * CELL * CELL / 2);
	u32 solid_cells = 0, leaf_cells = 0, plant_cells = 0;
	size_t i = 0;
	for (int cz = 0; cz < N; cz++)
	for (int cy = 0; cy < N; cy++)
	for (int cx = 0; cx < N; cx++, i++) {
		u32 counted = (u32)occ_acc[i] + water_acc[i] + lava_acc[i];
		if (counted == 0 && plant_acc[i] == 0)
			continue;
		// THE CELL'S BYTE IS A MATERIAL INDEX (2026-10-06), the same index
		// into the same palette as the near grid's, so a far cell is lit
		// by the one light law (physics-contract §7). matidx = {solid,
		// leaves, water, lava}, from game.cpp claudeMatIndex. The old
		// class bytes (255 / 180 / 100) had their own far-only shading.
		u8 cls = 0;
		// SOLID IF ANYTHING SOLID IS IN IT, and the box says where
		// (research 2026-10-06: conservative occupancy for the hit test,
		// a tight per-cell box to keep coverage honest for terrain). The
		// majority rule this replaces had to round flat ground to whole
		// cells, which put the far ground up to 1 m off the near ground.
		bool has_box = bxmin[i * 3] != 255;
		if (has_box && occ_acc[i] * 4 >= (u32)water_acc[i] + lava_acc[i])
			cls = leaf_acc[i] * 2 > occ_acc[i] ? matidx[1] : matidx[0];
		else if (lava_acc[i] >= half)
			cls = matidx[3];
		else if (water_acc[i] >= half && water_acc[i] > occ_acc[i])
			cls = matidx[2];
		else if (!has_box && plant_acc[i] > 0) {
			// ONLY PLANTS: a cloud of blades filling the cell, with the leaf
			// material (a blade is a leaf: it reflects and transmits)
			u32 pn = plant_acc[i];
			rgba[i * 4 + 0] = (u8)toSrgb(std::min((pr_acc[i] + pn / 2) / pn, LIN));
			rgba[i * 4 + 1] = (u8)toSrgb(std::min((pg_acc[i] + pn / 2) / pn, LIN));
			rgba[i * 4 + 2] = (u8)toSrgb(std::min((pb_acc[i] + pn / 2) / pn, LIN));
			rgba[i * 4 + 3] = matidx[1];
			const int unitp = std::max(1, CELL / 16);
			for (int a = 0; a < 3; a++)
				boxes[i * 4 + a] = (u8)(0 | ((CELL / unitp - 1) << 4));
			float rho = std::min(1.0f, (float)pn / (float)(CELL * CELL * CELL));
			boxes[i * 4 + 3] = (u8)(127 + std::clamp((int)std::lround(rho * 127.0f), 1, 127));
			solid_cells++;
			plant_cells++;
			continue;
		} else
			continue;
		u32 tn = std::max(top_n[i], (u16)1);
		rgba[i * 4 + 0] = (u8)toSrgb(std::min((r_acc[i] + tn / 2) / tn, LIN));
		rgba[i * 4 + 1] = (u8)toSrgb(std::min((g_acc[i] + tn / 2) / tn, LIN));
		rgba[i * 4 + 2] = (u8)toSrgb(std::min((b_acc[i] + tn / 2) / tn, LIN));
		rgba[i * 4 + 3] = cls;
		solid_cells++;
		const int unit = std::max(1, CELL / 16);
		if (cls == matidx[0] || cls == matidx[1]) {
			for (int a = 0; a < 3; a++)
				boxes[i * 4 + a] = (u8)((bxmin[i * 3 + a] / unit)
						| ((bxmax[i * 3 + a] / unit) << 4));
		} else {
			// liquids fill their whole cell (their surface is not boxed yet)
			for (int a = 0; a < 3; a++)
				boxes[i * 4 + a] = (u8)(0 | ((CELL / unit - 1) << 4));
		}
		// alpha: 255 = a solid box. A LEAF cell carries instead its leaf
		// density inside the box (leaf nodes / box volume, 1..254), which the
		// shader walks as a statistical cloud of leaf sheets
		// (claude_far_leaf_medium) rather than a solid lump.
		boxes[i * 4 + 3] = 255;
		if (cls == matidx[1]) {
			u32 vol = 1;
			for (int a = 0; a < 3; a++)
				vol *= (u32)((bxmax[i * 3 + a] / unit) - (bxmin[i * 3 + a] / unit) + 1)
						* unit;
			float rho = std::min(1.0f, (float)leaf_acc[i] / (float)std::max(vol, 1u));
			boxes[i * 4 + 3] = (u8)std::clamp((int)std::lround(rho * 127.0f), 1, 127);
			leaf_cells++;
		} else if (cls == matidx[0] && plant_acc[i] > 0) {
			// GROUND WITH GRASS ON IT: the solid box, and a layer of blades
			// in the room between the box top and the cell top.
			// alpha 128..254 = that layer's plant density
			int top = ((bxmax[i * 3 + 1] / unit) + 1) * unit;
			int room = CELL - top;
			if (room > 0) {
				float rho = std::min(1.0f, (float)plant_acc[i]
						/ (float)(CELL * CELL * room));
				boxes[i * 4 + 3] = (u8)(127
						+ std::clamp((int)std::lround(rho * 127.0f), 1, 127));
				plant_cells++;
			}
		}
	}
	// INSTRUMENT: how many cells the leaf medium can act on
	infostream << "[claude_lod] level cell " << CELL << " m: " << solid_cells
			<< " cells, " << leaf_cells << " leaf, " << plant_cells << " plant"
			<< std::endl;
	return solid_cells;
}

}
