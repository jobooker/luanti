// Luanti
// SPDX-License-Identifier: LGPL-2.1-or-later
//
// THE ISOLATING INSTRUMENT for "read real geometry from node_box".
//
// spec/handoffs/2026-08-16-nodebox-geometry.md sets a two-miss rule and
// names the instrument to build when it trips: "render a single node's
// 16^3 mask as a debug view and compare it against the nodebox definition
// directly, rather than iterating on a lit scene." This file is that
// instrument, built BEFORE the first miss rather than after the second —
// a lit scene has a shader, an accumulator, a settle and a golden between
// the mask and the eye, and none of them can tell you which bit is wrong.
//
// Every assertion here is on BITS, against a box list written out by hand
// from the nodebox definition. No frame, no GPU, no seat.
//
// Failures print claudeMaskAscii() — 16 y-slices of '#' and '.', which
// reads on shape alone (John is colourblind; environment-laws.md).

#include "test.h"

#include "client/claude_nodebox.h"
#include "constants.h"
#include "gamedef.h"
#include "mapnode.h"
#include "nodedef.h"
#include <cstring>

// mapnode.cpp's authoritative reader of every NodeBox type. It has no
// declaration in a header; this is the same signature, minus the default
// argument (which lives on the definition).
void transformNodeBox(const MapNode &n, const NodeBox &nodebox,
		const NodeDefManager *nodemgr, std::vector<aabb3f> *p_boxes,
		u8 neighbors);

class TestClaudeNodeBox : public TestBase
{
public:
	TestClaudeNodeBox() { TestManager::registerTestModule(this); }
	const char *getName() { return "TestClaudeNodeBox"; }

	void runTests(IGameDef *gamedef);

	void testRegularCubeHasNoBoxes();
	void testFullCubeFillsEveryBit();
	void testSlabIsTheBottomHalf();
	void testStairIsAnLShape();
	void testParam2RotatesTheStep();
	void testEdgesLandOnSubvoxelLines();
	void testSliverCannotFattenALayer();
	void testConnectedIsConverted();
	void testPaneAloneIsJustThePost();
	void testPaneWestAndEastIsAFullSlab();
	void testPaneNeighboursAreDifferentShapes();
};

static TestClaudeNodeBox g_test_instance;

////////////////////////////////////////////////////////////////////////////////
// Box lists written from the definitions, in BS units (-BS/2..+BS/2).

static aabb3f boxOf(float x0, float y0, float z0, float x1, float y1, float z1)
{
	return aabb3f(x0 * BS, y0 * BS, z0 * BS, x1 * BS, y1 * BS, z1 * BS);
}

// The vanilla stair: a bottom slab plus a step on the +z half. This is
// mcl_stairs / stairs:stair_* verbatim, and the shape gate 1 is about.
static void stairBoxes(std::vector<aabb3f> *out)
{
	out->push_back(boxOf(-0.5f, -0.5f, -0.5f, 0.5f, 0.0f, 0.5f));
	out->push_back(boxOf(-0.5f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f));
}

static int popcount(const u8 *mask)
{
	int n = 0;
	for (int i = 0; i < CLAUDE_NBOX_MASK_BYTES; i++)
		for (int b = 0; b < 8; b++)
			n += (mask[i] >> b) & 1;
	return n;
}

// A stair node registered into the test gamedef, so the rotation test
// runs Luanti's OWN transform rather than a re-derivation of it.
static content_t s_stair = CONTENT_IGNORE;
static const NodeDefManager *s_ndef = nullptr;

void TestClaudeNodeBox::runTests(IGameDef *gamedef)
{
	s_ndef = gamedef->getNodeDefManager();
	{
		NodeDefManager *ndef = (NodeDefManager *)s_ndef;
		ContentFeatures f;
		f.name = "claude_test:stair";
		f.drawtype = NDT_NODEBOX;
		f.param_type_2 = CPT2_FACEDIR;
		f.node_box.type = NODEBOX_FIXED;
		stairBoxes(&f.node_box.fixed);
		s_stair = ndef->set(f.name, f);
	}

	TEST(testRegularCubeHasNoBoxes);
	TEST(testFullCubeFillsEveryBit);
	TEST(testSlabIsTheBottomHalf);
	TEST(testStairIsAnLShape);
	TEST(testParam2RotatesTheStep);
	TEST(testEdgesLandOnSubvoxelLines);
	TEST(testSliverCannotFattenALayer);
	TEST(testConnectedIsConverted);
	TEST(testPaneAloneIsJustThePost);
	TEST(testPaneWestAndEastIsAFullSlab);
	TEST(testPaneNeighboursAreDifferentShapes);
}

////////////////////////////////////////////////////////////////////////////////

// GATE 2, the static half, AND THE TRAP UNDER IT.
//
// "Full-cube nodes must not move" reads like it needs no defending: a
// regular cube has no box list to convert. IT HAS ONE. transformNodeBox's
// final else is
//
//     else // NODEBOX_REGULAR
//         boxes.emplace_back(-BS/2,-BS/2,-BS/2, BS/2,BS/2,BS/2);
//
// (mapnode.cpp), so NODEBOX_REGULAR hands back a full-cube box like any
// other shape. A nodebox path gated on "did the box list come back
// non-empty" would therefore have fired on EVERY solid in cornell and
// both furnaces, tagged them class 250, and moved the three scored arms
// the gate exists to protect -- which is exactly the failure the handoff
// predicts ("if they move, the nodebox path is firing on NODEBOX_REGULAR
// and the conversion is wrong").
//
// So the gate is on the TYPE, in claudeNodeBoxConvertible(), and the
// full-cube bit count is a second, independent stop in the walk. This
// test pins both, and it pins the surprise itself so nobody re-derives it.
void TestClaudeNodeBox::testRegularCubeHasNoBoxes()
{
	NodeBox nb; // default-constructed == NODEBOX_REGULAR
	UASSERTEQ(int, (int)nb.type, (int)NODEBOX_REGULAR);

	std::vector<aabb3f> boxes;
	// nodemgr is only dereferenced by the FIXED/LEVELED branch, which a
	// REGULAR box does not enter.
	transformNodeBox(MapNode(CONTENT_AIR), nb, nullptr, &boxes, 0);
	UASSERTEQ(size_t, boxes.size(), 1u); // NOT empty -- a full cube

	u8 mask[CLAUDE_NBOX_MASK_BYTES];
	memset(mask, 0, sizeof(mask));
	UASSERTEQ(int, claudeRasterizeBoxes(boxes, mask), 4096);
	UASSERTEQ(int, popcount(mask), 4096);

	// STOP 1: the type gate never lets a regular cube reach the converter.
	UASSERT(!claudeNodeBoxConvertible(nb));
	// STOP 2: and if it ever did, 4096 bits is "no sub-voxel information",
	// which the walk treats as "leave it a 1 m cube" (claudeNodeBoxMaskId).
}

void TestClaudeNodeBox::testFullCubeFillsEveryBit()
{
	std::vector<aabb3f> boxes{boxOf(-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f)};
	u8 mask[CLAUDE_NBOX_MASK_BYTES];
	memset(mask, 0, sizeof(mask));

	UASSERTEQ(int, claudeRasterizeBoxes(boxes, mask), 4096);
	for (int i = 0; i < CLAUDE_NBOX_MASK_BYTES; i++)
		UASSERTEQ(int, mask[i], 0xFF);
}

void TestClaudeNodeBox::testSlabIsTheBottomHalf()
{
	std::vector<aabb3f> boxes{boxOf(-0.5f, -0.5f, -0.5f, 0.5f, 0.0f, 0.5f)};
	u8 mask[CLAUDE_NBOX_MASK_BYTES];
	memset(mask, 0, sizeof(mask));

	UASSERTEQ(int, claudeRasterizeBoxes(boxes, mask), 2048);
	for (int z = 0; z < 16; z++)
	for (int x = 0; x < 16; x++) {
		for (int y = 0; y < 8; y++)
			UASSERT(claudeMaskBit(mask, x, y, z));
		for (int y = 8; y < 16; y++)
			UASSERT(!claudeMaskBit(mask, x, y, z));
	}
}

// The shape gate 1 asks a human to recognise. Asserted here as bits so
// that "is the mask right" and "does the light honour the mask" are two
// separate questions with two separate answers.
void TestClaudeNodeBox::testStairIsAnLShape()
{
	std::vector<aabb3f> boxes;
	stairBoxes(&boxes);
	u8 mask[CLAUDE_NBOX_MASK_BYTES];
	memset(mask, 0, sizeof(mask));

	// 2048 (bottom slab) + 1024 (upper step, half in z) and no overlap
	UASSERTEQ(int, claudeRasterizeBoxes(boxes, mask), 3072);

	for (int x = 0; x < 16; x++)
	for (int z = 0; z < 16; z++) {
		for (int y = 0; y < 8; y++)
			UASSERT(claudeMaskBit(mask, x, y, z)); // slab: solid throughout
		for (int y = 8; y < 16; y++) {
			// THE STEP LINE: air over the -z half, solid over the +z half
			bool want = z >= 8;
			if (claudeMaskBit(mask, x, y, z) != want) {
				rawstream << claudeMaskAscii(mask);
				UASSERT(false);
			}
		}
	}
}

// STEP 0's landmine, demonstrated rather than argued: the same content id
// with a different param2 is a DIFFERENT SHAPE. Luanti's own
// transformNodeBox rotates a NODEBOX_FIXED box list by facedir, so this
// is true of stairs and slabs too, not only of NODEBOX_LEVELED /
// NODEBOX_WALLMOUNTED as the handoff's landmine list reads.
void TestClaudeNodeBox::testParam2RotatesTheStep()
{
	UASSERT(s_stair != CONTENT_IGNORE);
	const ContentFeatures &f = s_ndef->get(s_stair);

	// facedir 0..3 = the four horizontal yaws. The step must sit against
	// a different face each time, and the bit count must not change: a
	// rotation moves geometry, it never creates or destroys it.
	// Directions READ OFF transformNodeBox, not guessed: facedir 1 applies
	// rotateXZBy(-90), i.e. (x,z) -> (z,-x), so the +z step lands on +x.
	//   0 -> step on +z   1 -> step on +x   2 -> step on -z   3 -> step on -x
	// (The first guess written here had 1 and 3 swapped, and this test is
	// what said so -- before a lit scene could have blamed the shader.)
	struct { u8 p2; int ax; int lo; } want[4] = {
		{0, 2, 8}, {1, 0, 8}, {2, 2, -8}, {3, 0, -8},
	};
	u8 prev[CLAUDE_NBOX_MASK_BYTES];
	memset(prev, 0, sizeof(prev));

	for (int r = 0; r < 4; r++) {
		MapNode n(s_stair, 0, want[r].p2);
		std::vector<aabb3f> boxes;
		transformNodeBox(n, f.node_box, s_ndef, &boxes, 0);
		UASSERTEQ(size_t, boxes.size(), 2u);

		u8 mask[CLAUDE_NBOX_MASK_BYTES];
		memset(mask, 0, sizeof(mask));
		UASSERTEQ(int, claudeRasterizeBoxes(boxes, mask), 3072);

		// upper half: solid exactly on the named half of the named axis
		for (int x = 0; x < 16; x++)
		for (int y = 8; y < 16; y++)
		for (int z = 0; z < 16; z++) {
			int c = want[r].ax == 0 ? x : z;
			bool solid_hi = want[r].lo > 0;
			bool want_bit = solid_hi ? (c >= 8) : (c < 8);
			if (claudeMaskBit(mask, x, y, z) != want_bit) {
				rawstream << "facedir " << (int)want[r].p2 << ":\n"
						<< claudeMaskAscii(mask);
				UASSERT(false);
			}
		}
		// and each rotation is genuinely a different mask
		if (r > 0)
			UASSERT(memcmp(mask, prev, sizeof(mask)) != 0);
		memcpy(prev, mask, sizeof(mask));
	}
}

// Every vanilla nodebox edge is a multiple of 1/16 m. Centre sampling
// must put those edges exactly on the subvoxel line, with no off-by-one
// on either side, or a slab is 9/16 thick on one face and 7/16 on the
// other and the error is invisible until a shadow lands wrong.
void TestClaudeNodeBox::testEdgesLandOnSubvoxelLines()
{
	struct { float lo, hi; int i0, i1; } cases[] = {
		{-0.5f, 0.5f, 0, 15},   // full
		{-0.5f, 0.0f, 0, 7},    // lower half
		{ 0.0f, 0.5f, 8, 15},   // upper half
		{-0.5f, -0.4375f, 0, 0},// exactly one subvoxel (1/16)
		{ 0.4375f, 0.5f, 15, 15},
		{-0.0625f, 0.0625f, 7, 8}, // two subvoxels straddling the centre
		{-0.375f, 0.375f, 2, 13},  // 6/16 in from each face
	};
	for (const auto &c : cases) {
		std::vector<aabb3f> boxes{
				boxOf(c.lo, -0.5f, -0.5f, c.hi, 0.5f, 0.5f)};
		u8 mask[CLAUDE_NBOX_MASK_BYTES];
		memset(mask, 0, sizeof(mask));
		claudeRasterizeBoxes(boxes, mask);
		for (int x = 0; x < 16; x++) {
			bool want = x >= c.i0 && x <= c.i1;
			if (claudeMaskBit(mask, x, 0, 0) != want) {
				rawstream << "x span [" << c.lo << "," << c.hi
						<< "] expected " << c.i0 << ".." << c.i1 << "\n"
						<< claudeMaskAscii(mask);
				UASSERT(false);
			}
		}
	}
}

// A pane or a carpet can be thinner than one subvoxel. It may round to
// nothing on that axis, and it may round to one layer, but it must never
// round to two: a fattened box is a shadow that is wrong in the direction
// nobody looks for (too much occlusion reads as "the lighting is dark",
// not as "the geometry is wrong").
void TestClaudeNodeBox::testSliverCannotFattenALayer()
{
	for (int k = 0; k < 32; k++) {
		float lo = -0.5f + k * (1.0f / 32.0f);
		std::vector<aabb3f> boxes{
				boxOf(-0.5f, lo, -0.5f, 0.5f, lo + 1.0f / 32.0f, 0.5f)};
		u8 mask[CLAUDE_NBOX_MASK_BYTES];
		memset(mask, 0, sizeof(mask));
		int bits = claudeRasterizeBoxes(boxes, mask);
		// 0 or exactly one 16x16 y-layer, never more
		UASSERT(bits == 0 || bits == 256);
	}
}

// mcl_panes' `pane_nodebox`, verbatim from
// games/mineclonia/mods/ITEMS/mcl_panes/init.lua. A post through the
// middle of the cell, and one arm per horizontal face that connects.
// This is the shape every window in the world is made of.
static void paneNodeBox(NodeBox *nb)
{
	nb->type = NODEBOX_CONNECTED;
	nb->fixed.push_back(boxOf(-1.f / 16, -0.5f, -1.f / 16,
			1.f / 16, 0.5f, 1.f / 16));
	auto &c = nb->getConnected();
	// front = -z, left = -x, back = +z, right = +x. The names and the
	// bit order are Luanti's (mapnode.cpp transformNodeBox):
	//   1 top, 2 bottom, 4 front(-z), 8 left(-x), 16 back(+z), 32 right(+x)
	c.connect_front.push_back(boxOf(-1.f / 16, -0.5f, -0.5f,
			1.f / 16, 0.5f, -1.f / 16));
	c.connect_left.push_back(boxOf(-0.5f, -0.5f, -1.f / 16,
			-1.f / 16, 0.5f, 1.f / 16));
	c.connect_back.push_back(boxOf(-1.f / 16, -0.5f, 1.f / 16,
			1.f / 16, 0.5f, 0.5f));
	c.connect_right.push_back(boxOf(1.f / 16, -0.5f, -1.f / 16,
			0.5f, 0.5f, 1.f / 16));
}

// Rasterize one pane at one neighbour state. `neighbors` is the byte
// MapNode::getNeighbors() produces in the real walk.
static int paneMask(u8 neighbors, u8 *mask)
{
	NodeBox nb;
	paneNodeBox(&nb);
	std::vector<aabb3f> boxes;
	// nodemgr is only dereferenced by the FIXED/LEVELED branch
	transformNodeBox(MapNode(CONTENT_AIR), nb, nullptr, &boxes, neighbors);
	memset(mask, 0, CLAUDE_NBOX_MASK_BYTES);
	return claudeRasterizeBoxes(boxes, mask);
}

// TYPE GATE, THE OTHER WAY ROUND SINCE 2026-08-23 (PANES).
//
// This test used to assert the opposite: NODEBOX_CONNECTED was refused,
// because the shape depends on NEIGHBOURS and a one-block re-walk at a
// block boundary could compute the wrong shape while the per-block hash
// called it converged. Every glass pane with a neighbour is this type, so
// refusing it refused every window.
//
// The deferral was paid off in game.cpp, not here, and in two places: the
// incremental dirty box is grown by ONE NODE on each axis before it is
// turned into grid blocks (so every cell whose 6-neighbourhood an edit
// touched is re-walked), and the neighbour byte is part of the shape
// cache key (so the mask id the block hash mixes moves when the shape
// moves). What stays true is the other half of the old assertion:
// fixed[] alone is never the answer — see the two pane tests below, where
// the arms carry most of the geometry.
void TestClaudeNodeBox::testConnectedIsConverted()
{
	NodeBox nb;
	paneNodeBox(&nb);
	UASSERT(claudeNodeBoxConvertible(nb));

	// a fence: a thin post in fixed[], rails only in the connect_* lists
	NodeBox fence;
	fence.type = NODEBOX_CONNECTED;
	fence.fixed.push_back(boxOf(-0.125f, -0.5f, -0.125f,
			0.125f, 0.5f, 0.125f));
	fence.getConnected().connect_front.push_back(
			boxOf(-0.0625f, 0.1875f, -0.5f, 0.0625f, 0.375f, -0.125f));
	UASSERT(claudeNodeBoxConvertible(fence));

	// A CONNECTED box with an EMPTY fixed[] is still ours: its whole body
	// can live in the connect_*/disconnected_* lists, and emptiness is
	// decided where the neighbours are known, not at this gate.
	NodeBox armsonly;
	armsonly.type = NODEBOX_CONNECTED;
	armsonly.getConnected().connect_left.push_back(
			boxOf(-0.5f, -0.5f, -0.0625f, 0.0f, 0.5f, 0.0625f));
	UASSERT(armsonly.fixed.empty());
	UASSERT(claudeNodeBoxConvertible(armsonly));

	// and the two refusals that have not moved
	UASSERT(!claudeNodeBoxConvertible(NodeBox()));   // REGULAR
	NodeBox emptyfixed;
	emptyfixed.type = NODEBOX_FIXED;
	UASSERT(!claudeNodeBoxConvertible(emptyfixed));  // FIXED with no boxes
	NodeBox fixed;
	fixed.type = NODEBOX_FIXED;
	stairBoxes(&fixed.fixed);
	UASSERT(claudeNodeBoxConvertible(fixed));
}

// NO NEIGHBOURS: the post and nothing else. 2x2 sub-voxels in x and z,
// full height — a glass POST, which is what Mineclonia draws for a pane
// standing on its own.
void TestClaudeNodeBox::testPaneAloneIsJustThePost()
{
	u8 mask[CLAUDE_NBOX_MASK_BYTES];
	int bits = paneMask(0, mask);
	UASSERTEQ(int, bits, 2 * 2 * 16);

	for (int x = 0; x < 16; x++)
	for (int y = 0; y < 16; y++)
	for (int z = 0; z < 16; z++) {
		bool want = (x == 7 || x == 8) && (z == 7 || z == 8);
		if (claudeMaskBit(mask, x, y, z) != want) {
			rawstream << "pane, no neighbours:\n" << claudeMaskAscii(mask);
			UASSERT(false);
		}
	}
}

// THE WORKED EXAMPLE, and it is what makes a window a window. A pane with
// a WEST (-x, bit 8 "left") and an EAST (+x, bit 32 "right") neighbour
// unions three boxes:
//
//   fixed          x [-1/16, 1/16]  ->  sub-voxels x 7..8
//   connect_left   x [-1/2, -1/16]  ->  sub-voxels x 0..6
//   connect_right  x [ 1/16,  1/2]  ->  sub-voxels x 9..15
//
// all of them z [-1/16, 1/16] (sub-voxels 7..8) and full height. So the
// mask is a 16 x 16 pane TWO SUB-VOXELS THICK: 512 bits, a continuous
// sheet of glass across the whole cell with no seam at the post. Before
// this step the same node baked as a full opaque 1 m cube — 4096 bits of
// stone behaviour where the window is.
void TestClaudeNodeBox::testPaneWestAndEastIsAFullSlab()
{
	u8 mask[CLAUDE_NBOX_MASK_BYTES];
	int bits = paneMask(8 | 32, mask);
	UASSERTEQ(int, bits, 16 * 16 * 2);

	for (int x = 0; x < 16; x++)
	for (int y = 0; y < 16; y++)
	for (int z = 0; z < 16; z++) {
		bool want = (z == 7 || z == 8);
		if (claudeMaskBit(mask, x, y, z) != want) {
			rawstream << "pane, west+east:\n" << claudeMaskAscii(mask);
			UASSERT(false);
		}
	}
}

// THE LANDMINE THE DEFERRAL WAS ABOUT, ASSERTED AS BITS: the same content
// and the same param2 are DIFFERENT SHAPES at different neighbour states.
// That is why the neighbour byte is in claudeNodeBoxMaskId()'s cache key
// and why the incremental walk grows its dirty box by a node — if either
// were missing, one of these masks would be served for the other and the
// per-block hash would call it converged.
void TestClaudeNodeBox::testPaneNeighboursAreDifferentShapes()
{
	// west+east (a sheet), north+south (the same sheet turned 90 deg),
	// west only (half a sheet), a corner, and all four.
	const u8 states[5] = {8 | 32, 4 | 16, 8, 8 | 4, 4 | 8 | 16 | 32};
	const int want_bits[5] = {512, 512, 64 + 224, 64 + 224 + 224,
			512 + 512 - 64};
	u8 masks[5][CLAUDE_NBOX_MASK_BYTES];
	for (int i = 0; i < 5; i++) {
		int bits = paneMask(states[i], masks[i]);
		if (bits != want_bits[i]) {
			rawstream << "neighbours " << (int)states[i] << " -> "
					<< bits << " bits, expected " << want_bits[i] << "\n"
					<< claudeMaskAscii(masks[i]);
			UASSERT(false);
		}
	}
	// west+east and north+south have the same COUNT and must not have the
	// same bits: a count-only check would pass a mask rotated 90 degrees
	// out of true, which is a window that looks right end-on and is a
	// wall from the side.
	for (int i = 0; i < 5; i++)
	for (int j = i + 1; j < 5; j++)
		UASSERT(memcmp(masks[i], masks[j], CLAUDE_NBOX_MASK_BYTES) != 0);
}
