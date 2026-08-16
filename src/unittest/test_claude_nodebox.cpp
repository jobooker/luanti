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
	void testConnectedFallsBackWholeCell();
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
	TEST(testConnectedFallsBackWholeCell);
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

// DEFERRED TYPE, ASSERTED. NODEBOX_CONNECTED depends on neighbours, and
// the grid re-snap is incremental — a one-block re-walk at a block
// boundary can compute the wrong shape and the per-block hash will call
// it converged. So connected types are NOT converted, and this pins the
// fallback: a connected node must keep its full 1 m cube, never a
// half-converted shape built from the fixed[] list alone (which for a
// fence is the post, i.e. a fence whose rails have vanished).
void TestClaudeNodeBox::testConnectedFallsBackWholeCell()
{
	NodeBox nb;
	nb.type = NODEBOX_CONNECTED;
	// a fence: a thin post in fixed[], rails only in the connect_* lists
	nb.fixed.push_back(boxOf(-0.125f, -0.5f, -0.125f, 0.125f, 0.5f, 0.125f));
	auto &c = nb.getConnected();
	c.connect_front.push_back(boxOf(-0.0625f, 0.1875f, -0.5f,
			0.0625f, 0.375f, -0.125f));

	// The engine WOULD produce a shape here; the grid must not use it.
	std::vector<aabb3f> boxes;
	transformNodeBox(MapNode(CONTENT_AIR), nb, nullptr, &boxes, 0);
	UASSERT(!boxes.empty()); // proves the fallback is a choice, not an accident

	// claudeNodeBoxKind is the single place that decides. CONNECTED must
	// answer "not mine", so the walk leaves the cell class 255 / 1 m.
	UASSERT(!claudeNodeBoxConvertible(nb));
	UASSERT(!claudeNodeBoxConvertible(NodeBox())); // REGULAR too
	NodeBox fixed;
	fixed.type = NODEBOX_FIXED;
	stairBoxes(&fixed.fixed);
	UASSERT(claudeNodeBoxConvertible(fixed));
}
