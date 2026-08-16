// Luanti — sub-metre solids read from ContentFeatures::node_box
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "claude_nodebox.h"
#include "constants.h"
#include "nodedef.h"
#include <algorithm>
#include <cmath>

bool claudeNodeBoxConvertible(const NodeBox &nb)
{
	return nb.type == NODEBOX_FIXED && !nb.fixed.empty();
}

bool claudeMaskBit(const u8 *mask, int x, int y, int z)
{
	if (x < 0 || y < 0 || z < 0 || x > 15 || y > 15 || z > 15)
		return false;
	return (mask[(z * 16 + y) * 2 + (x >> 3)] >> (x & 7)) & 1;
}

int claudeRasterizeBoxes(const std::vector<aabb3f> &boxes, u8 *mask)
{
	// Half-open [lo, hi] in 1/16 units, from a box edge in BS units.
	// Sub-voxel i covers [i/16, (i+1)/16], so its centre is (i+0.5)/16
	// and the test "centre inside the box" is
	//     lo * 16 - 0.5  <=  i  <=  hi * 16 - 0.5
	// with lo/hi the box edge normalised to 0..1. Both ends round the
	// same way, which is what stops a box from growing or shrinking by
	// a subvoxel depending on which edge it sits on.
	auto span = [](float e_min, float e_max, int *i0, int *i1) {
		float lo = e_min / BS + 0.5f;
		float hi = e_max / BS + 0.5f;
		*i0 = std::max(0, (int)std::ceil(lo * 16.0f - 0.5f));
		*i1 = std::min(15, (int)std::floor(hi * 16.0f - 0.5f));
	};

	int bits = 0;
	for (const aabb3f &b : boxes) {
		// A box handed back by transformNodeBox is repaired (min <= max),
		// but a Lua-authored one need not be, and a reversed box would
		// silently rasterize to nothing rather than to the shape the
		// mesh generator draws. Normalise rather than trust.
		float x0 = std::min(b.MinEdge.X, b.MaxEdge.X);
		float x1 = std::max(b.MinEdge.X, b.MaxEdge.X);
		float y0 = std::min(b.MinEdge.Y, b.MaxEdge.Y);
		float y1 = std::max(b.MinEdge.Y, b.MaxEdge.Y);
		float z0 = std::min(b.MinEdge.Z, b.MaxEdge.Z);
		float z1 = std::max(b.MinEdge.Z, b.MaxEdge.Z);

		int ix0, ix1, iy0, iy1, iz0, iz1;
		span(x0, x1, &ix0, &ix1);
		span(y0, y1, &iy0, &iy1);
		span(z0, z1, &iz0, &iz1);
		if (ix0 > ix1 || iy0 > iy1 || iz0 > iz1)
			continue; // a box thinner than 1/16 m that misses every centre

		for (int z = iz0; z <= iz1; z++)
		for (int y = iy0; y <= iy1; y++) {
			u8 *row = &mask[(z * 16 + y) * 2];
			for (int x = ix0; x <= ix1; x++) {
				u8 &by = row[x >> 3];
				u8 bit = (u8)(1 << (x & 7));
				if (!(by & bit)) {
					by |= bit;
					bits++;
				}
			}
		}
	}
	return bits;
}

std::string claudeMaskAscii(const u8 *mask)
{
	std::string out;
	for (int y = 15; y >= 0; y--) {
		int n = 0;
		for (int z = 0; z < 16; z++)
			for (int x = 0; x < 16; x++)
				n += claudeMaskBit(mask, x, y, z) ? 1 : 0;
		out += "y=";
		if (y < 10)
			out += ' ';
		out += std::to_string(y);
		out += "  (";
		out += std::to_string(n);
		out += "/256)\n";
		// z increases DOWNWARD in the dump, x rightward: looking down at
		// the node from above, which is how a stair's tread reads.
		for (int z = 0; z < 16; z++) {
			out += "      ";
			for (int x = 0; x < 16; x++)
				out += claudeMaskBit(mask, x, y, z) ? '#' : '.';
			out += '\n';
		}
	}
	return out;
}
