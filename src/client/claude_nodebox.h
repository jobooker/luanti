// Luanti — sub-metre solids read from ContentFeatures::node_box
// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Stairs, slabs and beds already carry their true shape in Luanti's own
// node definitions (`ContentFeatures::node_box`), and the engine's mesh
// generator and raycast both read it. The trace grid does not: every
// solid cell is an honest 1 m cube unless an AUTHORED 16^3 model claims
// it (game.cpp, "THE VOXEL LAW").
//
// This is the missing SOURCE: it turns a node's box list into exactly
// the 16^3 occupancy mask the authored models produce, so both feed one
// consumer and there is no second sub-voxel format to keep in agreement.
//
// The rasterizer is split out of game.cpp deliberately: it is pure
// geometry with no client, no GL and no map, so it can be tested against
// a hand-written box list directly instead of by looking at a lit scene.
// That is the instrument spec/handoffs/2026-08-16-nodebox-geometry.md
// prescribes for the two-miss rule, built BEFORE the first miss.

#pragma once

#include "irr_aabb3d.h"
#include "irrlichttypes.h"
#include <string>
#include <vector>

struct NodeBox;

// Bytes in one 16^3 occupancy mask. Layout is the authored-model layout,
// verbatim (game.cpp claudeLoadModels):
//
//     byte = (z * 16 + y) * 2 + (x >> 3),  bit = 1 << (x & 7)
//
// so one byte is 8 x-subvoxels and a (z,y) row is 2 bytes. Do not invent
// a second packing: claudeTraceGridBakeSubvox copies these bytes into the
// R8 64x512x512 ring texture without reinterpreting them.
static constexpr int CLAUDE_NBOX_MASK_BYTES = 512;

// THE ONE PLACE that decides whether the trace grid reads a node's real
// geometry or leaves it a 1 m cube. Kept as a function rather than an
// inline test at the call site so the deferral below is testable.
//
//   NODEBOX_REGULAR   -> NO. A full cube is already a full cube, and a
//                        nodebox path that fires here would move cornell
//                        and the furnaces (handoff gate 2).
//   NODEBOX_FIXED     -> YES. Stairs, slabs, beds. transformNodeBox also
//                        rotates these by facedir, so the shape depends
//                        on param2 — see claudeTraceGridWalkBlock's hash.
//   NODEBOX_LEVELED   -> not yet.
//   NODEBOX_WALLMOUNTED -> not yet.
//   NODEBOX_CONNECTED -> DEFERRED, DELIBERATELY. The shape depends on
//                        NEIGHBOURS, and the grid re-snap re-walks single
//                        dirty blocks: a fence at a block boundary can be
//                        re-walked without its neighbour and the per-block
//                        hash will happily call the wrong shape converged.
//                        Falling back to the whole cell is the honest
//                        answer; falling back to fixed[] alone would draw
//                        a fence post with no rails.
bool claudeNodeBoxConvertible(const NodeBox &nb);

// Rasterize a node's box list (node-local, BS units, i.e. -BS/2..+BS/2 on
// each axis — what MapNode::getNodeBoxes hands back) into `mask`, which
// must hold CLAUDE_NBOX_MASK_BYTES and is NOT cleared by this call.
//
// A sub-voxel is set when its CENTRE lies inside a box. Centre sampling,
// not edge rounding: the handoff's `floor((c + 0.5) * 16)` is the same
// rule at a box edge that lands on a 1/16 line (every vanilla stair and
// slab edge does), but centre sampling is also correct for the edges that
// do not, and it can neither fatten a box by a subvoxel nor drop one.
//
// Returns the number of bits set (0..4096).
//
// CAUTION, measured 2026-08-16 and not what it looks like: a full-cube
// result does NOT mean "nothing came in". transformNodeBox's final else
// is `boxes.emplace_back(-BS/2,...,BS/2)` — NODEBOX_REGULAR hands back a
// FULL-CUBE BOX, not an empty list. So a caller that gates on "the box
// list was non-empty" fires on every plain cube in the world. Gate on
// claudeNodeBoxConvertible() (the type), and treat 4096 bits as "this
// mask carries no sub-voxel information" rather than as a shape.
int claudeRasterizeBoxes(const std::vector<aabb3f> &boxes, u8 *mask);

// Read one bit back. Instrument helper; the bake reads bytes directly.
bool claudeMaskBit(const u8 *mask, int x, int y, int z);

// Human-legible dump of one 16^3 mask: 16 y-slices, '#' solid and '.'
// air, laid out x across and z down, with a per-slice bit count. Reads
// on brightness/pattern/shape alone — no colour anywhere, so it is
// legible to a colourblind reader (environment law) and in a plain log.
std::string claudeMaskAscii(const u8 *mask);
