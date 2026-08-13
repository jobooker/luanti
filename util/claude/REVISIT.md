# Tricks worth revisiting (parked, not planned)

*Folding now has its own rulebook + instruments: see [FOLDING.md](FOLDING.md).*

*Captured 2026-08-13 from a design conversation (Evan Wallace's webgl demos,
graphics-API comparison, memory-layout audit of claude_accum). Standing design
docs live in the vault — `projects/luanti-renderer-design.md` is canonical;
this file is the fork-side reminder list so the ideas surface next session.*

## Memory layout — the biggest unclaimed constant factor

Audit of `claude_accum` (2026-08-13): the march is algorithmically grid-native
(cell-exact A&W DDA, mip-pyramid leaps, live 5-level far cascade) but the
*layout* isn't — each step pays 2–5 dependent texture fetches, and traversal
reads `.a` out of the RGBA8 `claudeVolume`, dragging 4-byte texels through
cache for a 1-byte question.

1. **Cheap instrument FIRST (an afternoon):** split the occupancy/class tag
   out of `claudeVolume.a` into its own R8 volume so traversal never touches
   the RGBA texture. Works even on GL 2.1. If frame time moves → we are
   bandwidth-bound and step 2 pays big; if not → bottleneck is elsewhere,
   stop here. Verify with `util/claude/verify_tracer.sh`.
2. **Bitmask bricks (needs `#version 410`, which gl41-core already earned):**
   pack each 4×4×4 brick's occupancy into 64 bits, upload as 32³ `RG32UI`
   (`usampler3D`). One integer fetch per *brick*; sub-stepping is
   `findLSB`/bit tests on a `uvec2` in registers — no memory. Whole 128³
   near-field occupancy = 256 KB, L2-resident. No int64 on Apple GL; uvec2 is
   the same 64 bits. **No Vulkan required** — bitCount/findLSB/usampler are
   GLSL 4.00+, and the march is per-pixel so fragment shaders suffice.
3. Smaller, same family: the coarse `textureLod` probe fires per CELL even
   inside occupied bricks — brick-granularity hierarchical DDA would cut the
   redundant probes. Per-cell Chebyshev distance field is an alternative /
   complement to the mip pyramid for empty-space leaps. Exact reprojection
   keys (voxel-ID + face instead of fuzzy depth) — already on the vault
   open-problems list, reaffirmed.

4. **Recursive 64-tree (John, 2026-08-13):** the bitmask brick generalizes —
   4³ = 64 voxels = one uint64, and 64 *bricks* = one uint64 one level up,
   recursively (the 16³ micro-grids are already two rungs: root mask + 64
   leaf masks, 520 B). Same primitive every level → the §6g "one code path,
   choose level, march" pyramid with its storage answer. Bit pyramid above
   128³ leaves: 4 KB + 62 B + 1 B — the whole skeleton ~260 KB. Caveats:
   bits are traversal only (color mips stay a parallel pyramid for the §6g
   merged-shading rule); descent costs steps, so likely flat+one-brick-level
   near, deep tree far; GLSL has no recursion — fixed-depth loop.

## Water — three liftable pieces from madebyevan.com/webgl-water/

- **Animated water normals** (highest vibe-per-effort): perturb the flat top
  face normal before reflect/refract. First step: flow-scrolled noise. Full
  version: his heightfield wave sim — a ~20-line ping-pong texture kernel
  (each texel averages 4 neighbours, wave equation, damped velocity),
  scene-independent, tiles over near-field water. = the "flow maps for water"
  item in the vault design doc §6b.
- **Refraction = one Snell bend, then keep DDA-ing.** His demo proves a single
  refraction event reads as fully convincing. Bend at the interface, continue
  the march underwater, tint by distance in-medium (Beer-Lambert — same math
  already chosen for leaves). Same shape fixes the "glass is opaque to rays"
  open problem. His hard-coded pool is our general case: the refracted ray
  just marches the grid.
- **Caustics** (beauty-pass, most work): render the water surface from the
  light direction, refract each point onto the floor, brightness = projected
  area ratio (screen derivatives), store in a caustics texture the floor
  samples. Needs heightfield-over-known-receiver — voxel water over voxel
  floor is exactly that. Dappled riverbeds; pairs with Riverflow.

## Far cells: decouple geometry rung from color rung (John, 2026-08-13)

"Far mountain = 16 m block with 1 m texels" — correct, and nearly free.
Don't store per-face textures (16x16/face over a 32^3 level = 144 MB;
4x4 = 9 MB). Instead: geometry/occlusion from level L, albedo sampled
from level L-1 (or L-2) at the hit point — colors the fold already
computes. 0 new bytes, +1 fetch per HIT. Folded albedo carries no
painted shading, so the §3 detail-over-average guarantee holds at scale
(mean 1.0 → can't fight traced light). The unifying rule, both ends of
the distance axis: bigger-than-a-pixel = geometry, smaller = averaged
color. Near converts texture→geometry; far converts geometry→texture.

## API verdicts — so we don't re-derive them

- API switch alone ≈ 0–20%. The 2–5× lives in representation (above).
- GL 4.1 (this branch) expresses the bitmask trick. Vulkan is wanted *later*
  for compute-shader radiance/face-cache passes (currently fragment-pass
  contortions over 2D atlases) + subgroup ops — and natively on the Linux rig.
- **RT cores: still not wanted.** They accelerate BVH-over-triangle-soup;
  a uniform grid IS its own acceleration structure and DDA beats renting
  hardware built for unstructured geometry. AMD's RT weakness is irrelevant
  to us (relevant when buying a GPU).
- Evan's "compile the scene into the shader" does not scale past toy scenes
  (instructions execute; every ray pays every inline test). The scaled
  equivalent is cache-resident occupancy: 256 KB in L2 + brick-in-register
  is the big-world version of constants-in-the-instruction-stream.

## Perf-measurement hygiene (Shadow PC 5 fps, unexplained as of 2026-08-13)

Before trusting any number from the Shadow box: `fps_max_unfocused = 200`
(the §6i trap — reads as ~10 fps and once caused a real bad decision),
confirm `video_driver` AFTER launch (client rewrites minetest.conf on exit —
a GL 2.1 run was once claimed as 4.1), vsync off. Then re-enable the
photo-mode de-optimizations one dial at a time and write the frame time down
at each step. Known magnitudes: coarse leap alone 59→35 ms; ungated
micro-march once cost 6×. Photo accuracy should come from temporal
accumulation while stationary (converges to the same image), not from paying
full cost every frame.

**Folded far normals — taste dial, decide LATER:** averaging fine normals
into coarse cells makes far terrain shade as slopes instead of facets.
Half taste (faceted may be MORE on-brand — §6h at landscape scale; Teardown
stays blocky at all distances), half correctness (fixes shading pops at rung
hand-offs, if they occur). Not evaluable until far irradiance cache + color
rung exist; then ship as `claude_far_normals` dial, judge on Everest scene
at golden hour.
