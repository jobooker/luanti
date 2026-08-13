# Folding — the rulebook

*Started 2026-08-13. Folding = building each coarse rung of the pyramid from
the rung below. It is the load-bearing skill of the distance work: every
far-field artifact so far has traced to a fold defect (snow layers counted
as full solids → the +1 m LOD wall, fixed `a7be463`) or a missing folded
channel (flat single-color coarse cells; faceted far shading). Canonical
design lives in the vault (`projects/luanti-renderer-design.md` §6g);
this file is the fork-side working contract + bug bank + instruments.*

## Why folding works — the aliasing frame (John, 2026-08-13)

Aliasing IS information loss: sampling detail smaller than a pixel returns
noise instead of signal (the measured ~0.8% converged-pixel flicker). The
folded channels are anti-aliasing applied to each quantity — coverage is
anti-aliased occupancy, folded albedo is anti-aliased color, the folded
normal is anti-aliased *shading*. One law, all channels: **when detail
drops below the pixel, keep its average or lose it entirely.**

The mirror holds near the camera: up close, cubeness IS the information —
hard facets at resolvable scale are fidelity, not loss. Facets are signal
when the eye can resolve them, noise when it can't; the pixel is the
dividing line. So the `claude_far_normals` dial is not "facets vs smooth"
— it is "at what distance do I admit the eye can no longer resolve the
truth." A taste number, found on screen (see `fold_normals_ab.png`:
axis-aligned far shading collapses rolling hills to three brightness
values — shape deleted, not style).

## The contract

A fold maps a group of children (2³ or 4³ cells) to one parent cell,
channel by channel. Every fold MUST be:

1. **Light-independent.** Fold materials, never lit results. Albedo is a
   property of the stuff; if any lighting leaks into the paint, the paint
   goes stale the moment the sun moves. (Same law as §6e "nothing is
   painted, so nothing lies" — folding inherits it.)
2. **Camera-independent.** No view direction anywhere in the fold. This is
   what makes coarse cells better than imposters/billboards: they answer
   ray queries from ANY direction (sun, bounce, reflection), not just the
   camera's.
3. **Local.** A parent depends only on its own children, so an edit refolds
   just the touched cell's ancestor chain — a handful of cells, not the
   pyramid.
4. **Energy-honest.** Means are preserved: the parent's albedo is the mean
   of what a perfect renderer would have gathered from the children at
   sub-pixel scale. Prefiltering, not approximation-by-fiat. (The §3
   detail-over-average guarantee — mean ratio 1.0 — is this same law at
   texture scale.)

## Per-channel rules

| Channel | Fold | Notes |
|---|---|---|
| Occupancy | **coverage fraction** (0..1), not any-solid bit | The traversal *bit* is `coverage > threshold`; shading/transmittance should use the fraction. Classify partial nodes (snow layers, slabs, plants) BEFORE folding so they contribute fractional volume — the LOD-wall lesson. |
| Albedo | weighted mean of children albedo | Open question: volume-weighted vs **exposed-surface-weighted** (a buried dark core shouldn't darken a grassy hill — fold what a ray could actually see). Start volume-weighted, instrument, revisit. |
| Normal | mean of exposed-face normals, or occupancy gradient | Non-axis-aligned by construction (staircase → 45°). Shading-only; geometry stays cubes. Taste + stability dial `claude_far_normals` — see REVISIT.md and `fold_normals_ab.png`. |
| Emissive | sum of child power, per parent volume | A village of torches folds to a faint glowing cell — far towns twinkle for free. |
| Class/material | dominant class + its coverage | Keeps water/stone/leaf behavior (transmittance, roughness) roughly right at distance. |

## Render-time counterpart (what the fold feeds)

- Geometry/occlusion: march level chosen by pixel footprint (cell ≈ 1 px;
  keep each rung's cells in the ~1–4 px band via hand-off tuning).
- Albedo at a hit on level L: sample level L-1 (or L-2) at the hit point —
  "16 m block with 1 m texels", zero extra storage, +1 fetch per hit.
- Light: computed live every frame against coarse geometry + far irradiance
  cache. The fold never sees it (rule 1).

## Fold bug bank (add every one found)

- 2026-08-12: thin snow LAYERS counted as full 1 m solids in every fold →
  second rung ground +1 m, abrupt LOD wall. Fix: classify LUT + fractional
  contribution in `summarizeBlock` (`a7be463`). Stale far files must be
  resampled after a fix — fold bugs persist in cached summaries.

## Instruments

- `fold_normals_ab.py` → `fold_normals_ab.png`: three panels from ONE real
  1-unit voxel grid — FINE (the truth), A (folded 8-unit blocks, face
  normals), B (same fold, normals = area-weighted average of the fine
  terrain's actual exposed cube faces). The fold is genuine: coarse
  occupancy from the mean of fine columns, folded normal from summed face
  areas — a worked example of this file's per-channel rules. Runs with
  `~/.venvs/voice/bin/python` (any python with numpy+PIL).
- TODO golden fold tests: hand-build a tiny scene (known slab/layer/plant
  mix), fold it, assert coverage and albedo sums analytically. Cheap, and
  would have caught the snow-layer bug before it reached the screen.

## Open questions

- Volume-weighted vs exposure-weighted albedo fold (above).
- Color rung at hit: L-1 or L-2? (finer = more face detail, one more level
  of cache pressure; judge on screen.)
- Transmittance from coverage: Beer-Lambert per crossed partially-covered
  coarse cell (a 30%-solid cell dims light instead of blocking) — would
  soften far shadow edges honestly and kill the >50%-solid promote hack's
  self-shadow bias.
- Folded normals dial verdict (taste; after far irradiance cache lands).
