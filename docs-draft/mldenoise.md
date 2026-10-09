---
title: "Learned denoiser prototype: the network against today's filter"
summary: "A 244k-parameter U-Net halves today's a-trous filter's pixel error at equal input on all four held-out scoreboard scenes, but costs about 9 ms per frame against the filter's 0.6-1.0 ms, and FLIP rates it worse on the cabin and the plains."
tags: [luanti, report]
author: claude
---

# Learned denoiser prototype: the network against today's filter

**Verdict: the quality is real, the cost is not yet.** Given the same noisy
frames, a small trained network (U-Net, 244k parameters) cuts the pixel error
(RMSE) of today's hand-tuned a-trous filter by **1.1-2.2x on all four held-out
scoreboard scenes**, at 1 frame and at the one-second frame counts. But it
costs about **9 ms per 960x540 frame** on this GPU (fp16, PyTorch), against the
filter's 0.6-1.0 ms. Even at equal time, which takes that cost out of the
frame budget, it still wins on RMSE (1.2-1.7x). FLIP, the perceptual judge,
splits: the network wins the forest, ties the torch room, and **loses the cabin
and the plains**. The reason is measured, not guessed: the network leaves fine
grain on flat walls, where the filter leaves blotches, and FLIP punishes grain
more. Against 0w's bar ("better quality for the same or better time") this is
not a ship. It is a strong signal that the quality is there to be had, once
the network gets about 10x cheaper and learns not to leave grain.

## The head-to-head (equal input)

The four scoreboard poses, never seen in training. The truth and the noisy
inputs were rendered in the same game session at the same pose. Each scene's
`grid_hash` and `area_emitters` matched between its truth and all three noisy
shots:

| scene | grid_hash | area_emitters |
|---|---|---|
| forest | 44b1c3e04c21aef5 | 1 |
| plains | 177276cd8234e788 | 16 |
| cabin | d8ffb3f26d7f4d05 | 16 |
| torchroom | d1bee6d929edd46f | 9 |

How it was run:
- **Noisy input:** the *anything* contender (play settings, denoiser on), at
  each scene's pinned scoreboard exposure, seeds 101-103. The network gets the
  accumulated radiance and guide buffers of that exact accumulation. Today's
  filter is that same frame's displayed output (`den`).
- **The three arms differ only in the denoiser.** Same samples, same truth,
  same exposure, same display transform (exposure, ACES, gamma 1/2.2), all at
  960x540.
- **Truth:** honest contender, 8192 frames, seed 0.
- **Numbers:** mean of 3 seeds. The spread between seeds is about ±0.0002 RMSE.

Each cell is noisy / today's filter / network. "Filter/net" is how many times
more RMSE the filter has (above 1.00x the network is ahead).

| scene | frames | RMSE noisy / filter / **net** | filter/net | FLIP noisy / filter / **net** |
|---|---|---|---|---|
| forest | 1 | 0.0599 / 0.0408 / **0.0213** | 1.92x | 0.252 / 0.129 / **0.112** |
| forest | 27 (one second) | 0.0415 / 0.0306 / **0.0171** | 1.79x | 0.183 / 0.107 / **0.093** |
| plains | 1 | 0.0245 / 0.0134 / **0.0078** | 1.72x | 0.098 / 0.045 / 0.046 |
| plains | 47 (one second) | 0.0117 / 0.0080 / **0.0041** | 1.93x | 0.071 / 0.029 / 0.030 |
| cabin | 1 | 0.0449 / 0.0113 / **0.0069** | 1.64x | 0.153 / **0.040** / 0.052 |
| cabin | 70 (one second) | 0.0172 / 0.0059 / **0.0035** | 1.66x | 0.086 / **0.029** / 0.031 |
| torchroom | 1 | 0.1441 / 0.0281 / **0.0244** | 1.15x | 0.410 / **0.114** / 0.121 |
| torchroom | 52 (one second) | 0.1527 / 0.0278 / **0.0220** | 1.26x | 0.424 / 0.117 / **0.112** |

All 16 captured depths (1-64 frames) are in `screenshots/mldenoise/test/unet/rows.json`.
The network is ahead on RMSE at every depth, scene and seed.

The one-second frame counts are the anything contender's from the scoreboard
run at 2026-10-08T17:16Z. The absolute values differ from the scoreboard page
because this scores at the trace resolution (960x540) against same-session
truths. The scoreboard scores the upscaled 1920x1080 picture against the
main checkout's truths. Comparisons inside this table are like for like.

## Equal time (the cost taken out of the budget)

**What each arm gets:**
- Today's filter: the anything contender's frames per second as the
  scoreboard measured them.
- The network: floor(1000 / (honest frame ms + network ms)) frames, scored at
  the deepest captured depth at or below that count. That is conservative
  for the network.

The network's cost is 8.8 ms, measured sustained (below).

| scene | filter: frames, RMSE, FLIP | network: frames (scored at), RMSE, FLIP | RMSE filter/net |
|---|---|---|---|
| forest | 27, 0.0306, 0.107 | 23 (20), **0.0183**, **0.097** | 1.68x |
| plains | 47, 0.0080, **0.029** | 31 (24), **0.0051**, 0.034 | 1.57x |
| cabin | 70, 0.0059, **0.029** | 38 (32), **0.0044**, 0.036 | 1.34x |
| torchroom | 52, 0.0278, 0.117 | 38 (32), **0.0234**, 0.116 | 1.19x |

At the one-frame (60 fps) budget the network does not fit: 8.8 ms on top of a
17-36 ms trace drops the frame rate.

## Cost per frame

| | ms per 960x540 frame |
|---|---|
| today's filter: six `claude_denoise` passes, on (scoreboard run 17:16Z, pass_ms[3..8]) | 0.62-1.02 |
| the same passes with the filter off (pass-through copies) | ~0.33 |
| **network, fp16**, sustained (200 frames back to back, wall clock) | **8.8** |
| network, fp16, replayed as a GPU graph (no Python launch cost) | 9.1 |
| network, fp32 | 14.4 |
| network, fp16, with input features and albedo-back-in included | ~12 |
| half-resolution variant (below), fp16 | 4.8 |

- **Throughput:** 9 ms for about 48 GFLOP is roughly 5.5 TFLOPS, well under
  what this card (RX 9070-class, RDNA4) can do. PyTorch's ROCm convolutions
  are slow at these small channel counts. Replaying the network as a GPU graph
  did not help, so the time is in the convolution kernels, not in Python's
  launch cost.
- **Timing method:** per-frame GPU-event timing, with a sync after each frame,
  came out bimodal (a 10th percentile of 2 ms, a median of 14 ms) and did not
  agree with wall clock. I trust the sustained wall clock above. The 2 ms
  readings are not a real cost.

## Pictures

Side-by-sides: noisy | today's filter | network | truth. The top row is 1 frame,
the bottom row is the one-second count, all for seed 101:

- `~/code/luanti-wt/mldenoise/screenshots/mldenoise/test/unet/side-by-side-forest.png`
- `.../side-by-side-plains.png`
- `.../side-by-side-cabin.png`
- `.../side-by-side-torchroom.png`

Every single picture (each arm, depth and seed) is next to them as PNG.

**What they show:**
- **Cabin:** today's filter keeps hard, stair-stepped edges at the grass
  ledges and the white block; that is the one-sample albedo. The network's
  edges are anti-aliased like the truth.
- **Flat walls (FLIP's complaint, measured):** I stretched the cabin's gray
  wall 8x around the truth's mean. The filter leaves soft blotches; the
  network leaves fine colour grain. Its pixel standard deviation is 1.3/255,
  against 0.5/255 for the truth and 0.3-0.6/255 for the filter. FLIP penalises
  that grain more than the blotches, so FLIP prefers the filter there even
  though the network's pixel error is half.
- **Torch room:** both the filter and the network show a horizontal band on
  the left wall at window height that the truth does not have. The noisy input
  has fireflies concentrated on that row of the wall. The filter keeps them as
  sparkles; the network smears them into a streak. It is not invented by the
  network, but it is not removed either.

## Training data recipe

- **Where:** 33 views in the gallery world, captured on the `mldenoise`
  branch with `util/claude_mldenoise_capture.py train`:
  - forest at seven spots (one of them at four hours of the day);
  - plains, four places;
  - the overlook (y 44.5), four headings;
  - the sky pad;
  - cozy-cabin interiors;
  - the Cornell room;
  - the skylight cave;
  - the glass rooms and the glass furnace;
  - the sealed plank house from outside.
- **Times of day:** 0.28-0.73.
- **Distance from the test poses:** every pose is at least 25 m from the four
  scoreboard poses and does not look back at them. The cozy interiors are
  15-20 m from the cabin pose but enclosed; no scoreboard frame sees inside
  them.
- **Held out for validation:** forest-far, plains-e, cozy-morning,
  cave-skylight-pm. That leaves 29 views and 232 samples for training.
- **Failed and skipped:** forest-e-dawn, sealed-outside and cave-skylight-am.
  The camera restarted or drifted under the shoot tool's checks.
- **Per view, two accumulations of the honest contender (denoiser off):**
  - A: seed sA, auto exposure, 256 frames. This is the scoreboard's exposure
    probe. Inputs are dumped at 1, 4, 16 and 64 frames, and the settled
    exposure is pinned.
  - B: seed sB, exposure pinned. Dumps at 1, 4, 16, 64 and 2048 frames; the
    2048-frame dump is the converged target.
- **What one dump holds** (`claude_dump_at`, a client change on this branch):
  accumulated radiance and its direct part, the guide (accumulated albedo and
  face code), the moments, and the displayed picture. A dump whose
  accumulation restarted after the schedule was armed is thrown away
  (`resets_since_arm`).
- **Correlation with the target:** B's shallow inputs share up to 64/2048 of
  their samples with the target. I measured A against B at 64 frames on every
  view: B is 3-5% closer, as expected, and nothing else differs.
- **GPU time:** about 68 minutes for the 33 views that succeeded (plus retries and skipped views), plus about 17 minutes for the
  test truths and inputs.
- **Size:** 20 GB in `~/data/mldenoise` (not in git).

## The network

**Inputs:** 20 channels at 960x540, computed from one dump:
- radiance divided by the accumulated albedo, times the exposure, through
  log(1+x);
- the direct part, the same way;
- sqrt(albedo);
- the normal, decoded from the face code;
- flags for sky, emissive or no surface, and mixed edge pixels;
- log distance;
- two flags for "same voxel face as the right / lower neighbour";
- the noise of the mean, from the moments;
- log N.

**Model and output:** a 3-level U-Net (24/32/48/64 channels) predicts a
residual on the log irradiance. The albedo is multiplied back in, and the
output is linear radiance.

**Why linear in, display-space loss:**
- **Output in linear radiance**, because everything downstream works in it:
  the present pass, auto exposure, white balance and the referees.
- **Input in log of demodulated radiance**, because a torch room at exposure
  4.7 and noon at 0.005 then look alike to the network, and texture is
  divided out, so it denoises the light, not the blocks. Today's filter does
  the same.
- **Loss: L1 in display space** (exposure, ACES, gamma: claude_present's
  transform). That is what the scoreboard scores and the eye sees, and it
  keeps the sun's x40 from dominating the gradient.

**Training:**
- AdamW at lr 5e-4 with cosine decay, about 21k steps in 25 minutes.
- Batch of 16 crops of 128x128, random horizontal flips.
- The display-space L1 fell from 0.0141 to 0.0089.

On the held-out views the network cut the noisy input's RMSE 1.8-2.5x
outdoors and 3-5.5x indoors.

## What did not work

- **Half-resolution network** (pixel unshuffle in, pixel shuffle out,
  32/48/64/96 channels): about half the cost (4.8 ms), but it lost to today's
  filter on the cabin (RMSE 0.0114 against 0.0059 at one second) and the
  torch room (0.068 against 0.028), and on FLIP everywhere. It reached a
  training loss of 0.0133, against 0.0089 for the full-resolution network.
  The shortcut throws away the per-pixel edges the filter's same-face test
  uses. Results are in `screenshots/mldenoise/test/half/`.
- **First training run: NaN from step 0.** The raw second moment reaches 4e9
  in a dusk interior and overflowed the fp16 the samples were stored in. Fixed
  by turning the moments into the noise of the mean in float32 at load time.
  The trainer now also refuses non-finite data or a non-finite loss.

## Found along the way

- **The torch room's linear truth in the main checkout is stale.** Its
  `truth.f32` (11:22) predates its `truth.png` and meta (13:16); through the
  display transform it is 0.044 RMSE from its PNG and darker. The other three
  scenes agree within 0.009-0.02. Not used here: every truth was re-rendered
  in session.
- **Dump rows are top-down**, the same order as the PNG. This contradicts the
  note in the brief. Flipped, they mismatch the PNG by 0.15-0.42 RMSE;
  unflipped, by 0.009-0.02.
- **The shoot tool can pass a shot whose accumulation restarted.** Twice
  (forest-e-B, forest-se-B) it passed still_frames and drift, yet the dump
  saw a restart after the shutter was armed. `resets_since_arm` catches it.
- **Pose height matters.** Poses at y 9 or 12 under `--play` jitter by
  0.05-0.2 units a frame and never settle; y 8.5 holds.

## What would make it better, cheapest first

1. **Stop the grain.** Add a term to the loss that punishes high-frequency
   error, or train on FLIP directly. A loss on image gradients is the usual
   choice. This targets exactly where FLIP prefers the filter. The training
   targets (2048 frames) also carry some grain of their own; 8192-frame targets
   for the flat-wall views would help.
2. **Make it 10x cheaper without the half-resolution shortcut.**
   - Predict per-pixel blend weights for a few a-trous passes instead of
     colours (a kernel-predicting network). Most of the work stays in today's
     cheap filter passes, and the network can only re-weigh real samples, so
     it cannot invent light.
   - Fewer full-resolution channels (12-16) with the depth at 1/4 resolution.
   - A fused fp16 shader in the engine instead of PyTorch's kernels.
3. **More and harder data.**
   - Interiors lit through openings, like the torch room: its 1.15x is the
     weakest win.
   - Moving-camera inputs: today's tests are still frames only.
   - More times of day and torchlit nights.
4. **History as an input** (the previous output, reprojected). That is where
   real-time denoisers get most of their quality, and it is the "history
   trust" row of 0w.

## Could not do

- **No in-engine run.** All numbers are offline on dumped buffers, at the
  trace resolution, not through the engine's upsampler. The scoreboard's
  1920x1080 numbers are not reproduced here.
- **No motion test.** Every test is a still camera.
- **Truth depth:** 8192 frames per scene, not the scoreboard's 16384. The
  truth's own noise is the same for every arm, so it moves no comparison.

## Tools (branch `mldenoise`)

- `src/client/game.cpp`, `src/client/render/secondstage.{h,cpp}`:
  `claude_dump_at`.
- `util/claude_mldenoise_capture.py`: `train` / `test` / `truth` / `batch`,
  holding the GPU lock.
- `util/claude_mldenoise.py`: `train` / `eval` / `test` / `time`.
- `util/claude_mldenoise_time.py`: sustained and graph timing.
- `util/claude_mldenoise_score.py`: FLIP and the side-by-sides (judge venv).
- `util/claude_mldenoise_equaltime.py`: the equal-time table.

Reproduce:
1. `python3 util/claude_mldenoise_capture.py batch ~/data/mldenoise/batch_capture.txt`
2. Then the same with `batch_learn.txt`.
3. Then `~/.venvs/judge/bin/python util/claude_mldenoise_score.py screenshots/mldenoise/test/unet/rows.json`.
