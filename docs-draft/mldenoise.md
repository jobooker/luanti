---
title: "Learned denoiser prototype: learned weights for today's filter clear the bar on the four scoreboard scenes"
summary: "A 20k-parameter network that only re-weighs today's a-trous passes beats the filter on RMSE and FLIP at equal input and at equal time on all four held-out scenes, at 1.47 ms per frame in PyTorch (plus an estimated ~1 ms for the passes' extra reads); the U-Net, even with a gradient loss, loses on time."
tags: [luanti, report]
author: claude
---

# Learned denoiser prototype

## Part 2 (2026-10-09): learned weights for today's filter

**Verdict: on these four held-out scenes, yes, it clears John's bar.** A small
network (20k parameters) that only re-weighs today's a-trous filter beats the
filter on both RMSE and FLIP:
- **At equal input:** at 1 frame and at the one-second frame counts, on all
  four scenes. The one exception is a FLIP tie on the plains.
- **At equal time:** with its cost taken out of the one-second budget, on all
  four scenes.

**What it costs, and how much is measured:**
- **Network:** 1.47 ms per 960x540 frame, measured on this GPU (PyTorch,
  compiled, fp16, sustained). That includes computing its inputs and
  upsampling its maps, and is just under the 1.5 ms target.
- **Filter passes:** the weighted passes read more per tap than today's. That
  surcharge is **estimated, not measured**, at about 1x today's 0.6-1.0 ms
  passes.
- **How much error there is room for:** the tightest scene (plains) keeps its
  win up to 3.9 ms extra per frame. The total above is about 2.5 ms.

**The U-Net:**
- **The gradient term helps it:** FLIP on the cabin at 1 frame goes from 0.052
  to 0.045, and on the torch room at one second from 0.112 to 0.104.
- **But it still loses at equal time** on the plains and the cabin (FLIP),
  because it costs 8.4 ms.

**Caveats:**
- **No in-engine run yet.** Both costs above are PyTorch measurements or
  estimates.
- **Four scenes, all still frames.** Nothing here tests a moving camera.

### What was built

- **A faithful port of today's filter**
  (`util/claude_mldenoise_atrous.py`). Every part is ported, none fitted:
  - the six passes;
  - the 7x7 young-history noise estimate and the moments path;
  - the 3x3 variance smoothing;
  - SIGMA_L 4 and the B3 kernel at steps 1-16;
  - the exact same-face test on the face code;
  - the young-pixel fade (claude_denoise_young 64, the game's default);
  - albedo divided out and put back.

  **Checked against what the engine displayed** (the `den` dumps of the
  anything contender), on all four scenes at 1, 4, 16 and 64 frames and the
  deepest depth: relative error at most **4.8e-6**, display-space RMSE between
  port and engine **0.000000**. It is the engine's filter to float rounding.
- **The weight network,** at 1/4 resolution: two 3x3 convs, one at 1/8
  resolution, one more at 1/4, then a 1x1 output. Its maps are upsampled
  bilinearly, which a shader's texture fetch does for free. Per pixel it
  supplies:
  - a factor on each pass's luminance edge-stopping width (5 maps);
  - 4 affinity features: a tap's weight is multiplied by exp(-|f_p - f_q|^2);
  - softmax weights over the six stages (unfiltered, after passes 1-5): how
    far to blur here.

  **Why it cannot invent light:** the output is a convex blend of same-face
  weighted averages of the pixel's own neighbourhood, so it cannot create
  light no sample carried. At zero weights it is today's filter (the
  stage-5 weight starts at 0.99997, measured 0.36% max deviation from the
  filter at init).

  **What it learned** (held-out views):
  - widen the brightness test 2.5-16x at pass 1 and pass 5;
  - tighten it to 0.3-0.5x at passes 2-4 on young pixels;
  - blend a few percent of pass 1 back in on the forest.
- **The loss** (idea 1): L1 in display space plus the L1 of the difference of
  neighbouring-pixel differences (the gradient term), weight 1. That weight
  is hand-picked: `TUNED | learn by: FLIP on the held-out views at 0 / 0.5 /
  1 / 2`. **Not separated:** the weight net was trained only with the
  gradient term, so its gain is not split between the architecture and the
  loss. Only the U-Net has the with/without-gradient-term comparison.
- **Training:** same data as part 1 (29 views, 232 samples), crops 256,
  batch 8, AdamW lr 1e-3, 25 min. That came to 4.1k steps for the weight net
  (it runs the six filter passes inside the training loop), against 21k for
  the U-Nets.

### Head-to-head: equal input

The four scoreboard poses, truths and inputs from the same session (as in
part 1), mean of 3 seeds. Each cell is RMSE / FLIP, lower is better.

| scene | frames | noisy | today's filter | U-Net | U-Net + grad | **weight net** |
|---|---|---|---|---|---|---|
| forest | 1 | 0.0599 / 0.252 | 0.0408 / 0.129 | 0.0213 / 0.112 | **0.0204 / 0.108** | 0.0324 / 0.122 |
| forest | 27 | 0.0415 / 0.183 | 0.0306 / 0.107 | 0.0171 / 0.093 | **0.0159 / 0.088** | 0.0243 / 0.098 |
| plains | 1 | 0.0245 / 0.098 | 0.0134 / 0.045 | 0.0078 / 0.046 | **0.0074** / 0.044 | 0.0110 / **0.044** |
| plains | 47 | 0.0117 / 0.071 | 0.0080 / 0.029 | **0.0041** / 0.030 | 0.0042 / 0.030 | 0.0054 / **0.027** |
| cabin | 1 | 0.0449 / 0.152 | 0.0113 / 0.040 | 0.0069 / 0.052 | **0.0065** / 0.045 | 0.0084 / **0.037** |
| cabin | 70 | 0.0172 / 0.086 | 0.0059 / 0.029 | **0.0035** / 0.031 | **0.0035** / 0.029 | 0.0043 / **0.027** |
| torchroom | 1 | 0.1441 / 0.410 | 0.0281 / 0.114 | 0.0244 / 0.121 | **0.0236** / 0.119 | 0.0258 / **0.107** |
| torchroom | 52 | 0.1527 / 0.424 | 0.0278 / 0.117 | 0.0220 / 0.112 | **0.0200** / 0.104 | 0.0216 / **0.097** |

### Head-to-head: equal time (the one-second budget)

How each arm's frames are counted:
- **Frames:** floor(1000 / frame ms). Each arm is scored at the deepest
  captured depth at or below that count, which is conservative for every arm
  except today's filter.
- **Frame times:** the scoreboard's (run 20261008-131615).
  - **Today's filter:** the anything contender.
  - **U-Nets:** honest + 8.8 / 8.4 ms.
  - **Weight net:** anything + 1.47 ms + the surcharge, estimated as today's
    passes' engine ms (0.62-1.02) x 0.96. The 0.96 is the PyTorch ratio,
    weighted passes over plain passes, minus 1.

| scene | today's filter | U-Net | U-Net + grad | **weight net** |
|---|---|---|---|---|
| forest | 27 fr: 0.0306 / 0.107 | 22 (20): 0.0183 / 0.097 | 23 (20): **0.0171 / 0.093** | 26 (24): 0.0251 / 0.100 |
| plains | 47 fr: 0.0080 / **0.029** | 31 (24): **0.0051** / 0.034 | 31 (24): **0.0050** / 0.034 | 42 (40): 0.0058 / **0.029** |
| cabin | 70 fr: 0.0059 / 0.029 | 37 (32): 0.0044 / 0.035 | 38 (32): **0.0042** / 0.032 | 60 (56): 0.0045 / **0.028** |
| torchroom | 52 fr: 0.0278 / 0.117 | 38 (32): 0.0234 / 0.116 | 38 (32): **0.0212** / 0.109 | 47 (40): 0.0221 / **0.098** |

**Against today's filter at equal time:**
- **The weight net** wins RMSE on all four scenes and FLIP on three, ties FLIP
  on the plains (0.029 against 0.029).
- **Both U-Nets** lose FLIP on the plains and the cabin.

**How much cost the weight net can absorb.** I read the extra ms per frame at
which its win still holds straight off the per-depth table: the depth at
which it matches the filter's one-second RMSE and FLIP. Captured depths
limit the precision, so these are conservative.

| scene | still wins with up to |
|---|---|
| forest | 14.3 ms extra |
| plains | 3.9 ms extra |
| cabin | 6.6 ms extra |
| torchroom | any (it beats the filter's one-second picture from 1 frame) |

Against the ~2.5 ms of network plus estimated surcharge, plains has 1.4 ms of
margin.

**The 60 fps budget:** every arm gets 1 frame. The weight net wins there too,
but it adds about 1.5 ms plus the surcharge to a frame.

### Cost per 960x540 frame (sustained wall clock, 200 frames back to back)

| | ms |
|---|---|
| today's filter in the engine (pass_ms 3..8, filter on): forest / plains / cabin / torchroom | 0.74 / 1.02 / 0.69 / 0.62 |
| **weight net, fp16, compiled: its inputs + network + maps upsampled to full resolution** | **1.47** |
| the same without the final upsample (a shader's bilinear fetch does it) | 1.28 |
| weight net, fp32, eager | 3.6 |
| the port of today's filter in PyTorch, compiled / eager | 4.3 / 88.8 |
| the port with the learned weights, compiled | 8.4 |
| U-Net + grad, fp16 | 8.4 |

**Reading the cost honestly:**
- **PyTorch is a poor stand-in for a shader.** The identical filter takes
  4.3 ms compiled in PyTorch against 0.6-1.0 ms in the engine. That suggests
  the network would cost less as a shader too, but I have not measured it.
  The 1.47 ms above is the PyTorch number.
- **The surcharge is an estimate.** It is the PyTorch ratio (weighted 8.4 ms
  over plain 4.3 ms) applied to the engine's passes. The weighted passes read
  4 feature channels per tap and one stage map per pass.

**What the first design taught** (MIOpen pathologies found by micro-benchmark,
`util/claude_mldenoise_bench.py` and `_prof.py`):
- A dilated 3x3 conv at 1/4 resolution took 12-16 ms, against 0.16 ms for a
  plain one.
- A full-resolution 1x1 conv took 1.0 ms (fp32), against 0.3 ms as a matrix
  product. In channels-last layout it took 2.4-9.2 ms.
- The first weight net, with a full-resolution head, cost 34 ms; with the
  dilation removed, 2.5-3 ms. Moving everything to 1/4 resolution got it to
  1.47 ms.

### Pictures

Noisy | today's filter | U-Net | U-Net + grad | weight net | truth. The top
row is 1 frame, the bottom row is the one-second count, seed 101:

- `~/code/luanti-wt/mldenoise/screenshots/mldenoise/test/h2h/side-by-side-forest.png`
- `.../side-by-side-plains.png`
- `.../side-by-side-cabin.png`
- `.../side-by-side-torchroom.png`

The head-to-head table is `.../test/h2h/h2h.md`.

**What they show:**
- **Cabin wall, stretched 8x:** the weight net keeps the filter's character,
  soft blotches with pixel spread 0.51/0.26/0.54 (the filter's is
  0.51/0.30/0.57, the truth's 0.66/0.51/0.81). It does not have the U-Net's
  grain (0.79/1.30/1.33). That is why FLIP likes it.
- **Torch room:** it softens the filter's firefly band at window height.
- **Cabin edges:** it keeps the filter's stair-stepped edges. Re-weighing
  cannot fix the one-sample albedo; the U-Net could. That is the next thing
  to give it.

### What would make it better next

1. **Measure it in the engine.** Port the weight net and the weighted passes
   to shaders behind a dial, and price them with the scoreboard's own frame
   timing. That replaces both the PyTorch cost and the surcharge estimate.
2. **Train longer, and split the effects.** It got 4.1k steps against the
   U-Net's 21k. Its batch loss (0.045 to 0.043) is too noisy to say whether
   it had stopped improving. Also separate the
   gradient term's effect on it (train without the term).
3. **Let it fix the albedo edges.** A per-pixel albedo-blend map (the
   accumulated albedo of neighbours on the same face) would cover the one
   thing the U-Net still does better.
4. **Motion.** Every test is a still frame; the weights should also learn
   history trust (0w's "history trust" row).

### Tools added

- `util/claude_mldenoise_atrous.py`: `check` (the port against the engine),
  `train`, `test`, `time`.
- `util/claude_mldenoise_h2h.py`: the head-to-head tables and the six-panel
  side-by-sides.
- `util/claude_mldenoise_wcheck.py`: the held-out check on the CPU, plus what
  the network asks for.
- `util/claude_mldenoise_bench.py` and `util/claude_mldenoise_prof.py`: the
  MIOpen micro-benchmarks.
- `util/claude_mldenoise.py`: new `--grad` option.
- `util/claude_mldenoise_time.py`: one output file per tag.

## Part 1 (2026-10-09 early): the U-Net against today's filter

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

### The head-to-head (equal input)

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

### Equal time (the cost taken out of the budget)

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

### Cost per frame

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

### Pictures

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

### Training data recipe

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

### The network

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

### What did not work

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

### Found along the way

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

### What would make it better, cheapest first

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

### Could not do

- **No in-engine run.** All numbers are offline on dumped buffers, at the
  trace resolution, not through the engine's upsampler. The scoreboard's
  1920x1080 numbers are not reproduced here.
- **No motion test.** Every test is a still camera.
- **Truth depth:** 8192 frames per scene, not the scoreboard's 16384. The
  truth's own noise is the same for every arm, so it moves no comparison.

### Tools (branch `mldenoise`)

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
