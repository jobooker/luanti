---
title: "Learned denoiser prototype: in the engine it beats today's filter at equal time, and holds up with a moving camera"
summary: "claude_denoise_learned (default 0) runs today's a-trous filter with per-pixel weights from a 20k-parameter network as GL compute passes; it matches its PyTorch twin to float precision, costs +0.3-0.6 ms of GPU pass time and +0.2-2.3 ms of whole frame, and still beats the filter on RMSE and FLIP at equal time on all four held-out scenes."
tags: [luanti, report]
author: claude
---

# Learned denoiser prototype

## Part 4 (2026-10-09): moving camera, and the lean path

**Verdict: the moving-camera test passes, with one small exception.**
`claude_denoise_learned=1` beats today's filter on video JOD in all three
play scenarios. In room-backup it all but removes the dark patches while
moving: 14.4% of the frame below 80% of truth, down to 1.7%.

**The exception:** the forest walk is slightly worse on dark patches, 5.9%
against 6.3% while moving and 1.0% against 1.3% after the stop.

**The lean path (`claude_denoise_learned=2`)** cuts the learned stage by
0.22-0.24 ms with no loss of quality. That brings it to 0.08-0.28 ms more
than today's filter.

### The playtest

How it was run:
- **Tool:** `util/claude_playtest.py` from this worktree, play settings.
- **Fixes picked up from one-tracer:** d6a9b8bb6, 8cdf54a2b (the view fix),
  82f2d3bc2, a4c1c809f. My branch already had `--rt-dial` and 3344225ba.
- **The dial:** `--rt-dial claude_denoise_learned=0|1`, on the real-time run
  only. Every truth run had it at 0.
- **Timing:** both arms back to back in one GPU lock hold, then both scored.
- **Run folders:** `screenshots/playtest/20261009-065441` (filter) and
  `20261009-070708` (learned).
- **Real pictures checked:** I looked at the middle frame of every real-time
  and reference dump (`util/claude_mldenoise_peek.py`, `peek.jpg` in each run
  folder). All are photo frames, not a debug view.

In the dark-patch columns, "dark" means the share of the frame below 80% of
truth (below 50% while moving in brackets); "after stop" is the first 10
frames after the camera stops.

| scenario | arm | fps (p99 frame) | JOD video | dark while moving | dark after stop | frames to JOD 9 after stop |
|---|---|---|---|---|---|---|
| room-backup | filter | 65.5 (28.1 ms) | 7.19 | 0.144 (0.000) | 0.039 | not reached |
| room-backup | **learned** | 61.2 (30.5 ms) | **7.81** | **0.017** (0.000) | **0.006** | not reached |
| turn | filter | 72.9 (29.2 ms) | 7.73 | 0.049 (0.022) | 0.009 | 24 |
| turn | **learned** | 67.4 (32.1 ms) | **7.84** | **0.045** (0.015) | **0.005** | **18** |
| forest-walk | filter | 37.5 (49.6 ms) | 5.90 | 0.059 (0.009) | 0.010 | not reached |
| forest-walk | **learned** | 36.9 (51.2 ms) | **6.04** | 0.063 (0.009) | 0.013 | not reached |

**The reveal curve** (brightness of newly revealed faces against truth, by
frames since reveal, ages 0-11):
- **room-backup:**
  - filter: 0.97 1.02 0.94 0.87 0.95 0.97 0.97 0.94 0.92 0.93 0.92 0.94
  - learned: 0.97 0.94 0.93 1.02 1.09 1.13 0.87 1.11 1.10 1.02 1.01 1.03
- **turn:**
  - filter: 0.78 0.80 0.80 0.76 1.02 1.00 ...
  - learned: 0.84 0.82 0.83 0.99 1.02 1.00 ...

  In the first three frames the learned filter is closer to truth. From
  frame 3 it matches the filter.
- **forest-walk:** too few revealed pixels after age 3 for a curve.
  - filter: 1.00 1.01 0.96 0.96
  - learned: 0.98 0.98 0.94

**Read with care:**
- **Room-backup reveals run bright.** From frames 4-8 after reveal the
  learned filter is 9-13% brighter than truth, where today's filter runs
  3-13% dark. It is closer to truth in size but on the other side.
- **fps comes from separate runs,** so it carries run-to-run noise. It drops
  4-7% with the learned arm in room-backup and turn, which is in line with
  the +0.3-0.5 ms stage cost plus the tracer-time effect from part 3.
- **The training data is all still cameras.** The network has never seen a
  moving camera, yet it holds up. History trust (the 0w row) is not used at
  all yet.

### Step 2: the lean path (`claude_denoise_learned=2`)

What changed from mode 1:
- **Dropped the 4 affinity features.** They had trained to ~0, so their
  25-tap reads per pass were doing nothing.
- **Maps in fp16.** They are now 11 maps in three RGBA16F images at 1/4
  resolution.
- **The full-resolution upsample pass is gone.** Each pass now reads the
  bilinear maps at its own pixel.

Mode 1 stays in place for A/B. The shader and C++ each have one `LEAN` path.

**Quality in PyTorch** (`util/claude_mldenoise_leaneval.py`): mode 2 against
mode 1 differs by display RMSE 0.00002-0.00003. The error against truth is
identical to 5 digits on all four scenes.

**Correctness in the engine** (`refcheck --arm learned2`, against a twin
that rounds the maps to fp16): display RMSE 0.00005-0.0001, relative error
p99 6e-4, no pixel above 1%. Mode 1 matched to 0.000000; the extra residue
comes from where the fp16 rounding falls.

**Cost** (pinned, 6 rotating rounds, medians, same session for all three
arms; stage = the six passes or the learned step):

| scene | filter stage | mode 1 stage | **mode 2 stage** | whole frame vs filter, mode 1 / mode 2 (paired) |
|---|---|---|---|---|
| forest | 0.697 | 1.088 | **0.867** | +1.93 / +1.73 |
| plains | 1.015 | 1.538 | **1.298** | +1.20 / +1.08 |
| cabin | 0.712 | 1.029 | **0.792** | +0.25 / +0.05 |
| torchroom | 0.699 | 1.009 | **0.813** | +0.89 / +0.72 |

- **Mode 2 against mode 1:** 0.22-0.24 ms less.
- **Mode 2 against today's filter:** 0.08-0.28 ms more.
- **Whole frame:** on the cabin, where the tracer's time holds still, mode 2
  costs the same as today's filter (+0.05 ms). The other scenes still carry
  the unexplained tracer-time shift from part 3. Its cause is the next thing
  to find.

**Not done:** equal-time and moving-camera scoring for mode 2. Its pictures
equal mode 1's to 3e-5 display RMSE, so mode 1's results stand for it, with
its smaller cost.

## Part 3 (2026-10-09): in the engine

**Verdict: yes, in the engine too.** Behind the dial `claude_denoise_learned`
(default 0), today's filter with learned per-pixel weights beats today's
filter at equal time on all four held-out scoreboard scenes:
- **RMSE:** better on all four.
- **FLIP:** better on three, equal on the plains.

**The cost is measured in the engine, not estimated:**
- **The denoise stage on the GPU:** +0.31-0.55 ms per frame.
- **The whole frame:** +0.24 to +2.3 ms. The whole frame grows more than the
  denoise stage because the tracer's own pass time moves with the arm
  (below). The win holds with the whole-frame number.

**The win is not equally wide everywhere:**
- **Forest and plains:** narrow. At its measured cost the learned filter gets
  91-92% of the filter's frames and needs about 78-81% to match the filter's
  FLIP.
- **Cabin and torch room:** wide. It needs 50% and under 6%.

**The engine and its PyTorch twin are the same function:** display-space
RMSE 0.000000 and largest relative pixel difference 3.4e-4, on all four
scenes.

### What was built (branch `mldenoise`)

**The dial and the code:**
- **`claude_denoise_learned` (0/1, live).** At 0 the six `claude_denoise`
  passes run exactly as before; each is wrapped in `ClaudeUnlessLearned`, and
  the new step returns at once.
  - **Checked:** at dial 0, every frame of the in-engine test's filter arm
    equals the exact PyTorch port of today's filter to a relative 4.5e-6
    (display RMSE 0.000000). That is the same check that passed before the
    change.
  - **Not done:** a bit-for-bit comparison against the old binary. Two
    sessions of the same pose and seed do not accumulate identical samples:
    the two arms of one session differed by 0.5% at 1 frame.
- **At 1, compute passes replace all six.** The code is
  `client/render/claude_learned.{h,cpp}` and
  `client/shaders/claude_denoise/learned.comp.glsl`. There are 12 dispatches:
  1. pass 0, exactly as today;
  2. the 20 features, pooled 4x4;
  3. three 3x3 convolutions at 1/4 and 1/8 resolution, then the head;
  4. the maps upsampled bilinearly;
  5. five weighted a-trous passes;
  6. the blend over stages, written into DEN_B, which the present and
     exposure passes read as before.

  The passes use image units 8-20 and SSBO binding 9. Neither is used
  elsewhere.
- **Weights:** `util/claude_mldenoise_weights.bin` (81 KB, 20,349 floats),
  exported by `claude_mldenoise_atrous.py export`. They are read at startup,
  and the log line carries the fnv64 hash. The file in git is `wnet2`, hash
  `34224733fdcbc15b`.
- **Irrlicht:** `ITexture::getNativeHandle()` (the GL name), so that engine
  passes can bind pipeline textures as images.
- **Row order:** the network works in the dumps' top-down row order. Its
  convolutions and the 1/8 branch are not symmetric under a vertical flip.

**One bug, found by an instrument, not a theory.** The first in-engine check
matched PyTorch closely on the plains and the cabin, but on the forest 5-10%
of pixels differed by more than 1% (display RMSE 0.0013). Two hypotheses
died on the measurement:
- the moments computed in fp32 instead of fp64;
- the wrong exposure: an exposure sweep had its minimum at the right value.

Then I dumped the network's intermediate images on a dump frame
(`claudeLearnedDump`) and re-did the GLSL formulas in numpy
(`util/claude_mldenoise_replica.py`):
- **Matched to 3e-6:** the features, conv a0 and conv a1.
- **Did not match:** the 1/8-branch sum, and only on even 1/4 columns from
  2 on.
- **The cause:** the GPU divides floats as a reciprocal times, so
  120.0/240.0 came out a hair under 0.5. Every even column then took the
  parent texel to its left.

With integer arithmetic, all four scenes match.

### Cost in the engine

How it was priced (`util/claude_mldenoise_price.py`):
- **Pinned:** the camera pinned at each scoreboard pose, in a fresh game per
  scene.
- **Scene identity:** printed on every arm, and unchanged throughout.
- **Arms:** filter (anything contender), learned, and honest (denoise off).
  Every dial is spelled on every arm.
- **Rounds:** 6, with the arm order rotating. The numbers are medians.

| scene | honest busy | filter busy | learned busy | six passes (filter arm) | learned step (learned arm) | learned - filter, whole frame (paired per round, median) |
|---|---|---|---|---|---|---|
| forest | 24.71 | 22.67 | 24.99 | 0.676 | 1.037 | +2.19 |
| plains | 17.60 | 16.73 | 18.27 | 1.014 | 1.564 | +1.41 |
| cabin | 12.52 | 12.86 | 13.13 | 0.708 | 1.021 | +0.24 |
| torchroom | 16.47 | 15.70 | 16.91 | 0.698 | 1.028 | +1.10 |

**Differs from the PyTorch picture:**
- **The denoise stage is cheaper than PyTorch said.** In the engine it costs
  1.02-1.56 ms against today's 0.68-1.01 ms, so it adds 0.31-0.55 ms.
  PyTorch had put the network alone at 1.47 ms, plus an estimated ~1 ms for
  the passes.
- **The whole frame grows more than the denoise stage**, except on the
  cabin.
  - The tracer's own pass time (pass_ms[2]) is lowest when today's filter is
    on (forest 17.1 ms median). It is higher with the learned filter (19.0)
    and higher still with the denoiser off (honest, 19.4), in every rotation
    order.
  - The tracer does not read the denoise dial. I suspect GPU timing or clock
    state carrying over between frames, but I have not shown it. The
    honest arm being slower than the filter arm, with less work, is the clue
    worth an instruments-lane look.
  - On the cabin, the tracer time held still and the whole-frame delta
    (+0.24 ms) equals the pass delta.
  - The equal-time table below uses the whole-frame numbers, the pessimistic
    ones.
- **An earlier pricing run was noisier.** The forest's first two rounds there
  had the trace at 30-34 ms, while my CPU scoring ran alongside. Those numbers
  are kept in `~/data/mldenoise/price_run2` but not used.

### Head-to-head on the held-out scenes (in the engine)

Same session per scene; truths re-rendered in that session wherever the
scene loaded differently. The grid hashes:

| scene | grid hash | truth |
|---|---|---|
| forest | 07c89acb… | re-rendered, 8192 frames |
| plains | 177276cd… | reused, same hash |
| cabin | a4622c80… | re-rendered, 8192 frames |
| torchroom | d1bee6d9… | reused, same hash |

Seeds 101-103, depths 1-64 plus the one-second counts. The arms:
- **filter, learned:** what the engine displayed;
- **noisy:** the filter arm's own input;
- **U-Net + grad:** PyTorch on the filter arm's inputs, for reference.

Each cell is RMSE / FLIP.

| scene | frames | noisy | today's filter | **learned (engine)** | U-Net + grad (PyTorch) |
|---|---|---|---|---|---|
| forest | 1 | 0.0436 / 0.162 | 0.0355 / 0.100 | **0.0294 / 0.096** | 0.0196 / 0.090 |
| forest | 27 | 0.0342 / 0.127 | 0.0265 / 0.081 | **0.0239 / 0.078** | 0.0156 / 0.073 |
| plains | 1 | 0.0244 / 0.098 | 0.0134 / 0.045 | **0.0107 / 0.044** | 0.0074 / 0.044 |
| plains | 47 | 0.0116 / 0.071 | 0.0080 / 0.029 | **0.0052 / 0.027** | 0.0042 / 0.030 |
| cabin | 1 | 0.0454 / 0.154 | 0.0114 / 0.040 | **0.0082 / 0.037** | 0.0066 / 0.046 |
| cabin | 70 | 0.0172 / 0.087 | 0.0060 / 0.029 | **0.0041 / 0.027** | 0.0036 / 0.029 |
| torchroom | 1 | 0.1466 / 0.415 | 0.0282 / 0.114 | **0.0258 / 0.107** | 0.0237 / 0.119 |
| torchroom | 52 | 0.1530 / 0.424 | 0.0279 / 0.118 | **0.0214 / 0.096** | 0.0201 / 0.104 |

**Equal time, the one-second budget, using the engine's measured whole-frame
ms.** Each arm gets floor(1000 / busy) frames, scored at the deepest captured
depth at or below that.

| scene | today's filter | **learned (engine)** | U-Net + grad (honest busy + 8.4 ms PyTorch) |
|---|---|---|---|
| forest | 44 (40): 0.0234 / 0.075 | 40 (40): **0.0214 / 0.072** | 30 (27): 0.0156 / 0.073 |
| plains | 59 (56): 0.0067 / 0.028 | 54 (48): **0.0052 / 0.027** | 38 (32): 0.0046 / 0.032 |
| cabin | 77 (70): 0.0060 / 0.029 | 76 (70): **0.0041 / 0.027** | 47 (40): 0.0041 / 0.031 |
| torchroom | 63 (56): 0.0279 / 0.118 | 59 (56): **0.0212 / 0.096** | 40 (40): 0.0207 / 0.107 |

The capture grid is coarse near the top: depths 40, 48, 56 and 64, and the
forest's two arms land on the same depth. The finer test that settles it
(`util/claude_mldenoise_need.py`) asks what share of the filter's frames the
learned filter needs to match the filter's metric, interpolating in log
frames between captured depths:

| scene | share it gets at its measured cost | share it needs to match FLIP (16 / 32 / top) | share it needs to match RMSE (top) |
|---|---|---|---|
| forest | 91% | 78% / 80% / 81% | 73% |
| plains | 92% | 81% / 77% / 78% | 42% |
| cabin | 99% | 45% / 46% / 50% | 28% |
| torchroom | 94% | 6% / 3% / 2% | 2% |

**Reading it:**
- **Forest and plains:** the win holds by about 10 points of frame share.
  More whole-frame cost would erase it: about 2 ms more on the forest, 1.5 ms
  on the plains.
- **Cabin and torch room:** the learned filter beats the filter's whole
  second with half its frames or fewer.

Side-by-sides (noisy | today's filter | learned (engine) | U-Net + grad |
truth; 1 frame on top, the one-second count below, seed 101):
`~/code/luanti-wt/mldenoise/screenshots/mldenoise/test/engine/side-by-side-{forest,plains,cabin,torchroom}.png`.
The tables are in `h2h_engine.md` in the same folder.

### Longer training, and the gradient term's own effect (PyTorch, part 2's test data)

**Training speed:** compiling the training step (torch.compile, after a
one-time compile of about 6 min) made it 3.7x faster: 10 steps/s against
2.7. 20 minutes gave 12.2k steps, against 4.1k before.

| scene | frames | filter | wnet (4.1k, grad) | **wnet2 (12.2k, grad)** | wnet2l1 (12.2k, no grad term) |
|---|---|---|---|---|---|
| forest | 1 | 0.0408 / 0.129 | 0.0324 / 0.122 | 0.0322 / 0.121 | 0.0324 / 0.120 |
| forest | 27 | 0.0306 / 0.107 | 0.0243 / 0.098 | 0.0240 / 0.098 | 0.0240 / 0.098 |
| plains | 47 | 0.0080 / 0.029 | 0.0054 / 0.027 | 0.0052 / 0.027 | 0.0054 / 0.027 |
| cabin | 70 | 0.0059 / 0.029 | 0.0043 / 0.027 | 0.0041 / 0.027 | 0.0042 / 0.027 |
| torchroom | 52 | 0.0278 / 0.117 | 0.0216 / 0.097 | 0.0213 / 0.096 | 0.0214 / 0.096 |

- **Three times the training** bought 1-4% RMSE and no FLIP. This design is
  capacity-limited, not training-limited.
- **The gradient term** is worth 0-4% RMSE and nothing in FLIP for the weight
  network. Its FLIP advantage comes from re-weighing today's filter, which
  leaves the filter's soft blotches rather than the U-Net's grain. That is
  unlike the U-Net, where the term moved FLIP (part 2).
- `wnet2` (with the term) was chosen before testing as the engine's weights.
- **Held-out training views** (CPU): `wnet2` is within 1% of `wnet` (forest-far
  at 1 frame 0.0204 against 0.0207).
- **Unused:** the 4 learned affinity features came out ~0 everywhere. That
  part of the design is unused and could be dropped to save reads.

### What is next

1. **Find why the tracer's pass time moves with the denoise arm.** It is most
   of the whole-frame delta on the forest (+2.2 ms against +0.4 ms of
   denoise passes). It also makes "denoiser off" slower than "filter on".
   Until it is understood, the whole-frame cost is the honest number.
2. **Cut reads:** drop the unused affinity features (25 reads per pixel per
   pass), use fp16 maps, and fuse the map upsample into the passes.
   My guess is that together these bring the stage close to today's cost;
   that needs pricing.
3. **Moving camera:** the history-trust row of 0w, still untested.
4. **Fix the stair-stepped edges:** a per-pixel albedo-blend map, as part 2
   proposed. The U-Net still has 30-40% less RMSE than the learned filter on
   the forest. My guess, not measured, is that part of that gap is these
   edges.

### Tools added in part 3

- `util/claude_mldenoise_capture.py test_engine`: both arms per seed, same
  session, the truth re-rendered when the scene loads differently.
- `util/claude_mldenoise_price.py`: the live-pricer pattern, rotating, N
  rounds.
- `util/claude_mldenoise_engine_score.py`: pictures and RMSE on the CPU, then
  FLIP and the tables.
- `util/claude_mldenoise_need.py`: frames needed against frames got.
- `util/claude_mldenoise_atrous.py`: `export` and `refcheck`.
- `util/claude_mldenoise_check0.py`: the dial-0 check.
- Instruments from the forest bug:
  - `util/claude_mldenoise_replica.py`, `_stages.py`, `_exprobe.py`,
    `_sens.py`, `_where2.py`;
  - `claudeLearnedDump`, which writes `<dump>.lF/.lM.f32` next to every set
    dump with the dial on.


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
