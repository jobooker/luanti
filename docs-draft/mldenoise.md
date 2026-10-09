---
title: "Learned denoiser prototype: status (paused 2026-10-08)"
summary: "Prototype of a learned denoiser against today's a-trous filter; data capture half done, no training run yet. Paused to free the GPU."
tags: [luanti, report]
author: claude
---

# Learned denoiser prototype: status (paused 2026-10-08)

**Where it stands:** 16 of 36 training views captured, the test captures and
training not started. No result yet on whether a network beats today's filter.
Paused at about 20:55 EDT so John can play.

## What exists (branch `mldenoise`)

- **One accumulation, many depths (`claude_dump_at`, C++).** `claude_dump_at =
  1,4,16:prefix` is armed by the next `claude_shutter`. On the frame where
  `still_frames` reaches each N, the client writes the denoiser's whole input
  set to `prefix_N.<name>.f32`: `accum` (radiance), `direct` (its direct part),
  `gbuf` (accumulated albedo and face code), `mom` (moments) and `den` (what
  the screen shows, i.e. today's filter output when `claude_denoise=1`). It also
  writes one `prefix_N.json` with `still_frames` and `resets_since_arm`. The
  stats file reports `dump_pending`, `dump_active` and `dump_written`.
  - Measured: rows in these dumps (and in `truth.f32`) are **top-down**, the
    same order as the PNG. Flipping them made the RMSE against `truth.png`
    jump from 0.009-0.02 to 0.15-0.42.
  - Measured: with `claude_denoise=0`, `den` equals `accum` exactly (max diff 0).
  - Measured: Python's copy of the display transform (exposure, ACES,
    gamma 1/2.2) on a 128-frame dump matches the game's PNG within 0.0045 RMSE,
    after box-downsampling the PNG to 960x540.
- **`util/claude_mldenoise_capture.py`** (GPU lock held):
  - `train`: per view, run A takes 256 frames with auto exposure (the
    scoreboard's probe) and dumps at 1/4/16/64. Run B uses the pinned exposure
    and a different seed, and dumps at 1/4/16/64/2048.
  - `test`: the four scoreboard poses, a fresh game per scene, the
    **anything** contender at the truth's exposure, seeds 101-103. It dumps at
    1-64 frames plus each scene's one-second count, so the network and today's
    filter see literally the same samples.
  - `truth`: re-renders a linear truth.
  - `batch FILE`: several jobs under one hold of the GPU lock.
  - A dump with `resets_since_arm != 0` is discarded and the view retried. A
    view that keeps failing is skipped.
- **`util/claude_mldenoise.py`** (PyTorch, `~/.venvs/rtm-torch`):
  - `train`: a compact U-Net (24/32/48/64 channels), checkpointed and resumable.
  - `eval`: the held-out views.
  - `test`: the scoreboard scenes, RMSE in display space at 960x540, all three
    arms put through the same transform.
  - `time`: milliseconds per 960x540 frame, fp32/fp16, with and without
    channels_last.
  - Dry-run end to end on CPU with fake data only.
- **`util/claude_mldenoise_score.py`** (judge venv): FLIP for every picture,
  plus the noisy | filter | network | truth side-by-sides.

## Design choices (and why)

- **Input:** radiance divided by the accumulated albedo, times the pinned
  exposure, through log(1+x). The direct part goes in the same way. Beside them:
  albedo, the normal decoded from the face code, sky / glow / mixed flags, log
  distance from the accum alpha, two "same face as the right / lower
  neighbour" flags (the exact same-surface test today's filter uses), the
  standard deviation of the mean from the moments, and log N. That is 20
  channels.
- **Output:** a residual on log irradiance. The albedo goes back in, and the
  result is linear radiance, which is what the present pass, auto exposure and
  the referees consume.
- **Loss:** L1 in display space (claude_present's transform). That is the space
  the scoreboard scores and the eye sees, and it stops the sun's x40 from
  dominating.

## Data so far (`~/data/mldenoise/views`, 5.8 GB, not in git)

| | |
|---|---|
| **Done** (16) | forest-n, forest-s, forest-e, forest-ne-up, forest-far*, forest-sw, plains-w, plains-s, plains-e*, overlook, overlook-west, overlook-dusk, skypad, skypad-low, cozy-golden, cozy-midday |
| **Skipped** | forest-se ("time did not freeze" three times); treeline-back (y 12 floats: camera drift) |
| **Not yet captured** (18) | cozy-morning*, cornell, cave-skylight, cave-skylight-pm*, glass-lit, glass-dark, glassfurnace, sealed-outside, sealed-outside-pm, forest-n-dusk, forest-e-dawn, forest-far-w, plains-w-dusk, plains-sw, overlook-down, glassfurnace-pm, cozy-golden-2, cave-skylight-am |

\* held out for validation (`holdout.json`).

Each view took 65-110 s of GPU time when nothing went wrong, and up to 480 s
with retries.

## Found along the way

- **The torch room's linear truth is stale.** In the main checkout,
  `screenshots/scoreboard/truth/torchroom/truth.f32` dates from 11:22, while its
  `truth.png` and `meta.json` were re-shot at 13:16. Put through the display
  transform, the f32 is 0.044 RMSE from the PNG and darker (mean 0.3485 against
  0.3667). The other three scenes agree within 0.009-0.02. The test plan
  re-renders it into this worktree (`truth torchroom`, 16384 frames) rather
  than using it.
- **The shoot tool can pass a shot whose accumulation restarted.** Twice,
  forest-e-B and forest-se-B passed every check in `claude_shoot` (still
  frames, drift), yet the 2048-frame dump showed a restart since the shutter
  was armed. `resets_since_arm` catches it.
- **Pose height matters.** Poses at y 9 (the CI vantages cornell, the skypad)
  or y 12 jitter by 0.05-0.2 units every frame under `--play` and never settle.
  y 8.5 holds.
- **Today's filter costs about 0.6-1.0 ms per frame.** That is the six
  `claude_denoise` passes, pass_ms[3..8] in the 17:16Z scoreboard run. About
  0.3 ms of it is the pass-through copies that run even with the filter off;
  switching the filter on adds 0.3-0.7 ms. This is the bar a network has to
  meet on cost.

## What remains, in order

1. Run the test capture and the torch room's truth
   (`~/data/mldenoise/batch_test.txt`, about 20 min of GPU).
2. Capture the remaining 18 training views
   (`~/data/mldenoise/batch_train2.txt` lines 1-2, about 35 min).
3. Train for about 25 min, then run `eval`, `test` and `time` (the rest of
   `batch_train2.txt`). Then `claude_mldenoise_score.py` on the CPU.
4. Write the result here: per-scene RMSE and FLIP at 1 frame and at one second,
   against today's filter on the same samples; ms per frame against the
   filter's 0.6-1.0 ms; the side-by-sides under
   `screenshots/mldenoise/test/<tag>/`.

Resume: `python3 util/claude_mldenoise_capture.py batch ~/data/mldenoise/batch_test.txt`,
then the same with `batch_train2.txt`. Both skip views already captured.
