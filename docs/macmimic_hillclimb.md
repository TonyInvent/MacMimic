# MacMimic hill-climb: what worked, what didn't

Record of the search for the fastest training recipe on an Apple-silicon Mac (MacBook Pro M4 Pro, 48 GB, 14 cores).
Last updated 2026-11-18 (night). Raw per-run lines are in `logs/hill/results.txt` and `logs/hill/progress.log` (git-ignored).

## Goal and how it is measured

Goal: train more than 2x faster than the BlinkMimic and XMimic ports on this Mac.

| Term | Definition |
|---|---|
| Strict target | normalized evaluation return >= 0.82 with 0 of 40 clips failed, at 3 consecutive checkpoints (100 iterations apart); the time is that of the first of the three |
| First clean | first checkpoint with return >= 0.82 and no failed clip |
| Equal quality | time to match BlinkMimic at its strict point: return 0.847, body position error 3.9 cm |

Baselines on this Mac (one run each):

| Run | Speed | First clean | Strict target |
|---|---|---|---|
| BlinkMimic port | 0.61 it/s | 38.2 min (it 1400) | 44.0 min (it 1600) |
| XMimic port | 0.65 it/s | 54.2 min (it 2100) | not reached in 57 min |

The BlinkMimic port matches upstream's published tracking metrics at iteration 3500 (body position 3.27 vs 3.44 cm,
orientation 6.79 vs 7.02 deg, joints 4.12 vs 4.19 deg, foot slip 6.67 vs 6.66 cm/s, contact 0.763 vs 0.765).

## Where it stands

Names: the preset `macmimic` is the 2026-11-18 recipe (called `macmimic2` while it was found); the first version is
`macmimic_v1`. Apart from the first two rows of the table below, this record was written before the rename:
`--preset macmimic` in it means the first version, `macmimic_v1`.

| Recipe | Strict target | vs BlinkMimic | vs XMimic |
|---|---|---|---|
| **`--preset macmimic`**, seeds 0 / 1 / 2 (runs CB1280fr_s0-2, 2026-11-18) | **21.3 / 21.0 / 22.5 min (it 1400 / 1400 / 1500), mean 21.62** | **2.04x** | **>= 2.6x** |
| `--preset macmimic_v1` (first version), seeds 0 / 1 / 2 | 29.2 / 30.1 / 27.9 min, mean 29.05 | 1.52x | >= 1.95x |
| `--preset macmimic --utd 0.2`, seeds 0 / 1 / 2 | it 1500 / 1500 / 1600 instead of 1700 / 1700 / 1600; ~27.0 / ~27.9 / ~29.3 min at clean speed, mean ~28.1 (estimate: the runs themselves were timed on a busy machine, 30.5 / 32.8 / 34.5 min) | ~1.57x, inside seed noise of the plain preset | |

`macmimic` = `macmimic_v1` + `--wm_lr_mult 8 --wm_batches 3 --ln_bf16 --lr_decay_by 800 --pi_batch 1280 --pi_batch_warm 1024 --pi_batch_switch 400 --focus_replay 0.1`. How it came together (details in open hypothesis 5): (1) the world model was under-trained from scratch; 8x its step size clears failed clips ~300 iterations earlier for free; (2) with that step size 3 world-model updates per iteration do as well as 5, -19% per iteration; (3) the learner then got light enough that the CPU (simulator plus probe sweeps) sets the pace during the sweeps, leaving the GPU slack, so from it 400 a larger policy batch (1280) costs little time and buys return margin; (4) that margin pays for 10% focus replay, which clears the hard clip `dance1_subject1` earlier. Timed with the machine mostly idle (1.09-1.11 it/s); quality at the strict point is lower than BlinkMimic's at its own (0.822 / 4.4 cm vs 0.847 / 3.9 cm, see RESULTS.md). The margin over 2x is small (seed 2 alone is 1.95x).

Earlier candidate (2026-11-18), three full runs from scratch: `--preset macmimic --wm_lr_mult 8 --lr_decay_by 800 --ln_bf16 --wm_batches 3` (open hypothesis 5, runs C3_s0-2). Strict target at it 1500 / 1600 / 1500 (seeds 0 / 1 / 2; the preset: 1700 / 1700 / 1600). One run on a quiet machine (C3_wmb3_s0, 1.16 it/s, bit-identical to C3_s0) puts it 1500 at 21.6 min and 1600 at 23.1 min: 21.6 / 23.1 / 21.6 min, mean 22.1 min, 1.99x BlinkMimic. The other two were timed with other workloads on the machine (26.2 / 27.3 min as measured). A variant with probe sweeps every 90 iterations (C3p90) reaches it 1400 / >= 1700 / 1500: the same mean iteration, a few percent faster per iteration, timed only under load (21.8 / >27.7 / 24.3 min). That is the 2x line, not past it: one late relapse (`dance1_subject1` at it 1500 on seed 1) is the difference, and the timing needs three runs on a quiet machine.

Earlier candidate, not validated by full runs on three seeds (2026-11-17): `--preset macmimic --utd 0.2` with the
step-size decay starting at iteration 600-800 instead of 1100.

| Decay from | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 1100 (`--utd 0.2` alone, full runs) | strict it 1500 | 1500 | 1600 |
| 800 (seed 0 fork from it 770; seeds 1 / 2 full runs) | 1300 | not by 1600 (first clean 1400, one relapse at 1500) | 1400 |
| 600 (seed 0 full run; seeds 1 / 2 forks from it 590) | not by 1500 (first clean 1400) | 1200 | not by 1500 (first clean 1400) |

Time at the speed of the validated runs (0.95 it/s once sweeps run, -5% for `--utd 0.2`; today's runs themselves
were ~25% slower with other workloads on the machine):

| Strict target at iteration | Time with `--utd 0.2` | vs BlinkMimic |
|---|---|---|
| 1200 | ~21.4 min | 2.06x |
| 1300 | ~23.3 min | 1.89x |
| 1400 | ~25.1 min | 1.75x |

Counting full runs from scratch only (seed 0 with decay from 600, seeds 1 and 2 with decay from 800), the first clean checkpoint is at iteration 1400 on all three seeds (~25.1 min, ~1.75x), but the strict target holds on one of them (seed 2, it 1400); the others relapse once. The forks spread it 1200-1400. So the candidate is at ~1.75-1.85x and not more reliable than the preset on the strict criterion; 2x needs the strict target at iteration 1200 on every seed. Follow-ups that did not get there (2026-11-17): probabilistic termination, focus replay, a 2x final step size, and all three combined (clean from it 1300-1400 on every seed, never at 1200).

Caveats:
- First clean is 29.1 min on average: 1.3x vs BlinkMimic, 1.9x vs XMimic.
- At equal quality the gain over BlinkMimic is about 1.25x (MacMimic needs ~34-35 min for 0.84 return / 3.9 cm). The
  0.82 threshold flatters runs that just pass it: `--utd 0.2` passes at iteration 1500 with 4.6 cm body error.
- Each baseline is a single run; seed-to-seed spread is about +-100 iterations (+-1.7 min).

## What worked (in the `macmimic` preset or on by default)

| Change | Effect | Evidence |
|---|---|---|
| Actor and learner overlapped on threads, bf16 autocast, automatic per-Mac configuration | BlinkMimic 0.61 -> XMimic-style 0.65-0.76 it/s | benchmark runs |
| Policy batch 1024 instead of 2048 | iteration time -38%; needs ~200 more iterations; net faster to 0.82 | M12, M13 vs M2 |
| Step-size decay from iteration 1100 instead of 1400 | recovers ~100 iterations of the smaller batch's deficit | M13 vs M12 |
| Probes from iteration 400, every 60, focus share 0.3, tile 600, lead 100-200 | failures found and fixed earlier | M1, M2, M10 |
| Probe termination in numpy, sweep on a worker thread | removes a GPU round trip per probe step; the sweep no longer gates the loop | throughput trials |
| Rotation math unrolled into fusable elementwise ops | FK + reward 5-8x faster in isolation, ~4% on the policy update | profile |
| Policy average 0.99 instead of 0.995 | fewer late failed clips in 3 of 4 runs | shadow averages |
| `--presample`: gather next batches during the learner wait | +2.5% it/s, bit-identical | 150-iteration comparison |

## Helps, not in the preset yet

| Change | Effect | Status |
|---|---|---|
| `--utd 0.2` (5 instead of 2.5 simulator steps per iteration) | 3 seeds: strict target at it 1500 / 1500 / 1600 instead of 1700 / 1700 / 1600; costs 5% it/s, so ~3% less time on average (paired differences -2.2, -2.2, +1.4 min). Failed clips are gone 200-400 iterations earlier on every seed; the return curve does not move (seed 2 runs 0.005-0.008 lower late) | not adopted in the preset: the gain is inside seed noise, as upstream XMimic found ("within seed noise or worse"). Useful when clean checkpoints matter more than return |
| Step-size decay from iteration 600-800 instead of 1100, with `--utd 0.2` (`--lr_decay_by 800` or `600`) | return +0.005 to +0.010 at equal iterations on all three seeds (about 100 iterations); strict target 1300 / >1600 / 1400 (decay 800) and >1500 / 1200 / >1500 (decay 600) vs 1500 / 1500 / 1600. Probe failures clear no slower. 600 and 800 are one plateau; a faster decay (150 instead of 300 iterations) is no better | best candidate so far (~1.85x on average); not in the preset, not validated by three full runs |
| Step size ending at 2x the base rate (`--pi_lr_final 2.0`) with decay from 600 | one fork (seed 1): return +0.003 to +0.005 from it 1300 (0.825 / 0.827 / 0.831 vs 0.822 / 0.825 / 0.826 at it 1300 / 1400 / 1500), same strict point (1200); 1.5x: +0.003 to +0.004, strict 1300 | weak positive inside the fork noise |
| `--focus_replay 0.25 --focus_replay_per 0.015` (policy starts from probe-failure windows, share shrinking with the failure count) | failed clips gone from it 1400 instead of 1700, but return ~0.007 lower (crosses 0.82 at 1600) | one seed; small net gain at best |
| `--deficit_replay 0.3` (30% of policy starts drawn by exp(-6 G_bin / G_max), the simulator's running return of the segment) | tail fork from it 1010: 1 failed clip-checkpoint in 1100-1700 instead of 5, return +0.002 to +0.004 late; the 0.82 crossing stays at it 1500 | one fork, inside the fork noise for return; not enough to move the strict iteration |

## What did not work

Compute cuts (each trades iterations for speed about 1:1, or worse):

| Change | Result |
|---|---|
| Narrow encoder/decoder (1024,512,256,256) | 0.84 it/s, 0.82 only at it 1500: a wash (M9) |
| Horizon 24 | best 0.803 by it 1600 (M15) |
| World-model batches 5 -> 3 from it 700 | best 0.818 (M16) |
| World-model batch 1024 | best 0.815 (M17) |
| Cheap ramp (horizon 16, 3 world-model batches until it 800) | 0.82 only at it 1600 (M5) |
| Narrow nets + 3 world-model batches + 2 critic epochs together | 1.61 it/s but 0.785 by it 1700 (A_fast) |
| Batch 1024 then 2048 from it 1100 | 0.82 at it 1400 at lower speed: a wash (M14) |

Algorithm knobs:

| Change | Result |
|---|---|
| Policy batch 512 for the first 400 iterations (`--pi_batch_warm 512 --pi_batch_switch 400`; 1.31 vs 1.00 it/s while it lasts) | probe failures 97 / 95 / 83 / 56 / 30 at sweeps 420-660 vs 91 / 67 / 61 / 28 / 25: about 60 iterations behind, which costs ~63 of the 88 s saved |
| Step size ending below the base rate (`--pi_lr_final 0.5` / `0.25`, decay from it 800, forks of seed 0 from it 770) | return at it 1200 / 1300 / 1400 / 1500: 0.815 / 0.819 / 0.821 / 0.824 (0.5) and 0.798 / 0.817 / 0.819 / 0.821 (0.25) vs 0.813 / 0.823 / 0.824 / 0.828 at the base rate: the late creep is learning that needs the step size, not optimizer noise |
| Probe restart after a failure (`--eprobe_resume 30`: a failed probe world restarts 30 frames later instead of stopping). Motivation: the strict point waits on `dance1_subject1`, which has the same inverted move twice (frames ~5350 and ~5800); the probes hide the second spot behind the first until it ~1080, and a failure map with restarts shows it already failing at it 800 | forks of seeds 0 / 1 / 2 with decay from 800: strict target not by 1500 / 1300 / not by 1500, vs 1300 / not by 1600 / 1400 without it. With the second spot visible from the first sweep, both spots still clear only at it ~1140-1200: the move is not starved of resets, it takes that long to learn |
| Late compute cuts: no probe sweeps and `--utd 0.4` from it 900 (`--eprobe_until 900 --utd_late 0.4 --utd_late_from 900`; +15% it/s on a quiet machine, more on a busy one), fork of seed 1 from it 590 with decay from 600 | relapses once the sweeps stop: 0.808/1 at it 1100, 0.783/4 at 1200 (two get-up clips and both dances), 0.819/1 at 1300; first clean 1400 vs 1100 for the control. The late sweeps and their focused resets keep cleared clips cleared |
| Four times the original simulation per iteration (`--utd 0.1`, decay from 800), forks of seeds 1 / 2 from it 590 | no failed clip from it 1200 on either seed (controls: 1400), but the return runs 0.004-0.008 lower from it 1300 (seed 1 0.814 / 0.820 / 0.822 at 1300-1500 vs 0.821 / 0.826 / 0.811; seed 2 0.810 / 0.814 / 0.817 vs 0.813 / 0.820 / 0.823): 0.82 crossed at it 1500 / not by 1500. Same trade as `--utd 0.2` and focus replay |
| 10% focus replay for the policy and the world model (`--focus_replay 0.1 --focus_replay_wm 0.1`), three seeds | failed clips gone 300-400 iterations earlier on two seeds, return 0.003-0.011 lower: strict not by 1500 / 1300 / not by 1500 vs 1300 / 1600+ / 1400 |
| Probabilistic termination while training (`--term_prob 0.05 --term_prob_until 950`), three seeds; and combined with decay from 600 and a 2x final step size | the hard clip clears 100-200 iterations earlier on two seeds, but late relapses of other clips still set the strict point: 1400+ / 1200 / 1500+ vs 1300 / 1600+ / 1400; combined: clean from it 1300-1400 on every seed, never at 1200 |
| Two half-batch policy updates per iteration (`--pi_steps_warm 2 --pi_batch_warm 512`) | ahead only before it 400 (68 vs 93 probe failures at 420), level by 540, behind at 600; +8% time per iteration |
| Muon momentum 0.975 from it 400 (`--pi_mom`) | behind the paired run by it 600 (50 vs 28 probe failures) |
| Muon momentum 0.975 only from it 1010 (tail fork) | return 0.006-0.013 below the control from it 1400 (0.812 vs 0.820 at 1500) |
| Shorter policy averages (0.98, 0.97) | no earlier 0.82 crossing (0.818 / 0.816 vs 0.820 at it 1500), more early failures |
| Fixed 25% focus replay | probe failures fall faster but return lags and a new persistent failure appears |
| Step-size decay from it 1000 with batch 2048 | higher late return, failures linger (M7) |
| Probe window 3 bins, policy-average schedule, goal stride 5 + critic clamp | no gain (M8, M11, M4) |
| Critic epochs 5 -> 3 together with Muon Newton-Schulz iterations 5 -> 3 (~70 ms less GPU work per iteration), on top of `--utd 0.2` | full run, seed 0: never reaches 0.82 by it 1800 (best 0.818); probe failures at sweeps 600 / 660 are 79 / 49 vs 28 / 25 without the trims. A tail fork from it 1010 had looked neutral: tail forks miss effects that build up from the start. Not merged |

CPU / GPU placement (all within +-2.5% noise or worse):

| Change | Result |
|---|---|
| Simulator threads 10 / 12 / 14, probe simulator threads 4 / 8 / 14 | flat |
| Learner thread at user-interactive QoS, `--learner_async`, `--eprobe_async` 12 / 16 / 20 | flat |
| Policy inference for the actor or the probes on the CPU | 0.79-0.83 vs 0.92 it/s: worse |
| Reward heads on the CPU | worse |
| Critic fit on the CPU, one iteration stale (as upstream XMimic does on a second GPU) | 0.813 vs 0.905-0.942 it/s; 0.600 with `--utd 0.2`; 0.695 / 0.677 with 1 / 2 BLAS threads vs 0.794-0.847. The fit always finishes in time (zero wait); the GPU learner itself slows while the CPU works. Patch not merged |
| Critic fit in NumPy (Accelerate BLAS + vForce, no MLX on the thread), one iteration stale | matches the GPU fit numerically, but costs ~0.6 s of CPU per iteration (1.0-1.3 s next to the simulator). Iterations without a probe sweep 3% faster (1012 / 1096 vs 1045 / 1135 ms), iterations during a sweep 8% slower (1288 / 1335 vs 1174 / 1246 ms): cycle 0.839 / 0.799 vs 0.886 / 0.828 it/s; with `--utd 0.2` 0.773 vs 0.769. Taking 94 ms off the GPU buys ~35 ms. Not merged |
| Goal encoder hoisted out of the rollout (`--fast_learner`) | 0.937 vs 0.905-0.942 it/s: flat, although stacked GEMMs measure faster in isolation |
| Upper bound for NumPy inference: zero actions in place of the actor's policy forward and no GPU call in the probe step | at most +7% it/s (0.860 vs 0.795-0.813 in a noisy window) before paying for the NumPy forward (~0.1 s per actor step for 4096 worlds): not worth the port |
| Twice the worlds (`--num_envs 8192`) | -6% it/s |
| Probe tile 300 (twice the worlds, 24 instead of 39 sweep iterations) | +1%, within noise: probe cost is per step, not per world |

## Main findings

1. The learner is GPU-bound. Per iteration at batch 1024: BPTT 421 ms, world model 349 ms (5 x (58 fwd+bwd + 12 Muon)),
   critic 94 ms, policy Muon 22 ms; about two thirds of the GPU's peak matmul rate. GPU 84-92% busy, CPU 13-35%.
2. CPU work slows the GPU learner: an outside process on ~10 cores cut 0.92 to 0.585 it/s, and every attempt to move
   learner or inference work to the idle CPU (critic fit, actor / probe inference, reward heads) lost more GPU time
   than it freed. A pure spin load on 8 cores costs only ~4%, so the cost is not core occupancy alone. The CPU itself
   measures 1.3-1.45 TFLOPS of fp32 matmul.
   Cause, from a microbenchmark (a BPTT-shaped learner loop on one thread, a 751-row MLP forward at 20 calls/s on
   another): work on MLX's CPU stream slows the GPU learner 2-6x; the same forward in NumPy costs ~3%, and as small
   MLX GPU evals ~3%. All the CPU placements above went through MLX's CPU stream. Do not mix MLX CPU-stream work with
   a GPU learner in one process; CPU-side work has to be NumPy. Even then the CPU is not free: a NumPy critic fit
   (~0.6 core-seconds per iteration) gave back two thirds of the GPU time it saved and lost during probe sweeps.
3. Every schedule tried reaches a given return after about the same GPU time (return 0.82 after ~25.5 min at current
   speeds): batch 1024 at it ~1500, batch 2048 from it 700 at ~1300, batch 2048 from it 1100 at ~1400. Progress after
   the bootstrap phase follows policy samples processed, not optimizer steps (half batches, momentum) or simulator
   data (`--utd 0.2`).
4. Two things bind the strict target and both land at iteration ~1200-1400: the return crossing 0.82, and one clip,
   `dance1_subject1` (BeyondMimic's ablation lists the same clip as its hard case: "Failed (cartwheel)"). The clip has
   the same inverted move twice (frames ~5300-5400 and ~5750-5850, root upside down). The policy tracks the
   orientation, but the pelvis ends 0.26-0.28 m below the reference, past the 0.25 m termination limit; from a
   reference start at frame 5350 or later the move passes. The failure frame creeps by about one frame per sweep
   (5337 at it 420, 5361 at it 1080). The clip first passes the evaluation at it 1100-1400 under every schedule tried
   (decay from 600 / 800 / 1100) and relapses once or twice before it is stable. More focused resets do not speed it
   up (probe restart), and neither does any step-size schedule.
5. Per-clip returns: walks sit at 0.85-0.91 by it 1500; dance, get-up, fight and jump clips at 0.72-0.80 carry almost
   all of the late return growth.
6. The GPU is saturated and not throttled. IOReport sampling during training (2026-11-17, quiet machine, 1.00 it/s):
   the GPU is off 3% of the time between probe sweeps and 8% during one; it spends 62-70% at its top performance
   state (1578 MHz) and 22-26% one state lower (1470 MHz), about 97% of the top frequency on average. GPU power is
   20-22 W, CPU power about 10 W: far from any thermal limit, so High Power Mode has nothing to unlock. More speed
   can only come from less arithmetic per unit of learning.

7. An earlier step-size decay is free and pulls the return forward by ~100 iterations, but all decays between 600
   and 1100 join the same late curve by it 1300-1500 (~0.826 at it 1500): the late creep is not optimizer noise. A
   step size below the base rate slows it (0.817 / 0.819 / 0.823 at it 1300 for 0.25x / 0.5x / 1x), a step size
   above it speeds it slightly (0.825 for 1.5x and 2x).
8. The late probe sweeps and doubled simulation are not overhead: stopping both at it 900 brings failed clips back
   (4 at it 1200).
9. Time estimates must use the validated runs' speed (it 400 at 6.4 min, then 0.95 it/s), not a single quiet-machine
   trial (1.00 it/s): with `--utd 0.2` iteration 1200 is ~21.4 min, 1300 is ~23.3 min.

10. The search ends on a trade-off between its two binding constraints. Everything that clears failed clips earlier
   (doubled or quadrupled simulation, focus replay, probabilistic termination) lowers or does not raise the average
   return, and everything that raises the return (earlier decay, a larger final step size) leaves the failure tail
   where it is. Across 2026-11-17's ~30 forks and runs, no recipe reached the strict target at iteration 1200 on all
   three seeds; the best are clean from it 1300-1400 (~1.75-1.9x), and single late relapses move a seed's strict
   point by +-100 iterations. More than 2x on the strict target would need something outside this search: a
   pretrained world model (not a from-scratch comparison) or substantially faster learner kernels.

## How things are measured

- Training is deterministic: runs sharing a prefix are bit-identical until a change acts, which gives paired comparisons.
- Throughput: 105-iteration resumed trials covering one probe sweep (`--resume logs/hill/MM1_s0`), interleaved with
  baselines, recording other processes' CPU. Trials with more than ~3 cores of outside load are discarded.
- Tail experiments: `--snap_at 1010` writes a fork point with the replay buffer; `--resume <run>/snap` continues it.
  A control fork reproduces the full run's strict iteration; single checkpoints differ by up to 0.006 return and by
  which clip fails, so only larger effects can be read from one fork.
- Tracking quality: `tools/eval_metrics.py` (port of upstream's protocol).
- Mid-run forks: `--snap_at 410 / 590 / 770` (between two sweeps) are the fork points used on 2026-11-17; a change
  that acts after the fork point is compared with the run that wrote it (`logs/hill/run_FK.sh`, ~20 min per fork).
- Why a clip fails: the `--eprobe it` lines of a run give the failing clip and frame per sweep;
  `tools/probe_failmap.py <ckpt>` repeats a sweep with restarts and lists failures hidden behind earlier ones;
  `tools/move_diag.py <ckpt>` starts `dance1_subject1` at a range of frames around its two inverted moves and prints
  where and by which limit (height or tilt) each start fails.
- Experimental flags added on 2026-11-17 (all off by default): `--pi_lr_final`, `--eprobe_resume`, `--eprobe_until`,
  `--utd_late` / `--utd_late_from`, `--term_prob` / `--term_prob_until`, `--init_wm_only`, `--wm_lr_mult`, `--v_lr_mult`, `--ln_bf16`, `--late_anneal_*`, `--term_margin` / `--term_margin_from` (training termination thresholds x m; evaluation unchanged).

## Related work consulted

- XMimic README: gains came from systems work (whole iterations compiled, bf16 backward, one GEMM per layer for weight
  gradients over all horizon x batch rows, batched Muon, the critic fit on another GPU one iteration stale). Tried
  without benefit there: higher policy learning rate, larger world-model batches, lower update-to-data ratio.
  https://github.com/nico-bohlinger/xmimic
- BlinkMimic README: 133 ms per iteration on an RTX 4090, strict target in 3.4 min; InstantTrack batches the goal
  encoding before the rollout (same result, fewer kernels). https://github.com/fuyichen0725/blinkmimic
- Cheaper Muon orthogonalization: Gram Newton-Schulz (https://tridao.me/blog/2026/gram-newton-schulz/), Turbo-Muon and
  "How much orthogonalization does Muon need?" (fewer Newton-Schulz iterations at parity).
- Adaptive motion sampling by failure or tracking error per time bin: BeyondMimic (https://arxiv.org/pdf/2508.08241),
  EGM (https://arxiv.org/pdf/2512.19043).

- Termination design for hard motions: Stubborn (https://arxiv.org/html/2606.12814) ends an episode past the tracking
  limit only with probability 0.005 per step; a loose-to-tight termination curriculum (https://arxiv.org/pdf/1907.11842).
- BeyondMimic's ablation names `dance1_subject1` (cartwheel) as the motion that fails without adaptive sampling: the
  clip that sets the strict point here is the known hard case of this dataset.
- Muon needs a decreasing step size late and works best as the early-phase optimizer
  (https://arxiv.org/pdf/2606.21514): consistent with the earlier decay helping here.

## Open hypotheses (ranked)

| # | Hypothesis | Expected | Risk to learning | Status |
|---|---|---|---|---|
| 1 | Faster learner: custom Metal kernels for the policy update | 10-20% per iteration at best | none | not started. Profile 2026-11-17 (batch 1024): BPTT 417 ms at 5.0 TFLOPS, world-model updates 347 ms at 3.3 TFLOPS, against 7.5 TFLOPS for large GEMMs on this GPU; the BPTT is 32 sequential steps of M=1024 GEMMs (3.4-5.6 TFLOPS each in isolation). At the 7.5 TFLOPS of large GEMMs the BPTT would take ~280 ms and the world-model updates ~150 ms (a learner ~1.6x faster), but the recurrence keeps the BPTT GEMMs at M = 1024, where MLX's kernels already run at the measured rates; the one restructuring that batches across steps (the hoisted goal encoder, `--fast_learner`) was flat in training. Checked on 2026-11-17: fp16 runs at the bf16 rate; padding the decoder input from 1022 to 1024 makes that layer 17% faster (~2% of the learner, not yet applied); MLX is the latest release (0.32.4); the BPTT and the world-model updates on two GPU streams do not overlap (765 ms vs 762 ms for the sum) |
| 2 | Pretrained world model of the G1 (`--init_from <run> --init_wm_only`): policy and critic from scratch, the world model of another seed's finished run | skips the bootstrap phase | none | outside the from-scratch comparison, reported separately. Seed 1 with seed 0's world model (iteration 2000), plain `macmimic` preset: probe failures 20 at sweep 420 (from scratch: 93); no failed clip at any checkpoint from it 900 to 1500 (from scratch: from 1700); return 0.802 / 0.808 / 0.812 / 0.815 / 0.820 / 0.826 / 0.829 at it 900-1500: strict target at it 1300, 21.4 min of training at 1.00 it/s (2.06x BlinkMimic's from-scratch 44.0 min, not counting the 34 min that trained the world model). The binding constraint is now only the return crossing 0.82. Seed 2 (world model of seed 0) and seed 0 (world model of seed 1): strict target at it 1400 / 1400, 23.7 / 23.9 min; failed clips gone from it 1000 / 1200. Three seeds: 21.4 / 23.7 / 23.9 min, mean 23.0 min, 1.91x. With the return-raising changes on top (`--lr_decay_by 800 --pi_lr_final 2.0`, same world models and seeds): strict target at it 1300 on seeds 2 and 0 (21.7 / 21.8 min, vs 1400 / 1400), but seed 1 relapses at it 1300 and 1400 (0.810 / 1 failed clip, 0.831 / 1) and misses it by 1500 (vs 1300); the return at it 1200 is 0.821 / 0.809 (1 failed) / 0.819 on seeds 1 / 2 / 0. No seed gets to it 1200. Mixed, not adopted |
| 3 | Deferred decoder weight gradients (one GEMM per layer after the backward recurrence) | ~1-1.5% per iteration | none | closed 2026-11-17: batching does not help on this GPU. The world-model MLP over 8 recurrent steps of 2048 rows takes 49.0 ms forward + backward, the same MLP on all 16384 rows at once 48.5 ms; the BPTT with the goal encoder hoisted out of the recurrence (one M = 33792 call) takes 419.9 vs 416.7 ms. MLX already overlaps the independent per-step GEMMs |
| 4 | LayerNorm in bf16 in the world model (`mx.fast.layer_norm` on the bf16 matmul output, fp32 accumulation inside; now LN runs on an fp32 copy) | ~5% per iteration | small numeric change (rounding before the activation instead of after) | microbenchmark 2026-11-17: world-model update forward + backward 56.5 -> 49.6 ms (x5 per iteration), BPTT 416.7 -> 403.5 ms: ~48 ms of ~1000 ms per iteration. In train.py as `--ln_bf16` (off by default). Training A/B (forks of D800_utd02_s1@590 to 700, two pairs): 0.969 / 0.815 it/s without, 0.929 / 0.885 with, under 75-100% CPU of other workloads (a VM): inconclusive, to repeat on a quiet machine. Full runs on a quiet machine: V4_s0 (with it) 0.974 it/s from it 400 to 1400 against 0.947 for MM1_s0 (without): ~3%, it 1300 at 21.5 instead of 22.2 min |
| 5 | The world model is under-trained from scratch: its 8-step error (`wm_err8`) is 0.86 / 0.57 / 0.46 / 0.57 / 0.45 / 0.36 at it 200 / 400 / 600 / 1000 / 1200 / 1500 (seed 0), against 0.28 / 0.24 / 0.24 / 0.37 / 0.34 / 0.22 with a pretrained world model (seed 1), which reaches the strict target ~300 iterations earlier. Fewer world-model updates hurt (M16, M17). A larger world-model step size (`--wm_lr_mult`) costs nothing per iteration | up to the pretrained-WM gain (~300 iterations) | instability | `--wm_lr_mult 2.0`, seed 0, from scratch (W2, vs MM1_s0): `wm_err8` 13-25% lower to it 550, then 5-7% higher (the policy reaches harder states: training return 0.66 vs 0.60 at it 900); probe failures 77 / 60 / 30 / 14 / 9 / 2 at sweeps 420 / 540 / 660 / 780 / 900 / 1020 vs 93 / 66 / 33 / 10 / 17 / 6; evaluation return / failed clips at it 900-1500: 0.765/3, 0.793/1, 0.796/2, 0.806/1, 0.811/1, 0.819/0, 0.822/0 vs 0.732/6, 0.757/4, 0.771/5, 0.803/1, 0.794/2, 0.802/2, 0.821/2. Failed clips gone from it 1400 instead of 1700, return +0.01-0.035 in it 900-1400, the same at 1500; first clean it 1500 vs 1700 (run ended at 1500); `dance1_subject1` fails the evaluation until it 1300. Same speed. `--wm_lr_mult 4.0` (W4): probe failures 58 / 38 / 24 / 9 / 6 / 3 / 0 at sweeps 420 / 540 / 660 / 780 / 900 / 1020 / 1260; evaluation at it 900-1500: 0.765/4, 0.790/1, 0.777/2, 0.802/1, 0.814/0, 0.819/1, 0.825/1; `dance1_subject1` passes from it 1100; the late failures are `jumps1_subject2` (it 1400, 1500), which the probes do not catch. No instability (0 non-finite). `--wm_lr_mult 8.0` (W8): probe failures 59 / 38 / 14 / 9 / 8 / 4 / 0 at the same sweeps; evaluation at it 900-1500: 0.762/3, 0.778/2, 0.784/2, 0.808/0, 0.810/1, 0.818/0, 0.824/0: like 4x, fewer late failed clips; 2-8x is one plateau above the base rate. 8x with the decay from it 800 and `--ln_bf16` (R): seed 1 strict target at it 1300, 22.87 min (0.821 / 0.827 / 0.831 / 0.833 at it 1300-1600, no failed clip; plain preset seed 1: it 1700, 30.1 min; decay 800 + `--utd 0.2`: not by 1600), timed with other workloads on the machine (a VM and browser tests). Seed 2: clean at it 1200 / 1300 (0.814 / 0.821), then `fight1_subject3` relapses at it 1400 and 1500 (the probes see it at sweeps 1380 and 1500), clean again at 1600 (0.834): strict not by 1600 (plain preset: 1600). Seed 0: clean at it 1200 / 1300 (0.812 / 0.819), `dance1_subject1` relapses at 1400 (0.822), clean at 1500 / 1600 (0.827 / 0.831): strict >= 1500 (plain preset: 1700). Three seeds: first clean 1300 / 1200 / 1200 (plain preset 1700 / 1600 / 1700), strict 1300 / >1600 / >=1500: one late relapse per seed on two seeds still sets the strict point. R + `--deficit_replay 0.3`, seed 2 (R2_def_s2): clean at 1200 (0.812), `dance1_subject1` relapses at 1300 (0.816), then clean 0.821 / 0.826 / 0.830 at 1400-1600: strict it 1400, 24.3 min (busy machine) vs >1600 without; return ~0.005 lower at 1300-1500. Late step-size anneal (`--late_anneal_from 1200 --late_anneal_mult 0.25 --late_anneal_len 100`), forks of R2_def_s2@1010 (forks are not bit-identical to their source run: the deficit weights and the sampling RNGs restart; control fork pending): policy only (LA1): 0.809/1, 0.815/1, 0.820/0, 0.823/0, 0.826/0 at it 1200-1600 (`dance1_subject1` until 1300); policy and world model (LA2): 0.809/1, 0.818/0, 0.819/0, 0.824/0, 0.825/0: no failed clip from 1300, but the return stalls just under 0.82 at 1300-1400. Control fork (LA0, no anneal; forks of one snapshot are bit-identical to each other, all three read 0.809/1 at it 1200): 0.817/0, 0.822/0, 0.826/1 (`jumps1_subject2`), 0.830/0 at it 1300-1600. Against it the anneal moves the return by -0.005 to +0.002 and trades one relapse for another: neutral, closed. Critic step size x4 on top of R (V4_s0, full run, seed 0): 0.781/2, 0.802/1, 0.824/0, 0.830/0, 0.834/0 at it 1200-1600 against R_s0 0.812/0, 0.819/0, 0.822/1, 0.827/0, 0.831/0: lower at 1200-1300, higher from 1400; strict it 1400, 23.2 min on a quiet machine (R_s0: >= 1500). World model only (LA3): 0.816/1, 0.822/0, 0.824/1, 0.827/1 at it 1300-1600: worse than the control; the late anneal is closed in all three forms. Cheaper world-model updates on top of R, seed 0: `--wm_batches 3` (C3_wmb3_s0) runs at 1.16 vs 0.97 it/s (+19%); its world-model error is back near the base-step-size level (0.63 / 0.62 at it 400 / 600 vs 0.48 / 0.50) and so are the early probe failures (93 / 64 / 24 at sweeps 420 / 540 / 660), but the late evaluation keeps up with R: 0.811/0, 0.800/2, 0.821/1, 0.829/0, 0.832/0 at it 1200-1600 (R_s0 0.812/0, 0.819/0, 0.822/1, 0.827/0, 0.831/0), so it 1500 comes at 21.6 min instead of 26.5 (busy) / ~25 (quiet). The strict point needs it 1700 (the run ended at 1600). `--wm_batch 1024` with 5 batches (C1): 1.09 it/s, 0.803/1, 0.811/1, 0.815/1, 0.820/1, 0.822/2 at it 1200-1600: worse than 3 batches of 2048. C3 on three seeds to it 1800: seed 1 (C3_s1, 1.02 it/s with outside load): 0.806/2, 0.816/0, 0.820/0, 0.821/1 (`dance1_subject1`), 0.826/0, 0.830/0, 0.832/0 at it 1200-1800: strict it 1600, 26.2 min (~23 min at C3_wmb3_s0's 1.16 it/s). Seed 2 (C3_s2, 0.92 it/s, slowed by other processes from it ~600): 0.792/2, 0.814/0, 0.816/1 (`dance1_subject1`), 0.826/0, 0.829/0, 0.830/0, 0.836/0 at it 1200-1800: strict it 1500, 27.3 min (21.6 min at 1.16 it/s). Seed 0 (C3_s0, bit-identical to C3_wmb3_s0 to it 1600, 0.87 it/s under outside load): 0.832/0 at 1700, 0.837/0 at 1800: strict it 1500. Three seeds: strict it 1500 / 1600 / 1500 (seeds 0 / 1 / 2): at the quiet run's 1.16 it/s 21.6 / 23.1 / 21.6 min, mean 22.1 (1.99x); as measured with other processes running 29.3 / 26.2 / 27.3 min. Training against tighter termination limits (`--term_margin 0.9 --term_margin_from 1010`, fork TM9_s2 of R2_def_s2@1010 vs LA0): 0.808/1, 0.814/1, 0.821/0, 0.823/0, 0.828/0 at it 1200-1600 vs 0.809/1, 0.817/0, 0.822/0, 0.826/1, 0.830/0: no relapse from 1400 (strict 1400 vs >= 1600), but `dance1_subject1` fails until 1300 and the return is 0.002-0.003 lower; one fork, inside the noise. `--wm_batches 2` (C2_wmb2_s0): probe failures 75 / 60 / 51 at sweeps 540-660 (3 batches: 64 / 43 / 24); evaluation 0.785/2, 0.789/2, 0.811/0, 0.819/0, 0.820/2, 0.824/0, 0.829/0 at it 1200-1800: strict >= 1700, too far behind for its ~8% speed; 3 batches is the sweet spot. 3 batches at 16x (C3w16_s0): probe failures 79 / 48 / 35 at sweeps 420 / 540 / 600 (8x: 93 / 64 / 43); evaluation 0.810/0, 0.815/1, 0.823/0, 0.825/1, 0.832/0, 0.835/0, 0.837/0 at it 1200-1800 (8x: 0.811/0, 0.800/2, 0.821/1, 0.829/0, 0.832/0, 0.832/0, 0.837/0): strict 1600 vs 1500, inside the noise; 8x stays. C3 + `--term_margin 0.9 --term_margin_from 1000`, seed 1 (C3m9_s1, bit-identical to C3_s1 until it 1000): `dance1_subject1` fails until 1500 (C3_s1: passes at 1300 / 1400), 0.831/0, 0.830/0, 0.832/1 at it 1600-1800: no strict point by 1800; the margin is closed. A longer policy average as a proxy (each checkpoint the mean of the EMA checkpoints k and k-100, evaluated for C3_s0-2): relapses as often, return no higher; closed. Critic fit with 2 instead of 4 minibatches: 50.5 vs 53.2 ms per iteration, not worth a learning change. C3 + `--pi_lr_final 2.0`, seed 1 (C3f2_s1; killed at it ~1500 by a macOS GPU recovery, `gpuEvent-python3.13-2026-11-18-031152.ips`, the second in two days): 0.806/1, 0.815/0, 0.815/2, 0.823/2 at it 1200-1500 vs C3_s1 0.806/2, 0.816/0, 0.820/0, 0.821/1: no better. Seed 0 (C3f2_s0): 0.793/2, 0.817/1, 0.824/0, 0.829/0, 0.830/1, 0.833/0, 0.833/1 at it 1200-1800 vs C3_s0 0.811/0, 0.800/2, 0.821/1, 0.829/0, 0.832/0, 0.832/0, 0.837/0: no strict point by 1800 (C3_s0: 1500); closed. Newton-Schulz iterations 5 -> 3 (`--wm_ns_steps`, `--pi_ns_steps`): 10.9 -> 6.8 ms per world-model update (two 1024x1024 matrices), ~12.6 ms per iteration with 3 world-model batches plus ~8 ms for the policy (~2.5%). Seed 0 (C3ns3_s0): 0.785/2, 0.803/1, 0.805/1, 0.816/0, 0.816/2, 0.807/1, 0.821/1 at it 1200-1800: far behind C3_s0, no strict point; the Muon updates need the full orthogonalization; closed. Re-timing C3_s1 (C3q_s1, bit-identical checkpoints) ran at 0.89 it/s with other processes on the machine (strict 1600 at 30.4 min): the 1.16 it/s of C3_wmb3_s0 is the only idle-machine speed measured; the same configuration ran at 1.02-1.08 it/s early under load. Next: a 768-wide world model on top of C3 (`--wm_hidden 768,768,768`; learner profile: BPTT -51 ms, world-model update -26 ms each, ~-128 ms per iteration with 3 batches, ~13-15%), seed 0 (C3w768_s0); at the base step size it was worse (proxy: 0.749 vs 0.788 at it 1000). Result: 1.00 vs ~0.87 it/s under the same load (~13% faster), but 0.771/4, 0.800/2, 0.819/0, 0.823/0, 0.807/1, 0.830/0, 0.832/0 at it 1200-1800 vs C3_s0 0.811/0, 0.800/2, 0.821/1, 0.829/0, 0.832/0, 0.832/0, 0.837/0: strict ~1700 vs 1500, a wash in time; closed. C3 + `--deficit_replay 0.3 --v_lr_mult 4` (both free per iteration; each moved one seed's strict point on R), three seeds (C3dv): seed 1: 0.806/1, 0.812/1, 0.818/1, 0.824/0, 0.829/0, 0.831/0, 0.835/0 at it 1200-1800: strict 1500 (C3_s1: 1600); seed 2: 0.805/2, 0.815/0, 0.818/1, 0.823/1, 0.823/2, 0.832/0, 0.834/0: strict >= 1700 (C3_s2: 1500). It moves which seed relapses, like the other relapse fixes; closed (seed 0 not run). Across the C3 family the return is >= 0.82 by it 1400 on most runs but ~60% of runs still fail a clip at 1400 (~15-20% per checkpoint at 1500-1800): the failure tail binds, not the return. C3 + `--utd 0.2` (C3u_s1, stopped at it ~900): 0.75-0.88 vs 1.06-1.36 it/s, 30-35% slower, not 5%: with 3 world-model batches the learner is light enough that the actor (MuJoCo on the CPU plus the 4096-world policy forward) is nearly on the critical path, and twice the simulation puts it there; probe failures 20 / 9 / 5 at sweeps 660 / 720 / 780 vs 26 / 17 / 15. Closed. The same balance caps further learner-only speedups at about the ~150 ms per iteration by which the learner still exceeds the actor. Policy batch 768 on C3: profiled (synchronized) 1.45 vs 1.20 it/s, learner 663 vs 806 ms per iteration; unsynchronized pre-sweep A/B (it 60-160) 1.49 / 1.43 / 1.39 / 1.33 it/s vs 1.33 / 1.19 for batch 1024: ~8-12% faster (less than the learner saving: the learner's kernel dispatch competes with the 14 MuJoCo threads during collection). Simulator threads 10: flat; learner thread at user-interactive QoS: worse (1.37). A first full run was stopped while other processes loaded the machine; running now: C3 + `--pi_batch 768`, three seeds (C3b768_s0-2). Its full run shows why the gain is small: 1.40-1.51 vs 1.33-1.37 it/s before the probe sweeps start (it 400), then the same ~1.05-1.15 it/s as C3 during sweeps. A sweep steps 751 probe worlds 20 times per iteration on the CPU (~15k world-steps, more than the actor's 2.5 x 4096), so with the lighter C3 learner the CPU simulation sets the pace for about two thirds of the run, and learner-side savings only count before it 400. Result, seed 0: 1.18 it/s overall (C3 on an idle machine: 1.16), but 0.786/2, 0.798/1, 0.791/1, 0.797/1, 0.819/0, 0.825/0, 0.828/0 at it 1200-1800: strict >= 1700 vs 1500; a net loss, closed (seeds 1 / 2 not run). Since the CPU sets the pace during sweeps, fewer sweeps: C3 + `--eprobe_every 90` (C3p90): seed 0: probe failures 86 / 39 / 15 / 7 / 2 / 1 / 0 at sweeps 450 / 630 / 810 / 990 / 1080 / 1440 / 1530; evaluation 0.803/2, 0.813/1, 0.820/0, 0.828/0, 0.834/0, 0.837/0, 0.839/0 at it 1200-1800: strict it 1400, 21.80 min as measured (other processes on the machine; 1.09 it/s), vs 1500 for C3. Seed 1: 0.785/2, 0.811/1, 0.818/1, 0.824/1, 0.827/1, 0.832/0, 0.833/0 at it 1200-1800: strict >= 1700 vs 1600 for C3 (`dance1_subject1` fails until 1600: fewer sweeps refresh its focused resets less often). Seed 2: 0.803/1, 0.809/1, 0.819/0, 0.825/0, 0.828/0, 0.834/0, 0.836/0: strict 1500 (C3: 1500), 24.25 min as measured with other processes running. Three seeds 1400 / >= 1700 / 1500, the same mean iteration as C3 (1500 / 1600 / 1500) at a somewhat higher speed per iteration; like the relapse levers, it moves which seed is late. Probe worlds that have finished their tile keep being stepped (`Batch.step` without ids): ~3% of the probe's world-steps, too little to matter. Next: the simulator thread split during sweeps (actor and probe each default to 14 threads on 10 P + 4 E cores), throughput forks of R2_def_s2@1010 with the C3 learner across the sweeps at 1020 and 1080. Result (it/s over it 1020-1130, two rounds, 62-84% CPU of other processes): default 1.21 / 1.02, probe threads 4: 1.20 / 1.05, simulator 10 + probe 4: 1.12 / 1.05, 7 + 7: 1.05 / 1.10: flat within the load noise; closed. Since the learner (~700 ms) is shorter than a CPU-bound sweep iteration (~900 ms), the GPU has slack during sweeps: next, a larger policy batch from it 400 (`--pi_batch 1536` / `1280 --pi_batch_warm 1024 --pi_batch_switch 400`; progress follows policy samples processed), seed 0 (CB1536_s0, CB1280_s0; bit-identical to C3_s0 until it 400). CB1536_s0: 0.89 it/s during sweeps (the learner binds again, ~800-900 ms), imagined reward 0.776 vs 0.757 at it 900; evaluation 0.817/1, 0.825/1, 0.827/1, 0.834/1, 0.839/0, 0.841/0, 0.844/0 at it 1200-1800 (C3_s0: 0.811/0, 0.800/2, 0.821/1, 0.829/0, 0.832/0, 0.832/0, 0.837/0): the return crosses 0.82 at 1300 and ends at BlinkMimic's equal-quality level (0.844), but `dance1_subject1` fails from 1200 to 1500: strict 1600 (28.1 min as measured under load). The return now has margin, so next: + focus replay 10% (policy starts from probe-failure windows: clears failed clips earlier at a 0.003-0.011 return cost in the old regime). CB1280_s0 (batch 1280 from it 400): 0.810/1, 0.815/1, 0.822/1, 0.829/0, 0.831/0, 0.832/0, 0.835/0 at it 1200-1800: strict 1500 like C3_s0, at 1.05 it/s under load: no gain. In every recipe since R, `dance1_subject1` clears at it ~1400-1600 and sets the strict point; batch 1536 moves the return crossing to 1300 but not that clip. Batch 1536 from it 400 + `--focus_replay 0.1` (CB1536fr_s0): the larger batch buys return margin and focus replay spends it on the failing windows: no failed clip from it 1100 (`dance1_subject1` passes from 1100), 0.809/0, 0.816/0, 0.822/0, 0.829/0, 0.832/0, 0.834/0, 0.836/0, 0.837/0 at it 1100-1800: strict target at it 1300 (also 6 in a row), 21.49 min as measured (1.00 it/s, other processes on the machine): 2.05x BlinkMimic on seed 0. Seed 1 (CB1536fr_s1, 1.00 it/s with a node process at ~120% CPU): 0.809/0, 0.812/1, 0.821/0, 0.816/1 (`fallAndGetUp2_subject2`), 0.830/0, 0.832/0, 0.838/0, 0.841/0 at it 1100-1800: strict 1500, 25.27 min as measured (first clean 1300, 21.64 min). Seed 2: 0.806/0, 0.811/0, 0.819/0, 0.823/0, 0.827/0, 0.833/0, 0.834/0, 0.838/0 at it 1100-1800: strict 1400, 23.12 min. Three seeds: strict it 1300 / 1500 / 1400, 21.5 / 25.3 / 23.1 min as measured with other processes running (1.00 it/s), mean 23.3 min (1.89x); no failed clip after it 1100 on two seeds, one relapse on seed 1. A re-timing of seed 1 on a quiet machine ran at the same speed (1.30-1.36 it/s before it 400, 0.92 after), so these times are not inflated: batch 1536 costs ~20% per iteration during the sweeps (the learner binds again) and the recipe is 1.89x, below C3. Focus replay is what clears `dance1_subject1`; next: focus replay with the cheaper batches (1024 = C3, 1280 from it 400), seed 0. C3 + focus replay (C3fr_s0, batch 1024, 1.19 it/s on a quiet machine): no failed clip at 1300-1600, but 0.804/1, 0.810/0, 0.817/0, 0.819/0, 0.824/0, 0.825/1, 0.830/0 at it 1200-1800: the return crosses 0.82 only at 1600; batch 1024 has no margin for focus replay's return cost. Batch 1280 from it 400 + focus replay (CB1280fr_s0, 1.09 it/s): 0.811/1, 0.818/0, 0.822/0, 0.827/0, 0.830/0, 0.832/0, 0.833/0 at it 1200-1800: strict it 1400, 21.31 min (2.07x on seed 0; machine mostly quiet). Seed 1 (CB1280fr_s1, 1.11 it/s, quiet machine): 0.810/1, 0.819/0, 0.822/0, 0.826/0, 0.831/0, 0.824/1, 0.825/1 at it 1200-1800: strict it 1400, 21.01 min (a later relapse at 1700-1800, after the strict point). Seed 2 running. Late relapses are clips that pass with a thin margin (`dance1_subject1`: pelvis 0.26-0.28 m off against the 0.25 m limit), so next: training against tighter termination limits than the evaluation's (`--term_margin 0.9`: 0.225 m / tilt 0.72), fork of R2_def_s2@1010 against LA0 now that the step size is right (`--wm_batches 3`, `--wm_batch 1024`; at the base step size both cost ~100+ iterations, M16 / M17) |

Closed on 2026-11-18 (details in open hypothesis 5): late step-size anneals (policy, world model, both), the 4x critic step size (mixed), deficit replay on top of R (helps one seed), a 1024-row world-model batch, 2 world-model batches, a 16x world-model step size with 3 batches, training termination margins, a 2x final policy step size on C3, 3 Newton-Schulz iterations, a 768-wide world model, a longer policy average (checkpoint averaging), padded first-layer inputs and 2 critic minibatches (no speed).

Closed on 2026-11-17 (see "What did not work"): earlier decay as a full recipe (helps, not reliable on the strict target), probe restart, focus replay 10%, probabilistic termination alone and combined, 4x simulation, final step size below or above the base rate, late compute cuts.

Noise: one run's strict point varies by about +-100 iterations between seeds (+-8% of the time), a fork's
checkpoint by +-0.006 return / +-1 failed clip, and throughput by up to 25% with other workloads on the machine.
A gain below ~10% needs three paired seeds to confirm. The validated result stays `--preset macmimic`: 1.52x vs
BlinkMimic, >= 1.95x vs XMimic; the early-decay candidate is at ~1.85x by iteration counts.

Closed on 2026-11-16 / 17 (details in "What did not work" and "Helps, not in the preset yet"): High Power Mode,
`--utd 0.2` alone, critic fit on the CPU (MLX and NumPy), NumPy inference, hoisted goal encoder, late-only momentum,
shorter policy averages, critic epochs 3 + Newton-Schulz 3, policy batch 512 early, step size ending below the base
rate, probe restart after a failure, late compute cuts (no sweeps and single simulation from it 900), decay from 600
vs 800 (no difference), faster decay (150 iterations).

Outside the from-scratch comparison: starting from a pretrained world model of the robot (`--resume` fine-tuning)
skips the bootstrap phase; it would be reported separately, not against the from-scratch baselines.

## Robustness note

One run in this campaign (2026-11-17 00:37, `--utd 0.2` seed 2, iteration ~650) died with
`[METAL] Command buffer execution failed: Discarded (victim of GPU error/recovery)`: macOS restarted the GPU
("firmware-detected lockup" in `/Library/Logs/DiagnosticReports/gpuEvent-python3.13-*.ips`). A second one killed C3f2_s1 on 2026-11-18 at 03:11 ("victim of GPU error/recovery", other
GPU workloads were running). A run killed this way can only be continued exactly from a `--snap_at` fork point (the
replay buffer is not in the regular checkpoints); `--resume` from `last` restarts with an empty buffer.
