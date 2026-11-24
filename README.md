# MacMimic for OpenSprite v5

## OpenSprite Xbox velocity policy · 100,000 updates

[![OpenSprite v5 following Xbox commands in MuJoCo with the final 100,000-update policy](docs/assets/opensprite-velocity-100k.webp)](policies/opensprite-velocity-100k/xbox_sequence.mp4)

*The 37-second replay shows forward walking, left and right strafing, reverse walking, turns and a stop on trigger release. [Play the final MP4](policies/opensprite-velocity-100k/xbox_sequence.mp4).*

The final Appendix G velocity policy trained on a 48 GB M4 Pro for 38 h 32 min. In MuJoCo it completed 2,688 five-second trials with zero falls and a median relative tracking error of 5.57%. It follows three-axis Xbox commands. Small reverse and lateral commands from rest remain weak. On OpenSprite v5 hardware, the same ONNX completed a full-gain 300-second Xbox control window with zero CAN deadline misses and all 31 drives confirmed disabled on exit. A subsequent hardware trial confirmed leftward and rightward walking with the ±0.50 m/s lateral command profile. [Jetson runtime, evidence and deployment](sim2real/README.md)

**OpenIMU on the physical robot.** The Jetson runtime now defaults to an OpenIMU connected through a USB-to-CAN FD adapter. OpenIMU and the original SPI board use the same ICM-45686 sensor, mounting rotation and attitude estimator. The original board remains available with `--imu-backend spi`. After the final mounting adjustment, the two sensors' mean gravity directions agreed within 0.26° at rest. On battery power, OpenSprite passed full-gain standing and the operator confirmed normal Xbox-controlled walking with OpenIMU. All 31 drives reported disabled at the end of the walking test. [OpenIMU firmware and hardware](https://github.com/TonyInvent/OpenIMU) · [Jetson setup and IMU selection](sim2real/README.md#imu-selection) · [Powered test record](sim2real/evidence/openimu-battery-validation.json)

OpenIMU streams raw accelerometer and gyroscope data over CAN FD at **800 Hz**. In the Jetson direct-read test, consecutive sensor timestamps were **1.25 ms** apart. After calibration, the runtime integrates each accepted frame using its sensor timestamp and publishes the updated attitude to shared memory. The **50 Hz** walking policy reads the latest estimate every 20 ms, after roughly 16 IMU updates. Across 803 read-only policy steps, the estimated age of the OpenIMU sample at inference was 0.64 ms at the median and 1.33 ms at p99. [Jetson timing measurements](sim2real/evidence/imu-direct-read-253-audit.json)

[Download the ONNX + contract package](https://github.com/TonyInvent/MacMimic/releases/download/opensprite-velocity-s49-it100000/opensprite-velocity-100k.zip) · [Sim-to-real interface and replay instructions](policies/opensprite-velocity-100k/README.md)

Motion sources are [Ubisoft La Forge LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed). Training data includes [Motion Data by Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the [BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license).

## Earlier Sprite SM-Walk R1 policy

[Watch the earlier 35-second R1 replay (update 1,800)](docs/opensprite-r1/xbox_sequence.mp4).

This fork trains OpenSprite on Apple silicon with MacMimic and the reviewed Sprite SM-Walk R1 walking and standing references. On an M4 Pro with 48 GB, seed 49 completed 6,000 updates in 2 h 27 min. **Update 1,800** is the selected Xbox velocity policy. It walked at 0.278 m/s under a 0.40 m/s command in MuJoCo and stopped on trigger release. Right turns remained weak. Hardware testing of these weights is pending.

[Run the policy or reproduce training](docs/opensprite.md) · [Download ONNX and MLX checkpoint](https://github.com/TonyInvent/MacMimic/releases/tag/opensprite-r1-mac-s49-it1800) · [Evaluation results](docs/opensprite-r1/evaluation)

## Original MacMimic G1 benchmark

<p align="center"><img src="docs/assets/macmimic.gif" width="640" alt="The simulated Unitree G1, in white, doing a cartwheel and getting up from the floor next to the human reference motion, in blue"></p>

**A simulated Unitree G1 learns to track 40 human motions in 22 minutes on a MacBook Pro, 2x faster than a BlinkMimic
port on the same Mac.** MacMimic trains whole-body motion tracking end to end on Apple silicon: MuJoCo simulates 4,096
robots on the CPU while MLX trains a differentiable world model and the policy on the GPU. No CUDA, no PyTorch, no cloud.

MacMimic is BlinkMimic's algorithm (SuperMimic's method, refined), run with XMimic's overlap of simulator and learner,
and tuned for a Mac.

[Project page](https://air23zj.github.io/macmimic/) · [Results](RESULTS.md) ·
[What worked and what didn't](docs/macmimic_hillclimb.md)

## Results

<p align="center"><img src="docs/assets/time_to_target.png" width="760" alt="Minutes to the benchmark target on one MacBook Pro: MacMimic 21.6 minutes (mean of 3 runs), BlinkMimic port 44 minutes, XMimic port not reached in 57 minutes"></p>

MacBook Pro, M4 Pro, 48 GB. Target: average evaluation return of at least 0.82 with none of the 40 clips failing,
held for three checkpoints in a row.

| Preset | Runs | Time to target | Iteration |
|---|---|---|---|
| `macmimic` | 3 seeds | 21.0 / 21.3 / 22.5 min, mean 21.6 | 1400 / 1400 / 1500 |
| `macmimic_v1` (first version) | 3 seeds | 27.9 / 29.2 / 30.1 min, mean 29.05 | 1600 / 1700 / 1700 |
| `blinkmimic` (this repository's port) | 1 | 44.0 min | 1600 |
| `xmimic` (this repository's port) | 1 | not reached in 57 min | |

- The baselines are this repository's MLX ports, timed on the same Mac; this is not a comparison with the original CUDA
  code on an NVIDIA GPU.
- `macmimic` is 2.04x faster than the BlinkMimic port on the mean of three runs (one seed alone: 1.95x). The target is a
  threshold: at their target checkpoints BlinkMimic tracks somewhat better (return 0.847, 3.9 cm body position error)
  than `macmimic` (0.822, 4.4 cm). Tracking metrics for every checkpoint shown are in [RESULTS.md](RESULTS.md).
- The G1 baseline results in this table are simulation measurements. The OpenSprite hardware runs are documented separately above.

## Quickstart

```bash
git clone https://github.com/TonyInvent/MacMimic && cd MacMimic
uv sync
bash tools/download_data.sh                                  # 40 LAFAN1 clips retargeted to the G1 -> data/
uv run python tools/urdf_to_mjcf.py
bash scripts/train.sh macmimic logs/macmimic_s0 --seed 0     # 2000 iterations, ~30 min on an M4 Pro
uv run python -m supermimic_mlx.evaluate --run_dir logs/macmimic_s0 --min_it 1000
uv run python tools/time_to_target.py logs/macmimic_s0
```

## What is in this repository

The code started as a port of [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (which also contains a faithful
SuperMimic baseline) from mjlab / MuJoCo Warp / PyTorch + CUDA to Apple silicon. All presets share the simulator, MDP,
reward and evaluator, as upstream:

| upstream (RTX 4090) | here (M-series Mac) |
|---|---|
| mjlab scene, MuJoCo Warp, 4096 worlds on the GPU | the same scene built with plain MuJoCo (`supermimic_mlx/scene.py`, checked field by field against mjlab's compile), 4096 worlds stepped by `mujoco.batch` on the CPU thread pool, per-world randomized model fields and solver warm starts |
| PyTorch networks, `torch.compile`, CUDA graphs, bf16 autocast | MLX networks, `mx.compile` over each whole update (world model; BPTT policy + critic), bf16 matmuls where the GPU runs them faster |
| `torch.optim.Adam` / `torch.optim.Muon` | `supermimic_mlx/optim.py`, same arithmetic (tested against PyTorch) |

- `--preset macmimic` — MacMimic: `macmimic_v1` with the world model at 8x the step size and 3 instead of 5 updates per
  iteration (19 % less time per iteration), LayerNorm on bf16 activations, the step-size decay from iteration 800, and
  from iteration 400, when the probe sweeps make the CPU the bottleneck and leave the GPU slack, a policy batch of 1280
  with 10 % of the policy's imagined rollouts started where the probes find failures.
- `--preset macmimic_v1` — the first MacMimic (29 min): the `xmimic` preset retuned for time to target on a Mac.
  Policy batch 1024 instead of 2048 (38 % less time per iteration for ~200 more iterations), step-size decay from
  iteration 1100, policy average 0.99, evaluation probes from iteration 400 every 60 iterations on a worker thread,
  2000 iterations. What was tried on the way, including what did not work, is in
  [docs/macmimic_hillclimb.md](docs/macmimic_hillclimb.md).
- `--preset supermimic` — SuperMimic (Liao et al., arXiv:2610.28480): upstream `train.py` without flags (paper hyperparameters).
- `--preset blinkmimic` — BlinkMimic: upstream `PRESET[main]` (structured semi-implicit world model, mirror
  augmentation, Muon on policy and world model, policy EMA, reference-action skip, contact reward, failure- and
  evaluation-guided resets, policy batch 2048 with a decaying 4× policy LR, 3500 iterations).
- `--preset xmimic` — [XMimic](https://github.com/nico-bohlinger/xmimic): BlinkMimic's algorithm and hyperparameters with
  XMimic's actor | learner split mapped onto one Mac: the learner (world model, policy, critic) runs from a worker thread
  on its own GPU stream while the main thread steps the CPU simulator and the probe sweeps with the policy of the
  previous iteration (XMimic's one-iteration staleness).

## Automatic configuration per Mac

`supermimic_mlx/hw.py` detects the chip (M1 … M6, base / Pro / Max / Ultra), its cores and GPU memory at startup and
sets what only affects speed and memory: simulator threads (all cores), MLP matmul precision (`--precision auto`:
bf16, upstream's autocast, wherever a short micro-benchmark shows this GPU runs it at least as fast as fp32 — M1/M2
emulate bf16 and get fp32), BPTT checkpointing when the estimated learner peak plus the 3.6 GB replay buffer would not
fit the GPU's working set, and MLX's wired-memory limit. The choices are printed (`[hw] ...`) and stored in `args.json`.
Every setting can be forced by its flag (`--sim_threads`, `--precision`, `--checkpoint_bptt`).

Measured on an M4 Pro (14 CPU cores, 48 GB), BlinkMimic's algorithm, 4096 worlds:

| | it/s | 3500 iterations | GPU peak |
|---|---|---|---|
| fp32 | 0.57 | 102 min | 9.2 GB |
| `blinkmimic` (auto: bf16) | 0.65 | 90 min | 6.4 GB |
| `xmimic` (auto: bf16 + actor \| learner) | 0.72 | 81 min | 6.5 GB |

The learner is GPU-compute bound (the BPTT forward runs at ~5.7 TFLOPS, ~70 % of the bf16 matmul peak overall), so the
remaining time is the algorithm's arithmetic; `--profile` prints a per-phase breakdown, `tools/bench_learner.py` times
the pieces of the policy update.

## Setup

macOS on Apple silicon, [uv](https://docs.astral.sh/uv/), `curl`, `ffmpeg` (videos only).

```bash
uv sync
bash tools/download_data.sh                            # 40 G1-retargeted LAFAN1 clips + G1 description -> data/
uv run python tools/urdf_to_mjcf.py                    # G1 MJCF (free-floating pelvis)
uv run python tools/test_env.py                        # state round-trip, FK vs MuJoCo, reference playback
uv run --group dev pytest -q                           # unit tests
# optional: the mjlab reference scene for tests/test_scene.py (mjlab installs on macOS; only used here)
uv run --with mjlab python tools/build_scene_mjlab.py
```

## Train, evaluate, render

```bash
bash scripts/train.sh blinkmimic logs/blinkmimic_s0 --seed 0     # 3500 iterations
bash scripts/train.sh xmimic logs/xmimic_s0 --seed 0
bash scripts/train.sh macmimic logs/macmimic_s0 --seed 0         # 2000 iterations, ~30 min on an M4 Pro
bash scripts/train.sh macmimic_v1 logs/macmimic_v1_s0 --seed 0   # the first version, ~35 min
bash scripts/train.sh supermimic logs/supermimic_s0 --seed 0 --iterations 16000
bash scripts/benchmark.sh logs/bench                              # all three, then evaluation and time to target
uv run python -m supermimic_mlx.evaluate --run_dir logs/blinkmimic_s0          # every checkpoint -> eval.csv
uv run python tools/time_to_target.py logs/blinkmimic_s0                       # minutes to return >= 0.82
uv run python tools/render.py --ckpt logs/blinkmimic_s0/ckpt_003500 --clip fallAndGetUp1_subject1 --out fall.mp4
# as on the project page: simulated from the clip's first frame, reference beside the robot
uv run python tools/render.py --ckpt logs/macmimic_s0/ckpt_001400 --clip dance1_subject1 --from_clip_start \
    --start_s 104.5 --seconds 5 --look studio --ghost side --azimuth 90 --out cartwheel.mp4
```

`python -m supermimic_mlx.evaluate --watch` evaluates checkpoints while a run trains (it slows the run down).

## What differs from upstream, and why

- **Simulator backend.** CPU MuJoCo instead of MuJoCo Warp, with the same model, options (5 ms implicitfast,
  pyramidal cone, Newton 100/50 iterations, tolerance 1e-6) and per-world state, including the solver warm start
  (cleared at resets, as MuJoCo Warp's reset does). Collision meshes are reduced to their convex-hull vertices
  (MuJoCo collides with the hulls anyway; the trajectories agree to round-off, 1e-7 over 4 s) — this makes per-world
  models cheap. Results are not bitwise equal to MuJoCo Warp (different solver implementation), and CPU MuJoCo is
  deterministic, so evaluation needs no replicas.
- **Ground friction.** Upstream sets the friction of the world body's geoms to the bottom of the randomization range,
  but mjlab's plane is a child body ("terrain"), so that set is empty and the ground keeps friction 1.0. This port
  keeps that behavior (MuJoCo takes the max of a contact pair, so the effective friction is in [1.0, 1.2]).
- **Systems flags.** `--cuda_graph`, `--collect_graph`, `--reset_masked`, `--graph_early`, `--fused_*`,
  `--compile_more`, `--no_sync`, `--eprobe_graph`, `--ckpt_light` are CUDA wall-clock engineering; they have no MLX
  counterpart (mx.compile covers the fusion). Their random-stream changes do not apply; the masked reset is the eager
  reset. `--eprobe_async` keeps its semantics (the sweep advances this many steps per iteration with a snapshot of the
  EMA policy taken when it starts, and its failures apply when it ends), run inline.
- **Precision.** MLP matmuls in bf16 with fp32 master weights, LayerNorm, heads and losses in fp32 (upstream's
  autocast), or fp32 on GPUs without fast bf16. Muon's Newton-Schulz iteration runs in fp32 (torch: bf16).
- **Memory.** `--checkpoint_bptt` recomputes each imagined step in the backward pass; chosen automatically from the
  GPU memory (the SuperMimic policy batch of 4096 needs ~15 GB in bf16 without it).
- **XMimic.** Its JAX/XLA engineering (whole iterations as one XLA program, multi-GPU layouts, bf16 kernels) is
  CUDA-specific; what carries over to one Mac is the actor | learner overlap. Its "fast learner" hoisting is available as
  `--fast_learner` but is slower under MLX (one large batched goal encoding compiles worse than the per-step one), so the
  preset leaves it off. XMimic's 256 constraint rows per world do not apply (CPU MuJoCo allocates per world as needed).

## Layout

```
src/supermimic_mlx/
  robots/g1.py      G1 constants (actuators, joints, bodies, randomization) and the LAFAN1 loader
  scene.py          the MuJoCo scene (= mjlab's), solver options
  sim.py            batched tracking simulator on mujoco.batch: resets, reset sampler, sim context xi
  kinematics.py     differentiable FK (MLX)          motion.py   reference library, goal features
  mdp.py            reward, termination, observation, critic features
  networks.py       world model, policy, critic      optim.py    Adam / Muon (torch semantics)
  buffer.py         replay buffer                    mirror.py   left-right mirror augmentation
  train.py          training (all presets)           evaluate.py checkpoint evaluation
tools/              data download, MJCF, mjlab reference scene, env check, render, time to target,
                    tracking metrics, failure diagnostics (probe_failmap.py, move_diag.py)
docs/               project page (index.html, GitHub Pages) and the hill-climb record
```

## Attribution and licenses

This combined project uses the [GNU AGPL-3.0](LICENSE). Upstream MacMimic retains its [MIT license](licenses/MacMimic-MIT.txt). See the [third-party notices](THIRD_PARTY_NOTICES.md) for component and motion credits. MacMimic was ported from [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (Fuyi Chen, MIT), which builds on
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (Chong Zhang, MIT) and
[InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (MIT); method: SuperMimic
([HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic), Apache-2.0; Liao et al. 2026), BeyondMimic
(reward, evaluation), SHAC / SuperTrack (learning through a differentiable model), Muon (Jordan et al. 2024; Liu et al.
2025). Simulation: [MuJoCo](https://mujoco.org) (Apache-2.0), [mjlab](https://github.com/mujocolab/mjlab) for the
reference scene. **Data:** LAFAN1 motions by Ubisoft La Forge, CC BY-NC-ND 4.0; G1 retargets and description by Unitree
(BSD-3-Clause), via the Hugging Face dataset `lvhaidong/LAFAN1_Retargeting_Dataset`. The data are not included;
trained checkpoints and videos derived from LAFAN1 (including the media in `docs/assets/`) are for
non-commercial research use only.
