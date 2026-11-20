# OpenSprite on Apple silicon

## Sprite SM-Walk R1

### Run the published policy in MuJoCo

Run from the repository root on an Apple silicon Mac with Git and uv installed. The simulation uses the pinned OpenSprite model and 33 verified meshes. The policy reads joint and IMU history and a velocity command. Motion reference files are only needed for motion tracking or training.

```sh
uv sync --locked --extra opensprite
mkdir -p data
git clone https://github.com/BeijingDynamics/OpenSprite-SuperMimic.git data/OpenSprite-SuperMimic
git -C data/OpenSprite-SuperMimic checkout --detach bc570e272f5ae07067d5c75c1c1b3051bf615b6a
.venv/bin/python data/OpenSprite-SuperMimic/scripts/fetch_robot_assets.py
mkdir -p logs/opensprite-release
curl -fL --retry 3 https://github.com/TonyInvent/MacMimic/releases/download/opensprite-r1-mac-s49-it1800/opensprite-r1-policy.zip -o logs/opensprite-release/policy.zip
unzip -o logs/opensprite-release/policy.zip -d logs/opensprite-release
.venv/bin/python tools/opensprite_control.py logs/opensprite-release/policy --sequence
```

For interactive simulation, pair an Xbox controller with macOS and run:

```sh
.venv/bin/mjpython tools/opensprite_control.py logs/opensprite-release/policy --xbox
```

Center both sticks, release LT, then hold LT to enable driving. Left-stick Y commands forward speed, right-stick X commands turning, and releasing LT commands a stop. The selected export includes its ONNX, input contract and local simulation measurements. See the [60-checkpoint comparison](opensprite-r1/evaluation/checkpoint_comparison.json).

The separate [upstream release](https://github.com/BeijingDynamics/OpenSprite-SuperMimic/releases) used for comparison was unpacked on the training machine under `data/OpenSprite-SuperMimic/outputs/r1-mac/release/official_policy`. Its ONNX SHA-256 is `5695b6980746534a11d075befc29bf5f7fe0ef2a7d81e3dadf3147c50f4d6930`. To drive the upstream release in the same simulator:

```sh
.venv/bin/mjpython tools/opensprite_control.py data/OpenSprite-SuperMimic/outputs/r1-mac/release/official_policy --xbox
```

The official 6,000-update PyTorch checkpoint identifies the training bank as `edd4ddd6f54e8a8a710481d11340ac6e38d6b944e75613c48e318fc3b32d296f`. Its two clips at 50 Hz are `pm01_clearance_walk` (572 frames, 80% of resets) and `Neutral_ID_001135_001495_stand` (300 frames, 20%). The public release contains the final checkpoint and policy. The exact bank and transfer parent are separate inputs. See the [R1 model card](https://github.com/BeijingDynamics/OpenSprite-SuperMimic/blob/bc570e272f5ae07067d5c75c1c1b3051bf615b6a/docs/SPRITE_SM_WALK_R1.md).

For retraining, obtain the exact reviewed bank from its owner and place `walk_stand.npz` in `data/OpenSprite-SuperMimic/outputs/r1-mac/original/`. The original standing medians are +4.3564° and −4.1396°. The separate local reconstruction is excluded. See the [reference provenance and fingerprints](opensprite-r1/reference.json).

MuJoCo prints the upstream robot model name `opensprite0825_waist20261105_run002` when it loads the v5 MJCF. For the motion input used by a run, check `opensprite_bank`, `reference_sha256` and `reference_clip_weights` in that run's `args.json`. The R1 training entry checks the file against the published hash before starting.

The local experiment starts with random weights and seed 49. It uses the MacMimic learner and simulation pipeline, the original two-clip bank with 80/20 reset sampling, and the R1 sole-height, posture, step-width and moving-foot clearance reward terms. The upstream PyTorch parent checkpoint is archived separately. Local and upstream training trajectories have different initialization, architecture and update schedules.

The [versioned release](https://github.com/TonyInvent/MacMimic/releases/tag/opensprite-r1-mac-s49-it1800) also includes the selected native MLX checkpoint, its companion JSON and archive checksums. To re-export from the repository root, unpack the checkpoint archive into `logs/opensprite-release` and run `.venv/bin/python tools/export_opensprite.py logs/opensprite-release/ckpt_001800 logs/opensprite-release/reexport`.

### Results on the M4 Pro

Seed 49 completed 6,000 iterations and 30,851,072 simulation steps in 2 h 27 min. Both training interfaces used the local Mac. The motion encoder used MacMimic's reference-relative goals. The official NVIDIA configuration uses robot-relative goals and transfer initialization.

We exported and tested all 60 saved checkpoints in the v5 simulator. Update 1,800 completed both motion clips with mean normalized return 0.974. It remained upright in four 35-second driving sequences, three steady-state tests, eight independent stick settings and a 60-second SDL virtual gamepad trial. For selection we checked forward motion, braking, bilateral stepping and correct turn direction across the driving trials, then compared the weaker of the left and right turn responses. The chosen package is `logs/opensprite_r1_mac_s49/velocity_001800`. The final update degraded. Its normalized motion return was 0.440 and its forward speed after the stop command was 0.218 m/s.

| Nominal 35-second simulation | Command | Mac-trained update 1,800 | Upstream R1 in same simulator |
|---|---:|---:|---:|
| Forward speed | 0.400 m/s | 0.278 m/s | 0.295 m/s |
| Left yaw rate while walking | +0.200 rad/s | +0.071 rad/s | +0.176 rad/s |
| Right yaw rate while walking | −0.200 rad/s | −0.013 rad/s | −0.157 rad/s |
| Forward speed after releasing LT | 0 m/s | −0.001 m/s | −0.003 m/s |

During a separate steady forward command of 0.40 m/s, the Mac-trained policy alternated feet and lifted each foot 17 times in 10 seconds. The upstream policy lifted each foot 13 times. We measured swing onset at 8 mm and return to stance at 2 mm. The Mac policy responds to starting and braking. Its turns, particularly right turns, are weak. Validation used MuJoCo and virtual gamepad input. Physical gamepad and robot testing remain outstanding.

```sh
bank=data/OpenSprite-SuperMimic/outputs/r1-mac/original/walk_stand.npz
caffeinate -is .venv/bin/python -u -m supermimic_mlx.train \
  --preset macmimic --opensprite_r1_bank "$bank" --seed 49 --goal_stride 5 \
  --num_envs 2048 --pi_batch 768 --pi_batch_warm 512 --wm_batch 1536 \
  --iterations 6000 --log_every 25 --log_dir logs/opensprite_r1_new_s49
```

## Historical run002 port experiment

This workspace uses the OpenSprite v5 asset and the five run002 source clips from [OpenSprite-SuperMimic](https://github.com/BeijingDynamics/OpenSprite-SuperMimic/tree/bc570e272f5ae07067d5c75c1c1b3051bf615b6a). The local checkout lives under `data/OpenSprite-SuperMimic`. Set `OPENSPRITE_REPO` if moving that checkout. Its 33 pinned STL meshes must be present.

### Earlier run002 policy trained on this Mac

The M4 Pro trained seed 43 for 3,000 iterations and 15,491,072 simulator steps in 56.7 minutes. Both iteration 2,800 and iteration 3,000 completed all five motion clips, with mean normalized tracking returns of 0.843 and 0.853 respectively. For Xbox driving, use the exported actor from iteration 2,800.

```sh
.venv/bin/mjpython tools/opensprite_control.py logs/opensprite_joint_v2_s43/velocity_002800 --xbox
```

We compared checkpoints 1,300 through 3,000 at intervals of 100. Each ran the same 35-second joystick sequence from the nominal pose and three slightly perturbed starts, plus eight independent stick settings. The score uses two-thirds weight for the continuous driving trials and one-third for the individual settings. It averages planar velocity errors divided by 0.40 m/s and yaw-rate errors divided by 0.20 rad/s. Iteration 2,800 had the lowest combined error in this comparison. The comparison scripts and rankings are in `logs/opensprite_joint_v2_s43/validation/`.

| Nominal 35-second trial | Command | Measured |
|---|---:|---:|
| Forward speed | 0.400 m/s | 0.339 m/s |
| Yaw rate while walking left | +0.200 rad/s | +0.175 rad/s |
| Yaw rate while walking right | −0.200 rad/s | −0.166 rad/s |
| Forward speed after releasing LT | 0.000 m/s | −0.001 m/s |

All four starting configurations completed the full sequence. A second 60-second closed-loop trial used an SDL virtual game controller, partial stick deflections, braking with the sticks still deflected, re-enabling, and starting again. The robot remained upright. Full reverse produced essentially zero travel. Stationary yaw produced approximately +0.031 and −0.008 rad/s at full left and right stick. Half forward stick from rest produced essentially zero travel, and three-quarter stick produced 0.247 m/s. These tests exercised the simulator and virtual input. A physical gamepad and physical robot have not been tested with this policy.

The ONNX, contract, video, and per-phase JSONs are in `logs/opensprite_joint_v2_s43/velocity_002800/`. In a 1,750-frame replay using the upstream observation-history implementation, the maximum absolute action difference was 1.44e-6 or less. The separate MLX-to-ONNX export check had a maximum error of 7.63e-6.

## Reference motions

The local reconstruction is `data/OpenSprite-SuperMimic/data/motions/run002_mac/v5_locomotion.npz`, SHA-256 `17ba162a93d9c6432aae6ed3d696c51398262d7ddbe3da2a72877dd1e96a59cd`. The five Geno crops match the published hashes. UMR correspondence training ran on CPU. Each sequence contains 400 robot frames at 50 Hz. The original run002 bank has a different hash.

The local reconstruction used the upstream contact optimizer for straight walking and stopping. For right turning, the resolver used collision penalties in the trajectory objective. For starting and left turning it also adjusted swing clearance during optimization. The solver variants are in the local OpenSprite checkout under `scripts/refine_contact_merit.py`, `scripts/refine_contact_swing.py`, and `scripts/refine_swing_cost.py`. The upstream acceptance thresholds were retained. The bank directory contains the QA results, source-bound reports and preview images.

## Train

Run commands from the MacMimic repository root. Use a fresh log directory for each new run.

```sh
uv sync --locked --extra opensprite
bank=data/OpenSprite-SuperMimic/data/motions/run002_mac/v5_locomotion.npz
run=logs/opensprite_new_s43
caffeinate -is .venv/bin/python -u -m supermimic_mlx.train \
  --preset macmimic --opensprite_bank "$bank" --log_dir "$run" --seed 43 \
  --num_envs 2048 --pi_batch 768 --pi_batch_warm 512 --wm_batch 1536 \
  --iterations 3000 --log_every 25
```

Specifying the bank enables OpenSprite joint training. Half the simulator worlds follow reference poses. The other half use planar body velocity and yaw-rate commands. Both actors use the same ten-frame proprioception history, action decoder and structured world model. Each has its own encoder and critic. The velocity encoder receives gradients through imagined velocity rewards and aligns its latent to the motion encoder on reference-derived commands. Only motion-policy gradients train the shared decoder. The command actor has no runtime dependency on reference frames. Training uses MacMimic's MLX learner, Muon updates, CPU MuJoCo collection, overlap, averaged weights and failure probes. Speed commands use a three-channel running normalizer. The shared decoder is trained without the reference-angle skip. Sagittal mirror augmentation is disabled for OpenSprite because actuator axes and the ankle geometry differ from G1. The Mac simulator uses implicit position PD, named gains and effort caps, and ten 2 ms substeps per policy tick.

## Export and evaluate

```sh
checkpoint="$run/ckpt_003000"
.venv/bin/python tools/export_opensprite.py "$checkpoint" "$run/velocity"
.venv/bin/python tools/opensprite_control.py "$run/velocity" --seconds 12 --out "$run/velocity/eval.json"
.venv/bin/python -m supermimic_mlx.evaluate --ckpt "$checkpoint"
```

Export uses the averaged motion decoder and velocity encoder. It compares ONNX Runtime outputs to MLX FP32 on 32 input histories and checks the upstream offline replay contract. The 993 floats contain histories of joint position relative to default, velocity, previous raw policy action, local angular velocity, projected gravity, and the current three-component command. Within each term the oldest frame comes first. The 31 output actions map to joint targets using the named default pose and 0.25 radians per action unit.

## Xbox in MuJoCo

Pair an Xbox controller with macOS, then run the following command.

```sh
.venv/bin/mjpython tools/opensprite_control.py "$run/velocity" --xbox
```

Center the sticks, release LT, then hold LT. Left-stick Y requests forward or reverse and right-stick X requests yaw. The lateral command is zero in this mapping. The forward, reverse and turn limits are 0.40 m/s, 0.05 m/s and 0.20 rad/s. Releasing LT ramps the target to zero. Closing the viewer exits. The demo drives a MuJoCo model and ends on a fall. The exported contract identifies the policy as simulation-tested only after separate rollout evaluation. Motor-space mapping and physical robot qualification belong to the hardware runtime.

To replay the same stick mapping without a controller, run the 35-second sequence below. It includes one continuous rollout from standing through forward walking, both turns and release. The JSON records the mean measured velocity over the final two seconds of each phase, or the full phase if shorter.

```sh
.venv/bin/python tools/opensprite_control.py "$run/velocity" --sequence --out "$run/velocity/xbox_sequence.json"
```

Use `--video "$run/velocity/xbox_sequence.mp4"` to record that rollout. The renderer uses `ffmpeg` from the shell path or the executable specified by `FFMPEG`.

## Replay the published validation

From the repository root, set `package=logs/opensprite-release/policy`. Use the following commands for steady velocity response, perturbed starts, a virtual SDL controller and upstream history replay.

```sh
package=logs/opensprite-release/policy
.venv/bin/python tools/eval_opensprite_r1.py "$package" --out "$package/r1_eval.json"
.venv/bin/python tools/opensprite_validation/stick_trials.py "$package"
.venv/bin/python tools/opensprite_validation/sdl_driving.py "$package"
.venv/bin/python tools/opensprite_validation/upstream_history.py "$package"
```

The gait comparison also reads the reviewed training bank at the path in the R1 training instructions. The [saved evaluation](opensprite-r1/evaluation) includes bilateral stepping counts, ONNX replay parity and the per-checkpoint selection results.
