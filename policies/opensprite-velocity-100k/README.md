# OpenSprite v5 Xbox velocity policy · update 100,000

This is the seed 49 MacMimic policy for the SuperMimic §4.5 / Appendix G velocity task on OpenSprite v5. Training from scratch took 38 h 32 min on a 48 GB M4 Pro. The archive includes the final actor, its matching JSON contract, the 37-second virtual Xbox replay and evaluation results. Hardware qualification is pending.

## Load the policy

Keep `policy.onnx` and `contract.json` together. Verify the package from this directory with `shasum -a 256 -c SHA256SUMS`. For simulation on an Apple silicon Mac, run the following from the MacMimic repository root after checking out the release tag `opensprite-velocity-s49-it100000`.

```sh
uv sync --locked --extra opensprite
mkdir -p data
git clone https://github.com/BeijingDynamics/OpenSprite-SuperMimic.git data/OpenSprite-SuperMimic
git -C data/OpenSprite-SuperMimic checkout --detach bc570e272f5ae07067d5c75c1c1b3051bf615b6a
.venv/bin/python data/OpenSprite-SuperMimic/scripts/fetch_robot_assets.py
package=policies/opensprite-velocity-100k
.venv/bin/python tools/opensprite_control.py "$package" --sequence
```

To drive in MuJoCo with a paired Xbox controller, run `.venv/bin/mjpython tools/opensprite_control.py "$package" --xbox`. Center the sticks, release LT and hold it to enable. Left Y commands forward/reverse, left X commands lateral velocity and right X commands yaw. Releasing LT ramps all three commands to zero.

## Robot runtime interface

| Item | Value |
|---|---|
| Robot and timing | OpenSprite v5, 50 Hz policy, 20 ms between policy calls |
| Input | Float32 `obs`, 993 values per sample. Each of the five proprioceptive terms holds 10 frames, oldest first within each term. The final three values are the current body-frame forward and leftward velocity (m/s) and yaw rate (rad/s). |
| State | Joint positions relative to the contract default pose, joint velocities, previous raw policy actions, root-local gyro and gravity projected into the base frame. Normalization is embedded in the ONNX. |
| Output | 31 raw actions in contract joint order. Target position in radians equals `action_offset + action_scale * policy_action`. Feed the raw actions back into the history. |
| Simulation command range | Forward −1.5 to +2.0 m/s, lateral ±1.32 m/s, yaw ±3.35 rad/s. Physical command limits require separate validation on the robot. |

The contract records the asset revision, control-profile fingerprint, joint names, gains, limits, startup handoff and ONNX fingerprint. For sim-to-real integration, verify joint-to-motor order, signs, zero offsets, the robot's ankle transmission and IMU frame against the specific hardware configuration. The export uses joint-space targets with `physical_ankle_differential.enabled = false` and `hardware_qualified = false`. Upstream R1 hardware results apply to its own weights and runtime configuration. [Pinned OpenSprite runtime](https://github.com/BeijingDynamics/OpenSprite-SuperMimic/tree/bc570e272f5ae07067d5c75c1c1b3051bf615b6a/runtime)

To compare the upstream runtime's history packing with this controller on simulated sensor data, run the command below from the repository root. It writes a fresh report in the package directory.

```sh
.venv/bin/python tools/opensprite_validation/upstream_history.py "$package"
```

## Simulation results

- 672 velocity commands × 4 starts × 5 seconds. There were zero falls in 2,688 trials. The median command-level relative tracking error was 5.57%. [Step tests](evaluation/paper_steps.json)
- One continuous 37-second virtual Xbox sequence finished upright. The final two seconds of the release phase averaged −0.0013 m/s forward, +0.0008 m/s lateral and +0.0065 rad/s yaw. [Video](xbox_sequence.mp4) · [Measurements](evaluation/xbox_sequence.json)
- Small commands from rest remain weak. The robot averaged −0.019 m/s for a −0.10 m/s reverse command and −0.0014 m/s for a +0.12 m/s lateral command. [Low-speed trials](evaluation/velocity_trials.json)
- The pinned upstream observation history and the local controller agreed to within 0.00000185 in raw action on the same 1,850 simulated frames. [Runtime comparison](evaluation/upstream_history_replay.json)

## Motion data

This port selected locomotion clips and retargeted them to OpenSprite. The final training bank contains 7,268,849 frames at 50 Hz (40.38 hours). Its SHA-256 is recorded in `contract.json`. The selection follows the Appendix G locomotion task. The clip selection and OpenSprite retargeting are specific to this run. [SuperMimic paper](https://arxiv.org/abs/2610.28480)

- **LAFAN1** by Ubisoft La Forge. The bank contains 760 segments from 18 walking, running and sprinting recordings, prepared from the Geno-resolved motions. [Official dataset and terms](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
- **BONES-SEED** by Bones Studio. The bank contains 24,156 locomotion clips retargeted from the G1 distribution. The source files matched official revision `2f59b2077b9da34dd4e43618e705c7cb962c9a66`. [Official dataset](https://huggingface.co/datasets/bones-studio/seed)

Training data includes [Motion Data by Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the [BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license).
