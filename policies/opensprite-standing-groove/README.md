# OpenSprite standing groove · 12-second lab test

This is the fixed choreography with the drum beat, exported from MacMimic seed 49 at update 100,000. The model embeds 600 motion latents and decodes them using current joint and IMU feedback. The original walking policy has separate hardware results. This choreography has been evaluated in ground-contact MuJoCo simulation. [Preview](standing_groove_preview.mp4)

## Jetson offline replay

Unzip the release and run these commands inside the `opensprite-standing-groove` directory using the Jetson Python environment with NumPy and ONNX Runtime CPU. The replay runs entirely in memory and prints the number of frames, maximum error against MLX and measured CPU p99 latency.

```sh
sha256sum -c standing_groove_SHA256SUMS
python3 standing_groove_replay.py
```

The fixture contains all 600 simulated observation histories packed in the walking runtime's layout and the corresponding raw MLX actions. The Mac CPU replay passes at a maximum absolute action difference of 0.0000124. The wall time of this offline inference test measures model calls only.

## Jetson hardware integration

The dedicated standing-groove mode in the Jetson runtime packs the walking runner's 993-float observation into the history tensor below and supplies the frame index. The Xbox walking launcher has its own policy binding. [Jetson setup and commands](../../sim2real/README.md#standing-groove-on-jetson-253)

| Name | Type and shape | Contents |
|---|---|---|
| `history` | float32 [1, 10, 99] | Ten frames from oldest to newest. Each contains 31 joint angles relative to default (rad), 31 joint velocities (rad/s), 3 root-local gyro values (rad/s), 3 projected-gravity components, then 31 previous raw actions. |
| `frame` | int64 [1] | Advance one frame per 20 ms policy tick. Start at 0 and finish at 599. |
| `actions` | float32 [1, 31] | Raw policy actions in contract joint order. Target radians = default_joint_pos + action_scale × actions. |

The package includes the exact training control profile (named Kp, Kd, limits and offsets). The replay script provides the packing function for a 993-float observation from the existing runtime. At initialization the Mac simulation repeats its first observed frame ten times with zero previous action. Subsequent histories include the previous raw output. The model governs motion through its frame index. The WAV is an optional playback cue.

In a powered OpenIMU trial on Jetson 253, the actor started frame 0 after the standing checkpoint, completed frame 599 and returned to zero-command standing. The controller recorded zero deadline misses, then confirmed all 31 drives disabled from individual feedback. [Powered trial record](../../sim2real/evidence/standing-groove-powered-253.json)

## Mac simulation

From the root of the matching MacMimic checkout, with the OpenSprite simulation dependencies and assets installed:

```sh
./scripts/play_opensprite_groove.sh
./scripts/play_opensprite_groove.sh --headless
```

The first command opens MuJoCo and plays the included WAV once. The reference NPZ is the original 50 Hz groove_0.88 choreography including lookahead. The ONNX contains the reference features needed for the 12-second performance.

The underlying policy used [Ubisoft La Forge LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed). Training data includes [Motion Data by Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the [BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license).
