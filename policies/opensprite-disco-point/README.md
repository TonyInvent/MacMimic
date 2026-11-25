# OpenSprite Disco Point · 100 BPM

A 32.8-second standing dance for OpenSprite v5 with alternating diagonal points, horizontal extensions and an overhead finish. The fixed-motion ONNX uses checkpoint 100,000 (seed 49). The latents for this choreography are embedded in the model. [Watch the simulated rollout with the beat](disco_point_preview.mp4)

## Replay the released model

From the extracted package directory on a Jetson with NumPy and CPU ONNX Runtime:

```sh
sha256sum -c disco_point_SHA256SUMS
python3 disco_point_replay.py
```

The replay checks all 1,640 simulated feedback histories against saved MLX actions. It uses no hardware IO. On a Mac with the repository's OpenSprite environment and assets installed, run the following from the matching repository checkout:

```sh
./scripts/play_opensprite_disco_point.sh
./scripts/play_opensprite_disco_point.sh --headless
```

## Lab handoff

Use `disco_point_100k.onnx` with `disco_point_contract.json`. At 50 Hz supply float32 `history[1,10,99]` and int64 `frame[1]` from 0 through 1,639 inclusive. The 31 raw actions are in contract order. Each history frame contains relative joint angles, joint velocities, root-local gyro, projected gravity and the previous raw action, oldest frame first. The included training control profile provides the named gains and limits. The WAV is a playback cue. The 50 Hz reference NPZ includes extra lookahead for the motion encoder.

The previously powered 12-second standing groove uses a separate model and evidence. At revision 3b9bca9 its native actor and verifier bind 600 frames and the original fingerprint set. For Disco Point, use a dedicated package binding and update the native frame count, replay length, motion window and end-of-motion handling together. Run a full native replay and target-specific commissioning before a powered trial. Preserve the operator stop, existing protections and confirmed drive-disable path. The Mac results below are for this model. [Current native actor](../../runtime/native/src/standing_groove.hpp) · [Original verifier](../../runtime/tools/verify_standing_groove.py)

## Simulation results

One nominal MuJoCo rollout completed 1,640 frames upright. Maximum heading change from the initial pelvis heading was 32.6°. Left and right foot collision-box centers moved up to 7.77 and 7.21 cm horizontally from their initial positions. The maximum raw action difference against MLX on the replay fixture was 0.00000620. [Measurements](disco_point_simulation.json)

Training data includes [Motion Data by Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the [BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license). Official motion sources are [Ubisoft La Forge LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed).
