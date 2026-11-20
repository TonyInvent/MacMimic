"""Count bilateral swing onsets from v5 sole geometry in recorded motion and local simulation."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path.cwd() / "tools"))
from opensprite_control import Controller


def sole_height(sim):
    feet = [sim.model.geom("robot/" + side + "_foot_collision").id for side in ("left", "right")]
    rotation = sim.data.geom_xmat[feet].reshape(2, 3, 3)
    return sim.data.geom_xpos[feet, 2] - (abs(rotation[:, 2]) * sim.model.geom_size[feet]).sum(-1)


def summarize(heights):
    heights = np.asarray(heights)
    in_swing = heights[0] >= .008
    sequence = []
    for row in heights[1:]:
        for side in range(2):
            if not in_swing[side] and row[side] >= .008:
                in_swing[side] = True
                sequence.append(side)
            elif in_swing[side] and row[side] <= .002:
                in_swing[side] = False
    return {
        "duration_s": len(heights) / 50,
        "swing_onsets_left_right": [sequence.count(i) for i in range(2)],
        "same_foot_consecutive_onsets": sum(a == b for a, b in zip(sequence, sequence[1:])),
        "sole_height_p95_mm": (np.percentile(heights, 95, axis=0) * 1000).tolist(),
    }


def policy(sim, speed):
    sim.reset()
    heights = []
    for tick in range(1000):
        _, fallen = sim.step([speed, 0, 0])
        if fallen:
            return {"fallen": True, "elapsed_s": (tick + 1) / 50}
        if tick >= 500:
            heights.append(sole_height(sim))
    return summarize(heights)


if __name__ == "__main__":
    package = Path(sys.argv[1])
    sim = Controller(package)
    original = Path("data/OpenSprite-SuperMimic/outputs/r1-mac/original/walk_stand.npz")
    with np.load(original, allow_pickle=False) as bank:
        root = list(bank["body_names"]).index("pelvis_link")
        reference = {}
        for name, start, length in zip(bank["motion_names"], bank["motion_starts"], bank["motion_lengths"]):
            heights = []
            for f in range(start, start + length):
                sim.data.qpos[:3] = bank["body_pos_w"][f, root]
                sim.data.qpos[3:7] = bank["body_quat_w"][f, root]
                sim.data.qpos[7:] = bank["joint_pos"][f]
                mujoco.mj_forward(sim.model, sim.data)
                heights.append(sole_height(sim))
            reference[str(name)] = summarize(heights)
    result = {"swing_onset_mm": 8, "swing_end_mm": 2, "reference": reference,
              "policy_stand": policy(sim, 0), "policy_walk_040": policy(sim, .4)}
    (package / "gait.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
