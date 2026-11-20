"""Measure standing posture, foot clearance and joystick response on the local v5 simulator."""
import argparse
import json
from pathlib import Path

import numpy as np

from opensprite_control import Controller, drive_sequence


def steady_state(sim, target):
    sim.reset()
    feet = [sim.model.geom("robot/" + side + "_foot_collision").id for side in ("left", "right")]
    knees = [sim.model.joint("robot/" + side + "_knee_joint").qposadr[0] for side in ("left", "right")]
    log = []
    for tick in range(1000):
        measured, fallen = sim.step(np.asarray(target, dtype=np.float32))
        if fallen:
            return {"fallen": True, "seconds": (tick + 1) / 50}
        if tick >= 500:
            rot = sim.data.geom_xmat[feet].reshape(2, 3, 3)
            sole_z = sim.data.geom_xpos[feet, 2] - (np.abs(rot[:, 2, :]) * sim.model.geom_size[feet]).sum(-1)
            log.append(np.r_[measured, np.rad2deg(sim.data.qpos[knees]), sole_z * 1000])
    values = np.array(log)
    return {
        "fallen": False,
        "seconds": 20,
        "velocity_mean": values[:, :3].mean(0).tolist(),
        "knee_median_deg": np.median(values[:, 3:5], axis=0).tolist(),
        "sole_height_p95_mm": np.percentile(values[:, 5:7], 95, axis=0).tolist(),
        "sole_height_max_mm": values[:, 5:7].max(0).tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sim = Controller(args.package)
    result = {"joystick": drive_sequence(sim)}
    for name, command in (("stand", [0, 0, 0]), ("forward_025", [.25, 0, 0]), ("forward_050", [.5, 0, 0])):
        result[name] = steady_state(sim, command)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
