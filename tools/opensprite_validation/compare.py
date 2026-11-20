"""Rank local R1 exports on two reference clips and identical Xbox simulation trials."""
import json
from pathlib import Path
import os

import numpy as np

root = Path(__file__).resolve().parents[2]
run = Path(os.environ.get("OPENSPRITE_RUN", root / "logs/opensprite_r1_mac_s49")).resolve()
baseline = root / "docs/opensprite-r1/evaluation/official_baseline.json"


def read(path):
    return json.loads(path.read_text())


def compare(package, reference):
    motion_log = (package / "motion_eval.log").read_text()
    motion = json.loads(motion_log[motion_log.index('{\n "norm_return"'):])
    evaluation = read(package / "r1_eval.json")
    sequences = list(read(package / "xbox_varied_starts.json").values())
    analog = read(package / "xbox_analog.json")
    steady = [evaluation[key] for key in ("stand", "forward_025", "forward_050")]
    fall_count = sum(any(row["fallen"] for row in seq) for seq in sequences)
    fall_count += sum(row["fallen"] for row in steady)
    correct_turns = 0
    joystick_errors = []
    for sequence in sequences:
        if any(row["fallen"] for row in sequence):
            continue
        phases = {row["phase"]: row["measured"] for row in sequence}
        correct_turns += int(phases["walking left"][2] > 0 and phases["walking right"][2] < 0)
        errors = np.array([np.subtract(row["measured"], row["command"]) for row in sequence[2:]])
        joystick_errors.append((np.linalg.norm(errors[:, :2], axis=1) / .4 + np.abs(errors[:, 2]) / .2).mean() / 2)
    result = {
        "iteration": int(package.name.split("_")[1]),
        "failed_motion_clips": motion["n_failed"],
        "motion_return": motion["norm_return"],
        "falls_in_seven_cases": fall_count,
        "falls_in_eight_stick_settings": sum(row["fallen"] for row in analog),
        "both_turn_directions_in_four_cases": correct_turns,
    }
    if not fall_count:
        knees = np.subtract(steady[0]["knee_median_deg"], reference["stand"]["knee_median_deg"])
        height = np.subtract(steady[2]["sole_height_p95_mm"], reference["forward_050"]["sole_height_p95_mm"])
        joystick = np.mean(joystick_errors)
        posture = np.mean(np.abs(knees)) / 20
        clearance = np.mean(np.abs(height)) / 25
        result.update(joystick=joystick, standing_knees=posture, moving_clearance=clearance,
                      score=.5 * joystick + .25 * (posture + clearance),
                      speed_at_050=steady[2]["velocity_mean"][0])
    return result


def rank(packages):
    reference = read(baseline)
    results = [compare(p, reference) for p in packages]
    results.sort(key=lambda r: (r["failed_motion_clips"], r["falls_in_seven_cases"] + r["falls_in_eight_stick_settings"],
                                -r["both_turn_directions_in_four_cases"], r.get("score", float("inf"))))
    return results


if __name__ == "__main__":
    packages = [p for p in sorted(run.glob("velocity_*")) if (p / "xbox_analog.json").is_file()]
    print(json.dumps(rank(packages), indent=2))
