"""SuperMimic Appendix G.2 velocity step test in the OpenSprite v5 simulator."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opensprite_control import Controller
from supermimic_mlx.command_ranges import COMMAND_RANGES


def grid():
    low, high = COMMAND_RANGES["paper"]
    axes = [np.linspace(lo, hi, 15) for lo, hi in zip(low, high)]
    for axis in axes:
        axis[np.argmin(np.abs(axis))] = 0.0
    for i, j in combinations(range(3), 2):
        for a, b in product(axes[i], axes[j]):
            if a == b == 0:
                continue
            cmd = np.zeros(3)
            cmd[[i, j]] = a, b
            yield i, j, cmd


def evaluate_batch(task):
    package, entries = task
    sim = Controller(package)
    results = []
    for i, j, cmd in entries:
        trials = []
        for seed in range(4):
            sim.reset()
            # Shared deterministic trial starts across commands and checkpoints.
            # Seed zero is nominal. Other starts use the Xbox varied-start protocol.
            if seed:
                rng = np.random.default_rng(seed)
                sim.data.qpos[7:] += rng.uniform(-.025, .025, len(sim.q0))
                sim.data.qvel[:2] = rng.uniform(-.04, .04, 2)
                sim.data.qvel[3:6] = rng.uniform(-.04, .04, 3)
                mujoco.mj_forward(sim.model, sim.data)
                sim.history[:] = sim.observation()
            samples = []
            for tick in range(250):
                measured, fallen = sim.step(cmd)
                if fallen:
                    break
                if tick >= 150:
                    samples.append(measured)
            error = None if fallen else float(np.linalg.norm(np.mean(samples, 0) - cmd) / np.linalg.norm(cmd))
            trials.append({"seed": seed, "fallen": bool(fallen), "seconds": (tick + 1) / 50, "relative_error": error})
        errors = [t["relative_error"] for t in trials if not t["fallen"]]
        results.append({"plane": [i, j], "command": cmd.tolist(), "trials": trials,
                        "median_relative_error": float(np.median(errors)) if errors else None})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    entries = list(grid())
    groups = [(args.package, entries[start:start + 16]) for start in range(0, len(entries), 16)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for batch in pool.map(evaluate_batch, groups) for row in batch]
    errors = [r["median_relative_error"] for r in rows if r["median_relative_error"] is not None]
    summary = {"step_responses": len(rows), "trials": len(rows) * 4,
               "fallen_trials": sum(t["fallen"] for r in rows for t in r["trials"]),
               "all_four_fell": len(rows) - len(errors),
               "median_relative_error_among_surviving_commands": float(np.median(errors)) if errors else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": rows}, allow_nan=False) + "\n")
    print(json.dumps(summary))
