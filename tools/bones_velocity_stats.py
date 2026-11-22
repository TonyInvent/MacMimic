"""Check candidate G1 recordings and measure planar commands in the source robot's heading frame."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from supermimic_mlx.robots.g1 import JOINT_NAMES


def read_g1(path, frames):
    # BONES-SEED G1 files have named columns, centimeter translations, and degree rotations.
    with path.open() as f:
        names = f.readline().strip().split(",")
        expected = (["Frame"] + ["root_translate" + axis for axis in "XYZ"]
                    + ["root_rotate" + axis for axis in "XYZ"] + [joint + "_dof" for joint in JOINT_NAMES])
        if names != expected:
            raise ValueError(f"Unknown G1 column layout: {path}")
        raw = np.loadtxt(f, delimiter=",", dtype=np.float64, ndmin=2)
    if (raw.shape != (frames, len(expected)) or not np.isfinite(raw).all()
            or not np.array_equal(raw[:, 0], np.arange(frames))):
        raise ValueError(f"Invalid G1 frames: {path}")
    # NVlabs/ProtoMotions uses extrinsic xyz in the G1 CSV importer.
    return raw[:, 1:4] / 100, Rotation.from_euler("xyz", raw[:, 4:7], degrees=True), np.deg2rad(raw[:, 7:])


def velocity_stats(path, frames):
    position, orientation, _ = read_g1(path, frames)
    target = np.arange((frames - 1) * 50 // 120 + 1, dtype=np.int64) * 120 / 50
    if len(target) < 2:
        raise ValueError(f"Too few frames for a velocity: {path}")
    floor = target.astype(np.int64)
    upper = np.minimum(floor + 1, frames - 1)
    p = position[floor] + (target - floor)[:, None] * (position[upper] - position[floor])
    rotation = Slerp(np.arange(frames), orientation)(target)
    R = rotation.as_matrix()
    heading = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    vx, vy = np.gradient(p[:, :2], .02, axis=0).T
    # Angular velocity projected onto the world z-axis as used by the paper.
    w = np.empty(len(p))
    w[1:-1] = (rotation[2:] * rotation[:-2].inv()).as_rotvec()[:, 2] / .04
    w[[0, -1]] = (rotation[[1, -1]] * rotation[[0, -2]].inv()).as_rotvec()[:, 2] / .02
    cmd = np.column_stack((np.cos(heading)*vx + np.sin(heading)*vy,
                           -np.sin(heading)*vx + np.cos(heading)*vy, w))
    padded = np.pad(cmd, ((12, 12), (0, 0)), mode="edge")
    cmd = np.lib.stride_tricks.sliding_window_view(padded, 25, axis=0).mean(-1)
    return np.round(np.quantile(cmd, [.05, .5, .95], axis=0).T.ravel(), 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/bones-seed"))
    args = parser.parse_args()
    dest = args.data / "velocity_source_stats.csv"
    tmp = dest.with_suffix(".tmp")
    with (args.data / "velocity_candidates.csv").open(newline="") as manifest, tmp.open("w", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(["move_g1_path", "source_frames"] + [f"{v}_p{q}" for v in ("vx", "vy", "wz") for q in (5, 50, 95)])
        count = 0
        for row in csv.DictReader(manifest):
            name = row["move_g1_path"]
            frames = int(row["move_duration_frames"])
            writer.writerow([name, frames, *velocity_stats(args.data / "velocity_sources" / name, frames)])
            count += 1
    tmp.replace(dest)
    print(f"Validated and measured {count} candidate G1 recordings", flush=True)
