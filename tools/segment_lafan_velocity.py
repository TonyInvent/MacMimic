"""Keep continuous feasible windows of complete LAFAN1 walking/running solves.

Use the OpenSprite retarget environment. Record every window on the resampled
50 Hz source timeline. Never join trajectories across an excluded frame.
"""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from retarget_lafan_velocity import OUT, SOURCE, UPSTREAM
import sys
sys.path.insert(0, str(UPSTREAM))
from opensprite_tracker.robot import ASSET_REVISION
from retarget.bvh_tools.bvh_adapters import inspect_bvh
from retarget.bvh_tools.motion_postprocess import process_motion
from retarget.processing import correct_small_ground_residual, refine_angles, robot_contract


def usable(model, poses):
    data = mj.MjData(model)
    ids = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    kinds, sizes = model.geom_type[ids], model.geom_size[ids]
    boxes = kinds == mj.mjtGeom.mjGEOM_BOX
    caps = kinds == mj.mjtGeom.mjGEOM_CAPSULE
    if not np.all(boxes | caps | (kinds == mj.mjtGeom.mjGEOM_SPHERE)):
        raise ValueError("Unsupported collision shape")
    valid = np.empty(len(poses), bool)
    for t, pose in enumerate(poses):
        data.qpos[:] = pose
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        mj.mj_collision(model, data)
        z = np.abs(data.geom_xmat[ids].reshape(-1,3,3)[:,2])
        extent = sizes[:,0].copy()
        extent[caps] += z[caps,2] * sizes[caps,1]
        extent[boxes] = (z[boxes] * sizes[boxes]).sum(1)
        floor = (data.geom_xpos[ids,2] - extent).min()
        valid[t] = floor >= -.0295 and all(c.dist >= -.01 for c in data.contact)
    return valid


def segment(name):
    info = inspect_bvh(SOURCE / (name + ".bvh"))
    model, joints, limits, speeds = robot_contract()
    with np.load(OUT / name / "native.npz", allow_pickle=True) as file:
        if (str(file["robot_name"]) != "opensprite0825" or str(file["source_format"]) != "nr_fbx_bvh"
                or file["robot_joint_names"].astype(str).tolist() != joints
                or not np.array_equal(file["frame_ids"], np.arange(info["frames"]))):
            raise ValueError(f"Invalid native solve: {name}")
        qpos = file["qpos"].astype(float)
    if (not np.isfinite(qpos).all() or np.max(np.abs(np.linalg.norm(qpos[:,3:7],axis=1)-1)) > 1e-6
            or np.any(qpos[:,7:] < limits[:,0]-1e-6) or np.any(qpos[:,7:] > limits[:,1]+1e-6)):
        raise ValueError(f"Invalid native pose: {name}")
    qpos, _, _ = process_motion(qpos, info["fps"], 50., model=model)
    qpos[:,7:], _ = refine_angles(qpos[:,7:], limits, speeds)
    edges = np.flatnonzero(np.diff(np.r_[False, usable(model, qpos), False]))
    target = OUT / name / "segments"
    target.mkdir(exist_ok=True)
    windows = []
    for start, end in edges.reshape(-1, 2):
        if end-start < 50:
            continue
        poses, _ = correct_small_ground_residual(model, qpos[start:end])
        path = target / f"{name}_{start}_{end}.npz"
        np.savez_compressed(path, qpos=poses, fps=50., joint_names=joints, asset_revision=ASSET_REVISION)
        windows.append(dict(file=path.name,start=int(start),end=int(end)))
    manifest = dict(source=name, frames=len(qpos), fps=50, windows=windows)
    tmp = target / "windows.tmp"
    tmp.write_text(json.dumps(manifest, indent=2)+"\n")
    tmp.replace(target / "windows.json")
    print(name,len(windows),sum(w["end"]-w["start"] for w in windows),len(qpos),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    allowed = {Path(s["file"]).stem for s in json.loads((SOURCE / "paper_locomotion_sources.json").read_text())["sources"]}
    for name in args.names:
        if name not in allowed:
            raise ValueError(f"Unlisted LAFAN1 source {name}")
        segment(name)
