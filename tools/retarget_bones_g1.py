"""Solve a BONES-SEED G1 clip on the OpenSprite v5 model for local reference review.

Run in the OpenSprite retarget environment with PYTHONPATH=src:tools. Outputs are
individual review candidates. The output includes the source path and fit diagnostics.
"""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from bones_velocity_stats import read_g1
from supermimic_mlx.robots import g1, opensprite
from retarget.processing import correct_small_ground_residual, geometry_quality, refine_angles, robot_contract


def resample(path, frames):
    pos, rot, joints = read_g1(path, frames)
    t = np.arange((frames - 1) * 50 // 120 + 1) * 120 / 50
    i = t.astype(int)
    a = (t - i)[:, None]
    j = np.minimum(i + 1, frames - 1)
    quat = Slerp(np.arange(frames), rot)(t).as_quat()[:, [3, 0, 1, 2]]
    return np.column_stack((pos[i] * (1-a) + pos[j] * a, quat, joints[i] * (1-a) + joints[j] * a))


def body_positions(model, data, names):
    return np.stack([data.xpos[model.body(name).id] for name in names])


def retarget(path, frames, output):
    poses = resample(path, frames)
    src = mj.MjModel.from_xml_path(g1.MJCF_PATH)
    dst, joints, limits, speeds = robot_contract()
    if [src.joint(i).name for i in range(1, src.njnt)] != list(g1.JOINT_NAMES):
        raise ValueError("G1 joint order differs from the BONES-SEED CSV")
    sd, dd = mj.MjData(src), mj.MjData(dst)
    mj.mj_forward(src, sd)
    mj.mj_forward(dst, dd)
    feet = [s+"_ankle_roll_link" for s in ("left","right")]
    knees = [s+"_knee_link" for s in ("left","right")]
    shoulders = [s+"_shoulder_roll_link" for s in ("left","right")]
    elbows = [s+"_elbow_link" for s in ("left","right")]
    # Each hand chain ends with different wrist axes on the two robots.
    hands_s = [s+"_wrist_yaw_link" for s in ("left","right")]
    hands_d = [s+"_wrist_roll_link" for s in ("left","right")]
    snames = feet+knees+shoulders+elbows+hands_s
    dnames = feet+knees+shoulders+elbows+hands_d
    s0 = body_positions(src, sd, snames) - sd.qpos[:3]
    d0 = body_positions(dst, dd, dnames) - dd.qpos[:3]
    sid = np.array([src.body(n).id for n in snames])
    did = np.array([dst.body(n).id for n in dnames])
    so = np.array([*sid[:2],src.body("torso_link").id])
    do = np.array([*did[:2],dst.body("waist_yaw_link").id])
    sr = sd.xmat[so].reshape(-1,3,3).copy()
    dr = dd.xmat[do].reshape(-1,3,3).copy()
    align = sr.transpose(0,2,1) @ dr
    # Scale stride by hip-to-ankle height. The pelvis origins differ between robots.
    hip = "left_hip_pitch_link"
    scale = (dd.xpos[dst.body(hip).id,2]-dd.xpos[did[0],2]) / (sd.xpos[src.body(hip).id,2]-sd.xpos[sid[0],2])
    sf = [np.flatnonzero((src.geom_bodyid == b) & (src.geom_contype != 0)) for b in sid[:2]]
    df = [dst.geom(s+"_foot_collision").id for s in ("left","right")]
    def source_ground():
        return np.array([(sd.geom_xpos[ids,2]-src.geom_size[ids,0]).min() for ids in sf])
    src_stand = -source_ground().mean()
    dst_stand = dd.qpos[2] - np.mean([dd.geom_xpos[b,2] - np.abs(dd.geom_xmat[b].reshape(3,3)[2]) @ dst.geom_size[b] for b in df])
    arm_len = [np.linalg.norm(d0[6:8]-d0[4:6],axis=1),np.linalg.norm(d0[8:10]-d0[6:8],axis=1)]
    # Fit a small vertical pelvis offset along with the 31 hinges. This lets a
    # shorter reachable leg meet the same ground contact while preserving stride.
    columns = np.r_[2, np.arange(6, dst.nv)]
    axes = np.arange(len(columns))
    jac, jr = np.empty((3,dst.nv)), np.empty((3,dst.nv))
    posed_ids = [0,1,2,3,6,7,8,9]
    weights = [1.,1.,.35,.35,.45,.45,.6,.6]
    # Orientation residuals use radians, positional residuals meters.
    ang_weights = np.array([.08,.08,.06])
    history, foot_errors = [], []
    for frame, pose in enumerate(poses):
        sd.qpos[:] = pose
        mj.mj_kinematics(src, sd)
        root_R = sd.xmat[src.body("pelvis").id].reshape(3,3)
        dd.qpos[:2] = (pose[:2]-poses[0,:2])*scale
        root_z = dst_stand + scale*(pose[2]-src_stand)
        dd.qpos[2] = root_z
        dd.qpos[3:7] = pose[3:7]
        if frame == 0:
            for side in ("left", "right"):
                knee = side + "_knee_joint"
                dd.qpos[dst.joint(knee).qposadr] = pose[src.joint(knee).qposadr]
                dd.qpos[dst.joint(side + "_elbow_joint").qposadr] = -.6
        sp = sd.xpos[sid]
        targets = dd.qpos[:3] + scale*(sp-pose[:3]) + (d0-scale*s0) @ root_R.T
        for child, parent, lengths in ((slice(6,8),slice(4,6),arm_len[0]),(slice(8,10),slice(6,8),arm_len[1])):
            vec = sp[child] - sp[parent]
            targets[child] = targets[parent] + vec * (lengths/np.linalg.norm(vec,axis=1))[:,None]
        rotations = sd.xmat[so].reshape(-1,3,3) @ align
        ground = np.maximum(0, source_ground()) * scale
        for k, geom in enumerate(df):
            R = rotations[k]
            # The box has its own local orientation and center relative to the ankle.
            box_R = Rotation.from_quat(dst.geom_quat[geom][[1,2,3,0]]).as_matrix()
            sole_z = R[2] @ dst.geom_pos[geom] - np.abs((R @ box_R)[2]) @ dst.geom_size[geom]
            targets[k,2] = ground[k] - sole_z
        prev = dd.qpos[7:].copy()
        for _ in range(32 if frame==0 else 5):
            mj.mj_kinematics(dst, dd)
            mj.mj_comPos(dst, dd)
            A, residual = [], []
            for idx, weight in zip(posed_ids,weights):
                mj.mj_jacBody(dst,dd,jac,None,did[idx])
                A.append(weight*jac[:,columns])
                residual.append(weight*(targets[idx]-dd.xpos[did[idx]]))
            current = dd.xmat[do].reshape(-1,3,3)
            delta = Rotation.from_matrix(rotations @ current.transpose(0,2,1)).as_rotvec()
            for body, angle, weight in zip(do,delta,ang_weights):
                mj.mj_jacBody(dst,dd,None,jr,body)
                A.append(weight*jr[:,columns])
                residual.append(weight*angle)
            A = np.concatenate(A)
            lhs = A.T @ A
            lhs[axes,axes] += .0004
            lhs[0,0] += .15
            rhs = A.T @ np.concatenate(residual)
            rhs[0] += .15*(root_z-dd.qpos[2])
            rhs[1:] += .00005*(prev-dd.qpos[7:]) - .00001*dd.qpos[7:]
            dq = np.linalg.solve(lhs,rhs)
            q = dd.qpos[7:]
            locked = ((q <= limits[:,0]+1e-6) & (dq[1:] < 0)) | ((q >= limits[:,1]-1e-6) & (dq[1:] > 0))
            if locked.any():
                free = np.r_[True, ~locked]
                dq[:] = 0
                dq[free] = np.linalg.solve(lhs[np.ix_(free,free)],rhs[free])
            dd.qpos[2] += np.clip(dq[0],-.015,.015)
            dd.qpos[7:] = np.clip(q + np.clip(dq[1:],-.16,.16), *limits.T)
        mj.mj_kinematics(dst,dd)
        foot_errors.append(np.linalg.norm(targets[:2]-dd.xpos[did[:2]], axis=1))
        history.append(dd.qpos.copy())
    qpos = np.stack(history)
    qpos[:,7:], refinement = refine_angles(qpos[:,7:],limits,speeds)
    qpos, correction = correct_small_ground_residual(dst,qpos)
    quality = geometry_quality(dst,qpos)
    output.mkdir(exist_ok=True,parents=True)
    file = output / (path.stem+'.npz')
    tmp = file.with_suffix(".npz.tmp")
    with tmp.open("wb") as dest:
        np.savez_compressed(dest, qpos=qpos, fps=50., joint_names=joints,
                            source=str(path), asset_revision=opensprite.source.ASSET_REVISION)
    tmp.replace(file)
    report = {"source":str(path),"output_frames":len(qpos),"leg_scale":float(scale),
              "ankle_target_error_p95_m_before_refine":float(np.quantile(foot_errors,.95)),
              "ground_correction":correction,"refinement":refinement,"quality":quality}
    tmp = file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report,indent=2)+'\n')
    tmp.replace(file.with_suffix(".json"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv",type=Path)
    parser.add_argument("--frames",type=int,required=True,help="Source frame count from the release manifest")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(retarget(args.csv,args.frames,args.output),indent=2))
