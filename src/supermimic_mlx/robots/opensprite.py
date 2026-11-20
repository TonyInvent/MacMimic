"""OpenSprite v5, using the named controls and asset from OpenSprite-SuperMimic."""

import os
import sys
from pathlib import Path

import mujoco
import numpy as np

UPSTREAM = Path(
    os.environ.get("OPENSPRITE_REPO", Path(__file__).resolve().parents[3] / "data/OpenSprite-SuperMimic")
).resolve()
sys.path.insert(0, str(UPSTREAM))
from opensprite_tracker import robot as source

JOINT_NAMES = source.BODY_JOINT_NAMES
TRACKED_BODIES = source.TRACKED_BODY_NAMES
FEET = source.FOOT_BODY_NAMES
DECIMATION = source.DECIMATION
MJCF_PATH = str(source.ROBOT_XML)
INIT_ROOT_POS = (0, 0, source.INITIAL_ROOT_HEIGHT)
COM_BODY = "waist_yaw_link"
COM_RANGE = ((-0.00588, 0.00588), (-0.01225, 0.01225), (-0.01102, 0.01102))
FRICTION_RANGE = (0.35, 1.2)
JOINT_OFFSET = 0.01
RESET_POS = (0.05, 0.05, 0.01)
RESET_ROT = (0.1, 0.1, 0.2)
RESET_LIN_VEL = (0.5, 0.5, 0.2)
RESET_ANG_VEL = (0.52, 0.52, 0.78)
RESET_JOINT_POS = 0.1
NOISE = dict(joint_pos=0.01, joint_vel=0.5, ang_vel=0.2, gravity=0.05, ref_ori=0.05)
MODEL = mujoco.MjModel.from_xml_path(MJCF_PATH)


def robot_spec():
    return mujoco.MjSpec.from_file(MJCF_PATH)


def motor_of(name):
    c = source.JOINT_CONTROL[name]
    return c.command_effort_nm, c.urdf_velocity_rad_s, MODEL.dof_armature[MODEL.joint(name).dofadr[0]]


def kp(name):
    return source.JOINT_CONTROL[name].kp


def kd(name):
    return source.JOINT_CONTROL[name].kd


def default_joint_pos():
    return np.array([source.JOINT_CONTROL[n].default_pos_rad for n in JOINT_NAMES], dtype=np.float32)


def action_scales():
    return np.full(len(JOINT_NAMES), 0.25, dtype=np.float32)


def motion_library(bank, kin):
    from ..motion import MotionLib

    with np.load(bank, allow_pickle=False) as data:
        if (
            tuple(data["joint_names"].astype(str)) != JOINT_NAMES
            or str(data["asset_revision"]) != source.ASSET_REVISION
        ):
            raise ValueError("Reference bank and OpenSprite joint/asset contract differ")
        root = list(data["body_names"].astype(str)).index("pelvis_link")
        names = data["motion_names"].astype(str).tolist()
        clips = {}
        for name, start, length in zip(names, data["motion_starts"], data["motion_lengths"], strict=True):
            frames = slice(start, start + length)
            clips[name] = (
                data["body_pos_w"][frames, root],
                data["body_quat_w"][frames, root],
                data["joint_pos"][frames],
                float(data["fps"]),
            )
        motion = MotionLib(names, kin, clips.__getitem__)
        from ..r1_reference import R1_RESET_WEIGHTS, validate_clip_layout

        if names == list(R1_RESET_WEIGHTS):
            import mlx.core as mx
            from ..rotations import T, mv3, quat_to_mat

            validate_clip_layout(names, data["motion_lengths"])
            motion.r1_reference = True
            if float(data["fps"]) != 50 or not np.array_equal(data["motion_starts"], [0, 572]):
                raise ValueError("Expected the R1 bank at its recorded frame rate and clip boundaries")
            body_indices = [data["body_names"].tolist().index(name) for name in TRACKED_BODIES]
            motion.body_p = mx.array(data["body_pos_w"][:, body_indices])
            motion.body_R = quat_to_mat(mx.array(data["body_quat_w"][:, body_indices]))
            motion.body_v = mx.array(data["body_lin_vel_w"][:, body_indices])
            motion.body_w = mx.array(data["body_ang_vel_w"][:, body_indices])
            root_index = TRACKED_BODIES.index("pelvis_link")
            for key, bodies in (("root_p", motion.body_p), ("root_R", motion.body_R),
                                ("root_v", motion.body_v), ("root_w", motion.body_w)):
                value = bodies[:, root_index]
                setattr(motion, key, value)
                motion.np[key] = np.array(value)
            motion.np["q"], motion.np["qd"] = data["joint_pos"], data["joint_vel"]
            motion.q, motion.qd = mx.array(motion.np["q"]), mx.array(motion.np["qd"])
            motion.root_v_local = mv3(T(motion.root_R), motion.root_v)
            motion.root_w_local = mv3(T(motion.root_R), motion.root_w)
            # Probe and learner threads use independent MLX streams.
            mx.eval(motion.body_p, motion.body_R, motion.body_v, motion.body_w,
                    motion.q, motion.qd, motion.root_p, motion.root_R, motion.root_v, motion.root_w,
                    motion.root_v_local, motion.root_w_local)
    return motion
