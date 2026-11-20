"""Unitree G1 (29 DoF), following BeyondMimic's actuator model (as upstream blinkmimic/g1/g1_constants.py).

Joint drives are implicit PD (MuJoCo <position> actuators) with a 10 Hz natural frequency and damping ratio 2 on the
reflected motor armature; the action scale is 0.25 * effort_limit / stiffness.
"""

from __future__ import annotations

import os
import re

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get("SUPERMIMIC_DATA", os.path.join(REPO, "data"))
MJCF_PATH = os.path.join(DATA_DIR, "g1_description", "g1_29dof.xml")
MOTION_DIR = os.path.join(DATA_DIR, "lafan1_g1")

DECIMATION = 4

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

# joint order of the LAFAN1 retarget CSVs (Unitree SDK order) = MJCF order = action order
JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)  # fmt: skip

# BeyondMimic's tracked bodies; the first one (pelvis) is the root/anchor
TRACKED_BODIES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)  # fmt: skip
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")

_M5020 = (25.0, 37.0, ARMATURE_5020)
_M4010 = (5.0, 22.0, ARMATURE_4010)
JOINT_MOTORS = {  # (effort limit, velocity limit (not simulated), armature)
    "hip_pitch": (88.0, 32.0, ARMATURE_7520_14),
    "hip_roll": (139.0, 20.0, ARMATURE_7520_22),
    "hip_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "knee": (139.0, 20.0, ARMATURE_7520_22),
    "ankle_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "ankle_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "waist_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "shoulder_pitch": _M5020,
    "shoulder_roll": _M5020,
    "shoulder_yaw": _M5020,
    "elbow": _M5020,
    "wrist_roll": _M5020,
    "wrist_pitch": _M4010,
    "wrist_yaw": _M4010,
}

DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

# domain randomization (SuperMimic paper G1 setup)
FRICTION_RANGE = (0.3, 1.2)  # robot geoms, once per env; the ground gets the lowest value (MuJoCo takes the pair max)
JOINT_OFFSET = 0.01  # encoder offset half-range (rad), per episode
COM_BODY = "torso_link"
COM_RANGE = ((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))  # once per env
RESET_POS = (0.05, 0.05, 0.01)
RESET_ROT = (0.1, 0.1, 0.2)
RESET_LIN_VEL = (0.5, 0.5, 0.2)
RESET_ANG_VEL = (0.52, 0.52, 0.78)
RESET_JOINT_POS = 0.1
# observation noise half-widths (paper Table A.1)
NOISE = dict(joint_pos=0.01, joint_vel=0.5, ang_vel=0.2, gravity=0.05, ref_ori=0.05)
INIT_ROOT_POS = (0.0, 0.0, 0.76)


def motor_of(name: str) -> tuple[float, float, float]:
    return JOINT_MOTORS[name.replace("left_", "").replace("right_", "").replace("_joint", "")]


def kp(name: str) -> float:
    return motor_of(name)[2] * NATURAL_FREQ**2


def kd(name: str) -> float:
    return 2.0 * DAMPING_RATIO * motor_of(name)[2] * NATURAL_FREQ


def action_scale(name: str) -> float:
    return 0.25 * motor_of(name)[0] / kp(name)


def _match(patterns: dict[str, float], name: str, default: float = 0.0) -> float:
    val = default
    for pat, v in patterns.items():
        if re.fullmatch(pat, name):
            val = v
    return val


def default_joint_pos() -> np.ndarray:
    return np.array([_match(DEFAULT_JOINT_POS, n) for n in JOINT_NAMES], dtype=np.float32)


def action_scales() -> np.ndarray:
    return np.array([action_scale(n) for n in JOINT_NAMES], dtype=np.float32)


def robot_spec() -> mujoco.MjSpec:
    """The G1 MJCF (tools/urdf_to_mjcf.py). Self-collisions off: collision geoms only touch the ground."""
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = 0, 1
    return spec


def load_motion(path: str):
    """LAFAN1 G1 retarget CSV: 30 fps rows of root position, root quaternion (xyzw), 29 joint angles."""
    raw = np.loadtxt(path, delimiter=",")
    p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
    quat = np.concatenate((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)
    return p, quat, j, 30.0
