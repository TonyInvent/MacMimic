"""Analytic MDP heads shared by the simulator and the world model (SuperMimic paper Appendix A).

State tuple ``S = (p, R, v, w, q, qd)``: root position, rotation matrix, world linear/angular velocity of the root
link frame, joint positions and velocities.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .kinematics import Kinematics
from .motion import MotionLib
from .robots import g1
from .rotations import mat_to_6d, mat_to_quat, mm3, mv3, quat_to_mat, rot_angle_sq, yaw_mat, T
from .scene import CONTROL_DT


def state_dim(J: int) -> int:
    return 3 + 4 + 3 + 3 + J + J


def obs_dim(J: int) -> int:
    return J + J + 3 + 3 + J + 6


def inv_dim(J: int) -> int:
    return 1 + 6 + 3 + 3 + J + J


CONTACT_Z = 0.045  # ankle-roll link origin height below which a foot counts as in contact
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


def flatten_state(S) -> mx.array:
    p, R, v, w, q, qd = S
    return mx.concatenate((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: mx.array):
    J = (x.shape[-1] - 13) // 2
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + J], x[..., 13 + J :]


def invariant_features(S) -> mx.array:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = T(R)
    return mx.concatenate((p[..., 2:3], mat_to_6d(mm3(T(H), R)), mv3(Rt, v), mv3(Rt, w), q, qd), -1)


class MDP:
    def __init__(self, kin: Kinematics, motion: MotionLib, xi_dim: int, cfg: RewardCfg = RewardCfg()):
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.default_q = mx.array(g1.default_joint_pos())
        J = self.default_q.shape[0]
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        self.num_j, self.obs_dim, self.xi_dim = J, obs_dim(J), xi_dim
        self.critic_extra = J + len(kin.tracked_idx) * 3 + 3 + 6 + xi_dim
        n = g1.NOISE
        self.obs_noise = mx.array(np.concatenate((np.full(J, n["joint_pos"]), np.full(J, n["joint_vel"]), np.full(3, n["ang_vel"]),
                                                  np.full(3, n["gravity"]), np.zeros(J), np.full(6, n["ref_ori"]))).astype(np.float32))  # fmt: skip
        self.feet: list[int] = [list(g1.TRACKED_BODIES).index(b) for b in g1.FEET]

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f) -> mx.array:
        """Noise-free proprioceptive observation (+ reference-relative root orientation)."""
        p, R, v, w, q, qd = S
        Rt = T(R)
        dq0 = xi[..., 1 : 1 + q.shape[-1]]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        rel = mat_to_6d(mm3(Rt, self.motion.root_R[f]))
        return mx.concatenate((q - self.default_q - dq0, qd, mv3(Rt, w), g_b, a_prev, rel), -1)

    def add_noise(self, o: mx.array, key=None) -> mx.array:
        return o + (2 * mx.random.uniform(shape=o.shape, key=key) - 1) * self.obs_noise

    # ---------------- reward / termination ----------------
    def _ref_bodies(self, p, R, f):
        """Reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic."""
        m = self.motion
        rp, rR = m.root_p[f], m.root_R[f]
        dR = yaw_mat(mm3(R, T(rR)))
        anchor = mx.concatenate((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = mx.expand_dims(anchor, -2) + mv3(mx.expand_dims(dR, -3), m.body_p[f] - mx.expand_dims(rp, -2))
        return ref_bp, dR, rp, rR

    def reward(self, S, f, a, a_prev, terms: bool = False, w_contact: float = 0.0, ref_contact=None):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f. ``w_contact`` > 0 adds the
        InstantTrack contact term (training only; the evaluation reward is the paper's)."""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        ref_bp, dR, rp, rR = self._ref_bodies(p, R, f)
        ref_bR = mm3(mx.expand_dims(dR, -3), m.body_R[f])
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_track = (
            mx.exp(-e_pos / c.body_pos_std**2)
            + mx.exp(-e_ori / c.body_ori_std**2)
            + mx.exp(-e_lv / c.body_linvel_std**2)
            + mx.exp(-e_av / c.body_angvel_std**2)
            + 0.5 * mx.exp(-e_rp / c.root_pos_std**2)
            + 0.5 * mx.exp(-e_ro / c.root_ori_std**2)
        )
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (mx.maximum(q - self.soft_hi, 0) + mx.maximum(self.soft_lo - q, 0)).sum(-1)
        r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim) * CONTROL_DT
        if w_contact > 0:
            feet = mx.array(self.feet)
            fz, rz = bp[..., feet, 2], m.body_p[f][..., feet, 2]
            c_sim = mx.sigmoid((CONTACT_Z - fz) / 0.005)
            c_ref = m.ref_contact[f] if m.ref_contact is not None else (rz < CONTACT_Z).astype(mx.float32)
            slip = (bv[..., feet, :2] ** 2).sum(-1)
            r_contact = (1 - (c_sim - c_ref) ** 2) + c_ref * mx.exp(-slip / 0.3**2)
            r = r + w_contact * r_contact.mean(-1) * CONTROL_DT
        if terms:
            return r, {"track": r_track * CONTROL_DT, "e_body_pos": e_pos, "e_body_ori": e_ori, "e_root_pos": e_rp,
                       "action_rate": r_rate, "joint_limit": r_lim}  # fmt: skip
        return r

    def termination(self, S, f) -> mx.array:
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = mx.abs(rp[..., 2] - p[..., 2])
        e_tilt = mx.abs(rR[..., 2, 2] - R[..., 2, 2])
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> mx.array:
        """Differentiable termination probability for imagined rollouts."""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = mx.abs(rp[..., 2] - p[..., 2])
        e_tilt = mx.abs(rR[..., 2, 2] - R[..., 2, 2])
        alive = (1 - mx.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - mx.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(self, S, f, xi_n) -> mx.array:
        """Critic input: invariant state, robot-vs-reference errors and the (normalized) sim context."""
        m = self.motion
        p, R, v, w, q, qd = S
        Ht = T(yaw_mat(R))
        bp = self.kin.forward(p, R, v, w, q, qd)[0]
        ref_bp, _, rp, rR = self._ref_bodies(p, R, f)
        e_body = mv3(mx.expand_dims(Ht, -3), ref_bp - bp)
        e_body = e_body.reshape(e_body.shape[:-2] + (-1,))
        e_root = mv3(Ht, rp - p)
        rel = mat_to_6d(mm3(T(R), rR))
        return mx.concatenate((invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi_n), -1)
