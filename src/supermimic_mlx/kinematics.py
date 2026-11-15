"""Differentiable forward kinematics of the robot in MLX, built from its MJCF.

The same FK computes tracked-body poses and velocities for the reference motion, for simulator states and for
world-model predictions, so the analytic reward head matches the reward the simulator transitions are labeled with.
"""

from __future__ import annotations

import mlx.core as mx
import mujoco
import numpy as np

from .robots import g1
from .rotations import axis_angle_to_mat, mm3, mv3, quat_to_mat


def cross(a: mx.array, b: mx.array) -> mx.array:
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return mx.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), -1)


class Kinematics:
    def __init__(self, mjcf_path: str = g1.MJCF_PATH, joint_names=g1.JOINT_NAMES, tracked=g1.TRACKED_BODIES):
        m = mujoco.MjModel.from_xml_path(mjcf_path)
        jname_to_idx = {n: i for i, n in enumerate(joint_names)}
        bodies, parent, joint_idx, axes, offs_p, offs_q = [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):  # moving bodies in tree order (parents first)
            if m.body_jntnum[b] == 0:
                continue
            jid = m.body_jntadr[b]
            mj_to_local[b] = len(bodies)
            bodies.append(m.body(b).name)
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1)
                joint_idx.append(-1)
            else:
                pb = m.body_parentid[b]
                if pb not in mj_to_local:
                    raise RuntimeError(f"fixed parent for {m.body(b).name}")
                parent.append(mj_to_local[pb])
                joint_idx.append(jname_to_idx[m.joint(jid).name])
            axes.append(m.jnt_axis[jid].copy())
            offs_p.append(m.body_pos[b].copy())
            offs_q.append(m.body_quat[b].copy())
        self.body_names, self.parent, self.joint_idx = bodies, parent, joint_idx
        self.axes = [mx.array(a, dtype=mx.float32) for a in axes]
        self.off_p = [mx.array(p, dtype=mx.float32) for p in offs_p]
        self.off_R = [quat_to_mat(mx.array(q, dtype=mx.float32)) for q in offs_q]
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        lo, hi = np.zeros(len(joint_names)), np.zeros(len(joint_names))
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo_np, self.q_hi_np = lo.astype(np.float32), hi.astype(np.float32)
        self.q_lo, self.q_hi = mx.array(self.q_lo_np), mx.array(self.q_hi_np)
        # only the chains that lead to tracked bodies are needed
        need = set()
        for i in self.tracked_idx:
            while i >= 0 and i not in need:
                need.add(i)
                i = parent[i]
        self.needed = sorted(need)
        mx.eval(self.axes, self.off_p, self.off_R, self.q_lo, self.q_hi)  # materialized: usable from any thread's stream

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions (..., nb, 3), rotations (..., nb, 3, 3), link-origin linear and angular
        velocities (..., nb, 3), from the root state (p, R, v, w: world frame) and joints (q, qd)."""
        n = self.num_bodies
        P, Rs, V, W = [None] * n, [None] * n, [None] * n, [None] * n
        for i in range(n) if all_bodies else self.needed:
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + mv3(Rp, self.off_p[i])
            R_off = mm3(Rp, self.off_R[i])
            Rs[i] = mm3(R_off, axis_angle_to_mat(self.axes[i], q[..., j]))
            W[i] = W[par] + mv3(R_off, self.axes[i]) * qd[..., j : j + 1]
            V[i] = V[par] + cross(W[par], pi - P[par])
            P[i] = pi
        idx = range(n) if all_bodies else self.tracked_idx
        return (
            mx.stack([P[i] for i in idx], -2),
            mx.stack([Rs[i] for i in idx], -3),
            mx.stack([V[i] for i in idx], -2),
            mx.stack([W[i] for i in idx], -2),
        )
