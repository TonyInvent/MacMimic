"""Left-right (sagittal-plane) mirror of states, actions and sim context, for world-model data augmentation.

The mirror is the reflection M = diag(1, -1, 1) of the world y axis: positions and linear velocities map to M x,
rotations to M R M, angular velocities (pseudo-vectors) to -M w, and joints to their left/right partner with the sign
flipped for roll and yaw joints.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .robots import g1

_FLIP_TYPES = ("roll", "yaw")


class Mirror:
    def __init__(self, xi_slices: dict):
        names = list(g1.JOINT_NAMES)
        perm, sign = [], []
        for n in names:
            partner = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
            perm.append(names.index(partner))
            sign.append(-1.0 if any(f"_{t}_" in n for t in _FLIP_TYPES) else 1.0)
        self.perm = mx.array(perm)
        self.sign = mx.array(sign, dtype=mx.float32)
        self.m = mx.array([1.0, -1.0, 1.0])
        self.M = mx.array(np.diag([1.0, -1.0, 1.0]).astype(np.float32))
        assert set(xi_slices) == {"friction", "joint_offset", "com"}, xi_slices
        self.xi_slices = xi_slices
        xs = np.ones(sum(s.stop - s.start for s in xi_slices.values()), dtype=np.float32)
        xs[xi_slices["com"].start + 1] = -1.0  # lateral COM offset
        xp = np.arange(len(xs))
        jo = xi_slices["joint_offset"]
        xp[jo] = jo.start + np.array(perm)
        self.xi_perm = mx.array(xp)
        xs2 = np.array(xs)
        xs2[jo] = np.array(sign)
        self.xi_sign = mx.array(xs2)

    def joints(self, x: mx.array) -> mx.array:
        return x[..., self.perm] * self.sign

    def state(self, S):
        p, R, v, w, q, qd = S
        M = self.M
        return (p * self.m, M @ R @ M, v * self.m, -(w * self.m), self.joints(q), self.joints(qd))

    def xi(self, xi: mx.array) -> mx.array:
        return xi[..., self.xi_perm] * self.xi_sign
