"""NumPy rotation helpers for the simulator's CPU path (resets, state readout); same conventions as rotations.py."""

from __future__ import annotations

import numpy as np


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    )  # fmt: skip
    return m.reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = np.sqrt(np.maximum(1 + m00 + m11 + m22, 1e-10)) / 2
    qx = np.sqrt(np.maximum(1 + m00 - m11 - m22, 1e-10)) / 2
    qy = np.sqrt(np.maximum(1 - m00 + m11 - m22, 1e-10)) / 2
    qz = np.sqrt(np.maximum(1 - m00 - m11 + m22, 1e-10)) / 2
    cands = np.stack(
        (
            np.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
            np.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
            np.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
            np.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ),
        axis=-2,
    )
    idx = np.argmax(np.stack((qw, qx, qy, qz), -1), -1)
    q = np.take_along_axis(cands, idx[..., None, None], axis=-2).squeeze(-2)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q * np.where(q[..., :1] < 0, -1.0, 1.0)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = np.zeros_like(x)
    return np.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(v.shape[:-1] + (3, 3))


def so3_exp(v: np.ndarray) -> np.ndarray:
    theta2 = (v * v).sum(-1, keepdims=True)
    theta = np.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = np.where(small, 1 - theta2 / 6, np.sin(theta) / theta)
    b = np.where(small, 0.5 - theta2 / 24, (1 - np.cos(theta)) / (theta2 + 1e-12))
    K = skew(v)
    return np.eye(3) + a[..., None] * K + b[..., None] * (K @ K)
