"""Batched rotation utilities in MLX (quaternions wxyz; rotation matrices act on column vectors).

3x3 products are written elementwise (multiply + sum), which mx.compile fuses (upstream --fused_rot).
"""

from __future__ import annotations

import mlx.core as mx


def mm3(A: mx.array, B: mx.array) -> mx.array:
    """A @ B for (..., 3, 3) matrices (broadcasting). Unrolled over the contraction index: elementwise multiply-adds that
    mx.compile fuses with their neighbours (a broadcast product + sum is a separate, slow reduction kernel); the
    same operations in the same order as the summation, so bit-identical."""
    return A[..., :, 0:1] * B[..., 0:1, :] + A[..., :, 1:2] * B[..., 1:2, :] + A[..., :, 2:3] * B[..., 2:3, :]


def mv3(A: mx.array, v: mx.array) -> mx.array:
    """A @ v for (..., 3, 3) matrices and (..., 3) vectors (broadcasting); unrolled as mm3."""
    return A[..., :, 0] * v[..., 0:1] + A[..., :, 1] * v[..., 1:2] + A[..., :, 2] * v[..., 2:3]


def sum_last(x: mx.array, keepdims: bool = False) -> mx.array:
    """x.sum(-1) for a short last axis (<= 9), unrolled into elementwise adds in index order: mx.compile fuses them with
    their neighbours, where a reduction is a separate kernel (slow on the many tiny (..., 3) reductions of FK)."""
    n = x.shape[-1]
    if n > 9:
        return x.sum(-1, keepdims=keepdims)
    out = x[..., 0:1] if keepdims else x[..., 0]
    for i in range(1, n):
        out = out + (x[..., i : i + 1] if keepdims else x[..., i])
    return out


def T(R: mx.array) -> mx.array:
    return mx.swapaxes(R, -1, -2)


def normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    # clamp before the square root: sqrt's gradient at an exact zero is inf (and inf * 0 = NaN downstream)
    return x / mx.sqrt(mx.maximum(sum_last(x * x, keepdims=True), eps * eps))


def quat_to_mat(q: mx.array) -> mx.array:
    q = normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = mx.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    )  # fmt: skip
    return m.reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: mx.array) -> mx.array:
    """Robust matrix -> quaternion (wxyz, w >= 0): the best-conditioned of the four candidates."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = mx.sqrt(mx.maximum(1 + m00 + m11 + m22, 1e-10)) / 2
    qx = mx.sqrt(mx.maximum(1 + m00 - m11 - m22, 1e-10)) / 2
    qy = mx.sqrt(mx.maximum(1 - m00 + m11 - m22, 1e-10)) / 2
    qz = mx.sqrt(mx.maximum(1 - m00 - m11 + m22, 1e-10)) / 2
    cands = mx.stack(
        (
            mx.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
            mx.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
            mx.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
            mx.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ),
        axis=-2,
    )
    idx = mx.stop_gradient(mx.argmax(mx.stack((qw, qx, qy, qz), -1), -1))
    q = mx.take_along_axis(cands, idx[..., None, None], axis=-2).squeeze(-2)
    q = normalize(q)
    return q * mx.where(q[..., :1] < 0, -1.0, 1.0)


def axis_angle_to_mat(axis: mx.array, angle: mx.array) -> mx.array:
    """Rodrigues for a fixed unit axis (3,) and angles (...)."""
    x, y, z = axis[0], axis[1], axis[2]
    c, s = mx.cos(angle), mx.sin(angle)
    C = 1 - c
    m = mx.stack(
        (
            c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C,
        ),
        axis=-1,
    )  # fmt: skip
    return m.reshape(angle.shape + (3, 3))


def skew(v: mx.array) -> mx.array:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = mx.zeros_like(x)
    return mx.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(v.shape[:-1] + (3, 3))


def so3_exp(v: mx.array) -> mx.array:
    """Rotation vector (..., 3) -> matrix, smooth at zero."""
    theta2 = sum_last(v * v, keepdims=True)
    theta = mx.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = mx.where(small, 1 - theta2 / 6, mx.sin(theta) / theta)
    b = mx.where(small, 0.5 - theta2 / 24, (1 - mx.cos(theta)) / (theta2 + 1e-12))
    K = skew(v)
    eye = mx.eye(3, dtype=v.dtype)
    return eye + a[..., None] * K + b[..., None] * mm3(K, K)


def quat_to_rotvec(q: mx.array) -> mx.array:
    q = q * mx.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sin_half = mx.sqrt(mx.maximum(sum_last(v * v, keepdims=True), 1e-30))  # finite gradient at the identity
    half = mx.arctan2(sin_half, q[..., :1])
    scale = mx.where(sin_half < 1e-6, 2.0 / mx.maximum(q[..., :1], 1e-6), 2 * half / (sin_half + 1e-12))
    return v * scale


def so3_log(R: mx.array) -> mx.array:
    """Matrix -> rotation vector via the quaternion (stable near 0 and pi)."""
    return quat_to_rotvec(mat_to_quat(R))


def rot_angle_sq(Ra: mx.array, Rb: mx.array) -> mx.array:
    """Squared geodesic angle between rotations (acos of the trace, clamped)."""
    tr = sum_last((Ra * Rb).reshape(Ra.shape[:-2] + (9,)))
    cos = mx.clip((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return mx.arccos(cos) ** 2


def yaw_mat(R: mx.array) -> mx.array:
    """Heading (yaw-only) rotation of R, from the projected x axis."""
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    n = mx.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / n, fy / n
    o, one = mx.zeros_like(c), mx.ones_like(c)
    return mx.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(R.shape[:-2] + (3, 3))


def mat_to_6d(R: mx.array) -> mx.array:
    """First two columns, flattened (6D continuous rotation representation)."""
    return R[..., :, :2].reshape(R.shape[:-2] + (6,))


def quat_slerp(q0: mx.array, q1: mx.array, t: mx.array) -> mx.array:
    dot = sum_last(q0 * q1, keepdims=True)
    q1 = mx.where(dot < 0, -q1, q1)
    dot = mx.minimum(mx.abs(dot), 1.0)
    theta = mx.arccos(dot)
    s = mx.sin(theta)
    t = t[..., None]
    w0 = mx.where(s < 1e-6, 1 - t, mx.sin((1 - t) * theta) / (s + 1e-12))
    w1 = mx.where(s < 1e-6, t, mx.sin(t * theta) / (s + 1e-12))
    return normalize(w0 * q0 + w1 * q1)
