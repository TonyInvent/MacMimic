"""Core checks: rotations, FK against MuJoCo, mirror symmetry, world-model integration, checkpoints, reset round-trip."""

import os

import mlx.core as mx
import mujoco
import numpy as np
import pytest

from supermimic_mlx import nprot
from supermimic_mlx.checkpoint import load_tree, save_tree
from supermimic_mlx.kinematics import Kinematics
from supermimic_mlx.mdp import flatten_state, unflatten_state
from supermimic_mlx.mirror import Mirror
from supermimic_mlx.networks import Policy, WorldModel
from supermimic_mlx.optim import flat
from supermimic_mlx.robots import g1
from supermimic_mlx.rotations import mat_to_quat, quat_to_mat, so3_exp, so3_log
from supermimic_mlx.scene import build_model
from supermimic_mlx.sim import XiLayout

HAVE_DATA = os.path.exists(g1.MJCF_PATH)
need_data = pytest.mark.skipif(not HAVE_DATA, reason="run tools/download_data.sh first")
rng = np.random.default_rng(0)


def rand_quat(n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_rotations_roundtrip():
    q = rand_quat(256).astype(np.float32)
    q *= np.where(q[:, :1] < 0, -1, 1)
    R = quat_to_mat(mx.array(q))
    np.testing.assert_allclose(np.array(mat_to_quat(R)), q, atol=1e-5)
    np.testing.assert_allclose(nprot.mat_to_quat(nprot.quat_to_mat(q.astype(np.float64))), q, atol=1e-6)
    v = rng.uniform(-1.5, 1.5, (256, 3)).astype(np.float32)  # |v| < pi: no wrap-around
    np.testing.assert_allclose(np.array(so3_log(so3_exp(mx.array(v)))), v, atol=1e-4)
    np.testing.assert_allclose(np.array(so3_exp(mx.array(v))), nprot.so3_exp(v.astype(np.float64)), atol=1e-5)


@need_data
def test_fk_matches_mujoco():
    m = build_model()
    d = mujoco.MjData(m)
    kin = Kinematics()
    bids = [m.body("robot/" + b).id for b in g1.TRACKED_BODIES]
    for _ in range(5):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[3:7] = rand_quat(1)[0]
        d.qpos[7:] += rng.normal(size=29) * 0.3
        d.qvel[:] = rng.normal(size=m.nv)
        mujoco.mj_forward(m, d)
        R = nprot.quat_to_mat(d.qpos[3:7])
        S = (d.qpos[:3], R, d.qvel[:3], R @ d.qvel[3:6], d.qpos[7:], d.qvel[6:])
        bp, bR, bv, bw = (np.array(x)[0] for x in kin.forward(*(mx.array(x[None].astype(np.float32)) for x in S)))
        for k, b in enumerate(bids):
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
            vel[3:] += np.cross(vel[:3], d.xpos[b] - d.xipos[b])
            np.testing.assert_allclose(bp[k], d.xpos[b], atol=1e-5)
            np.testing.assert_allclose(bR[k], d.xmat[b].reshape(3, 3), atol=1e-5)
            np.testing.assert_allclose(bw[k], vel[:3], atol=1e-4)
            np.testing.assert_allclose(bv[k], vel[3:], atol=1e-4)


@need_data
def test_mirror_is_an_involution_and_mirrors_fk():
    layout = XiLayout()
    mir = Mirror(layout.slices)
    n = 16
    s = np.concatenate((rng.normal(size=(n, 3)), rand_quat(n), rng.normal(size=(n, 6 + 58)) * 0.3), -1).astype(np.float32)
    S = unflatten_state(mx.array(s))
    S2 = mir.state(mir.state(S))
    for a, b in zip(S, S2):
        np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-5)
    xi = mx.array(rng.normal(size=(n, layout.dim)).astype(np.float32))
    np.testing.assert_allclose(np.array(mir.xi(mir.xi(xi))), np.array(xi), atol=1e-6)
    # body positions of the mirrored state are the mirrored body positions, left and right swapped
    kin = Kinematics()
    bp = np.array(kin.forward(*S)[0])
    bpm = np.array(kin.forward(*mir.state(S))[0])
    names = list(g1.TRACKED_BODIES)
    partner = [names.index(b.replace("left_", "@").replace("right_", "left_").replace("@", "right_")) for b in names]
    np.testing.assert_allclose(bpm, bp[:, partner] * np.array([1, -1, 1]), atol=2e-3)  # the G1 is symmetric to ~mm


def test_world_model_zero_output_is_drift():
    """With the zero-initialized last layer and identity normalizers, a structured model integrates the old
    velocities (semi-implicit drift) and keeps them; a plain one keeps the state."""
    J, n = 29, 8
    s = np.concatenate((rng.normal(size=(n, 3)), rand_quat(n), rng.normal(size=(n, 6 + 2 * J)) * 0.3), -1).astype(np.float32)
    S = unflatten_state(mx.array(s))
    a, xi = mx.zeros((n, J)), mx.zeros((n, 33))
    for structured in (False, True):
        wm = WorldModel(J, 33, hidden=(32, 32), structured=structured)
        wm.out_norm.var = mx.zeros_like(wm.out_norm.var)
        wm.out_norm.eps = 0.0
        S1 = wm.step(S, a, xi)
        dq = np.array(S1[4] - S[4])
        np.testing.assert_allclose(dq, 0.02 * np.array(S[5]) if structured else 0.0, atol=1e-5)
        np.testing.assert_allclose(np.array(S1[5]), np.array(S[5]), atol=1e-6)
    np.testing.assert_allclose(np.array(flatten_state(S)), s * np.where(s[:, 3:4] < 0, -1, 1) ** np.r_[np.zeros(3), np.ones(4), np.zeros(6 + 2 * J)],
                               atol=1e-5)  # fmt: skip


def test_checkpoint_roundtrip(tmp_path):
    pi = Policy(99, 730, 29, enc_hidden=(64, 32), dec_hidden=(64, 32))
    p = flat(pi.parameters())
    save_tree(str(tmp_path / "ck"), {"pi": p, "opt": {"t": mx.array(3.0), "m": {"a.b": mx.ones(2)}}, "meta": {"iteration": 7}})
    ck = load_tree(str(tmp_path / "ck"))
    assert ck["meta"]["iteration"] == 7 and float(ck["opt"]["t"]) == 3.0 and ck["opt"]["m"]["a.b"].shape == (2,)
    assert set(ck["pi"]) == set(p) and all(mx.array_equal(ck["pi"][k], p[k]) for k in p)


def test_backward_stays_finite_for_large_preactivations():
    """nn.elu's untaken exp branch overflows for x > ~88 and its gradient inf * 0 = NaN poisoned every policy gradient
    (bf16 runs from ~iteration 750); the MLP's ELU and the rotation helpers must keep finite gradients."""
    from supermimic_mlx.networks import MLP, elu

    x = mx.array([-3.0, 0.5, 100.0, 1000.0])
    assert bool(mx.isfinite(mx.grad(lambda x: elu(x).sum())(x)).all())
    net = MLP(4, [8], 2)
    g = mx.grad(lambda x: net(x).sum())(mx.full((3, 4), 500.0))
    assert bool(mx.isfinite(g).all())
    assert bool(mx.isfinite(mx.grad(lambda R: so3_log(R).sum())(mx.eye(3))).all())
