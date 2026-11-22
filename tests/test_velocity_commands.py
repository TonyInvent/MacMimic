import numpy as np
import pytest

from supermimic_mlx import velocity


@pytest.mark.parametrize("profile, low, high", [
    ("paper", [-1.5, -1.32, -3.35], [2, 1.32, 3.35]),
    ("opensprite", [-.3, -.2, -.6], [.6, .2, .6]),
])
def test_random_velocity_commands_and_hold(monkeypatch, profile, low, high):
    monkeypatch.setattr(velocity, "motion_commands", lambda m: np.zeros((1, 3), np.float32))
    n = 4096
    env = velocity.Commands(n, None, seed=7, profile=profile)
    frames = np.zeros(n, dtype=np.int32)
    sampled = env.sample(frames, frames)
    random = sampled[sampled[:, 3] == 0, :3]
    assert len(random) > 1000
    np.testing.assert_allclose(random.min(0), low, atol=.025)
    np.testing.assert_allclose(random.max(0), high, atol=.025)
    np.testing.assert_equal(env.sample(frames, frames + 249)[:, :4], sampled[:, :4])
    assert not np.array_equal(env.sample(frames, frames + 250)[:, :4], sampled[:, :4])


def test_compact_reference_preserves_clip_boundaries():
    from supermimic_mlx.kinematics import Kinematics
    from supermimic_mlx.motion import MotionLib
    from supermimic_mlx.robots import opensprite

    kin = Kinematics(opensprite.MJCF_PATH, opensprite.JOINT_NAMES, opensprite.TRACKED_BODIES)
    poses = np.tile(opensprite.MODEL.qpos0.astype(np.float32), (14, 1))
    poses[:,0] = np.r_[np.arange(7)*.01, 8+np.arange(7)*.02]
    poses[:,7] = np.sin(np.arange(14)*.2)
    names = ["a", "b"]
    compact = MotionLib.from_qpos50(names, [7,7], poses, kin)
    def load(name):
        clip = poses[:7] if name == "a" else poses[7:]
        return clip[:,:3], clip[:,3:7], clip[:,7:], 50.
    dense = MotionLib(names, kin, load)
    for key in dense.np:
        np.testing.assert_allclose(compact.np[key], dense.np[key], atol=1e-4)
    for a,b in zip(compact.reference_bodies(slice(None)), dense.reference_bodies(slice(None))):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-4)
    np.testing.assert_allclose(velocity.motion_commands(compact),velocity.motion_commands(dense),atol=1e-4)
