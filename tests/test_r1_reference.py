import numpy as np
import pytest

from supermimic_mlx.r1_reference import R1_RESET_WEIGHTS, check_r1_file, validate_clip_layout, with_clip_masses


def test_reset_mass_survives_difficulty_imbalance():
    clip_ids = np.repeat([0, 1], [12, 6])
    difficulty = np.geomspace(1e-9, 1.0, len(clip_ids))
    weights = with_clip_masses(difficulty, clip_ids, np.array(list(R1_RESET_WEIGHTS.values())))
    np.testing.assert_allclose(np.bincount(clip_ids, weights=weights), [0.8, 0.2])
    for clip in [0, 1]:
        mask = clip_ids == clip
        np.testing.assert_allclose(weights[mask] / weights[mask].sum(), difficulty[mask] / difficulty[mask].sum())


def test_reject_wrong_standing_interval():
    names = list(R1_RESET_WEIGHTS)
    validate_clip_layout(names, [572, 300])
    with pytest.raises(ValueError):
        validate_clip_layout(names, [572, 360])


def test_names_and_lengths_alone_cannot_identify_r1(tmp_path):
    path = tmp_path / "walk_stand.npz"
    np.savez(path, motion_names=list(R1_RESET_WEIGHTS), motion_lengths=[572, 300])
    with pytest.raises(ValueError, match="SHA256"):
        check_r1_file(path)
