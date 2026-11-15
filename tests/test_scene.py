"""The plain-MuJoCo scene equals mjlab's compile of the upstream scene (data/g1_scene_mjlab.npz, from
tools/build_scene_mjlab.py), apart from the visual geoms that build_model drops."""

import os

import numpy as np
import pytest

from supermimic_mlx.robots import g1
from supermimic_mlx.scene import build_model

REF = os.path.join(g1.DATA_DIR, "g1_scene_mjlab.npz")


@pytest.mark.skipif(not os.path.exists(REF), reason="run tools/build_scene_mjlab.py first")
def test_scene_matches_mjlab():
    ref = np.load(REF)
    m = build_model(visual=True)
    assert [m.body(i).name for i in range(m.nbody)] == list(ref["names"])
    for f in ref.files:
        if f in ("names", "opt"):
            continue
        a, b = np.asarray(getattr(m, f)), ref[f]
        assert a.shape == b.shape, (f, a.shape, b.shape)
        assert np.allclose(a, b, atol=1e-5 if f in ("geom_size", "geom_pos") else 1e-6), (f, np.abs(a - b).max())


def test_collision_model():
    m = build_model()
    assert m.nu == 29 and m.nq == 36
    assert all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 1 for g in range(1, m.ngeom))
