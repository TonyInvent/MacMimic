"""MuonAdam against torch.optim.Muon (match_rms_adamw, Newton-Schulz in fp32) + torch.optim.Adam: tests/data_optim_ref.npz
(5 steps, made by the PyTorch script in the commit history: W, grads G<i>, final parameters P)."""

import os

import mlx.core as mx
import numpy as np

from supermimic_mlx.optim import MuonAdam


def test_muon_adam_matches_torch():
    ref = np.load(os.path.join(os.path.dirname(__file__), "data_optim_ref.npz"))
    params = {k: mx.array(ref["W_" + k]) for k in "abc"}
    opt = MuonAdam(params, muon_paths=["a", "c"], betas=(0.9, 0.95))
    for i in range(5):
        params = opt.step(params, {k: mx.array(ref[f"G{i}_{k}"]) for k in "abc"}, mx.array(1e-2))
    for k in "abc":
        np.testing.assert_allclose(np.array(params[k]), ref["P_" + k], rtol=1e-4, atol=1e-5)


def test_gated_step_is_a_noop():
    params = {"w": mx.ones((4, 3)), "b": mx.ones(3)}
    opt = MuonAdam(params, muon_paths=["w"])
    out = opt.step(params, {"w": mx.ones((4, 3)), "b": mx.ones(3)}, mx.array(0.1), ok=mx.array(False))
    assert all(mx.array_equal(out[k], params[k]) for k in params)
    assert float(opt.state["t"]) == 0
