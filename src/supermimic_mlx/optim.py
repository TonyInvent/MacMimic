"""Optimizers with PyTorch semantics, on flat {path: array} parameter dicts.

- ``Adam``: torch.optim.Adam (bias-corrected, eps outside the square root).
- ``MuonAdam``: Muon (Jordan et al. 2024; torch.optim.Muon with nesterov, 5 Newton-Schulz steps, adjust_lr_fn=
  'match_rms_adamw' (Liu et al. 2025), no weight decay) on the given hidden weight matrices, Adam on the rest.
- ``clip_grad_norm`` / ``clip_grad_value``: torch.nn.utils semantics.

Every step is a pure function of (params, grads, state, lr, ok), so it can run inside mx.compile; ``ok`` = False skips
the step without changing parameters or optimizer state (upstream's gated non-finite update).
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

NS_COEF = (3.4445, -4.7750, 2.0315)


def flat(tree) -> dict:
    return dict(tree_flatten(tree))


def unflat(d: dict):
    return tree_unflatten(list(d.items()))


def global_norm(grads: dict) -> mx.array:
    return mx.sqrt(sum((g.astype(mx.float32) ** 2).sum() for g in grads.values()))


def clip_grad_norm(grads: dict, max_norm: float):
    n = global_norm(grads)
    coef = mx.minimum(max_norm / (n + 1e-6), 1.0)
    return {k: g * coef for k, g in grads.items()}, n


def clip_grad_value(grads: dict, v: float) -> dict:
    return {k: mx.clip(g, -v, v) for k, g in grads.items()}


NS_DTYPE = mx.float32  # train.py --ns_bf16 sets bf16 (as torch.optim.Muon runs it); off by default, no speedup measured here


def newton_schulz(G: mx.array, steps: int = 5, eps: float = 1e-7) -> mx.array:
    """torch.optim._muon._zeropower_via_newtonschulz (torch runs it in bf16; here in NS_DTYPE)."""
    a, b, c = NS_COEF
    X = G.astype(NS_DTYPE)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T
    X = X / mx.maximum(mx.sqrt((X * X).sum()), eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    X = X.T if transposed else X
    return X.astype(G.dtype)


class MuonAdam:
    """Muon on the parameters named in ``muon_paths``, Adam (betas) on all others. ``muon_paths`` empty = plain Adam."""

    def __init__(self, params: dict, muon_paths=(), betas=(0.9, 0.999), eps: float = 1e-8, momentum: float = 0.95):
        self.muon_paths = set(muon_paths)
        assert self.muon_paths <= set(params), self.muon_paths - set(params)
        self.b1, self.b2, self.eps, self.mom = betas[0], betas[1], eps, momentum
        self.state = {"t": mx.zeros(()), "m": {}, "v": {}, "buf": {}}
        for k, p in params.items():
            if k in self.muon_paths:
                self.state["buf"][k] = mx.zeros_like(p)
            else:
                self.state["m"][k], self.state["v"][k] = mx.zeros_like(p), mx.zeros_like(p)

    def step(self, params: dict, grads: dict, lr: mx.array, ok: mx.array | None = None) -> dict:
        """New parameters; updates self.state in place (so mx.compile can track it as an input / output tree).
        ``lr`` is a 0-dim array (a changing schedule does not recompile)."""
        st = self.state
        okf = mx.array(True) if ok is None else ok
        t = st["t"] + 1
        bc1, bc2 = 1 - self.b1**t, 1 - self.b2**t
        out = {}
        for k, p in params.items():
            g = grads[k]
            if k in self.muon_paths:
                buf = st["buf"][k]
                b = buf + (1 - self.mom) * (g - buf)
                upd = newton_schulz(g + self.mom * (b - g))
                ratio = 0.2 * math.sqrt(max(p.shape[0], p.shape[1]))
                out[k] = mx.where(okf, p - lr * ratio * upd, p)
                st["buf"][k] = mx.where(okf, b, buf)
            else:
                m = self.b1 * st["m"][k] + (1 - self.b1) * g
                v = self.b2 * st["v"][k] + (1 - self.b2) * g * g
                new = p - lr * (m / bc1) / (mx.sqrt(v) / mx.sqrt(bc2) + self.eps)
                out[k] = mx.where(okf, new, p)
                st["m"][k], st["v"][k] = mx.where(okf, m, st["m"][k]), mx.where(okf, v, st["v"][k])
        st["t"] = mx.where(okf, t, st["t"])
        return out
