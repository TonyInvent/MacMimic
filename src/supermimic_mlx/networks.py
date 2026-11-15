"""World model, policy (motion encoder + shared decoder) and critic (SuperMimic paper Table A.4), in MLX.

Running normalizers are frozen arrays of the modules (not trained); they are part of ``module.parameters()`` (so
checkpoints and the policy EMA include them) but not of ``trainable_parameters()``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .mdp import inv_dim, invariant_features
from .rotations import mm3, mv3, so3_exp, so3_log, yaw_mat, T


class MLP(nn.Module):
    def __init__(self, inp: int, hidden: list[int], out: int, act: str = "elu", layer_norm: bool = False):
        super().__init__()
        dims = [inp] + list(hidden)
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])] + [nn.Linear(dims[-1], out)]
        self.norms = [nn.LayerNorm(h) for h in hidden] if layer_norm else []
        self.act = nn.elu if act == "elu" else nn.silu

    def __call__(self, x):
        for i, l in enumerate(self.layers[:-1]):
            x = l(x)
            if self.norms:
                x = self.norms[i](x)
            x = self.act(x)
        return self.layers[-1](x)

    def hidden_weight_paths(self, prefix: str) -> list[str]:
        """Paths of the hidden weight matrices (all Linear layers but the first and the last): Muon's parameters."""
        return [f"{prefix}.layers.{i}.weight" for i in range(1, len(self.layers) - 1)]


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count`` samples."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.mean = mx.zeros((dim,))
        self.var = mx.ones((dim,))
        self.count = mx.zeros(())
        self.max_count, self.eps = max_count, eps
        self.freeze()

    def update(self, x: mx.array):
        x = mx.stop_gradient(x.reshape(-1, x.shape[-1]).astype(mx.float32))
        n = x.shape[0]
        m, v = x.mean(0), x.var(0)
        tot = self.count + n
        delta = m - self.mean
        mean = self.mean + delta * n / tot
        var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        live = self.count < self.max_count
        self.mean = mx.where(live, mean, self.mean)
        self.var = mx.where(live, var, self.var)
        self.count = mx.where(live, tot, self.count)

    def __call__(self, x):
        return (x - self.mean) / mx.sqrt(self.var + self.eps**2)

    def std(self):
        return mx.sqrt(self.var + self.eps**2)


def _split(x: mx.array, sizes) -> list[mx.array]:
    idx, s = [], 0
    for n in sizes[:-1]:
        s += n
        idx.append(s)
    return mx.split(x, idx, axis=-1)


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments.

    ``structured`` (BlinkMimic --wm_structured): semi-implicit integration, positions follow the predicted velocities
    and the network predicts velocity increments plus small position corrections."""

    DRIFT_W = 0.625  # weight of the new velocity in the position drift: 4 semi-implicit substeps -> (n + 1) / (2 n)

    def __init__(self, num_j: int, xi_dim: int, hidden=(2048, 2048, 2048), structured: bool = False, dt: float = 0.02):
        super().__init__()
        self.num_j, self.structured, self.dt = num_j, structured, dt
        self.in_dim = inv_dim(num_j) + num_j + xi_dim
        self.out_dim = 3 + 3 + 3 + 3 + num_j + num_j
        self.net = MLP(self.in_dim, list(hidden), self.out_dim, act="silu", layer_norm=True)
        last = self.net.layers[-1]
        last.weight = mx.zeros_like(last.weight)
        last.bias = mx.zeros_like(last.bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)
        self.inc_norm = RunningNorm(self.out_dim, eps=1e-4)  # scale of the plain increments (loss normalization)

    def muon_paths(self) -> list[str]:
        return self.net.hidden_weight_paths("net")

    def inputs(self, S, a, xi_n):
        return mx.concatenate((invariant_features(S), a, xi_n), -1)

    def _drift(self, S, v1, w1, qd1):
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        a, b = self.dt * (1 - self.DRIFT_W), self.dt * self.DRIFT_W
        return mv3(T(H), a * v + b * v1), mv3(T(R), a * w + b * w1), a * qd + b * qd1

    @staticmethod
    def increments(S, S1):
        """Frame-local increments from S to S1."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H, Rt = yaw_mat(R), T(R)
        return mx.concatenate((mv3(T(H), p1 - p), so3_log(mm3(Rt, R1)), mv3(Rt, v1 - v), mv3(Rt, w1 - w), q1 - q, qd1 - qd), -1)

    def targets(self, S, S1):
        inc = self.increments(S, S1)
        if not self.structured:
            return inc
        J = self.num_j
        dp, dth, dv, dw, dq, dqd = _split(inc, (3, 3, 3, 3, J, J))
        gp, gth, gq = self._drift(S, S1[2], S1[3], S1[5])
        return mx.concatenate((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)

    def step(self, S, a, xi_n):
        y = self.net(self.in_norm(self.inputs(S, a, xi_n)))
        return self.integrate(S, y * self.out_norm.std() + self.out_norm.mean)

    def integrate(self, S, d):
        p, R, v, w, q, qd = S
        J = self.num_j
        dp, dth, dv, dw, dq, dqd = _split(d, (3, 3, 3, 3, J, J))
        v1, w1, qd1 = v + mv3(R, dv), w + mv3(R, dw), qd + dqd
        if self.structured:
            gp, gth, gq = self._drift(S, v1, w1, qd1)
            dp, dth, dq = dp + gp, dth + gth, dq + gq
        return (p + mv3(yaw_mat(R), dp), mm3(R, so3_exp(dth)), v1, w1, q + dq, qd1)


def state_error(S_hat, S):
    """Frame-local error of S_hat against S (the layout of WorldModel.targets)."""
    return WorldModel.increments(S, S_hat)


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh.

    ``ref_skip`` = (offset, q0, alpha) (BlinkMimic --pi_ref_skip): a = (q_ref(f+1) - q0) / alpha + dec(...)."""

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hist: int = 10, z_dim: int = 32,
                 enc_hidden=(2048, 1024, 512, 256), dec_hidden=(2048, 1024, 512, 256), ref_skip=None):  # fmt: skip
        super().__init__()
        self.hist = hist
        self.ref_off = None
        if ref_skip is not None:
            self.ref_off = ref_skip[0]
            self.ref_q0 = mx.array(ref_skip[1], dtype=mx.float32)
            self.ref_alpha = mx.array(ref_skip[2], dtype=mx.float32)
        self.enc = MLP(goal_dim, list(enc_hidden), z_dim)
        self.dec = MLP(z_dim + obs_dim * hist, list(dec_hidden), act_dim)
        last = self.dec.layers[-1]
        last.weight = last.weight * 0.01
        last.bias = mx.zeros_like(last.bias)
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)
        if ref_skip is not None:
            self.freeze(keys=["ref_q0", "ref_alpha"], recurse=False)

    def muon_paths(self) -> list[str]:
        return self.enc.hidden_weight_paths("enc") + self.dec.hidden_weight_paths("dec")

    def encode(self, goal: mx.array) -> mx.array:
        """Motion latent z from the goal features (reference only, independent of the robot state)."""
        return mx.tanh(self.enc(self.goal_norm(goal)))

    def __call__(self, obs_hist: mx.array, goal: mx.array, z: mx.array | None = None) -> mx.array:
        """``z``: encode(goal), if computed beforehand (XMimic's hoisting of the goal encodings out of the rollout)."""
        o = self.obs_norm(obs_hist)
        o = o.reshape(o.shape[:-2] + (-1,))
        z = self.encode(goal) if z is None else z
        a = self.dec(mx.concatenate((z, o), -1))
        if self.ref_off is not None:
            q_ref = goal[..., self.ref_off : self.ref_off + a.shape[-1]]
            a = a + (q_ref - self.ref_q0) / self.ref_alpha
        return a


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = MLP(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def __call__(self, x):
        return self.net(self.norm(x)).squeeze(-1)


def policy_from_args(a_: dict, obs_dim: int, goal_dim: int, J: int) -> Policy:
    from .motion import GOAL_Q_OFFSET
    from .robots import g1

    widths = lambda k: tuple(int(x) for x in a_[k].split(","))  # noqa: E731
    ref_skip = (GOAL_Q_OFFSET, g1.default_joint_pos().tolist(), g1.action_scales().tolist()) if a_.get("pi_ref_skip") else None
    return Policy(obs_dim, goal_dim, J, hist=a_["hist"], enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"),
                  ref_skip=ref_skip)  # fmt: skip
