"""Fixed-size replay buffer of simulator transitions, laid out as (time, env), in host (unified) memory.

Writes are NumPy row copies; samples are gathered on the CPU and handed to MLX as arrays.
"""

from __future__ import annotations

import numpy as np

from .mdp import state_dim


def fields(num_j: int, xi_dim: int) -> dict:
    return {
        "s": (state_dim(num_j), np.float32),
        "s1": (state_dim(num_j), np.float32),
        "a": (num_j, np.float32),
        "a_prev": (num_j, np.float32),
        "xi": (xi_dim, np.float32),
        "f": ((), np.int32),
        "ep_step": ((), np.int32),
        "cut": ((), np.bool_),
    }


class ReplayBuffer:
    def __init__(self, T: int, N: int, num_j: int, xi_dim: int, seed: int = 0):
        self.T, self.N = T, N
        self.fields = fields(num_j, xi_dim)
        self.data = {k: np.zeros((T, N) + ((s,) if isinstance(s, int) else s), dtype=dt) for k, (s, dt) in self.fields.items()}
        self.ptr, self.size = 0, 0
        self.rng = np.random.default_rng(seed + 12345)

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: np.ndarray) -> np.ndarray:
        """Age-ordered time index (0 = oldest) -> storage row."""
        return (self.ptr - self.size + age) % self.T

    def focus_index(self, n: int, wframe: np.ndarray, span: int = 1, cand: int = 131072):
        """n (age, env) picks drawn in proportion to wframe[frame] among `cand` uniform candidates whose `span`-step
        window fits in the buffer; None when no candidate carries weight."""
        age = self.rng.integers(0, self.size - span + 1, cand)
        env = self.rng.integers(0, self.N, cand)
        w = wframe[self.data["f"][self._row(age), env]]
        tot = float(w.sum())
        if n <= 0 or tot <= 0:
            return None
        pick = self.rng.choice(cand, size=n, p=w / tot)
        return age[pick], env[pick]

    def sample_windows(self, B: int, k: int, focus=None) -> dict:
        """k-step windows: (B, k, ...) arrays and a validity mask (B, k). focus = (n, per-frame weights): n of the B
        windows start at frames drawn in proportion to the weights (--focus_replay_wm)."""
        t0 = self.rng.integers(0, self.size - k + 1, B)
        env = self.rng.integers(0, self.N, B)
        if focus is not None:
            idx = self.focus_index(focus[0], focus[1], span=k)
            if idx is not None:
                t0[: focus[0]], env[: focus[0]] = idx
        rows = self._row(t0[:, None] + np.arange(k))
        e = np.broadcast_to(env[:, None], rows.shape)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "a_prev", "xi", "cut")}
        # transition j is valid if the state stayed continuous (no episode end) after 0..j-1
        d = out.pop("cut").astype(np.float32)
        out["mask"] = np.cumprod(np.concatenate((np.ones((B, 1), np.float32), 1 - d[:, :-1]), -1), -1)
        return out

    def sample_starts(self, B: int, hist: int):
        age = self.rng.integers(0, self.size, B)
        env = self.rng.integers(0, self.N, B)
        return self.starts_at(age, env, hist)

    def starts_at(self, age: np.ndarray, env: np.ndarray, hist: int):
        """Start points at (age, env) plus their observation history (oldest first), staying inside the episode."""
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].astype(np.int64)
        back = np.arange(hist - 1, -1, -1)
        back = np.minimum(back[None], ep_step[:, None])
        back = np.minimum(back, age[:, None])
        hrows = self._row(age[:, None] - back)
        henv = np.broadcast_to(env[:, None], hrows.shape)
        keys = ("s", "a_prev", "xi", "f")
        return {k: self.data[k][row, env] for k in keys}, {k: self.data[k][hrows, henv] for k in keys}
