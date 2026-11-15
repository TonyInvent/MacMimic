"""Micro-benchmark of the policy update's pieces (BlinkMimic sizes) on this GPU: where the learner time goes.

    uv run python tools/bench_learner.py [--bf16] [--batch 2048]
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from supermimic_mlx import networks
from supermimic_mlx.kinematics import Kinematics
from supermimic_mlx.mdp import MDP, unflatten_state
from supermimic_mlx.motion import MotionLib
from supermimic_mlx.networks import Critic, Policy, WorldModel
from supermimic_mlx.optim import MuonAdam, flat, unflat
from supermimic_mlx.robots import g1
from supermimic_mlx.sim import XiLayout

p = argparse.ArgumentParser()
p.add_argument("--bf16", action="store_true")
p.add_argument("--batch", type=int, default=2048)
p.add_argument("--horizon", type=int, default=32)
a = p.parse_args()
if a.bf16:
    networks.set_compute_dtype(mx.bfloat16)
B, H, K, J = a.batch, a.horizon, 10, 29
kin = Kinematics()
motion = MotionLib.from_dir(g1.MOTION_DIR, kin, g1.load_motion)
layout = XiLayout()
mdp = MDP(kin, motion, layout.dim)
G = motion.goal_dim(K)
pi = Policy(mdp.obs_dim, G, J)
wm = WorldModel(J, layout.dim, hidden=(1024, 1024, 1024), structured=True)
critic = Critic(1 + 6 + 6 + 2 * J + mdp.critic_extra + G)
rng = np.random.default_rng(0)
f = mx.array(rng.integers(0, motion.num_frames - 100, B).astype(np.int32))
s = np.array(np.concatenate([np.array(x) for x in (motion.root_p[f], np.zeros((B, 4)), motion.root_v[f], motion.root_w[f], motion.q[f], motion.qd[f])], -1),
             dtype=np.float32)  # fmt: skip
s[:, 3] = 1
S0 = unflatten_state(mx.array(s))
xi = mx.zeros((B, layout.dim))
oh = mx.random.normal((B, 10, mdp.obs_dim))


def rollout(with_reward=True, with_obs=True):
    S, o_hist, fr, a_prev = S0, oh, f, mx.zeros((B, J))
    Ss, As, Aps, Fs = [], [], [], []
    for _ in range(H):
        g = motion.goal(fr, K)
        Aps.append(a_prev)
        a_prev = pi(o_hist, g)
        S = wm.step(S, a_prev, xi)
        fr = fr + 1
        if with_obs:
            o1 = mdp.add_noise(mdp.obs(S, a_prev, xi, fr))
            o_hist = mx.concatenate((o_hist[:, 1:], o1[:, None]), 1)
        Ss.append(S), As.append(a_prev), Fs.append(fr)
    if not with_reward:
        return sum(x.sum() for x in S) * 1e-3
    cat = lambda xs: mx.concatenate(xs, 0)  # noqa: E731
    S_all = tuple(cat([St[i] for St in Ss]) for i in range(6))
    r = mdp.reward(S_all, cat(Fs), cat(As), cat(Aps), w_contact=0.25)
    return -r.mean()


def timed(name, fn, n=5):
    fn_c = mx.compile(fn)
    mx.eval(fn_c())
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn_c())
    print(f"{name:42s} {1000 * (time.perf_counter() - t0) / n:8.1f} ms")


print(f"batch {B}, horizon {H}, {'bf16' if a.bf16 else 'fp32'} matmuls")
timed("rollout forward (policy+WM only)", lambda: rollout(False, False))
timed("rollout forward (+obs)", lambda: rollout(False, True))
timed("rollout forward (+obs +reward/FK)", lambda: rollout(True, True))
vg = nn.value_and_grad(pi, lambda: rollout(True, True))
timed("value_and_grad (policy BPTT)", lambda: vg()[1])
vg2 = nn.value_and_grad(pi, lambda: rollout(False, False))
timed("value_and_grad (no obs, no reward)", lambda: vg2()[1])
opt = MuonAdam(flat(pi.trainable_parameters()), pi.muon_paths())
grads = flat(pi.trainable_parameters())
timed("MuonAdam step (policy)", lambda: list(opt.step(flat(pi.trainable_parameters()), grads, mx.array(1e-4)).values()))
opt2 = MuonAdam(flat(pi.trainable_parameters()), ())
timed("Adam step (policy)", lambda: list(opt2.step(flat(pi.trainable_parameters()), grads, mx.array(1e-4)).values()))
X = mx.random.normal((H * B // 4, critic.norm.mean.shape[0]))
Y = mx.random.normal((H * B // 4,))
cvg = nn.value_and_grad(critic, lambda: ((critic(X) - Y) ** 2).mean())
timed("critic minibatch grad (x20 per update)", lambda: cvg()[1])
