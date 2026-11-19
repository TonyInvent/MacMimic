"""Compute profile of the learner on this GPU: GEMM efficiency at the shapes the learner uses, and the time / achieved
TFLOPS of each component of one training iteration (BlinkMimic sizes by default).

    uv run python tools/profile_learner.py [--pi_batch 2048 --horizon 32 --fp32]
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from supermimic_mlx import networks
from supermimic_mlx.kinematics import Kinematics
from supermimic_mlx.mdp import MDP, flatten_state, inv_dim, unflatten_state
from supermimic_mlx.motion import MotionLib
from supermimic_mlx.networks import Critic, Policy, WorldModel, state_error
from supermimic_mlx.optim import MuonAdam, clip_grad_norm, flat, unflat
from supermimic_mlx.robots import g1
from supermimic_mlx.sim import XiLayout

p = argparse.ArgumentParser()
p.add_argument("--pi_batch", type=int, default=2048)
p.add_argument("--horizon", type=int, default=32)
p.add_argument("--wm_batch", type=int, default=2048)
p.add_argument("--fp32", action="store_true")
p.add_argument("--reps", type=int, default=5)
p.add_argument("--breakdown", action="store_true", help="split the world-model update and the critic step into parts")
p.add_argument("--enc_hidden", type=str, default="2048,1024,512,256")
p.add_argument("--dec_hidden", type=str, default="2048,1024,512,256")
p.add_argument("--skip_gemm", action="store_true")
p.add_argument("--wm_hidden", type=str, default="1024,1024,1024")
a = p.parse_args()
dt = mx.float32 if a.fp32 else mx.bfloat16
if not a.fp32:
    networks.set_compute_dtype(mx.bfloat16)
B, H, K, J, W = a.pi_batch, a.horizon, 10, 29, 8


def timeit(fn, reps=a.reps):
    out = fn()
    mx.eval(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


rows = []


def report(name, ms, gflop=None):
    tf = f"{gflop / ms:7.2f}" if gflop else "      -"
    rows.append((name, ms, gflop))
    print(f"{name:58s} {ms:8.1f} ms  {gflop or 0:8.0f} GFLOP  {tf} TFLOPS", flush=True)


# ---------------------------------------------------------------- 1. GEMM efficiency at the learner's shapes
print(f"== GEMMs ({'fp32' if a.fp32 else 'bf16'}): forward x(M,K) @ W(K,N); weight grad x(M,K)^T @ g(M,N)")
shapes = {"dec L0 1022->2048": (1022, 2048), "enc L0 730->2048": (730, 2048), "L1 2048->1024": (2048, 1024),
          "L2 1024->512": (1024, 512), "wm L0 133->1024": (133, 1024), "wm L1 1024->1024": (1024, 1024)}  # fmt: skip
for nm, (k, n) in ({} if a.skip_gemm else shapes).items():
    for M in (B, B * H):
        x = mx.random.normal((M, k)).astype(dt)
        w = mx.random.normal((k, n)).astype(dt)
        g = mx.random.normal((M, n)).astype(dt)
        mx.eval(x, w, g)
        f = timeit(lambda: x @ w, reps=10)
        wg = timeit(lambda: x.T @ g, reps=10)
        gf = 2 * M * k * n / 1e9
        print(f"  {nm:20s} M={M:6d}: fwd {gf / f:5.2f} TFLOPS ({f:6.2f} ms) | weight-grad {gf / wg:5.2f} TFLOPS ({wg:6.2f} ms)")

# ---------------------------------------------------------------- 2. learner components
kin = Kinematics()
motion = MotionLib.from_dir(g1.MOTION_DIR, kin, g1.load_motion)
lay = XiLayout()
mdp = MDP(kin, motion, lay.dim)
G = motion.goal_dim(K)
pi = Policy(mdp.obs_dim, G, J, enc_hidden=tuple(int(x) for x in a.enc_hidden.split(",")), dec_hidden=tuple(int(x) for x in a.dec_hidden.split(",")), ref_skip=None)
wm = WorldModel(J, lay.dim, hidden=tuple(int(x) for x in a.wm_hidden.split(",")), structured=True)
critic = Critic(inv_dim(J) + mdp.critic_extra + G)
mx.eval(pi.parameters(), wm.parameters(), critic.parameters())
rng = np.random.default_rng(0)
f0 = mx.array(rng.integers(0, motion.num_frames - 200, B).astype(np.int32))
S0 = (motion.root_p[f0], motion.root_R[f0], motion.root_v[f0], motion.root_w[f0], motion.q[f0], motion.qd[f0])
xi = mx.zeros((B, lay.dim))
oh = mx.random.normal((B, 10, mdp.obs_dim))
P_pi = sum(v.size for _, v in flat(pi.trainable_parameters()).items())
P_enc = sum(v.size for k_, v in flat(pi.trainable_parameters()).items() if k_.startswith("enc"))
P_dec = P_pi - P_enc
P_wm = sum(v.size for _, v in flat(wm.trainable_parameters()).items())
P_v = sum(v.size for _, v in flat(critic.trainable_parameters()).items())
N = B * H
print(f"\n== components (policy batch {B}, horizon {H}; params: enc {P_enc / 1e6:.2f} M, dec {P_dec / 1e6:.2f} M, wm {P_wm / 1e6:.2f} M, critic {P_v / 1e6:.2f} M)")
goals = motion.goal(mx.concatenate([f0 + t for t in range(H)]), K)
mx.eval(goals)
report("encoder fwd, all steps batched", timeit(mx.compile(lambda: pi.encode(goals))), 2 * N * P_enc / 1e9)
dec_in = mx.random.normal((N, 32 + 10 * mdp.obs_dim))
report("decoder fwd, all steps batched", timeit(mx.compile(lambda: pi.dec(dec_in))), 2 * N * P_dec / 1e9)
S_big = tuple(mx.concatenate([x] * H) for x in S0)
a_big = mx.zeros((N, J))
xi_big = mx.zeros((N, lay.dim))
report("world-model step fwd, all steps batched", timeit(mx.compile(lambda: wm.step(S_big, a_big, xi_big)[0])), 2 * N * P_wm / 1e9)
report("FK + reward, all steps batched", timeit(mx.compile(lambda: mdp.reward(S_big, mx.concatenate([f0 + 1] * H), a_big, a_big, w_contact=0.25))))
report("observation + noise, all steps batched", timeit(mx.compile(lambda: mdp.add_noise(mdp.obs(S_big, a_big, xi_big, mx.concatenate([f0 + 1] * H))))))
report("goal features, all steps", timeit(mx.compile(lambda: motion.goal(mx.concatenate([f0 + t for t in range(H)]), K))))


def rollout(reward=True):
    S, o_hist, fr, a_prev = S0, oh, f0, mx.zeros((B, J))
    Ss, As, Aps, Fs = [], [], [], []
    g = motion.goal(fr, K)
    for _ in range(H):
        Aps.append(a_prev)
        a_prev = pi(o_hist, g)
        S = wm.step(S, a_prev, xi)
        fr = fr + 1
        o1 = mdp.add_noise(mdp.obs(S, a_prev, xi, fr))
        o_hist = mx.concatenate((o_hist[:, 1:], o1[:, None]), 1)
        g = motion.goal(fr, K)
        Ss.append(S), As.append(a_prev), Fs.append(fr)
    if not reward:
        return sum((x.astype(mx.float32) ** 2).mean() for x in S) + sum((x ** 2).mean() for x in As)
    cat = lambda xs: mx.concatenate(xs, 0)  # noqa: E731
    S_all = tuple(cat([St[i] for St in Ss]) for i in range(6))
    return -mdp.reward(S_all, cat(Fs), cat(As), cat(Aps), w_contact=0.25).mean()


fl_fwd = 2 * N * (P_enc + P_dec + P_wm) / 1e9
report("BPTT forward (recurrent)", timeit(mx.compile(lambda: rollout())), fl_fwd)
vg = nn.value_and_grad(pi, lambda: rollout())
# backward: policy input + weight grads (2x its forward), world model input grads only (1x)
fl_bwd = fl_fwd + 2 * N * (2 * (P_enc + P_dec) + P_wm) / 1e9
report("BPTT forward + backward", timeit(mx.compile(lambda: vg()[1])), fl_bwd)
vg2 = nn.value_and_grad(pi, lambda: rollout(reward=False))
report("BPTT forward + backward, no reward / FK", timeit(mx.compile(lambda: vg2()[1])), fl_bwd)
opt = MuonAdam(flat(pi.trainable_parameters()), pi.muon_paths(), betas=(0.9, 0.95))
gr = flat(pi.trainable_parameters())
report("policy MuonAdam step", timeit(mx.compile(lambda: list(opt.step(flat(pi.trainable_parameters()), gr, mx.array(1e-4)).values()))))
Xc = mx.random.normal(((H + 1) * B, critic.norm.mean.shape[0]))
report("critic features (FK) over (H+1) B", timeit(mx.compile(lambda: mdp.critic_features(tuple(mx.concatenate([x] * (H + 1)) for x in S0),
                                                                                     mx.concatenate([f0] * (H + 1)), mx.zeros(((H + 1) * B, lay.dim))))))  # fmt: skip
mb = H * B // 4
Xm, Ym = Xc[:mb], mx.random.normal((mb,))
cvg = nn.value_and_grad(critic, lambda: ((critic(Xm) - Ym) ** 2).mean())
copt = MuonAdam(flat(critic.trainable_parameters()), ())


def critic_step():
    _, g_ = cvg()
    g_, _ = clip_grad_norm(flat(g_), 1.0)
    critic.update(unflat(copt.step(flat(critic.trainable_parameters()), g_, mx.array(1e-4))))
    return list(flat(critic.trainable_parameters()).values())


ms_c = timeit(mx.compile(critic_step, inputs=[critic.state, copt.state], outputs=[critic.state, copt.state]))
report("critic: one minibatch step (x20 per iteration)", ms_c, 3 * 2 * mb * P_v / 1e9)
# world model update (one of 5)
Bw = a.wm_batch
fw = mx.array(rng.integers(0, motion.num_frames - 20, Bw).astype(np.int32))
s_w = mx.stack([flatten_state((motion.root_p[fw + j], motion.root_R[fw + j], motion.root_v[fw + j], motion.root_w[fw + j], motion.q[fw + j],
                               motion.qd[fw + j])) for j in range(W + 1)], 1)  # fmt: skip
a_w, xi_w = mx.zeros((Bw, W, J)), mx.zeros((Bw, W, lay.dim))


def wm_loss():
    S = unflatten_state(s_w[:, 0])
    es = []
    for j in range(W):
        S = wm.step(S, a_w[:, j], xi_w[:, j])
        es.append(((state_error(S, unflatten_state(s_w[:, j + 1])) / 0.1) ** 2).mean(-1))
    return mx.stack(es, 1).mean()


wvg = nn.value_and_grad(wm, wm_loss)
wopt = MuonAdam(flat(wm.trainable_parameters()), wm.muon_paths())


def wm_step():
    _, g_ = wvg()
    g_, _ = clip_grad_norm(flat(g_), 1.0)
    wm.update(unflat(wopt.step(flat(wm.trainable_parameters()), g_, mx.array(1e-4))))
    return list(flat(wm.trainable_parameters()).values())


report("world-model update (x5 per iteration)", timeit(mx.compile(wm_step, inputs=[wm.state, wopt.state], outputs=[wm.state, wopt.state])),
       3 * 2 * Bw * W * P_wm / 1e9)  # fmt: skip
# breakdown of one world-model update and one critic step: loss forward, forward + backward, the optimizer step alone
if a.breakdown:
    report("  wm: loss forward", timeit(mx.compile(wm_loss, inputs=[wm.state])), 2 * Bw * W * P_wm / 1e9)
    report("  wm: forward + backward", timeit(mx.compile(lambda: list(flat(wvg()[1]).values()), inputs=[wm.state], outputs=[wm.state])), 3 * 2 * Bw * W * P_wm / 1e9)
    g_fix = flat(wvg()[1]); mx.eval(g_fix)
    report("  wm: gradient clip + MuonAdam step alone", timeit(mx.compile(
        lambda: list(wopt.step(flat(wm.trainable_parameters()), clip_grad_norm(g_fix, 1.0)[0], mx.array(1e-4)).values()),
        inputs=[wm.state, wopt.state], outputs=[wm.state, wopt.state])))
    report("  critic: forward + backward", timeit(mx.compile(lambda: list(flat(cvg()[1]).values()), inputs=[critic.state], outputs=[critic.state])), 3 * 2 * mb * P_v / 1e9)
    gc_fix = flat(cvg()[1]); mx.eval(gc_fix)
    report("  critic: gradient clip + optimizer step alone", timeit(mx.compile(
        lambda: list(copt.step(flat(critic.trainable_parameters()), clip_grad_norm(gc_fix, 1.0)[0], mx.array(1e-4)).values()),
        inputs=[critic.state, copt.state], outputs=[critic.state, copt.state])))
r = {n: (ms, gf) for n, ms, gf in rows}
it_ms = r["BPTT forward + backward"][0] + r["policy MuonAdam step"][0] + r["critic features (FK) over (H+1) B"][0] + 20 * r["critic: one minibatch step (x20 per iteration)"][0] + 5 * r["world-model update (x5 per iteration)"][0]
print(f"\nestimated learner per iteration: {it_ms:.0f} ms = BPTT {r['BPTT forward + backward'][0]:.0f} + Muon {r['policy MuonAdam step'][0]:.0f} + critic "
      f"{r['critic features (FK) over (H+1) B'][0] + 20 * r['critic: one minibatch step (x20 per iteration)'][0]:.0f} + WM {5 * r['world-model update (x5 per iteration)'][0]:.0f}")
