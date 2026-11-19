"""How a checkpoint fails the inverted move of dance1_subject1: worlds reset onto the reference at different start
frames, each run until it terminates or passes frame END.  PYTHONPATH=src python tools/move_diag.py <ckpt> [more ckpts]"""
import sys
import numpy as np, mlx.core as mx
from supermimic_mlx import networks
from supermimic_mlx.checkpoint import load_tree
from supermimic_mlx.kinematics import Kinematics
from supermimic_mlx.mdp import MDP, unflatten_state
from supermimic_mlx.motion import MotionLib
from supermimic_mlx.networks import policy_from_args
from supermimic_mlx.optim import unflat
from supermimic_mlx.robots import g1
from supermimic_mlx.sim import TrackingSim, XiLayout

kin = Kinematics(); m = MotionLib.from_dir(g1.MOTION_DIR, kin, g1.load_motion); layout = XiLayout(); mdp = MDP(kin, m, layout.dim)
c = m.names.index("dance1_subject1"); c0 = int(m.clip_start_np[c])
STARTS = np.array([5100, 5200, 5250, 5280, 5300, 5310, 5320, 5330, 5340, 5350, 5360, 5370, 5380, 5700, 5740, 5760, 5780, 5790, 5800, 5810])
END = np.where(STARTS < 5500, 5450, 5900)
P = len(STARTS)
env = TrackingSim(P, m, mdp, layout, eval_mode=True, randomize=False, reset_noise=False, seed=0, nthread=4)
env.term_only = True
refz, refr = env._ref_z, env._ref_r22
print("term_height", mdp.cfg.term_height, "term_tilt", mdp.cfg.term_tilt)
for path in sys.argv[1:]:
    ck = load_tree(path); a_ = ck["meta"]["args"]; K, Hh = a_["goal_frames"], a_["hist"]
    networks.set_compute_dtype(mx.bfloat16 if a_.get("precision") == "bf16" else None)
    m.goal_stride = a_.get("goal_stride", 1)
    pi = policy_from_args(a_, mdp.obs_dim, m.goal_dim(K), len(g1.JOINT_NAMES)); pi.update(unflat(ck["pi"]))
    def act(s, a_prev, xi, f, hist):
        o = mdp.obs(unflatten_state(s), a_prev, xi, f)
        hist = mx.concatenate((hist[:, 1:], o[:, None]), 1)
        return pi(hist, m.goal(f, K)), hist
    env.done_eval[:] = False; env.failed_eval[:] = False
    env.reset(np.arange(P), c0 + STARTS)
    o = mdp.obs(unflatten_state(mx.array(env.state_flat())), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)))
    hist = mx.broadcast_to(o[:, None], (P, Hh, mdp.obs_dim))
    fail = np.full(P, -1); why = [""] * P; done = np.zeros(P, bool)
    for _ in range(int((END - STARTS).max())):
        a, hist = act(mx.array(env.state_flat()), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)), hist)
        a = np.array(a); f_now = env.f.copy()
        out = env.step_core(a)
        s1 = env.state_flat(); q = s1[:, 3:7] / np.linalg.norm(s1[:, 3:7], axis=-1, keepdims=True)
        r22 = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2); f1 = f_now + 1
        t = out["term"] & ~done
        for i in np.nonzero(t)[0]:
            fail[i] = f_now[i] - c0
            why[i] = f"dz {s1[i, 2] - refz[f1[i]]:+.2f} (z {s1[i, 2]:.2f} ref {refz[f1[i]]:.2f}) dr22 {r22[i] - refr[f1[i]]:+.2f} (r22 {r22[i]:+.2f} ref {refr[f1[i]]:+.2f})"
        done |= t | (env.f - c0 >= END)
        if done.all(): break
    print(path.split("logs/hill/")[-1])
    for i in range(P):
        print(f"   start {STARTS[i]}: " + (f"fails at {fail[i]} (+{fail[i] - STARTS[i]}) {why[i]}" if fail[i] >= 0 else "passes"))
