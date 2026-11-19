"""Failure map of a checkpoint: the probe sweep (tiles of 600 frames, lead 150) with a restart after each failure, so
that a failure does not hide the frames behind it.  PYTHONPATH=src python tools/probe_failmap.py <ckpt> [skip] [threads]"""
import sys, time
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

ck = load_tree(sys.argv[1]); SKIP = int(sys.argv[2]) if len(sys.argv) > 2 else 30; NT = int(sys.argv[3]) if len(sys.argv) > 3 else 6
TILE, LEAD = 600, 150
a_ = ck["meta"]["args"]; K, Hh = a_["goal_frames"], a_["hist"]
kin = Kinematics(); m = MotionLib.from_dir(g1.MOTION_DIR, kin, g1.load_motion); layout = XiLayout(); mdp = MDP(kin, m, layout.dim)
networks.set_compute_dtype(mx.bfloat16 if a_.get("precision") == "bf16" else None)
m.goal_stride = a_.get("goal_stride", 1)
pi = policy_from_args(a_, mdp.obs_dim, m.goal_dim(K), len(g1.JOINT_NAMES)); pi.update(unflat(ck["pi"]))
tiles = np.concatenate([np.arange(m.clip_start_np[c], m.clip_end_np[c] - 1, TILE) for c in range(len(m.names))])
P = len(tiles); cid = m.clip_id_np[tiles]
env = TrackingSim(P, m, mdp, layout, eval_mode=True, randomize=False, reset_noise=False, seed=0, nthread=NT)
env.term_only = True
f0 = np.maximum(tiles - LEAD, m.clip_start_np[cid]); stop = np.minimum(tiles + TILE, m.clip_end_np[cid]); fcap = np.minimum(stop, m.clip_end_np[cid] - 1)
def obs():
    return mdp.obs(unflatten_state(mx.array(env.state_flat())), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)))
def act(s, a_prev, xi, f, hist):
    o = mdp.obs(unflatten_state(s), a_prev, xi, f)
    hist = mx.concatenate((hist[:, 1:], o[:, None]), 1)
    return pi(hist, m.goal(f, K)), hist
act_c = mx.compile(act)
env.reset(np.arange(P), f0)
hist = mx.broadcast_to(obs()[:, None], (P, Hh, mdp.obs_dim))
done = np.zeros(P, bool); first = np.full(P, -1); events = []
t0 = time.time()
for step in range(LEAD + TILE):
    a, hist = act_c(mx.array(env.state_flat()), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)), hist)
    a = np.array(a); f_now = env.f.copy()
    out = env.step_core(a)
    term = (out["term"] | ~np.isfinite(a).all(-1)) & ~done
    for i in np.nonzero(term)[0]:
        events.append((int(i), int(f_now[i])))
        if first[i] < 0: first[i] = f_now[i]
    done |= env.f >= stop
    re = term & (f_now + SKIP < stop)
    done |= term & ~re
    if re.any():
        ids = np.nonzero(re)[0]
        env.reset(ids, f_now[ids] + SKIP)
        o = obs()
        hist = mx.where(mx.array(re)[:, None, None], o[:, None], hist)
    env.f = np.minimum(env.f, fcap)
    if done.all(): break
rel = lambda f: (m.names[int(m.clip_id_np[f])].replace("_subject", "_s"), int(f - m.clip_start_np[m.clip_id_np[f]]))
fs = sorted({rel(f) for f in first[first >= 0]})
ev = sorted({rel(f) for _, f in events})
hidden = [e for e in ev if e not in fs]
print(f"{sys.argv[1]} | first failures {int((first >= 0).sum())} worlds: {fs}")
print(f"   with restarts (skip {SKIP}): {len(events)} events; hidden behind an earlier failure: {hidden} | {time.time() - t0:.0f} s", flush=True)
