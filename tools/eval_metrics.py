"""Tracking-accuracy and contact metrics for a set of checkpoints (port of upstream BlinkMimic's tools/eval_metrics.py;
same rollout protocol as the evaluator: each clip once from its first frame, friction 1.0, no randomization or noise,
deterministic policy).

Per checkpoint, averaged over all alive control steps of all 40 clips:
  body_pos_cm   RMS error of the 14 tracked bodies (reference re-anchored to the robot root, as in the reward)
  body_ori_deg  RMS orientation error of the tracked bodies
  root_pos_cm   global root position error (drift)
  joint_deg     mean absolute joint-angle error
  slip_cm_s     horizontal foot speed while the simulated foot is in contact (ankle link < CONTACT_Z)
  ref_slip_cm_s the same on the reference motion (retargeting artifacts set a floor)
  contact_acc   fraction of foot-frames whose contact state matches the reference
  pen_mm        mean ankle-height deficit below the flat-foot height (ground penetration proxy)
plus normalized return and failed clips.

    uv run python tools/eval_metrics.py logs/<run_a>/ckpt_003500 logs/<run_b>/ckpt_003500 --out results/metrics
"""

import argparse
import json
import os

import mlx.core as mx
import numpy as np

from supermimic_mlx import networks
from supermimic_mlx.checkpoint import load_tree
from supermimic_mlx.kinematics import Kinematics
from supermimic_mlx.mdp import MAX_STEP_REWARD, MDP, unflatten_state
from supermimic_mlx.motion import MotionLib
from supermimic_mlx.networks import policy_from_args
from supermimic_mlx.optim import unflat
from supermimic_mlx.robots import g1
from supermimic_mlx.sim import TrackingSim, XiLayout

CONTACT_Z = 0.045  # ankle-roll link origin height below which a foot counts as in contact (flat foot: ~0.035 m)
FLAT_Z = 0.033

parser = argparse.ArgumentParser()
parser.add_argument("ckpts", nargs="+")
parser.add_argument("--out", type=str, default="results/metrics")
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--sim_backend", type=str, default="mujoco", choices=["mujoco", "superdex"])
args = parser.parse_args()

kin = Kinematics()
motion = MotionLib.from_dir(g1.MOTION_DIR, kin, g1.load_motion)
layout = XiLayout()
mdp = MDP(kin, motion, layout.dim)
C, J = len(motion.names), len(g1.JOINT_NAMES)
FEET = mdp.feet


def run(cks):
    M = len(cks)
    N = C * M
    env = TrackingSim(N, motion, mdp, layout, eval_mode=True, nthread=args.threads, backend=args.sim_backend)
    a0 = cks[0]["meta"]["args"]
    networks.set_compute_dtype(mx.bfloat16 if a0.get("precision") == "bf16" else None)
    K, Hh = a0["goal_frames"], a0["hist"]
    pis = []
    for ck in cks:
        pi = policy_from_args(ck["meta"]["args"], mdp.obs_dim, motion.goal_dim(K), J)
        pi.update(unflat(ck["pi"]))
        pis.append(pi)

    def act(s, a_prev, xi, f, hist):
        o = mdp.obs(unflatten_state(s), a_prev, xi, f)
        hist = mx.concatenate((hist[:, 1:], o[:, None]), 1)
        g = motion.goal(f, K)
        return mx.concatenate([pi(hist[m * C : (m + 1) * C], g[m * C : (m + 1) * C]) for m, pi in enumerate(pis)]), hist

    def metrics(s1, f1):
        S1 = unflatten_state(s1)
        _, t = mdp.reward(S1, f1, mx.zeros((N, J)), mx.zeros((N, J)), terms=True)
        bp, _, bv, _ = kin.forward(*S1)
        feet = mx.array(FEET)
        fz, fv = bp[:, feet, 2], mx.sqrt((bv[:, feet, :2] ** 2).sum(-1))
        rz = motion.body_p[f1][:, feet, 2]
        rv = mx.sqrt((motion.body_v[f1][:, feet, :2] ** 2).sum(-1))
        return t["e_body_pos"], t["e_body_ori"], mx.sqrt(t["e_root_pos"]), mx.abs(S1[4] - motion.q[f1]).mean(-1), fz, fv, rz, rv

    act_c, metrics_c = mx.compile(act), mx.compile(metrics)
    clip_of = np.arange(N) % C
    env.reset(np.arange(N), motion.clip_start_np[clip_of].copy())
    clip_end = motion.clip_end_np[clip_of]
    keys = ("n", "ret", "pos", "ori", "root", "joint", "slip", "slip_n", "ref_slip", "ref_slip_n", "c_ok", "c_n", "pen")
    acc = {k: np.zeros(N) for k in keys}
    s = env.state_flat()
    o0 = mdp.obs(unflatten_state(mx.array(s)), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)))
    hist = mx.broadcast_to(o0[:, None], (N, Hh, mdp.obs_dim))
    while not env.done_eval.all():
        s = env.state_flat()
        a, hist = act_c(mx.array(s), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)), hist)
        alive = (~env.done_eval).astype(np.float64)
        f1 = env.f + 1
        out = env.step(np.array(a))
        pos, ori, root, joint, fz, fv, rz, rv = (np.array(x, dtype=np.float64) for x in metrics_c(mx.array(env.state_flat()), mx.array(f1.astype(np.int32))))
        c_sim, c_ref = fz < CONTACT_Z, rz < CONTACT_Z
        acc["n"] += alive
        acc["ret"] += alive * out["r"]
        acc["pos"] += alive * pos
        acc["ori"] += alive * ori
        acc["root"] += alive * root
        acc["joint"] += alive * joint
        acc["slip"] += alive * (fv * c_sim).sum(-1)
        acc["slip_n"] += alive * c_sim.sum(-1)
        acc["ref_slip"] += alive * (rv * c_ref).sum(-1)
        acc["ref_slip_n"] += alive * c_ref.sum(-1)
        acc["c_ok"] += alive * (c_sim == c_ref).sum(-1)
        acc["c_n"] += alive * 2
        acc["pen"] += alive * np.maximum(FLAT_Z - fz, 0).sum(-1) / 2
        env.f = np.minimum(env.f, clip_end - 1)
    clip_len = motion.clip_len_np[clip_of]
    res = []
    for m in range(M):
        sl = slice(m * C, (m + 1) * C)
        sm = {k: v[sl].sum() for k, v in acc.items()}
        n = sm["n"]
        res.append({
            "norm_return": float((acc["ret"][sl] / ((clip_len[sl] - 1) * MAX_STEP_REWARD)).mean()),
            "n_failed": int(env.failed_eval[sl].sum()),
            "body_pos_cm": float(100 * (sm["pos"] / n) ** 0.5),
            "body_ori_deg": float(57.2958 * (sm["ori"] / n) ** 0.5),
            "root_pos_cm": float(100 * sm["root"] / n),
            "joint_deg": float(57.2958 * sm["joint"] / n),
            "slip_cm_s": float(100 * sm["slip"] / max(sm["slip_n"], 1)),
            "ref_slip_cm_s": float(100 * sm["ref_slip"] / max(sm["ref_slip_n"], 1)),
            "contact_acc": float(sm["c_ok"] / sm["c_n"]),
            "pen_mm": float(1000 * sm["pen"] / n),
        })  # fmt: skip
    return res


cks = [load_tree(p.removesuffix(".safetensors")) for p in args.ckpts]
res = run(cks)
cols = ["norm_return", "n_failed", "body_pos_cm", "body_ori_deg", "root_pos_cm", "joint_deg", "slip_cm_s", "ref_slip_cm_s", "contact_acc", "pen_mm"]
lines = ["| checkpoint | train h | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
for p, ck, r in zip(args.ckpts, cks, res):
    name = f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}"
    lines.append(f"| {name} | {ck['meta']['wall_time'] / 3600:.2f} | " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
table = "\n".join(lines)
print(table)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
open(args.out + ".md", "w").write(table + "\n")
json.dump({p: r for p, r in zip(args.ckpts, res)}, open(args.out + ".json", "w"), indent=1)
