"""Checkpoint evaluation (SuperMimic paper Appendix F.1) on all motions of the library, as upstream evaluate.py.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no observation or action
noise. A motion fails if the rollout triggers the training termination. Normalized return is the undiscounted reward
sum divided by its maximum attainable value. CPU MuJoCo is deterministic, so upstream's replicas (which average
MuJoCo Warp's run-to-run variation) are not needed.

  python -m supermimic_mlx.evaluate --run_dir logs/<run> [--watch]      # every checkpoint -> eval.csv
  python -m supermimic_mlx.evaluate --ckpt logs/<run>/ckpt_003500 [--record traj.npz]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import mlx.core as mx
import numpy as np

from . import networks
from .checkpoint import load_tree
from .kinematics import Kinematics
from .mdp import MAX_STEP_REWARD, MDP, unflatten_state
from .motion import MotionLib
from .networks import policy_from_args
from .optim import unflat
from .robots import g1
from .sim import TrackingSim, XiLayout


class Evaluator:
    def __init__(self, nthread: int | None = None, clips: list[str] | None = None):
        self.kin = Kinematics()
        self.motion = MotionLib.from_dir(g1.MOTION_DIR, self.kin, g1.load_motion)
        self.layout = XiLayout()
        self.mdp = MDP(self.kin, self.motion, self.layout.dim)
        m = self.motion
        self.clips = np.arange(len(m.names)) if clips is None else np.array([m.names.index(c) for c in clips])
        self.N = len(self.clips)
        self.env = TrackingSim(self.N, m, self.mdp, self.layout, eval_mode=True, nthread=nthread)

    def evaluate(self, ck: dict, record: str | None = None) -> dict:
        m, env, N = self.motion, self.env, self.N
        a_ = ck["meta"]["args"]
        K, Hh = a_["goal_frames"], a_["hist"]
        networks.set_compute_dtype(mx.bfloat16 if a_.get("precision") == "bf16" else None)  # as trained (upstream: AMP)
        pi = policy_from_args(a_, self.mdp.obs_dim, m.goal_dim(K), len(g1.JOINT_NAMES))
        pi.update(unflat(ck["pi"]))

        def act(s, a_prev, xi, f, hist):
            o = self.mdp.obs(unflatten_state(s), a_prev, xi, f)
            hist = mx.concatenate((hist[:, 1:], o[:, None]), 1)
            return pi(hist, m.goal(f, K)), hist

        act_c = mx.compile(act)
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(np.arange(N), m.clip_start_np[self.clips].copy())
        clip_end = m.clip_end_np[self.clips]
        ret, steps = np.zeros(N), np.zeros(N)
        s = env.state_flat()
        o0 = self.mdp.obs(unflatten_state(mx.array(s)), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)))
        hist = mx.broadcast_to(o0[:, None], (N, Hh, o0.shape[-1]))
        traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
        while not env.done_eval.all():
            # the first history is H copies of o_0; the first call shifts in o_0 again (upstream's first step)
            s = env.state_flat()
            a, hist = act_c(mx.array(s), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)), hist)
            a = np.array(a)
            alive = ~env.done_eval.copy()
            if record:
                traj["root_p"].append(s[:, :3]), traj["root_quat"].append(s[:, 3:7]), traj["q"].append(s[:, 13 : 13 + 29])
                traj["f"].append(env.f.copy()), traj["alive"].append(alive)
            out = env.step(a)
            ret += np.where(alive, out["r"], 0.0)
            steps += alive
            env.f = np.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f + 1 in range
        clip_len = m.clip_len_np[self.clips]
        norm = ret / ((clip_len - 1) * MAX_STEP_REWARD)
        names = [m.names[c] for c in self.clips]
        res = {
            "norm_return": float(norm.mean()),
            "norm_return_framew": float(ret.sum() / ((clip_len - 1).sum() * MAX_STEP_REWARD)),
            "n_failed": int(env.failed_eval.sum()),
            "failed": [names[i] for i in np.nonzero(env.failed_eval)[0]],
            "per_clip": {names[i]: round(float(norm[i]), 4) for i in range(N)},
        }
        if record:
            np.savez_compressed(record, names=np.array(names), **{k: np.stack(v) for k, v in traj.items()},
                                ref_root_p=m.np["root_p"], ref_root_R=m.np["root_R"], ref_q=m.np["q"],
                                clip_start=m.clip_start_np[self.clips], clip_len=clip_len)  # fmt: skip
        return res


def ckpt_paths(run_dir: str) -> list[str]:
    return sorted(p.removesuffix(".safetensors") for p in glob.glob(os.path.join(run_dir, "ckpt_*.safetensors")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--watch", action="store_true", help="keep evaluating new checkpoints until the run writes last.json")
    p.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
    p.add_argument("--record", type=str, default=None)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    ev = Evaluator(nthread=a.threads)
    if a.ckpt:
        res = ev.evaluate(load_tree(a.ckpt), a.record)
        print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
        return
    out_csv = os.path.join(a.run_dir, "eval.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {int(line.split(",")[0]) for line in open(out_csv).read().strip().splitlines()[1:]}
    else:
        open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
    while True:
        finished = os.path.exists(os.path.join(a.run_dir, "done"))
        todo = [c for c in ckpt_paths(a.run_dir)[:: a.every] if int(os.path.basename(c)[5:11]) not in done]
        for c in todo:
            it = int(os.path.basename(c)[5:11])
            ck = load_tree(c)
            t0 = time.time()
            res = ev.evaluate(ck)
            meta = ck["meta"]
            line = (f"{it},{meta['wall_time'] / 3600:.4f},{meta['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},"
                    f"{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n")  # fmt: skip
            open(out_csv, "a").write(line)
            json.dump(res, open(c + "_eval.json", "w"), indent=1)
            print(f"[eval] it {it} ({meta['wall_time'] / 60:.1f} min): norm return {res['norm_return']:.4f}, "
                  f"failed {res['n_failed']}/{ev.N} ({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
            done.add(it)
        if not a.watch or (finished and not todo):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
