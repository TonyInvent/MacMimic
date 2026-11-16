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
    def __init__(self, nthread: int | None = None, clips: list[str] | None = None, backend: str = "mujoco"):
        self.kin = Kinematics()
        self.motion = MotionLib.from_dir(g1.MOTION_DIR, self.kin, g1.load_motion)
        self.layout = XiLayout()
        self.mdp = MDP(self.kin, self.motion, self.layout.dim)
        m = self.motion
        self.clips = np.arange(len(m.names)) if clips is None else np.array([m.names.index(c) for c in clips])
        self.N = len(self.clips)
        self.env = TrackingSim(self.N, m, self.mdp, self.layout, eval_mode=True, nthread=nthread, backend=backend)

    def evaluate(self, ck: dict, record: str | None = None) -> dict:
        m, env, N = self.motion, self.env, self.N
        a_ = ck["meta"]["args"]
        K, Hh = a_["goal_frames"], a_["hist"]
        networks.set_compute_dtype(mx.bfloat16 if a_.get("precision") == "bf16" else None)  # as trained (upstream: AMP)
        self.motion.goal_stride = a_.get("goal_stride", 1)
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


class BatchEvaluator:
    """G checkpoints evaluated at once: one simulator of 40 G worlds (CPU MuJoCo is deterministic per world, so each
    slice gives the result of evaluating its checkpoint alone), one compiled step running every policy on its slice.
    The policies' parameters are compile inputs: a new group of checkpoints does not recompile."""

    def __init__(self, G: int, nthread: int | None = None, backend: str = "mujoco"):
        self.G = G
        self.kin = Kinematics()
        self.motion = MotionLib.from_dir(g1.MOTION_DIR, self.kin, g1.load_motion)
        self.layout = XiLayout()
        self.mdp = MDP(self.kin, self.motion, self.layout.dim)
        self.C = len(self.motion.names)
        self.env = TrackingSim(self.C * G, self.motion, self.mdp, self.layout, eval_mode=True, nthread=nthread, backend=backend)
        self.pis, self.act_c, self.sig = None, None, None

    def _setup(self, a_: dict):
        sig = tuple(a_.get(k) for k in ("goal_frames", "hist", "enc_hidden", "dec_hidden", "pi_ref_skip", "precision", "goal_stride"))
        if sig == self.sig:
            return
        self.sig = sig
        networks.set_compute_dtype(mx.bfloat16 if a_.get("precision") == "bf16" else None)
        self.motion.goal_stride = a_.get("goal_stride", 1)
        m, C, K = self.motion, self.C, a_["goal_frames"]
        self.pis = [policy_from_args(a_, self.mdp.obs_dim, m.goal_dim(K), len(g1.JOINT_NAMES)) for _ in range(self.G)]

        def act(s, a_prev, xi, f, hist):
            o = self.mdp.obs(unflatten_state(s), a_prev, xi, f)
            hist = mx.concatenate((hist[:, 1:], o[:, None]), 1)
            g = m.goal(f, K)
            return mx.concatenate([pi(hist[i * C : (i + 1) * C], g[i * C : (i + 1) * C]) for i, pi in enumerate(self.pis)]), hist

        self.act_c = mx.compile(act, inputs=[pi.state for pi in self.pis])

    def evaluate(self, cks: list[dict]) -> list[dict]:
        assert 0 < len(cks) <= self.G
        m, env, C, G = self.motion, self.env, self.C, self.G
        a_ = cks[0]["meta"]["args"]
        self._setup(a_)
        for i in range(G):  # unused slots replay the last checkpoint (results dropped)
            self.pis[i].update(unflat(cks[min(i, len(cks) - 1)]["pi"]))
        N = C * G
        clip_of = np.arange(N) % C
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(np.arange(N), m.clip_start_np[clip_of].copy())
        clip_end = m.clip_end_np[clip_of]
        ret = np.zeros(N)
        s = env.state_flat()
        o0 = self.mdp.obs(unflatten_state(mx.array(s)), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)))
        hist = mx.broadcast_to(o0[:, None], (N, a_["hist"], o0.shape[-1]))
        while not env.done_eval.all():
            s = env.state_flat()
            a, hist = self.act_c(mx.array(s), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)), hist)
            alive = ~env.done_eval.copy()
            out = env.step(np.array(a))
            ret += np.where(alive, out["r"], 0.0)
            env.f = np.minimum(env.f, clip_end - 1)
        clip_len = m.clip_len_np[clip_of]
        res = []
        for i in range(len(cks)):
            sl = slice(i * C, (i + 1) * C)
            norm = ret[sl] / ((clip_len[sl] - 1) * MAX_STEP_REWARD)
            failed = env.failed_eval[sl]
            res.append({
                "norm_return": float(norm.mean()),
                "norm_return_framew": float(ret[sl].sum() / ((clip_len[sl] - 1).sum() * MAX_STEP_REWARD)),
                "n_failed": int(failed.sum()),
                "failed": [m.names[c] for c in np.nonzero(failed)[0]],
                "per_clip": {m.names[c]: round(float(norm[c]), 4) for c in range(C)},
            })  # fmt: skip
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
    p.add_argument("--sim_backend", type=str, default="mujoco", choices=["mujoco", "superdex"],
                   help="simulator to evaluate in (superdex: cross-simulator evaluation; --threads worker processes)")
    p.add_argument("--clips", type=str, default=None, help="comma-separated clip names (default: all 40)")
    p.add_argument("--out", type=str, default="eval.csv", help="result file in the run directory")
    p.add_argument("--batch", type=int, default=1, help="evaluate this many checkpoints at once (all 40 clips each)")
    p.add_argument("--min_it", type=int, default=0, help="only checkpoints from this iteration on")
    a = p.parse_args()
    if a.batch > 1:
        assert a.run_dir and not a.clips and not a.ckpt
        return eval_run_batched(a)
    ev = Evaluator(nthread=a.threads, clips=a.clips.split(",") if a.clips else None, backend=a.sim_backend)
    if a.ckpt:
        res = ev.evaluate(load_tree(a.ckpt), a.record)
        print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
        return
    out_csv = os.path.join(a.run_dir, a.out)
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
            json.dump(res, open(c + "_" + a.out.removesuffix(".csv") + ".json", "w"), indent=1)
            print(f"[eval] it {it} ({meta['wall_time'] / 60:.1f} min): norm return {res['norm_return']:.4f}, "
                  f"failed {res['n_failed']}/{ev.N} ({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
            done.add(it)
        if not a.watch or (finished and not todo):
            break
        time.sleep(30)


def eval_run_batched(a):
    out_csv = os.path.join(a.run_dir, a.out)
    done = set()
    if os.path.exists(out_csv):
        done = {int(line.split(",")[0]) for line in open(out_csv).read().strip().splitlines()[1:]}
    else:
        open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
    todo = [c for c in ckpt_paths(a.run_dir)[:: a.every] if int(os.path.basename(c)[5:11]) not in done
            and int(os.path.basename(c)[5:11]) >= a.min_it]  # fmt: skip
    if not todo:
        return
    ev = BatchEvaluator(min(a.batch, len(todo)), nthread=a.threads, backend=a.sim_backend)
    rows = []
    for k in range(0, len(todo), ev.G):
        group = todo[k : k + ev.G]
        cks = [load_tree(c) for c in group]
        t0 = time.time()
        for c, ck, res in zip(group, cks, ev.evaluate(cks)):
            it, meta = int(os.path.basename(c)[5:11]), ck["meta"]
            rows.append((it, f"{it},{meta['wall_time'] / 3600:.4f},{meta['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},"
                             f"{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"))  # fmt: skip
            json.dump(res, open(c + "_" + a.out.removesuffix(".csv") + ".json", "w"), indent=1)
            print(f"[eval] it {it} ({meta['wall_time'] / 60:.1f} min): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/40", flush=True)
        print(f"[eval] group of {len(group)} checkpoints in {time.time() - t0:.0f} s", flush=True)
    with open(out_csv, "a") as f:
        for _, line in sorted(rows):
            f.write(line)


if __name__ == "__main__":
    main()
