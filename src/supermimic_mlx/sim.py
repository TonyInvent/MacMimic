"""Batched motion-tracking simulator on CPU MuJoCo (mujoco.batch), replacing upstream's mjlab / MuJoCo Warp.

Each world is one row of a mujoco.batch.Batch: its integration state (qpos, qvel, controls and the solver warm start,
which persists across steps and is cleared at a reset, as in MuJoCo Warp) and its own domain-randomized model fields
(friction of the robot geoms, torso CoM). A control step holds the PD setpoint for 4 physics substeps on the
batch's thread pool. The analytic reward / termination of :mod:`supermimic_mlx.mdp` label the transitions (on the
GPU, MLX), and episodes restart at reference-motion states drawn by return-based bin sampling (SuperMimic paper
Appendix D.1).
"""

from __future__ import annotations

import os
import time

import mlx.core as mx
import mujoco
import numpy as np
from mujoco.batch import Batch

from . import nprot
from .mdp import MAX_STEP_REWARD, MDP, unflatten_state
from .motion import MotionLib
from .robots import active as robot
from .scene import DECIMATION, Index, build_model


class XiLayout:
    """Slices of the sim context xi and its normalization (center, half-range) for the networks."""

    def __init__(self, J: int = len(robot.JOINT_NAMES)):
        self.slices = {"friction": slice(0, 1), "joint_offset": slice(1, 1 + J), "com": slice(1 + J, 4 + J)}
        lo, hi = robot.FRICTION_RANGE
        self.center = np.concatenate(([(lo + hi) / 2], np.zeros(J), np.zeros(3))).astype(np.float32)
        self.scale = np.concatenate(([(hi - lo) / 2], np.full(J, robot.JOINT_OFFSET), np.full(3, 0.05))).astype(np.float32)
        self.dim = 4 + J
        self.center_mx, self.scale_mx = mx.array(self.center), mx.array(self.scale)

    def normalize(self, xi: mx.array) -> mx.array:
        return (xi - self.center_mx) / self.scale_mx


def _bin_ema(stat: np.ndarray, bins: np.ndarray, x: np.ndarray, old: float) -> np.ndarray:
    """EMA of per-bin statistics with repeated bins: samples of a bin are averaged, then one EMA step per bin."""
    sums = np.bincount(bins, weights=x, minlength=len(stat))
    cnt = np.bincount(bins, minlength=len(stat))
    return np.where(cnt > 0, old * stat + (1 - old) * sums / np.maximum(cnt, 1), stat)


def _index_put_ema(stat: np.ndarray, bins: np.ndarray, x: np.ndarray, old: float) -> np.ndarray:
    """torch index_put_ of old * stat[b] + (1 - old) * x with duplicate bins (upstream without --bin_fix): one
    of the episodes of a bin wins (here: the last, as NumPy assignment does)."""
    out = stat.copy()
    out[bins] = old * stat[bins] + (1 - old) * x
    return out


class TrackingSim:
    def __init__(self, num_envs: int, motion: MotionLib, mdp: MDP, layout: XiLayout, gamma: float = 0.99,
                 eval_mode: bool = False, max_episode_steps: int = 500, reset_noise: bool = True,
                 randomize: bool | None = None, fail_reset: float = 0.0, bin_fix: bool = False, seed: int = 0,
                 nthread: int | None = None, backend: str = "mujoco"):  # fmt: skip
        """``backend``: "mujoco" (CPU MuJoCo, mujoco.batch) or "superdex" (SuperDex Physics, supermimic_mlx.sdx;
        ``nthread`` worker processes)."""
        self.N, self.J = num_envs, len(robot.JOINT_NAMES)
        self.motion, self.mdp, self.layout, self.gamma = motion, mdp, layout, gamma
        self.eval_mode = eval_mode
        self.randomize = (not eval_mode) if randomize is None else randomize
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.rng = np.random.default_rng(seed + (7 if eval_mode else 0))
        self.model = build_model()
        self.ix = Index(self.model)
        nthread = nthread if nthread is not None else int(os.environ.get("SIM_THREADS", os.cpu_count() or 8))
        self.backend = backend
        if backend == "superdex":
            from .sdx import SdxBatch

            self.batch = SdxBatch(self.model, num_envs, nthread=min(nthread, num_envs))
        else:
            self.batch = Batch(self.model, num_envs, nthread=min(nthread, max(1, num_envs // 8)))
        self.qpos, self.qvel, self.ctrl = self.batch.bind("qpos"), self.batch.bind("qvel"), self.batch.bind("ctrl")
        self.default_q = robot.default_joint_pos()
        self.alpha = robot.action_scales()
        # ---- sim context xi: friction (once per env), joint offsets (per episode), torso CoM offset (once per env) ----
        self.xi = np.zeros((num_envs, layout.dim), dtype=np.float32)
        sl = layout.slices
        if self.randomize:
            fr = self.rng.uniform(*robot.FRICTION_RANGE, size=num_envs).astype(np.float32)
            self.xi[:, sl["friction"]] = fr[:, None]
            gf = self.batch.expand("geom_friction")
            gf[:, self.ix.robot_geoms, 0] = fr[:, None]
            # The ground keeps its default friction (1.0): upstream sets the bodyid-0 geoms to the range's low end, but
            # mjlab's plane belongs to the "terrain" body, so that set is empty. MuJoCo takes the max of a contact pair.
            com = np.stack([self.rng.uniform(lo, hi, num_envs) for lo, hi in robot.COM_RANGE], -1).astype(np.float32)
            self.xi[:, sl["com"]] = com
            ipos = self.batch.expand("body_ipos")
            ipos[:, self.ix.com_body] = self.model.body_ipos[self.ix.com_body] + com
        else:
            self.xi[:, sl["friction"]] = 1.0
        # ---- episode bookkeeping ----
        N, nb = num_envs, motion.num_bins
        self.f = np.zeros(N, np.int64)
        self.ep_step = np.zeros(N, np.int64)
        self.a_prev = np.zeros((N, self.J), np.float32)
        self.start_bin = np.zeros(N, np.int64)
        self.ep_disc_ret = np.zeros(N, np.float64)
        self.ep_ret = np.zeros(N, np.float64)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = np.zeros(nb)
        self.bin_w_avg = np.full(nb, 1.0 / nb)
        self.fail_reset, self.bin_fix = fail_reset, bin_fix
        self.focus_q, self.focus_share = None, 0.0
        self.bin_fail = np.zeros(nb)
        prev = np.arange(nb) - 1
        bc = motion.bin_clip_np
        self.bin_prev = np.where((prev >= 0) & (bc[np.maximum(prev, 0)] == bc), prev, prev + 1)
        self.done_eval = np.zeros(N, bool)
        self.failed_eval = np.zeros(N, bool)
        self._reward_fn = mx.compile(self._reward_term)
        self._reward_fn_cpu = mx.compile(self._reward_term)  # traced separately on the CPU stream (compiled graphs keep their stream)
        self.cpu = False  # reward / termination heads on the CPU (train.py --actor_device cpu)
        self.command_outcome = None
        self.term_only = False  # termination only, in numpy, no reward (train.py's probe sweep: no GPU round trip)
        self.term_prob = 1.0  # training worlds: a world past the termination thresholds ends with this probability per step
        self._ref_z = np.array(motion.root_p[:, 2], dtype=np.float32)
        self._ref_r22 = np.array(motion.root_R[:, 2, 2], dtype=np.float32)
        self.t_physics = 0.0  # accumulated seconds in batch.step (profiling)
        self.reset(np.arange(N))

    # ------------------------------------------------------------------ state
    def state_flat(self) -> np.ndarray:
        """(N, 13 + 2J) float32: root position, quaternion (wxyz, w >= 0), world linear / angular velocity, q, qd."""
        qp, qv = self.qpos, self.qvel
        quat = qp[:, 3:7] / np.linalg.norm(qp[:, 3:7], axis=-1, keepdims=True)
        R = nprot.quat_to_mat(quat)
        w = np.einsum("nij,nj->ni", R, qv[:, 3:6])  # the free joint's angular velocity is in the root frame
        quat = quat * np.where(quat[:, :1] < 0, -1.0, 1.0)
        return np.concatenate((qp[:, :3], quat, qv[:, :3], w, qp[:, 7:], qv[:, 6:]), -1).astype(np.float32)

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int):
        m = self.motion
        z = -25.0 * self.bin_G / self.G_max
        soft = np.exp(z - z.max())
        soft /= soft.sum()
        self.bin_w_avg = self.bin_w_avg * 0.999 + 0.001 * soft
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        if self.fail_reset > 0:
            fail = np.bincount(self.bin_prev, weights=self.bin_fail, minlength=m.num_bins)
            fail = fail / max(fail.sum(), 1e-6)
            probs = (1 - self.fail_reset) * probs + self.fail_reset * (fail if fail.sum() > 0 else probs)
        if self.focus_q is not None and self.focus_share > 0:
            probs = (1 - self.focus_share) * probs + self.focus_share * self.focus_q
        if getattr(m, "reset_clip_mass", None) is not None:
            from .r1_reference import with_clip_masses
            probs = with_clip_masses(probs, m.bin_clip_np, m.reset_clip_mass)
        bins = self.rng.choice(m.num_bins, size=n, p=probs / probs.sum())
        start = m.bin_start_np[bins]
        end = np.minimum(start + 50, m.clip_end_np[m.bin_clip_np[bins]])  # keep >= 1 step before the clip end
        f = start + (self.rng.random(n) * (end - start)).astype(np.int64)
        return f, bins

    def set_focus(self, q: np.ndarray, share: float):
        self.focus_q, self.focus_share = q, share

    def reset(self, ids: np.ndarray, frames: np.ndarray | None = None):
        n = len(ids)
        if n == 0:
            return
        m = self.motion
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = m.frame_bin_np[frames]
        p, R, v, w, q, qd = (x.astype(np.float64) for x in m.ref_state_np(frames))
        if self.reset_noise:
            u = lambda half: np.stack([self.rng.uniform(-h, h, n) for h in half], -1)  # noqa: E731
            p = p + u(robot.RESET_POS)
            R = nprot.so3_exp(u(robot.RESET_ROT)) @ R
            v = v + u(robot.RESET_LIN_VEL)
            w = w + u(robot.RESET_ANG_VEL)
            q = q + self.rng.uniform(-robot.RESET_JOINT_POS, robot.RESET_JOINT_POS, q.shape)
        q = np.clip(q, self.mdp.kin.q_lo_np, self.mdp.kin.q_hi_np)
        # MuJoCo reset first (model defaults; clears controls and the solver warm start), then the writes
        self.batch.reset(ids, stage=mujoco.mjtStage.mjSTAGE_NONE)
        quat = nprot.mat_to_quat(R)
        qp = np.concatenate((p, quat, q), -1)
        qv = np.concatenate((v, np.einsum("nji,nj->ni", R, w), qd), -1)  # angular velocity into the new root frame
        self.qpos[ids] = qp
        self.qvel[ids] = qv
        self.ctrl[ids] = q  # position targets = the reset joint positions
        self.f[ids] = frames
        self.ep_step[ids] = 0
        self.a_prev[ids] = 0.0
        self.start_bin[ids] = bins
        self.ep_disc_ret[ids] = 0.0
        self.ep_ret[ids] = 0.0
        if self.randomize:
            self.xi[ids, self.layout.slices["joint_offset"]] = self.rng.uniform(-robot.JOINT_OFFSET, robot.JOINT_OFFSET, (n, self.J))

    # ------------------------------------------------------------------ observation
    def observe(self, s_flat: np.ndarray | None = None) -> mx.array:
        S = unflatten_state(mx.array(self.state_flat() if s_flat is None else s_flat))
        o = self.mdp.obs(S, mx.array(self.a_prev), mx.array(self.xi), mx.array(self.f.astype(np.int32)))
        return self.mdp.add_noise(o) if self.randomize else o

    # ------------------------------------------------------------------ step
    def _reward_term(self, s1, f1, a, a_prev):
        S1 = unflatten_state(s1)
        return self.mdp.reward(S1, f1, a, a_prev), self.mdp.termination(S1, f1)

    def termination_np(self, s1: np.ndarray, f1: np.ndarray) -> np.ndarray:
        """mdp.termination in numpy: root height / tilt (R[2, 2] of the normalized wxyz quaternion) vs frame f1."""
        q = s1[:, 3:7] / np.linalg.norm(s1[:, 3:7], axis=-1, keepdims=True)
        r22 = 1 - 2 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
        f1 = np.minimum(f1, len(self._ref_z) - 1)
        cfg = self.mdp.cfg
        return (np.abs(self._ref_z[f1] - s1[:, 2]) > cfg.term_height) | (np.abs(self._ref_r22[f1] - r22) > cfg.term_tilt)

    def step(self, a) -> dict:
        out = self.step_core(a)
        if self.eval_mode:
            self.failed_eval |= out["term"] & ~self.done_eval
            self.done_eval |= out["done"]
            return out
        return self.step_tail(out)

    def step_core(self, a, s: np.ndarray | None = None) -> dict:
        """One control step: PD setpoint held for DECIMATION physics substeps, reward / termination of the reached
        state against reference frame f + 1. ``s``: the current flat state if already computed."""
        a_np = np.asarray(a, dtype=np.float32)
        s = self.state_flat() if s is None else s
        f, xi, a_prev, ep_step = self.f.copy(), self.xi.copy(), self.a_prev.copy(), self.ep_step.copy()
        bad_a = ~np.isfinite(a_np).all(-1)
        target = self.default_q + xi[:, 1 : 1 + self.J] + self.alpha * a_np
        if bad_a.any():  # a non-finite action is not handed to MuJoCo (it would zero the forces and warn)
            target = np.where(bad_a[:, None], self.default_q + xi[:, 1 : 1 + self.J], target)
        self.ctrl[:] = target
        qp0, qv0 = self.qpos.copy(), self.qvel.copy()
        t0 = time.perf_counter()
        self.batch.step(nstep=DECIMATION)
        self.t_physics += time.perf_counter() - t0
        s1 = self.state_flat()
        f1 = f + 1
        if self.term_only:
            r, term = np.zeros(len(s1)), self.termination_np(s1, f1)
        elif self.cpu:
            with mx.stream(mx.cpu):
                r, term = self._reward_fn_cpu(mx.array(s1), mx.array(f1.astype(np.int32)), mx.array(a_np), mx.array(a_prev))
                mx.eval(r, term)
        else:
            r, term = self._reward_fn(mx.array(s1), mx.array(f1.astype(np.int32)), mx.array(a_np), mx.array(a_prev))
        r, term = np.asarray(r, dtype=np.float64), np.asarray(term)
        if self.command_outcome is not None:
            r, term = self.command_outcome(s1, a_np, a_prev, r, term)
        if self.term_prob < 1.0 and not self.eval_mode:
            term = term & (self.rng.random(len(term)) < self.term_prob)
        # a diverged world (non-finite state; MuJoCo's auto-reset is off, see scene.apply_options), a non-finite action
        # or reward ends the episode as a failure; the world is rolled back to its last finite state, so the stored
        # transition is a standstill and an evaluation world stays frozen (failed) instead of stepping NaNs
        bad = ~np.isfinite(s1).all(-1) | bad_a | ~np.isfinite(r)
        if bad.any():
            term = term | bad
            r = np.where(bad, 0.0, r)
            s1 = np.where(bad[:, None], s, s1)
            self.qpos[bad], self.qvel[bad] = qp0[bad], qv0[bad]
            a_np = np.where(bad_a[:, None], 0.0, a_np).astype(np.float32)
        clip_end = self.motion.clip_end_np[self.motion.clip_id_np[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {"s": s, "a": a_np, "a_prev": a_prev, "f": f.astype(np.int32), "s1": s1.astype(np.float32), "r": r,
               "term": term, "done": done, "ep_step": ep_step.astype(np.int32), "xi": xi, "cut": done}  # fmt: skip
        self.ep_disc_ret += (self.gamma**ep_step) * r
        self.ep_ret += r
        self.f, self.ep_step, self.a_prev = f1, ep_step + 1, a_np.copy()
        return out

    def step_tail(self, out: dict) -> dict:
        """Episode ends: bin statistics, then resets of the ended episodes."""
        done, term, f = out["done"], out["term"], out["f"]
        ids = np.nonzero(done)[0]
        if len(ids) > 0:
            Tn = self.ep_step[ids].astype(np.float64)
            boot = np.where(term[ids], 0.0, (self.gamma**Tn) * (self.ep_ret[ids] / Tn) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            ema = _bin_ema if self.bin_fix else _index_put_ema
            self.bin_G = ema(self.bin_G, b, G, 0.8)
            if self.fail_reset > 0:
                fb = self.motion.frame_bin_np[np.minimum(f[ids], self.motion.num_frames - 1)]
                self.bin_fail = ema(self.bin_fail * 0.999, fb, term[ids].astype(np.float64), 0.9)
            out["ep_len"] = Tn
            out["ep_ret_norm"] = self.ep_ret[ids] / (Tn * MAX_STEP_REWARD)
            self.reset(ids)
        return out
