"""Sim-to-sim gap: the same reference states and open-loop reference joint targets in CPU MuJoCo and SuperDex Physics.

    uv run --extra superdex python tools/compare_sims.py --worlds 4 --steps 50
"""

import argparse
import time

import numpy as np

from supermimic_mlx.kinematics import Kinematics
from supermimic_mlx.mdp import MDP
from supermimic_mlx.motion import MotionLib
from supermimic_mlx.robots import g1
from supermimic_mlx.sim import TrackingSim, XiLayout

p = argparse.ArgumentParser()
p.add_argument("--worlds", type=int, default=4)
p.add_argument("--steps", type=int, default=50, help="control steps (20 ms)")
p.add_argument("--workers", type=int, default=1)
a = p.parse_args()
kin = Kinematics()
motion = MotionLib.from_dir(g1.MOTION_DIR, kin, g1.load_motion)
lay = XiLayout()
mdp = MDP(kin, motion, lay.dim)
frames = motion.clip_start_np[np.arange(a.worlds) % len(motion.names)] + 200
sims = {}
for be, nt in (("mujoco", 1), ("superdex", a.workers)):
    t0 = time.time()
    sims[be] = TrackingSim(a.worlds, motion, mdp, lay, eval_mode=True, nthread=nt, backend=be)
    sims[be].reset(np.arange(a.worlds), frames.copy())
    print(f"{be}: setup {time.time() - t0:.1f} s", flush=True)
s0 = {be: e.state_flat() for be, e in sims.items()}
print("state after reset identical:", np.abs(s0["mujoco"] - s0["superdex"]).max() < 1e-5, flush=True)
traj = {be: [] for be in sims}
for be, e in sims.items():
    t0 = time.time()
    for t in range(a.steps):
        act = (motion.np["q"][np.minimum(e.f + 1, motion.num_frames - 1)] - e.default_q) / e.alpha
        out = e.step(act)
        traj[be].append(e.state_flat().copy())
    print(f"{be}: {1000 * (time.time() - t0) / a.steps:.1f} ms per control step, failed {e.failed_eval.sum()}/{a.worlds}", flush=True)
A, B = np.stack(traj["mujoco"]), np.stack(traj["superdex"])
for t in (0, 4, 9, 24, a.steps - 1):
    if t < a.steps:
        d = np.linalg.norm(A[t, :, :3] - B[t, :, :3], axis=-1)
        dq = np.abs(A[t, :, 13:42] - B[t, :, 13:42]).mean(-1)
        print(f"t {0.02 * (t + 1):.2f} s: root position gap mean {100 * d.mean():.1f} cm (max {100 * d.max():.1f}), "
              f"joint gap {np.degrees(dq.mean()):.1f} deg | root z mujoco {A[t, :, 2].mean():.3f} superdex {B[t, :, 2].mean():.3f}")
