"""Check the state round-trip, MLX FK against MuJoCo's body frames, and the reward on reference playback."""

import argparse
import time

import mlx.core as mx
import mujoco
import numpy as np

from supermimic_mlx.kinematics import Kinematics
from supermimic_mlx.mdp import MDP, unflatten_state
from supermimic_mlx.motion import MotionLib
from supermimic_mlx.robots import g1
from supermimic_mlx.sim import TrackingSim, XiLayout

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
args = parser.parse_args()

kin = Kinematics()
t0 = time.time()
motion = MotionLib.from_dir(g1.MOTION_DIR, kin, g1.load_motion)
print(f"motion: {len(motion.names)} clips, {motion.num_frames} frames ({motion.num_frames * 0.02 / 3600:.2f} h), "
      f"bins {motion.num_bins}, load {time.time() - t0:.1f}s")  # fmt: skip
layout = XiLayout()
mdp = MDP(kin, motion, layout.dim)
N = args.num_envs
env = TrackingSim(N, motion, mdp, layout, eval_mode=True)
frames = motion.clip_start_np[np.arange(N) % len(motion.names)] + 100
env.reset(np.arange(N), frames)
s = env.state_flat()
S = [np.array(x) for x in unflatten_state(mx.array(s))]
ref = motion.ref_state_np(frames)
for n, a, b in zip(["p", "R", "v", "w", "q", "qd"], S, ref):
    print(f"roundtrip {n}: max err {np.abs(a - b).max():.2e}")
# FK vs MuJoCo body frames (one world at a time)
m = env.model
d = mujoco.MjData(m)
bids = [m.body("robot/" + b).id for b in g1.TRACKED_BODIES]
Sm = unflatten_state(mx.array(s))
bp, bR, bv, bw = (np.array(x) for x in kin.forward(*Sm))
ep, eR, ev, ew = 0.0, 0.0, 0.0, 0.0
for i in range(N):
    d.qpos[:], d.qvel[:] = env.qpos[i], env.qvel[i]
    mujoco.mj_forward(m, d)
    vel = np.zeros(6)
    for k, b in enumerate(bids):
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)  # world frame, at the body COM: [ang, lin]
        vel[3:] += np.cross(vel[:3], d.xpos[b] - d.xipos[b])  # -> at the body origin (FK's link frame)
        ep = max(ep, np.linalg.norm(bp[i, k] - d.xpos[b]))
        eR = max(eR, np.abs(bR[i, k] - d.xmat[b].reshape(3, 3)).max())
        ew = max(ew, np.linalg.norm(bw[i, k] - vel[:3]))
        ev = max(ev, np.linalg.norm(bv[i, k] - vel[3:]))
print(f"FK vs MuJoCo: pos {ep:.2e}  rot {eR:.2e}  linvel {ev:.2e}  angvel {ew:.2e}")
print("ref body vs FK(sim) pos err", float(np.abs(np.array(motion.body_p)[frames] - bp).max()))
# open-loop reference playback through the joint targets
t0 = time.time()
for t in range(150):
    a = (motion.np["q"][env.f + 1] - env.default_q) / env.alpha
    out = env.step(a)
    if t % 25 == 0:
        print(t, "norm reward %.3f" % (out["r"].mean() / 0.1), "failed frac", env.failed_eval.mean())
print(f"{150 / (time.time() - t0):.0f} control steps/s with {N} worlds")
