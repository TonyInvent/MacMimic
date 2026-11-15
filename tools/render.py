"""Render a policy on one clip under evaluation conditions: the simulated G1 plus the reference motion as a blue ghost.

    uv run python tools/render.py --ckpt logs/blinkmimic_s0/ckpt_003500 --clip fallAndGetUp1_subject1 --seconds 20 --out out.mp4
"""

import argparse
import subprocess

import mlx.core as mx
import mujoco
import numpy as np

from supermimic_mlx.checkpoint import load_tree
from supermimic_mlx.evaluate import Evaluator
from supermimic_mlx.mdp import unflatten_state
from supermimic_mlx.networks import policy_from_args
from supermimic_mlx.optim import unflat
from supermimic_mlx.robots import g1
from supermimic_mlx.scene import build_model

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--clip", default="dance1_subject1")
p.add_argument("--start_s", type=float, default=0.0, help="start this far into the clip (on the reference)")
p.add_argument("--seconds", type=float, default=15.0)
p.add_argument("--out", default="render.mp4")
p.add_argument("--width", type=int, default=640)
p.add_argument("--height", type=int, default=480)
a = p.parse_args()

ev = Evaluator(nthread=1, clips=[a.clip])
m, env, mdp = ev.motion, ev.env, ev.mdp
ck = load_tree(a.ckpt)
ar = ck["meta"]["args"]
K, Hh = ar["goal_frames"], ar["hist"]
pi = policy_from_args(ar, mdp.obs_dim, m.goal_dim(K), len(g1.JOINT_NAMES))
pi.update(unflat(ck["pi"]))
c = m.names.index(a.clip)
f0 = int(m.clip_start_np[c] + round(a.start_s / 0.02))
env.reset(np.array([0]), np.array([f0]))
s = env.state_flat()
o = mdp.obs(unflatten_state(mx.array(s)), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)))
hist = mx.broadcast_to(o[:, None], (1, Hh, o.shape[-1]))
qs, fs, fails = [], [], []
for t in range(int(a.seconds / 0.02)):
    if env.f[0] + 1 > m.clip_end_np[c]:
        break
    s = env.state_flat()
    o = mdp.obs(unflatten_state(mx.array(s)), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)))
    hist = mx.concatenate((hist[:, 1:], o[:, None]), 1)
    act = np.array(pi(hist, m.goal(mx.array(env.f.astype(np.int32)), K)))
    qs.append(env.qpos[0].copy()), fs.append(int(env.f[0]))
    out = env.step(act)
    fails.append(bool(env.failed_eval[0]))
print(f"{len(qs)} steps, failed: {fails[-1] if fails else None}")

model = build_model(visual=True)
data, ref = mujoco.MjData(model), mujoco.MjData(model)
r = mujoco.Renderer(model, a.height, a.width)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 3.2, -15, 135
vopt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}",
                           "-r", "50", "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", a.out], stdin=subprocess.PIPE)  # fmt: skip
for q, f, failed in zip(qs, fs, fails):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    ref.qpos[:3], ref.qpos[7:] = m.np["root_p"][f], m.np["q"][f]
    rq = np.zeros(4)
    mujoco.mju_mat2Quat(rq, m.np["root_R"][f].astype(np.float64).flatten())
    ref.qpos[3:7] = rq
    mujoco.mj_forward(model, ref)
    cam.lookat[:] = data.qpos[:3]
    cam.lookat[2] = 0.8
    r.update_scene(data, cam)
    n0 = r.scene.ngeom
    mujoco.mjv_addGeoms(model, ref, vopt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, r.scene)
    for i in range(n0, r.scene.ngeom):
        r.scene.geoms[i].rgba[:] = (0.3, 0.5, 1.0, 0.35)
    if failed:
        for i in range(n0):
            r.scene.geoms[i].rgba[3] *= 0.5
    ffmpeg.stdin.write(r.render().tobytes())
ffmpeg.stdin.close()
ffmpeg.wait()
print("wrote", a.out)
