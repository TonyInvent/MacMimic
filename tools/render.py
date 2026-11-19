"""Render a policy on one clip under evaluation conditions: the simulated G1 plus the reference motion as a blue ghost.

    uv run python tools/render.py --ckpt logs/blinkmimic_s0/ckpt_003500 --clip fallAndGetUp1_subject1 --seconds 20 --out out.mp4
    # as evaluated: simulate from the clip's first frame, record 6 s from 104 s on
    uv run python tools/render.py --ckpt <ckpt> --clip dance1_subject1 --from_clip_start --start_s 104 --seconds 6 --out cartwheel.mp4
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
from supermimic_mlx.scene import apply_options, scene_spec

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--clip", default="dance1_subject1")
p.add_argument("--start_s", type=float, default=0.0, help="start this far into the clip (on the reference)")
p.add_argument("--from_clip_start", action="store_true",
               help="simulate from the clip's first frame, as the evaluation does, and record from --start_s on")
p.add_argument("--azimuth", type=float, default=135.0)
p.add_argument("--elevation", type=float, default=-15.0)
p.add_argument("--look_z", type=float, default=0.8, help="camera target height (m)")
p.add_argument("--ghost", default="overlay", choices=["overlay", "side", "none"],
               help="the reference motion: drawn where it is (overlay), next to the robot (side), or not at all")
p.add_argument("--look", default="plain", choices=["plain", "studio"], help="studio: dark gradient sky, grid floor, shadows")
p.add_argument("--distance", type=float, default=3.2)
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
m.goal_stride = ar.get("goal_stride", 1)
pi = policy_from_args(ar, mdp.obs_dim, m.goal_dim(K), len(g1.JOINT_NAMES))
pi.update(unflat(ck["pi"]))
c = m.names.index(a.clip)
f_rec = int(m.clip_start_np[c] + round(a.start_s / 0.02))  # first recorded frame
f0 = int(m.clip_start_np[c]) if a.from_clip_start else f_rec
env.reset(np.array([0]), np.array([f0]))
s = env.state_flat()
o = mdp.obs(unflatten_state(mx.array(s)), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)))
hist = mx.broadcast_to(o[:, None], (1, Hh, o.shape[-1]))
qs, fs, fails = [], [], []
for t in range(int(a.seconds / 0.02) + (f_rec - f0)):
    if env.f[0] + 1 > m.clip_end_np[c]:
        break
    s = env.state_flat()
    o = mdp.obs(unflatten_state(mx.array(s)), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)))
    hist = mx.concatenate((hist[:, 1:], o[:, None]), 1)
    act = np.array(pi(hist, m.goal(mx.array(env.f.astype(np.int32)), K)))
    rec = env.f[0] >= f_rec
    if rec:
        qs.append(env.qpos[0].copy()), fs.append(int(env.f[0]))
    out = env.step(act)
    if rec:
        fails.append(bool(env.failed_eval[0]))
    elif env.failed_eval[0]:
        print(f"failed at clip time {(env.f[0] - m.clip_start_np[c]) * 0.02:.2f} s, before the recorded window")
        break
print(f"{len(qs)} steps, failed: {fails[-1] if fails else None}")

def render_model(look: str) -> mujoco.MjModel:
    """The visual scene; ``studio`` adds a gradient sky, a grid floor and soft shadows (rendering only)."""
    spec = scene_spec(visual=True)
    if look == "studio":
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.16, 0.18, 0.22], rgb2=[0.02, 0.02, 0.03], width=512, height=3072)  # fmt: skip
        spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.13, 0.14, 0.16], rgb2=[0.11, 0.12, 0.14], width=512, height=512,
                         mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.24, 0.26, 0.3])  # fmt: skip
        mat = spec.add_material(name="grid", texrepeat=[1, 1], texuniform=True, reflectance=0.08)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        spec.geom("terrain").material = "grid"
        spec.visual.quality.shadowsize = 4096
        spec.visual.headlight.ambient = [0.35, 0.35, 0.38]
        spec.visual.headlight.diffuse = [0.55, 0.55, 0.58]
        spec.visual.headlight.specular = [0.1, 0.1, 0.1]
        spec.visual.rgba.haze = [0.05, 0.055, 0.07, 1.0]  # the horizon blends into the sky
    m_ = spec.compile()
    apply_options(m_)
    m_.vis.global_.offwidth, m_.vis.global_.offheight = max(a.width, 640), max(a.height, 480)  # offscreen buffer
    return m_


model = render_model(a.look)
data, ref = mujoco.MjData(model), mujoco.MjData(model)
r = mujoco.Renderer(model, a.height, a.width)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = a.distance, a.elevation, a.azimuth
az = np.deg2rad(a.azimuth)
side = np.array([-np.sin(az), np.cos(az)]) * (1.1 if a.ghost == "side" else 0.0)  # across the view
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
    if a.ghost == "side":  # the reference's pose and height, next to the robot
        ref.qpos[:2] = data.qpos[:2] + side
    mujoco.mj_forward(model, ref)
    cam.lookat[:2] = data.qpos[:2] + 0.5 * side
    cam.lookat[2] = a.look_z
    r.update_scene(data, cam)
    n0 = r.scene.ngeom
    if a.ghost != "none":
        mujoco.mjv_addGeoms(model, ref, vopt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, r.scene)
        for i in range(n0, r.scene.ngeom):
            r.scene.geoms[i].rgba[:] = (0.3, 0.55, 1.0, 0.45 if a.ghost == "side" else 0.35)
    if failed:
        for i in range(n0):
            r.scene.geoms[i].rgba[3] *= 0.5
    ffmpeg.stdin.write(r.render().tobytes())
ffmpeg.stdin.close()
ffmpeg.wait()
print("wrote", a.out)
