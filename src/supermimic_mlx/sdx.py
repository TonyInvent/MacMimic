"""SuperDex Physics backend (github.com/facebookresearch/project_superdex), for fine-tuning in a second simulator.

SuperDex is a contact-first engine (penalty contacts on SDF colliders, fully implicit integration). It is ~15x slower
per world than MuJoCo, and several G1s in one scene slow its solver down drastically, so every world lives in its own
scene and the worlds are spread over worker processes (one engine thread each). ``SdxBatch`` mimics the part of
``mujoco.batch.Batch`` that ``TrackingSim`` uses: live (nsim, ...) ``qpos`` / ``qvel`` / ``ctrl`` arrays in MuJoCo's
layout (free joint: world position, quaternion wxyz, world linear velocity, *body-frame* angular velocity), per-world
``geom_friction`` / ``body_ipos`` fields, ``reset`` and ``step(nstep)``. Rows that the caller writes between steps
(resets, roll-backs) are detected and pushed to the workers.

The G1 is the URDF the motions were retargeted to, as in MuJoCo: SuperDex's runtime URDF loader drops primitive
colliders, so the 4 foot spheres per foot are added as welded child links (the shoulder cylinders, which do not touch
the ground in the motions, are not); joint armature is the joint inertia; the PD drives are SuperDex's implicit pose
controller (stiffness, damping, torque saturation = BeyondMimic's kp, kd, effort limit) with a zero velocity target,
like MuJoCo's position actuator.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from multiprocessing import shared_memory
from multiprocessing.connection import Client, Listener

import numpy as np

from . import nprot
from .robots import g1

FOOT_SPHERES = ((-0.05, 0.025, -0.03), (-0.05, -0.025, -0.03), (0.12, 0.03, -0.03), (0.12, -0.03, -0.03))
FOOT_RADIUS = 0.005
NQ, NV, NU = 36, 35, 29


def _quat_wxyz_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    return q[1:] * (2 * np.arctan2(s, q[0]) / s if s > 1e-12 else 2.0)


def _rotvec_to_quat_wxyz(v: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate(([np.cos(th / 2)], np.sin(th / 2) * v / th))


class _World:
    """One G1 in its own SuperDex scene (inside a worker process)."""

    def __init__(self, sdp, sdr, prefab, ctx, dt: float):
        self.sdp, self.sdr, self.prefab, self.ctx, self.dt = sdp, sdr, prefab, ctx, dt
        self.scene = sdp.create_scene("g1")
        self.scene.set_gravity([0, 0, -9.81])
        self.ground = self.scene.create_rigid_actor(name="ground", shape=sdp.create_plane_shape(normal=[0, 0, 1], distance=0),
                                                    is_static=True)  # fmt: skip
        self.torso_li = [prefab.links[i].name for i in range(len(prefab.links))].index(g1.COM_BODY)
        self.com_offset = np.zeros(3)
        self.bot = None
        self._build()
        self.zero_v = np.zeros(NV, dtype=np.float32)
        self.pose = sdp.DynamicArrayReal(NV)
        self.vel = sdp.DynamicArrayReal(NV)

    def _build(self):
        """(Re)create the robot; the torso CoM offset is part of the prefab (a nested link's inertia cannot be set later)."""
        sdp, pre = self.sdp, self.prefab
        if self.bot is not None:
            self.sdr.destroy_bot(self.scene, self.bot)
        link = pre.links[self.torso_li]
        com0 = link.center_of_mass
        if np.any(self.com_offset != 0):
            c = np.array(list(com0) if com0 is not None else [0.0, 0.0, 0.0], dtype=np.float64) + self.com_offset
            link.center_of_mass = sdp.Real3(*c)
        try:
            self.bot = self.sdr.create_bot(self.scene, pre, self.ctx)
        finally:
            link.center_of_mass = com0  # the prefab is shared by the worker's worlds
        self.act = self.bot.get_articulated_actor()
        n_links = len(pre.links)
        pc = sdp.PoseControllerParams(n_links)
        jt = pc.joint_tracking
        for li in range(n_links):
            j = pre.joints[li]
            if j.type == sdp.ArticulatedJointType.REVOLUTE:
                jt[li] = sdp.PoseTrackingParams(stiffness=g1.kp(j.name), damping=g1.kd(j.name), saturation=g1.motor_of(j.name)[0])
        pc.joint_tracking = jt
        self.act.add_articulated_pose_controller(pc)
        nested = [self.scene.get_actor(h) for h in self.act.get_nested_link_actors()]  # handles -> actors
        self.contact_actors = []  # actors with a collider (links without a shape have no contact parameters)
        for a in nested + [self.ground]:
            try:
                a.get_contact_params()
                self.contact_actors.append(a)
            except Exception:
                pass

    def set_params(self, friction: float, com_offset: np.ndarray):
        if np.any(np.asarray(com_offset) != self.com_offset):
            self.com_offset = np.array(com_offset, dtype=np.float64)
            self._build()
        # MuJoCo's contact friction is the max of the pair (ground: 1.0); SuperDex's pair rule is not documented, so
        # the effective value goes on both the robot links and this world's ground
        mu = max(float(friction), 1.0)
        for a in self.contact_actors:
            cp = a.get_contact_params()
            cp.coulomb_friction_coefficient = mu
            a.set_contact_params(cp)

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray):
        R = nprot.quat_to_mat(qpos[3:7][None])[0]
        pose = np.concatenate((qpos[:3], _quat_wxyz_to_rotvec(qpos[3:7]), qpos[7:]))
        vel = np.concatenate((qvel[:3], R @ qvel[3:6], qvel[6:]))  # MuJoCo: body-frame angular velocity
        self.act.set_articulated_pose_from_joints(pose.astype(np.float32))
        self.act.set_articulated_joint_velocities(vel.astype(np.float32))

    def step(self, target: np.ndarray, nstep: int):
        self.act.get_articulated_pose(self.pose)
        p = np.array(self.pose, dtype=np.float32)
        p[6:] = target
        self.act.set_articulated_target_pose(p)
        self.act.set_articulated_target_velocity(self.zero_v)  # PD toward zero velocity, as MuJoCo's position actuator
        for _ in range(nstep):
            self.scene.step(self.dt)

    def get_state(self):
        self.act.get_articulated_pose(self.pose)
        self.act.get_articulated_joint_velocities(self.vel)
        p, v = np.array(self.pose, dtype=np.float64), np.array(self.vel, dtype=np.float64)
        quat = _rotvec_to_quat_wxyz(p[3:6])
        R = nprot.quat_to_mat(quat[None])[0]
        return np.concatenate((p[:3], quat, p[6:])), np.concatenate((v[:3], R.T @ v[3:6], v[6:]))


def build_prefab(sdp, sdr):
    pre = sdr.load_bot_prefab_from_urdf_file(os.path.join(g1.DATA_DIR, "g1_description", "g1_29dof_rev_1_0.urdf"))
    lidx = {pre.links[i].name: i for i in range(len(pre.links))}
    for side in ("left", "right"):
        for k, c in enumerate(FOOT_SPHERES):
            pre.links.append(sdr.BotLinkPrefab(
                name=f"{side}_foot_sphere{k}", parent_link=lidx[f"{side}_ankle_roll_link"],
                parent_joint_from_link=sdp.TransformRT(sdp.Real3(*c)), shape=sdp.create_sphere_shape(sdp.Real3(0, 0, 0), FOOT_RADIUS),
                mass=1e-4, center_of_mass=sdp.Real3(0, 0, 0), moment_of_inertia=sdp.Real6(1e-9, 1e-9, 1e-9, 0, 0, 0)))  # fmt: skip
            pre.joints.append(sdr.BotJointPrefab(name=f"{side}_foot_sphere{k}_joint", type=sdp.ArticulatedJointType.HARD))
    revolute = []
    for i in range(len(pre.joints)):
        j = pre.joints[i]
        if j.type == sdp.ArticulatedJointType.REVOLUTE:
            j.inertia = g1.motor_of(j.name)[2]  # reflected rotor armature
            revolute.append(j.name)
    assert revolute == list(g1.JOINT_NAMES), revolute
    pre.world_from_root = sdp.TransformRT(sdp.Real3(0.0, 0.0, 0.0))  # identity: the root pose DOFs are world coordinates
    return pre


def _worker(rank: int, ids: list[int], shm_names: dict, nsim: int, dt: float, conn):
    """Worker process body: own the worlds ``ids``, serve step / set / params commands over ``conn``."""
    os.environ.setdefault("SUPERDEX_PRECISION", "fp32")
    import superdex.physics as sdp
    import superdex.robotics as sdr

    sdp.initialize(num_worker_threads=0)
    sdp.enable_file_cache(True)
    # keep the handles referenced (their buffers back the arrays); track=False: the owner (SdxBatch) unlinks them
    shms = {k: shared_memory.SharedMemory(name=n, track=False) for k, (n, _) in shm_names.items()}
    arr = {k: np.ndarray(shape, dtype=np.float64, buffer=shms[k].buf) for k, (_, shape) in shm_names.items()}
    prefab = build_prefab(sdp, sdr)
    ctx = sdr.create_context()
    worlds = [_World(sdp, sdr, prefab, ctx, dt) for _ in ids]
    for w, i in zip(worlds, ids):
        arr["qpos"][i], arr["qvel"][i] = w.get_state()
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg[0] == "quit":
            return
        if msg[0] == "params":
            for w, i in zip(worlds, ids):
                w.set_params(arr["friction"][i, 0], arr["com"][i])
        elif msg[0] == "step":
            nstep, dirty = msg[1], msg[2]
            for w, i in zip(worlds, ids):
                if dirty[i]:
                    w.set_state(arr["qpos"][i], arr["qvel"][i])
                w.step(arr["ctrl"][i].astype(np.float32), nstep)
                qp, qv = w.get_state()
                arr["qpos"][i], arr["qvel"][i] = qp, qv
        elif msg[0] == "set":
            for w, i in zip(worlds, ids):
                if msg[1][i]:
                    w.set_state(arr["qpos"][i], arr["qvel"][i])
        conn.send("ok")


class SdxBatch:
    """The ``mujoco.batch.Batch`` surface ``TrackingSim`` uses, on SuperDex worker processes."""

    def __init__(self, model, nsim: int, nthread: int = 0, dt: float | None = None):
        import mujoco

        self.nsim, self.model = nsim, model
        self.dt = float(model.opt.timestep) if dt is None else dt
        nw = max(1, min(nthread or os.cpu_count() or 8, nsim))
        shapes = {"qpos": (nsim, NQ), "qvel": (nsim, NV), "ctrl": (nsim, NU), "friction": (nsim, 1), "com": (nsim, 3)}
        self._shm = {k: shared_memory.SharedMemory(create=True, size=int(np.prod(s)) * 8) for k, s in shapes.items()}
        self.arr = {k: np.ndarray(s, dtype=np.float64, buffer=self._shm[k].buf) for k, s in shapes.items()}
        self.arr["friction"][:] = 1.0
        self.arr["com"][:] = 0.0
        names = {k: (self._shm[k].name, shapes[k]) for k in shapes}
        # workers are separate interpreters (python -m supermimic_mlx.sdx), not multiprocessing children: spawn would
        # re-import the caller's main module, and train.py runs training at import
        self._dir = tempfile.mkdtemp(prefix="sdx_")
        addr = os.path.join(self._dir, "sock")
        authkey = os.urandom(16)
        listener = Listener(addr, family="AF_UNIX", authkey=authkey)
        self.procs, self.conns = [], []
        env = dict(os.environ, SDX_AUTHKEY=authkey.hex())
        for r in range(nw):
            cfg = json.dumps({"rank": r, "ids": list(range(r, nsim, nw)), "shm": names, "nsim": nsim, "dt": self.dt, "addr": addr})
            self.procs.append(subprocess.Popen([sys.executable, "-m", "supermimic_mlx.sdx", cfg], env=env))
        by_rank = {}
        for _ in range(nw):
            c = listener.accept()
            by_rank[c.recv()] = c
        listener.close()
        self.conns = [by_rank[r] for r in range(nw)]
        for c in self.conns:
            assert c.recv() == "ready"
        self.qpos, self.qvel, self.ctrl = self.arr["qpos"], self.arr["qvel"], self.arr["ctrl"]
        self._last_qpos, self._last_qvel = self.qpos.copy(), self.qvel.copy()
        self._key_qpos = np.array(model.key_qpos[0]) if model.nkey else np.array(model.qpos0)
        self._fields = {}
        self._mj = mujoco

    # ---- mujoco.batch.Batch surface
    def bind(self, name: str) -> np.ndarray:
        return {"qpos": self.qpos, "qvel": self.qvel, "ctrl": self.ctrl}[name]

    def expand(self, name: str) -> np.ndarray:
        if name not in self._fields:
            self._fields[name] = np.repeat(np.array(getattr(self.model, name))[None], self.nsim, 0)
        return self._fields[name]

    def _push_params(self):
        """Per-world friction (of the robot geoms) and torso CoM offset from the expanded MuJoCo fields."""
        if "geom_friction" in self._fields:
            gf = self._fields["geom_friction"]
            robot = np.array([g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] > 1])
            self.arr["friction"][:, 0] = gf[:, robot, 0].max(1)
        if "body_ipos" in self._fields:
            b = self.model.body("robot/" + g1.COM_BODY).id
            self.arr["com"][:] = self._fields["body_ipos"][:, b] - self.model.body_ipos[b]
        self._call("params")

    def reset(self, ids=None, keyframe=-1, stage=None):
        ids = np.arange(self.nsim) if ids is None else np.asarray(ids)
        if not getattr(self, "_params_pushed", False):
            self._push_params()
            self._params_pushed = True
        self.qpos[ids] = self.model.qpos0
        self.qvel[ids] = 0.0
        self.ctrl[ids] = 0.0

    def step(self, ids=None, nstep: int = 1, stage=None):
        assert ids is None
        dirty = ~(np.all(self.qpos == self._last_qpos, -1) & np.all(self.qvel == self._last_qvel, -1))
        self._call("step", nstep, dirty)
        self._last_qpos[:], self._last_qvel[:] = self.qpos, self.qvel

    def forward(self, ids=None, stage=None):
        dirty = ~(np.all(self.qpos == self._last_qpos, -1) & np.all(self.qvel == self._last_qvel, -1))
        self._call("set", dirty)
        self._last_qpos[:], self._last_qvel[:] = self.qpos, self.qvel

    def _call(self, *msg):
        for c in self.conns:
            c.send(msg)
        for c in self.conns:
            assert c.recv() == "ok"

    def close(self):
        for c in self.conns:
            try:
                c.send(("quit",))
            except Exception:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        for s in self._shm.values():
            s.close(), s.unlink()
        self._shm = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":  # worker process: python -m supermimic_mlx.sdx '<json config>'
    _cfg = json.loads(sys.argv[1])
    _conn = Client(_cfg["addr"], family="AF_UNIX", authkey=bytes.fromhex(os.environ["SDX_AUTHKEY"]))
    _conn.send(_cfg["rank"])
    _worker(_cfg["rank"], _cfg["ids"], {k: (n, tuple(sh)) for k, (n, sh) in _cfg["shm"].items()}, _cfg["nsim"], _cfg["dt"], _conn)
