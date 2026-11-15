"""The training scene with plain MuJoCo: what mjlab's Scene compiles for upstream BlinkMimic (a ground plane and the
G1 entity, attached under the prefix ``robot/``, with one <position> actuator per joint), plus the solver options of
upstream ``env.make_sim``. tests/test_scene.py checks the result against mjlab's own compile."""

from __future__ import annotations

import mujoco
import numpy as np

from .robots import g1

CONTROL_DT = 0.02
DECIMATION = 4  # 4 physics substeps of 5 ms per 50 Hz control step
PHYSICS_DT = CONTROL_DT / DECIMATION


def robot_entity_spec() -> mujoco.MjSpec:
    spec = g1.robot_spec()
    for name in g1.JOINT_NAMES:
        effort, _, arm = g1.motor_of(name)
        spec.joint(name).armature = arm
        act = spec.add_actuator(name=name, target=name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = g1.kp(name)
        act.biasprm[1] = -g1.kp(name)
        act.biasprm[2] = -g1.kd(name)
        act.inheritrange = 0.0
        act.ctrllimited = False  # setpoints beyond the joint limits are allowed (force is still bounded)
        act.forcelimited = True
        act.forcerange[:] = np.array([-effort, effort])
    return spec


def scene_spec(visual: bool = False) -> mujoco.MjSpec:
    world = mujoco.MjSpec()
    world.worldbody.add_body(name="terrain").add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01))
    if visual:
        world.worldbody.add_light(name="sun", pos=(0, 0, 1.5), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    robot = robot_entity_spec()
    frame = world.worldbody.add_frame()
    frame.attach_body(robot.body("pelvis"), "robot/", "")
    q0 = np.zeros(7 + len(g1.JOINT_NAMES))
    q0[:3], q0[3] = g1.INIT_ROOT_POS, 1.0
    q0[7:] = g1.default_joint_pos()
    world.add_key(name="init_state", qpos=q0, ctrl=g1.default_joint_pos())
    return world


def apply_options(m: mujoco.MjModel) -> None:
    """Upstream env.make_sim: 5 ms implicitfast, pyramidal cone, Newton (100 / 50 line-search iterations, tol 1e-6)."""
    o = m.opt
    o.timestep = PHYSICS_DT
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    o.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    o.iterations, o.ls_iterations, o.tolerance, o.ls_tolerance = 100, 50, 1e-6, 0.01
    o.ccd_iterations = 35
    o.impratio = 1.0
    o.gravity[:] = (0.0, 0.0, -9.81)
    # MuJoCo Warp (upstream) has no auto-reset: a diverged world must stay visibly non-finite (sim.step_core ends its
    # episode) instead of being silently reset to qpos0 mid-step and stored as a teleport transition
    o.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET


def build_model(visual: bool = False) -> mujoco.MjModel:
    """Compiled scene; ``visual`` keeps the visual geoms (rendering), otherwise they are removed (faster stepping and
    smaller models; they never collide)."""
    spec = scene_spec(visual)
    if not visual:
        for geom in list(spec.geoms):
            if geom.contype == 0 and geom.conaffinity == 0:
                spec.delete(geom)
        for mesh in list(spec.meshes):
            if not any(g.meshname == mesh.name for g in spec.geoms):
                spec.delete(mesh)
    m = spec.compile()
    apply_options(m)
    return m


class Index:
    """Addresses in the compiled scene."""

    def __init__(self, m: mujoco.MjModel):
        self.joint_q = np.array([m.jnt_qposadr[m.joint("robot/" + n).id] for n in g1.JOINT_NAMES])
        self.joint_v = np.array([m.jnt_dofadr[m.joint("robot/" + n).id] for n in g1.JOINT_NAMES])
        assert (self.joint_q == np.arange(7, 7 + len(g1.JOINT_NAMES))).all()
        assert (self.joint_v == np.arange(6, 6 + len(g1.JOINT_NAMES))).all()
        self.com_body = m.body("robot/" + g1.COM_BODY).id
        robot_bodies = {m.body(b).id for b in range(m.nbody) if m.body(b).name.startswith("robot/")}
        self.robot_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies])
        self.ground_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == 0])
