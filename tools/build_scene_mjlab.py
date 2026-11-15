"""Reference scene from mjlab (setup-time only): compile the training scene exactly as upstream BlinkMimic's env.py
does (mjlab Scene: plane terrain + G1 entity with BuiltinPositionActuators) and save it as an MJB. The MLX port
builds the same model with plain MuJoCo (supermimic_mlx.scene); tests/test_scene.py checks both are identical.

Run: uv run --with mjlab python tools/build_scene_mjlab.py
"""

import os
import sys

import mujoco
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from mjlab.actuator import BuiltinPositionActuatorCfg  # noqa: E402
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg  # noqa: E402
from mjlab.scene import Scene, SceneCfg  # noqa: E402
from mjlab.terrains import TerrainEntityCfg  # noqa: E402

from supermimic_mlx.robots import g1  # noqa: E402

acts = tuple(
    BuiltinPositionActuatorCfg(target_names_expr=(n,), stiffness=g1.kp(n), damping=g1.kd(n), effort_limit=g1.motor_of(n)[0],
                               armature=g1.motor_of(n)[2])
    for n in g1.JOINT_NAMES
)
cfg = EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.76), joint_pos=g1.DEFAULT_JOINT_POS, joint_vel={".*": 0.0}),
    spec_fn=g1.robot_spec,
    articulation=EntityArticulationInfoCfg(actuators=acts, soft_joint_pos_limit_factor=0.9),
)
scene = Scene(SceneCfg(num_envs=1, env_spacing=0.0, terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": cfg}),
              device="cpu")
m = scene.compile()
# model arrays for the equivalence test (an .mjb would tie the file to this MuJoCo version)
FIELDS = ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia", "jnt_range", "jnt_axis",
          "dof_armature", "dof_damping", "dof_frictionloss", "geom_type", "geom_size", "geom_friction", "geom_contype",
          "geom_conaffinity", "geom_solref", "geom_solimp", "geom_condim", "geom_priority", "geom_margin", "geom_pos",
          "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_forcelimited", "actuator_ctrllimited",
          "actuator_trnid", "qpos0", "key_qpos")
out = os.path.join(ROOT, "data", "g1_scene_mjlab.npz")
np.savez(out, **{f: getattr(m, f) for f in FIELDS}, names=np.array([m.body(i).name for i in range(m.nbody)]),
         opt=np.array([m.opt.timestep, m.opt.integrator, m.opt.cone, m.opt.solver, m.opt.iterations]))
print("saved", out, "nu", m.nu, "ngeom", m.ngeom, "nbody", m.nbody, "nkey", m.nkey)
