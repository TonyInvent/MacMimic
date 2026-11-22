"""Export this Mac-trained OpenSprite velocity policy and compare it to MLX."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["MACMIMIC_ROBOT"] = "opensprite"
import mlx.core as mx
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from supermimic_mlx.checkpoint import load_tree
from supermimic_mlx.networks import policy_from_args
from supermimic_mlx.optim import unflat
from supermimic_mlx.robots import opensprite as robot
from supermimic_mlx.velocity import CommandEncoder, action


def export(checkpoint, destination):
    ck = load_tree(str(checkpoint))
    args = ck["meta"]["args"]
    if args["embodiment"] != "opensprite" or args["hist"] != 10:
        raise ValueError("Expected an OpenSprite joint checkpoint with ten history frames")
    pi = policy_from_args(args, 99, 77 * args["goal_frames"], 31)
    pi.update(unflat(ck["pi"]))
    encoder = CommandEncoder(99, 10, normalized=args.get("velocity_command_norm", False))
    encoder.update(unflat(ck["velocity"]))
    nodes, constants = [], []

    def const(name, value):
        constants.append(numpy_helper.from_array(np.asarray(value), name=name))
        return name

    def op(kind, inputs, result, **attrs):
        nodes.append(helper.make_node(kind, inputs, [result], **attrs))
        return result

    widths = [31, 31, 31, 3, 3]
    start, blocks = 0, []
    for i, width in enumerate(widths + [3]):
        end = start + width * (10 if i < 5 else 1)
        label = f"term{i}"
        piece = op(
            "Slice",
            ["obs", const(label + "Start", [start]), const(label + "End", [end]), const(label + "Axis", [1])],
            label,
        )
        if i < 5:
            blocks.append(op("Reshape", [piece, const(label + "Shape", [-1, 10, width])], label + "Frames"))
        else:
            cmd = piece
        start = end
    history = op("Concat", [blocks[i] for i in (0, 1, 3, 4, 2)], "history", axis=-1)
    mean = const("obs_mean", np.array(pi.obs_norm.mean))
    std = const("obs_std", np.sqrt(np.array(pi.obs_norm.var) + pi.obs_norm.eps**2))
    centered = op("Sub", [history, mean], "centered")
    normed = op("Div", [centered, std], "normalized")
    features = op("Flatten", [normed], "features", axis=1)

    def mlp(module, x, name):
        for i, layer in enumerate(module.layers):
            prefix = name + str(i)
            x = op(
                "Gemm",
                [x, const(prefix + "W", np.array(layer.weight).T), const(prefix + "B", np.array(layer.bias))],
                prefix + "Out",
            )
            if i + 1 < len(module.layers):
                x = op("Elu", [x], prefix + "Act")
        return x

    if encoder.command_norm is not None:
        cmd_mean = const("commandMean", np.array(encoder.command_norm.mean))
        cmd_std = const("commandStd", np.sqrt(np.array(encoder.command_norm.var) + encoder.command_norm.eps**2))
        cmd = op("Div", [op("Sub", [cmd, cmd_mean], "centeredCommand"), cmd_std], "normalizedCommand")
    z = mlp(encoder.net, op("Concat", [cmd, features], "encoderInput", axis=1), "encoder")
    decode = op("Concat", [op("Tanh", [z], "latent"), features], "decoderInput", axis=1)
    raw = mlp(pi.dec, decode, "decoder")
    op("Clip", [raw, const("lower", np.float32(-20)), const("upper", np.float32(20))], "actions")
    value = lambda n, w: helper.make_tensor_value_info(n, TensorProto.FLOAT, ["batch", w])
    graph = helper.make_graph(nodes, "OpenSpriteVelocity", [value("obs", 993)], [value("actions", 31)], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(model.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(43)
    h = rng.normal(0, 0.25, (32, 10, 99)).astype("f")
    cmd = rng.uniform([-0.3, -0.2, -0.6], [0.6, 0.2, 0.6], (32, 3)).astype("f")
    cols = [h[:, :, :31], h[:, :, 31:62], h[:, :, 68:99], h[:, :, 62:65], h[:, :, 65:68]]
    x = np.column_stack([b.reshape(32, -1) for b in cols] + [cmd])
    expected = np.array(action(pi, encoder, mx.array(h), mx.array(cmd))[0])
    actual = sess.run(None, {"obs": x})[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "policy.onnx"
    output.write_bytes(model.SerializeToString())
    controls = [robot.source.JOINT_CONTROL[n] for n in robot.JOINT_NAMES]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    contract = dict(
        schema="sprite_supermimic_joint_velocity_v1",
        iteration=ck["meta"]["iteration"],
        reference_sha256=args["reference_sha256"],
        asset_revision=robot.source.ASSET_REVISION,
        control_profile_sha256=robot.source.CONTROL_PROFILE_SHA256,
        policy_onnx_sha256=sha(output),
        joint_names=robot.JOINT_NAMES,
        command_layout=["vx", "vy", "yaw_rate"],
        simulation_command_profile=args.get("velocity_commands", "opensprite"),
        policy_dt=0.02,
        physics_dt=0.002,
        observation_order=[
            "joint_pos_relative",
            "joint_vel",
            "previous_raw_action",
            "root_local_gyro",
            "projected_gravity",
            "velocity_commands",
        ],
        history_order="term_major_oldest_to_newest",
        actor_observation_dim=993,
        action_dim=31,
        default_joint_pos=robot.default_joint_pos().tolist(),
        action_scale=[0.25] * 31,
        stiffness=[c.kp for c in controls],
        damping=[c.kd for c in controls],
        effort_limit=[c.command_effort_nm for c in controls],
        parity_max_abs_error=float(abs(actual - expected).max()),
        hardware_qualified=False,
    )
    sys.path.insert(0, str(robot.UPSTREAM / "runtime/src"))
    from open_sprite_runtime.contracts import POLICY_REPLAY_PROFILES, PolicyContract

    offset = 0
    terms = []
    for name, width, history_len in POLICY_REPLAY_PROFILES["supermimic_v5_10f"]:
        length = width * max(1, history_len)
        terms.append(
            dict(
                name=name,
                start=offset,
                end=offset + length,
                shape=[length],
                frame_dim=width,
                history_length=history_len,
                history_order="oldest_to_newest",
                flatten_history_dim=True,
            )
        )
        offset += length
    contract.update(
        training_run=checkpoint.parent.name,
        policy_onnx=output.name,
        checkpoint_sha256=sha(checkpoint.with_suffix(".safetensors")),
        decimation=10,
        normalization="embedded_running_norm",
        observation_terms=terms,
        observation_has_horizontal_base_velocity=False,
        observation_has_global_position=False,
        observation_has_global_yaw=False,
        action_offset=contract["default_joint_pos"],
        action_formula="target_joint_pos = action_offset + action_scale * policy_action",
        velocity_limit=[c.urdf_velocity_rad_s for c in controls],
        maximum_absolute_action=20,
        base_ang_vel_frame="root_link_local",
        quaternion_order="wxyz",
        projected_gravity_definition="R_world_to_base @ [0, 0, -1]",
        deployment_handoff_seconds=2.0,
        deployment_handoff_mode="smoothstep_from_pose_equivalent_action",
        deployment_initial_velocity_mode="zero",
        physical_ankle_differential={"enabled": False},
    )
    # Exercise the same schema and ordering checks as the upstream offline runtime.
    PolicyContract(destination / "contract.json", json.loads(json.dumps(contract))).validate_for_replay(
        "supermimic_v5_10f"
    )
    (destination / "contract.json").write_text(json.dumps(contract, indent=2))
    print(output, contract["parity_max_abs_error"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    export(a.checkpoint, a.destination)
