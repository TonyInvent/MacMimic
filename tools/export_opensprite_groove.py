"""Specialize the OpenSprite motion-tracking actor to the 600 frames of the standing groove."""
import argparse
import hashlib
import json
from pathlib import Path

from play_opensprite_groove import PACKAGE, play, robot
import mlx.core as mx
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from supermimic_mlx.checkpoint import load_tree
from supermimic_mlx.kinematics import Kinematics
from supermimic_mlx.networks import policy_from_args
from supermimic_mlx.optim import unflat


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(checkpoint, reference):
    ck = load_tree(str(checkpoint))
    args = ck["meta"]["args"]
    frames = 600
    if args["embodiment"] != "opensprite" or args["hist"] != 10 or args["pi_ref_skip"]:
        raise ValueError("Expected the ten-frame OpenSprite actor without reference skip")
    motion = robot.motion_library(reference, Kinematics())
    if motion.names != ["standing_groove"] or motion.clip_end_np[0] < frames:
        raise ValueError("Expected the complete standing groove")
    motion.goal_stride = args["goal_stride"]
    goals = motion.goal(mx.arange(frames), args["goal_frames"])
    pi = policy_from_args(args, 99, goals.shape[-1], 31)
    pi.update(unflat(ck["pi"]))
    # The reference is fixed. Compute only its branch ahead of time. The decoder still
    # consumes live measurements on every tick and never reads a table of joint actions.
    arrays = {"motion_latent": np.array(pi.encode(goals)),
              "obs_mean": np.array(pi.obs_norm.mean), "obs_std": np.array(pi.obs_norm.std())}
    node = helper.make_node
    nodes = [node("Gather", ["motion_latent", "frame"], ["z"], axis=0),
             node("Sub", ["history", "obs_mean"], ["centered"]),
             node("Div", ["centered", "obs_std"], ["normalized"]),
             node("Flatten", ["normalized"], ["features"], axis=1),
             node("Concat", ["z", "features"], ["input"], axis=1)]
    x = "input"
    for i, layer in enumerate(pi.dec.layers):
        weight, bias = f"w{i}", f"b{i}"
        arrays[weight], arrays[bias] = np.array(layer.weight).T, np.array(layer.bias)
        last = i == len(pi.dec.layers) - 1
        y = "actions" if last else f"linear{i}"
        nodes.append(node("Gemm", [x, weight, bias], [y]))
        if not last:
            x = f"hidden{i}"
            nodes.append(node("Elu", [y], [x]))
    value = helper.make_tensor_value_info
    graph = helper.make_graph(nodes, "OpenSpriteStandingGroove",
                              [value("history", TensorProto.FLOAT, [1, 10, 99]),
                               value("frame", TensorProto.INT64, [1])],
                              [value("actions", TensorProto.FLOAT, [1, 31])],
                              [numpy_helper.from_array(v, k) for k, v in arrays.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    policy = PACKAGE / "standing_groove_100k.onnx"
    onnx.save(model, policy)
    contract = dict(schema="opensprite_fixed_motion_v1", hardware_qualified=False,
                    iteration=ck["meta"]["iteration"], frames=frames, policy_dt=.02, decimation=10,
                    reference_sha256=digest(reference), checkpoint_sha256=digest(checkpoint.with_suffix(".safetensors")),
                    policy_onnx_sha256=digest(policy), asset_revision=robot.source.ASSET_REVISION,
                    control_profile_sha256=robot.source.CONTROL_PROFILE_SHA256, joint_names=list(robot.JOINT_NAMES),
                    default_joint_pos=robot.default_joint_pos().tolist(), action_scale=robot.action_scales().tolist(),
                    history_order="oldest_to_newest", observation_order=["joint_pos_relative", "joint_vel",
                    "root_local_gyro", "projected_gravity", "previous_raw_action"], frame_range=[0, frames-1])
    (PACKAGE / "standing_groove_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    histories, actions = [], []

    def capture(frame, history, action):
        histories.append(history[0].copy())
        actions.append(action.copy())

    assert play(headless=True, compare=capture) == frames
    expected = np.array(pi(mx.array(np.stack(histories)), goals))
    np.testing.assert_allclose(np.stack(actions), expected, rtol=2e-4, atol=2e-4)
    h = np.stack(histories)
    # Native runtime layout: term-major q, qd, previous action, gyro, gravity, then command.
    terms = [h[:, :, sl].reshape(frames, -1) for sl in
             (slice(0, 31), slice(31, 62), slice(68, 99), slice(62, 65), slice(65, 68))]
    packed = np.column_stack([*terms, np.zeros((frames, 3), dtype=np.float32)])
    np.savez_compressed(PACKAGE / "standing_groove_replay_vectors.npz", packed_obs=packed, mlx_actions=expected)
    print(f"{frames} frames passed, max absolute action difference {np.max(np.abs(actions - expected)):.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.reference)
