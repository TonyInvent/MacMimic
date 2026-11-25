"""Run a fixed-motion OpenSprite ONNX choreography once in local MuJoCo."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ["MACMIMIC_ROBOT"] = "opensprite"
import mujoco
import numpy as np
import onnxruntime as ort

from supermimic_mlx.robots import opensprite as robot
from supermimic_mlx.scene import build_model

PACKAGE = Path(__file__).resolve().parents[1] / "policies/opensprite-standing-groove"


def play(package=PACKAGE, *, headless=False, compare=None, prefix="standing_groove"):
    contract = json.loads((package / f"{prefix}_contract.json").read_text())
    policy = package / f"{prefix}_100k.onnx"
    if (hashlib.sha256(policy.read_bytes()).hexdigest() != contract["policy_onnx_sha256"]
            or tuple(contract["joint_names"]) != robot.JOINT_NAMES
            or contract["control_profile_sha256"] != robot.source.CONTROL_PROFILE_SHA256):
        raise ValueError("Choreography package and OpenSprite v5 configuration do not match")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(policy), opts, providers=["CPUExecutionProvider"])
    model = build_model(not headless, studio=not headless)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    root = model.body("robot/pelvis_link").id
    default = np.array(contract["default_joint_pos"])
    scale = np.array(contract["action_scale"])
    prev = np.zeros(len(default), np.float32)

    def observation():
        return np.concatenate((data.qpos[7:] - default, data.qvel[6:], data.qvel[3:6],
                               -data.xmat[root].reshape(3, 3)[2], prev)).astype(np.float32)

    history = np.tile(observation(), (1, 10, 1))
    viewer = nullcontext(None)
    if not headless:
        import mujoco.viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
    with viewer as window:
        if window:
            window.cam.distance, window.cam.azimuth, window.cam.elevation = 1.82, 10, -6
            window.cam.lookat[:] = [0, 0, .50]
            window.sync()
        sound = subprocess.Popen(["afplay", str(package / f"{prefix}_beat.wav")]) if window else None
        try:
            start = time.monotonic()
            for frame in range(contract["frames"]):
                if window and not window.is_running():
                    return frame
                history[:, :-1] = history[:, 1:]
                history[:, -1] = observation()
                prev = session.run(None, {"history": history, "frame": np.array([frame], np.int64)})[0][0]
                if not np.isfinite(prev).all():
                    raise RuntimeError(f"Nonfinite action at frame {frame}")
                if compare is not None:
                    compare(frame, history, prev)
                data.ctrl[:] = default + scale * prev
                mujoco.mj_step(model, data, nstep=contract["decimation"])
                mujoco.mj_forward(model, data)
                if data.qpos[2] < .25 or data.xmat[root, 8] < .3:
                    raise RuntimeError(f"Fall at frame {frame}")
                if window:
                    window.sync()
                    time.sleep(max(0, start + (frame + 1) * contract["policy_dt"] - time.monotonic()))
        finally:
            if sound is not None and sound.poll() is None:
                sound.terminate()
                sound.wait()
    return contract["frames"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run the same simulation without graphics or audio")
    parser.add_argument("--disco-point", action="store_true", help="Play the 32.8-second disco choreography")
    args = parser.parse_args()
    if args.disco_point:
        frames = play(PACKAGE.with_name("opensprite-disco-point"), prefix="disco_point", headless=args.headless)
    else:
        frames = play(headless=args.headless)
    print(f"Simulated {frames} frames")
