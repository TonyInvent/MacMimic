"""Run the exported Xbox velocity actor in MuJoCo or evaluate scripted commands."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import subprocess
from contextlib import contextmanager

os.environ["MACMIMIC_ROBOT"] = "opensprite"
import mujoco
import numpy as np
import onnxruntime as ort
from supermimic_mlx.command_ranges import COMMAND_RANGES
from supermimic_mlx.scene import build_model
from supermimic_mlx.robots import opensprite as robot

FORWARD_LIMIT = 0.40
REVERSE_LIMIT = 0.05
YAW_LIMIT = 0.20


class Controller:
    def __init__(self, package, visual=False):
        contract = json.loads((package / "contract.json").read_text())
        path = package / "policy.onnx"
        if (
            contract["policy_onnx_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
            or tuple(contract["joint_names"]) != robot.JOINT_NAMES
            or contract["control_profile_sha256"] != robot.source.CONTROL_PROFILE_SHA256
        ):
            raise ValueError("Velocity policy package does not match the OpenSprite contract")
        self.command_profile = contract.get("simulation_command_profile", "opensprite")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.model = build_model(visual, studio=visual)
        self.root = self.model.body("robot/pelvis_link").id
        self.q0 = robot.default_joint_pos()
        self.data = mujoco.MjData(self.model)
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.prev = np.zeros(31, np.float32)
        self.history = np.repeat(self.observation()[None, :], 10, axis=0)

    def observation(self):
        d = self.data
        return np.concatenate(
            (d.qpos[7:] - self.q0, d.qvel[6:], self.prev, d.qvel[3:6], -d.xmat[self.root].reshape(3, 3)[2])
        ).astype("f")

    def step(self, cmd):
        self.history[:-1] = self.history[1:]
        self.history[-1] = self.observation()
        blocks = np.split(self.history, [31, 62, 93, 96], axis=1)
        row = np.concatenate([b.ravel() for b in blocks] + [cmd]).astype("f")
        self.prev = self.session.run(None, {"obs": row[None]})[0][0]
        if not np.isfinite(self.prev).all():
            raise FloatingPointError("Nonfinite action")
        self.data.ctrl[:] = self.q0 + 0.25 * self.prev
        mujoco.mj_step(self.model, self.data, nstep=10)
        mujoco.mj_forward(self.model, self.data)
        R = self.data.xmat[self.root].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        x, y = self.data.qvel[:2]
        measured = np.array([c * x + s * y, -s * x + c * y, (R @ self.data.qvel[3:6])[2]])
        fallen = self.data.qpos[2] < 0.25 or R[2, 2] < 0.3 or not np.isfinite(measured).all()
        return measured, fallen


def evaluate(controller, seconds):
    rows = []
    cases = {
        "stand": [0, 0, 0],
        "forward": [0.25, 0, 0],
        "reverse": [-0.1, 0, 0],
        "left": [0, 0, 0.2],
        "right": [0, 0, -0.2],
        "lateral": [0, 0.12, 0],
        "release": [0, 0, 0],
    }
    for name, target in cases.items():
        controller.reset()
        samples = []
        for i in range(round(seconds * 50)):
            cmd = [0.25, 0, 0] if name == "release" and i / 50 < seconds / 2 else target
            measured, fallen = controller.step(cmd)
            if i / 50 >= seconds * 0.65:
                samples.append(measured)
            if fallen:
                break
        rows.append(
            dict(
                case=name,
                target=target,
                seconds_alive=(i + 1) / 50,
                fallen=bool(fallen),
                measured=np.mean(samples, axis=0).tolist() if samples else None,
            )
        )
    return rows


class StickMapping:
    def __init__(self, profile="opensprite"):
        self.paper = profile == "paper"
        self.ready = False
        self.held = False
        self.cmd = np.zeros(3)

    def update(self, x, w, trigger, lateral=0.0):
        centered = max(abs(x), abs(w), abs(lateral) if self.paper else 0) <= 0.12
        if trigger < 0.2:
            self.ready = centered
            self.held = False
        elif trigger > 0.90 and self.ready:
            self.held = centered
            self.ready = False

        def axis(value):
            return np.sign(value) * max(0, min(1, abs(value)) - 0.12) / 0.88

        target = np.zeros(3)
        if self.held:
            target[0] = axis(x) * (FORWARD_LIMIT if x >= 0 else REVERSE_LIMIT)
            target[2] = axis(w) * YAW_LIMIT
        rates = np.array([0.05 if abs(target[0]) > abs(self.cmd[0]) else 0.1, 0, 0.2])
        if self.paper:
            low, high = (np.asarray(bound) for bound in COMMAND_RANGES["paper"])
            stick = np.array([axis(x), axis(lateral), axis(w)])
            target = np.where(stick >= 0, high, -low) * stick if self.held else np.zeros(3)
            rates = np.maximum(high, -low)  # full-scale command in one second in simulation
        self.cmd += np.clip(target - self.cmd, -0.02 * rates, 0.02 * rates)
        return self.cmd


class Xbox:
    def __init__(self, profile="opensprite"):
        import pygame
        from pygame._sdl2 import controller

        pygame.init()
        controller.init()
        found = [i for i in range(controller.get_count()) if controller.is_controller(i)]
        if not found:
            raise RuntimeError(
                "No SDL game controller connected. Pair the Xbox controller in macOS Bluetooth settings."
            )
        self.pad = controller.Controller(found[0])
        self.pg = pygame
        self.mapping = StickMapping(profile)
        if self.mapping.paper:
            print("Appendix G simulation: left X controls lateral speed.")
        print("Center sticks, release LT, then hold LT. Left Y moves, right X turns. Ctrl-C exits.")

    def sample(self):
        pg = self.pg
        pg.event.pump()
        if not self.pad.attached():
            raise RuntimeError("Controller disconnected")
        x = -self.pad.get_axis(pg.CONTROLLER_AXIS_LEFTY) / 32768
        w = -self.pad.get_axis(pg.CONTROLLER_AXIS_RIGHTX) / 32768
        trigger = self.pad.get_axis(pg.CONTROLLER_AXIS_TRIGGERLEFT) / 32767
        lateral = -self.pad.get_axis(pg.CONTROLLER_AXIS_LEFTX) / 32768 if self.mapping.paper else 0.0
        return self.mapping.update(x, w, trigger, lateral)


def drive_sequence(controller, record=None):
    """Replay stick inputs through the live mapping, without a reset between commands."""
    controller.reset()
    mapping = StickMapping(controller.command_profile)
    phases = [
        ("start", 1.0, 0, 0, 0),
        ("enable", 1.0, 0, 0, 1),
        ("forward", 12.0, 1, 0, 1),
        ("walking left", 7.0, 1, 1, 1),
        ("walking right", 7.0, 1, -1, 1),
        ("release", 7.0, 0, 0, 0),
    ]
    phases = [(*phase, 0) for phase in phases]
    if mapping.paper:
        phases = [
            ("start", 1, 0, 0, 0, 0), ("enable", 1, 0, 0, 1, 0),
            ("forward", 5, .4, 0, 1, 0), ("strafe left", 5, 0, 0, 1, .4),
            ("strafe right", 5, 0, 0, 1, -.4), ("reverse", 5, -.4, 0, 1, 0),
            ("turn left", 5, .3, .3, 1, 0), ("turn right", 5, .3, -.3, 1, 0),
            ("release", 5, 0, 0, 0, 0),
        ]
    rows = []
    for phase, seconds, forward, turn, trigger, lateral in phases:
        samples = []
        for i in range(round(seconds * 50)):
            cmd = mapping.update(forward, turn, trigger, lateral)
            measured, fallen = controller.step(cmd)
            if i >= round((seconds - min(2.0, seconds)) * 50):
                samples.append(measured.copy())
            if record is not None and i % 2 == 0:
                record(controller, phase, cmd, measured)
            if fallen:
                break
        rows.append(
            dict(
                phase=phase,
                seconds=(i + 1) / 50,
                fallen=bool(fallen),
                command=cmd.copy().tolist(),
                measured=np.mean(samples, axis=0).tolist() if samples else None,
            )
        )
        if fallen:
            break
    return rows


@contextmanager
def video_recorder(controller, path):
    from PIL import Image, ImageDraw

    m = controller.model
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.2, 125, -18
    process = subprocess.Popen(
        [
            os.getenv("FFMPEG", "ffmpeg"),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "640x480",
            "-r",
            "25",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        with mujoco.Renderer(m, 480, 640) as renderer:

            def frame(sim, phase, cmd, measured):
                camera.lookat[:] = [*sim.data.qpos[:2], 0.43]
                renderer.update_scene(sim.data, camera)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle([0, 0, 640, 47], fill=(24, 29, 32))
                draw.text((10, 6), f"OpenSprite   |   {phase}   |   {sim.data.time:.1f} s", fill="white")
                draw.text(
                    (10, 26),
                    f"Command {cmd[0]:+.2f} m/s, {cmd[2]:+.2f} rad/s     Measured {measured[0]:+.2f} m/s, {measured[2]:+.2f} rad/s",
                    fill="white",
                )
                process.stdin.write(image.tobytes())

            yield frame
    finally:
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("FFmpeg failed to write the sequence video")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("package", type=Path)
    p.add_argument("--xbox", action="store_true")
    p.add_argument("--sequence", action="store_true", help="Replay a continuous virtual stick sequence")
    p.add_argument("--video", type=Path, help="Record the simulated stick sequence to MP4")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    c = Controller(a.package, a.xbox or bool(a.video))
    if a.xbox:
        import mujoco.viewer

        pad = Xbox(c.command_profile)
        with mujoco.viewer.launch_passive(c.model, c.data) as view:
            while view.is_running():
                start = time.monotonic()
                _, fall = c.step(pad.sample())
                view.cam.lookat[:] = c.data.qpos[:3]
                view.sync()
                if fall:
                    break
                time.sleep(max(0, 0.02 - (time.monotonic() - start)))
    else:
        if a.video:
            with video_recorder(c, a.video) as record:
                rows = drive_sequence(c, record)
        else:
            rows = drive_sequence(c) if a.sequence else evaluate(c, a.seconds)
        print(json.dumps(rows, indent=2))
        if a.out:
            a.out.write_text(json.dumps(rows, indent=2))
