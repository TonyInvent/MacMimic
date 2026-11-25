"""Compare the protected actor's native groove inference against every supplied MLX frame."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def binding(package):
    hashes = {}
    for filename, expected in (
        ("standing_groove_contract.json", "acef7c5faf0c5f742555ff4e2633d9e740308fb42fd9c46f2621aedc01a39dc0"),
        ("standing_groove_control_profile.json", "184ec989fd24b4bc6ea5034e6fdae796b5b9ce83ef2c884dfe4fc61d8d6987dc"),
        ("standing_groove_100k.onnx", "ad35d34d4652bb9134c54788c0bf80f9ae6a4e5e77d4ac2ff8e00a33b490e796"),
        ("standing_groove_replay_vectors.npz", "673df5e7cc3f38e84650f84f2172eca37df71ed5f288d503cc727e02d7b1c507"),
    ):
        data = (package / filename).read_bytes()
        if filename.endswith(".json"):
            data = data.replace(b"\r\n", b"\n")
        hashes[filename] = hashlib.sha256(data).hexdigest()
        if hashes[filename] != expected:
            raise ValueError("Standing groove fingerprint mismatch: " + filename)
    return hashes


def check(package, actor, report):
    hashes = binding(package)
    actor = actor.resolve()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(actor.parent), env.get("LD_LIBRARY_PATH")]))
    with np.load(package / "standing_groove_replay_vectors.npz", allow_pickle=False) as vectors:
        obs, expected = vectors["packed_obs"], vectors["mlx_actions"]
    if obs.shape != (600, 993) or expected.shape != (600, 31) or not np.isfinite(obs).all():
        raise ValueError("Incomplete standing groove replay")
    with tempfile.TemporaryDirectory() as scratch:
        inputs, outputs = Path(scratch)/"packed.f32", Path(scratch)/"actions.f32"
        obs.astype("<f4").tofile(inputs)
        subprocess.run([actor, "--replay-standing-groove", package/"standing_groove_100k.onnx",
                        inputs, outputs], env=env, check=True)
        actual = np.fromfile(outputs, dtype="<f4").reshape(600, 31)
        contract = json.loads((package/"standing_groove_contract.json").read_text())
        defaults = np.asarray(contract["default_joint_pos"], dtype="<f8")
        states = np.zeros((600, 71), dtype="<f8")
        states[:, :31] = obs[:,279:310].astype(np.float64) + defaults
        states[:,31:62] = obs[:,589:620]
        states[:,62:65] = obs[:,957:960]
        states[:,65:68] = obs[:,987:990]
        with inputs.open("wb") as stream:
            defaults.tofile(stream)
            np.asarray(contract["action_scale"], dtype="<f8").tofile(stream)
            states.tofile(stream)
        subprocess.run([actor, "--replay-standing-groove-feedback", package/"standing_groove_100k.onnx",
                        inputs, outputs], env=env, check=True)
        live = np.fromfile(outputs, dtype="<f4").reshape(600, 31)
    if not all(np.isfinite(x).all() for x in (actual, live, expected)):
        raise ValueError("Nonfinite replay actions")
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(live, expected, rtol=2e-4, atol=2e-4)
    feedback_delta = np.abs(live - expected)
    delta = np.abs(actual - expected)
    frame, joint = map(int, np.unravel_index(delta.argmax(), delta.shape))
    summary = {"frames": 600, "actions_compared": int(delta.size), "max_abs_error": float(delta.max()),
               "worst_frame": frame, "worst_joint_index": joint, "rtol": 2e-4, "atol": 2e-4,
               "rebuilt_history_max_abs_error": float(feedback_delta.max())}
    if report:
        report.write_text(json.dumps(dict(summary, fingerprints=hashes,
            actor_sha256=hashlib.sha256(actor.read_bytes()).hexdigest(),
            per_frame_max_abs_error=delta.max(axis=1).tolist(),
            rebuilt_history_per_frame_max_abs_error=feedback_delta.max(axis=1).tolist()), indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("actor", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    check(args.package, args.actor, args.report)
