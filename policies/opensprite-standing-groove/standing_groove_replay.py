"""Jetson CPU inference check for the groove actor using recorded simulation observations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort


def groove_history(packed):
    """Convert the native walking runtime's term-major observation to dance frames."""
    obs = np.asarray(packed, dtype=np.float32).reshape(993)
    fields = [obs[a:b].reshape(10, -1) for a, b in
              [(0, 310), (310, 620), (930, 960), (960, 990), (620, 930)]]
    return np.concatenate(fields, axis=1)[None, :, :]


def check(package):
    contract = json.loads((package / "standing_groove_contract.json").read_text())
    model = package / "standing_groove_100k.onnx"
    if hashlib.sha256(model.read_bytes()).hexdigest() != contract["policy_onnx_sha256"]:
        raise ValueError("Unexpected model fingerprint")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
    with np.load(package / "standing_groove_replay_vectors.npz", allow_pickle=False) as vectors:
        observations, expected = vectors["packed_obs"], vectors["mlx_actions"]
    count = contract["frames"]
    if observations.shape != (count, 993) or expected.shape != (count, 31):
        raise ValueError("Incomplete replay vectors")
    results, elapsed = [], []
    for frame, obs in enumerate(observations):
        inputs = {"history": groove_history(obs), "frame": np.array([frame], dtype=np.int64)}
        start = time.perf_counter_ns()
        results.append(session.run(["actions"], inputs)[0][0])
        elapsed.append((time.perf_counter_ns()-start)/1e6)
    np.testing.assert_allclose(results, expected, rtol=2e-4, atol=2e-4)
    print(f"PASS: {count} frames; max raw action error {np.max(np.abs(results-expected)):.8g}; "
          f"CPU inference p99 {np.percentile(elapsed,99):.3f} ms; no hardware IO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    check(parser.parse_args().package)
