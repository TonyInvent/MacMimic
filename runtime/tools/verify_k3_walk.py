"""Compare all raw actions and joint targets with the reviewed 803-frame Jetson replay."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    dtype = np.dtype([("obs", "<f4", 993), ("raw", "<f4", 31),
                      ("action", "<f8", 31), ("target", "<f8", 31)])
    if args.replay.stat().st_size != 803 * dtype.itemsize:
        raise ValueError("Incomplete walking replay")
    actual = np.fromfile(args.replay, dtype=dtype)
    errors = {}
    with np.load(args.reference, allow_pickle=False) as ref:
        for field in ("raw", "target"):
            if ref[field].shape != (803, 31) or not all(np.isfinite(v).all() for v in (actual[field], ref[field])):
                raise ValueError("Invalid walking replay")
            np.testing.assert_allclose(actual[field], ref[field], rtol=2e-4, atol=2e-4)
            errors[field] = float(np.abs(actual[field] - ref[field]).max())
    print(json.dumps(dict(frames=803, max_abs_error=errors)))


if __name__ == "__main__":
    main()
