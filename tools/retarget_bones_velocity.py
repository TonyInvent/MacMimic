"""Retarget the verified BONES-SEED velocity candidates on this Mac, keeping rejections per clip."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import random

from retarget_bones_g1 import retarget
from verify_bones_seed import REVISION


def solve(task):
    path, frames, output = task
    report_path = output / (path.stem + ".json")
    npz = output / (path.stem + ".npz")
    failure = output / (path.stem + ".rejected.json")
    if report_path.exists() and npz.exists():
        return True
    if failure.exists():
        return False
    try:
        retarget(path, frames, output)
    except ValueError as exc:
        if "exceeding" not in str(exc) or "lift" not in str(exc):
            raise
        tmp = failure.with_suffix(".tmp")
        tmp.write_text(json.dumps({"source": str(path), "reason": str(exc)}, indent=2) + "\n")
        tmp.replace(failure)
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/bones-seed"))
    parser.add_argument("--output", type=Path, default=Path("data/bones-seed/opensprite-velocity"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Process a deterministic sample of candidates")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Expected positive workers and limit")
    provenance = json.loads((args.data / "source.json").read_text())
    if provenance.get("official_revision") != REVISION:
        raise ValueError("Verify source bytes with verify_bones_seed.py before retargeting")
    solver = Path(__file__).with_name("retarget_bones_g1.py")
    digest = hashlib.sha256(solver.read_bytes() + solver.with_name("bones_velocity_stats.py").read_bytes()).hexdigest()
    directory = args.output / digest[:12]
    directory.mkdir(parents=True, exist_ok=True)
    with (args.data / "velocity_candidates.csv").open(newline="") as manifest:
        tasks = []
        for row in csv.DictReader(manifest):
            path = args.data / "velocity_sources" / row["move_g1_path"]
            # Retain capture-date directories even if future sources reuse names.
            dest = directory / path.parent.name
            dest.mkdir(exist_ok=True)
            tasks.append((path, int(row["move_duration_frames"]), dest))
    random.Random(49).shuffle(tasks)
    tasks = tasks[:args.limit]
    ok = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(solve, task) for task in tasks]
        for count, f in enumerate(as_completed(futures), 1):
            ok += f.result()
            if count <= 5 or count % 100 == 0 or count == len(tasks):
                print(f"Processed {count}/{len(tasks)}: {ok} converted, {count-ok} rejected", flush=True)
