"""Retarget complete LAFAN1 sources independently and record per-clip conversion results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "data/OpenSprite-SuperMimic"
OUT = UPSTREAM / "outputs/paper-velocity-lafan"
SOURCE = UPSTREAM / "data/sources/lafan1-resolved"


def convert(name):
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    runtime = UPSTREAM / ".venv-umr/bin/python"
    common = [str(runtime), "-u", "scripts/retarget_lafan.py", "--motions", name,
              "--output-dir", str(OUT), "--correspondence-device", "cpu"]
    native = target / "native.npz"
    # Keep each convert-stage bank separate until all candidate clips pass QA.
    result = target / "converted" / (name + ".npz")
    with (target / "processing.log").open("a") as log:
        if not native.is_file():
            subprocess.run([*common, "--stage", "retarget"], cwd=UPSTREAM, stdout=log, stderr=log, check=True)
        if not result.with_suffix(".json").is_file():
            subprocess.run([*common, "--stage", "convert", "--motion-output-dir", str(result.parent)],
                           cwd=UPSTREAM, stdout=log, stderr=log, check=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+", help="Process only these complete recordings")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads((SOURCE / "paper_locomotion_sources.json").read_text())
    available = {Path(item["file"]).stem for item in manifest["sources"]}
    names = sorted(available) if not args.names else args.names
    if args.workers < 1 or not set(names) <= available or len(names) != len(set(names)):
        parser.error("Expected distinct names in the complete-source manifest and a positive worker count")
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {name: pool.submit(convert, name) for name in names}
        for name, future in futures.items():
            try:
                print(future.result(), flush=True)
            except subprocess.CalledProcessError:
                failures.append(name)
                print(f"{name} failed. See {OUT/name/'processing.log'}", flush=True)
    if failures:
        raise SystemExit(f"Incomplete retargets: {', '.join(failures)}")
