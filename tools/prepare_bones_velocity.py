"""Select reproducible BONES-SEED locomotion candidates and unpack their original G1 CSVs."""
import argparse
import csv
from pathlib import Path, PurePosixPath
import shutil
import tarfile

COLUMNS = ("move_g1_path", "move_duration_frames", "category", "content_name",
           "content_type_of_movement", "take_actor", "is_mirror")


def is_candidate(row):
    movements = {tag.strip() for tag in row["content_type_of_movement"].split(",")}
    gaits = {"walking", "running", "jogging"}
    name = set(row["content_name"].lower().split("_"))
    return (row["category"] in {"Basic Locomotion Neutral", "Baseline"}
            and row["package"] == "Locomotion" and row["content_uniform_style"] == "neutral"
            and row["content_body_position"] == "standing" and row["content_props"] == "0"
            and bool(gaits & movements) and movements <= gaits | {"turning", "standing"}
            and bool(name & {"walk", "run", "jog"}) and not name & {"weak", "amplified"})


def select(root):
    candidates = {}
    all_paths = set()
    with (root / "metadata/seed_metadata_v004.csv").open(newline="") as src:
        for row in csv.DictReader(src):
            path = row["move_g1_path"]
            parts = PurePosixPath(path).parts
            if parts[:2] != ("g1", "csv") or ".." in parts or not path.endswith(".csv") or path in all_paths:
                raise ValueError(f"Invalid or duplicate G1 path: {path}")
            all_paths.add(path)
            if is_candidate(row):
                candidates[path] = {key: row[key] for key in COLUMNS}
    manifest = root / "velocity_candidates.csv"
    tmp = manifest.with_suffix(".tmp")
    with tmp.open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(candidates.values())
    tmp.replace(manifest)
    print(f"{len(candidates)} candidates from {len(all_paths)} indexed G1 motions", flush=True)
    return all_paths, candidates


def unpack(root, expected, candidates, archive_path):
    seen = set()
    with tarfile.open(archive_path, "r|gz", bufsize=1024 * 1024) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            name = member.name.removeprefix("./")
            if name not in expected or name in seen:
                raise ValueError(f"Unexpected archive member: {name}")
            seen.add(name)
            if name in candidates:
                dest = root / "velocity_sources" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                partial = dest.with_suffix(".part")
                with partial.open("wb") as output, archive.extractfile(member) as source:
                    shutil.copyfileobj(source, output)
                if partial.stat().st_size != member.size:
                    raise ValueError(f"Incomplete member: {name}")
                partial.replace(dest)
    if expected != seen:
        raise ValueError(f"{len(expected - seen)} indexed G1 CSVs missing from archive")
    print(f"Archive matched all {len(seen)} paths. Extracted {len(candidates)} velocity candidates.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/bones-seed"))
    parser.add_argument("--unpack", action="store_true", help="Extract candidates and verify full G1 path coverage")
    parser.add_argument("--archive", type=Path, help="Alternate G1 tar.gz path or stream")
    args = parser.parse_args()
    expected, candidates = select(args.data)
    if args.unpack:
        unpack(args.data, expected, candidates, args.archive or args.data / "g1.tar.gz")
