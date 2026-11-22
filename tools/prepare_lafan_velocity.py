"""Acquire the complete Geno LAFAN1 walk/run/sprint sources for local OpenSprite retargeting."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "data/OpenSprite-SuperMimic"
sys.path.insert(0, str(UPSTREAM / "scripts"))
from download_lafan_walks import RemoteZip, URL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=UPSTREAM / "data/sources/lafan1-resolved")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(RemoteZip()) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            name = Path(info.filename).name
            if not re.fullmatch(r"(walk|run|sprint)\d+_subject\d+\.bvh", name):
                continue
            path = args.output / name
            if path.exists():
                content = path.read_bytes()
            else:
                content = archive.read(info)
            if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
                raise ValueError(f"Size or CRC mismatch: {name}")
            if not path.exists():
                temp = path.with_suffix(".bvh.part")
                temp.write_bytes(content)
                temp.replace(path)
            manifest.append({"file": name, "archive_entry": info.filename,
                             "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            print(name, len(content), flush=True)
    if not manifest:
        raise ValueError("No matching complete sequences")
    (args.output / "paper_locomotion_sources.json").write_text(json.dumps(
        {"url": URL, "selection": "whole walk/run/sprint recordings", "sources": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
