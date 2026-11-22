"""Verify BONES-SEED local bytes against the public Git tree of the official release."""
import argparse
import hashlib
import json
from pathlib import Path

REVISION = "2f59b2077b9da34dd4e43618e705c7cb962c9a66"
OFFICIAL = "https://huggingface.co/datasets/bones-studio/seed"
MIRROR = "https://modelscope.cn/datasets/bones-studio/seed"
# File size, Git blob ID, and whether that Git blob is an LFS pointer.
FILES = {
    "LICENSE.md": (21781, "a76739044205e2171bcfef654af57c34b2c7f476", False),
    "README.md": (16901, "3bafc6a6de418d1f9269d71141d8c9c37180cbab", False),
    "metadata/seed_metadata_v004.csv": (146411608, "653c9904238434074e9c44794093e6b84e7675b3", True),
    "g1.tar.gz": (23499973647, "1af40344d78baf58baee9a02e1693d09c60d35eb", True),
}


def verify(path, size, oid, is_lfs):
    if path.stat().st_size != size:
        raise ValueError(f"Unexpected size: {path}")
    with path.open("rb") as src:
        digest = hashlib.file_digest(src, "sha256").hexdigest()
    if is_lfs:
        blob = f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {size}\n".encode()
    else:
        blob = path.read_bytes()
    git_oid = hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest()
    if git_oid != oid:
        raise ValueError(f"File content differs from official Git object: {path}")
    return {"bytes": size, "sha256": digest, "official_git_blob": git_oid, "git_lfs": is_lfs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--source", choices=("huggingface", "modelscope"), required=True)
    args = parser.parse_args()
    report = {"official_dataset": OFFICIAL, "official_revision": REVISION,
              "downloaded_from": {"huggingface": OFFICIAL, "modelscope": MIRROR}[args.source], "files": {}}
    for name, spec in FILES.items():
        report["files"][name] = verify(args.data / name, *spec)
        print(f"Matched official Git object: {name}", flush=True)
    tmp = args.data / "source.json.tmp"
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(args.data / "source.json")
