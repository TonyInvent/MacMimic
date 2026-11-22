"""Pack feasible BONES-SEED and LAFAN1 references for the OpenSprite RL velocity run."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from supermimic_mlx.robots import opensprite

LAFAN = Path("data/OpenSprite-SuperMimic/outputs/paper-velocity-lafan")
SOURCES = Path("data/OpenSprite-SuperMimic/data/sources/lafan1-resolved/paper_locomotion_sources.json")
BONES = Path("data/bones-seed")


def build(output, conversions, smoke=False):
    files, skipped = [], []
    with (BONES / "velocity_candidates.csv").open() as reader:
        for row in csv.DictReader(reader):
            src = Path(row["move_g1_path"])
            report = conversions / src.parent.name / src.with_suffix(".json").name
            if not report.is_file():
                if report.with_suffix(".rejected.json").is_file() or smoke:
                    skipped.append(str(src))
                    continue
                raise ValueError(f"Pending BONES-SEED conversion: {src}")
            qa = json.loads(report.read_text())
            if qa["quality"]["self_minimum_signed_distance_m"] < -.01 or qa["output_frames"] < 50:
                skipped.append(str(src))
                continue
            files.append(("bones_"+src.stem,report.with_suffix(".npz")))
            if smoke and len(files) == 8:
                break
    bones_count = len(files)
    for source in json.loads(SOURCES.read_text())["sources"]:
        name = Path(source["file"]).stem
        manifest = LAFAN / name / "segments/windows.json"
        if not manifest.exists() and smoke:
            continue
        for window in json.loads(manifest.read_text())["windows"]:
            path = manifest.parent / window["file"]
            files.append(("lafan_"+path.stem,path))
            if smoke:
                break
    if not 0 < bones_count < len(files):
        raise ValueError("Expected both BONES-SEED and LAFAN1")
    lengths, arrays = [], []
    for name, path in files:
        with np.load(path, allow_pickle=False) as z:
            names = z["joint_names"].tolist()
            poses = z["qpos"].astype(np.float32, copy=False)
            if (names != list(opensprite.JOINT_NAMES) or str(z["asset_revision"]) != opensprite.source.ASSET_REVISION
                    or float(z["fps"]) != 50 or poses.shape[1:] != (38,) or not np.isfinite(poses).all()
                    or not np.allclose(np.linalg.norm(poses[:,3:7],axis=1), 1, atol=1e-5)):
                raise ValueError(f"Invalid reference: {name}")
            lengths.append(len(poses))
            arrays.append(poses)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".npz.tmp")
    with tmp.open("wb") as dest:
        np.savez(dest, qpos=np.concatenate(arrays), motion_names=[n for n,_ in files],
                 motion_lengths=lengths, fps=50., joint_names=opensprite.JOINT_NAMES,
                 asset_revision=opensprite.source.ASSET_REVISION)
    tmp.replace(output)
    summary = dict(bones_clips=bones_count, lafan_segments=len(files)-bones_count, frames=sum(lengths),
                   smoke_test=smoke, excluded_or_pending_bones=len(skipped))
    output.with_suffix(".json").write_text(json.dumps(dict(summary=summary,
        inputs=[str(p) for _,p in files], excluded_or_pending=skipped),indent=2)+"\n")
    print(summary,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--conversions",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    build(a.output,a.conversions,a.smoke)
