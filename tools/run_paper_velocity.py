"""Finish prepared references, train Appendix G on OpenSprite, and score the final velocity policy.

Launch from the project root after preparing both source manifests. The BONES and
LAFAN retarget queues can run independently. If a queue has exited with missing
outputs, resume those conversions before assembling the fixed training bank.
"""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import is_zipfile

from retarget_lafan_velocity import OUT, SOURCE, UPSTREAM

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "logs/opensprite_paper_velocity/rl_s49"
UMR = UPSTREAM / ".venv-umr/bin/python"
PYTHON = ROOT / ".venv/bin/python"
BONES = ROOT / "data/bones-seed"
BANK = ROOT / "data/opensprite-paper-velocity/locomotion.npz"
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")+os.pathsep+str(ROOT / "tools"),
       "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def running(pattern):
    result = subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL)
    if result.returncode not in (0, 1):
        raise RuntimeError("Could not check for existing retarget workers")
    return result.returncode == 0


def run(cmd, log=None, cwd=ROOT):
    if log:
        with log.open("a") as out:
            subprocess.run(cmd, check=True, env=ENV, cwd=cwd, stdout=out, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, env=ENV, cwd=cwd)


def prepare():
    sources = [Path(s["file"]).stem for s in json.loads((SOURCE/"paper_locomotion_sources.json").read_text())["sources"]]
    digest = hashlib.sha256((ROOT/"tools/retarget_bones_g1.py").read_bytes()
                            + (ROOT/"tools/bones_velocity_stats.py").read_bytes()).hexdigest()
    conversions = BONES/"opensprite-velocity"/digest[:12]
    with (BONES/"velocity_candidates.csv").open() as f:
        expected = sum(1 for _ in csv.DictReader(f))
    while True:
        for name in sources:
            native = OUT/name/"native.npz"
            if not (OUT/name/"segments/windows.json").exists() and is_zipfile(native):
                run([UMR, "-u", "tools/segment_lafan_velocity.py", name])
        complete = sum((OUT/name/"segments/windows.json").exists() for name in sources)
        count = sum(1 for _ in conversions.glob("*/*.json"))
        print(f"References: LAFAN1 {complete}/{len(sources)} recordings, BONES-SEED {count}/{expected} decisions",flush=True)
        if complete == len(sources) and count == expected:
            break
        if count < expected and not running("[r]etarget_bones_velocity.py"):
            run([UMR,"-u","tools/retarget_bones_velocity.py","--workers","3"])
        if complete < len(sources) and not running("[r]etarget_lafan_velocity.py|[s]cripts/retarget_lafan.py"):
            for name in sources:
                if not (OUT/name/"native.npz").exists():
                    run([UMR,"-u","scripts/retarget_lafan.py","--stage","retarget","--motions",name,
                         "--output-dir",str(OUT),"--correspondence-device","cpu"],cwd=UPSTREAM)
                    break
        time.sleep(30)
    run([PYTHON, "tools/build_paper_velocity_bank.py", BANK, "--conversions", conversions])


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not BANK.exists() or not BANK.with_suffix(".json").exists():
            prepare()
        if not (RUN/"done").exists():
            args = [PYTHON,"-u","-m","supermimic_mlx.train","--preset","macmimic","--log_dir",RUN,
                    "--opensprite_bank",BANK,"--velocity_commands","paper","--iterations","100000",
                    "--num_envs","4096","--seed","49","--precision","bf16","--eprobe_share","0",
                    "--focus_replay","0","--fail_reset","0","--ckpt_every","1000","--log_every","25"]
            if (RUN/"last.json").exists():
                args.extend(["--resume",RUN])
            print("Starting 100000-update Appendix G velocity training",flush=True)
            run(args,RUN/"stdout.log")
        if (RUN/"done").read_text().strip() != "100000":
            raise RuntimeError("Expected completion of all 100000 updates")
        run([PYTHON,"tools/export_opensprite.py",RUN/"ckpt_100000",RUN/"velocity_100000"])
        run([PYTHON,"tools/opensprite_validation/paper_steps.py",RUN/"velocity_100000",
             "--out",RUN/"paper_steps.json"])
        run([PYTHON,"tools/opensprite_control.py",RUN/"velocity_100000","--sequence","--out",RUN/"xbox_sequence.json"])


if __name__ == "__main__":
    main()
