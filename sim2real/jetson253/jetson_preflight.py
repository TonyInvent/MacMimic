"""Run the MacMimic no-CAN checks on Jetson using recorded feedback."""
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

stage = Path(__file__).resolve().parent
runtime = stage / "upstream/runtime"
sys.path.insert(0, str(runtime / "tools"))
from run_supermimic_commissioning import (prepare, HARDWARE_SHA, LIMITS_SHA, MATCHED_MAPPING_SHA,
                                          MATCHED_PROFILE_SHA, apply_training_gains, startup_joint_roles)
from open_sprite_runtime.contracts import PolicyContract
from open_sprite_runtime.supermimic_replay import NATIVE_RECORD, verify_case, file_sha256
from open_sprite_runtime.supermimic_run002 import verify_bundle

def check(condition, message):
    if not condition:
        raise ValueError(message)

root = Path.home() / "supermimic-run002-20261106"
baseline = Path.home() / "sprite-sm-walk-r1-xbox300-20261119"
bundle = stage / "verified-bundle"
contract = PolicyContract.load(bundle / "contract.json")
report = verify_bundle(bundle)
report["runtime_revision"] = "fe4e74f96117debbd66e5511af256418e1edcd23"
output = Path(tempfile.mkdtemp(prefix="preflight-", dir=stage))
args = Namespace(runtime_root=runtime, duration=10,
    hardware=root / "live-shadow-1700/hardware.json",
    joint_limits=root / "strict-sm-20261109/sprite0825_urdf_limit_candidates.json")
check(file_sha256(args.hardware)==HARDWARE_SHA and file_sha256(args.joint_limits)==LIMITS_SHA,"Robot configuration mismatch")
reviewed = root / "walk025-sync-en-1111/reviewed-profile"
for name, digest in MATCHED_PROFILE_SHA.items():
    check(file_sha256(reviewed/name)==digest, name)
prepare(args,output,contract)
for name, digest in MATCHED_MAPPING_SHA.items():
    check(file_sha256(output/name)==digest, "Mapping mismatch: "+name)
(output/"reviewed-joints.tsv").write_bytes((reviewed/"joints.tsv").read_bytes())
target = apply_training_gains(output/"reviewed-joints.tsv", (reviewed/"target.bin").read_bytes(), contract)
check(len(target)>0,"Empty target profile")
with open("/dev/input/js0","rb",buffering=0) as joystick:
    axes=fcntl.ioctl(joystick,0x80016a11,bytes(1))[0]
    axis_map=list(fcntl.ioctl(joystick,0x80406a32,bytes(64))[:axes])
    check(axis_map[0]==0, "Joystick axis 0 must be ABS_X")
report["linux_js_axis_map"]=axis_map
trace=subprocess.check_output([stage/"build/test_gamepad_control","--trace-lateral"],text=True)
vy=np.array([float(row.split()[1]) for row in trace.splitlines() if row.startswith("LATERAL_SAMPLE ")])
check(len(vy)==330 and max(vy)==.10 and min(vy)==-.10 and vy[-1]==0, "Lateral command sequence failed")
states=np.fromfile(baseline/"disabled-04/native.bin.inputs",dtype="<f8").reshape(-1,71)[:len(vy)].copy()
states[:,68:]=0
states[:,69]=vy
case=verify_case(contract,states,stage/"build/sprite_supermimic_replay",output/"replay",
                 handoff=True,benchmark=True,threads=2,warmup=100)
check(case["passed"],"Native/Python replay mismatch")
records=np.fromfile(output/"replay/native.bin",dtype=NATIVE_RECORD)
check(np.array_equal(records["obs"][:,-3:],states[:,-3:].astype("f")), "Xbox observation tail mismatch")
report.update(replay=case, lateral_m_s=[min(vy), max(vy)], final_vy=vy[-1], report_directory=str(output),
              physical_mapping="Stored calibration and limit files verified offline",
              device_input="Axis numbering only; stick-direction observations require operator movement")
(output/"preflight.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
