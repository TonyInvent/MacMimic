"""Prepare a bound no-enable job or check prerequisites without CAN access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

stage = Path(__file__).resolve().parent
runtime = stage / "upstream/runtime"
sys.path.insert(0, str(runtime / "src"))
from open_sprite_runtime.supermimic_run002 import verify_bundle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("operation", choices=("check", "disabled", "deadline-fault"), default="check", nargs="?")
args = parser.parse_args()
bundle = stage / "verified-bundle"
binding = verify_bundle(bundle)
if binding["policy_sha256"] != "2ab5220171e9119028194ce6f0e0700beecfa87deb1af0d8b882340a3ae2de34":
    sys.exit("MacMimic 100k model required")
preflight = json.loads((stage/"preflight-3v3v6y_6/preflight.json").read_text())
if not preflight["replay"]["passed"] or preflight["policy_sha256"] != binding["policy_sha256"]:
    sys.exit("Matching replay evidence required")
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
can = stage/"commissioning-bin/sprite_can_shadow"
if digest(can) != "d94fd40af0d27801fcb15ffb50b4081d3f735b7a612e142d5f52af52e3b59826":
    sys.exit("CAN executable mismatch")
subprocess.run([can, "--check-commissioning-scheduler","2"],check=True)
driver = Path("/sys/module/kcan/srcversion").read_text().strip()
if driver != "404E8D65A6EFF60079EAEC1":
    sys.exit(f"KCAN driver mismatch: {driver}. The R1 echo-order driver is required.")
if args.operation == "check":
    print("Model, replay, CAN scheduler and driver checks passed. No motor frames sent.")
    sys.exit(0)
root=Path.home()/"supermimic-run002-20261106"
label=args.operation+"-"+datetime.now().strftime("%Y%m%d-%H%M%S")
env={**os.environ,"LD_LIBRARY_PATH":str(stage/"build"),"PYTHONPATH":str(runtime/"src"),
     "SPRITE_INFERENCE_THREADS":"2","SPRITE_NO_ORT_SPINNING":"1"}
env.pop("SPRITE_INFERENCE_DIAGNOSTICS",None)
command=[sys.executable,runtime/"tools/run_supermimic_commissioning.py",
    "--runtime-root",runtime,"--bundle",bundle,"--hardware",root/"live-shadow-1700/hardware.json",
    "--joint-limits",root/"strict-sm-20261109/sprite0825_urdf_limit_candidates.json",
    "--build",stage/"commissioning-bin","--icm",root/"zero-gain-1618/build/icm/sprite_icm45686",
    "--lock",root/"commissioning.lock","--output",stage/label,
    "--tracked-startup",root/"walk025-sync-en-1111/reviewed-profile",
    "--training-full-body","--direct-policy-start","--coherent-action-send","--sling-window",
    "--operator-status",stage/(label+"-operator.log"),"--gamepad-control-device","/dev/input/js0",
    "--cpu-frequency-socket","/run/open-sprite-cpu-frequency/lease.sock"]
if args.operation=="deadline-fault":
    command.append("--deadline-fault")
os.execve(sys.executable,list(map(str,command)),env)
