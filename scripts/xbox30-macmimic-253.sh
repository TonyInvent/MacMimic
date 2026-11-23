#!/bin/bash
# Run matched disabled, deadline and operator-controlled trials in one session.
set -euo pipefail
stage=/home/tony/macmimic-s49-100k-7947489
base=/home/tony/supermimic-run002-20261106
cd "$stage/xbox30-1"
[[ $(hostname -s) == tony-jetson && $(id -un) == tony && -t 0 && -t 1 ]]
python=/home/tony/open_sprite_runtime/.venv/bin/python
[[ -r /dev/input/js0 && $(cat /sys/module/kcan/srcversion) == 404E8D65A6EFF60079EAEC1 ]]
sha256sum --check trial.sha256
"$python" "$stage/commissioning_no_enable.py" check
export LD_LIBRARY_PATH="$PWD/build" PYTHONPATH="$PWD/runtime/src"
export SPRITE_INFERENCE_THREADS=2 SPRITE_NO_ORT_SPINNING=1
unset SPRITE_INFERENCE_DIAGNOSTICS
# Give only this launcher and its descendants a temporary RT priority limit.
# CAN selects FIFO 50 on CPU 2. Only the two compute threads select RR 1.
sudo prlimit --pid "$$" --rtprio=50:50
build/sprite_can_shadow --check-commissioning-scheduler 2
output="$PWD/trial-$(date +%Y%m%d-%H%M%S)"
mkdir "$output"
run_phase() {
  local phase=$1
  shift
  "$python" -u runtime/tools/run_supermimic_commissioning.py \
    --runtime-root "$PWD/runtime" --bundle "$stage/verified-bundle" \
    --hardware "$base/live-shadow-1700/hardware.json" \
    --joint-limits "$base/strict-sm-20261109/sprite0825_urdf_limit_candidates.json" \
    --build "$PWD/build" --icm "$base/zero-gain-1618/build/icm/sprite_icm45686" \
    --lock "$base/commissioning.lock" --output "$output/$phase" \
    --tracked-startup "$base/walk025-sync-en-1111/reviewed-profile" \
    --training-full-body --direct-policy-start --coherent-action-send --sling-window \
    --operator-status "$output/$phase-operator.log" --gamepad-control-device /dev/input/js0 \
    --cpu-frequency-socket /run/open-sprite-cpu-frequency/lease.sock "$@"
}
echo "Stand still with fall protection during the two disabled checks. Evidence: $output"
run_phase disabled
run_phase fault --deadline-fault
echo "Next: full-gain standing for 20 seconds, then a 30-second Xbox input window."
echo "Keep fall protection, clear the travel area, and keep the motor power cutoff within reach."
echo "After XBOX_WINDOW_OPEN, center sticks, release LT, then hold LT to move. Releasing LT brakes."
read -r -p "With all drives disabled and the robot supported and stationary, type XBOX30 to start: " ready
[[ $ready == XBOX30 ]] || exit 2
run_phase powered --enable-suspended --passed-disabled "$output/disabled" --passed-deadline-fault "$output/fault"
