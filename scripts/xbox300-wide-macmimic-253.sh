#!/bin/bash
# Reuse matching no-enable qualification for the 300-second Xbox profile.
set -euo pipefail
stage=/home/tony/macmimic-s49-100k-7947489
base=/home/tony/supermimic-run002-20261106
cd "$stage/xbox300-wide-1"
case "${1:-}" in
  ""|--qualify|--check-cache) ;;
  *) echo "Usage: bash operator.sh [--qualify|--check-cache]" >&2; exit 2 ;;
esac
[[ $# -le 1 ]]
[[ $(hostname -s) == tony-jetson && $(id -un) == tony  ]]
python=/home/tony/open_sprite_runtime/.venv/bin/python
if [[ -z ${1:-} && ( ! -t 0 || ! -t 1 ) ]]; then
  echo "Start the powered trial at the operator terminal." >&2
  exit 2
fi
if [[ ${1:-} != --check-cache && ! -r /dev/input/js0 ]]; then
  echo "Connect the Xbox controller before starting." >&2
  exit 2
fi
sha256sum --check trial.sha256
"$python" "$stage/commissioning_no_enable.py" check
export LD_LIBRARY_PATH="$PWD/build" PYTHONPATH="$PWD/runtime/src"
export SPRITE_INFERENCE_THREADS=2 SPRITE_NO_ORT_SPINNING=1
unset SPRITE_INFERENCE_DIAGNOSTICS
# Give only this launcher and its descendants a temporary RT priority limit.
# CAN selects FIFO 50 on CPU 2. Only the two compute threads select RR 1.
if [[ $(ulimit -r) != 50 ]]; then sudo prlimit --pid "$$" --rtprio=50:50; fi
build/sprite_can_shadow --check-commissioning-scheduler 2
mkdir -p qualification
exec 9>qualification/operator.lock
flock -n 9 || { echo "Another Xbox 300 session is active." >&2; exit 1; }
output=$PWD
new_output() {
  if [[ $output == "$PWD" ]]; then
    output=$(mktemp -d "$PWD/trial-$(date +%Y%m%d-%H%M%S)-XXXX")
    echo "Evidence: $output"
  fi
}
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
ensure_phase() {
  local phase=$1 evidence_flag=$2
  shift 2
  local cached="$PWD/qualification/$phase"
  if [[ -f $cached/commissioning.json ]] &&
     run_phase check --check-saved-evidence "$evidence_flag" "$cached"; then
    echo "Reusing $phase: $(readlink -f "$cached")"
    return
  fi
  if [[ $operation == --check-cache ]]; then
    echo "No matching saved $phase qualification." >&2
    return 1
  fi
  new_output
  echo "Running $phase once for this build. Keep the robot supported and stationary."
  run_phase "$phase" "$@"
  ln -sfnT "$output/$phase" "$cached"
}
operation=${1:-}
ensure_phase disabled --passed-disabled
ensure_phase fault --passed-deadline-fault --deadline-fault
if [[ -n $operation ]]; then exit 0; fi
echo "Command limits: forward +0.70, reverse -0.50, lateral +/-0.42 m/s; yaw +/-0.70 rad/s. Full-speed forward braking takes up to 4 seconds."
echo "Full-gain standing for 20 seconds after IMU calibration, then a 300-second Xbox input window."
echo "Keep fall protection, clear the travel area, and keep the motor power cutoff within reach."
echo "After XBOX_WINDOW_OPEN, center sticks, release LT, then hold LT to move. Releasing LT brakes."
read -r -p "With the robot supported, stationary and drives disabled, type XBOX300 to start: " ready
[[ $ready == XBOX300 ]] || exit 2
new_output
run_phase powered --enable-suspended --passed-disabled "$PWD/qualification/disabled" --passed-deadline-fault "$PWD/qualification/fault"
