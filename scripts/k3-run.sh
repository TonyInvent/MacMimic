#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repo=$PWD
w="$repo/sim2real/build/k3"
export PYTHONPATH="$repo/runtime/src" OPENBLAS_NUM_THREADS=1 SPRITE_INFERENCE_THREADS=2 SPRITE_NO_ORT_SPINNING=1
unset SPRITE_INFERENCE_DIAGNOSTICS
case ${1:-} in
  xbox) args=(--gamepad-control-device /dev/input/js0); token=XBOX300 ;;
  standing-groove) args=(--standing-groove "$repo/policies/opensprite-standing-groove"); token=GROOVE12 ;;
  disco-point) args=(--disco-point "$repo/policies/opensprite-disco-point"); token=DISCO1640 ;;
  *) echo "Usage: $0 {xbox|standing-groove|disco-point} [--replay|--qualify|--powered]" >&2; exit 2 ;;
esac
[[ $# -le 2 ]]
case ${2:---replay} in
  --replay)
    if [[ $1 == xbox ]]; then
      out=$(mktemp -d "$w/replay-XXXXXX")
      "$w/build/sprite_supermimic_replay" policies/opensprite-velocity-100k/policy.onnx sim2real/evidence/xbox803-input.bin "$out/walk.bin" --threads 2 --warmup 100 --macmimic-100k
      exec python3 runtime/tools/verify_k3_walk.py "$out/walk.bin" sim2real/evidence/xbox803-jetson-reference.npz
    fi
    exec python3 runtime/tools/verify_standing_groove.py "policies/opensprite-$1" "$w/build/sprite_supermimic_protected_ipc" --motion "${1//-/_}" ;;
  --qualify|--powered) ;;
  *) exit 2 ;;
esac
python3 runtime/tools/k3_board.py
[[ -r /dev/input/js0 ]] || { echo "Connect the Xbox operator stop controller." >&2; exit 1; }
if [[ $(ulimit -r) != 50 ]]; then sudo prlimit --pid "$$" --rtprio=50:50; fi
cache="$w/qualification/$1"
mkdir -p "$cache"
exec 9>"$cache/operator.lock"
flock -n 9
out=$(mktemp -d "$w/$1-XXXXXX")
echo "Evidence: $out"
phase() {
  local name=$1
  shift
  python3 -u runtime/tools/run_supermimic_commissioning.py --k3 --runtime-root "$repo/runtime" --bundle "$repo/policies/opensprite-velocity-100k" \
    "${args[@]}" --hardware sim2real/hardware/hardware.json --joint-limits sim2real/hardware/joint-limits.json \
    --build "$w/build" --icm "$w/icm/sprite_icm45686" --imu-backend canfd --lock "$HOME/.macmimic-k3.lock" \
    --tracked-startup sim2real/hardware/reviewed-profile --training-full-body --direct-policy-start --coherent-action-send \
    --sling-window --operator-status "$out/$name-operator.log" --output "$out/$name" "$@"
}
if [[ $2 == --qualify ]]; then
  phase disabled
  phase fault --deadline-fault
  ln -sfnT "$out" "$cache/passed"
else
  [[ -t 0 && -t 1 && -f $cache/passed/fault/commissioning.json ]] || { echo "Run matching disabled qualification first." >&2; exit 1; }
  read -r -p "Sling supported, stationary, drives disabled and cutoff within reach? Type $token to enable: " ready
  [[ $ready == "$token" ]]
  phase powered --enable-suspended --passed-disabled "$cache/passed/disabled" --passed-deadline-fault "$cache/passed/fault"
fi
