#!/usr/bin/env bash
# Shared protected launcher for the two approved choreography packages.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo/sim2real/build/jetson253"
python=${SPRITE_PYTHON:-/home/tony/open_sprite_runtime/.venv/bin/python}
export LD_LIBRARY_PATH="$PWD/build" PYTHONPATH="$repo/runtime/src"
motion=standing_groove
label=groove
frames=600
confirmation=GROOVE12
if [[ ${1:-} == --disco-point ]]; then
  shift
  motion=disco_point
  label=disco-point
  frames=1640
  confirmation=DISCO1640
fi
flag="--${motion//_/-}"
package="$repo/policies/opensprite-${flag#--}"
operation=${1:---replay}
backend=${2:-canfd}
[[ $# -le 2 && ( $backend == canfd || $backend == spi ) ]] || exit 2
case "$operation" in
  --replay)
    exec "$python" "$repo/runtime/tools/verify_standing_groove.py" "$package" \
      "$PWD/build/sprite_supermimic_protected_ipc" --motion "$motion" --report "$PWD/$label-native-$frames.json" ;;
  --qualify|--powered) ;;
  *) echo "Usage: $0 [--replay|--qualify|--powered] [canfd|spi]" >&2; exit 2 ;;
esac
[[ $(hostname -s) == tony-jetson && $(id -un) == tony ]]
[[ -r /dev/input/js0 ]] || { echo "Connect the Xbox operator stop controller." >&2; exit 1; }
[[ $(cat /sys/module/kcan/srcversion) == 404E8D65A6EFF60079EAEC1 ]] || { echo "Load the echo-order KCAN driver." >&2; exit 1; }
sha256sum --check trial.sha256
export SPRITE_INFERENCE_THREADS=2 SPRITE_NO_ORT_SPINNING=1
unset SPRITE_INFERENCE_DIAGNOSTICS
if [[ $(ulimit -r) != 50 ]]; then sudo prlimit --pid "$$" --rtprio=50:50; fi
cache="$PWD/qualification/$label-$backend"
mkdir -p "$cache"
exec 9>"$cache/operator.lock"
flock -n 9 || { echo "Another groove session is active." >&2; exit 1; }
out=$(mktemp -d "$PWD/$label-XXXXXX")
echo "Evidence: $out"
phase() {
  local name=$1
  shift
  "$python" -u "$repo/runtime/tools/run_supermimic_commissioning.py" \
    --runtime-root "$repo/runtime" --bundle "$repo/policies/opensprite-velocity-100k" \
    "$flag" "$package" --hardware "$repo/sim2real/hardware/hardware.json" \
    --joint-limits "$repo/sim2real/hardware/joint-limits.json" --build "$PWD/build" \
    --icm "$PWD/icm/sprite_icm45686" --imu-backend "$backend" \
    --lock /home/tony/supermimic-run002-20261106/commissioning.lock --output "$out/$name" \
    --tracked-startup "$repo/sim2real/hardware/reviewed-profile" --training-full-body \
    --direct-policy-start --coherent-action-send --sling-window --operator-status "$out/$name-operator.log" \
    --cpu-frequency-socket /run/open-sprite-cpu-frequency/lease.sock "$@"
}
if [[ $operation == --qualify ]]; then
  phase disabled
  phase fault --deadline-fault
  ln -sfnT "$out" "$cache/passed"
  exit 0
fi
[[ -t 0 && -t 1 && -f $cache/passed/fault/commissioning.json ]] || {
  echo "Use the operator terminal after --qualify passes for this IMU." >&2; exit 1;
}
echo "Retain sling support and keep the cutoff in reach. Holding LB+RB for 3 seconds disables all drives."
read -r -p "Supported, stationary and drives disabled? Type $confirmation to enable: " ready
[[ $ready == "$confirmation" ]] || exit 2
phase powered --enable-suspended --passed-disabled "$cache/passed/disabled" --passed-deadline-fault "$cache/passed/fault"
