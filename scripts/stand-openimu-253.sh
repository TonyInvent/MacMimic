#!/usr/bin/env bash
# Full-gain zero-command standing with the selected ICM-45686 backend.
set -euo pipefail
imu_backend=canfd
imu_options=()
while (($#)); do
  case "$1" in
    --imu-backend) [[ ${2:-} == canfd || ${2:-} == spi ]] || exit 2; imu_backend=$2; shift ;;
    --imu-can-interface) [[ -n ${2:-} ]] || exit 2; imu_options+=(--imu-can-interface "$2"); shift ;;
    *) echo "Usage: $0 [--imu-backend canfd|spi] [--imu-can-interface IFACE]" >&2; exit 2 ;;
  esac
  shift
done
[[ $imu_backend == canfd || ${#imu_options[@]} == 0 ]] || exit 2
repo=$(cd "$(dirname "$0")/.." && pwd)
runtime="$repo/runtime"
hardware="$repo/sim2real/hardware"
cd "$repo/sim2real/build/jetson253"
[[ $(hostname -s) == tony-jetson && $(id -un) == tony && $# == 0 && -t 0 && -t 1 ]] || {
  echo "Run this standing test at the robot terminal as tony." >&2; exit 2;
}
sha256sum -c trial.sha256
[[ $(cat /sys/module/kcan/srcversion) == 404E8D65A6EFF60079EAEC1 ]]
for interface in kcan{1..4}; do
  usb=$(readlink -f "/sys/class/net/$interface/device")/..
  [[ $(cat "$usb/idVendor") == 395e && $(cat "$usb/idProduct") == 0020 ]] || {
    echo "Unexpected motor adapter for $interface" >&2; exit 1;
  }
done
python=${SPRITE_PYTHON:-/home/tony/open_sprite_runtime/.venv/bin/python}
export LD_LIBRARY_PATH="$PWD/build" PYTHONPATH="$runtime/src"
export SPRITE_INFERENCE_THREADS=2 SPRITE_NO_ORT_SPINNING=1
unset SPRITE_INFERENCE_DIAGNOSTICS
if [[ $(ulimit -r) != 50 ]]; then sudo prlimit --pid "$$" --rtprio=50:50; fi
build/sprite_can_shadow --check-commissioning-scheduler 2
out=$(mktemp -d "$PWD/stand-XXXXXX")
echo "Standing evidence: $out"
run_phase() {
  local phase=$1
  shift
  "$python" -u "$runtime/tools/run_supermimic_commissioning.py" \
    --runtime-root "$runtime" --bundle "$repo/policies/opensprite-velocity-100k" \
    --hardware "$hardware/hardware.json" --joint-limits "$hardware/joint-limits.json" \
    --build "$PWD/build" --icm "$PWD/icm/sprite_icm45686" --imu-backend "$imu_backend" "${imu_options[@]}" \
    --lock /home/tony/supermimic-run002-20261106/commissioning.lock --output "$out/$phase" \
    --tracked-startup "$hardware/reviewed-profile" --training-full-body --direct-policy-start \
    --coherent-action-send --sling-window --operator-status "$out/$phase-operator.log" \
    --cpu-frequency-socket /run/open-sprite-cpu-frequency/lease.sock "$@"
}
run_phase disabled
run_phase fault --deadline-fault
mapfile -t desktop < <(pgrep -u gdm -x gnome-shell)
if (( ${#desktop[@]} == 1 )); then
  pid=${desktop[0]}
  affinity=$(awk '/^Cpus_allowed_list:/ {print $2}' "/proc/$pid/status")
  if [[ $affinity == 0-5 ]]; then
    start=$(awk '{print $22}' "/proc/$pid/stat")
    sudo -v
    restore_desktop() {
      if [[ -r /proc/$pid/stat && $(awk '{print $22}' "/proc/$pid/stat") == "$start" ]]; then
        sudo -n taskset -pc "$affinity" "$pid"
      fi
    }
    trap restore_desktop EXIT
    sudo -n taskset -pc 0,1 "$pid"
  fi
  [[ $(awk '/^Cpus_allowed_list:/ {print $2}' "/proc/$pid/status") == 0-1 ]]
elif (( ${#desktop[@]} > 1 )); then
  echo "Ambiguous desktop process affinity." >&2; exit 1
fi
read -r -p "Supported and stationary, drives disabled, area clear and cutoff in reach? Type STAND for full-gain zero-speed standing: " ready
[[ $ready == STAND ]] || exit 2
run_phase powered --enable-suspended --passed-disabled "$out/disabled" --passed-deadline-fault "$out/fault"
