#!/bin/bash
# Zero-command sling standing with the MacMimic contract gains.
set -euo pipefail
stage=/home/tony/macmimic-s49-100k-7947489
base=/home/tony/supermimic-run002-20261106
cd "$stage"
[[ $(hostname -s) == tony-jetson && $(id -un) == tony ]]
python=/home/tony/open_sprite_runtime/.venv/bin/python
"$python" commissioning_no_enable.py check
mode=${1:-}
extra=()
case "$mode" in
  disabled) [[ $# == 1 ]] ;;
  deadline-fault) [[ $# == 1 ]]; extra+=(--deadline-fault) ;;
  powered)
    [[ $# == 3 && -t 0 && -t 1 ]] || { echo "Run at the robot with both standing evidence directories." >&2; exit 2; }
    for evidence in "$2" "$3"; do test -f "$evidence/commissioning.json"; done
    extra+=(--enable-suspended --passed-disabled "$2" --passed-deadline-fault "$3")
    mapfile -t desktop < <(pgrep -u gdm -x gnome-shell)
    [[ ${#desktop[@]} == 1 ]]
    pid=${desktop[0]}
    start=$(awk '{print $22}' "/proc/$pid/stat")
    affinity=$(awk '/^Cpus_allowed_list:/ {print $2}' "/proc/$pid/status")
    [[ $affinity == 0-5 || $affinity == 0-1 ]]
    if [[ $affinity == 0-5 ]]; then
      sudo -v
      restore() {
        if [[ -r /proc/$pid/stat && $(awk '{print $22}' "/proc/$pid/stat") == "$start" ]]; then
          sudo -n taskset -pc "$affinity" "$pid"
        fi
      }
      trap restore EXIT
      sudo -n taskset -pc 0,1 "$pid"
    fi
    [[ $(awk '/^Cpus_allowed_list:/ {print $2}' "/proc/$pid/status") == 0-1 ]]
    echo "Powered zero-speed standing, full training gains. Keep fall protection and the physical motor cutoff within reach."
    read -r -p "With the robot supported and stationary, area clear and all drives disabled, type STAND to start: " ready
    [[ $ready == STAND ]] || exit 2 ;;
  *) echo "Usage: bash stand-macmimic-253.sh disabled|deadline-fault|powered [disabled_dir fault_dir]" >&2; exit 2 ;;
esac
out="$stage/stand-$mode-$(date +%Y%m%d-%H%M%S)"
echo "Evidence: $out"
export LD_LIBRARY_PATH="$stage/build" PYTHONPATH="$stage/upstream/runtime/src"
export SPRITE_INFERENCE_THREADS=2 SPRITE_NO_ORT_SPINNING=1
unset SPRITE_INFERENCE_DIAGNOSTICS
"$python" -u "$stage/upstream/runtime/tools/run_supermimic_commissioning.py" \
  --runtime-root "$stage/upstream/runtime" --bundle "$stage/verified-bundle" \
  --hardware "$base/live-shadow-1700/hardware.json" \
  --joint-limits "$base/strict-sm-20261109/sprite0825_urdf_limit_candidates.json" \
  --build "$stage/commissioning-bin" --icm "$base/zero-gain-1618/build/icm/sprite_icm45686" \
  --lock "$base/commissioning.lock" --output "$out" \
  --tracked-startup "$base/walk025-sync-en-1111/reviewed-profile" \
  --training-full-body --direct-policy-start --coherent-action-send --sling-window \
  --operator-status "$out-operator.log" --cpu-frequency-socket /run/open-sprite-cpu-frequency/lease.sock \
  "${extra[@]}"
