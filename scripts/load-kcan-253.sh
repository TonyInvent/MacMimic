#!/usr/bin/env bash
# Load the temporary echo-order KCAN driver for the OpenSprite Jetson.
set -euo pipefail

case "${1:-}" in
  ""|--check) ;;
  *) echo "Usage: bash load-kcan-253.sh [--check]" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "Unexpected arguments." >&2; exit 2; }

stage=/home/tony/sprite-sm-walk-r1-fast-20261119
loader="$stage/load-echo-order-driver.sh"
module="$stage/driver-echo-order/kcan.ko"
expected=404E8D65A6EFF60079EAEC1

[[ $(hostname -s) == tony-jetson && $(uname -r) == 5.15.199-tegra ]] || {
  echo "This loader requires tony-jetson running 5.15.199-tegra." >&2
  exit 1
}
sha256sum --check <<EOF
9aa0eb44f35714d072f149101e6c2aa6f9f2bafc07610a9d3f7159d648752d80  $loader
c8587bb874ad973d50ee59136ce6e3695af162cdf070e89673b84fe6c985528a  $module
EOF
[[ $(modinfo -F srcversion "$module") == "$expected" ]] || {
  echo "Candidate module version mismatch." >&2; exit 1
}
bash "$loader" --check
echo "Current KCAN $(cat /sys/module/kcan/srcversion 2>/dev/null || echo unloaded), requested $expected."
if [[ ${1:-} == --check ]]; then exit 0; fi
[[ $EUID -eq 0 ]] || { echo "Run the load step with sudo bash." >&2; exit 1; }

# The existing loader holds the shared commissioning lock during the switch.
bash "$loader"
[[ $(cat /sys/module/kcan/srcversion) == "$expected" ]] || {
  echo "Loaded module version mismatch." >&2; exit 1
}
echo "Loaded KCAN $expected."
ip -brief link show type can
