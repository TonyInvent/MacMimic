#!/usr/bin/env bash
# Start the qualified OpenIMU choreography from the operator terminal after a battery swap.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $(hostname -s) == tony-jetson && $(id -un) == tony && -t 0 && -t 1 ]] || {
  echo "Run this script as tony at the Jetson operator terminal." >&2; exit 1;
}
[[ -r /dev/input/js0 ]] || { echo "Connect the Xbox controller and run again." >&2; exit 1; }
sudo bash scripts/load-kcan-253.sh
sudo python3 runtime/tools/openimu_interface.py --up
exec bash scripts/standing-groove-253.sh --powered canfd
