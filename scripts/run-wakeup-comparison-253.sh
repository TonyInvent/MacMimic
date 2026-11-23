#!/bin/bash
# Run the no-enable comparison as the operator as soon as ftrace is ready.
set -euo pipefail
[[ $(hostname -s) == tony-jetson && $(id -un) == tony ]] || {
  echo "Run this script as tony on tony-jetson." >&2
  exit 2
}
cd /home/tony/macmimic-s49-100k-7947489
python=../open_sprite_runtime/.venv/bin/python
[[ -r /dev/input/js0 ]] || { echo "Connect the Xbox controller first." >&2; exit 2; }
"$python" commissioning_no_enable.py check
sudo -v
sudo -n bash ./trace-actor-wakeup-253-ab.sh --compare-gdm | (
  started=0
  result=0
  while IFS= read -r message; do
    printf '%s\n' "$message"
    if [[ $started == 0 && $message == 'READY: '* ]]; then
      started=1
      "$python" commissioning_no_enable.py disabled </dev/null || result=$?
    fi
  done
  (( started == 1 )) || exit 3
  exit "$result"
)
