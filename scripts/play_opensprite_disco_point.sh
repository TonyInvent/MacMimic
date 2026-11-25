#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ "${1:-}" = "--headless" ]; then
    exec .venv/bin/python tools/play_opensprite_groove.py --disco-point "$@"
fi
exec .venv/bin/mjpython tools/play_opensprite_groove.py --disco-point "$@"
