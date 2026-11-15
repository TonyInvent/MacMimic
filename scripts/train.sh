#!/bin/bash
# Train a preset. Usage: bash scripts/train.sh <supermimic|blinkmimic|instanttrack> <log_dir> [train flags, e.g. --seed 1]
set -eu
cd "$(dirname "$0")/.."
PRESET=${1:?usage: bash scripts/train.sh <preset> <log_dir> [flags]}; shift
LOG_DIR=${1:?usage: bash scripts/train.sh <preset> <log_dir> [flags]}; shift
mkdir -p "$LOG_DIR"
PYTHONPATH=src ${PYTHON:-uv run python} -u -m supermimic_mlx.train --preset "$PRESET" --log_dir "$LOG_DIR" "$@" 2>&1 \
  | tee "$LOG_DIR/stdout.log"
