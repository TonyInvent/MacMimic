#!/bin/bash
# Training benchmark on this Mac: SuperMimic, BlinkMimic and XMimic, one after the other on an otherwise idle machine.
# Each run is evaluated on all 40 clips at every checkpoint after it finishes (evaluation never shares the machine with
# training, so the training wall-clock is clean), then time to target and curves are summarized.
#
#   bash scripts/benchmark.sh logs/bench [seed]            # environment: SUPERMIMIC_HOURS (default 10)
set -u
cd "$(dirname "$0")/.."
OUT=${1:-logs/bench}
SEED=${2:-0}
SUPER_H=${SUPERMIMIC_HOURS:-10}
mkdir -p "$OUT"
run() {  # <name> <preset> [flags]
  local name=$1 preset=$2; shift 2
  local dir="$OUT/$name"
  if [ -f "$dir/done" ]; then echo "[bench] $name: already trained"; else
    echo "[bench] $(date '+%F %T') train $name"
    bash scripts/train.sh "$preset" "$dir" --seed "$SEED" "$@" > /dev/null
  fi
  echo "[bench] $(date '+%F %T') evaluate $name"
  PYTHONPATH=src uv run python -m supermimic_mlx.evaluate --run_dir "$dir" --threads 8 > "$dir/eval_stdout.log" 2>&1
  uv run python tools/time_to_target.py "$dir" | tee -a "$OUT/summary.txt"
}
run blinkmimic blinkmimic
run xmimic xmimic
run supermimic supermimic --iterations 16000 --max_hours "$SUPER_H" --ckpt_every 200 --ckpt_minutes 0
uv run python tools/plot_curves.py "$OUT/supermimic" "$OUT/blinkmimic" "$OUT/xmimic" --out "$OUT/curves.png"
echo "[bench] $(date '+%F %T') done"
