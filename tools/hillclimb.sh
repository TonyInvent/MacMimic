#!/bin/bash
# One hill-climbing trial: train under a wall-clock budget, evaluate every checkpoint (batched), time to target.
#   bash tools/hillclimb.sh <name> <budget_min> <train flags...>      -> logs/hill/<name>/, logs/hill/results.txt
cd "$(dirname "$0")/.."
NAME=$1; BUDGET=$2; shift 2
D=logs/hill/$NAME
mkdir -p "$D"
echo "[hill] $(date '+%T') $NAME: $*" | tee -a logs/hill/progress.log
PYTHONPATH=src .venv/bin/python -u -m supermimic_mlx.train --log_dir "$D" --max_hours "$(echo "$BUDGET / 60" | bc -l)" \
  --ckpt_every 100 --ckpt_minutes 0 "$@" > "$D/stdout.log" 2>&1
echo "[hill] $(date '+%T') $NAME: trained $(grep '^it' "$D/stdout.log" | tail -1 | cut -c1-24), nonfinite $(grep -c 'nonfinite\]' "$D/stdout.log")" | tee -a logs/hill/progress.log
PYTHONPATH=src .venv/bin/python -m supermimic_mlx.evaluate --run_dir "$D" --batch 40 --threads 14 > "$D/eval_stdout.log" 2>&1
r=$(uv run python tools/time_to_target.py "$D" | head -1)
ips=$(awk -F, 'NR>1 {s+=$15; n++} END {printf "%.2f", s/n}' "$D/train_log.csv")
best=$(awk -F, 'NR>1 && $5>b {b=$5; i=$1} END {printf "best %.4f (it %s)", b, i}' "$D/eval.csv")
echo "$NAME | $ips it/s | $r | $best | $*" | tee -a logs/hill/results.txt
