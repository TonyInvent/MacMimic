#!/bin/bash
# Cheap sample-efficiency proxy: train to iteration 1000 (no probe sweeps yet), evaluate checkpoints 800-1000 (batched).
# Reference (BlinkMimic, seed 0): 800 0.721 | 900 0.760 | 1000 0.788.
#   bash tools/proxy.sh <name> <train flags...>      -> logs/hill/proxy.txt
cd "$(dirname "$0")/.."
NAME=$1; shift
D=logs/hill/p_$NAME
mkdir -p "$D"
PYTHONPATH=src .venv/bin/python -u -m supermimic_mlx.train --log_dir "$D" --iterations 1000 --ckpt_every 100 --ckpt_minutes 0 "$@" \
  > "$D/stdout.log" 2>&1
PYTHONPATH=src .venv/bin/python -m supermimic_mlx.evaluate --run_dir "$D" --batch 3 --min_it 800 --threads 14 > "$D/eval_stdout.log" 2>&1
ev=$(tail -n +2 "$D/eval.csv" | awk -F, '{printf "%s %.3f/%s (%.1fm) | ", $1, $5, $7, $2 * 60}')
ips=$(awk -F, 'NR>1 && $1>100 {s+=$15; n++} END {printf "%.2f", s/n}' "$D/train_log.csv")
echo "$NAME | $ips it/s | $ev nonfinite $(grep -c 'nonfinite\]' "$D/stdout.log") | $*" | tee -a logs/hill/proxy.txt
