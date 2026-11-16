#!/bin/bash
# Steady-state training throughput of configurations (before the probe sweeps start): 60 iterations each.
#   bash tools/throughput.sh "<flags A>" "<flags B>" ...      -> logs/hill/throughput.csv
cd "$(dirname "$0")/.."
mkdir -p logs/hill
for X in "$@"; do
  d=$(mktemp -d)
  out=$(PYTHONPATH=src .venv/bin/python -m supermimic_mlx.train $X --iterations 60 --log_every 20 --log_dir "$d" --ckpt_every 100000 2>&1)
  last=$(echo "$out" | grep "^it 60 ")
  ips=$(echo "$last" | grep -oE "[0-9.]+ it/s" | cut -d' ' -f1)
  ms=$(echo "$last" | grep -oE "collect [0-9]+ wm [0-9]+ pi [0-9]+" | tr ' ' ',')
  mem=$(echo "$last" | grep -oE "gpu peak [0-9.]+" | cut -d' ' -f3)
  echo "$X | ${ips:-ERR} it/s | $ms | peak ${mem} GB"
  echo "\"$X\",${ips:-},${mem:-}" >> logs/hill/throughput.csv
  [ -z "$ips" ] && echo "$out" | grep -E "Error|Traceback" | head -3
  rm -rf "$d"
done
