"""Evaluation return and failed clips over training wall-clock for runs with an eval.csv.

    uv run python tools/plot_curves.py logs/blinkmimic_s0 logs/supermimic_s0 --out docs/curves.png
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--out", default="curves.png")
p.add_argument("--thr", type=float, default=0.82)
a = p.parse_args()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for run in a.runs:
    rows = sorted((float(r["train_wall_h"]) * 60, float(r["norm_return"]), int(float(r["n_failed"])))
                  for r in csv.DictReader(open(os.path.join(run, "eval.csv"))))  # fmt: skip
    if not rows:
        continue
    preset = json.load(open(os.path.join(run, "args.json"))).get("preset", os.path.basename(run))
    t, ret, nf = zip(*rows)
    label = f"{preset} ({os.path.basename(run.rstrip('/'))})"
    ax1.plot(t, ret, label=label)
    ax2.plot(t, nf, label=label)
ax1.axhline(a.thr, color="gray", ls="--", lw=0.8)
ax1.set(xlabel="training wall-clock (min, M4 Pro)", ylabel="evaluation normalized return", ylim=(0, 1))
ax2.set(xlabel="training wall-clock (min, M4 Pro)", ylabel="failed clips (of 40)", ylim=(0, 41))
ax1.legend(), ax2.legend()
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(a.out, dpi=120)
print("wrote", a.out)
