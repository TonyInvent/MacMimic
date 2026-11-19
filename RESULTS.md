# Results on a MacBook Pro M4 Pro

Machine: MacBook Pro (Mac16,8), M4 Pro, 14 CPU cores (10 P + 4 E), 48 GB, macOS 26.5, automatic power mode. 4096
training worlds, 40 evaluation clips. The machine was shared with other workloads during these runs, so wall-clock
times carry a few percent of noise; iterations are exact.

## Time to target

Target: normalized evaluation return >= 0.82 with 0 of 40 clips failed. "First clean" is the first such checkpoint;
"strict" additionally asks for three consecutive checkpoints (100 iterations apart) and reports the first of them.

| Preset | Runs | Speed | First clean | Strict |
|---|---|---|---|---|
| `blinkmimic` | 1 | 0.61 it/s | 38.2 min (it 1400) | 44.0 min (it 1600) |
| `xmimic` | 1 | 0.65 it/s | 54.2 min (it 2100) | not reached in 57 min |
| `macmimic_v1` | 3 seeds | 0.94-0.97 it/s | 29.1 min (mean) | 29.2 / 30.1 / 27.9 min (it 1700 / 1700 / 1600), mean 29.05 |
| `macmimic` | 3 seeds | 1.09-1.11 it/s | 21.6 min (mean) | 21.3 / 21.0 / 22.5 min (it 1400 / 1400 / 1500), mean 21.62 |

`macmimic` reaches the strict target 2.04x faster than `blinkmimic` (44.0 / 21.6 min) and at least 2.6x faster than
`xmimic`; its first version, `macmimic_v1`, does it 1.52x / at least 1.95x faster. The baselines are single runs;
between seeds the strict iteration varies by about +-100 iterations (seed 2 of `macmimic`: 22.5 min, 1.95x on its own).
The `macmimic` runs (2026-11-18) were timed with the machine mostly idle (other processes under ~20% of one core at
their start).

`macmimic` differs from `macmimic_v1` in five settings (`--wm_lr_mult 8 --wm_batches 3 --ln_bf16 --lr_decay_by 800`,
and from iteration 400 a policy batch of 1280 with `--focus_replay 0.1`);
[docs/macmimic_hillclimb.md](docs/macmimic_hillclimb.md) has the reasoning and every variant tried.

The threshold flatters a run that has just crossed it. At equal tracking quality (the `blinkmimic` checkpoint at its
strict point: return 0.847, body position error 3.9 cm) `macmimic_v1` needs about 34-35 min, a gain of about 1.25x;
`macmimic` reaches 0.833 / 4.1 cm by 27.7 min (it 1800) and has not been run longer, so its equal-quality gain is
not measured.

## Tracking quality

`tools/eval_metrics.py`, upstream's protocol.

| Checkpoint | Train time | Return | Failed clips | Body position | Body orientation | Joints | Foot slip | Contact accuracy |
|---|---|---|---|---|---|---|---|---|
| `blinkmimic`, it 1600 | 44 min | 0.847 | 0 | 3.94 cm | 7.90 deg | 3.97 deg | 7.05 cm/s | 0.750 |
| `blinkmimic`, it 3500 | 101 min | 0.841 | 1 | 3.27 cm | 6.79 deg | 4.12 deg | 6.67 cm/s | 0.763 |
| `macmimic_v1` seed 0, it 1700 | 29 min | 0.834 | 0 | 4.09 cm | 8.04 deg | 4.19 deg | 7.29 cm/s | 0.743 |
| `macmimic_v1` seed 0, it 2000 | 34 min | 0.840 | 0 | 3.93 cm | 7.79 deg | 4.21 deg | 7.33 cm/s | 0.746 |
| `macmimic` seed 0, it 1400 | 21 min | 0.822 | 0 | 4.42 cm | 8.83 deg | 4.42 deg | 7.47 cm/s | 0.739 |
| `macmimic` seed 0, it 1800 | 28 min | 0.833 | 0 | 4.13 cm | 8.22 deg | 4.37 deg | 7.20 cm/s | 0.745 |
| BlinkMimic upstream (published), it 3500 | - | 0.862 | - | 3.44 cm | 7.02 deg | 4.19 deg | 6.66 cm/s | 0.765 |

The `blinkmimic` port at iteration 3500 matches upstream's published tracking errors; its return is 0.860 at
iteration 3400 and 0.841 at 3500, where one clip fails.

## Reproduce

```bash
bash scripts/train.sh macmimic logs/macmimic_s0 --seed 0
uv run python -m supermimic_mlx.evaluate --run_dir logs/macmimic_s0 --min_it 1000
uv run python tools/time_to_target.py logs/macmimic_s0
uv run python tools/eval_metrics.py logs/macmimic_s0/ckpt_001400 logs/macmimic_s0/ckpt_001800
```

`macmimic` reaches the strict target at iteration 1400-1500, so `--iterations 1800` is enough to see it.
`macmimic_v1` the same way, with `macmimic_v1` as the preset and the checkpoints of iterations 1700 and 2000.

How the presets were found, with every rejected variant: [docs/macmimic_hillclimb.md](docs/macmimic_hillclimb.md).
