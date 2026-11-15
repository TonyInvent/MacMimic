"""Apple-silicon hardware detection and automatic systems configuration.

Nothing here changes the algorithm; it picks the settings that only change speed and memory for the machine at hand
(M1 ... M6, base / Pro / Max / Ultra, any memory size):

- simulator threads: all CPU cores (performance + efficiency; mujoco.batch scales across both);
- matmul precision: bf16 (upstream's autocast) where this GPU runs bf16 matmuls at least about as fast as fp32 (M1 and
  M2 have no bf16 hardware, MLX emulates it); measured by a short micro-benchmark at startup;
- BPTT checkpointing: on when the estimated learner peak plus the replay buffer would not fit in the GPU's recommended
  working set;
- wired memory limit: the recommended working set, so MLX's buffers are not paged out (macOS 15+).
"""

from __future__ import annotations

import os
import subprocess
import time

import mlx.core as mx


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:
        return ""


def detect() -> dict:
    info = {}
    try:
        info.update({k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in mx.metal.device_info().items()})
    except Exception:
        try:
            info.update({k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in mx.device_info().items()})
        except Exception:
            pass
    info["chip"] = _sysctl("machdep.cpu.brand_string") or "unknown"
    p, e = _sysctl("hw.perflevel0.physicalcpu"), _sysctl("hw.perflevel1.physicalcpu")
    info["p_cores"] = int(p) if p.isdigit() else os.cpu_count() or 8
    info["e_cores"] = int(e) if e.isdigit() else 0
    gpu = _sysctl("machdep.gpu.core_count")  # not on every macOS version
    if gpu.isdigit():
        info["gpu_cores"] = int(gpu)
    mem = info.get("memory_size") or int(_sysctl("hw.memsize") or 0)
    info["memory_gb"] = round(mem / 2**30, 1)
    info["working_set_gb"] = round(info.get("max_recommended_working_set_size", 0.75 * mem) / 2**30, 1)
    return info


def matmul_tflops(dtype, n: int = 2048, m: int = 8192, reps: int = 8) -> float:
    a = mx.random.normal((m, n)).astype(dtype)
    w = mx.random.normal((n, n)).astype(dtype)
    mx.eval(a, w)
    for _ in range(2):
        mx.eval(a @ w)
    t0 = time.perf_counter()
    for _ in range(reps):
        mx.eval(a @ w)
    return 2 * m * n * n * reps / (time.perf_counter() - t0) / 1e12


def learner_peak_gb(pi_batch: int, horizon: int, wm_width: int, bf16: bool, checkpoint: bool) -> float:
    """Peak GPU memory of the BPTT policy update, from measurements on an M4 Pro (fp32: 9.2 GB at batch 2048 with a
    1024-wide world model, 21.7 GB at 4096 with 2048; bf16 ~0.7x)."""
    per = 110e3 + 30e3 * wm_width / 1024  # bytes per imagined sample-step
    gb = pi_batch * horizon * per / 2**30 * (0.7 if bf16 else 1.0)
    return gb / 4 if checkpoint else gb


def buffer_gb(num_envs: int, buffer_steps: int, J: int = 29, xi_dim: int = 33) -> float:
    return num_envs * buffer_steps * (4 * (2 * (13 + 2 * J) + 2 * J + xi_dim) + 9) / 2**30


def configure(args) -> dict:
    """Fill the 'auto' systems settings of the training arguments in place; returns the hardware record."""
    info = detect()
    if args.sim_threads is None:
        args.sim_threads = info["p_cores"] + info["e_cores"]
    if args.precision == "auto":
        f32, b16 = matmul_tflops(mx.float32), matmul_tflops(mx.bfloat16)
        info["tflops_fp32"], info["tflops_bf16"] = round(f32, 2), round(b16, 2)
        args.precision = "bf16" if b16 >= 0.95 * f32 else "fp32"
    wm_w = max(int(x) for x in args.wm_hidden.split(","))
    bf16 = args.precision == "bf16"
    need = learner_peak_gb(args.pi_batch, args.horizon, wm_w, bf16, False) + 1.5
    budget = info["working_set_gb"] - buffer_gb(args.num_envs, args.buffer_steps)
    if args.checkpoint_bptt is None:
        args.checkpoint_bptt = need > 0.9 * budget
    info["learner_peak_est_gb"] = round(learner_peak_gb(args.pi_batch, args.horizon, wm_w, bf16, args.checkpoint_bptt), 1)
    info["buffer_gb"] = round(buffer_gb(args.num_envs, args.buffer_steps), 1)
    if info["learner_peak_est_gb"] + 1.5 > 0.9 * budget:
        print(f"[hw] warning: the run needs ~{info['learner_peak_est_gb'] + info['buffer_gb'] + 1.5:.0f} GB, the GPU's "
              f"working set is {info['working_set_gb']} GB: expect swapping (reduce --pi_batch or --num_envs)", flush=True)  # fmt: skip
    try:
        ws = info.get("max_recommended_working_set_size")
        if ws:
            mx.set_wired_limit(int(ws))
            info["wired_limit_gb"] = round(ws / 2**30, 1)
    except Exception:
        pass
    return info
