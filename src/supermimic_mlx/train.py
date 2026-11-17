"""SuperMimic / BlinkMimic training on Apple silicon (SuperMimic paper, Algorithm A.1): collect -> fit the world model ->
BPTT policy + critic update through imagined rollouts.

Port of upstream BlinkMimic's train.py: the simulator is CPU MuJoCo (mujoco.batch), the networks and every
differentiable head run in MLX on the GPU. ``--preset supermimic`` is the SuperMimic baseline (upstream train.py without
flags), ``--preset blinkmimic`` the released BlinkMimic recipe (upstream scripts/presets.sh PRESET[main]), ``--preset xmimic``
XMimic (github.com/nico-bohlinger/xmimic): BlinkMimic's algorithm and hyperparameters with XMimic's systems changes mapped
to Apple silicon (--overlap: actor | learner, --fast_learner); individual flags override the preset. Upstream's CUDA-only systems flags (CUDA graphs, fused kernels, host-sync removal, background
streams) have no MLX counterpart and are absent; mx.compile plays their role.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRESETS = {
    # SuperMimic (paper hyperparameters, as upstream train.py's defaults)
    "supermimic": {},
    # BlinkMimic (upstream PRESET[main], without the CUDA systems flags)
    "blinkmimic": dict(pi_ema=0.995, lr=4e-4, wm_batches=5, wm_hidden="1024,1024,1024", fail_reset=0.3, contact_reward=0.25,
                       wm_structured=True, wm_mirror=0.5, pi_opt="muon", wm_opt="muon", pi_ref_skip=True, wm_only_iters=0,
                       bin_fix=True, ref_contact_fix=True, ckpt_minutes=0, eprobe_share=0.15, pi_batch=2048, pi_lr_mult=4.0,
                       lr_decay_ret=0.75, lr_decay_len=300, iterations=3500, ckpt_every=100, eprobe_async=20,
                       eprobe_start=1000, eprobe_every=125, eprobe_decay=0.7, lr_decay_by=1400),
    # InstantTrack (upstream PRESET[instanttrack])
    "xmimic": {},  # filled below: blinkmimic + overlap
    "instanttrack": dict(pi_ema=0.995, lr=4e-4, wm_batches=5, wm_hidden="1024,1024,1024", fail_reset=0.3, contact_reward=0.25),
}


PRESETS["xmimic"] = dict(PRESETS["blinkmimic"], overlap=True)
# MacMimic: XMimic re-budgeted for an Apple-silicon GPU, where the learner is compute-bound (the policy update costs
# linear in the batch) and a small GPU call from the actor thread queues behind the learner. Policy batch 1024 (-38%
# per iteration; the return ends ~0.01 lower, recovered by the earlier step-size decay), probes from it 400 with a
# focus share of 0.3, their termination in numpy and their sweep on a worker thread, a faster policy average (fewer
# late failed clips than 0.995 in 3 of 4 runs). See logs/hill/results.txt for the trials behind each setting.
PRESETS["macmimic"] = dict(PRESETS["xmimic"], pi_batch=1024, pi_ema=0.99, lr_decay_by=1100, iterations=2000,
                           eprobe_start=400, eprobe_every=60, eprobe_tile=600, eprobe_share=0.3, eprobe_lead_min=100,
                           eprobe_lead_max=200, eprobe_np_term=True, eprobe_thread=True)
# fine-tuning a pretrained (MuJoCo) run in SuperDex Physics: use with --init_from <run_dir>. Few worlds (SuperDex is
# ~15x slower per world), the world model adapts first (--wm_only_iters), the policy at the base learning rate; the
# source run's network flags are taken over by --init_from
PRESETS["finetune_superdex"] = dict(PRESETS["blinkmimic"], sim_backend="superdex", num_envs=256, wm_only_iters=50, pi_lr_mult=1.0,
                                    lr=1e-4, lr_decay_ret=0.0, lr_decay_by=0, eprobe_share=0.0, iterations=1000, ckpt_every=50,
                                    warmup_steps=200)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    a = p.add_argument
    a("--preset", type=str, default="supermimic", choices=sorted(PRESETS))
    a("--num_envs", type=int, default=4096)
    a("--iterations", type=int, default=320_000)
    a("--buffer_steps", type=int, default=1000)
    a("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
    a("--horizon", type=int, default=32)
    a("--goal_frames", type=int, default=10)
    a("--hist", type=int, default=10)
    a("--wm_window", type=int, default=8)
    a("--wm_batches", type=int, default=10)
    a("--wm_batch", type=int, default=2048)
    a("--pi_batch", type=int, default=4096)
    a("--lr", type=float, default=2e-4)
    a("--gamma", type=float, default=0.99)
    a("--lam", type=float, default=0.95)
    a("--action_noise", type=float, default=0.1)
    a("--critic_epochs", type=int, default=5)
    a("--critic_minibatches", type=int, default=4)
    a("--wm_hidden", type=str, default="2048,2048,2048")
    a("--enc_hidden", type=str, default="2048,1024,512,256")
    a("--dec_hidden", type=str, default="2048,1024,512,256")
    a("--fail_reset", type=float, default=0.0, help="share of resets one bin before recent termination points")
    a("--pi_beta1", type=float, default=0.9)
    a("--pi_mom", type=float, default=0.0,
      help="Muon momentum of the policy from iteration --pi_mom_from (0: the optimizer's 0.95 throughout). The update "
           "direction averages ~1 / (1 - momentum) batches: a longer average is a larger effective batch at no compute")
    a("--pi_mom_from", type=int, default=0, help="first iteration of --pi_mom")
    a("--pi_clip_value", type=float, default=0.1, help="element-wise policy gradient clip after the norm clip (<= 0: off)")
    a("--pi_ema", type=float, default=0.0, help="Polyak average of the policy (decay per update); checkpoints store it")
    a("--pi_ema_early", type=float, default=0.0,
      help="EMA decay before --pi_ema_switch (0: --pi_ema throughout); e.g. 0.99: the average follows the fast-improving "
           "policy more closely in the fit phase, then --pi_ema smooths the tail")
    a("--pi_ema_switch", type=int, default=0, help="iteration from which --pi_ema applies (with --pi_ema_early)")
    a("--pi_ema_shadow", type=str, default="",
      help="measurement only: comma-separated extra EMA decays (0 = the live policy) checkpointed to <log_dir>/shadow_<d>/ "
           "next to every checkpoint; training and the probes use --pi_ema only")
    a("--warmup_steps", type=int, default=64)
    a("--contact_reward", type=float, default=0.0, help="weight of the contact-matching reward in imagination")
    a("--wm_structured", action=argparse.BooleanOptionalAction, default=False)
    a("--wm_mirror", type=float, default=0.0, help="share of each world-model batch replaced by its mirror image")
    a("--pi_opt", type=str, default="adam", choices=["adam", "muon"])
    a("--wm_opt", type=str, default="adam", choices=["adam", "muon"])
    a("--wm_only_iters", type=int, default=500)
    a("--bin_fix", action=argparse.BooleanOptionalAction, default=False)
    a("--pi_lr_mult", type=float, default=1.0)
    a("--lr_decay_ret", type=float, default=0.0)
    a("--lr_decay_len", type=int, default=300)
    a("--lr_decay_by", type=int, default=0)
    a("--ref_contact_fix", action=argparse.BooleanOptionalAction, default=False)
    a("--pi_ref_skip", action=argparse.BooleanOptionalAction, default=False)
    a("--eprobe_share", type=float, default=0.0, help="evaluation-guided reset allocation (0 = off)")
    a("--eprobe_tile", type=int, default=300)
    a("--eprobe_lead_min", type=int, default=200)
    a("--eprobe_lead_max", type=int, default=450)
    a("--eprobe_every", type=int, default=250)
    a("--eprobe_start", type=int, default=1250)
    a("--eprobe_win", type=int, default=10)
    a("--eprobe_decay", type=float, default=0.5)
    a("--eprobe_async", type=int, default=0, help="sweep steps per iteration (0: the whole sweep at once)")
    a("--learner_qos", action=argparse.BooleanOptionalAction, default=False,
      help="--overlap, macOS: run the learner thread (the GPU's feeder) at user-interactive QoS so the simulator's "
           "threads cannot delay it")
    a("--learner_async", action=argparse.BooleanOptionalAction, default=False,
      help="--overlap: start evaluating the world-model updates (mx.async_eval) before the policy update's graph is "
           "built, so the GPU does not idle during the build")
    a("--presample", action=argparse.BooleanOptionalAction, default=True,
      help="--overlap: sample the next iteration's batches while the learner is still running (same buffer, same "
           "draws), so the GPU does not idle during the host-side gather")
    a("--pi_steps", type=int, default=1,
      help="policy (+ critic) updates per iteration, each on its own --pi_batch starts: more optimizer steps for the "
           "same imagination compute when paired with a smaller batch")
    a("--deficit_replay", type=float, default=0.0,
      help="share of the policy update's imagination starts drawn in proportion to exp(-deficit_temp * G_bin / G_max), "
           "G_bin the simulator's running return of the segment: the low-return segments get more of the imagination "
           "compute (the resets already favour them, but the buffer's frames stay close to uniform)")
    a("--deficit_temp", type=float, default=6.0, help="--deficit_replay: sharpness of the weights")
    a("--focus_replay", type=float, default=0.0,
      help="share of the policy update's imagination starts drawn from the windows before the probe sweeps' failures "
           "(0 = off): the reset focus of --eprobe_share applied to what the update trains on")
    a("--focus_replay_per", type=float, default=0.0,
      help="adaptive share: this much per failing probe world (failures per sweep, smoothed by --eprobe_decay), capped at "
           "--focus_replay (0: the fixed share --focus_replay)")
    a("--focus_replay_wm", type=float, default=0.0, help="the same share for the world-model windows")
    a("--focus_replay_win", type=int, default=2, help="bins before a probe failure that count as its replay window")
    a("--lr_decay_iters", type=int, default=250_000)
    a("--ckpt_every", type=int, default=2000)
    a("--ckpt_minutes", type=float, default=3.0)
    a("--max_hours", type=float, default=None)
    a("--log_every", type=int, default=50)
    a("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
    a("--seed", type=int, default=0)
    a("--resume", type=str, default=None, help="run directory with last/ to continue from")
    a("--snap_at", type=int, default=0,
      help="write a fork point after this many iterations: <log_dir>/snap with the full training state (as last/) and "
           "the replay buffer (~1 kB per transition); --resume <log_dir>/snap --log_dir <new> continues from it with the "
           "buffer restored, so variants of the later iterations can be compared without retraining the prefix")
    a("--sim_backend", type=str, default="mujoco", choices=["mujoco", "superdex"],
      help="simulator: CPU MuJoCo (mujoco.batch) or SuperDex Physics (supermimic_mlx.sdx, --sim_threads worker processes)")
    a("--init_from", type=str, default=None,
      help="fine-tuning: start from the networks (world model, policy + EMA, critic, normalizers) of this run's last "
           "checkpoint, with fresh optimizers and schedules; its network flags are taken over")
    a("--reset_critic", action="store_true", help="--resume: start a fresh critic (and its optimizer); debugging")
    a("--sim_threads", type=int, default=None, help="default: all CPU cores")
    a("--probe_sim_threads", type=int, default=None, help="the probe sweep's simulator threads (default: --sim_threads)")
    a("--overlap", action=argparse.BooleanOptionalAction, default=False,
      help="XMimic actor | learner: the world-model and policy updates of iteration i run on a GPU stream while the CPU "
           "simulator collects (and the probe sweep advances) with the policy and normalizers of iteration i - 1")
    a("--goal_stride", type=int, default=1, help="spacing of the K goal frames (supermimic_vibed v2: 5, ~0.9 s lookahead)")
    a("--critic_clamp", action=argparse.BooleanOptionalAction, default=False,
      help="clamp critic values to the maximum attainable discounted return (supermimic_vibed v2)")
    a("--pi_batch_warm", type=int, default=0,
      help="compute schedule: policy batch for iterations < --pi_batch_switch (0 = off); cheap updates while the world "
           "model and the policy ramp up, the full --pi_batch afterwards")
    a("--pi_steps_warm", type=int, default=0,
      help="compute schedule: --pi_steps before --pi_batch_switch (0 = off); e.g. --pi_batch_warm 512 --pi_steps_warm 2: "
           "twice the optimizer steps on half batches while progress is step-limited, full batches for the tail")
    a("--horizon_warm", type=int, default=0, help="compute schedule: imagination horizon before --pi_batch_switch (0 = off)")
    a("--wm_batches_warm", type=int, default=0, help="compute schedule: world-model updates per iteration before --pi_batch_switch (0 = off)")
    a("--pi_batch_switch", type=int, default=0, help="first iteration of the full compute (the end of the --*_warm settings)")
    a("--ns_bf16", action=argparse.BooleanOptionalAction, default=False,
      help="with --precision bf16: Muon's Newton-Schulz iteration in bf16 (as torch.optim.Muon). Off by default: no "
           "measurable speedup here (Muon is ~2%% of an iteration) and the validated runs used fp32")
    a("--eprobe_thread", action=argparse.BooleanOptionalAction, default=False,
      help="--overlap: run each iteration's probe-sweep chunk on a worker thread (own GPU stream) concurrently with the "
           "collection and the learner; same sweep, same results (the final chunk still precedes its collection)")
    a("--eprobe_np_term", action=argparse.BooleanOptionalAction, default=False,
      help="probe sweep: termination in numpy instead of the reward / termination graph (no GPU round trip; the probe "
           "never uses the reward)")
    a("--reward_cpu", action=argparse.BooleanOptionalAction, default=False,
      help="--overlap: compute the actor's and the probe sweep's reward / termination heads on the CPU while the policy "
           "forwards stay on the GPU (one GPU round trip per sequential sim step instead of two)")
    a("--probe_device", type=str, default="gpu", choices=["gpu", "cpu"],
      help="--overlap: run the probe sweep's policy forwards (a few hundred worlds) on the CPU, so each sequential sweep "
           "step does not queue behind the learner on the GPU (the 4096-world actor stays on --actor_device)")
    a("--actor_device", type=str, default="gpu", choices=["gpu", "cpu"],
      help="--overlap: run the actor's and the probe sweep's policy forwards on the CPU (fp32, MLX's CPU backend), so they "
           "do not queue behind the learner on the GPU")
    a("--fast_learner", action=argparse.BooleanOptionalAction, default=False,
      help="XMimic's learner hoisting: frames, goals and goal encodings of all imagined steps in one batch before the "
           "rollout; the windows of all world-model batches sampled at once and their updates in one compiled call")
    a("--precision", type=str, default="auto", choices=["fp32", "bf16", "auto"],
      help="MLP matmul precision (upstream autocasts to bf16); auto: measured on this GPU at startup")
    a("--profile", action="store_true", help="print a per-phase timing breakdown (adds synchronizations)")
    a("--checkpoint_bptt", action=argparse.BooleanOptionalAction, default=None,
      help="recompute each imagined step in the backward pass (saves memory; default: automatic from the GPU memory)")
    return p


def parse_args(argv=None):
    p = build_parser()
    pre, _ = p.parse_known_args(argv)
    p.set_defaults(**PRESETS[pre.preset])
    return p.parse_args(argv)


args = parse_args()
NET_KEYS = ("wm_hidden", "enc_hidden", "dec_hidden", "wm_structured", "pi_ref_skip", "hist", "goal_frames", "horizon", "goal_stride")
if args.init_from:  # the pretrained run fixes the network architecture
    _src = json.load(open(os.path.join(args.init_from, "args.json")))
    for _k in NET_KEYS:
        setattr(args, _k, _src[_k])

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

from supermimic_mlx.buffer import ReplayBuffer  # noqa: E402
from supermimic_mlx.checkpoint import load_tree, save_tree  # noqa: E402
from supermimic_mlx.kinematics import Kinematics  # noqa: E402
from supermimic_mlx.mdp import MAX_STEP_REWARD, MDP, flatten_state, inv_dim, unflatten_state  # noqa: E402
from supermimic_mlx.mirror import Mirror  # noqa: E402
from supermimic_mlx.motion import MotionLib  # noqa: E402
from supermimic_mlx import networks  # noqa: E402
from supermimic_mlx.networks import Critic, WorldModel, policy_from_args, state_error  # noqa: E402
from supermimic_mlx.optim import MuonAdam, clip_grad_norm, clip_grad_value, flat, unflat  # noqa: E402
from supermimic_mlx.robots import g1  # noqa: E402
from supermimic_mlx.scene import CONTROL_DT  # noqa: E402
from supermimic_mlx.sim import TrackingSim, XiLayout  # noqa: E402

PROF: dict = {}


class prof:
    """--profile: accumulate wall-clock of a block (ms) under a name; ``sync`` evaluates these arrays inside it."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *a):
        PROF[self.name] = PROF.get(self.name, 0.0) + 1000 * (time.perf_counter() - self.t)


def prof_eval(name_encode, name_wait, *xs):
    """Evaluate xs; with --profile split into the encode (async_eval returns) and the GPU wait."""
    if not args.profile:
        mx.eval(*xs)
        return
    with prof(name_encode):
        mx.async_eval(*xs)
    with prof(name_wait):
        mx.eval(*xs)


from supermimic_mlx import hw  # noqa: E402

HW = hw.configure(args)  # automatic systems settings for this Mac (threads, precision, checkpointing, wired memory)
print("[hw] " + ", ".join(f"{k} {v}" for k, v in HW.items() if k in ("chip", "p_cores", "e_cores", "gpu_cores", "memory_gb",
      "working_set_gb", "tflops_fp32", "tflops_bf16", "buffer_gb", "learner_peak_est_gb")), flush=True)  # fmt: skip
print(f"[hw] sim_threads {args.sim_threads}, precision {args.precision}, checkpoint_bptt {args.checkpoint_bptt}", flush=True)
if args.precision == "bf16":
    networks.set_compute_dtype(mx.bfloat16)
    import supermimic_mlx.optim as _optim

    if args.ns_bf16:
        _optim.NS_DTYPE = mx.bfloat16  # Muon's Newton-Schulz in bf16, as torch.optim.Muon
mx.random.seed(args.seed)
np.random.seed(args.seed)
os.makedirs(args.log_dir, exist_ok=True)
json.dump(dict(vars(args), hw=HW), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)
print("args:", json.dumps(vars(args)), flush=True)

J = len(g1.JOINT_NAMES)
kin = Kinematics()
motion = MotionLib.from_dir(g1.MOTION_DIR, kin, g1.load_motion)
layout = XiLayout()
mdp = MDP(kin, motion, layout.dim)
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = motion.goal_dim(K)
CRITIC_IN = inv_dim(J) + mdp.critic_extra + GOAL_DIM
FEET = mdp.feet
if args.ref_contact_fix:
    motion.set_ref_contact(FEET)
motion.goal_stride = args.goal_stride
widths = lambda s: tuple(int(x) for x in s.split(","))  # noqa: E731
CKPT_BPTT = bool(args.checkpoint_bptt)

wm = WorldModel(J, layout.dim, hidden=widths(args.wm_hidden), structured=args.wm_structured, dt=CONTROL_DT)
pi = policy_from_args(vars(args), mdp.obs_dim, GOAL_DIM, J)
# maximum per-step reward incl. the contact term (2 feet x 2 terms each up to 1): (4 + 2 x 0.5 + 2 w_c) dt
V_MAX = (MAX_STEP_REWARD + 2 * args.contact_reward * CONTROL_DT) / (1 - args.gamma)
critic = Critic(CRITIC_IN, v_max=V_MAX if args.critic_clamp else None)
mx.eval(wm.parameters(), pi.parameters(), critic.parameters(), ERR_FLOOR if "ERR_FLOOR" in globals() else ())
opt_wm = MuonAdam(flat(wm.trainable_parameters()), wm.muon_paths() if args.wm_opt == "muon" else (), betas=(0.9, 0.999))
opt_pi = MuonAdam(flat(pi.trainable_parameters()), pi.muon_paths() if args.pi_opt == "muon" else (), betas=(args.pi_beta1, 0.95))
opt_v = MuonAdam(flat(critic.trainable_parameters()), (), betas=(0.9, 0.999))
EMA = {"p": flat(pi.parameters()), "n": 0} if args.pi_ema > 0 else None  # Polyak average of all policy arrays (+ norms)
CTR = {"nonfinite": mx.zeros((), dtype=mx.int32)}
SHADOW = [{"d": float(x), "p": flat(pi.parameters())} for x in args.pi_ema_shadow.split(",") if x.strip()]
SHADOW = [sh for sh in SHADOW if sh["d"] > 0] + [sh for sh in SHADOW if sh["d"] == 0]  # live (0) needs no state
n_params = lambda m: sum(v.size for _, v in flat(m.trainable_parameters()).items())  # noqa: E731
print(f"world model {n_params(wm) / 1e6:.2f} M, policy {n_params(pi) / 1e6:.2f} M, critic {n_params(critic) / 1e6:.2f} M "
      f"params | goal {GOAL_DIM}, obs {mdp.obs_dim}, critic in {CRITIC_IN} | bptt checkpointing {CKPT_BPTT}", flush=True)  # fmt: skip

# every array a compiled step may read or write
STATE = [wm.state, pi.state, critic.state, opt_wm.state, opt_pi.state, opt_v.state, CTR, mx.random.state]
STATE += [sh["p"] for sh in SHADOW if sh["d"] > 0]
if EMA is not None:
    STATE.append(EMA["p"])

start_it, wall_offset, sim_steps = 0, 0.0, 0
RESUME_CTRL = None
if args.init_from:
    ck = load_tree(os.path.join(args.init_from, "last"))
    wm.update(unflat(ck["wm"])), pi.update(unflat(ck["pi_live"])), critic.update(unflat(ck["critic"]))
    if EMA is not None:
        EMA["p"].update(ck.get("pi_avg", ck["pi_live"]))
        EMA["n"] = 1  # continue the average from the pretrained one
    mx.eval(wm.parameters(), pi.parameters(), critic.parameters(), EMA["p"] if EMA is not None else ())
    print(f"--init_from {args.init_from}: networks of iteration {ck['meta']['iteration']} ({ck['meta']['args'].get('preset')})", flush=True)
    del ck
if args.resume:
    ck = load_tree(os.path.join(args.resume, "last"))
    wm.update(unflat(ck["wm"])), pi.update(unflat(ck["pi_live"]))
    if not args.reset_critic:
        critic.update(unflat(ck["critic"]))
    for o, k in ((opt_wm, "opt_wm"), (opt_pi, "opt_pi")) + (() if args.reset_critic else ((opt_v, "opt_v"),)):
        o.state["t"] = ck[k]["t"]
        for sub in ("m", "v", "buf"):
            o.state[sub].update(ck[k].get(sub, {}))
    if EMA is not None and "pi_avg" in ck:
        EMA["p"].update(ck["pi_avg"])
        EMA["n"] = int(ck["meta"]["ema_n"])
        for sh in SHADOW:  # the shadows are not checkpointed: they continue from the main average
            sh["p"].update(ck["pi_avg"])
    meta = ck["meta"]
    start_it, wall_offset, sim_steps = int(meta["iteration"]), float(meta["wall_time"]), int(meta["sim_steps"])
    RESUME_CTRL = meta.get("ctrl")
    print(f"--resume: iteration {start_it}, {wall_offset / 3600:.2f} h", flush=True)

# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = mx.array(np.concatenate((np.full(3, 3e-3), np.full(3, 5e-3), np.full(3, 2e-2), np.full(3, 5e-2),
                                     np.full(J, 3e-3), np.full(J, 5e-2))).astype(np.float32))  # fmt: skip
mirror = Mirror(layout.slices) if args.wm_mirror > 0 else None
N_MIRROR = int(args.wm_batch * args.wm_mirror)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


# ------------------------------------------------------------------------------------------------ world model update
def _wm_loss(s, s1, a, xi_n, mask):
    S = unflatten_state(s[:, 0])
    std = mx.maximum(wm.inc_norm.std(), ERR_FLOOR)
    es = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], xi_n[:, j])
        e = (state_error(S, unflatten_state(s1[:, j])) / std) ** 2
        es.append(e.mean(-1))
    e = mx.stack(es, 1)  # (B, k)
    return (e * mask).sum() / s.shape[0], e[:, -1]


_wm_vg = nn.value_and_grad(wm, _wm_loss)


def _mirror_windows(s, s1, a, xi):
    if mirror is None or N_MIRROR == 0:
        return s, s1, a, xi
    n = N_MIRROR
    ms = lambda x: flatten_state(mirror.state(unflatten_state(x)))  # noqa: E731
    cat = lambda fn, x: mx.concatenate((fn(x[:n]), x[n:]))  # noqa: E731
    return cat(ms, s), cat(ms, s1), cat(mirror.joints, a), cat(mirror.xi, xi)


def _wm_update(s, s1, a, xi, mask, lr):
    s, s1, a, xi = _mirror_windows(s, s1, a, xi)
    (loss, e_last), grads = _wm_vg(s, s1, a, layout.normalize(xi), mask)
    grads, _ = clip_grad_norm(flat(grads), 1.0)
    wm.update(unflat(opt_wm.step(flat(wm.trainable_parameters()), grads, lr)))
    m = mask[:, -1]
    return loss / args.wm_window, (e_last * m).sum() / mx.maximum(m.sum(), 1)


wm_update_c = mx.compile(_wm_update, inputs=STATE, outputs=STATE)


def _wm_update_all(s, s1, a, xi, mask, lr):
    """--fast_learner: the wm_batches updates of an iteration in one call, on windows sampled at once."""
    B = args.wm_batch
    outs = [_wm_update(s[i * B : (i + 1) * B], s1[i * B : (i + 1) * B], a[i * B : (i + 1) * B], xi[i * B : (i + 1) * B],
                       mask[i * B : (i + 1) * B], lr) for i in range(args.wm_batches)]  # fmt: skip
    return outs[-1]


wm_update_all_c = mx.compile(_wm_update_all, inputs=STATE, outputs=STATE)


# --focus_replay: per-frame sampling weights from the probe sweeps' failures (None until a sweep found one)
REPLAY = {"pi": None, "wm": None, "scale": 1.0}  # scale: the adaptive fraction of the configured shares
DEFICIT = {"n": 0, "w": None}  # --deficit_replay: per-frame start weights from the simulator's segment returns


def wm_inputs() -> list:
    """Host side of the world-model updates: the sampled windows of each batch (one list per compiled call)."""
    keys = ("s", "s1", "a", "xi", "mask")
    foc = lambda B: (int(round(args.focus_replay_wm * REPLAY["scale"] * B)), REPLAY["wm"]) if args.focus_replay_wm > 0 and REPLAY["wm"] is not None else None  # noqa: E731
    if args.fast_learner:
        with prof("wm.sample"):
            w = buffer.sample_windows(args.wm_batch * args.wm_batches, args.wm_window, foc(args.wm_batch * args.wm_batches))
        return [[mx.array(w[k]) for k in keys]]
    out = []
    with prof("wm.sample"):
        for _ in range(args.wm_batches):
            w = buffer.sample_windows(args.wm_batch, args.wm_window, foc(args.wm_batch))
            out.append([mx.array(w[k]) for k in keys])
    return out


def wm_run(inputs: list, lr):
    """The world-model updates of an iteration on sampled windows; returns the last (loss, 8-step error)."""
    fn = wm_update_all_c if args.fast_learner else wm_update_c
    for xs in inputs:
        out = fn(*xs, lr)
    return out


def wm_updates(lr):
    return wm_run(wm_inputs(), lr)


# ------------------------------------------------------------------------------------------------ policy update
def _imag_step(S, o_hist, f1, xi, xi_n, g):
    """One imagined step from S (goal g of its frame): action, world-model state, noisy observation history at f1."""
    a = pi(o_hist, g)
    S1 = wm.step(S, a, xi_n)
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    return S1, a, mx.concatenate((o_hist[:, 1:], o1[:, None]), 1)


def _flat_step(p, R, v, w, q, qd, o_hist, f1, xi, xi_n, g):
    """_imag_step on flat arguments (mx.checkpoint). Frame indices and goals are computed outside: every output of a
    checkpointed call depends on the parameters, and integer indices must not."""
    S1, a, o_hist = _imag_step((p, R, v, w, q, qd), o_hist, mx.stop_gradient(f1), xi, xi_n, g)
    return (*S1, a, o_hist)


def _critic_input(S, f, xi_n, g):
    return mx.concatenate((mdp.critic_features(S, f, xi_n), g), -1)


def actor_objective(r, d, v_boot, gamma):
    """SHAC objective J = sum_{t<H} gamma^t c_t r_t + gamma^H c_H V(z_H), c_t = prod_{k<t} (1 - d_k)."""
    H = r.shape[0]
    alive = mx.cumprod(1 - d, axis=0)
    c_t = mx.concatenate((mx.ones((1,) + r.shape[1:]), alive[:-1]), 0)
    disc = mx.array([gamma**t for t in range(H)])[:, None]
    J_ = (disc * c_t * r).sum(0) + (gamma**H) * alive[-1] * v_boot
    return J_, c_t, alive[-1]


def _pi_loss_fast(S, o_hist, a_prev, f, xi, xi_n):
    """--fast_learner (XMimic): frames, goals and goal encodings of all HZ + 1 steps computed before the rollout in one
    batch (they do not depend on the imagined state). Same objective as _pi_loss."""
    B = f.shape[0]
    f_all = mx.concatenate([f + t for t in range(HZ + 1)])
    G = motion.goal(f_all, K)
    Z = pi.encode(G).reshape(HZ + 1, B, -1)
    G = G.reshape(HZ + 1, B, -1)
    Fs = f_all.reshape(HZ + 1, B)
    Ss, As, Aps = [], [], []
    for t in range(HZ):
        Aps.append(a_prev)
        a_prev = pi(o_hist, G[t], z=Z[t])
        S = wm.step(S, a_prev, xi_n)
        o1 = mdp.add_noise(mdp.obs(S, a_prev, xi, Fs[t + 1]))
        o_hist = mx.concatenate((o_hist[:, 1:], o1[:, None]), 1)
        Ss.append(S), As.append(a_prev)
    flt = lambda xs: mx.concatenate(xs, 0)  # noqa: E731  (H*B, ...)
    S_all = tuple(flt([St[i] for St in Ss]) for i in range(6))
    fr = f_all[B:]
    r = mdp.reward(S_all, fr, flt(As), flt(Aps), w_contact=args.contact_reward).reshape(HZ, -1)
    d = mdp.soft_termination(S_all, fr).reshape(HZ, -1)
    v_boot = critic(_critic_input(S, Fs[HZ], xi_n, G[HZ]))
    J_, c_t, c = actor_objective(r, d, v_boot, args.gamma)
    aux = (mx.stop_gradient(r), mx.stop_gradient(d), mx.stop_gradient(c_t), mx.stop_gradient(c),
           tuple(mx.stop_gradient(mx.stack([St[i] for St in Ss])) for i in range(6)), Fs[1:], mx.stop_gradient(G[1:]),
           mx.stop_gradient(v_boot))  # fmt: skip
    return -J_.mean(), aux


def _pi_loss(S, o_hist, a_prev, f, xi, xi_n):
    g = motion.goal(f, K)
    step = nn.utils.checkpoint(pi, _flat_step) if CKPT_BPTT else None
    Ss, As, Aps, Fs, Gs = [], [], [], [], []
    for _ in range(HZ):
        Aps.append(a_prev)
        f1 = f + 1
        if CKPT_BPTT:
            out = step(*S, o_hist, f1, xi, xi_n, g)
            S, a_prev, o_hist = tuple(out[:6]), out[6], out[7]
        else:
            S, a_prev, o_hist = _imag_step(S, o_hist, f1, xi, xi_n, g)
        f, g = f1, motion.goal(f1, K)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
    flt = lambda xs: mx.concatenate(xs, 0)  # noqa: E731  (H*B, ...)
    S_all = tuple(flt([St[i] for St in Ss]) for i in range(6))
    f_all = flt(Fs)
    r = mdp.reward(S_all, f_all, flt(As), flt(Aps), w_contact=args.contact_reward).reshape(HZ, -1)
    d = mdp.soft_termination(S_all, f_all).reshape(HZ, -1)
    v_boot = critic(_critic_input(S, f, xi_n, g))
    J_, c_t, c = actor_objective(r, d, v_boot, args.gamma)
    aux = (mx.stop_gradient(r), mx.stop_gradient(d), mx.stop_gradient(c_t), mx.stop_gradient(c),
           tuple(mx.stop_gradient(mx.stack([St[i] for St in Ss])) for i in range(6)), mx.stack(Fs), mx.stack(Gs),
           mx.stop_gradient(v_boot))  # fmt: skip
    return -J_.mean(), aux


_pi_vg = nn.value_and_grad(pi, _pi_loss_fast if args.fast_learner and not CKPT_BPTT else _pi_loss)
_v_vg = nn.value_and_grad(critic, lambda Xn, Y: ((critic.head(Xn) - Y) ** 2).mean())  # on normalized inputs


def _gae_targets(V, rews, dones):
    R = V[-1]
    out = []
    for t in range(HZ - 1, -1, -1):
        R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
        out.append(R)
    return mx.stack(out[::-1])


def _policy_update(s, hs, ha, hxi, hf, xi, f, a_prev, lr_pi, lr_v, ema_w):
    S = unflatten_state(s)
    xi_n = layout.normalize(xi)
    o_hist = mdp.add_noise(mdp.obs(unflatten_state(hs), ha, hxi, hf))
    (loss, aux), grads = _pi_vg(S, o_hist, a_prev, f, xi, xi_n)
    r, d, c_t, c, Ss, Fs, Gs, v_boot = aux
    nf = lambda x: (~mx.isfinite(x)).sum().astype(mx.int32)  # noqa: E731
    gflat = flat(grads)
    # non-finite counts: loss, rewards, terminations, critic bootstrap, imagined p, R, v, w, q, qd, encoder / decoder grads
    diag = mx.stack([nf(loss), nf(r), nf(d), nf(v_boot)] + [nf(x) for x in Ss]
                    + [sum(nf(g) for k, g in gflat.items() if k.startswith(pre)) for pre in ("enc", "dec")])  # fmt: skip
    grads, gn = clip_grad_norm(gflat, 1.0)
    if args.pi_clip_value > 0:
        grads = clip_grad_value(grads, args.pi_clip_value)
    finite = mx.isfinite(gn)
    CTR["nonfinite"] = CTR["nonfinite"] + (~finite).astype(mx.int32)
    grads = {k: mx.where(finite, g, 0.0) for k, g in grads.items()}
    pi.update(unflat(opt_pi.step(flat(pi.trainable_parameters()), grads, lr_pi, ok=finite)))
    if EMA is not None:
        live = flat(pi.parameters())
        for k in EMA["p"]:
            EMA["p"][k] = EMA["p"][k] + ema_w * (live[k] - EMA["p"][k])
        for sh in SHADOW:
            if sh["d"] > 0:
                w_s = mx.where(ema_w >= 1.0, 1.0, 1.0 - sh["d"])  # the first update copies, as the main EMA
                for k in sh["p"]:
                    sh["p"][k] = sh["p"][k] + w_s * (live[k] - sh["p"][k])
    # ---- critic: GAE(lambda) targets over the imagined transitions ----
    B = s.shape[0]
    S_c = tuple(mx.concatenate((S[i][None], Ss[i]), 0).reshape((-1,) + S[i].shape[1:]) for i in range(6))
    f_c = mx.concatenate((f[None], Fs), 0).reshape(-1)
    g_c = mx.concatenate((motion.goal(f, K)[None], Gs), 0).reshape(-1, GOAL_DIM)
    xi_c = mx.broadcast_to(xi_n[None], (HZ + 1,) + xi_n.shape).reshape(-1, xi_n.shape[-1])
    X = mx.stop_gradient(_critic_input(S_c, f_c, xi_c, g_c)).reshape(HZ + 1, B, -1)
    # a non-finite imagined batch must not reach the critic's normalizer or weights (they would stay NaN for good)
    ok_c = mx.isfinite(X).all() & mx.isfinite(r).all() & mx.isfinite(d).all()
    X = mx.where(ok_c, X, 0.0)
    critic.norm.update(X, ok=ok_c)
    targets = mx.stop_gradient(_gae_targets(critic(X), mx.where(ok_c, r, 0.0), mx.where(ok_c, d, 0.0)))
    # normalized once (the statistics are fixed during the epochs): the same values as normalizing each minibatch
    Xf, Yf = critic.norm(X[:-1].reshape(-1, X.shape[-1])), targets.reshape(-1)
    n = Xf.shape[0]
    mb = n // args.critic_minibatches
    v_loss = mx.zeros(())
    for _ in range(args.critic_epochs):
        perm = mx.random.permutation(n)
        Xp, Yp = Xf[perm], Yf[perm]  # one gather per epoch; minibatch i is rows perm[i * mb : (i + 1) * mb], as before
        for i in range(args.critic_minibatches):
            v_loss, vg = _v_vg(Xp[i * mb : (i + 1) * mb], Yp[i * mb : (i + 1) * mb])
            vg, vgn = clip_grad_norm(flat(vg), 1.0)
            ok_v = ok_c & mx.isfinite(vgn)
            vg = {k: mx.where(ok_v, g, 0.0) for k, g in vg.items()}
            critic.update(unflat(opt_v.step(flat(critic.trainable_parameters()), vg, lr_v, ok=ok_v)))
    r_sum = (c_t * r).sum(0)
    return loss, v_loss, (r_sum / HZ / 0.1).mean(), gn, c.mean(), diag


_PU = {}


def policy_update_c(*xs):
    """The compiled policy + critic update for the current horizon HZ. The horizon is a Python constant of the trace, so
    each horizon gets its own compiled wrapper (a distinct function object: its own compile cache entry)."""
    if HZ not in _PU:
        def _pu(*a):
            return _policy_update(*a)

        _PU[HZ] = mx.compile(_pu, inputs=STATE, outputs=STATE)
    return _PU[HZ](*xs)


def sample_valid_starts(B: int):
    """B imagination starts at least HZ frames before their clip's end (2B candidates, the valid ones first); with
    --focus_replay a share of them from the windows before the probe sweeps' failures."""
    picks = []
    if args.focus_replay > 0 and REPLAY["pi"] is not None:
        picks.append(buffer.focus_index(int(round(args.focus_replay * REPLAY["scale"] * B)), REPLAY["pi"]))
    if args.deficit_replay > 0:
        if DEFICIT["n"] % 20 == 0:  # the segment returns move slowly: refresh the per-frame weights every 20 draws
            ok = np.arange(motion.num_frames) + max(HZ, args.horizon) <= motion.clip_end_np[motion.clip_id_np]
            DEFICIT["w"] = np.exp(-args.deficit_temp * env.bin_G / env.G_max)[motion.frame_bin_np] * ok
        DEFICIT["n"] += 1
        picks.append(buffer.focus_index(int(round(args.deficit_replay * B)), DEFICIT["w"]))
    picks = [x for x in picks if x is not None]
    foc = (np.concatenate([x[0] for x in picks]), np.concatenate([x[1] for x in picks])) if picks else None
    nu = B - (len(foc[0]) if foc is not None else 0)
    s, h = buffer.sample_starts(2 * nu, Hh)
    f = s["f"]
    ok = f + HZ <= motion.clip_end_np[motion.clip_id_np[f]]
    idx = np.argsort(~ok, kind="stable")[:nu]
    s, h = {k: v[idx] for k, v in s.items()}, {k: v[idx] for k, v in h.items()}
    if foc is not None:
        sf, hf = buffer.starts_at(foc[0], foc[1], Hh)
        s, h = {k: np.concatenate((sf[k], v)) for k, v in s.items()}, {k: np.concatenate((hf[k], v)) for k, v in h.items()}
    return s, h


def pi_starts():
    with prof("pi.sample"):
        return sample_valid_starts(args.pi_batch)


def pi_inputs(lr_pi, lr_v, starts=None):
    """Host side of the policy update: sampled starts and histories, learning rates, the EMA weight."""
    s, h = starts if starts is not None else pi_starts()
    ema_w = 0.0
    if EMA is not None:
        d = args.pi_ema_early if args.pi_ema_early > 0 and it < args.pi_ema_switch else args.pi_ema
        ema_w = 1.0 if EMA["n"] == 0 else 1.0 - d  # AveragedModel: the first update copies
        EMA["n"] += 1
    with prof("pi.to_mx"):
        xs = (mx.array(s["s"]), mx.array(h["s"]), mx.array(h["a_prev"]), mx.array(h["xi"]), mx.array(h["f"]),
              mx.array(s["xi"]), mx.array(s["f"]), mx.array(s["a_prev"]), mx.array(lr_pi), mx.array(lr_v),
              mx.array(ema_w, dtype=mx.float32))  # fmt: skip
    return xs


def policy_update(lr_pi, lr_v):
    xs = pi_inputs(lr_pi, lr_v)
    with prof("pi.build"):
        return policy_update_c(*xs)


# ------------------------------------------------------------------------------------------------ collection
env = TrackingSim(args.num_envs, motion, mdp, layout, gamma=args.gamma, fail_reset=args.fail_reset, bin_fix=args.bin_fix,
                  seed=args.seed, nthread=args.sim_threads, backend=args.sim_backend)  # fmt: skip
if RESUME_CTRL is not None:
    env.bin_G, env.bin_fail, env.bin_w_avg = (np.array(RESUME_CTRL[k]) for k in ("bin_G", "bin_fail", "bin_w_avg"))
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, J, layout.dim, seed=args.seed)
if args.resume and os.path.exists(os.path.join(args.resume, "buffer.npz")):  # a fork point written by --snap_at
    with np.load(os.path.join(args.resume, "buffer.npz")) as _z:
        if _z["s"].shape == buffer.data["s"].shape:
            for _k in buffer.data:
                buffer.data[_k][...] = _z[_k]
            buffer.ptr, buffer.size = int(_z["ptr"]), int(_z["size"])
            buffer.data["cut"][(buffer.ptr - 1) % buffer.T] = True  # the worlds restart: no window continues across the fork
            print(f"--resume: replay buffer restored ({buffer.size} steps)", flush=True)
        else:
            print(f"--resume: buffer.npz has shape {_z['s'].shape}, this run {buffer.data['s'].shape}: starting empty", flush=True)
OBS_HIST = {"h": mx.zeros((args.num_envs, Hh, mdp.obs_dim))}
ACT_PI = policy_from_args(vars(args), mdp.obs_dim, GOAL_DIM, J) if args.overlap else None  # --overlap: the actor's snapshot
ACT_CPU = args.overlap and args.actor_device == "cpu"
if ACT_CPU:
    ACT_PI.fix_compute_dtype(None)  # fp32 on the CPU (Accelerate); the learner keeps bf16 on the GPU


def actor_ctx():
    """The stream of the actor phase (--overlap): the CPU with --actor_device cpu, else the default (GPU)."""
    import contextlib

    return mx.stream(mx.cpu) if ACT_CPU else contextlib.nullcontext()


PROBE_CPU = ACT_CPU or (args.overlap and args.probe_device == "cpu")


def probe_ctx():
    """The stream of the probe sweep: the CPU with --probe_device cpu (or --actor_device cpu), else the default."""
    import contextlib

    return mx.stream(mx.cpu) if PROBE_CPU else contextlib.nullcontext()
ACT_KEY = {"k": mx.random.key(args.seed + 1)}


def _obs_goal(s, a_prev, xi, f, fresh, hist, key):
    o = mdp.add_noise(mdp.obs(unflatten_state(s), a_prev, xi, f), key=key)
    hist = mx.where(fresh[:, None, None], mx.broadcast_to(o[:, None], hist.shape), mx.concatenate((hist[:, 1:], o[:, None]), 1))
    return o, hist, motion.goal(f, K)


def _act(s, a_prev, xi, f, fresh, hist, k1, k2):
    """Sequential collection (upstream): the normalizers see o and g before the action is taken."""
    o, hist, g = _obs_goal(s, a_prev, xi, f, fresh, hist, k1)
    pi.obs_norm.update(o)
    pi.goal_norm.update(g)
    a = pi(hist, g)
    return mx.clip(a + args.action_noise * mx.random.normal(a.shape, key=k2), -20, 20), hist


def _act_snap(s, a_prev, xi, f, fresh, hist, k1, k2):
    """--overlap: the action of the snapshot policy (normalizers of iteration i - 1); o and g are returned so that the
    normalizer updates can be applied once the learner is done."""
    o, hist, g = _obs_goal(s, a_prev, xi, f, fresh, hist, k1)
    a = ACT_PI(hist, g)
    return mx.clip(a + args.action_noise * mx.random.normal(a.shape, key=k2), -20, 20), hist, o, g


def _pi_norms(o, g):
    pi.obs_norm.update(o)
    pi.goal_norm.update(g)
    return mx.zeros(())


def _wm_norms(s, s1, a, xi):
    S0, S1 = unflatten_state(s), unflatten_state(s1)
    wm.in_norm.update(wm.inputs(S0, a, layout.normalize(xi)))
    wm.out_norm.update(wm.targets(S0, S1))
    wm.inc_norm.update(WorldModel.increments(S0, S1))
    return mx.zeros(())


act_c = mx.compile(_act, inputs=STATE, outputs=STATE)
act_snap_c = mx.compile(_act_snap, inputs=[ACT_PI.state], outputs=[]) if args.overlap else None
pi_norms_c = mx.compile(_pi_norms, inputs=STATE, outputs=STATE)
wm_norms_c = mx.compile(_wm_norms, inputs=STATE, outputs=STATE)


def _keys():
    ks = mx.random.split(ACT_KEY["k"], 3)
    ACT_KEY["k"] = ks[0]
    return ks[1], ks[2]


def collect(n_steps: int, stats: dict, deferred: list | None = None):
    """n_steps collection steps. ``deferred`` (--overlap): act with the snapshot ACT_PI and append the normalizer
    updates to the list instead of applying them (apply_deferred, after the learner)."""
    global sim_steps
    for _ in range(n_steps):
        s = env.state_flat()
        args_ = (mx.array(s), mx.array(env.a_prev), mx.array(env.xi), mx.array(env.f.astype(np.int32)), mx.array(env.ep_step == 0),
                 OBS_HIST["h"], *_keys())  # fmt: skip
        with prof("col.act"):
            if deferred is None:
                a, OBS_HIST["h"] = act_c(*args_)
            else:
                with actor_ctx():
                    a, OBS_HIST["h"], o, g = act_snap_c(*args_)
                    mx.eval(a)
            a = np.array(a)
        with prof("col.step"):
            out = env.step_core(a, s)
        tr = (mx.array(out["s"]), mx.array(out["s1"]), mx.array(out["a"]), mx.array(out["xi"]))
        if deferred is None:
            wm_norms_c(*tr)
        else:
            deferred.append(((o, g), tr))
        with prof("col.buffer"):
            buffer.add(out)
        with prof("col.tail"):
            out = env.step_tail(out)
        sim_steps += args.num_envs
        stats["r"].append(out["r"].mean() / 0.1)
        if "ep_len" in out:
            stats["ep_len"].append(out["ep_len"].mean()), stats["ep_ret"].append(out["ep_ret_norm"].mean())
            stats["term"].append(out["term"].sum()), stats["done"].append(out["done"].sum())


def apply_deferred(deferred: list):
    for (o, g), tr in deferred:
        pi_norms_c(o, g)
        wm_norms_c(*tr)
    deferred.clear()


class LearnerThread:
    """--overlap: a worker thread that owns a GPU stream (MLX streams are per thread) and runs one learner job at a time;
    MLX releases the GIL while it evaluates, so the main thread steps the CPU simulator meanwhile."""

    def __init__(self, seed: int, qos: bool = False):
        import queue
        import threading

        self.seed, self.qos = seed, qos
        self.jobs, self.results = queue.Queue(), queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.pending = None

    def _run(self):
        # MLX's default random key sequence is per thread and a new thread's is seeded from the clock: seed it, or the
        # learner's draws (observation noise, critic minibatch permutations) differ from run to run
        mx.random.seed(self.seed)
        if self.qos and sys.platform == "darwin":  # --learner_qos: this thread feeds the GPU; it must not queue behind the simulator's threads
            import ctypes

            rc = ctypes.CDLL("/usr/lib/libSystem.B.dylib").pthread_set_qos_class_self_np(ctypes.c_uint(0x21), ctypes.c_int(0))  # USER_INTERACTIVE
            print(f"--learner_qos: learner thread at user-interactive QoS (rc {rc})", flush=True)
        stream = mx.new_stream(mx.gpu)
        while True:
            fn = self.jobs.get()
            if fn is None:
                return
            try:
                with mx.stream(stream):
                    self.results.put(("ok", fn()))
            except BaseException as e:  # re-raised in the main thread
                self.results.put(("err", e))

    def submit(self, fn):
        self.jobs.put(fn)
        self.pending = None

    def wait_idle(self):
        self.pending = self.results.get()

    def result(self):
        kind, val = self.pending if self.pending is not None else self.results.get()
        self.pending = None
        if kind == "err":
            raise val
        return val


# ------------------------------------------------------------------------------------------------ evaluation probes
EPROBE = args.eprobe_share > 0
if EPROBE:
    assert EMA is not None, "--eprobe_share sweeps the EMA policy: needs --pi_ema"
    EP_TILE = np.concatenate([np.arange(motion.clip_start_np[c], motion.clip_end_np[c] - 1, args.eprobe_tile) for c in range(len(motion.names))])
    P_EP = len(EP_TILE)
    penv = TrackingSim(P_EP, motion, mdp, layout, eval_mode=True, randomize=False, reset_noise=False, seed=args.seed,
                       nthread=args.probe_sim_threads or args.sim_threads, backend=args.sim_backend)  # fmt: skip
    EP_PI = policy_from_args(vars(args), mdp.obs_dim, GOAL_DIM, J)  # the sweep's policy: a snapshot of the EMA
    if PROBE_CPU:
        EP_PI.fix_compute_dtype(None)
    mx.eval(EP_PI.parameters())  # materialized: arrays cross to the probe worker thread (--eprobe_thread)
    EP_FAIL = np.zeros(motion.num_bins)
    _nxt = np.arange(motion.num_bins) + 1
    EP_NEXT = np.where((_nxt < motion.num_bins) & (motion.bin_clip_np[np.minimum(_nxt, motion.num_bins - 1)] == motion.bin_clip_np), _nxt, _nxt - 1)
    EP = {}
    print(f"--eprobe: {P_EP} probe worlds (tiles of {args.eprobe_tile} frames)", flush=True)

    def _ep_act(s, a_prev, xi, f, hist):
        o = mdp.obs(unflatten_state(s), a_prev, xi, f)
        hist = mx.concatenate((hist[:, 1:], o[:, None]), 1)
        return EP_PI(hist, motion.goal(f, K)), hist

    ep_act_c = mx.compile(_ep_act, inputs=[EP_PI.state], outputs=[EP_PI.state])
    penv.cpu = PROBE_CPU or (args.overlap and args.reward_cpu)
    penv.term_only = args.eprobe_np_term
    # --eprobe_thread: the sweep's steps run on their own worker thread / GPU stream (compiled graphs keep the stream
    # they were traced on, so the probe functions are traced there)
    PROBER = LearnerThread(args.seed + 202) if (args.overlap and args.eprobe_thread and args.eprobe_async > 0) else None

    def _probe_trace():
        """Trace the probe step now, before any learner thread runs (--overlap)."""
        _s = penv.state_flat()
        with probe_ctx():
            mx.eval(ep_act_c(mx.array(_s), mx.array(penv.a_prev), mx.array(penv.xi), mx.array(penv.f.astype(np.int32)),
                             mx.zeros((P_EP, Hh, mdp.obs_dim)))[0])  # fmt: skip
        if penv.term_only:
            return
        with mx.stream(mx.cpu) if penv.cpu else probe_ctx():  # a compiled graph keeps the stream it was traced on
            _fn = penv._reward_fn_cpu if penv.cpu else penv._reward_fn
            mx.eval(_fn(mx.array(_s), mx.array(penv.f.astype(np.int32) + 1), mx.zeros((P_EP, J)), mx.zeros((P_EP, J))))

    if PROBER is not None:
        PROBER.submit(_probe_trace)
        PROBER.result()
    else:
        _probe_trace()

    def eprobe_begin(it, ema_params):
        m = motion
        cid = m.clip_id_np[EP_TILE]
        lead = int(np.random.randint(args.eprobe_lead_min, args.eprobe_lead_max + 1))
        f0 = np.maximum(EP_TILE - lead, m.clip_start_np[cid])
        stop = np.minimum(EP_TILE + args.eprobe_tile, m.clip_end_np[cid])
        fcap = np.minimum(stop, m.clip_end_np[cid] - 1)
        penv.done_eval[:] = False
        penv.failed_eval[:] = False
        penv.reset(np.arange(P_EP), f0)
        EP_PI.update(unflat(dict(ema_params)))
        s = penv.state_flat()
        with probe_ctx():
            o = mdp.obs(unflatten_state(mx.array(s)), mx.array(penv.a_prev), mx.array(penv.xi), mx.array(penv.f.astype(np.int32)))
            mx.eval(o)
        EP.update(it=it, lead=lead, t0=time.time(), left=lead + args.eprobe_tile, stop=stop, fcap=fcap,
                  fail_f=np.full(P_EP, -1), hist=mx.broadcast_to(o[:, None], (P_EP, Hh, mdp.obs_dim)))  # fmt: skip
        mx.eval(EP["hist"])  # materialized: --eprobe_thread hands it to the worker thread

    def eprobe_steps(k):
        _ep_advance(k)
        if EP["left"] == 0:
            eprobe_finish()

    def _ep_advance(k):
        t_chunk = time.perf_counter()
        for _ in range(min(k, EP["left"])):
            s = penv.state_flat()
            with prof("ep.act"), probe_ctx():
                a, EP["hist"] = ep_act_c(mx.array(s), mx.array(penv.a_prev), mx.array(penv.xi), mx.array(penv.f.astype(np.int32)), EP["hist"])
                mx.eval(a)
            a = np.array(a)
            alive = ~penv.done_eval.copy()
            f_now = penv.f.copy()
            with prof("ep.step"):
                out = penv.step(a)
            EP["fail_f"] = np.where(alive & (out["term"] | ~np.isfinite(a).all(-1)), f_now, EP["fail_f"])
            penv.done_eval |= penv.f >= EP["stop"]
            penv.f = np.minimum(penv.f, EP["fcap"])
            EP["left"] -= 1
            if penv.done_eval.all() and not args.eprobe_async:  # upstream's eager sweep stops early, its graphed
                EP["left"] = 0  # (async) sweep always runs lead + tile steps
        mx.eval(EP["hist"])  # materialized: the history crosses threads between chunks with --eprobe_thread
        PROF["ep.chunk"] = PROF.get("ep.chunk", 0.0) + 1000 * (time.perf_counter() - t_chunk)

    def replay_weights():
        """--focus_replay: per-frame weights, the failure bins of the recent sweeps and the bins just before them."""
        if not (args.focus_replay > 0 or args.focus_replay_wm > 0) or EP_FAIL.sum() <= 0:
            return
        w, q = EP_FAIL.copy(), EP_FAIL.copy()
        for _ in range(args.focus_replay_win):
            w = w[EP_NEXT] * (EP_NEXT != np.arange(motion.num_bins))
            q += w
        wf = q[motion.frame_bin_np]
        ok = np.arange(motion.num_frames) + max(HZ, args.horizon) <= motion.clip_end_np[motion.clip_id_np]
        REPLAY["wm"], REPLAY["pi"] = wf, wf * ok
        if args.focus_replay_per > 0:  # failures per sweep, smoothed: EP_FAIL sums them with weights decay^age
            rate = float(EP_FAIL.sum()) * (1.0 - args.eprobe_decay)
            REPLAY["scale"] = min(1.0, args.focus_replay_per * rate / max(args.focus_replay, 1e-9))

    def eprobe_finish():
        global EP_FAIL
        m, fail_f = motion, EP["fail_f"]
        EP_FAIL = EP_FAIL * args.eprobe_decay
        fb = m.frame_bin_np[fail_f[fail_f >= 0]]
        np.add.at(EP_FAIL, fb, 1.0)
        if EP_FAIL.sum() > 0:
            w, q = EP_FAIL.copy(), EP_FAIL.copy()
            for _ in range(args.eprobe_win):  # q[b] = sum of w over b .. b + win (same clip): the window before each failure
                w = w[EP_NEXT] * (EP_NEXT != np.arange(m.num_bins))
                q += w
            env.set_focus(q / q.sum(), args.eprobe_share)
            replay_weights()
        nf = int((fail_f >= 0).sum())
        fl = sorted({(m.names[int(m.clip_id_np[f])].replace("_subject", "_s"), int(f - m.clip_start_np[m.clip_id_np[f]])) for f in fail_f[fail_f >= 0]})
        rp = ""
        if REPLAY["wm"] is not None:  # the buffer's share of frames inside the replay windows (what uniform sampling would give)
            rp = f" | replay share {100 * args.focus_replay * REPLAY['scale']:.1f}%"
        print(f"--eprobe it {EP['it']}: lead {EP['lead']} | {nf}/{P_EP} failures {fl[:8]} | {time.time() - EP['t0']:.1f} s{rp}", flush=True)
        EP.clear()

    if RESUME_CTRL is not None and RESUME_CTRL.get("ep_fail") is not None:
        EP_FAIL = np.array(RESUME_CTRL["ep_fail"])
        replay_weights()
if RESUME_CTRL is not None and RESUME_CTRL.get("focus_q") is not None:
    env.set_focus(np.array(RESUME_CTRL["focus_q"]), RESUME_CTRL["focus_share"])


# ------------------------------------------------------------------------------------------------ checkpoints
def save_ckpt(it: int, wall: float, full: bool, snap: bool = False):
    """The policy checkpoint of iteration `it` (+ the full training state in last/ with `full`); `snap`: only the full
    state and the replay buffer, in snap/ (--snap_at)."""
    pol = EMA["p"] if EMA is not None else flat(pi.parameters())
    meta = {"iteration": it, "wall_time": wall, "sim_steps": sim_steps, "args": vars(args), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_dir = args.log_dir
    if snap:
        out_dir = os.path.join(args.log_dir, "snap")
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "buffer.npz"), ptr=buffer.ptr, size=buffer.size, **buffer.data)
    else:
        save_tree(os.path.join(args.log_dir, f"ckpt_{it:06d}"), {"pi": pol, "meta": meta})
        for sh in SHADOW:
            d = os.path.join(args.log_dir, f"shadow_{sh['d']:g}")
            os.makedirs(d, exist_ok=True)
            save_tree(os.path.join(d, f"ckpt_{it:06d}"), {"pi": sh["p"] if sh["d"] > 0 else flat(pi.parameters()), "meta": meta})
    if full or snap:
        meta = dict(meta, ema_n=EMA["n"] if EMA is not None else 0,
                    ctrl={"decay_at": DECAY_AT[0], "ep_fail": EP_FAIL.tolist() if EPROBE else None,
                          "focus_q": env.focus_q.tolist() if env.focus_q is not None else None, "focus_share": env.focus_share,
                          "bin_G": env.bin_G.tolist(), "bin_fail": env.bin_fail.tolist(), "bin_w_avg": env.bin_w_avg.tolist()})  # fmt: skip
        tree = {"wm": flat(wm.parameters()), "pi_live": flat(pi.parameters()), "critic": flat(critic.parameters()), "meta": meta}
        for o, k in ((opt_wm, "opt_wm"), (opt_pi, "opt_pi"), (opt_v, "opt_v")):
            tree[k] = {"t": o.state["t"], **{sub: o.state[sub] for sub in ("m", "v", "buf") if o.state[sub]}}
        if EMA is not None:
            tree["pi_avg"] = EMA["p"]
        save_tree(os.path.join(out_dir, "last"), tree)


LEARNER = LearnerThread(args.seed + 101, qos=args.learner_qos) if args.overlap else None

# ------------------------------------------------------------------------------------------------ main loop
stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
t_c = time.time()
collect(args.warmup_steps, stats)
print(f"warmup: {args.warmup_steps} steps in {time.time() - t_c:.1f} s", flush=True)
env.cpu = ACT_CPU or (args.overlap and args.reward_cpu)  # from here the actor phase computes the reward heads on the CPU (--actor_device cpu)
log_path = os.path.join(args.log_dir, "train_log.csv")
new_log = start_it == 0 or not os.path.exists(log_path)
log_f = open(log_path, "a")
if new_log:
    log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,"
                "ep_ret_norm,term_frac,it_per_s,lr,pi_lr,pi_mult,nonfinite,ms_collect,ms_wm,ms_pi\n")  # fmt: skip
    log_f.flush()
t_start = time.time() - wall_offset
next_ckpt_s = wall_offset + args.ckpt_minutes * 60
t_log = time.time()
step_acc = 0.0
agg = {k: [] for k in ("wm", "wm8", "pi", "v", "ir", "gn", "surv")}
tm = {"collect": 0.0, "wm": 0.0, "pi": 0.0}
POLICY_START = args.wm_only_iters if (start_it > 0 or args.wm_only_iters == 0) else None
DECAY_AT = [RESUME_CTRL["decay_at"] if RESUME_CTRL is not None else None]

PI_BATCH_FULL, HZ_FULL, WMB_FULL, PI_STEPS_FULL = args.pi_batch, HZ, args.wm_batches, args.pi_steps
WARM = bool(args.pi_batch_warm or args.horizon_warm or args.wm_batches_warm or args.pi_steps_warm)
PRE = None  # --presample: the next iteration's sampled batches


def sched_at(i: int):
    """(policy batch, horizon, world-model batches, policy steps) of iteration i under the compute schedule."""
    warm = WARM and i < args.pi_batch_switch
    return (args.pi_batch_warm if warm and args.pi_batch_warm else PI_BATCH_FULL,
            args.horizon_warm if warm and args.horizon_warm else HZ_FULL,
            args.wm_batches_warm if warm and args.wm_batches_warm else WMB_FULL,
            args.pi_steps_warm if warm and args.pi_steps_warm else PI_STEPS_FULL)  # fmt: skip


for it in range(start_it, args.iterations):
    if args.pi_mom > 0 and it == max(args.pi_mom_from, start_it):  # a Python constant of the trace: retrace once
        opt_pi.mom = args.pi_mom
        _PU.clear()
    if WARM:  # compute schedule: cheap updates during the ramp (the compiled updates retrace once for the new shapes)
        args.pi_batch, HZ, args.wm_batches, args.pi_steps = sched_at(it)
    lr = lr_at(it)
    pi_mult = args.pi_lr_mult
    if DECAY_AT[0] is not None:
        pi_mult = args.pi_lr_mult + (1.0 - args.pi_lr_mult) * min(1.0, (it - DECAY_AT[0]) / max(args.lr_decay_len, 1))
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    t0 = time.time()
    if POLICY_START is None and it + 1 >= args.wm_only_iters:
        POLICY_START = args.wm_only_iters
        print(f"policy updates start at iteration {POLICY_START}", flush=True)
    do_pi = POLICY_START is not None and it >= POLICY_START
    lr_mx = mx.array(lr, dtype=mx.float32)
    if not args.overlap:
        # upstream order: probe steps, collect, world-model updates, policy + critic update
        if EPROBE:
            if EP and args.eprobe_async:
                eprobe_steps(args.eprobe_async)
            if it >= args.eprobe_start and it % args.eprobe_every == 0:
                assert not EP, "--eprobe_async too small: the previous sweep has not finished"
                eprobe_begin(it, EMA["p"])
                if not args.eprobe_async:
                    eprobe_steps(EP["left"])
        collect(n, stats)
        t1 = time.time()
        l, e8 = wm_updates(lr_mx)
        prof_eval("wm.encode", "wm.gpu", l, e8, STATE)
        t2 = time.time()
        res = None
        if do_pi:
            for _ in range(args.pi_steps):
                res = policy_update(lr * pi_mult, lr)
                prof_eval("pi.encode", "pi.gpu", res, STATE)  # STATE: the EMA would otherwise stay a lazy chain of copies
        t3 = time.time()
    else:
        # XMimic actor | learner: the learner (world model, then policy + critic) of iteration i is queued on its GPU
        # stream with the buffer up to iteration i - 1; meanwhile the actor collects (and the probe sweep advances) on
        # the CPU simulator with the policy, normalizers and EMA of iteration i - 1; then the actor's normalizer updates
        # are applied
        snap = flat(pi.parameters())
        ACT_PI.update(unflat(snap))
        ema_snap = dict(EMA["p"]) if EMA is not None else None
        if PRE is not None and PRE["it"] == it:  # sampled after the previous collection: the same buffer and draws as now
            wm_in, starts = PRE["wm"], PRE["pi"]
        else:  # sampled now: the actor overwrites the oldest buffer rows while the learner runs
            wm_in = wm_inputs()
            starts = [pi_starts() for _ in range(args.pi_steps)] if do_pi else None
        PRE = None
        pi_in = [pi_inputs(lr * pi_mult, lr, st) for st in starts] if do_pi else None

        def learner_job(wm_in=wm_in, pi_in=pi_in, lr_mx=lr_mx):
            with prof("lrn.build_wm"):
                l_, e8_ = wm_run(wm_in, lr_mx)
                if args.learner_async:  # the GPU starts on the world model while the policy update's graph is built
                    mx.async_eval(l_, e8_, STATE)
            res_ = None
            for xs in pi_in or ():
                with prof("lrn.build_pi"):
                    res_ = policy_update_c(*xs)
                if len(pi_in) > 1:
                    mx.eval(res_, STATE)  # one update's graph at a time
            with prof("lrn.eval"):
                mx.eval(l_, e8_, res_ if res_ is not None else (), STATE)
            return l_, e8_, res_

        # the first iterations, and the first policy update after --wm_only_iters, trace and compile one side at a time
        # (MLX's compile cache is not known to be safe to fill from two threads); the probe functions are traced at setup
        concurrent = it >= start_it + 2 and it != POLICY_START and not (WARM and it in (args.pi_batch_switch, args.pi_batch_switch + 1))
        concurrent = concurrent and not (args.pi_mom > 0 and it in (args.pi_mom_from, args.pi_mom_from + 1))  # its retrace
        concurrent = concurrent and not os.environ.get("SMX_OVERLAP_SERIAL")  # debug: overlap semantics, no concurrency
        mx.eval(STATE, wm_in, pi_in if pi_in is not None else (), lr_mx)  # MLX streams are per thread: hand over materialized arrays
        LEARNER.submit(learner_job)
        if not concurrent:
            LEARNER.wait_idle()
        t1 = time.time()
        deferred = []
        actx = actor_ctx()
        actx.__enter__()
        probe_pending = False
        if EPROBE:
            if EP and args.eprobe_async and PROBER is not None:
                final = EP["left"] <= args.eprobe_async  # read before the worker starts decrementing it
                PROBER.submit(lambda: _ep_advance(args.eprobe_async))
                if final:  # the final chunk: its focus update applies to this collection, as inline
                    PROBER.result()
                    eprobe_finish()
                else:
                    probe_pending = True  # runs concurrently with the collection and the learner
            elif EP and args.eprobe_async:
                eprobe_steps(args.eprobe_async)
            if it >= args.eprobe_start and it % args.eprobe_every == 0:
                assert not EP, "--eprobe_async too small: the previous sweep has not finished"
                eprobe_begin(it, ema_snap)
                if not args.eprobe_async:
                    eprobe_steps(EP["left"])
        collect(n, stats, deferred)
        actx.__exit__(None, None, None)
        t2 = time.time()
        # the buffer is now what iteration it + 1 samples from: gather its batches during the learner wait (only between
        # two policy iterations of the same shapes, so the draws are exactly those of sampling at the top of it + 1)
        if args.presample and do_pi and it + 1 < args.iterations and sched_at(it + 1) == sched_at(it):
            PRE = {"it": it + 1, "wm": wm_inputs(), "pi": [pi_starts() for _ in range(args.pi_steps)]}
        with prof("wait.learner"):
            l, e8, res = LEARNER.result()
        if probe_pending:
            with prof("wait.probe"):
                PROBER.result()
        apply_deferred(deferred)
        mx.eval(STATE)
        t3 = time.time()
    agg["wm"].append(l), agg["wm8"].append(e8)
    if res is not None:
        for k, v in zip(("pi", "v", "ir", "gn", "surv"), res[:5]):
            agg[k].append(v)
        dg = np.array(res[5])
        if dg.any():
            names = ("loss", "r", "d", "v_boot", "p", "R", "v", "w", "q", "qd", "g_enc", "g_dec")
            print(f"[nonfinite] it {it}: " + ", ".join(f"{k} {int(x)}" for k, x in zip(names, dg) if x), flush=True)
    # sequential: collect / world model / policy; --overlap: queue the learner / actor phase / remaining learner wait
    tm["collect"] += t1 - t0
    tm["wm"] += t2 - t1
    tm["pi"] += t3 - t2
    if (it + 1) % args.log_every == 0:
        mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall_h = (time.time() - t_start) / 3600
        term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
        row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
               mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr,
               lr * pi_mult, float(pi_mult), int(CTR["nonfinite"].item())] + [1000 * tm[k] / args.log_every for k in ("collect", "wm", "pi")]  # fmt: skip
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        log_f.flush()
        deadline = args.lr_decay_by > 0 and it + 1 >= args.lr_decay_by
        if args.lr_decay_ret > 0 and DECAY_AT[0] is None and (row[12] >= args.lr_decay_ret or deadline):
            DECAY_AT[0] = it + 1
            why = f"training return {row[12]:.3f} >= {args.lr_decay_ret}" if row[12] >= args.lr_decay_ret else f"deadline --lr_decay_by {args.lr_decay_by}"
            print(f"--lr_decay_ret: {why} at it {it + 1}: policy LR x{args.pi_lr_mult} -> x1 over {args.lr_decay_len} its", flush=True)
        print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} "
              f"surv {row[9]:.3f} | sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | "
              f"{ips:.2f} it/s (collect {row[-3]:.0f} wm {row[-2]:.0f} pi {row[-1]:.0f} ms) | gpu peak {mx.get_peak_memory() / 2**30:.1f} GB, active {mx.get_active_memory() / 2**30:.1f} GB",
              flush=True)  # fmt: skip
        if args.profile and PROF:
            print("  profile (ms / iteration): " + ", ".join(f"{k} {v / args.log_every:.1f}" for k, v in sorted(PROF.items())), flush=True)
            PROF.clear()
        stats = {k: [] for k in stats}
        agg = {k: [] for k in agg}
        tm = {k: 0.0 for k in tm}
    wall = time.time() - t_start
    out_of_time = args.max_hours is not None and wall >= args.max_hours * 3600
    timed = args.ckpt_minutes > 0 and wall >= next_ckpt_s
    if timed:
        next_ckpt_s += args.ckpt_minutes * 60
    final = it + 1 == args.iterations or out_of_time
    if (it + 1) % args.ckpt_every == 0 or timed or final:
        save_ckpt(it + 1, wall, full=final or (it + 1) % (args.ckpt_every * 5) == 0)
    if args.snap_at and it + 1 == args.snap_at:
        assert not (EPROBE and EP), "--snap_at: a probe sweep is in progress (pick an iteration between two sweeps)"
        t_snap = time.time()
        save_ckpt(it + 1, wall, full=True, snap=True)
        t_start += time.time() - t_snap  # the write is not training time
        print(f"--snap_at: fork point of iteration {it + 1} in {os.path.join(args.log_dir, 'snap')} ({time.time() - t_snap:.1f} s)", flush=True)
    if out_of_time:
        print(f"time budget of {args.max_hours} h reached at it {it + 1}", flush=True)
        break
open(os.path.join(args.log_dir, "done"), "w").write(f"{it + 1}\n")
print("done", flush=True)
sys.exit(0)
