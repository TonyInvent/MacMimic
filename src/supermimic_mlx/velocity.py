"""OpenSprite command interface. The decoder and dynamics are shared with motion tracking."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .kinematics import cross
from .command_ranges import COMMAND_RANGES
from .mdp import invariant_features, inv_dim, unflatten_state
from .networks import Critic, MLP, RunningNorm
from .optim import MuonAdam, clip_grad_norm, flat, unflat
from .rotations import mv3, T, yaw_mat
from .robots import active as robot

class CommandEncoder(nn.Module):
    def __init__(self, obs_dim, hist, normalized=True):
        super().__init__()
        self.command_norm = RunningNorm(3) if normalized else None
        self.net = MLP(3 + hist * obs_dim, [256, 256], 32)

    def __call__(self, history_features, cmd):
        cmd = self.command_norm(cmd) if self.command_norm is not None else cmd
        return mx.tanh(self.net(mx.concatenate((cmd, history_features), -1)))


def action(pi, encoder, history, cmd):
    h = pi.obs_norm(history).reshape(history.shape[0], -1)
    z = encoder(h, cmd)
    return mx.clip(pi.dec(mx.concatenate((z, h), -1)), -20, 20), z


def motion_commands(motion):
    planar = mv3(T(yaw_mat(motion.root_R)), motion.root_v)[:, :2]
    raw = np.array(mx.concatenate((planar, motion.root_w[:, 2:3]), -1))
    for start, end in zip(motion.clip_start_np, motion.clip_end_np):
        padded = np.pad(raw[start : end + 1], ((12, 12), (0, 0)), mode="edge")
        raw[start : end + 1] = np.lib.stride_tricks.sliding_window_view(padded, 25, axis=0).mean(-1)
    return raw


class Commands:
    def __init__(self, n, motion, seed, profile="opensprite"):
        self.rng = np.random.default_rng(seed)
        self.low, self.high = np.asarray(COMMAND_RANGES[profile], dtype=np.float32)
        self.references = motion_commands(motion)
        self.cmd = np.zeros((n, 3), np.float32)
        self.reference = np.zeros(n, bool)

    def sample(self, frame, step):
        refresh = step % 250 == 0
        n = refresh.sum()
        if n:
            sample = self.rng.uniform(self.low, self.high, (n, 3))
            sample[self.rng.random(n) < 0.1] = 0
            self.cmd[refresh] = sample
            self.reference[refresh] = self.rng.random(n) < 0.5
        self.cmd[self.reference] = self.references[frame[self.reference]]
        return np.column_stack((self.cmd, self.reference, step % 250)).astype(np.float32)


def reward(mdp, state, cmd, a, prev):
    _, R, v, w, q, _ = state
    inv_heading = T(yaw_mat(R))
    v_body = mv3(inv_heading, v)
    track = 2 * mx.exp(-mx.sum((v_body[..., :2] - cmd[..., :2]) ** 2, -1) / 0.25)
    track += 2 * mx.exp(-((w[..., 2] - cmd[..., 2]) ** 2) / 0.5)
    torso = robot.TRACKED_BODIES.index("waist_yaw_link")
    body_p, body_R, body_v, body_w = mdp.kin.forward(*state)
    up = mv3(inv_heading, body_R[..., torso, :, 2])
    omega = mv3(inv_heading, body_w[..., torso, :])
    roll = up[..., 1] ** 2 / (up[..., 1] ** 2 + up[..., 2] ** 2 + 1e-8)
    limit = (mx.maximum(q - mdp.soft_hi, 0) + mx.maximum(mdp.soft_lo - q, 0)).sum(-1)
    if mdp.r1_soles:
        # Published R1 posture/width/swing extension on the two sole boxes.
        centers, rot, offset, height = mdp.sole_geometry(body_p, body_R)
        lateral = inv_heading[..., 1, :]
        separation = ((centers[..., 0, :] - centers[..., 1, :]) * lateral).sum(-1)
        extent = (mx.abs((lateral[..., None, :, None] * rot).sum(-2)) * mdp.sole_half).sum((-1, -2))
        narrow = mx.maximum(mx.maximum(0.12 - separation, 0.04 - (separation - extent)), 0.0) / 0.04
        wide = mx.maximum(separation - 0.18, 0.0) / 0.04
        pitch = up[..., 0] / mx.sqrt(up[..., 0] ** 2 + up[..., 2] ** 2 + 1e-8)
        posture_cost = 32.0 * mx.maximum(mx.abs(pitch) - np.sin(np.deg2rad(1.0)).item(), 0.0) ** 2
        feet = mx.array(mdp.feet)
        sole_v = body_v[..., feet, :] + cross(body_w[..., feet, :], offset)
        speed2 = (sole_v[..., :2] ** 2).sum(-1)
        command2 = (cmd[..., :2] ** 2).sum(-1) + (0.2 * cmd[..., 2]) ** 2
        short = (mx.maximum(0.02 - height, 0.0) / 0.02) ** 2
        swing = command2 / (command2 + 0.05**2) * (short * speed2 / (speed2 + 0.08**2)).mean(-1)
        track -= posture_cost + 0.4 * narrow**2 + 0.1 * wide**2 + 0.15 * swing
    return 0.02 * (
        track - 0.3 * ((a - prev) ** 2).sum(-1) - 2 * limit - 4 * roll - 0.05 * (omega[..., :2] ** 2).sum(-1)
    )


class Learner:
    def __init__(self, pi, wm, mdp, layout, motion, args):
        self.pi, self.wm, self.mdp, self.layout, self.motion = pi, wm, mdp, layout, motion
        self.encoder = CommandEncoder(mdp.obs_dim, args.hist)
        self.snapshot = CommandEncoder(mdp.obs_dim, args.hist)
        self.critic = Critic(inv_dim(mdp.num_j) + layout.dim + 3)
        self.opt = MuonAdam(
            flat(self.encoder.trainable_parameters()), self.encoder.net.hidden_weight_paths("net"), betas=(0.9, 0.95)
        )
        self.opt_v = MuonAdam(flat(self.critic.trainable_parameters()), (), betas=(0.9, 0.999))
        self.avg = flat(self.encoder.parameters())
        self.refs = mx.array(motion_commands(motion))
        self.args = args
        self.command_low, self.command_high = np.asarray(COMMAND_RANGES[args.velocity_commands], dtype=np.float32)
        self.grad = nn.value_and_grad(self.encoder, self.rollout)
        self.vgrad = nn.value_and_grad(self.critic, lambda x, y: ((self.critic.head(x) - y) ** 2).mean())
        self.state = [self.encoder.state, self.critic.state, self.opt.state, self.opt_v.state, self.avg]
        self.real_reward = mx.compile(lambda s, c, a, p: reward(mdp, unflatten_state(s), c, a, p))

    def rollout(self, state, history, prev, f, xi, packed, horizon):
        cmd, ref, age = packed[:, :3], packed[:, 3] > 0, packed[:, 4]
        normalized = self.layout.normalize(xi)
        features, rewards, deaths, errors = [], [], [], []
        end = self.motion.clip_end[self.motion.clip_id[f]]
        for _ in range(horizon):
            features.append(mx.concatenate((invariant_features(state), normalized, cmd), -1))
            a, z = action(self.pi, self.encoder, history, cmd)
            teacher = mx.stop_gradient(self.pi.encode(self.motion.goal(f, self.args.goal_frames)))
            errors.append(((z - teacher) ** 2).mean(-1) * ref)
            state = self.wm.step(state, a, normalized)
            rewards.append(reward(self.mdp, state, cmd, a, prev))
            deaths.append(1 - mx.sigmoid((state[0][:, 2] - 0.25) / 0.02) * mx.sigmoid((state[1][:, 2, 2] - 0.3) / 0.04))
            f = mx.minimum(f + 1, end)
            age = age + 1
            refresh = age >= 250
            ref = mx.where(refresh, mx.random.uniform(shape=ref.shape) < 0.5, ref)
            new_cmd = mx.random.uniform(mx.array(self.command_low), mx.array(self.command_high), shape=cmd.shape)
            new_cmd = mx.where(mx.random.uniform(shape=age.shape)[:, None] < 0.1, 0, new_cmd)
            cmd = mx.where(ref[:, None], self.refs[f], mx.where(refresh[:, None], new_cmd, cmd))
            age = mx.where(refresh, 0, age)
            obs = self.mdp.add_noise(self.mdp.obs(state, a, xi, f))
            history, prev = mx.concatenate((history[:, 1:], obs[:, None]), 1), a
        features.append(mx.concatenate((invariant_features(state), normalized, cmd), -1))
        x, r, d = mx.stack(features), mx.stack(rewards), mx.stack(deaths)
        survival = mx.cumprod(1 - d, axis=0)
        weights = mx.concatenate((mx.ones_like(d[:1]), survival[:-1]), 0)
        discount = mx.array([self.args.gamma**t for t in range(horizon)])[:, None]
        ret = (weights * discount * r).sum(0) + self.args.gamma**horizon * survival[-1] * self.critic(x[-1])
        return -ret.mean() + 0.1 * mx.stack(errors).mean(), tuple(mx.stop_gradient(t) for t in (x, r, d))

    def update(self, s, hs, ha, hxi, hf, xi, f, prev, packed, lr, v_lr, ema_w, horizon):
        history = self.mdp.add_noise(self.mdp.obs(unflatten_state(hs), ha, hxi, hf))
        (loss, (x, r, d)), grads = self.grad(unflatten_state(s), history, prev, f, xi, packed, horizon)
        grad, norm = clip_grad_norm(flat(grads), 1.0)
        valid = mx.isfinite(norm) & mx.isfinite(loss)
        grad = {k: mx.where(valid, mx.clip(v, -0.1, 0.1), 0) for k, v in grad.items()}
        self.encoder.update(unflat(self.opt.step(flat(self.encoder.trainable_parameters()), grad, lr, ok=valid)))
        for k, v in flat(self.encoder.parameters()).items():
            self.avg[k] += ema_w * (v - self.avg[k])
        self.critic.norm.update(mx.where(valid, x, 0), ok=valid)
        v = mx.stop_gradient(self.critic(x))
        targets = []
        target = v[-1]
        for t in reversed(range(horizon)):
            target = r[t] + self.args.gamma * (1 - d[t]) * ((1 - self.args.lam) * v[t + 1] + self.args.lam * target)
            targets.append(target)
        y = mx.stack(targets[::-1]).reshape(-1)
        xn = self.critic.norm(x[:-1].reshape(-1, x.shape[-1]))
        for _ in range(self.args.critic_epochs):
            for ids in mx.split(mx.random.permutation(y.size), self.args.critic_minibatches):
                vloss, g = self.vgrad(xn[ids], y[ids])
                g, gn = clip_grad_norm(flat(g), 1.0)
                ok = valid & mx.isfinite(gn)
                g = {k: mx.where(ok, a, 0) for k, a in g.items()}
                self.critic.update(unflat(self.opt_v.step(flat(self.critic.trainable_parameters()), g, v_lr, ok=ok)))
        return loss
