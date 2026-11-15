"""Reference motion library: retargeted clips resampled to 50 Hz, with FK of the tracked bodies.

Tables live on the GPU as MLX arrays (goal features, rewards) and on the CPU as NumPy arrays (simulator resets and
the reset sampler).
"""

from __future__ import annotations

import glob
import os

import mlx.core as mx
import numpy as np

from .kinematics import Kinematics
from .rotations import mat_to_6d, mm3, mv3, quat_slerp, quat_to_mat, so3_log, yaw_mat, T
from .scene import CONTROL_DT

BIN_FRAMES = 50  # 1 s bins for return-based reset sampling
GOAL_Q_OFFSET = 15  # index of q(f+1) in the goal vector (first future frame: dp 3, rot 6, v 3, w 3, then q)


def _gradient(x: np.ndarray, dt: float) -> np.ndarray:
    g = np.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g


class MotionLib:
    def __init__(self, files: list[str], kin: Kinematics, load_motion):
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        ps, qs, js, lens = [], [], [], []
        for f in files:
            p, quat, j, src_fps = load_motion(f)
            p, quat, j = (np.asarray(x, dtype=np.float32) for x in (p, quat, j))
            T_src = p.shape[0]
            dur = (T_src - 1) / src_fps
            n = int(np.floor(dur / CONTROL_DT)) + 1
            t = np.arange(n, dtype=np.float32) * CONTROL_DT * src_fps
            i0 = np.minimum(np.floor(t).astype(np.int64), T_src - 2)
            a = np.clip(t - i0.astype(np.float32), 0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = np.array(quat_slerp(mx.array(quat[i0]), mx.array(quat[i0 + 1]), mx.array(a)))
            ps.append(p), qs.append(quat), js.append(j), lens.append(n)
        self.clip_len_np = np.array(lens, dtype=np.int64)
        self.clip_start_np = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int64)
        self.clip_end_np = self.clip_start_np + self.clip_len_np - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id_np = np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(lens)])

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = np.array(quat_to_mat(mx.array(quat)))
            v, qd = _gradient(p, CONTROL_DT), _gradient(j, CONTROL_DT)
            w = np.zeros_like(p)  # world angular velocity by central differences of rotations
            Rm = mx.array(R)
            w[1:-1] = np.array(so3_log(mm3(Rm[2:], T(Rm[:-2])))) / (2 * CONTROL_DT)
            w[0] = np.array(so3_log(mm3(Rm[1], T(Rm[0])))) / CONTROL_DT
            w[-1] = np.array(so3_log(mm3(Rm[-1], T(Rm[-2])))) / CONTROL_DT
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: np.concatenate(xs).astype(np.float32)  # noqa: E731
        self.np = {"root_p": cat(root_p), "root_R": cat(root_R), "root_v": cat(root_v), "root_w": cat(root_w),
                   "q": cat(jq), "qd": cat(jqd)}  # fmt: skip
        g = {k: mx.array(v) for k, v in self.np.items()}
        self.root_p, self.root_R, self.root_v, self.root_w, self.q, self.qd = (
            g["root_p"], g["root_R"], g["root_v"], g["root_w"], g["q"], g["qd"])  # fmt: skip
        bp, bR, bv, bw = [], [], [], []
        for s in range(0, self.num_frames, 65536):
            e = min(s + 65536, self.num_frames)
            out = kin.forward(self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e])
            mx.eval(out)
            bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
        self.body_p, self.body_R, self.body_v, self.body_w = (mx.concatenate(x) for x in (bp, bR, bv, bw))
        Rt = T(self.root_R)
        self.root_v_local = mv3(Rt, self.root_v)
        self.root_w_local = mv3(Rt, self.root_w)
        # 1 s bins
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            for b in range(int(np.ceil(lens[c] / BIN_FRAMES))):
                bin_clip.append(c), bin_start.append(int(self.clip_start_np[c]) + b * BIN_FRAMES)
        self.bin_clip_np = np.array(bin_clip, dtype=np.int64)
        self.bin_start_np = np.array(bin_start, dtype=np.int64)
        self.num_bins = len(bin_start)
        self.frame_bin_np = np.zeros(self.num_frames, dtype=np.int64)
        for b in range(self.num_bins):
            c, s = bin_clip[b], bin_start[b]
            self.frame_bin_np[s : min(s + BIN_FRAMES, int(self.clip_end_np[c]) + 1)] = b
        self.clip_end = mx.array(self.clip_end_np.astype(np.int32))
        self.clip_id = mx.array(self.clip_id_np.astype(np.int32))
        self.ref_contact = None
        mx.eval(self.body_p, self.body_R, self.body_v, self.body_w, self.root_v_local, self.root_w_local)

    @staticmethod
    def from_dir(path: str, kin: Kinematics, load_motion, glob_pat: str = "*.csv") -> "MotionLib":
        files = sorted(glob.glob(os.path.join(path, glob_pat)))
        assert files, f"no motions in {path} (run tools/download_data.sh)"
        return MotionLib(files, kin, load_motion)

    def set_ref_contact(self, feet: list[int], margin: float = 0.015, vmax: float = 0.5, q: float = 0.05) -> None:
        """Reference foot contact (num_frames, len(feet)) from a per-clip floor (the q-quantile of the foot height in
        the clip) + margin and horizontal speed below vmax (upstream --ref_contact_fix)."""
        z = np.array(self.body_p[:, feet, 2])
        vh = np.linalg.norm(np.array(self.body_v[:, feet, :2]), axis=-1)
        floor = np.zeros_like(z)
        for c in range(len(self.names)):
            s = self.clip_id_np == c
            floor[s] = np.quantile(z[s], q, axis=0, method="linear")
        self.ref_contact = mx.array(((z < floor + margin) & (vh < vmax)).astype(np.float32))

    def goal_dim(self, K: int) -> int:
        return K * (3 + 6 + 3 + 3 + 2 * self.q.shape[-1])

    def future_frames(self, f: mx.array, K: int) -> mx.array:
        end = self.clip_end[self.clip_id[f]]
        fut = f[:, None] + mx.arange(1, K + 1, dtype=f.dtype)
        return mx.minimum(fut, end[:, None])

    def goal(self, f: mx.array, K: int) -> mx.array:
        """Goal features from the reference only: K future frames relative to the current reference frame."""
        fut = self.future_frames(f, K)
        Ht = mx.expand_dims(T(yaw_mat(self.root_R[f])), 1)
        dp = mv3(Ht, self.root_p[fut] - mx.expand_dims(self.root_p[f], 1))
        rot = mat_to_6d(mm3(Ht, self.root_R[fut]))
        feats = mx.concatenate((dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1)
        return feats.reshape(f.shape[0], -1)

    def ref_state_np(self, f: np.ndarray):
        d = self.np
        return d["root_p"][f], d["root_R"][f], d["root_v"][f], d["root_w"][f], d["q"][f], d["qd"][f]
