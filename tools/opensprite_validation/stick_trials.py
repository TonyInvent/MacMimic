from pathlib import Path
import json
import sys

import mujoco
import numpy as np

root = Path.cwd()
sys.path.insert(0, str(root / "tools"))
from opensprite_control import Controller, StickMapping, drive_sequence


class VariedStart(Controller):
    seed = 0

    def reset(self):
        super().reset()
        if self.seed:
            rng = np.random.default_rng(self.seed)
            self.data.qpos[7:] += rng.uniform(-0.025, 0.025, len(self.q0))
            self.data.qvel[:2] = rng.uniform(-0.04, 0.04, 2)
            self.data.qvel[3:6] = rng.uniform(-0.04, 0.04, 3)
            mujoco.mj_forward(self.model, self.data)
            self.history[:] = self.observation()


package = Path(sys.argv[1])
c = VariedStart(package)
trials = {}
for seed in [0, 1, 2, 3]:
    c.seed = seed
    trials[str(seed)] = drive_sequence(c)
(package / "xbox_sequence.json").write_text(json.dumps(trials["0"], indent=2))
(package / "xbox_varied_starts.json").write_text(json.dumps(trials, indent=2))
c.seed = 0
analog = []
for fwd, turn in [(-1, 0), (0.5, 0), (0.75, 0), (1, 0), (1, 0.5), (1, -0.5), (0, 1), (0, -1)]:
    c.reset()
    stick = StickMapping()
    samples = []
    for step in range(650):
        if step < 50:
            cmd = stick.update(0, 0, 0)
        elif step < 100:
            cmd = stick.update(0, 0, 1)
        else:
            cmd = stick.update(fwd, turn, 1)
        vel, fell = c.step(cmd)
        if step >= 550:
            samples.append(vel.copy())
        if fell:
            break
    analog.append(dict(stick=[fwd, turn], command=cmd.tolist(), measured=np.mean(samples, 0).tolist() if samples else None, fallen=bool(fell)))
(package / "xbox_analog.json").write_text(json.dumps(analog, indent=2))
