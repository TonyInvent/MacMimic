import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path.cwd() / 'tools'))
from opensprite_control import Controller, drive_sequence, robot

sys.path.insert(0, str(robot.UPSTREAM / "runtime/src"))
from open_sprite_runtime.contracts import PolicyContract
from open_sprite_runtime.policy_shadow import PolicyObservationHistory
from open_sprite_runtime.supermimic_replay import checked_model


class RuntimeShadow(Controller):
    def __init__(self, package):
        contract_path = package / 'contract.json'
        contract = PolicyContract(contract_path, json.loads(contract_path.read_text()))
        self.file = checked_model(contract)
        self.contract = contract
        self.timings = []
        self.max_err = 0.
        super().__init__(package)

    def reset(self):
        super().reset()
        self.shadow_history = PolicyObservationHistory(self.contract)

    def step(self, cmd):
        obs = self.observation()
        self.shadow_history.append(self.data.qpos[7:], self.data.qvel[6:], self.prev, obs[93:96], obs[96:99])
        vector = self.shadow_history.observation(cmd)
        start = time.perf_counter_ns()
        shadow_action = self.session.run(None, {"obs": vector[None]})[0][0]
        self.timings.append((time.perf_counter_ns() - start) * 1e-6)
        result = super().step(cmd)
        self.max_err = max(self.max_err, float(abs(self.prev - shadow_action).max()))
        return result


p=Path(sys.argv[1])
sim=RuntimeShadow(p)
rows=drive_sequence(sim)
result=dict(frames=len(sim.timings),max_abs_action_difference=sim.max_err,
            cpu_inference_ms_p50=float(np.percentile(sim.timings[20:],50)),
            cpu_inference_ms_p95=float(np.percentile(sim.timings[20:],95)),
            model_sha256=hashlib.sha256(sim.file.read_bytes()).hexdigest(),
            completed=not any(r['fallen'] for r in rows),source='upstream PolicyObservationHistory on simulated joint and IMU feedback')
print(result)
(p/'upstream_history_replay.json').write_text(json.dumps(result,indent=2))
assert result['completed'] and sim.max_err < 2e-4
