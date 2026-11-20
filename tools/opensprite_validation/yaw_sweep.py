"""Inspect R1 velocity commands throughout the trained yaw and forward ranges."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path.cwd()/'tools'))
from opensprite_control import Controller

p=Path(sys.argv[1]); sim=Controller(p); rows=[]
for speed in (.25,.4,.5):
    for yaw in (-.6,-.4,-.2,0,.2,.4,.6):
        sim.reset()
        samples=[]
        for tick in range(750):
            command=np.array([speed,0,yaw]) * min(tick/150,1)
            velocity,fallen=sim.step(command)
            if fallen: break
            if tick>=500: samples.append(velocity)
        rows.append({'command':[speed,0,yaw], 'fallen':bool(fallen), 'mean':np.mean(samples,0).tolist() if samples else None})
(p/'yaw_sweep.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:
    if row['command'][0]==.4: print(row)
