import ctypes as C
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame
from pygame._sdl2 import controller
sys.path.insert(0, str(Path.cwd() / "tools"))
from opensprite_control import Xbox, Controller

p=Path(sys.argv[1])
sim=Controller(p)
pygame.init()
controller.init()
lib=C.CDLL(str(Path(pygame.__file__).parent/'.dylibs/libSDL2-2.0.0.dylib'))
lib.SDL_JoystickAttachVirtual.argtypes=[C.c_int]*4
lib.SDL_JoystickOpen.argtypes=[C.c_int]
lib.SDL_JoystickOpen.restype=C.c_void_p
lib.SDL_JoystickSetVirtualAxis.argtypes=[C.c_void_p,C.c_int,C.c_int16]
lib.SDL_JoystickClose.argtypes=[C.c_void_p]
idx=lib.SDL_JoystickAttachVirtual(1,6,16,0)
assert idx>=0 and controller.is_controller(idx)
joy=lib.SDL_JoystickOpen(idx)
pad=Xbox()
assert pad.pad.id == idx, 'unexpected controller'
schedule=[
    ("neutral", 1, 0, 0, 0), ("enable", 1, 0, 0, 1),
    ("cruise", 12, .85, 0, 1), ("left", 8, .85, .75, 1),
    ("right", 9, .95, -.85, 1), ("brake", 8, .95, -.85, 0),
    ("center", 1, 0, 0, 0), ("re-enable", 1, 0, 0, 1),
    ("restart", 12, 1, 0, 1), ("release", 7, 0, 0, 0)]
rows=[]
try:
    for label,secs,x,w,lt in schedule:
        for axis,val in [(1,-x),(2,-w),(4,2*lt-1)]:
            assert lib.SDL_JoystickSetVirtualAxis(joy,axis,round(val*(32767 if val>=0 else 32768)))==0
        lib.SDL_JoystickUpdate()
        vel=[]
        for i in range(50*secs):
            cmd=pad.sample()
            measured,fallen=sim.step(cmd)
            if i>=(secs-min(secs,2))*50: vel.append(measured.copy())
            if fallen: break
        rows.append(dict(phase=label, stick=[x,w,lt], seconds=(i+1)/50, command=cmd.tolist(),measured=np.mean(vel,axis=0).tolist() if vel else None,fallen=bool(fallen)))
        if fallen: break
finally:
    lib.SDL_JoystickClose(joy)
    lib.SDL_JoystickDetachVirtual(idx)
    pygame.quit()
result=dict(input='SDL2 virtual game controller, end-to-end simulation',rows=rows)
(p/'sdl_driving.json').write_text(json.dumps(result,indent=2))
print([(r['phase'],r['seconds'],np.round(r['measured'],3).tolist() if r['measured'] else None) for r in rows])
