"""Select an embodiment before constructing simulator and learner state."""
import os

name = os.environ.get("MACMIMIC_ROBOT", "g1")
if name == "g1":
    from . import g1 as active
elif name == "opensprite":
    from . import opensprite as active
else:
    raise ValueError(f"Unknown MACMIMIC_ROBOT: {name}")
