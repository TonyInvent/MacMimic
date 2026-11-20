"""Published Sprite SM-Walk R1 reference identity and clip reset distribution."""

import hashlib

import numpy as np

R1_BANK_SHA256 = "edd4ddd6f54e8a8a710481d11340ac6e38d6b944e75613c48e318fc3b32d296f"
R1_RESET_WEIGHTS = {"pm01_clearance_walk": 0.8, "Neutral_ID_001135_001495_stand": 0.2}


def check_r1_file(path):
    with open(path, "rb") as bank:
        checksum = hashlib.file_digest(bank, "sha256").hexdigest()
    if checksum != R1_BANK_SHA256:
        raise ValueError(f"R1 requires walk_stand.npz SHA256 {R1_BANK_SHA256}; supplied {checksum}")
    return checksum


def validate_clip_layout(names, lengths):
    if list(names) != list(R1_RESET_WEIGHTS) or list(lengths) != [572, 300]:
        raise ValueError("R1 needs the 572-frame corrected walk and the 300-frame training-split stand")


def with_clip_masses(bin_weights, bin_clip, clip_mass):
    """Preserve MacMimic reset adaptation inside each clip while imposing its total mass."""
    totals = np.bincount(bin_clip, weights=bin_weights, minlength=len(clip_mass))
    if np.any(totals <= 0):
        raise ValueError("Every reference clip needs a positive reset bin weight")
    return bin_weights * (clip_mass / totals)[bin_clip]
