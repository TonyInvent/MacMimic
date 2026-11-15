"""Checkpoints: nested dicts of MLX arrays in one safetensors file (<path>.safetensors, keys joined with '/'), plus the
JSON-serializable 'meta' entry in <path>.json. Writes are atomic (temporary file + rename)."""

from __future__ import annotations

import json
import os

import mlx.core as mx


def _flatten(tree: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in tree.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "/"))
        else:
            out[key] = v
    return out


def save_tree(path: str, tree: dict) -> None:
    tree = dict(tree)
    meta = tree.pop("meta", {})
    arrays = _flatten(tree)
    mx.save_safetensors(path + ".tmp.safetensors", arrays)
    os.replace(path + ".tmp.safetensors", path + ".safetensors")
    with open(path + ".tmp.json", "w") as f:
        json.dump(meta, f)
    os.replace(path + ".tmp.json", path + ".json")


def load_tree(path: str) -> dict:
    """{top-level key: {rest of the key: array}} (the policy / model dicts keep their dotted parameter paths)."""
    path = path.removesuffix(".safetensors")
    out: dict = {}
    for k, v in mx.load(path + ".safetensors").items():
        parts = k.split("/")
        d = out
        for p in parts[:-2] if len(parts) > 2 else parts[:-1]:
            d = d.setdefault(p, {})
        if len(parts) > 2:
            d.setdefault(parts[-2], {})[parts[-1]] = v
        else:
            d[parts[-1]] = v
    with open(path + ".json") as f:
        out["meta"] = json.load(f)
    return out
