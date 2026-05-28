"""Inference wrapper: load a trained checkpoint, expose `agent(obs)`."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from rl.action_space import sample_action
from rl.features import encode
from rl.policy import OrbitWarsPolicy

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CKPT = _HERE.parent / "checkpoints" / "best.pt"


_POLICY: OrbitWarsPolicy | None = None
_DEVICE = torch.device("cpu")  # submissions run on Kaggle CPU


def _load_policy(checkpoint_path: Path):
    global _POLICY
    if _POLICY is not None:
        return _POLICY
    payload = torch.load(checkpoint_path, map_location="cpu")
    # Try to recover architecture hints from the payload if present;
    # otherwise default to training defaults.
    d_model = payload.get("d_model", 96)
    n_heads = payload.get("n_heads", 4)
    n_layers = payload.get("n_layers", 3)
    _POLICY = OrbitWarsPolicy(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    _POLICY.load_state_dict(payload["policy"])
    _POLICY.eval()
    return _POLICY


def make_agent(checkpoint_path: Path | None = None, *, deterministic: bool = True):
    ckpt = Path(checkpoint_path or _DEFAULT_CKPT)
    policy = _load_policy(ckpt)

    def agent(obs):
        enc = encode(obs)
        if len(enc.my_planet_slots) == 0:
            return []
        ent = torch.from_numpy(enc.entities).unsqueeze(0)
        mask = torch.from_numpy(enc.mask).unsqueeze(0)
        gl = torch.from_numpy(enc.globals_).unsqueeze(0)
        with torch.no_grad():
            out = policy(ent, mask, gl)
        moves, _ = sample_action(out, enc, obs, deterministic=deterministic)
        return moves

    return agent


# Default callable for submission imports
agent = None
if _DEFAULT_CKPT.exists():
    agent = make_agent(_DEFAULT_CKPT, deterministic=True)
