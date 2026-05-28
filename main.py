"""Orbit Wars — competition submission entrypoint.

Auto-selects the best available agent:
  1. If checkpoints/best.pt exists, load the RL policy.
  2. Otherwise, fall back to agents/heuristic_v1.

The Kaggle competition runner imports this module and calls `agent(obs)`.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

_BEST_CKPT = os.path.join(_here, "checkpoints", "best.pt")

if os.path.exists(_BEST_CKPT):
    from agents.rl_inference import make_agent
    agent = make_agent(_BEST_CKPT, deterministic=True)
else:
    from agents.heuristic_v1 import agent  # noqa: F401
