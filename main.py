"""Orbit Wars — competition submission entrypoint.

Auto-selects the best available agent:
  1. If checkpoints/best.pt exists, load the RL policy (TODO once trained).
  2. Otherwise, fall back to agents/heuristic_v1.

The kaggle competition runner imports this module and calls `agent(obs)`
each turn.
"""

import os
import sys

# Make sibling packages importable when this file is the submission root.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)


_USE_RL = os.path.exists(os.path.join(_here, "checkpoints", "best.pt"))

if _USE_RL:
    # Placeholder — wire up once we have a trained checkpoint.
    from agents.heuristic_v1 import agent  # noqa: F401
else:
    from agents.heuristic_v1 import agent  # noqa: F401
