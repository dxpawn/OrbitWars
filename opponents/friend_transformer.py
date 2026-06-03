"""Lazy loader for the friend's imitation-transformer agent (LB ~1140.9).

Multi-file submission in other_adversaries/submission_feature46_transformer_v2_late_recapture_2p_v1/
(main.py + orbit_base.py + feature46_weights_2p/4p.py). We load his main.py under a
UNIQUE module name (so it doesn't clobber our repo's main.py) with his folder on
sys.path (so his sibling imports `orbit_base`, `feature46_weights_2p/4p` resolve).

Loaded lazily on first agent() call so merely importing the opponents registry
doesn't pay the ~7.6 MB weight-import cost in every spawn worker that never uses it.
"""
import os
import sys
import importlib.util

_FRIEND_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "other_adversaries",
    "submission_feature46_transformer_v2_late_recapture_2p_v1",
))

_agent = None


def _load():
    global _agent
    if _agent is not None:
        return _agent
    if _FRIEND_DIR not in sys.path:
        sys.path.insert(0, _FRIEND_DIR)  # so his sibling imports resolve
    spec = importlib.util.spec_from_file_location(
        "friend_main_transformer", os.path.join(_FRIEND_DIR, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs his main.py (imports orbit_base + weights from sys.path)
    _agent = mod.agent
    return _agent


def agent(obs, config=None):
    return _load()(obs, config)
