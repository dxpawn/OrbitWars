"""OUR distilled agent: the friend's hull + feature extractor, with HIS transformer
scorer replaced by OUR distilled student (rl/student/student_weights_{2p,4p}.py).

We load his main.py but inject our student modules under the names his code imports
(`feature46_weights_2p/4p`), so his `_candidate_features` + `_attn_nearest_targets`
+ `orbit_base` hull all run unchanged while the SCORE comes from our model. His
7.6 MB transformer is never loaded. sys.modules is restored after binding so this
never pollutes a process that also runs his real agent.
"""
import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
FRIEND_DIR = os.path.abspath(os.path.join(
    _HERE, "..", "other_adversaries",
    "submission_feature46_transformer_v2_late_recapture_2p_v1"))
STUDENT_DIR = os.path.abspath(os.path.join(_HERE, "..", "rl", "student"))

_agent = None


def _load():
    global _agent
    if _agent is not None:
        return _agent
    for d in (FRIEND_DIR, STUDENT_DIR):
        if d not in sys.path:
            sys.path.insert(0, d)
    import student_weights_2p
    import student_weights_4p
    keys = ("feature46_weights_2p", "feature46_weights_4p")
    saved = {k: sys.modules.get(k) for k in keys}
    sys.modules["feature46_weights_2p"] = student_weights_2p
    sys.modules["feature46_weights_4p"] = student_weights_4p
    try:
        spec = importlib.util.spec_from_file_location(
            "distilled_main", os.path.join(FRIEND_DIR, "main.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)  # his main binds scorer_2p/4p = OUR student now
        _agent = m.agent
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return _agent


def agent(obs, config=None):
    return _load()(obs, config)
