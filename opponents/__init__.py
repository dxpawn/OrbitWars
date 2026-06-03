"""Opponent registry — maps name → agent (callable OR absolute file path).

Callables come from local opponents. File paths are passed through to
kaggle_environments.env.run, which loads them as agent modules — used for
the real-world adversaries dumped into ../other_adversaries/.
"""

import os

from agents import heuristic_v1, heuristic_v2, heuristic_v3, heuristic_v4, heuristic_v5, heuristic_v6, heuristic_v6_1017, heuristic_tune
from opponents import defender, do_nothing, nearest_sniper, random_bot, rusher, friend_transformer, distilled

_HERE = os.path.dirname(os.path.abspath(__file__))
_ADV = os.path.abspath(os.path.join(_HERE, "..", "other_adversaries"))

REGISTRY: dict[str, object] = {
    # In-house opponents
    "random": random_bot.agent,
    "do_nothing": do_nothing.agent,
    "nearest_sniper": nearest_sniper.agent,
    "defender": defender.agent,
    "rusher": rusher.agent,
    "heuristic_v1": heuristic_v1.agent,
    "heuristic_v2": heuristic_v2.agent,
    "heuristic_v3": heuristic_v3.agent,
    "heuristic_v4": heuristic_v4.agent,
    "heuristic_v5": heuristic_v5.agent,
    "heuristic_v6": heuristic_v6.agent,
    "heuristic_v6_1017": heuristic_v6_1017.agent,  # FROZEN snapshot of the Kaggle 1017 upload (fallback)
    "heuristic_tune": heuristic_tune.agent,
    # Real Kaggle submissions (file paths — env.run loads them as modules)
    "adv_distance": os.path.join(_ADV, "Distance-Prioritized Agent.py"),
    "adv_lbmax": os.path.join(_ADV, "LBMAX1224.py"),
    "adv_structured": os.path.join(_ADV, "Structured Baseline.py"),
    "adv_rf_v0": os.path.join(_ADV, "rf_v0.py"),
    "adv_rf_v1": os.path.join(_ADV, "rf_v1.py"),
    "adv_rf_v2": os.path.join(_ADV, "rf_v2.py"),
    # Newer / stronger public adversaries
    "adv_ver16": os.path.join(_ADV, "ver16-800score.py"),  # teammate's 825-pt agent
    "adv_lb958": os.path.join(_ADV, "LB958.py"),           # name suggests ~958-pt LB
    "adv_hellburner": os.path.join(_ADV, "hellburner.py"),
    "adv_proto_v15": os.path.join(_ADV, "Proto-V15.py"),
    "adv_in_progress": os.path.join(_ADV, "inProgress.py"),
    "adv_heuristic1000": os.path.join(_ADV, "HEURISTIC1000.py"),  # user reports 1000-1100 LB, beats teammate
    "adv_friend_tf": friend_transformer.agent,  # imitation-transformer re-ranker, LB ~1140.9 (the teacher)
    "ours_distilled": distilled.agent,           # OUR student re-ranker on the reused hull (the deliverable)
}


def get(name):
    """Return the agent (callable or file path) for `name`. Raises KeyError."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown opponent: {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def names():
    return sorted(REGISTRY)


def adversary_names():
    """The real-world Kaggle adversaries (not in-house)."""
    return sorted(n for n in REGISTRY if n.startswith("adv_"))
