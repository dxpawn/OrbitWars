"""Easier opponent pool for fast initial training.

Excludes the heaviest adversaries (3000+ line heuristics that slow rollouts).
Use this for first training pass, then swap to default_league once policy
isn't pure random.
"""

from rl.league import League


def easy_league() -> League:
    L = League()
    for name in ("random", "do_nothing", "nearest_sniper", "rusher", "defender"):
        L.add(name, weight=1.0)
    # Mid-difficulty
    L.add("heuristic_v1", weight=1.2)
    L.add("adv_rf_v2", weight=1.5)  # smallest adversary (~560 lines)
    L.add("adv_rf_v1", weight=1.5)  # ~700 lines
    return L
