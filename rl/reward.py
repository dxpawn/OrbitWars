"""Reward shaping for Orbit Wars RL.

Per-step reward = small shaping signal (planet capture diffs + ship-advantage
delta). Terminal reward = +1 for outright win, +0.5 for tied-for-first, -1
for loss.

Shaping coefficient should be ANNEALED toward 0 during training so the
agent ultimately optimizes the true terminal signal.
"""

from __future__ import annotations


def compute_step_reward(prev_obs, cur_obs, player: int, shape_coef: float = 1.0) -> float:
    """Reward for the transition prev_obs → cur_obs."""
    if prev_obs is None:
        return 0.0

    def _stats(obs):
        if isinstance(obs, dict):
            planets = obs.get("planets") or []
            fleets = obs.get("fleets") or []
        else:
            planets = getattr(obs, "planets", None) or []
            fleets = getattr(obs, "fleets", None) or []
        my_planets = 0
        my_ships = 0
        my_prod = 0
        enemy_ships = 0
        for p in planets:
            if p[1] == player:
                my_planets += 1
                my_ships += p[5]
                my_prod += p[6]
            elif p[1] != -1:
                enemy_ships += p[5]
        for f in fleets:
            if f[1] == player:
                my_ships += f[6]
            elif f[1] != -1:
                enemy_ships += f[6]
        return my_planets, my_ships, my_prod, enemy_ships

    pm, ps, pp, pe = _stats(prev_obs)
    cm, cs, cp, ce = _stats(cur_obs)

    # Planet capture / loss
    delta_planets = (cm - pm)
    # Ship advantage delta (small)
    delta_adv = (cs - ce) - (ps - pe)
    # Production delta
    delta_prod = (cp - pp)

    reward = (
        0.05 * delta_planets
        + 0.0002 * delta_adv
        + 0.02 * delta_prod
    )
    return float(reward * shape_coef)


def compute_terminal_reward(final_obs, player: int, n_players: int) -> float:
    """Engine-style reward at episode end.

    Engine awards +1 to all players tied for max score (when max_score > 0),
    -1 to everyone else. We mirror but treat ties as +0.5 to discourage
    coasting into a draw.
    """
    if isinstance(final_obs, dict):
        planets = final_obs.get("planets") or []
        fleets = final_obs.get("fleets") or []
    else:
        planets = getattr(final_obs, "planets", None) or []
        fleets = getattr(final_obs, "fleets", None) or []

    scores = [0.0] * n_players
    for p in planets:
        if p[1] != -1 and p[1] < n_players:
            scores[p[1]] += p[5]
    for f in fleets:
        if f[1] < n_players:
            scores[f[1]] += f[6]

    max_score = max(scores) if scores else 0
    if max_score <= 0:
        return -1.0
    winners = [i for i, s in enumerate(scores) if s == max_score]
    if player not in winners:
        return -1.0
    if len(winners) == 1:
        return 1.0
    return 0.5  # tied for first
