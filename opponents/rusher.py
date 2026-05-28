"""Rusher — early aggression on closest enemy.

Strategy: hold ships until a critical mass, then send them at the nearest
enemy-owned planet from every owned planet. Treats neutrals as obstacles, not
targets (no expansion). Loses badly if it can't kill the enemy fast, but
hard to handle if the opponent isn't ready.
"""

import math


def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)

    my_planets = [p for p in planets if p[1] == player]
    enemy_planets = [p for p in planets if p[1] not in (player, -1)]
    if not my_planets or not enemy_planets:
        # In opening, also rush nearest neutral as a beach-head
        targets = [p for p in planets if p[1] != player]
        if not targets:
            return moves
    else:
        targets = enemy_planets

    # Hold buildup until step 30 to mass forces, then commit
    if step < 30:
        return moves

    for mine in my_planets:
        if mine[5] < 25:
            continue
        nearest = min(targets, key=lambda t: math.hypot(t[2] - mine[2], t[3] - mine[3]))
        # Send almost everything (leave 3 ships)
        send = max(1, mine[5] - 3)
        angle = math.atan2(nearest[3] - mine[3], nearest[2] - mine[2])
        moves.append([mine[0], angle, int(send)])
    return moves
