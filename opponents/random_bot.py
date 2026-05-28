"""Random agent — uniform random moves from owned planets.

Each turn, with probability 0.5, picks a random owned planet and sends a
random fraction of its garrison at a random angle. Functions as the floor
of any opponent pool.
"""

import math
import random


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

    my_planets = [p for p in raw_planets if p[1] == player and p[5] >= 2]
    if not my_planets:
        return []

    moves = []
    for p in my_planets:
        if random.random() < 0.5:
            continue
        pid, _owner, _x, _y, _r, ships, _prod = p
        send = random.randint(1, max(1, int(ships) // 2))
        angle = random.uniform(-math.pi, math.pi)
        moves.append([pid, angle, send])
    return moves
