"""Defender — only reinforces threatened planets, never attacks neutrals/enemies.

Strategy: when an enemy fleet is heading toward one of our planets, ship
extra ships from the closest ally to absorb the impact. Otherwise sit on
ships and let production accumulate.
"""

import math


def _est_fleet_arrival(fleet, target_planet):
    """Rough estimate: time for fleet to reach `target_planet` along its ray.
    Returns None if the fleet isn't heading there.
    """
    fx, fy, fa, fs = fleet[2], fleet[3], fleet[4], fleet[6]
    tx, ty, tr = target_planet[2], target_planet[3], target_planet[4]
    cos_a, sin_a = math.cos(fa), math.sin(fa)
    dx, dy = tx - fx, ty - fy
    # Project onto the ray direction
    proj = dx * cos_a + dy * sin_a
    if proj < 0:
        return None
    perp_sq = dx * dx + dy * dy - proj * proj
    if perp_sq >= tr * tr:
        return None
    hit_dist = max(0.0, proj - math.sqrt(max(0.0, tr * tr - perp_sq)))
    # Fleet speed
    speed = 1.0 + 5.0 * (math.log(max(1, int(fs))) / math.log(1000)) ** 1.5
    speed = min(speed, 6.0)
    return hit_dist / max(speed, 1e-6)


def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets

    my_planets = [p for p in planets if p[1] == player]
    if not my_planets:
        return moves

    # For each owned planet, compute incoming enemy ships within the next 40 turns
    for mine in my_planets:
        incoming = 0
        for f in fleets:
            if f[1] == player:
                continue
            eta = _est_fleet_arrival(f, mine)
            if eta is None or eta > 40:
                continue
            incoming += f[6]

        if incoming == 0:
            continue

        deficit = incoming - mine[5]
        if deficit <= 0:
            continue  # We can hold without help

        # Find closest ally with surplus ships (keep at least 5 in reserve)
        donors = sorted(
            (p for p in my_planets if p[0] != mine[0] and p[5] > 5),
            key=lambda p: math.hypot(p[2] - mine[2], p[3] - mine[3]),
        )
        for ally in donors:
            available = ally[5] - 5
            send = min(available, deficit)
            if send < 1:
                continue
            angle = math.atan2(mine[3] - ally[3], mine[2] - ally[2])
            moves.append([ally[0], angle, int(send)])
            deficit -= send
            if deficit <= 0:
                break

    return moves
