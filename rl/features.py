"""Observation → tensor encoding.

Variable-length entity list (planets + fleets + structural tokens), padded
to a fixed length with a mask. Designed to be permutation-invariant via the
downstream transformer.

Per-entity feature dim: 32. See ENTITY_DIM.
Per-global feature dim: 12.  See GLOBAL_DIM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Constants ---------------------------------------------------------------
MAX_ENTITIES = 96       # planets + fleets + 1 sun token + 1 global token
ENTITY_DIM = 32
GLOBAL_DIM = 12
BOARD = 100.0
CENTER = 50.0
TOTAL_STEPS = 500
SUN_R = 10.0
ROTATION_RADIUS_LIMIT = 50.0

# Entity type onehots (one-hot at indices [0..3] in the feature vector)
TYPE_PLANET = 0
TYPE_FLEET = 1
TYPE_SUN = 2
TYPE_GLOBAL = 3


@dataclass
class EncodedObs:
    entities: np.ndarray   # (MAX_ENTITIES, ENTITY_DIM) float32
    mask: np.ndarray       # (MAX_ENTITIES,) bool — True = real entity, False = padding
    globals_: np.ndarray   # (GLOBAL_DIM,) float32
    # Index-mapping helpers for action decoding:
    planet_slot_ids: np.ndarray  # (MAX_ENTITIES,) int32 — planet_id at each slot (-1 if not a planet)
    fleet_slot_ids: np.ndarray   # (MAX_ENTITIES,) int32 — fleet_id at each slot (-1 if not a fleet)
    my_planet_slots: np.ndarray  # (num_my_planets,) int32 — slot indices of owned planets
    my_planet_ids: np.ndarray    # (num_my_planets,) int32 — corresponding planet ids
    step: int


def _log_scale(x: float, base: float = 100.0) -> float:
    return math.log1p(max(0.0, x)) / math.log(1.0 + base)


def encode(obs, *, max_entities: int = MAX_ENTITIES) -> EncodedObs:
    """Encode a game observation into fixed-size tensors.

    Layout per slot:
      [0:4]  type onehot (planet, fleet, sun, global)
      [4:9]  owner onehot: me, ally(=-1 unused for now), enemy_a, enemy_b, neutral
      [9]    x_norm   (0..1)
      [10]   y_norm
      [11]   radius_norm
      [12]   ships_log
      [13]   production (0..5)
      [14]   is_orbiting
      [15]   is_comet
      [16]   is_home_relative_dist (from any of my planets, normalized)
      [17]   angle_to_center_sin
      [18]   angle_to_center_cos
      [19]   orbital_radius_norm
      [20]   fleet_vx (cos angle, only for fleets)
      [21]   fleet_vy (sin angle, only for fleets)
      [22]   fleet_speed_norm
      [23]   turns_to_impact_est (only for fleets, capped at 60)
      [24]   is_mine                    (owner == player)
      [25]   ships_normalized_linear (ships / 200, clipped)
      [26]   production_norm (prod / 5)
      [27]   step_norm
      [28..31]  reserved zeros
    """
    if isinstance(obs, dict):
        player = int(obs.get("player", 0) or 0)
        step = int(obs.get("step", 0) or 0)
        planets = obs.get("planets") or []
        fleets = obs.get("fleets") or []
        initial_planets = obs.get("initial_planets") or []
        omega = float(obs.get("angular_velocity", 0.03) or 0.03)
        comet_ids = set(obs.get("comet_planet_ids") or [])
    else:
        player = int(getattr(obs, "player", 0) or 0)
        step = int(getattr(obs, "step", 0) or 0)
        planets = getattr(obs, "planets", None) or []
        fleets = getattr(obs, "fleets", None) or []
        initial_planets = getattr(obs, "initial_planets", None) or []
        omega = float(getattr(obs, "angular_velocity", 0.03) or 0.03)
        comet_ids = set(getattr(obs, "comet_planet_ids", None) or [])

    entities = np.zeros((max_entities, ENTITY_DIM), dtype=np.float32)
    mask = np.zeros(max_entities, dtype=bool)
    planet_slot_ids = np.full(max_entities, -1, dtype=np.int32)
    fleet_slot_ids = np.full(max_entities, -1, dtype=np.int32)
    initial_by_id = {p[0]: p for p in initial_planets}

    my_planet_slots: list[int] = []
    my_planet_ids: list[int] = []

    my_planet_coords = [(p[2], p[3]) for p in planets if p[1] == player]

    def _min_dist_to_mine(x, y):
        if not my_planet_coords:
            return 0.0
        return min(math.hypot(x - mx, y - my) for mx, my in my_planet_coords)

    def _owner_onehot(owner):
        v = np.zeros(5, dtype=np.float32)
        if owner == player:
            v[0] = 1.0
        elif owner == -1:
            v[4] = 1.0
        else:
            # Distinguish up to 2 enemies (slots 2 and 3 reserved for them)
            # In 4-player, owners 0..3 minus player → enemies. We mod into [0,1].
            v[2 + (owner % 2)] = 1.0
        return v

    slot = 0

    # ---- Planets ----
    for p in planets:
        if slot >= max_entities - 2:
            break
        pid, owner, x, y, r, ships, prod = p[0], p[1], float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])
        v = np.zeros(ENTITY_DIM, dtype=np.float32)
        v[TYPE_PLANET] = 1.0
        v[4:9] = _owner_onehot(owner)
        v[9] = x / BOARD
        v[10] = y / BOARD
        v[11] = r / 5.0
        v[12] = _log_scale(ships, 100.0)
        v[13] = prod
        # is_orbiting from initial position
        init = initial_by_id.get(pid)
        ix0, iy0 = (init[2], init[3]) if init else (x, y)
        ir0 = init[4] if init else r
        orbital_r = math.hypot(ix0 - CENTER, iy0 - CENTER)
        v[14] = 1.0 if (orbital_r + ir0 < ROTATION_RADIUS_LIMIT) else 0.0
        v[15] = 1.0 if pid in comet_ids else 0.0
        v[16] = min(1.0, _min_dist_to_mine(x, y) / 60.0)
        ang = math.atan2(y - CENTER, x - CENTER)
        v[17] = math.sin(ang)
        v[18] = math.cos(ang)
        v[19] = math.hypot(x - CENTER, y - CENTER) / 60.0
        v[24] = 1.0 if owner == player else 0.0
        v[25] = min(1.0, ships / 200.0)
        v[26] = prod / 5.0
        v[27] = step / TOTAL_STEPS

        entities[slot] = v
        mask[slot] = True
        planet_slot_ids[slot] = pid
        if owner == player:
            my_planet_slots.append(slot)
            my_planet_ids.append(pid)
        slot += 1

    # ---- Fleets ----
    for f in fleets:
        if slot >= max_entities - 2:
            break
        fid, owner, x, y, ang, _from, ships = f[0], f[1], float(f[2]), float(f[3]), float(f[4]), f[5], float(f[6])
        v = np.zeros(ENTITY_DIM, dtype=np.float32)
        v[TYPE_FLEET] = 1.0
        v[4:9] = _owner_onehot(owner)
        v[9] = x / BOARD
        v[10] = y / BOARD
        v[12] = _log_scale(ships, 100.0)
        v[20] = math.cos(ang)
        v[21] = math.sin(ang)
        # speed: 1 + 5 * (log(ships)/log(1000))^1.5
        sp = 1.0 + 5.0 * (math.log(max(1, int(ships))) / math.log(1000.0)) ** 1.5
        sp = min(sp, 6.0)
        v[22] = sp / 6.0
        v[24] = 1.0 if owner == player else 0.0
        v[25] = min(1.0, ships / 200.0)
        v[27] = step / TOTAL_STEPS

        entities[slot] = v
        mask[slot] = True
        fleet_slot_ids[slot] = fid
        slot += 1

    # ---- Sun token ----
    if slot < max_entities:
        v = np.zeros(ENTITY_DIM, dtype=np.float32)
        v[TYPE_SUN] = 1.0
        v[9] = CENTER / BOARD
        v[10] = CENTER / BOARD
        v[11] = SUN_R / 5.0
        entities[slot] = v
        mask[slot] = True
        slot += 1

    # ---- Global token ----
    if slot < max_entities:
        v = np.zeros(ENTITY_DIM, dtype=np.float32)
        v[TYPE_GLOBAL] = 1.0
        v[27] = step / TOTAL_STEPS
        entities[slot] = v
        mask[slot] = True
        slot += 1

    # ---- Global feature vector ----
    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    g[0] = step / TOTAL_STEPS
    g[1] = omega * 20.0  # 0.025-0.05 → 0.5-1.0
    # Per-player stats
    by_owner_ships = {}
    by_owner_prod = {}
    by_owner_count = {}
    for p in planets:
        if p[1] != -1:
            by_owner_ships[p[1]] = by_owner_ships.get(p[1], 0) + p[5]
            by_owner_prod[p[1]] = by_owner_prod.get(p[1], 0) + p[6]
            by_owner_count[p[1]] = by_owner_count.get(p[1], 0) + 1
    for f in fleets:
        by_owner_ships[f[1]] = by_owner_ships.get(f[1], 0) + f[6]

    my_ships = by_owner_ships.get(player, 0)
    my_prod = by_owner_prod.get(player, 0)
    my_count = by_owner_count.get(player, 0)
    enemy_ships = [s for o, s in by_owner_ships.items() if o != player]
    max_enemy = max(enemy_ships, default=0)
    g[2] = _log_scale(my_ships, 1000.0)
    g[3] = _log_scale(max_enemy, 1000.0)
    g[4] = my_prod / 50.0
    g[5] = max((p for o, p in by_owner_prod.items() if o != player), default=0) / 50.0
    g[6] = my_count / 20.0
    g[7] = max((c for o, c in by_owner_count.items() if o != player), default=0) / 20.0
    g[8] = len(planets) / 40.0
    g[9] = len(fleets) / 30.0
    g[10] = len(comet_ids) / 8.0
    g[11] = float(len(by_owner_ships) >= 4)  # is_4_player

    return EncodedObs(
        entities=entities,
        mask=mask,
        globals_=g,
        planet_slot_ids=planet_slot_ids,
        fleet_slot_ids=fleet_slot_ids,
        my_planet_slots=np.array(my_planet_slots, dtype=np.int32),
        my_planet_ids=np.array(my_planet_ids, dtype=np.int32),
        step=step,
    )
