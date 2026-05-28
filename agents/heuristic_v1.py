"""Heuristic v1 — strong handcrafted Orbit Wars agent.

Built from scratch based on direct inspection of the orbit_wars engine
(NOT based on the START.ipynb code, which scores poorly).

Pipeline per turn:
  1. Parse obs into typed structures.
  2. Predict positions of orbiting planets at the target turn for each
     candidate src->dst pair (fixed-point iteration on ETA).
  3. Build arrival ledger: which fleets are heading where and when.
  4. Simulate timeline per planet to find planets we'll lose and planets
     we can take cheaply with current commitments.
  5. Defense pass: reinforce threatened planets from nearest safe ally.
  6. Expansion pass: score (target, src) candidates by ROI; greedily
     commit until budget or sources exhausted.
  7. Dedupe moves and clamp to garrison.

Key design points:
  - Angle from src to predicted intercept (not the planet's *current*
    position), with sun avoidance via tangent routing.
  - Combat simulation matches engine rules exactly: per-owner sum, top vs
    second difference, then survivor vs garrison (flip with surplus).
  - Per-source launch caps and a global attack budget prevent overspread.
  - Phase-aware (early/mid/late) targeting and reserves.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


# ---- Constants (must match orbit_wars engine) -------------------------------
BOARD = 100.0
CENTER = 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
ROTATION_RADIUS_LIMIT = 50.0
TOTAL_STEPS = 500
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)


# ---- Field-index aliases (planets/fleets are lists, not named tuples here) --
P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)


# ---- Geometry ---------------------------------------------------------------

def fleet_speed(ships: int) -> float:
    ships = max(1, int(ships))
    if ships <= 1:
        return 1.0
    r = math.log(ships) / math.log(1000.0)
    r = max(0.0, min(1.0, r))
    return 1.0 + (MAX_SPEED - 1.0) * (r ** 1.5)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def seg_point_dist(p, a, b):
    """Distance from point p to segment a-b."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    ls = dx * dx + dy * dy
    if ls < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / ls
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def sun_blocked(x1, y1, x2, y2, margin: float = 1.5) -> bool:
    return seg_point_dist((CENTER, CENTER), (x1, y1), (x2, y2)) < SUN_R + margin


def normalize_angle(a: float) -> float:
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def safe_angle(x1, y1, x2, y2):
    """Direct angle if sun isn't blocking, else tangent angle around the sun."""
    direct = math.atan2(y2 - y1, x2 - x1)
    if not sun_blocked(x1, y1, x2, y2):
        return direct

    d = math.hypot(x1 - CENTER, y1 - CENTER)
    if d <= SUN_R + 1.0:
        return direct  # We're inside or on the sun — fallback

    half = math.asin(min(1.0, (SUN_R + 2.0) / d))
    to_sun = math.atan2(CENTER - y1, CENTER - x1)
    cw = to_sun + half
    ccw = to_sun - half
    if abs(normalize_angle(cw - direct)) < abs(normalize_angle(ccw - direct)):
        return cw
    return ccw


def is_orbiting(p, initial_planets_by_id):
    """A planet rotates iff its INITIAL orbital_radius + radius < limit.
    Current position is unreliable (it might already be rotated).
    """
    init = initial_planets_by_id.get(p[P_ID])
    src = init if init is not None else p
    r = math.hypot(src[P_X] - CENTER, src[P_Y] - CENTER)
    return r + src[P_R] < ROTATION_RADIUS_LIMIT


def predict_position(planet, omega: float, dt: float, initial_planets_by_id) -> tuple[float, float]:
    """Predict (x, y) of `planet` `dt` turns into the future."""
    if not is_orbiting(planet, initial_planets_by_id):
        return planet[P_X], planet[P_Y]
    px, py = planet[P_X], planet[P_Y]
    r = math.hypot(px - CENTER, py - CENTER)
    th = math.atan2(py - CENTER, px - CENTER)
    return CENTER + r * math.cos(th + omega * dt), CENTER + r * math.sin(th + omega * dt)


# ---- Comet position prediction ---------------------------------------------

def predict_comet_position(comet_planet_id, comets_groups, dt_turns) -> tuple[float, float] | None:
    """Walk forward `dt_turns` along the comet's known path. Returns None if
    the comet would have left the board by then.
    """
    for group in comets_groups:
        if comet_planet_id not in group["planet_ids"]:
            continue
        i = group["planet_ids"].index(comet_planet_id)
        path = group["paths"][i]
        future_idx = group["path_index"] + int(dt_turns)
        if 0 <= future_idx < len(path):
            return float(path[future_idx][0]), float(path[future_idx][1])
        return None
    return None


# ---- Intercept solver -------------------------------------------------------

def solve_intercept(src, tgt, ships, omega, initial_planets_by_id,
                    comet_ids, comets_groups, max_iters: int = 10):
    """Find (intercept_x, intercept_y, eta_turns) for fleet of `ships` from
    `src` chasing `tgt`. Stable fixed-point iteration.
    """
    ships = max(1, int(ships))
    sp = fleet_speed(ships)
    sx, sy = src[P_X], src[P_Y]
    tx, ty = tgt[P_X], tgt[P_Y]

    # Initial guess: time to reach tgt's CURRENT position
    d = max(0.0, math.hypot(tx - sx, ty - sy) - src[P_R] - tgt[P_R])
    t = d / sp

    is_comet = tgt[P_ID] in comet_ids
    orbiting = is_orbiting(tgt, initial_planets_by_id)

    if not is_comet and not orbiting:
        return tx, ty, t

    ix, iy = tx, ty
    for _ in range(max_iters):
        if is_comet:
            future = predict_comet_position(tgt[P_ID], comets_groups, t)
            if future is None:
                return tx, ty, float("inf")  # comet gone by then
            ix, iy = future
        else:
            ix, iy = predict_position(tgt, omega, t, initial_planets_by_id)
        d_new = max(0.0, math.hypot(ix - sx, iy - sy) - src[P_R] - tgt[P_R])
        t_new = d_new / sp
        if abs(t_new - t) < 0.05:
            t = t_new
            break
        t = 0.5 * t + 0.5 * t_new
    return ix, iy, t


# ---- Fleet target inference (for arrival ledger) ----------------------------

def fleet_target_estimate(fleet, planets, horizon: int = 80):
    """Estimate which planet a fleet will hit and when, by raycast.
    Returns (planet_id, eta) or (None, None).
    """
    fx, fy, fa, fs = fleet[F_X], fleet[F_Y], fleet[F_ANGLE], fleet[F_SHIPS]
    cos_a, sin_a = math.cos(fa), math.sin(fa)
    sp = fleet_speed(fs)
    best_id, best_t = None, 1e18
    for p in planets:
        dx, dy = p[P_X] - fx, p[P_Y] - fy
        proj = dx * cos_a + dy * sin_a
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        r2 = p[P_R] * p[P_R]
        if perp_sq >= r2:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, r2 - perp_sq)))
        t = hit_d / sp
        if t <= horizon and t < best_t:
            best_id, best_t = p[P_ID], t
    if best_id is None:
        return None, None
    return best_id, int(math.ceil(best_t))


# ---- Combat simulation (matches engine rules) -------------------------------

def _resolve_arrivals(owner, garrison, arrivals):
    """Resolve all arrivals at a single planet on a single turn."""
    by_owner = defaultdict(int)
    for _, o, s in arrivals:
        by_owner[o] += int(s)

    # Same-owner reinforcements just add to garrison (engine sums before combat)
    if owner in by_owner:
        garrison += by_owner.pop(owner)
    if not by_owner:
        return owner, garrison

    # Top vs second; difference survives unless tied (then 0)
    items = sorted(by_owner.items(), key=lambda x: -x[1])
    top_owner, top_s = items[0]
    second_s = items[1][1] if len(items) > 1 else 0
    surv = top_s - second_s
    if surv <= 0:
        return owner, garrison

    if owner == top_owner:
        # Shouldn't happen since we popped same-owner above, but be defensive.
        return owner, garrison + surv
    # Different owner: subtract from garrison; flip if it goes negative.
    new_g = garrison - surv
    if new_g < 0:
        return top_owner, -new_g
    return owner, new_g


def simulate_timeline(planet, arrivals, horizon):
    """Walk forward `horizon` turns from `planet`'s current state. Returns
    (owner_by_turn, ships_by_turn) where index 0 = now, index `horizon` = end.
    """
    horizon = max(1, int(horizon))
    by_turn = defaultdict(list)
    for eta, o, s in arrivals:
        t = max(1, int(math.ceil(eta)))
        if t <= horizon and s > 0:
            by_turn[t].append((t, o, int(s)))

    owner = planet[P_OWNER]
    g = float(planet[P_SHIPS])
    owners = [owner]
    ships = [g]
    for turn in range(1, horizon + 1):
        if owner != -1:
            g += planet[P_PROD]
        if turn in by_turn:
            owner, g = _resolve_arrivals(owner, g, by_turn[turn])
            g = max(0.0, g)
        owners.append(owner)
        ships.append(g)
    return owners, ships


def ships_needed_to_take(planet, eta, player, base_arrivals, planned_extra):
    """How many ships do WE need to land at turn `eta` for the planet to be
    ours at end of that turn? Exact via binary search on top of timeline sim.
    """
    eta = max(1, int(math.ceil(eta)))
    all_base = [a for a in (list(base_arrivals) + list(planned_extra))
                if int(math.ceil(a[0])) <= eta and a[2] > 0]

    owners, _ = simulate_timeline(planet, all_base, eta)
    if owners[eta] == player:
        return 0

    def owns_with(extra: int) -> bool:
        test = all_base + [(eta, player, int(extra))]
        owners, _ = simulate_timeline(planet, test, eta)
        return owners[eta] == player

    # Upper bound on what could ever be needed: starting ships + production
    # over the horizon + a safety buffer.
    hi = max(1, int(planet[P_SHIPS]) + int(planet[P_PROD]) * eta + 4)
    cap = 5000
    while hi <= cap and not owns_with(hi):
        hi *= 2
    if hi > cap:
        return cap + 1

    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if owns_with(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---- Phase classification ---------------------------------------------------

def game_phase(step: int, my_count: int, enemy_count: int,
               my_prod: float, max_enemy_prod: float, n_players: int) -> str:
    """Phase classification.

    `max_enemy_prod` is the production of the STRONGEST single enemy
    (not the sum). In 4-player FFA we measure ourselves against the leader,
    not the field — winning means out-producing #2.
    """
    progress = step / float(TOTAL_STEPS)
    if step < 30 or my_count <= 2:
        return "early"
    if progress > 0.80 or (my_count >= 5 and my_prod >= max_enemy_prod * 1.15):
        return "late"
    if enemy_count == 0:
        return "cleanup"
    if my_prod < max_enemy_prod * 0.85 or my_count < enemy_count // max(1, n_players - 1):
        return "defend"
    return "mid"


# ---- Main agent -------------------------------------------------------------

def agent(obs):
    # ---- Parse obs ----------------------------------------------------------
    if isinstance(obs, dict):
        step = int(obs.get("step", 0) or 0)
        player = int(obs.get("player", 0) or 0)
        omega = float(obs.get("angular_velocity", 0.03) or 0.03)
        planets = [list(p) for p in (obs.get("planets") or [])]
        fleets = [list(f) for f in (obs.get("fleets") or [])]
        initial_planets = [list(p) for p in (obs.get("initial_planets") or [])]
        comets = obs.get("comets") or []
        comet_ids = set(obs.get("comet_planet_ids") or [])
    else:
        step = int(getattr(obs, "step", 0) or 0)
        player = int(getattr(obs, "player", 0) or 0)
        omega = float(getattr(obs, "angular_velocity", 0.03) or 0.03)
        planets = [list(p) for p in (getattr(obs, "planets", None) or [])]
        fleets = [list(f) for f in (getattr(obs, "fleets", None) or [])]
        initial_planets = [list(p) for p in (getattr(obs, "initial_planets", None) or [])]
        comets = getattr(obs, "comets", None) or []
        comet_ids = set(getattr(obs, "comet_planet_ids", None) or [])

    if not planets:
        return []

    initial_by_id = {p[P_ID]: p for p in initial_planets}

    my_planets = [p for p in planets if p[P_OWNER] == player]
    if not my_planets:
        return []

    enemy_planets = [p for p in planets if p[P_OWNER] != player and p[P_OWNER] != -1]
    neutral_planets = [p for p in planets if p[P_OWNER] == -1]
    target_pool = enemy_planets + neutral_planets

    # Per-enemy stats (important in 4-player to identify the leader)
    enemy_prod_by_id: dict[int, float] = defaultdict(float)
    enemy_planet_count: dict[int, int] = defaultdict(int)
    enemy_ships_by_id: dict[int, float] = defaultdict(float)
    for p in enemy_planets:
        enemy_prod_by_id[p[P_OWNER]] += p[P_PROD]
        enemy_planet_count[p[P_OWNER]] += 1
        enemy_ships_by_id[p[P_OWNER]] += p[P_SHIPS]
    for f in fleets:
        if f[F_OWNER] != player and f[F_OWNER] != -1:
            enemy_ships_by_id[f[F_OWNER]] += f[F_SHIPS]

    my_prod = sum(p[P_PROD] for p in my_planets)
    max_enemy_prod = max(enemy_prod_by_id.values()) if enemy_prod_by_id else 0.0
    strongest_enemy = max(enemy_ships_by_id, key=enemy_ships_by_id.get) if enemy_ships_by_id else None

    # Detect 2p vs 4p (based on how many distinct owners we've seen)
    seen_owners = {p[P_OWNER] for p in planets if p[P_OWNER] != -1}
    seen_owners.update(f[F_OWNER] for f in fleets if f[F_OWNER] != -1)
    n_players = max(2, len(seen_owners))

    phase = game_phase(step, len(my_planets), len(enemy_planets),
                       my_prod, max_enemy_prod, n_players)
    remaining = max(1, TOTAL_STEPS - step)

    # ---- Build arrival ledger ----------------------------------------------
    ledger: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for f in fleets:
        pid, eta = fleet_target_estimate(f, planets)
        if pid is None:
            continue
        ledger[pid].append((eta, f[F_OWNER], int(f[F_SHIPS])))

    # ---- Defense pass: reinforce planets about to flip ---------------------
    moves: list[list] = []
    spent: dict[int, int] = defaultdict(int)
    planned: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    # 4p needs faster expansion (snipers grab everything otherwise);
    # smaller reserves let us push harder. Scale down for multi-player.
    reserve_scale = 0.7 if n_players > 2 else 1.0

    def reserve_for(p) -> int:
        if phase == "early":
            base = 6 + 1.5 * p[P_PROD]
        elif phase == "late":
            base = 3 + 0.8 * p[P_PROD]
        else:
            base = 5 + 1.2 * p[P_PROD]
        return int(base * reserve_scale)

    DEF_HORIZON = 30
    threatened: list[tuple[int, list]] = []
    for mine in my_planets:
        owners, _ = simulate_timeline(mine, ledger[mine[P_ID]], DEF_HORIZON)
        for t in range(1, DEF_HORIZON + 1):
            if owners[t] != player:
                threatened.append((t, mine))
                break

    threatened.sort(key=lambda x: x[0])
    for fall_t, mine in threatened:
        need = ships_needed_to_take(
            mine, fall_t, player,
            base_arrivals=ledger[mine[P_ID]],
            planned_extra=planned[mine[P_ID]],
        )
        if need <= 0:
            continue
        need = int(need + mine[P_PROD] + 2)  # small buffer

        donors = sorted(
            (a for a in my_planets if a[P_ID] != mine[P_ID]),
            key=lambda a: math.hypot(a[P_X] - mine[P_X], a[P_Y] - mine[P_Y]),
        )
        for ally in donors:
            avail = ally[P_SHIPS] - spent[ally[P_ID]] - reserve_for(ally)
            if avail < 5:
                continue
            ix, iy, eta = solve_intercept(ally, mine, max(5, avail), omega,
                                          initial_by_id, comet_ids, comets)
            if eta >= fall_t - 0.5 or eta >= 999:
                continue
            if sun_blocked(ally[P_X], ally[P_Y], ix, iy):
                continue
            send = min(int(avail), need)
            if send < 5:
                continue
            ang = safe_angle(ally[P_X], ally[P_Y], ix, iy)
            moves.append([ally[P_ID], float(ang), int(send)])
            spent[ally[P_ID]] += send
            eta_int = max(1, int(math.ceil(eta)))
            planned[mine[P_ID]].append((eta_int, player, send))
            need -= send
            if need <= 0:
                break

    # ---- Attack pass: score (src, tgt) candidates --------------------------
    ATT_HORIZON = 65

    def src_available(src) -> int:
        return max(0, int(src[P_SHIPS]) - spent[src[P_ID]] - reserve_for(src))

    candidates: list[tuple[float, list, list, float, float, float, int]] = []
    for src in my_planets:
        if src_available(src) < 5:
            continue
        for tgt in target_pool:
            if tgt[P_ID] == src[P_ID]:
                continue
            ships_guess = max(15, min(40, src_available(src)))
            ix, iy, eta_f = solve_intercept(src, tgt, ships_guess, omega,
                                            initial_by_id, comet_ids, comets)
            if eta_f >= ATT_HORIZON or eta_f >= 999:
                continue
            if sun_blocked(src[P_X], src[P_Y], ix, iy):
                continue
            eta = max(1, int(math.ceil(eta_f)))
            need = ships_needed_to_take(
                tgt, eta, player,
                base_arrivals=ledger[tgt[P_ID]],
                planned_extra=planned[tgt[P_ID]],
            )
            if need <= 0:
                continue
            if need > src_available(src):
                # Skip if we can't afford it (no single-source partial commits yet)
                continue

            turns_left = max(1, remaining - eta)
            value = tgt[P_PROD] * float(turns_left)
            if tgt[P_OWNER] == -1:
                value *= 1.0
            else:
                # Base enemy multiplier; bigger boost for strongest enemy
                # (denies the leader; matters in 4-player FFA).
                base = 1.5
                if tgt[P_OWNER] == strongest_enemy and n_players > 2:
                    base = 2.1
                value *= base
            if tgt[P_ID] in comet_ids:
                # Comets are temporary; discount their long-tail value
                value *= 0.6
            cost = max(1.0, need + 1.5 * eta)
            score = value / cost

            # Slight bonus for being near our cluster (consolidation)
            d0 = math.hypot(src[P_X] - tgt[P_X], src[P_Y] - tgt[P_Y])
            score *= 1.0 + max(0.0, 0.2 * (30.0 - d0) / 30.0)

            candidates.append((score, src, tgt, ix, iy, eta_f, need))

    candidates.sort(key=lambda c: -c[0])

    # ---- Greedy commit with budget caps ------------------------------------
    if phase == "early":
        per_src_cap = 3 if n_players > 2 else 2
    elif phase in ("mid", "defend"):
        per_src_cap = 4 if n_players > 2 else 3
    else:
        per_src_cap = 5 if n_players > 2 else 4

    src_launches: dict[int, int] = defaultdict(int)
    total_my_ships = sum(p[P_SHIPS] for p in my_planets)
    if phase == "early":
        attack_budget = int(0.45 * total_my_ships)
    elif phase in ("mid", "defend"):
        attack_budget = int(0.60 * total_my_ships)
    else:
        attack_budget = int(0.85 * total_my_ships)
    # In 4-player FFA, only #1 wins — be more aggressive overall.
    if n_players > 2:
        attack_budget = int(attack_budget * 1.25)
    total_attacked = 0

    for score, src, tgt, ix, iy, eta_f, need in candidates:
        if total_attacked >= attack_budget:
            break
        if src_launches[src[P_ID]] >= per_src_cap:
            continue
        # Re-check need (other commits may have changed it)
        eta = max(1, int(math.ceil(eta_f)))
        latest_need = ships_needed_to_take(
            tgt, eta, player,
            base_arrivals=ledger[tgt[P_ID]],
            planned_extra=planned[tgt[P_ID]],
        )
        if latest_need <= 0:
            continue
        avail = src_available(src)
        if avail < latest_need:
            continue
        # Send exact need + small overcommit cushion in late game
        send = latest_need
        if phase == "late":
            send = min(avail, latest_need + max(2, latest_need // 4))

        ang = safe_angle(src[P_X], src[P_Y], ix, iy)
        moves.append([src[P_ID], float(ang), int(send)])
        spent[src[P_ID]] += send
        src_launches[src[P_ID]] += 1
        total_attacked += send
        planned[tgt[P_ID]].append((eta, player, send))

    # ---- Final sanity: clamp by current garrison, dedup same-angle launches
    by_src_ang: dict[tuple[int, float], list] = {}
    for sid, a, s in moves:
        key = (sid, round(float(a), 3))
        if key in by_src_ang:
            by_src_ang[key][2] += int(s)
        else:
            by_src_ang[key] = [sid, float(a), int(s)]
    out = []
    used_per_src: dict[int, int] = defaultdict(int)
    src_by_id = {p[P_ID]: p for p in my_planets}
    for sid, a, s in by_src_ang.values():
        src = src_by_id.get(sid)
        if src is None:
            continue
        cap = max(0, int(src[P_SHIPS]) - used_per_src[sid])
        send = min(int(s), cap)
        if send >= 1:
            out.append([sid, float(a), int(send)])
            used_per_src[sid] += send
    return out
