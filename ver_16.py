%%writefile submission.py
import os
os.environ['KAGGLE_ENVELOPES'] = '0'

import math

SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
DECOY_THRESHOLD = 8


def fleet_speed(ships: int) -> float:
    if ships <= 0:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5


def travel_time(x1: float, y1: float, x2: float, y2: float, ships: int) -> float:
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist / fleet_speed(ships) if ships > 0 else 999.0


def line_seg_min_dist(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(x1 - px, y1 - py)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return math.hypot(x1 + t * dx - px, y1 + t * dy - py)


def path_crosses_sun(x1: float, y1: float, x2: float, y2: float, margin: float = 1.5) -> bool:
    return line_seg_min_dist(x1, y1, x2, y2, SUN_X, SUN_Y) < SUN_RADIUS + margin


def predict_orbit(x: float, y: float, omega: float, dt: float):
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r = math.hypot(x - SUN_X, y - SUN_Y)
    return SUN_X + r * math.cos(theta + omega * dt), SUN_Y + r * math.sin(theta + omega * dt)


def predict_comet_position(planet_id, comets, turns):
    for g in comets:
        pids = g.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = g.get("paths", [])
        path_index = g.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        future_idx = path_index + int(turns)
        if 0 <= future_idx < len(path):
            return path[future_idx][0], path[future_idx][1]
        return None
    return None


def comet_remaining_life(planet_id, comets):
    for g in comets:
        pids = g.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = g.get("paths", [])
        path_index = g.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0


def solve_intercept(fx: float, fy: float, tx: float, ty: float, target_id: int, orbiting: bool, omega: float, ships: int, comets, comet_ids, iterations: int = 25):
    if target_id in comet_ids:
        t = travel_time(fx, fy, tx, ty, ships)
        ix, iy = tx, ty
        valid = False
        for _ in range(iterations):
            pos = predict_comet_position(target_id, comets, t)
            if pos is None:
                break
            ix, iy = pos
            t2 = travel_time(fx, fy, ix, iy, ships)
            if abs(t2 - t) < 0.05:
                valid = True
                break
            t = t2
        return ix, iy, (t if valid else 999.0)

    if not orbiting:
        t = travel_time(fx, fy, tx, ty, ships)
        return tx, ty, t
    theta = math.atan2(ty - SUN_Y, tx - SUN_X)
    r = math.hypot(tx - SUN_X, ty - SUN_Y)
    t = travel_time(fx, fy, tx, ty, ships)
    ix, iy = tx, ty
    for _ in range(iterations):
        ix, iy = predict_orbit(tx, ty, omega, t)
        t2 = travel_time(fx, fy, ix, iy, ships)
        if abs(t2 - t) < 0.05:
            break
        t = t2
    return ix, iy, t


def safe_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    direct = math.atan2(y2 - y1, x2 - x1)
    if not path_crosses_sun(x1, y1, x2, y2, margin=1.5):
        return direct
    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)
    if d <= SUN_RADIUS + 1.0:
        return direct
    half = math.asin(min(1.0, (SUN_RADIUS + 1.0) / d))
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    cw = to_sun + half
    ccw = to_sun - half
    def adiff(a):
        dd = (a - direct) % (2 * math.pi)
        return min(dd, 2 * math.pi - dd)
    return cw if adiff(cw) < adiff(ccw) else ccw


def is_decoy_fleet(fleet, planets, omega):
    if fleet['ships'] < DECOY_THRESHOLD:
        return True
    tgt_id = None
    best_dist = float('inf')
    for p in planets.values():
        d = math.hypot(fleet['x'] - p['x'], fleet['y'] - p['y'])
        if d < best_dist:
            best_dist = d
            tgt_id = p['id']
    if tgt_id is None:
        return True
    tgt = planets.get(tgt_id)
    if tgt is None:
        return True
    r = math.hypot(tgt['x'] - SUN_X, tgt['y'] - SUN_Y)
    is_orb = (r + tgt['radius']) < 48.0
    ships_needed = tgt['ships'] + 1
    if fleet['ships'] < ships_needed * 0.4:
        return True
    return False


def ships_needed_for_takeover(tgt_ships, tgt_prod, tt, owner, margin=1.05):
    if owner == -1:
        return int(tgt_ships * margin) + 1
    growth = tgt_prod * tt
    return int((tgt_ships + growth) * margin) + 1


def planet_under_threat(p_id, fleets, planets, player, omega):
    incoming = 0
    for f in fleets.values():
        if f['owner'] == player:
            continue
        best_tgt, best_d = None, float('inf')
        for p in planets.values():
            if p['id'] == f['from']:
                continue
            d = math.hypot(f['x'] - p['x'], f['y'] - p['y'])
            if d < best_d:
                best_d = d
                best_tgt = p['id']
        if best_tgt == p_id:
            r = math.hypot(planets[p_id]['x'] - SUN_X, planets[p_id]['y'] - SUN_Y)
            is_orbiting = (r + planets[p_id]['radius']) < 48.0
            if is_orbiting:
                ix, iy = predict_orbit(planets[p_id]['x'], planets[p_id]['y'], omega, travel_time(f['x'], f['y'], planets[p_id]['x'], planets[p_id]['y'], int(f['ships'])))
                d = math.hypot(ix - planets[p_id]['x'], iy - planets[p_id]['y'])
            else:
                d = math.hypot(f['x'] - planets[p_id]['x'], f['y'] - planets[p_id]['y'])
            if d < 50:
                incoming += f['ships']
    return incoming


def compute_tangent_points(x1: float, y1: float, margin: float = 2.0):
    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)
    if d <= SUN_RADIUS + margin:
        return None, None
    half_angle = math.asin(min(1.0, (SUN_RADIUS + margin) / d))
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    return to_sun + half_angle, to_sun - half_angle


def multi_leg_path(x1: float, y1: float, x2: float, y2: float, margin: float = 2.0):
    if not path_crosses_sun(x1, y1, x2, y2, margin):
        return [(x2, y2)], math.hypot(x2 - x1, y2 - y1)
    
    beacon_ring = SUN_RADIUS + 15.0
    waypoints = []
    for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
        bx = SUN_X + beacon_ring * math.cos(angle)
        by = SUN_Y + beacon_ring * math.sin(angle)
        if not path_crosses_sun(x1, y1, bx, by, margin) and not path_crosses_sun(bx, by, x2, y2, margin):
            waypoints.append((bx, by))
    
    if not waypoints:
        return None, float('inf')
    
    best_wp = None
    best_dist = float('inf')
    for wx, wy in waypoints:
        d = math.hypot(wx - x1, wy - y1) + math.hypot(x2 - wx, y2 - wy)
        if d < best_dist:
            best_dist = d
            best_wp = (wx, wy)
    
    if best_wp:
        return [best_wp, (x2, y2)], best_dist
    
    return None, float('inf')


def estimate_capture_bonus(src_x: float, src_y: float, planet, omega: float, ships: int) -> float:
    r = math.hypot(planet['x'] - SUN_X, planet['y'] - SUN_Y)
    if (r + planet['radius']) >= 48.0:
        return 0.0
    
    if not path_crosses_sun(src_x, src_y, planet['x'], planet['y'], margin=2.0):
        return 3.0
    
    safe_count = 0
    for offset in range(-6, 7):
        fx, fy = predict_orbit(planet['x'], planet['y'], omega, offset)
        if not path_crosses_sun(src_x, src_y, fx, fy, margin=2.0):
            safe_count += 1
    
    return (safe_count / 13.0) * 5.0


def agent(obs):
    if isinstance(obs, dict):
        player = obs.get('player', 0)
        planets_data = obs.get('planets', [])
        fleets_data = obs.get('fleets', [])
        step = obs.get('step', 0)
        omega = obs.get('angular_velocity', 0.03)
        comets = obs.get('comets', [])
        comet_ids = set(obs.get('comet_planet_ids', []))
    else:
        player = getattr(obs, 'player', 0)
        planets_data = getattr(obs, 'planets', [])
        fleets_data = getattr(obs, 'fleets', [])
        step = getattr(obs, 'step', 0)
        omega = getattr(obs, 'angular_velocity', 0.03)
        comets = getattr(obs, 'comets', [])
        comet_ids = set(getattr(obs, 'comet_planet_ids', []))

    planets = {}
    for p in planets_data:
        pid, owner, x, y, radius, ships, prod = p[:7]
        r = math.hypot(x - SUN_X, y - SUN_Y)
        planets[pid] = {
            'id': pid, 'owner': owner, 'x': x, 'y': y,
            'radius': radius, 'ships': float(ships), 'prod': float(prod),
            'is_orb': (r + radius) < 48.0
        }

    fleets = {}
    for f in fleets_data:
        fleets[f[0]] = {
            'id': f[0], 'owner': f[1], 'x': f[2], 'y': f[3],
            'angle': f[4], 'from': f[5], 'ships': float(f[6])
        }

    my = [p for p in planets.values() if p['owner'] == player]
    if not my:
        return []

    enemy = [p for p in planets.values() if p['owner'] != player and p['owner'] != -1]
    neutrals = [p for p in planets.values() if p['owner'] == -1]

    my_prod = sum(p['prod'] for p in my)
    my_ships = sum(p['ships'] for p in my)
    enemy_prod = sum(p['prod'] for p in enemy) if enemy else 0
    enemy_ships = sum(p['ships'] for p in enemy) if enemy else 0

    prod_ratio = my_prod / enemy_prod if enemy_prod > 0 else 999
    ship_ratio = my_ships / enemy_ships if enemy_ships > 0 else 999

    my_planet_count = len(my)
    neighbor_count = sum(1 for t in neutrals if any(math.hypot(t['x'] - p['x'], t['y'] - p['y']) < 35 for p in my))

    nearby_larger_planets = []
    for src in my:
        for t in (neutrals + enemy):
            d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
            if d < 40 and t['prod'] >= src['prod'] * 0.8 and t['radius'] >= src['radius'] * 0.8:
                nearby_larger_planets.append((src['id'], t['id'], d))

    in_flight_from = set()
    in_flight_to = set()
    for f in fleets.values():
        if f['owner'] == player and f['from'] is not None:
            # CHỈ bỏ qua việc ghi nhận nếu hạm đội của ta có đúng 1 tàu (tàu nghi binh Ghost Fleet)
            if f['ships'] > 1:
                in_flight_from.add(f['from'])
                best_tgt, best_d = None, float('inf')
                for p in planets.values():
                    if p['id'] == f['from']:
                        continue
                    d = math.hypot(f['x'] - p['x'], f['y'] - p['y'])
                    if d < best_d:
                        best_d = d
                        best_tgt = p['id']
                if best_tgt:
                    in_flight_to.add(best_tgt)

    threats = {}
    for p in planets.values():
        if p['owner'] == player:
            threats[p['id']] = planet_under_threat(p['id'], fleets, planets, player, omega)

    smash_targets = set()
    for e in enemy:
        nearby_my_ships = sum(p['ships'] for p in my if math.hypot(p['x'] - e['x'], p['y'] - e['y']) < 50)
        if nearby_my_ships > e['ships'] * 0.95:
            smash_targets.add(e['id'])

    if smash_targets:
        phase = 'smash'
    elif my_ships > 120 and my_planet_count < 4 and enemy:
        phase = 'rush'
    elif my_planet_count < 3 or (neighbor_count > 0 and my_planet_count < 5):
        phase = 'expand'
    elif threats and any(t > my_ships * 0.25 for t in threats.values()):
        phase = 'counter_attack'
    elif prod_ratio > 4 and my_ships > 80 and my_planet_count >= 3:
        phase = 'crush'
    elif prod_ratio > 2.0 or ship_ratio > 2.5:
        phase = 'aggressive'
    elif my_prod < enemy_prod * 0.7:
        phase = 'defend'
    elif len(enemy) > 0 and len(my) >= 3 and my_prod > enemy_prod * 1.0:
        phase = 'dominate'
    else:
        phase = 'grow'

    moves = []
    targeted_this_turn = set()

    for src in my:
        if src['id'] in in_flight_from:
            continue

        if src['ships'] < 10:
            continue

        if phase == 'expand':
            nearby_larger = {nl[1] for nl in nearby_larger_planets if nl[0] == src['id']}
            best_target = None
            best_score = -1e9
            for t in neutrals:
                if t['id'] == src['id']:
                    continue
                if t['id'] in in_flight_to or t['id'] in targeted_this_turn:
                    continue
                d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
                
                # Nâng cấp thuật toán chấm điểm mở rộng: Ưu tiên tuyệt đối các hành tinh sản lượng cao ở xa
                score = -d * 2.0 + t['prod'] * 6.5
                
                if nearby_larger and t['radius'] < src['radius'] * 0.7 and d > 25:
                    score -= 50
                if score > best_score:
                    best_score = score
                    best_target = t
            if best_target:
                r = math.hypot(best_target['x'] - SUN_X, best_target['y'] - SUN_Y)
                is_orbiting = (r + best_target['radius']) < 48.0
                ix, iy, tt = solve_intercept(src['x'], src['y'], best_target['x'], best_target['y'], best_target['id'], is_orbiting, omega, int(src['ships']), comets, comet_ids)
                
                is_comet_and_leaving = False
                if best_target['id'] in comet_ids:
                    life = comet_remaining_life(best_target['id'], comets)
                    if tt >= life:
                        is_comet_and_leaving = True
                
                if not is_comet_and_leaving and not path_crosses_sun(src['x'], src['y'], ix, iy, margin=1.5):
                    send = ships_needed_for_takeover(best_target['ships'], best_target['prod'], tt, best_target['owner'])
                    if src['ships'] >= send:
                        angle = safe_angle(src['x'], src['y'], ix, iy)
                        moves.append([src['id'], angle, send])
                        targeted_this_turn.add(best_target['id'])
                        src['ships'] -= send
                        if src['ships'] < 5:
                            break
            elif src['ships'] > 40:
                decoy_tgt = None
                decoy_score = -1e9
                for t in (enemy + neutrals):
                    if t['id'] == src['id']:
                        continue
                    if t['id'] in targeted_this_turn:
                        continue
                    d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
                    
                    # Nâng cấp chấm điểm quấy rối/decoy
                    score = -d * 1.5 + (t['prod'] if t['owner'] != -1 else 0) * 6.0
                    
                    if nearby_larger and t['radius'] < src['radius'] * 0.7 and d > 25:
                        score -= 50
                    if score > decoy_score:
                        decoy_score = score
                        decoy_tgt = t
                if decoy_tgt and src['ships'] > 25:
                    send = min(8, int(src['ships'] * 0.15))
                    if send >= 5:
                        r = math.hypot(decoy_tgt['x'] - SUN_X, decoy_tgt['y'] - SUN_Y)
                        is_orbiting = (r + decoy_tgt['radius']) < 48.0
                        ix, iy, tt = solve_intercept(src['x'], src['y'], decoy_tgt['x'], decoy_tgt['y'], decoy_tgt['id'], is_orbiting, omega, int(src['ships']), comets, comet_ids)
                        
                        is_comet_and_leaving = False
                        if decoy_tgt['id'] in comet_ids:
                            life = comet_remaining_life(decoy_tgt['id'], comets)
                            if tt >= life:
                                is_comet_and_leaving = True
                        
                        if not is_comet_and_leaving and not path_crosses_sun(src['x'], src['y'], ix, iy, margin=1.5):
                            angle = safe_angle(src['x'], src['y'], ix, iy)
                            moves.append([src['id'], angle, send])
                            targeted_this_turn.add(decoy_tgt['id'])
                            src['ships'] -= send
                            if src['ships'] < 10:
                                break

        need_defense = threats.get(src['id'], 0) > src['ships'] * 0.3

        if need_defense and phase != 'counter_attack':
            continue

        if need_defense and phase == 'counter_attack' and threats.get(src['id'], 0) >= src['ships'] * 0.5:
            continue

        if phase == 'counter_attack':
            best_enemy = None
            best_score = -1e9
            for t in enemy:
                if t['id'] in targeted_this_turn:
                    continue
                d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
                score = t['ships'] * 0.8 + t['prod'] * 8 - d
                if t['id'] in smash_targets:
                    score += 50
                if score > best_score:
                    best_score = score
                    best_enemy = t
            if best_enemy:
                r = math.hypot(best_enemy['x'] - SUN_X, best_enemy['y'] - SUN_Y)
                is_orbiting = (r + best_enemy['radius']) < 48.0
                ix, iy, tt = solve_intercept(src['x'], src['y'], best_enemy['x'], best_enemy['y'], best_enemy['id'], is_orbiting, omega, int(src['ships']), comets, comet_ids)
                
                is_comet_and_leaving = False
                if best_enemy['id'] in comet_ids:
                    life = comet_remaining_life(best_enemy['id'], comets)
                    if tt >= life:
                        is_comet_and_leaving = True
                
                if not is_comet_and_leaving and not path_crosses_sun(src['x'], src['y'], ix, iy, margin=1.5):
                    send = int(src['ships'] * 0.8)
                    send = max(send, ships_needed_for_takeover(best_enemy['ships'], best_enemy['prod'], tt, best_enemy['owner']))
                    send = min(send, int(src['ships'] * 0.95))
                    if src['ships'] > send + 3:
                        angle = safe_angle(src['x'], src['y'], ix, iy)
                        moves.append([src['id'], angle, send])
                        targeted_this_turn.add(best_enemy['id'])
                        src['ships'] -= send

        best_tgt = None
        best_score = -1e9

        if phase == 'smash':
            candidates = [t for t in enemy if t['id'] in smash_targets]
        elif phase == 'rush':
            candidates = enemy
        elif phase in ('expand', 'opportunistic', 'aggressive', 'dominate'):
            candidates = neutrals if phase not in ('aggressive', 'dominate') else (enemy + neutrals)
        elif phase == 'grow':
            candidates = [t for t in neutrals if threats.get(t['id'], 0) == 0]
        else:
            candidates = []

        for t in candidates:
            if t['id'] == src['id']:
                continue
            if t['id'] in in_flight_to:
                continue
            if t['id'] in targeted_this_turn:
                continue

            incoming = threats.get(t['id'], 0)
            if incoming > 0:
                continue

            r = math.hypot(t['x'] - SUN_X, t['y'] - SUN_Y)
            is_orbiting = t['is_orb']

            ix, iy, tt = solve_intercept(src['x'], src['y'], t['x'], t['y'], t['id'], is_orbiting, omega, int(src['ships']), comets, comet_ids)

            if t['id'] in comet_ids:
                life = comet_remaining_life(t['id'], comets)
                if tt >= life:
                    continue

            if path_crosses_sun(src['x'], src['y'], ix, iy, margin=1.5):
                waypoints, _ = multi_leg_path(src['x'], src['y'], ix, iy)
                if waypoints is None:
                    continue
                final_x, final_y = waypoints[-1]
                if path_crosses_sun(src['x'], src['y'], final_x, final_y, margin=1.5):
                    continue

            if is_orbiting:
                planet_future = predict_orbit(t['x'], t['y'], omega, tt)
                to_planet = math.atan2(planet_future[1] - src['y'], planet_future[0] - src['x'])
                to_target = math.atan2(t['y'] - src['y'], t['x'] - src['x'])
                diff = abs((to_planet - to_target) % (2 * math.pi))
                if diff > 0.5 and diff < (2 * math.pi - 0.5):
                    continue
            elif t['id'] in comet_ids:
                planet_future = predict_comet_position(t['id'], comets, tt)
                if planet_future is None:
                    continue
                to_planet = math.atan2(planet_future[1] - src['y'], planet_future[0] - src['x'])
                to_target = math.atan2(t['y'] - src['y'], t['x'] - src['x'])
                diff = abs((to_planet - to_target) % (2 * math.pi))
                if diff > 0.5 and diff < (2 * math.pi - 0.5):
                    continue

            # Nâng cao điểm sản lượng kinh tế trong tính toán mục tiêu chính
            score = t['prod'] * 21 - tt * 2.3

            if t['owner'] == -1:
                score += 25

            if phase == 'aggressive' and t['owner'] != -1:
                score += 35 - t['ships'] * 0.12

            if phase == 'dominate' and t['owner'] != -1:
                score += 45 - t['ships'] * 0.08

            if phase == 'dominate' and t['owner'] == -1:
                score += 20

            if is_orbiting:
                score -= 6

            if src['ships'] > 50 and t['owner'] == -1:
                score += 12

            if src['prod'] > t['prod'] * 0.7:
                score += 8

            score += estimate_capture_bonus(src['x'], src['y'], t, omega, int(src['ships']))

            if t['id'] in comet_ids:
                score += 10

            if score > best_score:
                best_score = score
                best_tgt = (t, ix, iy, tt)

        if best_tgt is None:
            continue

        tgt, ix, iy, tt = best_tgt

        if phase == 'smash':
            send = int(src['ships'] * 0.9)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
        elif phase == 'rush':
            send = int(src['ships'] * 0.8)
        elif phase == 'aggressive':
            send = int(src['ships'] * 0.4)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
            send = min(send, int(src['ships'] * 0.7))
        elif phase == 'dominate':
            send = int(src['ships'] * 0.5)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
            send = min(send, int(src['ships'] * 0.8))
        elif phase == 'opportunistic':
            send = ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner'])
            send = min(send, int(src['ships'] * 0.5))
        else:
            send = ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner'])

        if src['ships'] < send:
            continue

        angle = safe_angle(src['x'], src['y'], ix, iy)
        moves.append([src['id'], angle, send])
        targeted_this_turn.add(tgt['id'])
        src['ships'] -= send

    if phase == 'expand':
        for src in my:
            if src['id'] in in_flight_from:
                continue
            if src['ships'] < 10:
                continue
            nearby_larger = [nl for nl in nearby_larger_planets if nl[0] == src['id']]
            if not nearby_larger:
                continue
            candidates = [t for t in (neutrals + enemy)
                          if t['id'] not in targeted_this_turn
                          and t['id'] not in in_flight_to
                          and t['owner'] != player]
            if not candidates:
                continue
            best_tgt = None
            best_score = -1e9
            for t in candidates:
                d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
                if d > 40:
                    continue
                score = t['prod'] * 5 - d
                if t['radius'] >= src['radius'] * 0.8 and t['prod'] >= src['prod'] * 0.8:
                    score += 40
                if score > best_score:
                    best_score = score
                    best_tgt = t
            if best_tgt:
                r = math.hypot(best_tgt['x'] - SUN_X, best_tgt['y'] - SUN_Y)
                is_orbiting = (r + best_tgt['radius']) < 48.0
                ix, iy, tt = solve_intercept(src['x'], src['y'], best_tgt['x'], best_tgt['y'], best_tgt['id'], is_orbiting, omega, int(src['ships']), comets, comet_ids)
                
                is_comet_and_leaving = False
                if best_tgt['id'] in comet_ids:
                    life = comet_remaining_life(best_tgt['id'], comets)
                    if tt >= life:
                        is_comet_and_leaving = True
                
                if not is_comet_and_leaving and not path_crosses_sun(src['x'], src['y'], ix, iy, margin=1.5):
                    send = ships_needed_for_takeover(best_tgt['ships'], best_tgt['prod'], tt, best_tgt['owner'])
                    if src['ships'] >= send:
                        angle = safe_angle(src['x'], src['y'], ix, iy)
                        moves.append([src['id'], angle, send])
                        targeted_this_turn.add(best_tgt['id'])
                        src['ships'] -= send

    # Spoofing Protocol (Ghost Fleets) - Nghi binh quấy nhiễu đối phương
    if step % 18 == 0 and enemy and len(my) >= 1:
        enemy_strengths = {}
        for e in enemy:
            enemy_strengths[e['owner']] = enemy_strengths.get(e['owner'], 0) + e['ships']
        if enemy_strengths:
            strongest_owner = max(enemy_strengths, key=enemy_strengths.get)
            strongest_planets = [p for p in enemy if p['owner'] == strongest_owner]
            if strongest_planets:
                best_src = max(my, key=lambda p: p['ships'])
                if best_src['ships'] > 30:
                    target = min(strongest_planets, key=lambda p: math.hypot(best_src['x'] - p['x'], best_src['y'] - p['y']))
                    r = math.hypot(target['x'] - SUN_X, target['y'] - SUN_Y)
                    is_orbiting = (r + target['radius']) < 48.0
                    ix, iy, tt = solve_intercept(best_src['x'], best_src['y'], target['x'], target['y'], target['id'], is_orbiting, omega, 1, comets, comet_ids)
                    if not path_crosses_sun(best_src['x'], best_src['y'], ix, iy, margin=1.5):
                        angle = safe_angle(best_src['x'], best_src['y'], ix, iy)
                        moves.append([best_src['id'], angle, 1])
                        best_src['ships'] -= 1

    return moves