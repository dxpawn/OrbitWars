# main.py
# SIMPLE ORBIT WARS BOT

import math


def distance(a, b):
    return math.hypot(a[2] - b[2], a[3] - b[3])


def predict_orbit_position(planet, angular_velocity, steps_ahead):
    """Dự đoán vị trí orbit planet sau steps_ahead lượt"""
    if planet[4] + 10 >= 50:  # static planet
        return planet[2], planet[3]
    
    x, y = planet[2], planet[3]
    center_x, center_y = 50, 50
    
    current_angle = math.atan2(y - center_y, x - center_x)
    new_angle = current_angle + angular_velocity * steps_ahead
    
    radius = math.hypot(x - center_x, y - center_y)
    new_x = center_x + radius * math.cos(new_angle)
    new_y = center_y + radius * math.sin(new_angle)
    
    return new_x, new_y


def will_cross_sun(x1, y1, x2, y2, sun_radius=10):
    """Kiểm tra đường đi có qua sun không"""
    center_x, center_y = 50, 50
    numerator = abs((y2 - y1) * center_x - (x2 - x1) * center_y + x2 * y1 - y2 * x1)
    denominator = math.hypot(y2 - y1, x2 - x1)
    
    if denominator == 0:
        return False
    
    dist_to_sun = numerator / denominator
    return dist_to_sun < sun_radius


def agent(obs, config):

    planets = obs["planets"]
    fleets = obs.get("fleets", [])
    player = obs["player"]
    angular_velocity = obs.get("angular_velocity", 0.03)
    comets = obs.get("comets", [])
    comet_planet_ids = obs.get("comet_planet_ids", [])
    step = obs.get("step", 0)

    my_planets = []
    neutral_planets = []
    enemy_planets = []

    # planet format: [id, owner, x, y, radius, ships, production]
    # fleet format: [id, owner, x, y, angle, from_planet_id, ships]

    for p in planets:

        owner = p[1]

        if owner == player:
            my_planets.append(p)

        elif owner == -1:
            neutral_planets.append(p)

        else:
            enemy_planets.append(p)

    actions = []

    # ===== PHÒNG THỦ HOME PLANET =====
    # Kiểm tra fleet enemy đang đến home planets
    home_planets = [p for p in my_planets if p[6] >= 3]  # production >= 3 là home
    
    for fleet in fleets:
        if fleet[1] != player:  # enemy fleet
            for home in home_planets:
                dist = math.hypot(fleet[2] - home[2], fleet[3] - home[3])
                if dist < 15:  # fleet đang đến gần
                    # gửi thêm ships về phòng thủ
                    for source in my_planets:
                        if source[0] != home[0] and source[5] > 20:
                            send_ships = min(source[5] - 5, 10)
                            angle = math.atan2(home[3] - source[3], home[2] - source[2])
                            actions.append([source[0], angle, send_ships])
                            source[5] -= send_ships  # update ships
    
    # không còn hành tinh
    if not my_planets:
        return actions

    # sort theo production cao nhất (không phải ships)
    my_planets.sort(key=lambda p: p[6], reverse=True)

    used_targets = set()

    # mỗi hành tinh tự đi đánh
    for source in my_planets:

        ships = source[5]

        # ít quân thì skip (trừ khi đang phòng thủ)
        if ships < 20:
            continue

        target = None

        # ===== ĐÁNH GIÁ TẤT CẢ TARGET =====
        target = None
        best_score = -float('inf')
        best_angle = 0
        
        all_targets = neutral_planets + enemy_planets
        
        for t in all_targets:
            if t[0] in used_targets:
                continue
            
            # ===== ORBIT PREDICTION =====
            dist = distance(source, t)
            travel_time = dist / 3.0  # ước tính tốc độ
            pred_x, pred_y = predict_orbit_position(t, angular_velocity, travel_time)
            
            # tính angle đến vị trí dự đoán
            angle = math.atan2(pred_y - source[3], pred_x - source[2])
            
            # ===== SUN AVOIDANCE =====
            target_x = source[2] + math.cos(angle) * dist
            target_y = source[3] + math.sin(angle) * dist
            if will_cross_sun(source[2], source[3], target_x, target_y):
                continue  # skip target nếu đi qua sun
            
            # ===== COMET STRATEGY =====
            # Ưu tiên comets khi chúng xuất hiện
            if t[0] in comet_planet_ids and step in [50, 150, 250, 350, 450]:
                score_bonus = 30
            else:
                score_bonus = 0
            
            # tính score cho target
            score = 0
            
            # ưu tiên production cao
            score += t[6] * 15
            
            # ưu tiên gần
            score -= distance(source, t) * 0.3
            
            # ưu tiên neutral (dễ chiếm)
            if t[1] == -1:
                score += 25
            
            # ưu tiên target yếu
            score -= t[5] * 0.2
            
            # nếu là home planet của enemy, ưu tiên đánh
            if t[1] != -1 and t[6] >= 3:
                score += 10
            
            # bonus cho comets
            score += score_bonus
            
            if score > best_score:
                best_score = score
                target = t
                best_angle = angle

        if target is None:
            continue

        source_id = source[0]
        target_id = target[0]

        # tính số ships cần gửi
        ships_needed = target[5] + 5
        send_ships = min(ships_needed, source[5] - 5)


        if send_ships <= 0:
            continue

        # tính angle (đã tính ở trên)
        angle = best_angle
        
        actions.append([source_id, angle, send_ships])

        used_targets.add(target_id)

    return actions