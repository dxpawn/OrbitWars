# bot_v2.py
# IMPROVED ORBIT WARS BOT

import math


def distance(a, b):
    return math.hypot(a[2] - b[2], a[3] - b[3])


def predict_orbit_position(planet, angular_velocity, steps_ahead):
    """Dự đoán vị trí orbit planet sau steps_ahead lượt"""
    if planet[4] + 10 >= 50:  # static planet (radius + sun_radius >= 50)
        return planet[2], planet[3]
    
    # orbit planet - tính vị trí mới
    x, y = planet[2], planet[3]
    center_x, center_y = 50, 50
    
    # tính góc hiện tại
    current_angle = math.atan2(y - center_y, x - center_x)
    
    # góc mới sau steps_ahead
    new_angle = current_angle + angular_velocity * steps_ahead
    
    # tính vị trí mới
    radius = math.hypot(x - center_x, y - center_y)
    new_x = center_x + radius * math.cos(new_angle)
    new_y = center_y + radius * math.sin(new_angle)
    
    return new_x, new_y


def will_cross_sun(x1, y1, x2, y2, sun_radius=10):
    """Kiểm tra xem đường đi có qua sun không"""
    center_x, center_y = 50, 50
    
    # khoảng cách từ đường thẳng đến center sun
    # sử dụng công thức khoảng cách từ điểm đến đường thẳng
    numerator = abs((y2 - y1) * center_x - (x2 - x1) * center_y + x2 * y1 - y2 * x1)
    denominator = math.hypot(y2 - y1, x2 - x1)
    
    if denominator == 0:
        return False
    
    dist_to_sun = numerator / denominator
    
    return dist_to_sun < sun_radius


def agent(obs, config):
    planets = obs["planets"]
    player = obs["player"]
    angular_velocity = obs.get("angular_velocity", 0.03)
    
    my_planets = []
    neutral_planets = []
    enemy_planets = []
    
    # planet format: [id, owner, x, y, radius, ships, production]
    for p in planets:
        owner = p[1]
        if owner == player:
            my_planets.append(p)
        elif owner == -1:
            neutral_planets.append(p)
        else:
            enemy_planets.append(p)
    
    actions = []
    
    if not my_planets:
        return actions
    
    # sort theo production cao nhất
    my_planets.sort(key=lambda p: p[6], reverse=True)
    
    used_targets = set()
    
    for source in my_planets:
        ships = source[5]
        
        # giữ lại ít nhất 5 ships để phòng thủ
        if ships < 10:
            continue
        
        target = None
        best_score = -float('inf')
        
        # đánh giá tất cả target có thể
        all_targets = neutral_planets + enemy_planets
        
        for t in all_targets:
            if t[0] in used_targets:
                continue
            
            # dự đoán vị trí target khi tàu đến
            dist = distance(source, t)
            travel_time = dist / 3.0  # ước tính tốc độ trung bình
            pred_x, pred_y = predict_orbit_position(t, angular_velocity, travel_time)
            
            # tính angle đến vị trí dự đoán
            angle = math.atan2(pred_y - source[3], pred_x - source[2])
            
            # kiểm tra có qua sun không
            target_x = source[2] + math.cos(angle) * dist
            target_y = source[3] + math.sin(angle) * dist
            if will_cross_sun(source[2], source[3], target_x, target_y):
                continue
            
            # tính score cho target
            score = 0
            
            # ưu tiên production cao
            score += t[6] * 10
            
            # ưu tiên gần
            score -= dist * 0.5
            
            # ưu tiên neutral (dễ chiếm hơn)
            if t[1] == -1:
                score += 20
            
            # ưu tiên target yếu
            score -= t[5] * 0.3
            
            # nếu là home planet của enemy, ưu tiên đánh
            if t[1] != -1 and t[6] >= 3:
                score += 15
            
            if score > best_score:
                best_score = score
                target = t
                target_angle = angle
        
        if target is None:
            continue
        
        # tính số ships cần gửi
        ships_needed = target[5] + 5
        send_ships = min(ships_needed, ships - 5)
        
        if send_ships <= 0:
            continue
        
        actions.append([source[0], target_angle, send_ships])
        used_targets.add(target[0])
    
    return actions
