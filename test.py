import math
from kaggle_environments import make

# =========================
# SIMPLE PLANET BOT
# =========================

def distance(a, b):
    return math.hypot(a[2] - b[2], a[3] - b[3])


def agent(obs, config):
    planets = obs["planets"]
    player = obs["player"]

    my_planets = []
    neutral_planets = []
    enemy_planets = []

    # planet format:
    # [id, owner, x, y, production, ships, level]

    for p in planets:
        owner = p[1]

        if owner == player:
            my_planets.append(p)

        elif owner == -1:
            neutral_planets.append(p)

        else:
            enemy_planets.append(p)

    actions = []

    # nếu không có hành tinh của mình thì thôi
    if not my_planets:
        return actions

    # lấy hành tinh mạnh nhất
    strongest = max(my_planets, key=lambda p: p[5])

    # chỉ attack nếu đủ quân
    if strongest[5] < 30:
        return actions

    target = None

    # ưu tiên hành tinh neutral gần nhất
    if neutral_planets:
        target = min(
            neutral_planets,
            key=lambda p: distance(strongest, p)
        )

    # nếu hết neutral thì đánh enemy yếu nhất
    elif enemy_planets:
        target = min(
            enemy_planets,
            key=lambda p: p[5]
        )

    if target is None:
        return actions

    source_id = strongest[0]
    target_id = target[0]

    # gửi nửa quân
    ships_to_send = strongest[5] // 2

    # tính angle để gửi tàu
    angle = math.atan2(target[3] - strongest[3], target[2] - strongest[2])
    
    # action format: [from_planet_id, angle, num_ships]
    actions.append([source_id, angle, ships_to_send])

    return actions


# =========================
# TEST LOCAL
# =========================

if __name__ == "__main__":

    env = make("orbit_wars", debug=True)

    # main.py = bot của mình
    # random = bot random mặc định
    steps = env.run([agent, "random"])

    print("GAME FINISHED")
    print("FINAL REWARD:", steps[-1][0]["reward"])

    # in replay json
    print(env.toJSON())