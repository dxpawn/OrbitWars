from kaggle_environments import make
import pandas as pd
from collections import defaultdict
from itertools import combinations
import importlib.util
import sys
import os

# ============================================================
# IMPORT AGENTS
# ============================================================

# Import NovaHeuristic
spec_nova = importlib.util.spec_from_file_location("novaHeuristic", "adversaries/novaHeuristic.py")
nova_module = importlib.util.module_from_spec(spec_nova)
spec_nova.loader.exec_module(nova_module)
nova_agent = nova_module.agent

# Import Evogen
spec_evo = importlib.util.spec_from_file_location("evogen", "adversaries/evogen.py")
evo_module = importlib.util.module_from_spec(spec_evo)
spec_evo.loader.exec_module(evo_module)
evo_agent = evo_module.agent

# Import Hellburner
spec_hell = importlib.util.spec_from_file_location("hellburner", "adversaries/hellburner.py")
hell_module = importlib.util.module_from_spec(spec_hell)
spec_hell.loader.exec_module(hell_module)
hell_agent = hell_module.agent

# Import User (upgraded)
spec_user = importlib.util.spec_from_file_location("user_upgraded", "user(hellburner_upgraded).py")
user_module = importlib.util.module_from_spec(spec_user)
spec_user.loader.exec_module(user_module)
user_agent = user_module.agent

# Import heuristic1025
spec_heur = importlib.util.spec_from_file_location("heuristic1025", "heuristic1025.py")
heur_module = importlib.util.module_from_spec(spec_heur)
spec_heur.loader.exec_module(heur_module)
heur_agent = heur_module.agent

# ============================================================
# MATCH FUNCTIONS
# ============================================================

def run_match(agent1, agent2):
    env = make("orbit_wars", debug=True)
    env.run([agent1, agent2])

    r1 = env.steps[-1][0].reward
    r2 = env.steps[-1][1].reward

    r1 = 0 if r1 is None else r1
    r2 = 0 if r2 is None else r2

    return r1, r2


def run_ffa(agent_list):
    env = make("orbit_wars", debug=True)
    env.run(agent_list)

    rewards = []

    for player in env.steps[-1]:
        reward = player.reward
        reward = 0 if reward is None else reward
        rewards.append(reward)

    return rewards

# ============================================================
# BOTS
# ============================================================

bots = {
    "Model 0 (NovaHeur)": nova_agent,
    "Model 1 (Evogen)": evo_agent,
    "Model 2 (User)": user_agent,
    "Model 3 (Heuristic1025)": heur_agent
}

# ============================================================
# RESULTS
# ============================================================

results = defaultdict(lambda: {
    "Wins": 0,
    "Losses": 0,
    "Games": 0
})

# ============================================================
# 1V1 TOURNAMENT
# ============================================================

print("--- ĐANG CHẠY GIẢI ĐẤU 1v1 ---")

matchups = list(combinations(bots.keys(), 2))

for name1, name2 in matchups:

    print(f"Đang đấu: {name1} vs {name2}...")

    for i in range(20):

        r1, r2 = run_match(
            bots[name1],
            bots[name2]
        )

        results[name1]["Games"] += 1
        results[name2]["Games"] += 1

        if r1 > r2:
            results[name1]["Wins"] += 1
            results[name2]["Losses"] += 1

        elif r2 > r1:
            results[name2]["Wins"] += 1
            results[name1]["Losses"] += 1

# ============================================================
# FREE FOR ALL (4-PLAYER)
# ============================================================

print("\n--- ĐANG CHẠY GIẢI ĐẤU 4-PLAYER (FFA) ---")

bot_names = list(bots.keys())
bot_files = list(bots.values())

# Run FFA with all 4 models
for game in range(20):  # 20 games for FFA

    print(f"  Game {game+1}/20")

    rewards = run_ffa(bot_files)

    best_reward = max(rewards)

    winners = [
        i for i, r in enumerate(rewards)
        if r == best_reward
    ]

    for name in bot_names:
        results[name]["Games"] += 1

    # chỉ tính thắng nếu có đúng 1 winner
    if len(winners) == 1:

        winner_idx = winners[0]
        winner_name = bot_names[winner_idx]

        results[winner_name]["Wins"] += 1

        # những thằng còn lại tính thua
        for i, name in enumerate(bot_names):
            if i != winner_idx:
                results[name]["Losses"] += 1

# ============================================================
# DISPLAY RESULTS
# ============================================================

df = pd.DataFrame.from_dict(results, orient='index')

df['Win Rate %'] = (
    df['Wins'] / df['Games']
) * 100

print("\n--- KẾT QUẢ CUỐI CÙNG ---")

print(df.sort_values(
    by="Win Rate %",
    ascending=False
))