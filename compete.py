import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_environments import make
from adversaries.hellburner import agent as hellburner_agent
from adversaries.evogen import agent as evogen_agent  
from adversaries.novaHeuristic import agent as nova_agent
import importlib.util

# Import user models
spec1 = importlib.util.spec_from_file_location("user_base", "user(hellburner_base).py")
user_base_module = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(user_base_module)
user_base_agent = user_base_module.agent

spec2 = importlib.util.spec_from_file_location("user_upgraded", "user(hellburner_upgraded).py")
user_upgraded_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(user_upgraded_module)
user_upgraded_agent = user_upgraded_module.agent

def run_competition(num_games=20):
    """Run competition between all models."""
    
    models = [
        ("Evogen", evogen_agent),
        ("Hellburner", hellburner_agent),
        ("NovaHeuristic", nova_agent),
        ("User (Base)", user_base_agent),
        ("User (Upgraded)", user_upgraded_agent),
    ]
    
    results = {name: {"wins": 0, "losses": 0, "total": 0} for name, _ in models}
    
    env = make("orbit_wars", configuration={"episodeSteps": 500})
    
    print(f"Running {num_games} games between each pair of models...")
    print("=" * 80)
    
    for i, (name1, agent1) in enumerate(models):
        for j, (name2, agent2) in enumerate(models):
            if i >= j:  # Avoid duplicate pairs and self-play
                continue
            
            print(f"\n{name1} vs {name2}:")
            
            for game in range(num_games):
                # Run game
                env.reset(num_agents=2)
                env.configuration["agent_names"] = [name1, name2]
                
                # Alternate who goes first
                if game % 2 == 0:
                    agents = [agent1, agent2]
                else:
                    agents = [agent2, agent1]
                
                # Run simulation
                steps = env.run(agents)
                
                # Determine winner
                final_state = steps[-1]
                rewards = final_state[0]['reward'], final_state[1]['reward']
                
                if rewards[0] > rewards[1]:
                    winner = name1 if game % 2 == 0 else name2
                    loser = name2 if game % 2 == 0 else name1
                    results[winner]["wins"] += 1
                    results[loser]["losses"] += 1
                elif rewards[1] > rewards[0]:
                    winner = name2 if game % 2 == 0 else name1
                    loser = name1 if game % 2 == 0 else name2
                    results[winner]["wins"] += 1
                    results[loser]["losses"] += 1
                
                results[name1]["total"] += 1
                results[name2]["total"] += 1
                
                # Progress indicator
                if (game + 1) % 5 == 0:
                    print(f"  Game {game + 1}/{num_games} completed")
    
    # Print final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    # Sort by win rate
    sorted_results = sorted(
        results.items(),
        key=lambda x: (x[1]["wins"] / x[1]["total"] if x[1]["total"] > 0 else 0),
        reverse=True
    )
    
    for rank, (name, stats) in enumerate(sorted_results, 1):
        win_rate = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"{rank}. {name:20s} | Wins: {stats['wins']:3d} | Losses: {stats['losses']:3d} | Total: {stats['total']:3d} | Win Rate: {win_rate:5.1f}%")
    
    print("=" * 80)

if __name__ == "__main__":
    # Run competition with 20 games per pair (adjust as needed)
    run_competition(num_games=20)
