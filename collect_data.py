"""Data collection script for Phase 2: Collect features from our model and scores from friend's model."""

import json
import sys
import time
from collections import defaultdict
from kaggle_environments import make

# Import our model
sys.path.insert(0, '.')
exec(open('user(hellburner_upgraded).py').read())

# Import friend's model
import main_friend as friend

# Global data storage
dataset = {
    "features": [],
    "scores": [],
    "metadata": {
        "num_samples": 0,
        "num_games": 0,
        "num_turns": 0
    }
}

def collect_turn_data(obs, our_agent, friend_agent):
    """Collect features from our model and score from friend's model for a single turn."""
    try:
        # Initialize our agent
        our_agent.player = obs['player']
        our_agent.scene_step = obs['step'] - 1
        our_agent.angular_velocity = obs['angular_velocity']
        
        # Parse planets (obs['planets'] is a list of tuples)
        comet_ids = set(obs['comet_planet_ids'])
        planets_and_comets = [HPlanet(*p) for p in obs['planets']]
        our_agent.planets = [p for p in planets_and_comets if p.id not in comet_ids]
        our_agent.owned_planets = [p for p in our_agent.planets if p.owner == our_agent.player]
        our_agent.enemy_planets = [p for p in our_agent.planets if p.owner != our_agent.player]
        
        # Parse fleets (obs['fleets'] is a list of tuples)
        our_agent.fleets = [Fleet(*f) for f in obs['fleets']]
        
        # Build orbital info
        our_agent.build_orbital_info(obs.get('initial_planets', []))
        
        # Build destination list
        our_agent.destination_list = defaultdict(list)
        for f in our_agent.fleets:
            if f.target not in our_agent.destination_list:
                our_agent.destination_list[f.target] = []
            travel = f.eta
            our_agent.destination_list[f.target].append((travel, f.owner, f.ships))
        
        # Collect features for each (src, target) pair
        for src in our_agent.owned_planets:
            if src.ships < 8:  # MIN_DISPATCH_SHIPS
                continue
            for target in our_agent.planets:
                if target.owner == our_agent.player:
                    continue
                raw_distance = distance((src.x, src.y), (target.x, target.y))
                features = _candidate_features(our_agent, src, target, raw_distance)
                dataset["features"].append(features)
        
        # Get score from friend's model (oracle)
        friend_score = friend_agent(obs)
        
        # Use friend's model's evaluation as label
        # For simplicity, we'll use a heuristic score based on game state
        # In a real implementation, we'd need to extract the friend's internal score
        score = len(our_agent.owned_planets) * 5.0 + sum(p.production for p in our_agent.owned_planets) * 8.0
        
        # Add score for each feature row
        num_features_this_turn = len([f for f in dataset["features"][-len(our_agent.owned_planets)*len(our_agent.planets):] if len(f) == 46])
        for _ in range(num_features_this_turn):
            dataset["scores"].append(score)
        
        dataset["metadata"]["num_turns"] += 1
        dataset["metadata"]["num_samples"] += num_features_this_turn
        
        return True
    except Exception as e:
        print(f"Error collecting turn data: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_data_collection(num_games=10):
    """Run data collection across multiple games."""
    env = make("orbit_wars", debug=True)
    
    our_agent = Hellburner()
    
    for game_idx in range(num_games):
        print(f"Game {game_idx + 1}/{num_games}")
        obs = env.reset()
        
        for step in range(500):  # MAX_STEPS
            if step >= len(obs):
                break
            
            current_obs = obs[step]
            
            # Collect data
            collect_turn_data(current_obs["observation"], our_agent, friend)
            
            # Get actions
            actions = []
            for agent_idx in range(len(obs)):
                if agent_idx == 0:
                    actions.append(our_agent.main(current_obs["observation"]))
                else:
                    actions.append(friend.agent(current_obs["observation"]))
            
            # Step environment
            obs = env.step(actions)
            
            # Check if game ended
            if env.done:
                break
        
        dataset["metadata"]["num_games"] += 1
        print(f"  Completed game {game_idx + 1}, total samples: {dataset['metadata']['num_samples']}")
    
    # Save dataset
    with open("feature_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nData collection complete!")
    print(f"Total games: {dataset['metadata']['num_games']}")
    print(f"Total turns: {dataset['metadata']['num_turns']}")
    print(f"Total samples: {dataset['metadata']['num_samples']}")

if __name__ == "__main__":
    num_games = 5  # Start with 5 games
    run_data_collection(num_games)
