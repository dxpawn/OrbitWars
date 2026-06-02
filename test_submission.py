from kaggle_environments import make
import submission

def test_agent(num_games=10):
    env = make("orbit_wars", debug=True)
    total_reward = 0
    wins = 0
    
    for i in range(num_games):
        steps = env.run([submission.agent, "random"])
        reward = steps[-1][0]["reward"]
        total_reward += reward
        if reward > 0:
            wins += 1
        print(f"Game {i+1}: reward = {reward}")
    
    avg_reward = total_reward / num_games
    win_rate = wins / num_games * 100
    
    print(f"\n=== Summary ===")
    print(f"Total games: {num_games}")
    print(f"Wins: {wins}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Avg reward: {avg_reward:.2f}")
    
    return avg_reward, win_rate

def test_self_play(num_games=10):
    """Test agent vs chính nó để xem độ ổn định"""
    env = make("orbit_wars", debug=True)
    total_reward = 0
    wins = 0
    
    for i in range(num_games):
        steps = env.run([submission.agent, submission.agent])
        reward = steps[-1][0]["reward"]
        total_reward += reward
        if reward > 0:
            wins += 1
        print(f"Self-play Game {i+1}: reward = {reward}")
    
    avg_reward = total_reward / num_games
    win_rate = wins / num_games * 100
    
    print(f"\n=== Self-play Summary ===")
    print(f"Total games: {num_games}")
    print(f"Wins: {wins}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Avg reward: {avg_reward:.2f}")
    
    return avg_reward, win_rate

if __name__ == "__main__":
    print("=== Test vs Random ===")
    test_agent(num_games=10)
    
    print("\n=== Test Self-play ===")
    test_self_play(num_games=10)
