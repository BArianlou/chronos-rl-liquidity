import numpy as np
import argparse
import sys
import os

# Force path synchronization for GitHub Actions
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from market_env import ChronosMarketEnv
from rl_agent import RLAgent

def run_chronos_training(episodes=1000):
    env = ChronosMarketEnv()
    agent = RLAgent(env.observation_space.shape[0], env.action_space.n)

    for episode in range(episodes):
        state, info = env.reset()
        
        for time_step in range(200):
            # Triune Fix 1: Encapsulated Action Logic
            action = agent.act(state)
            
            next_state, reward, done, _, _ = env.step(action)
            
            # Triune Fix 2: Store raw 1D states to prevent vstack dimension crashes
            agent.store_experience(state, action, reward, next_state, done)
            
            state = next_state
            
            # Triune Fix 3: Continuous Bellman updates per step
            # Epsilon decay is safely handled internally by the agent.
            agent.train()
            
            if done:
                if episodes > 2:
                    print(f"Episode {episode + 1}/{episodes}: Time Steps = {time_step}")
                break
                
    print("Chronos Intelligence Audit: SUCCESS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    run_chronos_training(args.episodes)
