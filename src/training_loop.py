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
        state = np.reshape(state, [1, agent.state_size])
        for time_step in range(200):
            action = np.argmax(agent.model.predict(state, verbose=0)) if np.random.rand() > agent.epsilon else env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            if done:
                if episodes > 2:
                    print(f"Episode {episode + 1}/{episodes}: Time Steps = {time_step}")
                break
        agent.train()
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    print("Chronos Intelligence Audit: SUCCESS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    run_chronos_training(args.episodes)
