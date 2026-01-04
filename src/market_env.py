import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ChronosMarketEnv(gym.Env):
    """Institutional Environment: Simulates asymmetric market impact and liquidity."""
    def __init__(self):
        super(ChronosMarketEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([100.0, 0.02, 1.0, 0.0], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        price, vol, depth, pos = self.state
        self.steps += 1
        price_change = np.random.normal(0, price * vol)
        new_price = price + price_change
        reward = 0
        
        if action == 1: # Buy
            impact = (0.01 / depth) * price
            reward = -impact
            pos += 1
        elif action == 2: # Sell
            impact = (0.01 / depth) * price
            reward = (new_price - price) - impact
            pos = 0
            
        self.state = np.array([new_price, vol, depth, pos], dtype=np.float32)
        terminated = self.steps >= 200
        return self.state, reward, terminated, False, {}
