import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ChronosMarketEnv(gym.Env):
    """Institutional Environment: Simulates asymmetric market impact and liquidity."""
    def __init__(self):
        super(ChronosMarketEnv, self).__init__()
        
        # Triune Fix 1: Deterministic Position Bounds
        self.max_position = 5.0 
        
        # Ge's Precision Tweak: Explicit State-Space Boundaries
        self.observation_space = spaces.Box(
            low=np.array([0.01, 0.0, 0.01, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, self.max_position], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([100.0, 0.02, 1.0, 0.0], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        price, vol, depth, pos = self.state
        self.steps += 1
        
        # Triune Fix 2: Prevent negative prices (Geometric boundary)
        price_change = np.random.normal(0, price * vol)
        new_price = max(0.01, price + price_change)
        
        execution_cost = 0.0
        
        # Triune Fix 3: Position-Aware Execution & Kinetic Friction
        if action == 1 and pos < self.max_position: # Buy
            impact = (0.01 / depth) * price
            execution_cost = impact
            pos += 1
        elif action == 2 and pos > 0: # Sell
            impact = (0.01 / depth) * price
            execution_cost = impact
            pos -= 1
            
        # Triune Fix 4: Mark-to-Market (MtM) Reward Logic
        mtm_pnl = pos * (new_price - price)
        reward = mtm_pnl - execution_cost
            
        self.state = np.array([new_price, vol, depth, pos], dtype=np.float32)
        terminated = self.steps >= 200
        
        return self.state, reward, terminated, False, {}
