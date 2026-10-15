"""
===============================================================================
MODULE MANIFEST: CHRONOS MARKET ENVIRONMENT (GYMNASIUM)
===============================================================================
System Purpose:
    Serves as the deterministic simulation engine for the CHRONOS reinforcement
    learning agent. Implements a high-fidelity Markov Decision Process (MDP)
    modeling asymmetric market impact, slippage friction, and liquidity
    fragmentation encountered in institutional execution sequencing.

State Boundaries:
    - Encapsulates state evolution: price generation, liquidity depth
      degradation, and order impact dynamics.
    - Emits 4D observation tensors, scalar rewards, and termination/truncation
      flags to the DQN policy graph.
    - Strictly decoupled from policy optimization, Q-value estimation, and order
      routing infrastructures.

Mathematical/Physical Invariants:
    1. Liquidity Fragmentation Penalty (LFP):
       Execution cost is inversely proportional to available book depth:
       Impact_t = (eta / Depth_t) * Price_t
    2. Geometric Price Bounds:
       Asset prices are clamped strictly above zero (P_t >= 0.01) to prevent
       mathematical breakdown in return and variance calculations.
    3. Mark-to-Market (MtM) Conservation:
       Total reward balances unrealized portfolio inventory delta against
       incurred execution slippage:
       R_t = (Pos_t * Delta_P_t) - Execution_Cost_t

Design Rationale:
    Standard academic execution environments assume infinite liquidity at the
    mid-price. This environment enforces realistic microstructure penalties,
    forcing the agent to resolve the fundamental trade-off between execution
    velocity (capturing price alpha) and market impact (slippage decay).
===============================================================================
"""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ChronosMarketEnv(gym.Env):
    """Discrete-action market execution simulation environment.

    Simulates order book dynamics, inventory risk, and liquidity fragmentation
    under an institutional trade execution mandate.

    Attributes:
        max_position (float): Absolute ceiling on held inventory lots.
        observation_space (spaces.Box): 4-dimensional continuous state bounds:
            [Price, Volatility, Liquidity Depth, Inventory Position].
        action_space (spaces.Discrete): Action primitives {0: Hold, 1: Buy, 2: Sell}.
        state (np.ndarray): Current 4D observation vector [P_t, Vol_t, Depth_t, Pos_t].
        steps (int): Step counter for the active episode horizon.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        """Initializes the market environment spaces and configuration parameters."""
        super().__init__()

        # Inventory Exposure Limit: Caps maximum allowable position size to prevent
        # unconstrained leverage exploits during early policy exploration.
        self.max_position: float = 5.0

        # Observation Space Bounds:
        # [Price (P >= 0.01), Volatility (V >= 0.0), Depth (D >= 0.01), Position (0.0 <= Pos <= Max)]
        # Enforces non-degenerate market state boundaries.
        self.observation_space: spaces.Box = spaces.Box(
            low=np.array([0.01, 0.0, 0.01, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, self.max_position], dtype=np.float32),
            dtype=np.float32,
        )

        # Action Space: 0 (Hold/Observe), 1 (Execute Buy), 2 (Execute Sell)
        self.action_space: spaces.Discrete = spaces.Discrete(3)

        self.state: np.ndarray = np.empty((4,), dtype=np.float32)
        self.steps: int = 0
        self.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment state to canonical baseline anchors.

        Args:
            seed: PRNG seed initializing `self.np_random` for reproducible trajectories.
            options: Supplementary configuration flags (reserved for interface parity).

        Returns:
            Tuple containing:
                - state (np.ndarray): Baseline state [P=100.0, Vol=0.02, Depth=1.0, Pos=0.0].
                - info (dict): Empty auxiliary diagnostic dictionary.
        """
        super().reset(seed=seed)
        self.state = np.array([100.0, 0.02, 1.0, 0.0], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advances the market simulation by one discrete temporal increment.

        Generates stochastic price motion via the environment-scoped PRNG, applies
        liquidity impact penalties contingent on order direction and prevailing depth,
        and computes mark-to-market rewards.

        Mathematical Invariants:
            - Price Diffusion: P_(t+1) = max(0.01, P_t + epsilon), where epsilon ~ N(0, (P_t * Vol_t)^2)
            - Impact Penalty: Impact = (eta / Depth_t) * P_t
            - Net Reward: R_t = (Pos_(t+1) * Delta_P_t) - Execution_Cost_t

        Args:
            action: Discrete execution signal (0 = Hold, 1 = Buy, 2 = Sell).

        Returns:
            Tuple containing:
                - next_state (np.ndarray): Updated 4D state vector.
                - reward (float): Mark-to-market PnL net of execution slippage costs.
                - terminated (bool): True if asset price hits absorbing lower bound (0.01).
                - truncated (bool): True if episode reaches the 200-step time horizon.
                - info (dict): Diagnostic metadata dictionary.
        """
        price, vol, depth, pos = self.state
        self.steps += 1

        # Scoped PRNG Diffusion & Lower-Bound Price Clamp:
        # Uses Gymnasium's seeded `self.np_random` generator for deterministic replay
        # and clamps price at 0.01 to preserve numerical stability.
        price_change: float = float(self.np_random.normal(0.0, price * vol))
        new_price: float = max(0.01, price + price_change)

        execution_cost: float = 0.0

        # Order Book Impact & Slippage Calculation:
        # Execution cost is inversely proportional to available order book depth,
        # penalizing aggressive liquidity consumption in thin books.
        if action == 1 and pos < self.max_position:  # Buy
            impact = (0.01 / depth) * price
            execution_cost = impact
            pos += 1.0
        elif action == 2 and pos > 0:  # Sell
            impact = (0.01 / depth) * price
            execution_cost = impact
            pos -= 1.0

        # Mark-to-Market (MtM) Net Reward:
        # Balances unrealized valuation change of held inventory against
        # instantaneous slippage and market impact costs.
        mtm_pnl: float = pos * (new_price - price)
        reward: float = float(mtm_pnl - execution_cost)

        self.state = np.array([new_price, vol, depth, pos], dtype=np.float32)

        # Gymnasium v0.26+ Terminal vs. Truncation Semantics:
        # - `terminated`: True only if asset reaches absorbing distress boundary (P <= 0.01).
        # - `truncated`: True when the fixed 200-step episodic horizon is reached.
        terminated: bool = bool(new_price <= 0.01)
        truncated: bool = self.steps >= 200

        return self.state, reward, terminated, truncated, {}
