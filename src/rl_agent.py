"""
===============================================================================
MODULE MANIFEST: CHRONOS AUTONOMIC AGENT (RL EXECUTION CORE)
===============================================================================
System Purpose:
    Serves as the sovereign execution controller for the CHRONOS engine.
    Orchestrates autonomic feedback loops, governing policy exploitation,
    experience storage, and experience replay optimization. Serves as the
    operational bridge between the continuous market environment and the
    underlying Deep Q-Network (DQN) policy approximator.

State Boundaries:
    - Governs episodic experience ingestion, ring-buffer replay memory, and
      stochastic action selection.
    - Encapsulates epsilon-decay exploration schedules and TD-error target updates.
    - Strictly delegates computational graph definitions, backpropagation updates,
      and loss compilation to the ChronosAI backbone.

Mathematical/Physical Invariants:
    1. Chronos_RL_Sequencing:
       Q(s, a) <- Q(s, a) + alpha * [Reward + gamma * max_a' Q(s', a') - Q(s, a)]
    2. Monotonic Entropy Decay:
       Exploration rate epsilon decays strictly monotonically:
       epsilon_(t+1) = max(epsilon_min, epsilon_t * epsilon_decay)
       driving transitions from high-entropy exploration to deterministic policy execution.

Design Rationale:
    A uniform random-sampling Experience Replay buffer breaks temporal autocorrelation
    in sequential financial time series, mitigating catastrophic forgetting.
    Mini-batch tensor operations parallelize target estimation, minimizing kinetic
    execution bottlenecks during neural weight updates.
===============================================================================
"""

from collections import deque
import random
from typing import Any, Deque, List, Tuple, Union
import numpy as np
import tensorflow as tf

from chronos_ai import ChronosAI


class RLAgent:
    """Autonomic Deep Q-Network (DQN) reinforcement learning agent.

    Coordinates experience replay ingestion, exploration-exploitation trade-offs,
    and mini-batch gradient updates for optimal order execution sequencing.

    Attributes:
        state_size (int): Dimensionality of the incoming observation state tensor.
        action_size (int): Total count of discrete execution primitives.
        memory (Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]]): Cyclic
            replay buffer storing state-action transition tuples.
        gamma (float): Discount factor for future expected utilities.
        epsilon (float): Dynamic probability threshold for exploratory action selection.
        epsilon_min (float): Lower bound on exploration probability.
        epsilon_decay (float): Multiplicative decay rate applied to epsilon per update.
        model (tf.keras.Model): Compiled Q-value regression model from ChronosAI.
    """

    def __init__(self, state_size: int, action_size: int) -> None:
        """Initializes the RLAgent parameters, replay queue, and policy backbone.

        Args:
            state_size: Dimensionality of the environment state space.
            action_size: Cardinality of the discrete action space.
        """
        self.state_size: int = state_size
        self.action_size: int = action_size

        # [STRUCTURAL CALLOUT] Finite Memory Horizon
        # Physical Constraint: Caps replay memory at 2000 transitions.
        # Operates as a FIFO queue to prevent training on stale microstructure regimes.
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=2000)

        self.gamma: float = 0.95
        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.01
        self.epsilon_decay: float = 0.995

        # Instantiates the underlying neural backbone
        self.model: tf.keras.Model = ChronosAI(state_size, action_size).model

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Appends a discrete state transition tuple to the FIFO replay buffer.

        Args:
            state: Pre-transition observation vector of shape (state_size,).
            action: Discrete action executed by the agent in range [0, action_size - 1].
            reward: Scalar feedback emitted by the environment (MtM PnL net of impact).
            next_state: Post-transition observation vector of shape (state_size,).
            done: Terminal boundary indicator for the transitioned episode.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """Selects an action primitive via an epsilon-greedy decision policy.

        Mathematical Invariants:
            - Exploration Probability: P(a = random) = epsilon
            - Exploitation Probability: P(a = argmax_a Q(s, a)) = 1 - epsilon

        Args:
            state: Current environment observation vector.

        Returns:
            int: Discrete action primitive selected for execution.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # [STRUCTURAL CALLOUT] Dimension-Safe Tensor Reshaping
        # Dynamic projection guarantees a 2D batch shape (1, state_size)
        # without mutating source array references.
        state_tensor: np.ndarray = (
            np.reshape(state, (1, self.state_size)) if state.ndim == 1 else state
        )
        q_values: np.ndarray = self.model.predict(state_tensor, verbose=0)
        return int(np.argmax(q_values[0]))

    def train(self, batch_size: int = 32) -> None:
        """Performs a mini-batch gradient descent update on the Q-network.

        Samples decorrelated transitions from replay memory, calculates target
        Q-values using the Bellman optimality operator, and updates network weights.

        Mathematical Invariants:
            - Terminal Step: Target Q(s_t, a_t) = r_t
            - Non-Terminal Step: Target Q(s_t, a_t) = r_t + gamma * max_a' Q(s_(t+1), a')
            - Epsilon Schedule: epsilon <- max(epsilon_min, epsilon * epsilon_decay)

        Args:
            batch_size: Minimum batch threshold required to execute gradient optimization.
        """
        if len(self.memory) < batch_size:
            return

        batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = random.sample(
            self.memory, batch_size
        )

        # [STRUCTURAL CALLOUT] Triune Vectorization & Kinetic Speedup
        # Stacks sampled transition tuples into contiguous memory buffers to
        # leverage SIMD/GPU parallelization across inference passes.
        states: np.ndarray = np.vstack([x[0] for x in batch])
        actions: np.ndarray = np.array([x[1] for x in batch], dtype=np.int32)
        rewards: np.ndarray = np.array([x[2] for x in batch], dtype=np.float32)
        next_states: np.ndarray = np.vstack([x[3] for x in batch])
        dones: np.ndarray = np.array([x[4] for x in batch], dtype=bool)

        # Batched inference for current and next-state Q-distributions
        target_values: np.ndarray = self.model.predict(states, verbose=0)
        next_q_values: np.ndarray = self.model.predict(next_states, verbose=0)

        # [STRUCTURAL CALLOUT] Vectorized Bellman Target Update
        # Direct implementation of Chronos_RL_Sequencing.
        for i in range(batch_size):
            if dones[i]:
                target_values[i, actions[i]] = rewards[i]
            else:
                target_values[i, actions[i]] = rewards[i] + self.gamma * float(
                    np.max(next_q_values[i])
                )

        # Gradient update on batched targets
        self.model.fit(
            states, target_values, batch_size=batch_size, epochs=1, verbose=0
        )

        # [STRUCTURAL CALLOUT] Triune Entropy Decay
        # Enforces deterministic policy convergence by decaying exploration entropy.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
