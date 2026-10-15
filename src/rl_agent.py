"""
===============================================================================
MODULE MANIFEST: CHRONOS RL AGENT (EXPERIENCE REPLAY & POLICY CONTROLLER)
===============================================================================
"""

from collections import deque
import random
from typing import Deque, List, Tuple
import numpy as np
import tensorflow as tf

from chronos_ai import ChronosAI


class RLAgent:
    """Coordinates experience replay ingestion, exploration-exploitation trade-offs,
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
        self.state_size: int = state_size
        self.action_size: int = action_size

        # Replay Buffer Management:
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=2000)

        self.gamma: float = 0.95
        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.01
        self.epsilon_decay: float = 0.995

        self.model: tf.keras.Model = ChronosAI(state_size, action_size).model

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor: np.ndarray = (
            np.reshape(state, (1, self.state_size)) if state.ndim == 1 else state
        )
        q_values: np.ndarray = self.model.predict(state_tensor, verbose=0)
        return int(np.argmax(q_values[0]))

    def train(self, batch_size: int = 32) -> None:
        if len(self.memory) < batch_size:
            return

        batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = random.sample(
            self.memory, batch_size
        )

        states: np.ndarray = np.vstack([x[0] for x in batch])
        actions: np.ndarray = np.array([x[1] for x in batch], dtype=np.int32)
        rewards: np.ndarray = np.array([x[2] for x in batch], dtype=np.float32)
        next_states: np.ndarray = np.vstack([x[3] for x in batch])
        dones: np.ndarray = np.array([x[4] for x in batch], dtype=bool)

        target_values: np.ndarray = self.model.predict(states, verbose=0)
        next_q_values: np.ndarray = self.model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                target_values[i, actions[i]] = rewards[i]
            else:
                target_values[i, actions[i]] = rewards[i] + self.gamma * float(
                    np.max(next_q_values[i])
                )

        self.model.fit(
            states, target_values, batch_size=batch_size, epochs=1, verbose=0
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
