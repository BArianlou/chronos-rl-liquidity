import numpy as np
import random
from collections import deque
from .chronos_ai import ChronosAI

class RLAgent:
    """The Sovereign Actor: Executes policies derived from the ChronosAI backbone."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = ChronosAI(state_size, action_size).model

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.model.predict(next_state, verbose=0))
            target_values = self.model.predict(state, verbose=0)
            target_values[0][action] = target
            self.model.fit(state, target_values, epochs=1, verbose=0)
