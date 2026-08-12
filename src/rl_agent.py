import numpy as np
import random
from collections import deque
from chronos_ai import ChronosAI 

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

    def act(self, state):
        """Epsilon-greedy action selection with dimension-safe tensor reshaping."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Reshape single state array for model inference: (1, state_size)
        state_tensor = np.reshape(state, (1, self.state_size)) if state.ndim == 1 else state
        q_values = self.model.predict(state_tensor, verbose=0)
        return np.argmax(q_values[0])

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        
        # Triune Vectorization: Extract batch into contiguous matrices
        states = np.vstack([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.vstack([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        # Predict all current and next Q-values in TWO matrix operations instead of 64
        target_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        # Vectorized Bellman Equation update
        for i in range(batch_size):
            if dones[i]:
                target_values[i][actions[i]] = rewards[i]
            else:
                target_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Single fit call for the entire batch (Massive kinetic speedup)
        self.model.fit(states, target_values, batch_size=batch_size, epochs=1, verbose=0)

        # Triune Entropy Decay: Enforce the deterministic shift
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
