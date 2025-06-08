import numpy as np
import pandas as pd
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(self, n_actions=8, state_size=16, alpha=0.3, gamma=0.9, epsilon=0.5):
        self.n_actions = n_actions
        self.state_size = state_size
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

    def get_state_key(self, state):
        return tuple((state * 10).astype(int))  # ~10x faster than round + tuple


    def act(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.alpha * (target - self.q_table[state_key][action])
