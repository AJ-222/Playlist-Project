import numpy as np
import pandas as pd

class RandomAgent:
    def __init__(self, songs, user, length=10):
        self.songs = songs
        self.user = user
        self.length = length
        self.playlist = []
        self.currentPos = 0

    def reset(self):
        self.playlist = []
        self.currentPos = 0
        self.startMood = self.user.startMood
        self.endMood = self.user.endMood
        return self.getState()

    def getState(self):
        progress = self.currentPos / self.length
        return np.array([self.startMood, self.endMood, progress], dtype=np.float32)

    def act(self):
        return np.random.choice([0, 1, 2])

    def step(self, env):
        state = self.getState()
        action = self.act()
        next_state, reward, done, _ = env.step(action)
        self.playlist.append((state, action, reward))
        self.currentPos += 1
        return next_state, reward, done
