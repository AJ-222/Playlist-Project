import numpy as np
import pandas as pd

class FavouriteClusterAgent:
    def __init__(self, songs, user, length=10):
        self.songs = songs
        self.user = user
        self.name = "FavouriteClusterAgent"
        self.length = length
        self.playlist = []
        self.currentPos = 0

    def reset(self):
        self.playlist = []
        self.currentPos = 0
        self.startMood = self.user.startMood
        self.endMood = self.user.endMood
        self.startMoodVec = self.user.startMoodVec
        self.endMoodVec = self.user.endMoodVec
        return self.getState()

    def getState(self):
        progress = self.currentPos / self.length
        return np.array([
            *self.user.startMoodVec,
            *self.user.endMoodVec,
            progress
        ], dtype=np.float32)

    def act(self, state=None):
        return self.user.preferredCluster  # Always select from the preferred cluster

    def step(self, env):
        state = self.getState()
        action = self.act(state)
        next_state, reward, done, _ = env.step(action)
        self.playlist.append((state, action, reward))
        self.currentPos += 1
        return next_state, reward, done
