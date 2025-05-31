import gym
from gym.spaces import Box, Discrete
import numpy as np
import pandas as pd
import random

class MusicEnv(gym.Env):
    def __init__(self, songs, user, length=10):
        super(MusicEnv, self).__init__()

        self.songs = songs
        self.user = user
        self.length = length

        self.observation_space = Box(low=0, high=1, shape=(5), dtype=np.float32)
        self.action_space = Discrete(3)
        self.reset()
    def reset(self):
        self.playlist = []
        self.currentPos = 0
        self.songEnergy = []
        self.songDanceability = []

        self.startMood = self.user.get("startMood",0.2)
        self.endMood = self.user.get("endMood",0.8)
        return self.getState()
    
    def getState(self):
        avgEnergy = np.mean(self.songEnergy) if self.songEnergy else 0
        avgDanceability = np.mean(self.songDanceability) if self.songDanceability else 0
        progress = self.currentPos / self.length
        return np.array([
            self.startMood,
            self.endMood,
            progress,
            avgEnergy,
            avgDanceability
        ], dtype=np.float32)	

    def step(self, action):
        song = self.selectSong(action)
        self.playlist.append(song)
        self.songEnergy.append(song["Energy"])
        self.songDanceability.append(song["Danceability"])
        
        reward = self.calculateReward(song)
        self.currentPos += 1
        done = self.currentPos >= self.length
        return self.getState(), reward, done, {}

    def selectSong(self, action): #proxy action selection function
        if action == 0: #safe action
            return random.choice([
                s for s in self.songs
                if abs(s["Mood"] - self.startMood) < 0.1
                and s["Cluster"] == self.user["preferredCluster"]
            ])
        elif action == 1: #risky action
            return random.choice([
                s for s in self.songs
                if abs(s["Mood"] - self.startMood) < 0.1
                and s["Cluster"] != self.user["preferredCluster"]
            ])
        else:   #popular action
            return random.choice([
                s for s in self.songs
                if s["Hotttnesss"] > 0.7 and abs(s["Mood"] - self.startMood) < 0.2
            ])

        
    def calculateReward(self, song): #proxy reward function
        target_energy = np.linspace(self.startMood, self.endMood, self.length)[self.currentPos]
        return 1 - abs(song["Energy"] - target_energy)
    
    def simulateUser(self, song):
        pass