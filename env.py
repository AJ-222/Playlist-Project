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

        self.observation_space = Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = Discrete(3)
        self.reset()
    def reset(self):
        self.playlist = []
        self.currentPos = 0
        self.startMood = self.user.get("startMood",0.2)
        self.endMood = self.user.get("endMood",0.8)
        return self.getState()
    
    def getState(self):
        progress = self.currentPos / self.length
        return np.array([
            self.startMood,
            self.endMood,
            progress,
        ], dtype=np.float32)	

    def step(self, action):
        song = self.selectSong(action)
        self.playlist.append(song)
        
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

        
    def calculateReward(self, song):
        reward, feedback = self.simulateUser(song)
        print(f"[Feedback] {song['Title']} by {song['Artist']} → {feedback}")
        return reward

    def simulateUser(self, song):
        target_mood = np.linspace(self.startMood, self.endMood, self.length)[self.currentPos]
        mood_diff = abs(song["Mood"] - target_mood)

        cluster_match = (song["Cluster"] == self.user["preferredCluster"])

        if mood_diff > 0.5 or (not cluster_match and random.random() < 0.5):
            skipped = random.random() < 0.8 #probably gonna skip
        else:
            skipped = random.random() < 0.2

        if skipped:
            #skipped
            if random.random() < 0.5:
                return -1.0, "Skipped before halfway -1"
            else:
                return -0.5, "Skipped after halfway -0.5"
        else:
            #listened
            emotion_roll = random.random()
            if mood_diff < 0.2 and cluster_match:
                if emotion_roll < 0.7:
                    return 1.0, "Liked the song +1"
            if mood_diff > 0.4:
                if emotion_roll < 0.5:
                    return -0.25, "Disliked the song -0.25"
            if emotion_roll < 0.3:
                return 0.1, "Neutral after full listen +0.1"
            return 0.0, "No clear reaction 0"

    def render(self):
        print("\nFinal Playlist:")
        for i, song in enumerate(self.playlist):
            print(f"{i+1}. {song['Title']} by {song['Artist']} (Mood: {song['Mood']:.2f}, Cluster: {song['Cluster']})")
