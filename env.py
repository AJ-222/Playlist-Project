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
        self.observation_space = Box(low=0, high=1, shape=(7,), dtype=np.float32)
        self.action_space = Discrete(8)
        self.reset()

    def reset(self):
        self.playlist = []
        self.currentPos = 0
        return self.getState()
    
    def getState(self):
        progress = self.currentPos / self.length
        return np.concatenate([
            self.user.startMoodVec,
            self.user.endMoodVec,
            [progress]
        ]).astype(np.float32)

    def step(self, action):
        song = self.selectSong(action)
        self.playlist.append(song)
        
        reward = self.calculateReward(song)
        self.currentPos += 1
        done = self.currentPos >= self.length
        return self.getState(), reward, done, {}

    def selectSong(self, action):
        valence_target = self.user.startMoodVec[0]

    # Select from the specific cluster corresponding to the action number
        candidates = [
            row for _, row in self.songs.iterrows()
            if row["Cluster"] == action and abs(row["MoodValence"] - valence_target) < 0.2
        ]

        return random.choice(candidates) if candidates else self.songs.sample(1).iloc[0]


    def calculateReward(self, song):
        reward, feedback = self.simulateUser(song)
        print(f"[Feedback] {song['Title']} by {song['Artist']} → {feedback}")
        return reward

    def simulateUser(self, song):
        # use linear interpolation in valence space only for simplicity
        mood_gradient = np.linspace(self.user.startMoodVec[0], self.user.endMoodVec[0], self.length)
        target_valence = mood_gradient[self.currentPos]
        mood_diff = abs(song["MoodValence"] - target_valence)

        cluster_match = song["Cluster"] == self.user.preferredCluster

        if mood_diff > 0.5 or (not cluster_match and random.random() < 0.5):
            skipped = random.random() < 0.8
        else:
            skipped = random.random() < 0.2

        if skipped:
            if random.random() < 0.5:
                return -1.0, "Skipped before halfway -1"
            else:
                return -0.5, "Skipped after halfway -0.5"
        else:
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
            print(f"{i+1}. {song['Title']} by {song['Artist']} (MoodValence: {song['MoodValence']:.2f}, Cluster: {song['Cluster']})")
