import gym
from gym.spaces import Box, Discrete
import numpy as np
import pandas as pd
import random

class MusicEnv(gym.Env):
    def __init__(self, songs, user, length=10, verbose=False):
        super(MusicEnv, self).__init__()
        self.songs = songs.copy()
        self.user = user
        self.length = length
        self.verbose = verbose
        self.observation_space = Box(low=0, high=1, shape=(16,), dtype=np.float32)
        self.action_space = Discrete(8)

        self.songs["TagDict"] = self.songs["FilteredTerms"].map(self.parseTagDict)
        self.title_to_song = {
            row["Title"].strip().lower(): row
            for _, row in self.songs.iterrows()
            if isinstance(row["Title"], str)
        }

        self.reset()

    def parseTagDict(self, tag_str):
        try:
            return {
                term.split(":")[0].strip().lower(): float(term.split(":")[1])
                for term in str(tag_str).split(";") if ":" in term
            }
        except:
            return {}

    def reset(self):
        self.playlist = []
        self.currentPos = 0
        self.cacheFavouriteTags()
        self.fav_vecs = self.getFavouriteMoodVecs()
        return self.getState()

    def cacheFavouriteTags(self):
        self.fav_tags = set()
        for title in self.user.favourite_titles:
            song = self.title_to_song.get(title.strip().lower(), None)
            if song is not None:
                tags = song.get("TagDict", {})
                self.fav_tags.update(tags.keys())

    def getState(self):
        progress = self.currentPos / self.length
        return np.concatenate([
            self.user.startMoodVec,
            self.user.endMoodVec,
            [progress],
            self.fav_vecs
        ]).astype(np.float32)

    def tag_score(self, tag_dict):
        return sum(weight for tag, weight in tag_dict.items() if tag in self.fav_tags)

    def selectSong(self, action):
        cluster_songs = self.songs[self.songs["Cluster"] == action].copy()
        cluster_songs["SimilarityScore"] = cluster_songs["TagDict"].map(self.tag_score)

        if not cluster_songs.empty and cluster_songs["SimilarityScore"].max() > 0:
            song = cluster_songs.sort_values("SimilarityScore", ascending=False).iloc[0]
        elif not cluster_songs.empty:
            song = cluster_songs.sample(1).iloc[0]
        else:
            song = self.songs.sample(1).iloc[0]
        return song

    def step(self, action):
        song = self.selectSong(action)
        self.playlist.append(song)
        reward = self.calculateReward(song)
        self.currentPos += 1
        done = self.currentPos >= self.length
        return self.getState(), reward, done, {}

    def calculateReward(self, song):
        reward, feedback = self.simulateUser(song)
        if self.verbose:
            print(f"[Feedback] {song['Title']} by {song['Artist']} → {feedback}")
        return reward

    def simulateUser(self, song):
        mood_gradient = np.linspace(self.user.startMoodVec[0], self.user.endMoodVec[0], self.length)
        target_valence = mood_gradient[self.currentPos]
        mood_diff = abs(song["MoodValence"] - target_valence)
        cluster_match = song["Cluster"] == self.user.preferredCluster
        mood_alignment_bonus = max(0, 1 - mood_diff) * 2

        if mood_diff > 0.5 or (not cluster_match and random.random() < 0.5):
            skipped = random.random() < 0.8
        else:
            skipped = random.random() < 0.2

        if skipped:
            reward = -10 if random.random() < 0.5 else -5
            feedback = "Skipped before halfway -10" if reward == -10 else "Skipped after halfway -5"
        else:
            emotion_roll = random.random()
            if mood_diff < 0.2 and cluster_match and emotion_roll < 0.7:
                reward, feedback = 10, "Liked the song +10"
            elif mood_diff > 0.4 and emotion_roll < 0.5:
                reward, feedback = -2.5, "Disliked the song -2.5"
            elif emotion_roll < 0.3:
                reward, feedback = 1.0, "Neutral after full listen +1"
            else:
                reward, feedback = 0.5, "No clear reaction +0.5"

        reward += mood_alignment_bonus
        return reward, feedback

    def getFavouriteMoodVecs(self):
        moods = []
        for title in self.user.favourite_titles:
            song = self.title_to_song.get(title.strip().lower(), None)
            if song is not None:
                moods.append([
                    song["MoodValence"],
                    song["MoodEnergy"],
                    song["MoodDepth"]
                ])
            else:
                moods.append([0.5, 0.5, 0.5])
        return np.array(moods).flatten()

    def render(self):
        print("\nFinal Playlist:")
        for i, song in enumerate(self.playlist):
            print(f"{i+1}. {song['Title']} by {song['Artist']} (MoodValence: {song['MoodValence']:.2f}, Cluster: {song['Cluster']})")
