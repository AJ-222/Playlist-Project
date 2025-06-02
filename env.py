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
        #(startMood) + (endMood) + (progress) + (3 moods × 3 features)
        self.observation_space = Box(low=0, high=1, shape=(16,), dtype=np.float32)
        self.fav_tags = set() 
        self.action_space = Discrete(8)
        self.reset()

    def reset(self):
        self.playlist = []
        self.currentPos = 0
        self.cacheFavouriteTags()
        return self.getState()
    
    
    def cacheFavouriteTags(self):
        self.fav_tags = set()
        #print("\n🎵 Caching Favourite Song Tags:")
        
        for title in self.user.favourite_titles:
            # Normalize both sides for reliable matching
            matches = self.songs[self.songs["Title"].str.strip().str.lower() == title.strip().lower()]
            
            if not matches.empty:
                song_row = matches.iloc[0]
                raw_terms = str(song_row.get("FilteredTerms", ""))
                tags = [term.split(":")[0].strip().lower() for term in raw_terms.split(";") if ":" in term]
                
#                if tags:
 #                   print(f" - {title}:")
  #                  for tag in tags[:10]:
   #                     print(f"     • {tag}")
    #            else:
     #               print(f" - {title}: (no tags found)")
      #              
       #         self.fav_tags.update(tags)
        #    else:
                #print(f" - {title}: ⚠️ Not found in dataset.")


    def getState(self):
        progress = self.currentPos / self.length
        favs = self.getFavouriteMoodVecs()  # Shape: (9,)
        return np.concatenate([
            self.user.startMoodVec,    
            self.user.endMoodVec,       
            [progress],                 
            favs                        
        ]).astype(np.float32)

    def step(self, action):
        song = self.selectSong(action)
        self.playlist.append(song)
        
        reward = self.calculateReward(song)
        self.currentPos += 1
        done = self.currentPos >= self.length
        return self.getState(), reward, done, {}

    def selectSong(self, action):
        cluster_songs = self.songs[self.songs["Cluster"] == action].copy()

        def tag_similarity(song_terms):
            tag_dict = {
                term.split(":")[0].strip().lower(): float(term.split(":")[1])
                for term in str(song_terms).split(";") if ":" in term
            }
            return sum(weight for tag, weight in tag_dict.items() if tag in self.fav_tags)


        cluster_songs["SimilarityScore"] = cluster_songs["FilteredTerms"].apply(tag_similarity)

        if not cluster_songs.empty and cluster_songs["SimilarityScore"].max() > 0:
            song = cluster_songs.sort_values("SimilarityScore", ascending=False).iloc[0]
            #print(f"\n🎯 Selected from Cluster {action}: {song['Title']} by {song['Artist']}")
            #print(f"   ➤ Similarity Score: {song['SimilarityScore']}")
            return song
        elif not cluster_songs.empty:
            return cluster_songs.sample(1).iloc[0]
        else:
            return self.songs.sample(1).iloc[0]





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
    def getFavouriteMoodVecs(self):
        moods = []
        for title in self.user.favourite_titles:  # List of 3 song titles
            matching_song = self.songs[self.songs["Title"] == title]
            if not matching_song.empty:
                song = matching_song.iloc[0]
                moods.append([
                    song["MoodValence"],
                    song["MoodEnergy"],
                    song["MoodDepth"]
                ])
            else:
                # If the song is not found, use a neutral fallback
                moods.append([0.5, 0.5, 0.5])
        return np.array(moods).flatten()




    def render(self):
        print("\nFinal Playlist:")
        for i, song in enumerate(self.playlist):
            print(f"{i+1}. {song['Title']} by {song['Artist']} (MoodValence: {song['MoodValence']:.2f}, Cluster: {song['Cluster']})")