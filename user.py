import numpy as np
import pandas as pd
class User:
    def __init__(self, name, startMood, endMood, preferredCluster, mood_dict):
        self.name = name
        self.startMood = startMood
        self.endMood = endMood
        self.preferredCluster = preferredCluster

        self.startMoodVec = np.array(mood_dict[startMood])
        self.endMoodVec = np.array(mood_dict[endMood])

    def get_profile(self):
        return {
            "name": self.name,
            "startMood": self.startMood,
            "endMood": self.endMood,
            "preferredCluster": self.preferredCluster
        }
