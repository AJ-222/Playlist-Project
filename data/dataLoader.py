import pandas as pd
import numpy as np
from data.user import User

class MusicDataset:
    def __init__(self,
                 song_path="data/clustered_songs.csv",
                 mood_path="data/moods.csv",
                 cluster_mood_path="data/cluster_to_mood.csv",
                 user_archetype_path="data/userArchetypes.csv"):

        self.songs = pd.read_csv(song_path)
        self.songs.dropna(subset=["Title"], inplace=True)
        self.mood_df = pd.read_csv(mood_path)
        self.cluster_moods = pd.read_csv(cluster_mood_path)
        self.user_df = pd.read_csv(user_archetype_path)

        self._preprocess_songs()
        self._build_mood_dict()

    def _preprocess_songs(self):
        required_columns = ["Tempo", "Loudness", "Duration", "Key", "Mode",
                             "Year", "Hotttnesss", "AvgTimbre", "AvgPitches", "Cluster"]
        for col in required_columns:
            self.songs[col] = pd.to_numeric(self.songs[col], errors="coerce")
        self.songs.dropna(subset=["Cluster"], inplace=True)

        cluster_to_mood = {
            row["Cluster"]: {
                "Valence": row["Valence"],
                "Energy": row["Energy"],
                "Depth": row["Depth"]
            } for _, row in self.cluster_moods.iterrows()
        }

        self.songs["MoodValence"] = self.songs["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Valence", 0.5))
        self.songs["MoodEnergy"] = self.songs["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Energy", 0.5))
        self.songs["MoodDepth"] = self.songs["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Depth", 0.5))

        def parse_tags(tag_string):
            if pd.isna(tag_string):
                return {}
            try:
                return {
                    term.split(":")[0].strip().lower(): float(term.split(":")[1])
                    for term in tag_string.split(",")
                }
            except:
                return {}

        self.songs["ParsedTags"] = self.songs["FilteredTerms"].apply(parse_tags)

    def _build_mood_dict(self):
        self.mood_dict = {
            row["Mood"]: [row["Valence"], row["Energy"], row["Depth"]]
            for _, row in self.mood_df.iterrows()
        }

    def get_user(self, idx=0):
        row = self.user_df.iloc[idx]
        return User(
            name=row["archetype"],
            startMood=row["startMood"],
            endMood=row["endMood"],
            preferredCluster=int(row["favouriteCluster"]),
            mood_dict=self.mood_dict,
            favourite_titles=row["FavouriteSongs"].split(";") if pd.notna(row["FavouriteSongs"]) else []
        )
