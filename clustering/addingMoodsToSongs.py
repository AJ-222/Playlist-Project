import pandas as pd

# Load the CSV file that already includes the 'Cluster' column
songs = pd.read_csv("clustering/dataWithClusters.csv")

# Arbitrary mapping from cluster number to mood score (0 = calm, 1 = energetic)
cluster_to_mood = {
    0: 0.2,
    1: 0.7,
    2: 0.4,
    3: 0.9,
    4: 0.3,
    5: 0.8,
    6: 0.6,
    7: 0.1,
}

# Apply the mapping to assign a mood score to each song
songs["Mood"] = songs["Cluster"].map(cluster_to_mood)

# Optionally save to a new file
songs.to_csv("track_features_with_mood.csv", index=False)
