import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the clustered songs CSV
df = pd.read_csv("clustered_songs.csv")

# Define genre labels for each cluster
genre_labels = {
    0: "Popular Classics",
    1: "Soul & Jazz",
    2: "EDM",
    3: "Indie Rock",
    4: "Folk",
    5: "Latin Pop",
    6: "Mainstream Pop",
    7: "Chill & Lo-Fi"
}

# Add genre label to dataframe
df["GenreLabel"] = df["Cluster"].map(genre_labels)

# Features used for clustering
CLUSTER_FEATURES = [
    "Tempo", "Loudness", "Duration", "Key", "Mode",
    "Year", "Hotttnesss", "AvgTimbre", "AvgPitches"
]

# Refit scaler and KMeans to get cluster centers
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[CLUSTER_FEATURES])
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(scaled_features)

# Convert cluster centers back to original scale
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers_original, columns=CLUSTER_FEATURES)
centers_df["Cluster"] = range(8)
centers_df["GenreLabel"] = centers_df["Cluster"].map(genre_labels)

df.to_csv("clustered_songs_with_genres.csv", index=False)

# Write the cluster center feature values with genre labels
centers_df.to_csv("cluster_centers_with_genres.csv", index=False)

print("✅ Files saved: 'clustered_songs_with_genres.csv' and 'cluster_centers_with_genres.csv'")