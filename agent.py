import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def plot_elbow_curve(songs, max_k=15):
    features = songs[["AvgTimbre", "AvgPitches"]].fillna(0)
    scaled = StandardScaler().fit_transform(features)
    
    inertias = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-cluster SSE)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.show()





def kMeans(songs, n_clusters=10):
    features = songs[["AvgTimbre", "AvgPitches"]].fillna(0)

    scaled = StandardScaler().fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    songs["Cluster"] = kmeans.fit_predict(scaled)
    
    return songs, kmeans

def parseSongsfromCSV(path="track_raw_features.csv"):
    return pd.read_csv(path)

def preview_clusters(songs, n_samples=5):
    grouped = songs.groupby("Cluster")
    for cluster_id, group in grouped:
        print(f"\nCluster {cluster_id} — {len(group)} songs")
        sample = group.sample(min(n_samples, len(group)), random_state=42)
        for _, row in sample.iterrows():
            print(f"{row['Title']} by {row['Artist']} (Energy: {row['Energy']:.2f}, Danceability: {row['Danceability']:.2f})")

songs, _ = kMeans(parseSongsfromCSV())
preview_clusters(songs)
plot_elbow_curve(songs) 