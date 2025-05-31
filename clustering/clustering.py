import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

CLUSTER_FEATURES = [
    "Tempo", "Loudness", "Duration", "Key", "Mode",
    "Year", "Hotttnesss", "AvgTimbre", "AvgPitches"
]

def parseSongsfromCSV(path="track_raw_features.csv"):
    songs = pd.read_csv(path)
    for col in CLUSTER_FEATURES:
        songs[col] = pd.to_numeric(songs[col], errors="coerce")
    songs[CLUSTER_FEATURES] = songs[CLUSTER_FEATURES].fillna(songs[CLUSTER_FEATURES].mean())
    return songs

def kMeans(songs, n_clusters=10):
    scaled = StandardScaler().fit_transform(songs[CLUSTER_FEATURES])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    songs["Cluster"] = kmeans.fit_predict(scaled)
    return songs, kmeans

def plot_elbow_curve(songs, max_k=15):
    scaled = StandardScaler().fit_transform(songs[CLUSTER_FEATURES])
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

def plot_pca_clusters(songs, title="🎼 PCA Visualization of Clusters"):
    scaled = StandardScaler().fit_transform(songs[CLUSTER_FEATURES])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)

    songs["PCA1"] = pca_result[:, 0]
    songs["PCA2"] = pca_result[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=songs, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=60, edgecolor="k")
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def preview_clusters(songs, n_samples=5):
    grouped = songs.groupby("Cluster")
    for cluster_id, group in grouped:
        print(f"\nCluster {cluster_id} — {len(group)} songs")
        sample = group.sample(min(n_samples, len(group)), random_state=42)
        for _, row in sample.iterrows():
            print(f"{row['Title']} by {row['Artist']} (Hotttnesss: {row['Hotttnesss']:.2f})")

# Run the process
songs = parseSongsfromCSV()
songs, _ = kMeans(songs, n_clusters=8)
preview_clusters(songs)
plot_elbow_curve(songs)
plot_pca_clusters(songs, "🎶 PCA Visualization of Music Clusters")
songs.to_csv("clustered_songs.csv", index=False)
print("✅ Clustered songs exported to 'clustered_songs.csv'")