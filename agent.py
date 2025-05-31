import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

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


def plot_pca_clusters(songs, title="🎼 PCA Visualization of Clusters"):
    features = songs[["AvgTimbre", "AvgPitches"]].fillna(0)
    scaled = StandardScaler().fit_transform(features)

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
plot_pca_clusters(songs, "🎶 PCA Visualization of Music Clusters")