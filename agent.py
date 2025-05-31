import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def kMeans(songs):
    features = songs[[f"Timbre_{i}" for i in range(12)] + [f"Pitch_{i}" for i in range(12)]]
    scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=10, random_state=42)
    songs["Cluster"] = kmeans.fit_predict(scaled)
    return songs

def parseSongsfromCSV():
    songs = pd.read_csv("track_raw_features.csv")
    return songs

