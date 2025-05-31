# mood_mapping.py
import numpy as np
cluster_to_mood = {
    0: {"Mood": "Relaxed", "Valence": 0.7, "Energy": 0.2, "Depth": 0.2},
    1: {"Mood": "Focused", "Valence": 0.5, "Energy": 0.4, "Depth": 0.4},
    2: {"Mood": "Sad", "Valence": 0.1, "Energy": 0.2, "Depth": 0.9},
    3: {"Mood": "Nostalgic", "Valence": 0.4, "Energy": 0.3, "Depth": 1.0},
    4: {"Mood": "Romantic", "Valence": 0.6, "Energy": 0.5, "Depth": 0.8},
    5: {"Mood": "Happy", "Valence": 0.9, "Energy": 0.7, "Depth": 0.4},
    6: {"Mood": "Inspired", "Valence": 0.8, "Energy": 0.8, "Depth": 0.7},
    7: {"Mood": "Energetic", "Valence": 0.6, "Energy": 0.9, "Depth": 0.25},
}

# Optionally, inverse mapping for mood name to float vector
mood_to_vector = {v["Mood"]: {"Valence": v["Valence"], "Energy": v["Energy"], "Depth": v["Depth"]}
                  for v in cluster_to_mood.values()}
