import pandas as pd
from env import MusicEnv
from randomAgent import RandomAgent
from favClusterAgent import FavouriteClusterAgent  # Make sure this file/class exists
from user import User

# Load songs
songs = pd.read_csv("clustered_songs.csv")

# Ensure required columns are numeric and clean
required_columns = ["Tempo", "Loudness", "Duration", "Key", "Mode",
                    "Year", "Hotttnesss", "AvgTimbre", "AvgPitches", "Cluster"]
for col in required_columns:
    songs[col] = pd.to_numeric(songs[col], errors="coerce")
songs = songs.dropna(subset=["Cluster"])  # Ensure Cluster exists

# Load mood info per cluster
cluster_moods = pd.read_csv("cluster_to_mood.csv")
cluster_to_mood = {
    row["Cluster"]: {
        "Valence": row["Valence"],
        "Energy": row["Energy"],
        "Depth": row["Depth"]
    }
    for _, row in cluster_moods.iterrows()
}
songs["MoodValence"] = songs["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Valence", 0.5))
songs["MoodEnergy"]  = songs["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Energy", 0.5))
songs["MoodDepth"]   = songs["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Depth",  0.5))

# Load user archetypes
user_df = pd.read_csv("userArchetypes.csv")
mood_df = pd.read_csv("moods.csv")
mood_dict = {
    row["Mood"]: [row["Valence"], row["Energy"], row["Depth"]]
    for _, row in mood_df.iterrows()
}

# Pick a single user archetype
user_row = user_df.iloc[0]
user = User(
    name=user_row["archetype"],
    startMood=user_row["startMood"],
    endMood=user_row["endMood"],
    preferredCluster=int(user_row["favouriteCluster"]),
    mood_dict=mood_dict,
    favourite_titles=user_row["FavouriteSongs"].split(";") if pd.notna(user_row["FavouriteSongs"]) else []
)

# List of agents to test
agent_classes = [RandomAgent, FavouriteClusterAgent]

for agent_cls in agent_classes:
    print("\n" + "="*60)
    agent = agent_cls(songs, user)
    env = MusicEnv(songs, user)

    print(f"\n👤 User: {user.name}")
    print(f"🎯 Agent: {agent.name}")
    print(f"🌈 Start Mood: {user.startMood} → End Mood: {user.endMood}\n")

    state = agent.reset()
    env.reset()
    total_reward = 0

    for t in range(env.length):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        print(f"Step {t+1}: Action={action}, Reward={reward:.2f}")

    print(f"\n🎵 Total Reward for {agent.name}: {total_reward:.2f}")
    print("="*60)
