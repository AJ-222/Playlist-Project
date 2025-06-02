import pandas as pd
from env import MusicEnv
from randomAgent import RandomAgent
from favClusterAgent import FavouriteClusterAgent
from user import User

# Load song data
songs = pd.read_csv("clustered_songs.csv")
required_columns = ["Tempo", "Loudness", "Duration", "Key", "Mode",
                    "Year", "Hotttnesss", "AvgTimbre", "AvgPitches", "Cluster"]
for col in required_columns:
    songs[col] = pd.to_numeric(songs[col], errors="coerce")
songs = songs.dropna(subset=["Cluster"])

# Load cluster moods
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
songs["MoodDepth"]   = songs["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Depth", 0.5))

# Load mood dictionary
mood_df = pd.read_csv("moods.csv")
mood_dict = {
    row["Mood"]: [row["Valence"], row["Energy"], row["Depth"]]
    for _, row in mood_df.iterrows()
}

# Load all users
user_df = pd.read_csv("userArchetypes.csv")
agent_classes = [RandomAgent, FavouriteClusterAgent]

# Store results here
results = []

# Run for each user
for _, user_row in user_df.iterrows():
    user = User(
        name=user_row["archetype"],
        startMood=user_row["startMood"],
        endMood=user_row["endMood"],
        preferredCluster=int(user_row["favouriteCluster"]),
        mood_dict=mood_dict,
        favourite_titles=user_row["FavouriteSongs"].split(";") if pd.notna(user_row["FavouriteSongs"]) else []
    )

    for agent_cls in agent_classes:
        rewards = []
        for run in range(100):
            agent = agent_cls(songs, user)
            env = MusicEnv(songs, user)
            state = agent.reset()
            env.reset()
            total_reward = 0

            for t in range(env.length):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward

            rewards.append(total_reward)

        avg_reward = sum(rewards) / len(rewards)
        results.append({
            "User": user.name,
            "StartMood": user.startMood,
            "EndMood": user.endMood,
            "Agent": agent_cls.__name__,
            "AvgReward": round(avg_reward, 2)
        })

# Save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("agent_comparison_results.csv", index=False)
print("\n✅ Results written to 'agent_comparison_results.csv'")
