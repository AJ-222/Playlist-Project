import pandas as pd
from env import MusicEnv
from randomAgent import RandomAgent
from user import User
from moodmapping import cluster_to_mood
# Load songs
songs = pd.read_csv("finalData.csv")

# Ensure required columns are numeric and clean
required_columns = ["Tempo", "Loudness", "Duration", "Key", "Mode",
                    "Year", "Hotttnesss", "AvgTimbre", "AvgPitches", "Cluster"]
for col in required_columns:
    songs[col] = pd.to_numeric(songs[col], errors="coerce")
songs = songs.dropna(subset=["Cluster"])  # Make sure Cluster exists

# Assign mood info from cluster
songs["MoodValence"] = songs["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Valence", 0.5))
songs["MoodEnergy"] = songs["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Energy", 0.5))
songs["MoodDepth"]   = songs["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Depth",  0.5))

# Load user archetypes
user_df = pd.read_csv("userArchetypes.csv")
mood_df = pd.read_csv("moods.csv")
mood_dict = {
    row["Mood"]: [row["Valence"], row["Energy"], row["Depth"]]
    for _, row in mood_df.iterrows()
}

# Pick a user archetype (just the first one for now)
user_row = user_df.iloc[0]
user = User(
    name=user_row["archetype"],
    startMood=user_row["startMood"],
    endMood=user_row["endMood"],
    preferredCluster=int(user_row["favouriteCluster"]),
    mood_dict=mood_dict  # You need to make sure this is loaded above
)

# Init environment
env = MusicEnv(songs, user)
agent = RandomAgent(songs, user)

# Run one episode
state = agent.reset()
env.reset()

print(f"\nUser: {user.name}...\n")
print(f"Start Mood: {user.startMood} → End Mood: {user.endMood}")
print(f"\Agent: {agent.name}...\n")
total_reward = 0
for t in range(env.length):
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward
    print(f"Step {t + 1}: Action={action}, Reward={reward:.2f}")

#print("\nPlaylist:")
#for i, song in enumerate(env.playlist, 1):
#    print(f"{i}. {song['Title']} by {song['Artist']}")

print(f"\nTotal Reward: {total_reward:.2f}")
