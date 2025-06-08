import pandas as pd
from collections import deque
import random
import numpy as np
from env import MusicEnv
from qLearnAgent import QLearningAgent
from user import User
import matplotlib.pyplot as plt

# Load songs
df = pd.read_csv("clustered_songs.csv")
required_columns = ["Tempo", "Loudness", "Duration", "Key", "Mode",
                    "Year", "Hotttnesss", "AvgTimbre", "AvgPitches", "Cluster"]
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["Cluster"])

# Load mood features per cluster
cluster_moods = pd.read_csv("cluster_to_mood.csv")
cluster_to_mood = {
    row["Cluster"]: {
        "Valence": row["Valence"],
        "Energy": row["Energy"],
        "Depth": row["Depth"]
    } for _, row in cluster_moods.iterrows()
}
df["MoodValence"] = df["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Valence", 0.5))
df["MoodEnergy"]  = df["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Energy", 0.5))
df["MoodDepth"]   = df["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Depth",  0.5))

# Load user archetypes
user_df = pd.read_csv("userArchetypes.csv")
mood_df = pd.read_csv("moods.csv")
mood_dict = {
    row["Mood"]: [row["Valence"], row["Energy"], row["Depth"]]
    for _, row in mood_df.iterrows()
}

# Pick a random user
# Fix user selection: use only 1 archetype consistently
user_row = user_df.iloc[0]
user = User(
    name=user_row["archetype"],
    startMood=user_row["startMood"],
    endMood=user_row["endMood"],
    preferredCluster=int(user_row["favouriteCluster"]),
    mood_dict=mood_dict,
    favourite_titles=user_row["FavouriteSongs"].split(";") if pd.notna(user_row["FavouriteSongs"]) else []
)


# Initialize environment and agent
env = MusicEnv(df, user)
agent = QLearningAgent(n_actions=8, state_size=16)  # Assuming 8 clusters and 16-dim state

# Training loop
episodes = 10000
epsilonDecay = 0.999
minEpsilon = 0.01
reward_history = []

for episode in range(episodes):
    env = MusicEnv(df, user)
    state = env.reset()
    total_reward = 0

    for t in range(env.length):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    reward_history.append(total_reward)
    agent.epsilon = max(agent.epsilon * 0.995, 0.01)  # Exploration decay
        

pd.DataFrame({
    "Episode": list(range(1, len(reward_history) + 1)),
    "TotalReward": reward_history
}).to_csv("qlearning_training_rewards_10000.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(range(len(reward_history)), reward_history, label="Total reward per episode")
plt.title("Q-Learning Progress Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("qlearning_training_plot.png")
plt.show()
