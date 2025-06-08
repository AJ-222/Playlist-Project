import pandas as pd
import numpy as np
from env import MusicEnv
from user import User
from dQAgent import DQNAgent  # Ensure this is your DQN agent implementation
import torch

# Load and prepare song data
songs_df = pd.read_csv("clustered_songs.csv")
cluster_moods = pd.read_csv("cluster_to_mood.csv")

cluster_to_mood = {
    row["Cluster"]: {
        "Valence": row["Valence"],
        "Energy": row["Energy"],
        "Depth": row["Depth"]
    } for _, row in cluster_moods.iterrows()
}

songs_df["MoodValence"] = songs_df["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Valence", 0.5))
songs_df["MoodEnergy"]  = songs_df["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Energy", 0.5))
songs_df["MoodDepth"]   = songs_df["Cluster"].map(lambda c: cluster_to_mood.get(c, {}).get("Depth",  0.5))

# Load user archetype
user_df = pd.read_csv("userArchetypes.csv")
user_row = user_df.iloc[0]  # Use only one archetype
mood_df = pd.read_csv("moods.csv")
mood_dict = {
    row["Mood"]: [row["Valence"], row["Energy"], row["Depth"]]
    for _, row in mood_df.iterrows()
}

user = User(
    name=user_row["archetype"],
    startMood=user_row["startMood"],
    endMood=user_row["endMood"],
    preferredCluster=int(user_row["favouriteCluster"]),
    mood_dict=mood_dict,
    favourite_titles=user_row["FavouriteSongs"].split(";") if pd.notna(user_row["FavouriteSongs"]) else []
)

# Environment and agent
env = MusicEnv(songs_df, user)
state_size = len(env.getState())
action_size = 8  # One per cluster
agent = DQNAgent(state_size, action_size)

# Training loop
episodes = 10000
batch_size = 32

rewards = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for t in range(env.length):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay(batch_size)
        state = next_state
        total_reward += reward
        if done:
            break

    rewards.append(total_reward)
    if ep % 20 == 0:
        agent.update_target_network()

    if (ep + 1) % 100 == 0:
        avg = np.mean(rewards[-100:])
        print(f"Episode {ep + 1} | Total Reward: {total_reward:.2f} | Avg(100): {avg:.2f} | Epsilon: {agent.epsilon:.3f}")

# Save results
pd.DataFrame({
    "Episode": np.arange(1, episodes + 1),
    "TotalReward": rewards
}).to_csv("dqn_training_rewards.csv", index=False)

print("Training complete. Rewards saved to dqn_training_rewards.csv")
