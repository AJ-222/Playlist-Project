import torch
import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from env import MusicEnv
from Agents.deepQAgent.dQAgent import DQNAgent
from user import User

# Load data
songs = pd.read_csv("finalData.csv")
user_df = pd.read_csv("userArchetypes.csv")
mood_df = pd.read_csv("moods.csv")
mood_dict = {
    row["Mood"]: [row["Valence"], row["Energy"], row["Depth"]]
    for _, row in mood_df.iterrows()
}


# Preprocess song mood features (already done in your data)
required_cols = ["MoodValence", "MoodEnergy", "MoodDepth", "Cluster"]
for col in required_cols:
    songs[col] = pd.to_numeric(songs[col], errors="coerce")
songs.dropna(subset=required_cols, inplace=True)

# Select ONE archetype
user_row = user_df.iloc[0]
user = User(
    name=user_row["archetype"],
    startMood=user_row["startMood"],
    endMood=user_row["endMood"],
    preferredCluster=int(user_row["favouriteCluster"]),
    mood_dict=mood_dict,
    favourite_titles=user_row.get("FavouriteSongs", "").split(";") if pd.notna(user_row.get("FavouriteSongs")) else []
)

# Environment and Agent
env = MusicEnv(songs, user)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Training Parameters
episodes = 10000
batch_size = 32
target_update_freq = 20
reward_window = []
average_rewards = []

# Live Plot
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label="Avg Reward (100)", color='orange')
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_title("DQN Music Agent Training Progress")
ax.legend()
ax.grid(True)

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for t in range(env.length):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break

    agent.replay(batch_size)
    if episode % target_update_freq == 0:
        agent.update_target_network()

    reward_window.append(total_reward)
    avg_reward = np.mean(reward_window[-100:])
    average_rewards.append(avg_reward)

    # Live plot update
    line.set_xdata(np.arange(len(average_rewards)))
    line.set_ydata(average_rewards)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

    if (episode + 1) % 100 == 0:
        print(f"Ep {episode+1}/{episodes} | Total: {total_reward:.2f} | Avg(100): {avg_reward:.2f} | Eps: {agent.epsilon:.3f}")

# Final save
plt.ioff()
plt.savefig("dqn_training_plot.png")

# Export rewards
pd.DataFrame({
    "Episode": np.arange(1, episodes + 1),
    "Reward": reward_window,
    "AvgReward100": pd.Series(average_rewards)
}).to_csv("dqn_training_rewards.csv", index=False)
