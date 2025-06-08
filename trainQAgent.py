import pandas as pd
from collections import deque
import random
import numpy as np
from env import MusicEnv
from qLearnAgent import QLearningAgent
from data.user import User
import matplotlib.pyplot as plt
from data.dataLoader import MusicDataset

data = MusicDataset()
songs = data.songs
user = data.get_user(idx=0)  # Always use first archetype (or change index)

# Initialize environment and agent
env = MusicEnv(songs, user)
agent = QLearningAgent(n_actions=8, state_size=16)  # Assuming 8 clusters and 16-dim state

# Training loop
episodes = 10000
epsilonDecay = 0.999
minEpsilon = 0.01
reward_history = []

for episode in range(episodes):
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
    if (episode + 1) % 100 == 0 or episode == 0:
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

        

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
