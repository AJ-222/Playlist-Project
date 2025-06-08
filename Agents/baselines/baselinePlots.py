import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Load the CSV file
df = pd.read_csv("randomResults.csv")

# Create a label for plotting
df['Label'] = df['User']
df_sorted = df.sort_values('User')  # Sort by agent, then user if needed

# Color mapping: -10 = red, 0 = yellow, +10 = green
norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
cmap = cm.RdYlGn
colors = cmap(norm(df_sorted['AvgReward']))

# Plot
fig, ax = plt.subplots(figsize=(14, 9))
bars = ax.barh(df_sorted['Label'], df_sorted['AvgReward'], color=colors)

# Formatting
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel("Average Reward", fontsize=12)
ax.set_title("Average Performance of Random Agent", fontsize=16)
ax.grid(axis='x', linestyle=':', linewidth=0.5)

# Axis limits with a bit of padding
ax.set_xlim(-11, 11)

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Average Reward (–10 = Red, 0 = Yellow, +10 = Green)', fontsize=10)

# Improve spacing
plt.tight_layout()
plt.show()
