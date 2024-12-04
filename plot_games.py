import matplotlib.pyplot as plt
import glob
import numpy as np
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# Get all game*.txt files
game_files = sorted(glob.glob('game*.txt'), key=natural_sort_key)

plt.figure(figsize=(12, 8))

# Plot each game file
for game_file in game_files:
    # Read the rewards from the file
    with open(game_file, 'r') as f:
        rewards = [float(line.strip()) for line in f if line.strip()]
    
    # Only take first 200 episodes
    rewards = rewards[:200]
    episodes = range(len(rewards))
    
    # Plot this game's rewards
    plt.plot(episodes, rewards, label=game_file.replace('.txt', ''))

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress Across Different Games')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('games_comparison.png')
plt.show()
