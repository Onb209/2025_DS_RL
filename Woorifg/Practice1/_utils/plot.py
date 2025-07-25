import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from env.gridworld_env import Action, TileType

ARROWS = {
    Action.UP: '↑',
    Action.DOWN: '↓',
    Action.LEFT: '←',
    Action.RIGHT: '→'
}

if os.path.exists("outputs"):
    shutil.rmtree("outputs")
os.makedirs("outputs", exist_ok=True)

# 매 iteration마다 value 값과 policy를 plot 
def plot_value_and_policy(V, policy, grid, iteration, width, height, prefix='vi'):
    value_grid = np.full((height, width), np.nan)
    policy_grid = np.full((height, width), '', dtype=object)

    for (y, x), v in V.items():
        value_grid[y][x] = v
        policy_grid[y][x] = ARROWS[policy[(y, x)]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im = axes[0].imshow(value_grid, cmap='coolwarm', interpolation='nearest')
    for y in range(height):
        for x in range(width):
            if not np.isnan(value_grid[y, x]):
                axes[0].text(x, y, f"{value_grid[y, x]:.1f}", ha='center', va='center', color='black')
    axes[0].set_title(f"Value Function - Iteration {iteration}")

    axes[1].imshow(np.ones_like(value_grid), cmap='gray', vmin=0, vmax=1)
    for y in range(height):
        for x in range(width):
            if policy_grid[y][x]:
                axes[1].text(x, y, policy_grid[y][x], ha='center', va='center', fontsize=16)
            if grid[y][x] == TileType.WALL:
                axes[1].add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black'))
            elif grid[y][x] == TileType.TRAP:
                axes[1].add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='red'))
            elif grid[y][x] == TileType.GOAL:
                axes[1].add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='green'))
    axes[1].set_title(f"Policy - Iteration {iteration}")

    for ax in axes:
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(f"outputs/{prefix}_iteration_{iteration}.png")
    plt.close()
