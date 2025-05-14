import numpy as np
import matplotlib.pyplot as plt
import os
from env.gridworld_env import Action, TileType
from env.gridworld_mdp import GridWorldMDP

ARROWS = {
    Action.UP: '↑',
    Action.DOWN: '↓',
    Action.LEFT: '←',
    Action.RIGHT: '→'
}

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

    os.makedirs("outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"outputs/{prefix}_iteration_{iteration}.png")
    plt.close()

def value_iteration(mdp, gamma=0.99, theta=1e-4):
    V = {s: 0 for s in mdp.states}
    policy = {s: Action.UP for s in mdp.states}
    iteration = 0

    while True:
        delta = 0
        new_V = V.copy()
        for s in mdp.states:
            max_value = float('-inf')
            best_action = None
            for a in mdp.actions:
                next_s, reward, done = mdp.get_transition(s, a)
                value = reward + gamma * (0 if done else V[next_s])
                if value > max_value:
                    max_value = value
                    best_action = a
            new_V[s] = max_value
            policy[s] = best_action
            delta = max(delta, abs(V[s] - new_V[s]))

        V = new_V
        plot_value_and_policy(V, policy, mdp.env.grid, iteration, mdp.width, mdp.height, prefix='vi')
        iteration += 1
        if delta < theta:
            break

    plot_value_and_policy(V, policy, mdp.env.grid, 'final', mdp.width, mdp.height, prefix='vi')
    return V, policy

def policy_evaluation(policy, mdp, gamma=0.99, theta=1e-4):
    V = {s: 0 for s in mdp.states}
    while True:
        delta = 0
        for s in mdp.states:
            a = policy[s]
            next_s, reward, done = mdp.get_transition(s, a)
            V_new = reward + gamma * V.get(next_s, 0) * (not done)
            delta = max(delta, abs(V[s] - V_new))
            V[s] = V_new
        if delta < theta:
            break
    return V

def policy_improvement(V, mdp, gamma=0.99):
    policy_stable = True
    policy = {}
    for s in mdp.states:
        old_action = policy.get(s)
        action_values = {}
        for a in mdp.actions:
            next_s, reward, done = mdp.get_transition(s, a)
            action_values[a] = reward + gamma * V.get(next_s, 0) * (not done)
        best_action = max(action_values, key=action_values.get)
        policy[s] = best_action
        if old_action is not None and old_action != best_action:
            policy_stable = False
    return policy, policy_stable

def policy_iteration(mdp, gamma=0.99):
    policy = {s: np.random.choice(mdp.actions) for s in mdp.states}
    iteration = 0

    while True:
        V = policy_evaluation(policy, mdp, gamma)
        old_policy = policy.copy()  # <-- 반드시 이전 정책을 저장해야 함
        policy, _ = policy_improvement(V, mdp, gamma)

        # 수렴 여부 비교
        policy_stable = all(policy[s] == old_policy[s] for s in mdp.states)

        plot_value_and_policy(V, policy, mdp.env.grid, iteration, mdp.width, mdp.height, prefix='pi')
        print(f"Policy Iteration {iteration} completed")

        if policy_stable:
            break
        iteration += 1

    plot_value_and_policy(V, policy, mdp.env.grid, 'final', mdp.width, mdp.height, prefix='pi')
    return V, policy

