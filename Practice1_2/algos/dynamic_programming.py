import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from env.gridworld_env import Action, TileType
from env.gridworld_mdp import GridWorldMDP
import random

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


#-----Policy Iteration-----#
def policy_evaluation(policy, mdp, gamma=0.95, theta=1e-3):
    V = {s: 0 for s in mdp.states} # 모든 state의 초기 value 0
    while True:
        delta = 0
        for s in mdp.states:
            a = policy[s] # 현재 policy에서의 action
            next_s, reward, done = mdp.get_transition(s, a)
            V_new = reward + gamma * V.get(next_s, 0) * (not done) # Bellman Equation
            delta = max(delta, abs(V[s] - V_new))
            V[s] = V_new

        if delta < theta:
            break
    return V

def policy_improvement(V, mdp, gamma=0.95):
    policy_stable = True
    policy = {}
    for s in mdp.states:
        old_action = policy.get(s) 
        action_values = {}

        # 각 state에서 가능한 모든 action에 대해 q(s,a) 계산
        for a in mdp.actions: 
            next_s, reward, done = mdp.get_transition(s, a)
            action_values[a] = reward + gamma * V.get(next_s, 0) * (not done) # Bellman Equation

        best_action = max(action_values, key=action_values.get) # action_values 딕셔너리에서 값이 가장 큰 키를 찾음
        policy[s] = best_action # 각 state에서 action value가 가장 큰 action을 선택

        # 이전 policy와 다르면 수렴하지 않았다고 판단
        if old_action is not None and old_action != best_action:
            policy_stable = False

    return policy, policy_stable

def policy_iteration(mdp, gamma=0.95, max_iterations=80):
    # policy = {s: random.choice(mdp.actions) for s in mdp.states} # 초기 policy를 랜덤으로
    policy = {s: Action.RIGHT for s in mdp.states} # 초기 policy를 '오른쪽으로 이동'으로. 랜덤보다 빠른 수렴 유도 가능.
    iteration = 0
    policy_changes = []

    while iteration < max_iterations:
        # 현재 policy에서의 value function 계산
        V = policy_evaluation(policy, mdp, gamma) 
        
        # if iteration == 0:
        #     plot_value_and_policy(V, policy, mdp.env.grid, 'initial', mdp.width, mdp.height, prefix='pi')
        
        old_policy = policy.copy()
        # 현재 value function을 기반으로 더 나은 policy를 계산
        policy, _ = policy_improvement(V, mdp, gamma) # 각 state에서 가장 좋은 action 선택

        # policy 변화량 체크. 얼마나 많은 state에서 policy가 변경되었는지.
        changed = sum(old_policy[s] != policy[s] for s in mdp.states) 
        policy_changes.append(changed)

        plot_value_and_policy(V, policy, mdp.env.grid, iteration, mdp.width, mdp.height, prefix='pi')
        print(f"[PI Iter {iteration}] policy changes: {changed}")

        # policy가 더 이상 바뀌지 않으면 수렴했다고 판단
        if changed == 0:
            break

        iteration += 1

    return V, policy



#-----Value Iteration-----#
def value_iteration(mdp, gamma=0.95, theta=1e-3, max_iterations=80):
    V = {s: 0 for s in mdp.states} # 모든 state의 초기 value 0
    policy = {s: Action.UP for s in mdp.states} # 초기 policy를 '위로 이동'으로
    iteration = 0
    deltas = []

    while iteration < max_iterations:
        delta = 0
        new_V = V.copy()

        for s in mdp.states:
            max_value = float('-inf')
            best_action = None

            # 각 state에서 가능한 모든 action에 대해 q(s,a) 계산
            for a in mdp.actions:
                next_s, reward, done = mdp.get_transition(s, a)
                value = reward + gamma * (0 if done else V[next_s]) # Bellman equation

                # 가장 큰 Q-value를 가지는 action을 선택
                if value > max_value:
                    max_value = value
                    best_action = a

            # V(s)와 policy 둘 다 업데이트
            new_V[s] = max_value
            policy[s] = best_action

            delta = max(delta, abs(V[s] - new_V[s])) # 얼마나 value 값이 감소했는지

        V = new_V
        deltas.append(delta)
        plot_value_and_policy(V, policy, mdp.env.grid, iteration, mdp.width, mdp.height, prefix='vi')

        print(f"[VI Iter {iteration}] max Δ: {delta:.5f}")

        # 모든 state에서의 value 변화가 기준값 theta 보다 작으면 수렴했다고 판단
        if delta < theta:
            break

        iteration += 1

    return V, policy



