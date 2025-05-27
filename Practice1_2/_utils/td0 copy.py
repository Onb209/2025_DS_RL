import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from env.gridworld_env import Action, TileType


def td0_prediction(env, gamma=0.95, alpha=0.1, episodes=500):
    V = defaultdict(float)
    policy = {
        (y, x): random.choice(list(Action))
        for y in range(env.height)
        for x in range(env.width)
        if env.grid[y][x] != TileType.WALL
    }
    deltas = []

    for i in range(episodes):
        state = tuple(env.reset())
        done = False
        delta = 0

        while not done:
            action = policy[state]
            next_state, reward, done = env.step(action)
            next_state = tuple(next_state)

            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]
            V[state] += alpha * td_error
            delta = max(delta, abs(td_error))

            state = next_state

        deltas.append(delta)
        if i % 20 == 0 or i == episodes - 1:
            plot_value_function(V, env, i, 'td0')

    plot_convergence(deltas, method='td0')
    return V

def plot_value_function(V, env, episode, method='td0'):
    grid = np.zeros((env.height, env.width))
    for y in range(env.height):
        for x in range(env.width):
            grid[y, x] = V.get((y, x), 0)

    plt.figure()
    plt.imshow(grid, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title(f'{method.upper()} Value - Episode {episode}')
    plt.savefig(f'outputs/{method}_value_{episode}.png')
    plt.close()

def plot_convergence(deltas, method='td0'):
    plt.figure()
    plt.plot(deltas)
    plt.title(f'{method.upper()} Value Convergence')
    plt.xlabel('Episode')
    plt.ylabel('Max ΔV')
    plt.grid()
    plt.savefig(f'outputs/{method}_convergence.png')
    plt.close()
