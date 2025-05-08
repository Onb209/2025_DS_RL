import random
from collections import defaultdict
from env.gridworld_env import Action
import time

def td0(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1, render=False):
    V = defaultdict(float)

    for episode in range(episodes):
        state = tuple(env.reset())
        done = False

        while not done:
            if render and episode % 500 == 0:
                env.render()
                time.sleep(0.05)
            if random.random() < epsilon:
                action = random.choice(list(Action))
            else:
                action = random.choice(list(Action))  # TD(0) evaluates a given policy; here we use random policy
            next_state, reward, done = env.step(action.value)
            next_state = tuple(next_state)
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state

    return V
