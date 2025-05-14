import random
from collections import defaultdict
from env.gridworld_env import Action
import time

def sarsa(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1, render=False):
    Q = defaultdict(lambda: {a: 0.0 for a in Action})

    for episode in range(episodes):
        state = tuple(env.reset())
        if random.random() < epsilon:
            action = random.choice(list(Action))
        else:
            action = max(Q[state], key=Q[state].get)
        done = False

        while not done:
            if render and episode % 500 == 0:
                env.render()
                time.sleep(0.05)

            next_state, reward, done = env.step(action.value)
            next_state = tuple(next_state)
            if random.random() < epsilon:
                next_action = random.choice(list(Action))
            else:
                next_action = max(Q[next_state], key=Q[next_state].get)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state, action = next_state, next_action

    policy = {state: max(actions, key=actions.get) for state, actions in Q.items()}
    return Q, policy
