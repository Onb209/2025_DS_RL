import random
from collections import defaultdict
from env.gridworld_env import Action
import time

def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1, render=False):
    Q = defaultdict(lambda: {a: 0.0 for a in Action})

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
                action = max(Q[state], key=Q[state].get)

            next_state, reward, done = env.step(action.value)
            next_state = tuple(next_state)
            max_next = max(Q[next_state].values()) if Q[next_state] else 0
            Q[state][action] += alpha * (reward + gamma * max_next - Q[state][action])
            state = next_state

    policy = {state: max(actions, key=actions.get) for state, actions in Q.items()}
    return Q, policy

