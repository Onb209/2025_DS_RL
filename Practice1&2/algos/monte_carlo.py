import random
from collections import defaultdict
from env.gridworld_env import Action

def monte_carlo(env, episodes=5000, gamma=0.99, epsilon=0.1, render=False):
    Q = defaultdict(lambda: {a: 0.0 for a in Action})
    returns = defaultdict(list)

    for _ in range(episodes):
        episode = []
        state = tuple(env.reset())
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.choice(list(Action))
            else:
                action = max(Q[state], key=Q[state].get)
            next_state, reward, done = env.step(action.value)
            episode.append((state, action, reward))
            state = tuple(next_state)

        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if (state, action) not in visited:
                returns[(state, action)].append(G)
                Q[state][action] = sum(returns[(state, action)]) / len(returns[(state, action)])
                visited.add((state, action))

    policy = {state: max(actions, key=actions.get) for state, actions in Q.items()}
    return Q, policy
