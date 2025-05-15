import random
from collections import defaultdict
from env.gridworld_env import Action
from tqdm import tqdm  # pip install tqdm
import time

def monte_carlo(env, episodes=5000, gamma=0.99, epsilon=0.1, render=False, log_interval=100):
    Q = defaultdict(lambda: {a: 0.0 for a in Action})  # 낙관적 초기값
    returns = defaultdict(list)

    all_rewards = []
    success_count = 0

    for episode in tqdm(range(episodes), desc="Training Monte Carlo"):
        episode_data = []
        state = tuple(env.reset())
        done = False
        total_reward = 0

        while not done:
            if render and episode % log_interval == 0:
                env.render()
                time.sleep(0.05)

            if random.random() < epsilon:
                action = random.choice(list(Action))
            else:
                action = max(Q[state], key=Q[state].get)

            next_state, reward, done = env.step(action.value)
            episode_data.append((state, action, reward))
            total_reward += reward
            state = tuple(next_state)

        all_rewards.append(total_reward)
        if reward == 100:
            success_count += 1

        G = 0
        visited = set()
        for state, action, reward in reversed(episode_data):
            G = gamma * G + reward
            if (state, action) not in visited:
                returns[(state, action)].append(G)
                Q[state][action] = sum(returns[(state, action)]) / len(returns[(state, action)])
                visited.add((state, action))

        if (episode + 1) % log_interval == 0:
            avg_reward = sum(all_rewards[-log_interval:]) / log_interval
            success_rate = success_count / log_interval * 100
            print(f"[Episode {episode+1}] Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.1f}%")
            success_count = 0

    policy = {state: max(actions, key=actions.get) for state, actions in Q.items()}
    return Q, policy
