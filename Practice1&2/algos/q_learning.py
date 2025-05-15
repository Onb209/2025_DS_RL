import random
from collections import defaultdict
from env.gridworld_env import Action
import time
from tqdm import tqdm

def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1, render=False):
    Q = defaultdict(lambda: {a: 0.0 for a in Action})
    total_reward = 0
    success_count = 0

    for episode in tqdm(range(episodes), desc="Training Q-Learning"):
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

            max_next = max(Q[next_state].values())
            Q[state][action] += alpha * (reward + gamma * max_next - Q[state][action])
            state = next_state
            total_reward += reward
            if reward == 100:
                success_count += 1

        # ε 감소 (옵션)
        # epsilon = max(0.01, epsilon * 0.995)

        if (episode + 1) % 100 == 0:
            avg_reward = total_reward / 100
            success_rate = success_count
            print(f"[Ep {episode+1}] Avg Reward: {avg_reward:.2f}, Successes: {success_rate}")
            total_reward = 0
            success_count = 0

    policy = {state: max(actions, key=actions.get) for state, actions in Q.items()}
    return Q, policy


