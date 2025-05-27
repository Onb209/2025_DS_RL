import random
from collections import defaultdict
from env.gridworld_env import Action
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# 15: 500 0.3 0.95
def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1, render=False):
    Q = defaultdict(lambda: {a: 0.0 for a in Action})
    reward_history = []
    success_rate_history = []
    total_reward = 0
    success_count = 0
    max_steps = 100

    for episode in tqdm(range(episodes), desc="Training Q-Learning"):
        state = tuple(env.reset())
        done = False
        episode_reward = 0
        steps = 0

        # while not done:
        while not done and steps < max_steps:
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

            episode_reward += reward
            if reward == 100:
                success_count += 1

            steps += 1

        reward_history.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = sum(reward_history[-100:]) / 100
            success_rate = success_count
            success_rate_history.append(success_rate)
            print(f"[Ep {episode+1}] Avg Reward: {avg_reward:.2f}, Successes: {success_rate}")
            success_count = 0

    policy = {state: max(actions, key=actions.get) for state, actions in Q.items()}
    plot_rewards(reward_history)
    visualize_q_table(Q, env.width, env.height)

    return Q, policy

def plot_rewards(reward_history):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Curve Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

def visualize_q_table(Q, width, height):
    direction_map = {
        Action.UP: '↑',
        Action.DOWN: '↓',
        Action.LEFT: '←',
        Action.RIGHT: '→'
    }

    grid_policy = [['' for _ in range(width)] for _ in range(height)]
    for (y, x), actions in Q.items():
        best_action = max(actions, key=actions.get)
        grid_policy[y][x] = direction_map[best_action]

    print("\nOptimal Policy (Q-table):")
    for row in grid_policy:
        print(' '.join(row))
