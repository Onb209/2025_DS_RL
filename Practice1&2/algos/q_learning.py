import random
from collections import defaultdict
from env.gridworld_env import Action
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from algos.dynamic_programming import plot_value_and_policy


def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1, render=False, log_interval=100):
    # Q table 초기화
    Q = defaultdict(lambda: {a: 0.0 for a in Action})
    
    reward_history = []
    success_rate_history = []
    total_reward = 0
    success_count = 0
    max_steps = 100

    for episode in tqdm(range(episodes), desc="Training Q-Learning"):
        state = tuple(env.reset()) # 환경을 초기화하고, state를 tuple 형태로 변환해 사용
        done = False
        episode_reward = 0
        steps = 0

        # while not done:
        while not done and steps < max_steps:
            if render and episode % 500 == 0:
                env.render()
                time.sleep(0.05)

            # epsilon-greedy 하게 action 선택
            if random.random() < epsilon:
                action = random.choice(list(Action))
            else:
                action = max(Q[state], key=Q[state].get)

            next_state, reward, done = env.step(action.value)
            next_state = tuple(next_state)

            max_next = max(Q[next_state].values())
            Q[state][action] += alpha * (reward + gamma * max_next - Q[state][action]) # Q-learning update
            
            state = next_state

            episode_reward += reward
            if reward == 100:
                success_count += 1

            steps += 1

        reward_history.append(episode_reward)

        if (episode + 1) % log_interval == 0 or episode == episodes - 1:
            V = {s: max(Q[s].values()) for s in Q}
            policy = {s: max(Q[s], key=Q[s].get) for s in Q}
            plot_value_and_policy(V, policy, env.grid, episode, env.width, env.height, prefix='q_learning')

            avg_reward = sum(reward_history[-log_interval:]) / log_interval
            success_rate = success_count
            success_rate_history.append(success_rate)
            print(f"[Ep {episode+1}] Avg Reward: {avg_reward:.2f}, Successes: {success_rate}")
            success_count = 0

    policy = {state: max(actions, key=actions.get) for state, actions in Q.items()}

    plt.figure()
    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning - Episode Rewards')
    plt.grid(True)
    plt.savefig('outputs/q_learning_rewards.png')
    plt.close()

    return Q, policy

